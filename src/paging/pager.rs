//! Pager implementation.
//!
//! This module contains the public API (for the crate) to access pages on disk.
//! Pages are also cached in memory and the implementation takes advantage of
//! the Rust type system to automatically send pages that are acquired using
//! `&mut` to a write queue. Additionally, commit and rollback operations are
//! implemented here.

use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashSet},
    fmt::Debug,
    io::{self, Read, Seek, Write},
    mem,
    path::PathBuf,
};

use super::{
    cache::{Cache, FrameId},
    io::{BlockIo, FileOps},
};
use crate::{
    db::{DbError, DEFAULT_PAGE_SIZE},
    storage::page::{DbHeader, FreePage, MemPage, Page, PageTypeConversion, PageZero, MAGIC},
};

/// Are we gonna have more than 4 billion pages? Probably not ¯\_(ツ)_/¯
pub(crate) type PageNumber = u32;

/// Default value for [`Journal::max_pages`].
const DEFAULT_MAX_JOURNAL_BUFFERED_PAGES: usize = 10;

/// IO page manager that operates on top of a "block device" or disk.
///
/// Inspired mostly by [SQLite 2.8.1 pager].
///
/// [SQLite 2.8.1 pager]: https://github.com/antoniosarosi/sqlite2-btree-visualizer/blob/master/src/pager.c
///
/// This structure is built on top of [`BlockIo`], [`Cache`] and [`Journal`].
/// The [`Pager`] provides a high level API to access disk pages by simply
/// calling [`Pager::get`] or [`Pager::get_mut`] without being concerned about
/// when and how the page will be read from or written to disk.
///
/// Commit and rollback operations are also implemented through a "journal"
/// file. This is the exact same approach that SQLite 2.X.X takes. The journal
/// file maintains copies of the original unmodified pages while we modify the
/// actual database file.
///
/// If the user wants to "commit" the changes, then we simply delete the journal
/// file. Otherwise, if the user wants to "rollback" the changes, we copy the
/// original pages back to the database file, leaving it in the same state as
/// it was prior to modification. See [`Journal`] for more details.
#[derive(PartialEq)]
pub(crate) struct Pager<F> {
    /// Wrapped IO/file handle/descriptor.
    file: BlockIo<F>,
    /// Hardware/ file system block size or prefered IO read/write buffer size.
    pub block_size: usize,
    /// High level page size.
    pub page_size: usize,
    /// Page cache.
    cache: Cache,
    /// Keeps track of modified pages.
    dirty_pages: HashSet<PageNumber>,
    /// Transaction journal.
    journal: Journal<F>,
    /// Keeps track of pages written to the journal file.
    journal_pages: HashSet<PageNumber>,
}

// The derive Debug impl for the Pager prints too much stuff (the internal
// io Cursor buffer, every cache allocated page, etc). We'll just print some
// basic configs, this is only needed for tests.
impl<F> Debug for Pager<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("Pager")
            .field("page_size", &self.page_size)
            .field("cache_size", &self.cache.max_size())
            .field("journal_max_buf_pages", &self.journal.max_pages)
            .finish()
    }
}

/// Builder for [`Pager`].
///
/// There's nothing in this project that's easy to "build" for some reason.
pub(crate) struct Builder {
    block_size: Option<usize>,
    page_size: usize,
    cache: Option<Cache>,
    journal_file_path: PathBuf,
    max_journal_buffered_pages: usize,
}

impl Builder {
    /// Prepares a default [`Pager`].
    pub fn new() -> Self {
        Self {
            block_size: None,
            page_size: DEFAULT_PAGE_SIZE,
            cache: None,
            journal_file_path: PathBuf::new(),
            max_journal_buffered_pages: DEFAULT_MAX_JOURNAL_BUFFERED_PAGES,
        }
    }

    /// Sets the page size of the [`Pager`] and also the [`Cache`].
    pub fn page_size(mut self, page_size: usize) -> Self {
        self.page_size = page_size;
        self
    }

    /// Sets the block size of the underlying [`BlockIo`] instance.
    pub fn block_size(mut self, block_size: usize) -> Self {
        self.block_size = Some(block_size);
        self
    }

    /// Uses this cache for the [`Pager`].
    pub fn cache(mut self, cache: Cache) -> Self {
        self.cache = Some(cache);
        self
    }

    /// Path of the journal file.
    ///
    /// The file doesn't need to exist, it will be created when needed.
    pub fn journal_file_path(mut self, journal_file_path: PathBuf) -> Self {
        self.journal_file_path = journal_file_path;
        self
    }

    /// How many pages to buffer in memory before writing them to the journal
    /// file.
    pub fn max_journal_buffered_pages(mut self, max_journal_buffered_pages: usize) -> Self {
        self.max_journal_buffered_pages = max_journal_buffered_pages;
        self
    }

    /// Takes ownership of the file handle/descriptor and returns the final
    /// instance of [`Pager`].
    pub fn wrap<F>(self, file: F) -> Pager<F> {
        let Builder {
            block_size,
            page_size,
            cache,
            journal_file_path,
            max_journal_buffered_pages,
        } = self;

        let block_size = block_size.unwrap_or(page_size);

        // This one allocates a bunch of stuff so we evaluate it lazily.
        let mut cache = cache.unwrap_or_else(|| Cache::with_page_size(page_size));

        // Cache page size must be the same as the pager.
        cache.page_size = page_size;

        Pager {
            file: BlockIo::new(file, self.page_size, block_size),
            block_size,
            page_size,
            cache,
            dirty_pages: HashSet::new(),
            journal_pages: HashSet::new(),
            journal: Journal::new(JournalConfig {
                file_path: journal_file_path,
                max_pages: max_journal_buffered_pages,
                page_size,
            }),
        }
    }
}

impl<F> Pager<F> {
    /// Choose your own adventure.
    pub fn builder() -> Builder {
        Builder::new()
    }
}

impl<F: Seek + Read> Pager<F> {
    /// Manually read a page from disk.
    ///
    /// The cache system is not involved at all, this goes straight to disk.
    pub fn read(&mut self, page_number: PageNumber, buf: &mut [u8]) -> io::Result<usize> {
        self.file.read(page_number, buf)
    }
}

impl<F: Seek + Write> Pager<F> {
    /// Manually write a page to disk.
    ///
    /// Unlike normal writes there is no use of the cache/buffer pool. The page
    /// is written directly to disk.
    pub fn write(&mut self, page_number: PageNumber, buf: &[u8]) -> io::Result<usize> {
        self.file.write(page_number, buf)
    }
}

impl<F: Write + FileOps> Pager<F> {
    /// Appends the given page to the write queue and writes it to the journal
    /// if it's not alrady written.
    fn push_to_write_queue(&mut self, page_number: PageNumber, index: FrameId) -> io::Result<()> {
        self.cache.mark_dirty(page_number);
        self.dirty_pages.insert(page_number);

        if !self.journal_pages.contains(&page_number) {
            self.journal.push(page_number, &self.cache[index])?;
            self.journal_pages.insert(page_number);
        }

        Ok(())
    }
}

impl<F: Seek + Write + FileOps> Pager<F> {
    /// Writes all the pages present in the dirty queue and marks them as clean.
    ///
    /// Changes will most likely not be persisted to disk until [`Self::commit`]
    /// is called.
    pub fn write_dirty_pages(&mut self) -> io::Result<()> {
        if self.dirty_pages.is_empty() {
            return Ok(());
        }

        // Persist the original pages to disk first.
        self.journal.persist()?;

        // Sequential IO bruh blazingly fast :)
        let page_numbers = BinaryHeap::from_iter(self.dirty_pages.iter().copied().map(Reverse));

        for Reverse(page_number) in page_numbers {
            // Self::dirty_pages should never contain uncached pages, so
            // unwrapping should be safe here.
            let index = self.cache.get(page_number).unwrap();
            let page = &self.cache[index];
            self.file.write(page_number, page.as_ref())?;
            self.cache.mark_clean(page_number);
            self.dirty_pages.remove(&page_number);
        }

        Ok(())
    }

    /// If this succeeds then we can tell the client/user that data is
    /// persisted on disk.
    pub fn commit(&mut self) -> io::Result<()> {
        // If there are no pages in the journal it means we didn't modify
        // anything. The transaction was read-only.
        if self.journal_pages.is_empty() {
            return Ok(());
        }

        // Make sure everything goes to disk.
        self.write_dirty_pages()?;
        self.file.flush()?;
        self.file.sync()?;

        // Clear the journal hash set.
        self.journal_pages.clear();

        // Commit is confirmed when the journal file is deleted.
        self.journal.invalidate()
    }
}

impl<F: Write> Pager<F> {
    /// Flush buffered writes.
    ///
    /// See [`FileOps`] for details.
    pub fn flush(&mut self) -> io::Result<()> {
        self.file.flush()
    }
}

impl<F: FileOps> Pager<F> {
    /// Ensure writes reach their destination.
    ///
    /// See [`FileOps::sync`] for details.
    pub fn sync(&self) -> io::Result<()> {
        self.file.sync()
    }
}

impl<F: Seek + Read + Write + FileOps> Pager<F> {
    /// Initialize the database file.
    pub fn init(&mut self) -> io::Result<()> {
        // Manually read one block without involving the cache system, because
        // if the DB file already exists we might have to set the page size to
        // that defined in the file.
        let mut page_zero = PageZero::alloc(self.page_size);

        // alloc() initializes the page headers, zero the buffer to leave it
        // uninit.
        //
        // TODO: create another function alloc_zeroed() or something.
        page_zero.as_mut().fill(0);

        self.file.read(0, page_zero.as_mut())?;

        let magic = page_zero.header().magic;
        let page_size = page_zero.header().page_size as usize;

        // Magic number is written in the file, we'll assume that it is already
        // initialized.
        //
        // TODO: This is getting out of hand, we need a centralized place
        // to access the page size (and ideally not a global variable).
        if magic == MAGIC {
            self.page_size = page_size;
            self.cache.page_size = page_size;
            self.journal.page_size = page_size;
            self.file.page_size = page_size;
            return Ok(());
        }

        // Magic number is written but using the opposite endianness. We could
        // techincally make this work by calling `.to_le()` or `.to_be()`
        // everywhere or using a custom `LittleEndian<I>(I)` type, which would
        // be a no-op if the endianness of the machine is correct. But instead
        // we could just implement some functionality for dumping the SQL
        // insert statements just like MySQL or any other database does and not
        // deal with flipping bits around.
        if magic.swap_bytes() == MAGIC {
            panic!("the database file has been created using a different endianness than the one used by this machine");
        }

        // Initialize page zero.
        let page_zero = PageZero::alloc(self.page_size);
        self.write(0, page_zero.as_ref())?;

        Ok(())
    }

    /// Moves the page copies from the journal file back to the database file.
    ///
    /// See the journal file format in the documentation of [`Pager`] to have
    /// an understanding of what's going on here.
    pub fn rollback(&mut self) -> Result<usize, DbError> {
        // No-op if already open. Only necessary for the initial rollback on
        // startup.
        self.journal.open_if_exists()?;

        let mut num_pages_rolled_back = 0;
        let mut journal_pages = self.journal.iter()?;

        while let Some((page_number, content)) = journal_pages.try_next()? {
            self.file.write(page_number, content)?;
            self.cache.invalidate(page_number);
            self.dirty_pages.remove(&page_number);
            num_pages_rolled_back += 1;
        }

        self.file.flush()?;
        self.file.sync()?;

        self.journal_pages.clear();
        self.journal.invalidate()?;

        Ok(num_pages_rolled_back)
    }

    /// Returns a page as a concrete type.
    ///
    /// # Panics
    ///
    /// Panics if the requested type is not the actual type of the page at
    /// runtime. There's no way to recover from that as it would be a bug.
    ///
    /// # Implementation Notes
    ///
    /// We have a rare situation where we need to store multiple types of pages
    /// in the cache but at the same time we know the exact type of each page
    /// we're gonna use at compile time. `dyn Trait` doesn't cut it because we
    /// don't need a VTable for anything at all and it also introduces an extra
    /// level of indirection since we have to box it.
    ///
    /// Two different "manual" solutions have been tried:
    ///
    /// 1. The current solution based on an enum and [`TryFrom`]. Works pretty
    /// well and doesn't require much magic, the only downside is that it needs
    /// a lot of duplicated code to implement [`TryFrom`] for both `&P` and
    /// `&mut P` and for every enum variant. Basically two identical `impl`
    /// blocks for each variant, see the [`TryFrom`] impls below [`MemPage`].
    ///
    /// 2. The previous solution was using [`std::any::Any`] to downcast to the
    /// the concrete type at runtime, which is definitely not "elegant" or
    /// "good practice", but it works and it doesn't require so much duplicated
    /// code or macros. It does introduce some boilerplate if we want to know
    /// which type conversion failed though. Here's the last [commit] that used
    /// to do so.
    ///
    /// [commit]: https://github.com/antoniosarosi/mkdb/blob/master/src/paging/pager.rs#L197-L236
    ///
    /// We've managed to keep this project free of dependencies so far, so we're
    /// not gonna introduce one just for this feature, but we could use
    /// [`enum_dispatch`](https://docs.rs/enum_dispatch/) which basically
    /// automates what we're doing here by implementing [`TryFrom`] for each
    /// enum member using macros (didn't actually test if it works in our case
    /// since we need a specific lifetime). Another solution would be writing
    /// our own macro.
    pub fn get_as<'p, P>(&'p mut self, page_number: PageNumber) -> io::Result<&P>
    where
        P: PageTypeConversion + AsMut<[u8]>,
        &'p P: TryFrom<&'p MemPage>,
        <&'p P as TryFrom<&'p MemPage>>::Error: Debug,
    {
        let index = self.lookup::<P>(page_number)?;
        let mem_page = &self.cache[index];

        Ok(mem_page.try_into().expect("page type conversion error"))
    }

    /// Sames as [`Self::get_as`] but returns a mutable reference.
    ///
    /// This function also marks the page as dirty and adds it to the write
    /// queue.
    pub fn get_mut_as<'p, P>(&'p mut self, page_number: PageNumber) -> io::Result<&mut P>
    where
        P: PageTypeConversion + AsMut<[u8]>,
        &'p mut P: TryFrom<&'p mut MemPage>,
        <&'p mut P as TryFrom<&'p mut MemPage>>::Error: Debug,
    {
        let index = self.lookup::<P>(page_number)?;
        self.push_to_write_queue(page_number, index)?;

        let mem_page = &mut self.cache[index];

        Ok(mem_page.try_into().expect("page type conversion error"))
    }

    /// Returns a read-only reference to a BTree page.
    ///
    /// BTree page are the ones used most frequently, so we'll consider this
    /// function the default of [`Self::get_as`].
    pub fn get(&mut self, page_number: PageNumber) -> io::Result<&Page> {
        self.get_as::<Page>(page_number)
    }

    /// Default return type for [`Self::get_mut_as`].
    pub fn get_mut(&mut self, page_number: PageNumber) -> io::Result<&mut Page> {
        self.get_mut_as::<Page>(page_number)
    }

    /// Returns mutable references to many pages at the same time.
    ///
    /// If it's not possible or duplicated pages are received then [`None`] is
    /// returned instead.
    pub fn get_many_mut<const N: usize>(
        &mut self,
        pages: [PageNumber; N],
    ) -> io::Result<Option<[&mut Page; N]>> {
        if self.cache.max_size() < N {
            return Ok(None);
        }

        // TODO: This algorithm can be optimized by telling the cache to
        // give us space, then once we have all the space we need figure out
        // which pages are not in memory, load them from disk and finally build
        // the mutable refs. Easier said than done :)
        let frames = pages
            .iter()
            .map(|page| self.lookup::<Page>(*page))
            .collect::<Result<Vec<_>, _>>()?;

        // Couldn't cache all pages, bail out. Ideally we should use pin() and
        // then once we have all the space unpin() and return the refs, but that
        // could introduce some hard to spot bugs.
        if pages.iter().any(|page| !self.cache.contains(page)) {
            return Ok(None);
        }

        for (page, frame) in pages.iter().zip(frames) {
            self.push_to_write_queue(*page, frame)?;
        }

        Ok(self
            .cache
            .get_many_mut(pages)
            .map(|pages| pages.map(|page| page.try_into().expect("page type conversion error"))))
    }

    /// Returns the cache index of the given `page_number`.
    ///
    /// This function doesn't fail if the page is not cached, it will load the
    /// page from disk instead. So Best case scenario is when the page is
    /// already cached in memory. Worst case scenario is when we have to evict a
    /// dirty page in order to load the new one, which requires at least two IO
    /// operations, but possibly more since we flush the entire write queue at
    /// that point. Evicting a clean page doesn't require IO.
    ///
    /// Note that this function does not mark the page as dirty.
    fn lookup<P: PageTypeConversion + AsMut<[u8]>>(
        &mut self,
        page_number: PageNumber,
    ) -> io::Result<usize> {
        if let Some(index) = self.cache.get(page_number) {
            return Ok(index);
        }

        // We always know the type of page zero. See [`Self::get_as`] for
        // details on page types.
        if page_number == 0 {
            self.load_from_disk::<PageZero>(page_number)?;
        } else {
            self.load_from_disk::<P>(page_number)?;
        }

        // Unwrapping is safe because we've just loaded the page into cache.
        Ok(self.cache.get(page_number).unwrap())
    }

    /// Loads a page from disk into the cache.
    ///
    /// The page is not marked dirty, it will not be written back to disk
    /// unless [`Self::get_mut_as`] is called.
    fn load_from_disk<P: PageTypeConversion + AsMut<[u8]>>(
        &mut self,
        page_number: PageNumber,
    ) -> io::Result<()> {
        let index = self.map_page::<P>(page_number)?;
        self.file.read(page_number, self.cache[index].as_mut())?;

        Ok(())
    }

    /// Maps a page number to a cache entry.
    ///
    /// This process involves two expensive steps in the worst case scenario:
    ///
    /// 1. Execute the eviction algorithm to find which page must be evicted.
    ///
    /// 2. If the page that must be evicted is dirty, then use the opportunity
    /// to write all dirty pages sequentially at once.
    ///
    /// The eviction algorithm is O(n) so worst case requires O(n) work in
    /// memory and O(n) disk IO to write all the pages.
    fn map_page<P: PageTypeConversion>(&mut self, page_number: PageNumber) -> io::Result<usize> {
        if self.cache.must_evict_dirty_page() {
            self.write_dirty_pages()?;
        }

        let index = self.cache.map(page_number);
        self.cache[index].reinit_as::<P>();

        Ok(index)
    }

    /// Allocates a new page on disk that can be used to write data.
    pub fn alloc_disk_page(&mut self) -> io::Result<PageNumber> {
        let mut header = self.read_header()?;

        let free_page = if header.first_free_page == 0 {
            // If there are no free pages, then simply increase length by one.
            let page = header.total_pages;
            header.total_pages += 1;
            page
        } else {
            // Otherwise use one of the free pages.
            let page = header.first_free_page;
            let free_page = self.get_as::<FreePage>(page)?;
            header.first_free_page = free_page.header().next;
            header.free_pages -= 1;
            page
        };

        if header.first_free_page == 0 {
            header.last_free_page = 0;
        }

        self.write_header(header)?;

        Ok(free_page)
    }

    /// Allocates a page on disk and creates the cache entry for it.
    pub fn alloc_page<P: PageTypeConversion>(&mut self) -> io::Result<PageNumber> {
        let page_number = self.alloc_disk_page()?;
        self.map_page::<P>(page_number)?;

        Ok(page_number)
    }

    /// Adds the given page to the free list.
    ///
    /// **Important**: do not use the page after calling this function, since it
    /// will be replaced by a [`FreePage`] instance and all the data will be
    /// lost. Consider that a "use after free" bug.
    pub fn free_page(&mut self, page_number: PageNumber) -> io::Result<()> {
        // Bring the page into memory if it's not already there. The lookup
        // function will initialize the page as "free page" but then it will
        // write the data from disk into its buffer so it doesn't matter. We
        // don't know the type of the page here anyway.
        let index = self.lookup::<FreePage>(page_number)?;

        // Push the page to the write queue before modifying it so that the
        // journal gets its original contents. We don't care about the type of
        // the page here, we only care about the binary buffer, which is what
        // the journal will get.
        self.push_to_write_queue(page_number, index)?;

        // Now it's safe to reinitialize the page as a free page. The page
        // number is already stored in the write queue so at some point this
        // will be written to disk.
        self.cache[index].reinit_as::<FreePage>();

        let mut header = self.read_header()?;

        if header.first_free_page == 0 {
            // No previous free pages, initialize freelist.
            header.first_free_page = page_number;
        } else {
            // Grab the last free and make it point to the new last free.
            let last_free = self.get_mut_as::<FreePage>(page_number)?;
            last_free.header_mut().next = page_number;
        }

        header.last_free_page = page_number;
        header.free_pages += 1;

        self.write_header(header)
    }

    /// Returns a copy of the DB header.
    ///
    /// Since the header is small it's gonna be faster to copy it once, modify
    /// it and then write it back instead of accessing it through the cache
    /// system, which requires making sure that the borrow rules are met and
    /// other details.
    pub(crate) fn read_header(&mut self) -> io::Result<DbHeader> {
        self.get_as::<PageZero>(0).map(PageZero::header).copied()
    }

    /// Writes the header back to page zero. See [`Self::read_header`].
    fn write_header(&mut self, header: DbHeader) -> io::Result<()> {
        *self.get_mut_as::<PageZero>(0)?.header_mut() = header;
        Ok(())
    }
}

/// Type of the journal magic number. See [`Journal`].
type JournalMagic = u64;

/// Type used to write page numbers of total number of pages in a journal chunk.
type JournalPageNum = u32;

/// Type of the journal page checksum.
type JournalChecksum = u32;

/// Journal file magic number. See [`Journal`].
const JOURNAL_MAGIC: JournalMagic = 0x9DD505F920A163D6;

/// Size of [`JOURNAL_MAGIC`].
const JOURNAL_MAGIC_SIZE: usize = mem::size_of::<JournalMagic>();

/// Size of [`JournalPageNum`].
const JOURNAL_PAGE_NUM_SIZE: usize = mem::size_of::<JournalPageNum>();

/// Size of [`JournalChecksum`].
const JOURNAL_CHECKSUM_SIZE: usize = mem::size_of::<JournalChecksum>();

/// Total size of a journal chunk header.
const JOURNAL_HEADER_SIZE: usize = JOURNAL_MAGIC_SIZE + JOURNAL_PAGE_NUM_SIZE;

/// The journal is a file used to implement "commit" and "rollback" operations.
///
/// See the documentation of [`Pager`] for the basic idea.
///
/// # Journal File Format
///
/// The journal file format is almost the same as the one described in
/// [SQLite 2.8.1 docs].
///
/// [SQLite 2.8.1 docs]: https://github.com/antoniosarosi/sqlite2-btree-visualizer/blob/master/www/fileformat.tcl#L74
///
/// It has some important differences though. Mainly, we don't store the number
/// of pages that the database file had before we started modifying it and we
/// don't store the total amount of pages in the journal file either. Instead,
/// the journal file is made of "chunks" that look like this:
///
/// ```text
/// +--------------------+
/// |     Magic Number   | 8 bytes
/// +--------------------+
/// | Num Pages In Chunk | 4 bytes
/// +--------------------+
/// |    Page 0 Number   | 4 bytes
/// +--------------------+
/// |         ...        |
/// |   Page 0 Content   | PAGE SIZE bytes
/// |         ...        |
/// +--------------------+
/// |   Page 0 Checksum  | 4 bytes
/// +--------------------+
/// |    Page 1 Number   | 4 bytes
/// +--------------------+
/// |         ...        |
/// |   Page 1 Content   | PAGE SIZE bytes
/// |         ...        |
/// +--------------------+
/// |   Page 1 Checksum  | 4 bytes
/// +--------------------+
/// ```
///
/// Each chunk has a "header" that stores the magic number, ([`JOURNAL_MAGIC`],
/// which is the same value that SQLite 2 uses) and the number of pages in that
/// individual chunk. Let's call such number N. The header is then followed by
/// N instances of page metadata blocks. Each block stores one single page using
/// this format:
///
///
/// ```text
/// +------------------+
/// |    Page Number   | 4 bytes
/// +------------------+
/// |        ...       |
/// |   Page Content   | PAGE SIZE bytes
/// |        ...       |
/// +------------------+
/// |   Page Checksum  | 4 bytes
/// +------------------+
/// ```
///
/// The "checksum" is not actually a real checksum. In our case it's simply the
/// sum of the first 4 bytes of [`JOURNAL_MAGIC`] and the page number. SQLite 2
/// uses a random number that's stored at the beginning of the journal file to
/// do the additions, so each journal file would have different checksums for
/// the same page. That's definitely somewhat better than our useless aproach,
/// but it's still not a real checksum.
///
/// The reason we're not using random numbers is because we'd have to bring in
/// some third party dependency (this project is free of dependencies), link to
/// libc and use FFI or fallback to poor man's random numbers, namely, using
/// Unix timestamps as random numbers, using memory addresses from pointers as
/// random numbers, etc.
///
/// At that point we might as well just use some library that can compute a real
/// checksum. But anway, this is a toy database, you get the idea, want a
/// checksum? You can store it after the page content.
///
/// Going back to the file format, the reason we're storing multiple chunks is
/// because we have an in-memory buffer where we make copies of pages using the
/// format described above until it fills up and then simply dump the buffer to
/// the file. That way we don't have to rewind() or seek back to the start of
/// the file to update the total number of pages and then seek to the end of the
/// file again to keep writing new pages. We can just keep writing to the file
/// continously until we're done. The in-memory buffer also saves us from making
/// a syscall every single time we want to write a page to the journal file,
/// which should make this more efficient. But without any benchmarks to prove
/// it you can call it yet another useless micro-optimization :)
#[derive(Debug, PartialEq)]
struct Journal<F> {
    /// In-memory page buffer.
    buffer: Vec<u8>,
    /// Size of each page without including checksums and page numbers.
    page_size: usize,
    /// Number of pages currently stored in [`Self::buffer`].
    buffered_pages: u32,
    /// Maximum number of pages that can be buffered in memory.
    max_pages: usize,
    /// Path of the journal file.
    file_path: PathBuf,
    /// File handle/descriptor.
    file: Option<F>,
}

/// Wrote some many "builders" at this point that we have to try something new.
struct JournalConfig {
    page_size: usize,
    max_pages: usize,
    file_path: PathBuf,
}

impl<F> Journal<F> {
    /// Creates a new empty journal.
    pub fn new(
        JournalConfig {
            page_size,
            max_pages,
            file_path,
        }: JournalConfig,
    ) -> Self {
        let mut buffer = Vec::with_capacity(journal_chunk_size(page_size, max_pages));

        buffer.extend_from_slice(&JOURNAL_MAGIC.to_le_bytes());
        buffer.extend_from_slice(&[0; JOURNAL_PAGE_NUM_SIZE]);

        Self {
            buffer,
            max_pages,
            file_path,
            page_size,
            buffered_pages: 0,
            file: None,
        }
    }

    /// Clears the in-memory page buffer.
    pub fn clear(&mut self) {
        self.buffer.drain(JOURNAL_MAGIC_SIZE..);
        self.buffer.extend_from_slice(&[0; JOURNAL_PAGE_NUM_SIZE]);
        self.buffered_pages = 0;
    }
}

impl<F: Write> Journal<F> {
    pub fn flush(&mut self) -> io::Result<()> {
        self.file.as_mut().unwrap().flush()
    }
}

impl<F: FileOps> Journal<F> {
    pub fn sync(&mut self) -> io::Result<()> {
        self.file.as_mut().unwrap().sync()
    }

    /// Opens the journal file if it exists and is not already open.
    pub fn open_if_exists(&mut self) -> io::Result<()> {
        if self.file.is_some() {
            return Ok(());
        }

        // Miri doesn't work with system calls. All the tests run in memory
        // anyway and won't actually open any file. This is only needed when
        // the program runs on top of a real file system.
        #[cfg(not(miri))]
        if self.file_path.is_file() {
            self.file = Some(F::open(&self.file_path)?);
        }

        Ok(())
    }
}

/// Computes the size in bytes of a journal chunk that contains `num_pages`.
///
/// See the file format described in [`Journal`] for details.
fn journal_chunk_size(page_size: usize, num_pages: usize) -> usize {
    JOURNAL_MAGIC_SIZE + JOURNAL_PAGE_NUM_SIZE + (num_pages) * journal_page_size(page_size)
}

/// Computes the size that a page takes on the journal file considering its
/// metadata.
///
/// See the file format described in [`Journal`] for details.
fn journal_page_size(page_size: usize) -> usize {
    JOURNAL_PAGE_NUM_SIZE + page_size + JOURNAL_CHECKSUM_SIZE
}

impl<F: Write + FileOps> Journal<F> {
    /// Writes the in-memory buffer to the journal file and clears everything.
    ///
    /// The journal will be able to buffer pages in memory again after this
    /// function succeeds.
    pub fn write(&mut self) -> io::Result<()> {
        if self.buffer.len() <= JOURNAL_HEADER_SIZE {
            return Ok(());
        }

        // Create the journal file if it doesn't exist yet.
        if self.file.is_none() {
            self.file = Some(FileOps::create(&self.file_path)?);
        }

        // Persist copies to disk.
        self.file.as_mut().unwrap().write_all(&self.buffer)?;

        // Clear the in-memory buffer.
        self.clear();

        Ok(())
    }

    /// Ads the given page to this buffer.
    ///
    /// The [`Journal`] acts like a [`io::BufWriter`], buffering pages until
    /// the in-memory buffer is full. At that point, calling [`Self::push`]
    /// again will write the memory buffer to the journal file and clear its
    /// contents so that it can buffer subsequent pages. Therefore, IO is only
    /// required when the buffer cannot fit the given `page`.
    pub fn push(&mut self, page_number: PageNumber, page: impl AsRef<[u8]>) -> io::Result<()> {
        // If the buffer is full write it and clear it before pushing the new
        // page.
        if self.buffered_pages as usize >= self.max_pages {
            self.write()?;
        }

        // Write page number
        self.buffer.extend_from_slice(&page_number.to_le_bytes());

        // Write page content.
        self.buffer.extend_from_slice(page.as_ref());

        // TODO: We should generate a random number here but we can't without
        // adding dependencies. If we must add dependencies we might as well
        // compute a CRC checksum or something like that.
        let checksum = (JOURNAL_MAGIC as u32).wrapping_add(page_number);

        // Write "checksum" (if we can call this a "checksum").
        self.buffer.extend_from_slice(&checksum.to_le_bytes());

        let num_pages_range = JOURNAL_MAGIC_SIZE..JOURNAL_MAGIC_SIZE + JOURNAL_PAGE_NUM_SIZE;

        // Increase number of pages written to journal.
        self.buffered_pages += 1;
        self.buffer[num_pages_range].copy_from_slice(&self.buffered_pages.to_le_bytes());

        Ok(())
    }

    /// Makes sure that all the data written so far to the journal reaches the
    /// disk.
    pub fn persist(&mut self) -> io::Result<()> {
        self.write()?;
        self.flush()?;
        self.sync()?;

        Ok(())
    }

    /// Deletes the journal files and resets the journal state to empty.
    pub fn invalidate(&mut self) -> io::Result<()> {
        self.clear();

        if let Some(file) = self.file.take() {
            drop(file);
            F::remove(&self.file_path)?;
        }

        Ok(())
    }
}

impl<F: Seek + Read> Journal<F> {
    /// Returns a fallible iterator over the journal pages.
    ///
    /// The iterator does not return pages in FIFO order given that the memory
    /// buffer might still contain pages that were not written to disk. In that
    /// case, the in-memory pages are returned first and then the rest of pages
    /// in the journal file are returned in FIFO order.
    ///
    /// Order doesn't matter because we only need to write the pages back.
    /// Ideally we should do it sequentially to avoid random IO, but that's an
    /// optimization for another day.
    ///
    /// Another important detail is that consuming the iterator will reuse the
    /// [`Journal::buffer`] for buffered reads, but that doesn't matter since
    /// a single transaction will either rollback or commit, it can't rollback
    /// and later commit, so it's safe to discard values in the journal when
    /// rolling back.
    fn iter(&mut self) -> io::Result<JournalPagesIter<'_, F>> {
        if let Some(file) = self.file.as_mut() {
            file.rewind()?;
        };

        Ok(JournalPagesIter {
            journal: self,
            cursor: JOURNAL_HEADER_SIZE,
            eof: false,
        })
    }
}

/// See [`Journal::iter`].
///
/// This acts like a smart version of [`io::BufReader`] that is aware of the
/// journal file format. It only needs 2 IO operations for each journal chunk
/// and one memory buffer whereas using [`io::BufReader`] would require two
/// buffers, the internal buffer of the reader and our external buffer where we
/// actually read stuff. [`io::BufReader`] would probably require more IO
/// operations as well depending on the configured capacity. The capacity of
/// the [`JournalPagesIter`] is always one entire journal chunk.
struct JournalPagesIter<'j, F> {
    /// Journal instance.
    journal: &'j mut Journal<F>,
    /// Points to the beginning of a page in [`Journal::buffer`].
    cursor: usize,
    /// `true` if we reached EOF or there are no more pages otherwise.
    eof: bool,
}

impl<'j, F: Read> JournalPagesIter<'j, F> {
    /// Returns the next page in the journal.
    ///
    /// See [`Journal::iter`] for page order details.
    pub fn try_next(&mut self) -> Result<Option<(PageNumber, &[u8])>, DbError> {
        // Reached EOF, no more data.
        if self.eof {
            return Ok(None);
        }

        let corrupted_error =
            || DbError::Corrupted(String::from("journal file is corrupted or invalid"));

        // Returned all data from memory, read the next chunk in the file.
        if self.cursor >= self.journal.buffer.len() {
            let Some(file) = self.journal.file.as_mut() else {
                self.eof = true;
                return Ok(None);
            };

            // TODO: We can optimize this read operation by prefetching the
            // next header when reading the chunk.
            let mut header_buf = [0; JOURNAL_HEADER_SIZE];
            let bytes = file.read(&mut header_buf)?;

            if bytes == 0 {
                self.eof = true;
                return Ok(None);
            }

            if bytes != header_buf.len() {
                return Err(corrupted_error());
            }

            let magic =
                JournalMagic::from_le_bytes(header_buf[..JOURNAL_MAGIC_SIZE].try_into().unwrap());

            if magic != JOURNAL_MAGIC {
                return Err(corrupted_error());
            }

            let num_pages =
                JournalPageNum::from_le_bytes(header_buf[JOURNAL_MAGIC_SIZE..].try_into().unwrap());

            let total_bytes = journal_page_size(self.journal.page_size) * num_pages as usize;

            self.journal
                .buffer
                .resize(JOURNAL_HEADER_SIZE + total_bytes, 0);

            if file.read(&mut self.journal.buffer[JOURNAL_HEADER_SIZE..])? != total_bytes {
                return Err(corrupted_error());
            }

            self.journal.buffered_pages = 0;
            self.cursor = JOURNAL_HEADER_SIZE;
        }

        // Return pages from memory until we're done with this chunk.
        let page_number = u32::from_le_bytes(
            self.journal.buffer[self.cursor..self.cursor + JOURNAL_PAGE_NUM_SIZE]
                .try_into()
                .unwrap(),
        );
        self.cursor += JOURNAL_PAGE_NUM_SIZE;

        let page_buf = &self.journal.buffer[self.cursor..self.cursor + self.journal.page_size];
        self.cursor += self.journal.page_size;

        let checksum = u32::from_le_bytes(
            self.journal.buffer[self.cursor..self.cursor + JOURNAL_CHECKSUM_SIZE]
                .try_into()
                .unwrap(),
        );
        self.cursor += JOURNAL_CHECKSUM_SIZE;

        if checksum != (JOURNAL_MAGIC as u32).wrapping_add(page_number) {
            return Err(corrupted_error());
        }

        Ok(Some((page_number, page_buf)))
    }
}

#[cfg(test)]
mod tests {
    use std::io;

    use super::{Builder, Pager};
    use crate::{
        db::DbError,
        paging::{
            cache::Cache,
            io::MemBuf,
            pager::{journal_chunk_size, PageNumber},
        },
        storage::page::{Cell, OverflowPage, Page},
    };

    fn init_pager(builder: Builder) -> io::Result<Pager<MemBuf>> {
        let mut pager = builder.wrap(io::Cursor::new(Vec::new()));

        pager.init()?;

        Ok(pager)
    }

    fn init_pager_with_cache(cache: Cache) -> io::Result<Pager<MemBuf>> {
        init_pager(
            Pager::<MemBuf>::builder()
                .page_size(cache.page_size)
                .cache(cache),
        )
    }

    fn init_default_pager() -> io::Result<Pager<MemBuf>> {
        init_pager_with_cache(Cache::builder().page_size(64).max_size(64).build())
    }

    #[test]
    fn alloc_disk_page() -> io::Result<()> {
        let mut pager = init_default_pager()?;

        for i in 1..=10 {
            assert_eq!(pager.alloc_disk_page()?, i);
        }

        let header = pager.read_header()?;

        assert_eq!(header.first_free_page, 0);
        assert_eq!(header.last_free_page, 0);
        assert_eq!(header.total_pages, 11);
        assert_eq!(header.free_pages, 0);

        Ok(())
    }

    #[test]
    fn free_page() -> io::Result<()> {
        let mut pager = init_default_pager()?;

        for _ in 1..=10 {
            pager.alloc_disk_page()?;
        }

        for p in [5, 7, 9] {
            pager.free_page(p)?;
        }

        let header = pager.read_header()?;

        assert_eq!(header.first_free_page, 5);
        assert_eq!(header.last_free_page, 9);
        assert_eq!(header.free_pages, 3);

        Ok(())
    }

    #[test]
    fn write_queue() -> io::Result<()> {
        let mut pager = init_default_pager()?;

        for i in 1..=10 {
            let mut page = Page::alloc(pager.page_size);
            page.push(Cell::new(vec![
                i;
                Page::ideal_max_payload_size(pager.page_size, 1)
                    as usize
            ]));

            let page_number = pager.alloc_disk_page()?;

            pager.write(page_number, page.as_ref())?;
        }

        let update_pages = [5, 7, 9];

        for p in &update_pages {
            pager.get_mut(*p)?.cell_mut(0).content.fill(10 + *p as u8);
        }

        pager.write_dirty_pages()?;
        pager.flush()?;
        pager.sync()?;

        for i in 1..=10 {
            let mut expected = Page::alloc(pager.page_size);
            expected.push(Cell::new(vec![
                if update_pages.contains(&i) {
                    10 + i as u8
                } else {
                    i as u8
                };
                Page::ideal_max_payload_size(pager.page_size, 1)
                    as usize
            ]));

            let mut page = Page::alloc(pager.page_size);
            pager.read(i, page.as_mut())?;

            assert_eq!(page, expected);
        }

        assert!(!pager.cache.must_evict_dirty_page());
        assert!(pager.dirty_pages.is_empty());

        Ok(())
    }

    #[test]
    fn write_pages_when_dirty_page_is_evicted() -> io::Result<()> {
        let mut pager = init_pager_with_cache(Cache::builder().max_size(3).page_size(64).build())?;

        // Cache size is 3 and pager needs to read page 0 to allocate. So this
        // will fill the cache.
        for i in 1..=2 {
            let page_number = pager.alloc_page::<OverflowPage>()?;
            pager
                .get_mut_as::<OverflowPage>(page_number)?
                .content_mut()
                .fill(i as u8);
        }

        pager.cache.pin(0);

        // Trying to allocate another page will evict one of the previous.
        let causes_evict = pager.alloc_page::<OverflowPage>()?;
        pager
            .get_mut_as::<OverflowPage>(causes_evict)?
            .content_mut()
            .fill(3);

        for i in 1..=2 {
            let mut page = OverflowPage::alloc(pager.page_size);
            pager.read(i, page.as_mut())?;

            let mut expected = OverflowPage::alloc(pager.page_size);
            expected.content_mut().fill(i as u8);

            assert_eq!(page, expected);
        }

        assert!(!pager.cache.must_evict_dirty_page());

        Ok(())
    }

    #[test]
    fn get_many_mut_ok() -> io::Result<()> {
        let mut pager = init_pager_with_cache(Cache::with_max_size(3))?;

        let mut cells = Vec::new();

        for i in 1..=3_u32 {
            let page_number = pager.alloc_page::<Page>()?;
            let cell = Cell::new(Vec::from(&i.to_le_bytes()));
            pager.get_mut(page_number)?.push(cell.clone());
            cells.push(cell);
        }

        let mut_refs = pager.get_many_mut([1, 2, 3])?;

        assert!(mut_refs.is_some());

        for (mut_ref, cell) in mut_refs.unwrap().into_iter().zip(cells) {
            assert_eq!(mut_ref.len(), 1);
            assert_eq!(mut_ref.cell(0), cell.as_ref());
        }

        Ok(())
    }

    #[test]
    fn get_many_mut_fail() -> io::Result<()> {
        let mut pager = init_pager_with_cache(Cache::with_max_size(3))?;

        // Read page zero into cache.
        pager.get(0)?;
        // Pin it.
        pager.cache.pin(0);

        // Cache size is 3 and one page is pinned, so this is impossible.
        let mut_refs = pager.get_many_mut([1, 2, 3])?;

        assert!(mut_refs.is_none());

        Ok(())
    }

    #[test]
    fn cache_pressure() -> io::Result<()> {
        let mut pager = init_pager_with_cache(Cache::builder().max_size(5).page_size(256).build())?;

        let max_pages = 25;

        let mut page_numbers = Vec::new();

        for _ in 1..max_pages {
            let page_number = pager.alloc_page::<OverflowPage>()?;
            page_numbers.push(page_number);

            pager
                .get_mut_as::<OverflowPage>(page_number)?
                .content_mut()
                .fill(page_number as u8);
        }

        for page_number in page_numbers {
            let mut expected = OverflowPage::alloc(pager.page_size);
            expected.content_mut().fill(page_number as u8);

            assert_eq!(pager.get_as::<OverflowPage>(page_number)?, &expected);
        }

        Ok(())
    }

    #[test]
    fn write_to_journal_before_writing_dirty_pages() -> io::Result<()> {
        let mut pager = init_pager(
            Pager::<MemBuf>::builder()
                .page_size(64)
                .max_journal_buffered_pages(10)
                .cache(Cache::with_max_size(3)),
        )?;

        let modified_pages = 3;

        // We'll bypass page allocations to make sure the pager doesn't use
        // page 0. That way we know exactly how many pages should be written to
        // the journal.
        for page_number in 1..=modified_pages {
            pager.get_mut_as::<OverflowPage>(page_number)?;
        }

        // Should sync the journal.
        pager.write_dirty_pages()?;

        // We're not going to check the written content here basically because
        // we'd have to reimplement the rollback algorithm. At that point we'd
        // be getting into the "test the test" situation. We could decouple
        // reading and rolling back... so... TODO.
        assert!(pager.journal.file.is_some());
        assert_eq!(
            pager.journal.file.unwrap().into_inner().len(),
            journal_chunk_size(pager.page_size, modified_pages as usize)
        );

        Ok(())
    }

    #[test]
    fn write_journal_pages_when_journal_buf_fills_up() -> io::Result<()> {
        let buffered_pages = 6;

        let mut pager = init_pager(
            Pager::<MemBuf>::builder()
                .page_size(64)
                .cache(Cache::with_max_size(64))
                .max_journal_buffered_pages(buffered_pages),
        )?;

        for page_number in 1..=buffered_pages + 1 {
            pager.get_mut_as::<OverflowPage>(page_number as PageNumber)?;
        }

        assert!(pager.journal.file.is_some());
        assert_eq!(
            pager.journal.buffer.len(),
            journal_chunk_size(pager.page_size, 1)
        );
        assert_eq!(
            pager.journal.file.unwrap().into_inner().len(),
            journal_chunk_size(pager.page_size, buffered_pages)
        );

        Ok(())
    }

    #[test]
    fn write_multiple_journal_chunks() -> io::Result<()> {
        let buffered_pages = 3;
        let modified_pages = 8;

        let mut pager = init_pager(
            Pager::<MemBuf>::builder()
                .page_size(64)
                .cache(Cache::with_max_size(64))
                .max_journal_buffered_pages(buffered_pages),
        )?;

        for page_number in 1..=modified_pages {
            pager.get_mut_as::<OverflowPage>(page_number as PageNumber)?;
        }

        assert!(pager.journal.file.is_some());

        // 2 complete chunks should be written to the file.
        assert_eq!(
            pager.journal.file.unwrap().into_inner().len(),
            journal_chunk_size(pager.page_size, buffered_pages) * 2
        );

        // There should be a partial chunk in memory.
        assert_eq!(
            pager.journal.buffer.len(),
            journal_chunk_size(pager.page_size, 2)
        );

        Ok(())
    }

    #[test]
    fn rollback() -> Result<(), DbError> {
        // Modifying 8 pages when the buffer can only store 3 will cause the
        // rollback to read two chunks from the journal file and one partial
        // chunk from memory.
        let buffered_pages = 3;
        let modified_pages = 8;
        let total_pages = modified_pages * 2;

        let mut pager = init_pager(
            Pager::<MemBuf>::builder()
                .page_size(64)
                .cache(Cache::with_max_size(64))
                .max_journal_buffered_pages(buffered_pages),
        )?;

        let update_key = 170;

        let mut expected_pages = Vec::new();

        // Load initialized pages from disk into mem.
        for page_number in 1..=total_pages {
            let mut ovf_page = OverflowPage::alloc(pager.page_size);
            ovf_page.content_mut().fill(page_number as u8);
            expected_pages.push(ovf_page.clone());

            pager.write(page_number as PageNumber, ovf_page.as_ref())?;
            pager.get_as::<OverflowPage>(page_number)?;
        }

        for i in 1..=total_pages {
            if i % 2 == 0 {
                pager
                    .get_mut_as::<OverflowPage>(i)?
                    .content_mut()
                    .fill(update_key);
            }
        }

        pager.rollback()?;

        assert!(pager.journal.file.is_none());
        assert_eq!(
            pager.journal.buffer.len(),
            journal_chunk_size(pager.page_size, 0)
        );

        for (page_number, expected_page) in (1..=total_pages).zip(expected_pages.iter()) {
            assert_eq!(pager.get_as::<OverflowPage>(page_number)?, expected_page);
        }

        Ok(())
    }
}
