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
    io::{self, BufRead, BufReader, Read, Seek, Write},
    mem,
    path::PathBuf,
};

use super::{
    cache::Cache,
    io::{BlockIo, FileOps},
};
use crate::{
    db::{DbError, DEFAULT_PAGE_SIZE},
    storage::page::{DbHeader, FreePage, MemPage, Page, PageTypeConversion, PageZero, MAGIC},
};

/// Are we gonna have more than 4 billion pages? Probably not ¯\_(ツ)_/¯
pub(crate) type PageNumber = u32;

/// Journal file magic number. See [`Pager`].
const JOURNAL_MAGIC: u64 = 0x9DD505F920A163D6;

/// Default value for calculating [`Pager::max_journal_buf_size`].
const DEFAULT_MAX_JOURNAL_BUFFERED_PAGES: usize = 10;

/// IO page manager that operates on top of a "block device" or disk.
///
/// Inspired mostly by [SQLite 2.8.1 pager].
///
/// [SQLite 2.8.1 pager]: https://github.com/antoniosarosi/sqlite2-btree-visualizer/blob/master/src/pager.c
///
/// This structure is built on top of [`BlockIo`] and [`Cache`] and provides a
/// high level API to access disk pages. The user calls [`Pager::get`] or
/// [`Pager::get_mut`] with a page number and they can read and write the
/// contents of that page without being concerned about when and how the page
/// will be read from or written to disk.
///
/// Commit and rollback operations are also implemented through a "journal"
/// file. This is the exact same approach that SQLite 2.X.X takes. The journal
/// file maintains copies of the original unmodified pages while we modify the
/// actual database file.
///
/// If the user wants to "commit" the changes, then we simply delete the journal
/// file. Otherwise, if the user wants to "rollback" the changes, we copy the
/// original pages back to the database file, leaving it in the same state as
/// it was prior to modification.
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
pub(crate) struct Pager<F> {
    /// Wrapped IO/file handle/descriptor.
    io: BlockIo<F>,
    /// Hardware block size or prefered IO read/write buffer size.
    pub block_size: usize,
    /// High level page size.
    pub page_size: usize,
    /// Page cache.
    cache: Cache,
    /// Keeps track of modified pages.
    dirty_pages: HashSet<PageNumber>,
    /// Keeps track of pages written to the journal file.
    journal_pages: HashSet<PageNumber>,
    /// Copies of pages are kept in memory until we actually need to write them.
    journal_buffer: Vec<u8>,
    /// Once the max size is reached we have to write to the journal file.
    max_journal_buf_size: usize,
    /// Number of pages in [`Self::journal_buffer`].
    buffered_journal_pages: u32,
    /// Journal path.
    journal_file_path: PathBuf,
    /// Journal file descriptor or handle.
    journal: Option<F>,
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
    pub fn wrap<F>(self, io: F) -> Pager<F> {
        let Builder {
            block_size,
            page_size,
            cache,
            journal_file_path,
            max_journal_buffered_pages,
        } = self;

        let block_size = block_size.unwrap_or(page_size);

        // This one allocates a bunch of stuff so we evaluate it lazily.
        let cache = cache.unwrap_or_else(|| Cache::with_page_size(page_size));

        assert_eq!(
            page_size,
            cache.page_size(),
            "conflicting page sizes for cache and pager"
        );

        let journal_magic_size = mem::size_of_val(&JOURNAL_MAGIC);

        // Magic number + num_pages size + (page num + page size + checksum size) * num_pages
        let max_journal_buf_size = journal_magic_size
            + mem::size_of::<u32>()
            + (mem::size_of::<PageNumber>() + page_size + mem::size_of::<u32>())
                * max_journal_buffered_pages;

        let mut journal_buffer = Vec::from(JOURNAL_MAGIC.to_le_bytes());
        journal_buffer.extend_from_slice(&0u32.to_le_bytes());

        Pager {
            io: BlockIo::new(io, self.page_size, block_size),
            block_size,
            page_size,
            cache,
            journal_file_path,
            max_journal_buf_size,
            dirty_pages: HashSet::new(),
            journal_pages: HashSet::new(),
            journal_buffer,
            buffered_journal_pages: 0,
            journal: None,
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
        self.io.read(page_number, buf)
    }
}

impl<F: Seek + Write + FileOps> Pager<F> {
    /// Manually write a page to disk.
    ///
    /// Unlike normal writes there is no use of the cache/buffer pool. The page
    /// is written directly to disk.
    pub fn write(&mut self, page_number: PageNumber, buf: &[u8]) -> io::Result<usize> {
        self.io.write(page_number, buf)
    }

    /// Writes all the in-memory copies of original pages to disk but does not
    /// `fsync()`.
    ///
    /// The OS will probably buffer the writes. [`Self::sync_journal`] should be
    /// called when we need to make sure everything reaches the disk.
    fn write_journal_buffer(&mut self) -> io::Result<()> {
        // Journal buffer is empty, only the magic number is written.
        if self.journal_buffer.len() <= mem::size_of_val(&JOURNAL_MAGIC) {
            return Ok(());
        }

        // Create the journal file if it doesn't exist yet.
        if self.journal.is_none() {
            self.journal = Some(FileOps::create(&self.journal_file_path)?);
        }

        let journal = self.journal.as_mut().unwrap();

        // Persist copies to disk.
        journal.write_all(&self.journal_buffer)?;

        // Clear the in-memory buffer.
        self.journal_buffer
            .drain(mem::size_of_val(&JOURNAL_MAGIC)..);
        self.journal_buffer.extend_from_slice(&0u32.to_le_bytes());
        self.buffered_journal_pages = 0;

        Ok(())
    }

    /// Makes sure that everything we've written to the journal so far reaches
    /// the disk.
    fn sync_journal(&mut self) -> io::Result<()> {
        self.write_journal_buffer()?;

        let journal = self.journal.as_mut().unwrap();

        journal.flush()?;
        journal.sync()
    }

    /// Appends the given page to the write queue and writes it to the journal
    /// if it's not alrady written.
    ///
    /// See the journal file format described in the documentation of [`Pager`]
    /// to understand what's going on here.
    fn push_to_write_queue(&mut self, page_number: PageNumber, index: usize) -> io::Result<()> {
        // Mark page as dirty.
        self.cache.mark_dirty(page_number);
        self.dirty_pages.insert(page_number);

        // Already written to the journal, early return.
        if self.journal_pages.contains(&page_number) {
            return Ok(());
        }

        self.journal_pages.insert(page_number);

        // Write page number
        self.journal_buffer
            .extend_from_slice(&page_number.to_le_bytes());

        // Write page content.
        self.journal_buffer
            .extend_from_slice(self.cache[index].as_ref());

        // TODO: We should generate a random number here but we can't without
        // adding dependencies. If we must add dependencies we might as well
        // compute a CRC checksum or something like that.
        let checksum = (JOURNAL_MAGIC as u32).wrapping_add(page_number);

        // Write "checksum" (if we can call this a "checksum").
        self.journal_buffer
            .extend_from_slice(&checksum.to_le_bytes());

        let num_pages_range = mem::size_of_val(&JOURNAL_MAGIC)
            ..mem::size_of_val(&JOURNAL_MAGIC) + mem::size_of::<u32>();

        // Increase number of pages written to journal.
        self.buffered_journal_pages += 1;
        self.journal_buffer[num_pages_range]
            .copy_from_slice(&self.buffered_journal_pages.to_le_bytes());

        // If the buffer is full we'll write it now. Otherwise we can wait
        // until we either have to write dirty pages or the buffer becomes full.
        if self.journal_buffer.len() >= self.max_journal_buf_size {
            self.write_journal_buffer()?;
        }

        Ok(())
    }

    /// Writes all the pages present in the dirty queue and marks them as clean.
    ///
    /// Changes might not be reflected unless [`Self::flush`] and [`Self::sync`]
    /// are called.
    pub fn write_dirty_pages(&mut self) -> io::Result<()> {
        if self.dirty_pages.is_empty() {
            return Ok(());
        }

        // Persist the original pages to disk first.
        self.sync_journal()?;

        // Sequential IO bruh blazingly fast :)
        let page_numbers = BinaryHeap::from_iter(self.dirty_pages.iter().copied().map(Reverse));

        for Reverse(page_number) in page_numbers {
            // Self::dirty_pages should never contain uncached pages, so
            // unwrapping should be safe here.
            let index = self.cache.get(page_number).unwrap();
            let page = &self.cache[index];
            self.io.write(page_number, page.as_ref())?;
            self.cache.mark_clean(page_number);
            self.dirty_pages.remove(&page_number);
        }

        Ok(())
    }

    /// If this succeeds then we can tell the client/user that data is
    /// persisted on disk.
    pub fn commit(&mut self) -> io::Result<()> {
        if self.journal_pages.is_empty() {
            return Ok(());
        }

        // Make sure everything goes to disk. This includes both the journal
        // and the DB pages, since write_dirty_pages() calls sync_journal().
        self.write_dirty_pages()?;
        self.flush()?;
        self.sync()?;

        // Move the journal file out and drop it.
        drop(self.journal.take());

        // Clear the journal hash set.
        self.journal_pages.clear();

        // Commit is confirmed when the journal file is deleted.
        F::remove(&self.journal_file_path)
    }
}

impl<F: Write> Pager<F> {
    /// Flush buffered writes.
    ///
    /// See [`FileOps::sync`] for details.
    pub fn flush(&mut self) -> io::Result<()> {
        self.io.flush()
    }
}

impl<F: FileOps> Pager<F> {
    /// Ensure writes reach their destination.
    ///
    /// See [`FileOps::sync`] for details.
    pub fn sync(&self) -> io::Result<()> {
        self.io.sync()
    }
}

impl<F: Seek + Read + Write + FileOps> Pager<F> {
    /// Initialize the database file.
    pub fn init(&mut self) -> io::Result<()> {
        // Manually read one block without involving the cache system, because
        // if the DB file already exists we might have to set the page size to
        // that defined in the file.
        let (magic, page_size) = {
            let mut page_zero = PageZero::alloc(self.block_size);
            self.read(0, page_zero.as_mut())?;
            (page_zero.header().magic, page_zero.header().page_size)
        };

        // Magic number is written in the file, we'll assume that it is already
        // initialized.
        if magic == MAGIC {
            self.page_size = page_size as usize;
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
        // Try to open the journal file first. If we can't open it then we're
        // going to create the file, dump the in-memory buffer and rollback
        // after that. We should rollback from memory but... laziness. TODO.
        if self.journal.is_none() {
            self.journal = Some(
                FileOps::open(&self.journal_file_path)
                    .or_else(|_| FileOps::create(&self.journal_file_path))?,
            );
        }

        // TODO: This sync call be optimized away by reading the remaining
        // pages from the in-memory buffer once we're done with the file.
        self.sync_journal()?;

        let mut journal = self.journal.take().unwrap();
        journal.rewind()?;

        let mut reader = BufReader::new(journal);

        let mut num_pages_rolled_back = 0;

        let corrupted_error =
            || DbError::Corrupted(String::from("journal file is corrupted or invalid"));

        while reader.has_data_left()? {
            let mut u64buf = [0; mem::size_of::<u64>()];
            reader.read_exact(&mut u64buf)?;

            if u64::from_le_bytes(u64buf) != JOURNAL_MAGIC {
                return Err(corrupted_error());
            }

            let mut u32buf = [0; mem::size_of::<u32>()];
            reader.read_exact(&mut u32buf)?;

            let num_pages = u32::from_le_bytes(u32buf);

            for _ in 0..num_pages {
                reader.read_exact(&mut u32buf)?;

                let page_number = PageNumber::from_le_bytes(u32buf);

                let mut page_buf = vec![0; self.page_size];
                reader.read_exact(&mut page_buf)?;
                reader.read_exact(&mut u32buf)?;

                let checksum = (JOURNAL_MAGIC as u32).wrapping_add(page_number);

                // TODO: At this point we might already have written some
                // pages back to the database. What do we do with those pages?
                // Should we sync them?
                if u32::from_le_bytes(u32buf) != checksum {
                    return Err(corrupted_error());
                }

                self.cache.invalidate(page_number);
                // TODO: Manual writes or go through the cache system and
                // write everything at once?
                self.write(page_number, &page_buf)?;

                num_pages_rolled_back += 1;
            }
        }

        // TODO: Should we sync at the end or should we sync after every page
        // write? The second option seems inefficient.
        self.sync()?;

        // If we managed to sync the changes then the journal file no longer
        // serves any purpose.
        F::remove(&self.journal_file_path)?;

        self.journal_pages.clear();

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
        self.io.read(page_number, self.cache[index].as_mut())?;

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
            let page = self.get_as::<FreePage>(header.last_free_page)?;
            header.first_free_page = page.header().next;
            header.free_pages -= 1;
            header.last_free_page
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
        // Initialize last free page.
        let index = self.lookup::<FreePage>(page_number)?;
        self.cache[index].reinit_as::<FreePage>();

        self.push_to_write_queue(page_number, index)?;

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
    fn read_header(&mut self) -> io::Result<DbHeader> {
        self.get_as::<PageZero>(0).map(PageZero::header).copied()
    }

    /// Writes the header back to page zero. See [`Self::read_header`].
    fn write_header(&mut self, header: DbHeader) -> io::Result<()> {
        *self.get_mut_as::<PageZero>(0)?.header_mut() = header;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::{io, mem};

    use super::Pager;
    use crate::{
        db::DbError,
        paging::{
            cache::Cache,
            io::MemBuf,
            pager::{PageNumber, DEFAULT_MAX_JOURNAL_BUFFERED_PAGES, JOURNAL_MAGIC},
        },
        storage::page::{Cell, OverflowPage, Page},
    };

    fn init_pager_with_cache(cache: Cache) -> io::Result<Pager<MemBuf>> {
        let mut pager = Pager::<MemBuf>::builder()
            .page_size(cache.page_size())
            .cache(cache)
            .wrap(io::Cursor::new(Vec::new()));

        pager.init()?;

        Ok(pager)
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

    fn expected_journal_len(page_size: usize, written_pages: usize) -> usize {
        mem::size_of_val(&JOURNAL_MAGIC)
            + mem::size_of::<u32>()
            + (written_pages) * (mem::size_of::<u32>() + page_size + mem::size_of::<u32>())
    }

    #[test]
    fn write_to_journal_before_writing_dirty_pages() -> io::Result<()> {
        let mut pager = init_pager_with_cache(Cache::builder().max_size(3).page_size(64).build())?;

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
        assert!(pager.journal.is_some());
        assert_eq!(
            pager.journal.unwrap().into_inner().len(),
            expected_journal_len(pager.page_size, modified_pages as usize)
        );

        Ok(())
    }

    #[test]
    fn write_journal_pages_when_journal_buf_fills_up() -> io::Result<()> {
        let mut pager = init_pager_with_cache(Cache::builder().max_size(64).page_size(64).build())?;

        let modified_pages = DEFAULT_MAX_JOURNAL_BUFFERED_PAGES;

        for page_number in 1..=modified_pages + 1 {
            pager.get_mut_as::<OverflowPage>(page_number as PageNumber)?;
        }

        assert!(pager.journal.is_some());
        assert_eq!(
            pager.journal_buffer.len(),
            expected_journal_len(pager.page_size, 1)
        );
        assert_eq!(
            pager.journal.unwrap().into_inner().len(),
            expected_journal_len(pager.page_size, modified_pages)
        );

        Ok(())
    }

    #[test]
    fn rollback() -> Result<(), DbError> {
        let mut pager = init_pager_with_cache(Cache::builder().max_size(64).page_size(64).build())?;

        let update_key = 170;

        let mut expected_pages = Vec::new();

        // Load initialized pages from disk into mem.
        for page_number in 1..=10 {
            let mut ovf_page = OverflowPage::alloc(pager.page_size);
            ovf_page.content_mut().fill(page_number as u8);
            expected_pages.push(ovf_page.clone());

            pager.write(page_number as PageNumber, ovf_page.as_ref())?;
            pager.get_as::<OverflowPage>(page_number)?;
        }

        for update_page in [5, 7, 9] {
            pager
                .get_mut_as::<OverflowPage>(update_page)?
                .content_mut()
                .fill(update_key);
        }

        pager.rollback()?;

        assert!(pager.journal.is_none());
        assert_eq!(
            pager.journal_buffer.len(),
            expected_journal_len(pager.page_size, 0)
        );

        for (page_number, expected_page) in (1..=10).zip(expected_pages.iter()) {
            assert_eq!(pager.get_as::<OverflowPage>(page_number)?, expected_page);
        }

        Ok(())
    }
}
