//! Pager implementation.
//!
//! This module contains the public API (for the crate) to access pages on disk.
//! Pages are also cached in memory and the implementation takes advantage of
//! the Rust type system to automatically send pages that are acquired using
//! `&mut` to a write queue.

use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashSet},
    fmt::Debug,
    io::{self, Read, Seek, Write},
};

use super::{cache::Cache, io::BlockIo};
use crate::storage::page::{AllocPageInMemory, DbHeader, FreePage, MemPage, Page, PageZero, MAGIC};

/// Are we gonna have more than 4 billion pages? Probably not ¯\_(ツ)_/¯
pub(crate) type PageNumber = u32;

/// IO block device page manager.
pub(crate) struct Pager<I> {
    /// Wrapped IO resource handle.
    io: BlockIo<I>,
    /// Hardware block size or prefered IO read/write buffer size.
    pub block_size: usize,
    /// High level page size.
    pub page_size: usize,
    /// Page cache.
    cache: Cache,
    /// Keeps track of modified pages.
    dirty_pages: HashSet<PageNumber>,
}

impl<I> Pager<I> {
    /// Creates a new pager on top of `io`.
    ///
    /// `block_size` should evenly divide `page_size` or viceversa. Ideally,
    /// both should be powers of 2, but it's convinient to support any size for
    /// testing.
    pub fn new(io: I, page_size: usize, block_size: usize) -> Self {
        Self {
            io: BlockIo::new(io, page_size, block_size),
            block_size,
            page_size,
            cache: Cache::new(),
            dirty_pages: HashSet::new(),
        }
    }

    /// Same as [`Self::new`] but allows a custom cache instance.
    pub fn with_cache(io: I, page_size: usize, block_size: usize, cache: Cache) -> Self {
        Self {
            io: BlockIo::new(io, page_size, block_size),
            block_size,
            page_size,
            cache,
            dirty_pages: HashSet::new(),
        }
    }
}

impl<I> Pager<I> {
    /// Appends the given page to the write queue.
    fn push_to_write_queue(&mut self, page_number: PageNumber) {
        self.cache.mark_dirty(page_number);
        self.dirty_pages.insert(page_number);
    }
}

impl<I: Seek + Read> Pager<I> {
    /// Manually read a page from disk.
    pub fn read(&mut self, page_number: PageNumber, buf: &mut [u8]) -> io::Result<usize> {
        self.io.read(page_number, buf)
    }
}

impl<I: Seek + Write> Pager<I> {
    /// Manually write a page to disk.
    pub fn write(&mut self, page_number: PageNumber, buf: &[u8]) -> io::Result<usize> {
        self.io.write(page_number, buf)
    }

    /// Writes all the pages present in the dirty queue and marks them as clean.
    ///
    /// Changes might not be reflected unless [`Self::flush`] and [`Self::sync`]
    /// are called.
    pub fn write_dirty_pages(&mut self) -> io::Result<()> {
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
}

impl<I: Write> Pager<I> {
    /// Flush buffered writes.
    ///
    /// See [`super::io::Sync`] for details.
    pub fn flush(&mut self) -> io::Result<()> {
        self.io.flush()
    }
}

impl<I: super::io::Sync> Pager<I> {
    /// Ensure writes reach their destination.
    ///
    /// See [`super::io::Sync`] for details.
    pub fn sync(&self) -> io::Result<()> {
        self.io.sync()
    }
}

impl<I: Seek + Read + Write> Pager<I> {
    /// Initialize the database file.
    pub fn init(&mut self) -> io::Result<()> {
        // Manually read one block without involving the cache system, because
        // if the DB file already exists we might have to set the page size to
        // that defined in the file.
        let (magic, page_size) = {
            let mut page_zero = PageZero::alloc_in_memory(0, self.block_size);
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
        // insert statements just like MySQL or any other database does.
        if magic.swap_bytes() == MAGIC {
            panic!("the database file has been created using a different endianness than the one used by this machine");
        }

        // Initialize page zero.
        let page_zero = PageZero::alloc_in_memory(0, self.page_size);
        self.write(0, page_zero.as_ref())?;

        Ok(())
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
    fn lookup<P: Into<MemPage> + AllocPageInMemory + AsMut<[u8]>>(
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
    /// since we need a specific lifetime).
    pub fn get_as<'p, P>(&'p mut self, page_number: PageNumber) -> io::Result<&P>
    where
        P: Into<MemPage> + AllocPageInMemory + AsMut<[u8]>,
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
        P: Into<MemPage> + AllocPageInMemory + AsMut<[u8]>,
        &'p mut P: TryFrom<&'p mut MemPage>,
        <&'p mut P as TryFrom<&'p mut MemPage>>::Error: Debug,
    {
        let index = self.lookup::<P>(page_number)?;
        self.push_to_write_queue(page_number);

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
        for i in 0..N {
            self.lookup::<Page>(pages[i])?;
        }

        // Couldn't cache all pages, bail out. Ideally we should use pin() and
        // then once we have all the space unpin() and return the refs, but that
        // could introduce some hard to spot bugs.
        if pages.iter().any(|page| !self.cache.contains(page)) {
            return Ok(None);
        }

        for page in pages {
            self.push_to_write_queue(page);
        }

        Ok(self
            .cache
            .get_many_mut(pages)
            .map(|pages| pages.map(|page| page.try_into().expect("page type conversion error"))))
    }

    /// Loads the given page into the cache buffer.
    ///
    /// If the cache evicts a dirty page we use the opportunity to write all
    /// the dirty pages that we currenlty track. This doesn't mean that the
    /// dirty pages will be written to disk, they might be buffered by the
    /// underlying OS until [`Self::sync`] is called.
    fn load_page_into_cache(&mut self, page: impl Into<MemPage>) -> io::Result<()> {
        if self.cache.must_evict_dirty_page() {
            self.write_dirty_pages()?;
        }

        self.cache.load(page.into());

        Ok(())
    }

    /// Loads a page from disk into the cache.
    ///
    /// The page is not marked dirty, it will not be written back to disk
    /// unless [`Self::get_mut_as`] is called.
    fn load_from_disk<P: Into<MemPage> + AllocPageInMemory + AsMut<[u8]>>(
        &mut self,
        page_number: PageNumber,
    ) -> io::Result<()> {
        let mut page = P::alloc_in_memory(page_number, self.page_size);
        self.io.read(page_number, page.as_mut())?;
        self.load_page_into_cache(page)
    }

    /// Loads a page created in memory into the cache.
    ///
    /// Doing so automatically marks the page as dirty, since it's gonna have
    /// to be written to disk at some point.
    pub fn load_from_mem(&mut self, page: impl Into<MemPage>) -> io::Result<()> {
        let page: MemPage = page.into();
        let page_number = page.number();

        self.load_page_into_cache(page)?;
        self.push_to_write_queue(page_number);

        Ok(())
    }

    /// Initializes the contents of a disk page.
    ///
    /// This does not immediately write to disk as it would be inefficient.
    /// Instead, the initialization goes through the cache system. The page
    /// is initialized in memory, cached and pushed to the write queue.
    /// Eventually, the page will be written to disk.
    pub fn init_disk_page<P: Into<MemPage> + AllocPageInMemory>(
        &mut self,
        page_number: PageNumber,
    ) -> io::Result<()> {
        self.load_from_mem(P::alloc_in_memory(page_number, self.page_size))
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

    /// Allocates a new page that can be used to write data.
    pub fn alloc_page(&mut self) -> io::Result<PageNumber> {
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
            page.number
        };

        if header.first_free_page == 0 {
            header.last_free_page = 0;
        }

        self.write_header(header)?;

        Ok(free_page)
    }

    /// Adds the given page to the free list.
    ///
    /// **Important**: do not use the page after calling this function, since it
    /// will be replaced by a free page and all the data will be lost. Consider
    /// that a "use after free" bug.
    pub fn free_page(&mut self, page_number: PageNumber) -> io::Result<()> {
        // Initialize last free page.
        self.load_from_mem(FreePage::alloc_in_memory(page_number, self.page_size))?;

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
}

#[cfg(test)]
mod tests {
    use std::io;

    use super::Pager;
    use crate::{
        paging::{cache::Cache, io::MemBuf},
        storage::page::{AllocPageInMemory, Cell, OverflowPage, Page},
    };

    fn init_pager_with_cache(cache: Cache) -> io::Result<Pager<MemBuf>> {
        let mut pager = Pager::with_cache(io::Cursor::new(Vec::new()), 256, 256, cache);
        pager.init()?;

        Ok(pager)
    }

    fn init_pager() -> io::Result<Pager<MemBuf>> {
        init_pager_with_cache(Cache::default())
    }

    #[test]
    fn alloc_page() -> io::Result<()> {
        let mut pager = init_pager()?;

        for i in 1..=10 {
            assert_eq!(pager.alloc_page()?, i);
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
        let mut pager = init_pager()?;

        for _ in 1..=10 {
            pager.alloc_page()?;
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
        let mut pager = init_pager()?;

        for i in 1..=10 {
            let mut page = Page::alloc_in_memory(pager.alloc_page()?, pager.page_size);
            page.push(Cell::new(vec![
                i;
                Page::ideal_max_payload_size(pager.page_size, 1)
                    as usize
            ]));

            pager.write(page.number, page.as_ref())?;
        }

        let update_pages = [5, 7, 9];

        for p in &update_pages {
            pager.get_mut(*p)?.cell_mut(0).content.fill(10 + *p as u8);
        }

        pager.write_dirty_pages()?;
        pager.flush()?;
        pager.sync()?;

        for i in 1..=10 {
            let mut expected = Page::alloc_in_memory(i, pager.page_size);
            expected.push(Cell::new(vec![
                if update_pages.contains(&i) {
                    10 + i as u8
                } else {
                    i as u8
                };
                Page::ideal_max_payload_size(pager.page_size, 1)
                    as usize
            ]));

            let mut page = Page::alloc_in_memory(i, pager.page_size);
            pager.read(page.number, page.as_mut())?;

            assert_eq!(page, expected);
        }

        assert!(!pager.cache.must_evict_dirty_page());

        Ok(())
    }

    #[test]
    fn write_pages_when_dirty_page_is_evicted() -> io::Result<()> {
        let mut pager = init_pager_with_cache(Cache::with_max_size(3))?;

        for i in 1..=3 {
            let mut page = OverflowPage::alloc_in_memory(i, pager.page_size);
            page.content_mut().fill(i as u8);
            pager.load_from_mem(page)?;
        }

        let mut causes_evict = OverflowPage::alloc_in_memory(4, pager.page_size);
        causes_evict.content_mut().fill(4);

        pager.load_from_mem(causes_evict)?;

        for i in 1..=3 {
            let mut page = OverflowPage::alloc_in_memory(i, pager.page_size);
            pager.read(i, page.as_mut())?;

            let mut expected = OverflowPage::alloc_in_memory(i, pager.page_size);
            expected.content_mut().fill(i as u8);

            assert_eq!(expected, page);
        }

        assert!(!pager.cache.must_evict_dirty_page());

        Ok(())
    }

    #[test]
    fn get_many_mut_ok() -> io::Result<()> {
        let mut pager = init_pager_with_cache(Cache::with_max_size(3))?;

        let mut cells = Vec::new();

        for i in 1..=3 {
            let mut page = Page::alloc_in_memory(i, pager.page_size);
            let cell = Cell::new(Vec::from(&i.to_le_bytes()));
            page.push(cell.clone());
            cells.push(cell);
            pager.load_from_mem(page)?;
        }

        let mut_refs = pager.get_many_mut([1, 2, 3])?;

        assert!(mut_refs.is_some());

        for ((mut_ref, page_num), cell) in mut_refs.unwrap().into_iter().zip(1..=3).zip(cells) {
            assert_eq!(mut_ref.number, page_num);
            assert_eq!(mut_ref.len(), 1);
            assert_eq!(mut_ref.cell(0), cell.as_ref());
        }

        Ok(())
    }

    #[test]
    fn get_many_mut_fail() -> io::Result<()> {
        let mut pager = init_pager_with_cache(Cache::with_max_size(3))?;

        pager.load_from_mem(Page::alloc_in_memory(0, pager.page_size))?;
        pager.cache.pin(0);

        let mut_refs = pager.get_many_mut([1, 2, 3])?;

        assert!(mut_refs.is_none());

        Ok(())
    }
}
