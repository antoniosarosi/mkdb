//! Pager implementation.
//!
//! This module contains the public API (for the crate) to access pages on disk.
//! Pages are also cached in memory and the implementation takes advantage of
//! the Rust type system to automatically send pages that are acquired using
//! `&mut` to a write queue.

use std::{
    any::{Any, TypeId},
    cmp::Reverse,
    collections::{BinaryHeap, HashMap, HashSet},
    io::{self, Read, Seek, Write},
};

use super::{cache::Cache, io::BlockIo};
use crate::storage::page::{
    DbHeader, FreePage, InitEmptyPage, MemPage, OverflowPage, Page, PageZero, MAGIC,
};

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
    /// Creates a new pager on top of `io`. `block_size` should evenly divide
    /// `page_size` or viceversa. Ideally, both should be powers of 2, but it's
    /// convinient to support any size for testing.
    pub fn new(io: I, page_size: usize, block_size: usize) -> Self {
        Self {
            io: BlockIo::new(io, page_size, block_size),
            block_size,
            page_size,
            cache: Cache::new(),
            dirty_pages: HashSet::new(),
        }
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
    fn sync(&self) -> io::Result<()> {
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
            let mut page_zero = PageZero::init(0, self.block_size);
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
        let page_zero = PageZero::init(0, self.page_size);
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
    fn lookup<P: Into<MemPage> + InitEmptyPage + AsMut<[u8]>>(
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
    /// We're currently using [`Any`] to downcast to to the concrete type at
    /// runtime, which is definitely not "elegant" or "good practice", but it
    /// works and it doesn't require a lot of duplicated code or macros.
    ///
    /// We've managed to keep this project free of dependencies so far, so we're
    /// not gonna introduce one just for this feature, but we could use
    /// [`enum_dispatch`](https://docs.rs/enum_dispatch/) which basically
    /// automates what we're doing here using [`TryFrom`] for each enum member.
    ///
    /// Implementing [`TryFrom`] manually requires two impls per member, one for
    /// `&P` and another one for `&mut P`, and possibly a third one for `P`
    /// which we might use in tests. So for now `downcast_ref` is good enough to
    /// reduce boilerplate.
    pub fn get_as<P: Any + 'static + Into<MemPage> + InitEmptyPage + AsMut<[u8]>>(
        &mut self,
        page_number: PageNumber,
    ) -> io::Result<&P> {
        let index = self.lookup::<P>(page_number)?;

        let downcast = <dyn Any>::downcast_ref(match &self.cache[index] {
            MemPage::Btree(page) => page,
            MemPage::Overflow(overflow_page) => overflow_page,
            MemPage::Zero(page_zero) => {
                if TypeId::of::<P>() == TypeId::of::<Page>() {
                    page_zero.as_btree_page()
                } else {
                    page_zero
                }
            }
        });

        if cfg!(debug_assertions) {
            let types = HashMap::from([
                (TypeId::of::<Page>(), "Page"),
                (TypeId::of::<PageZero>(), "PageZero"),
                (TypeId::of::<OverflowPage>(), "OverflowPage"),
            ]);

            if !types.contains_key(&TypeId::of::<P>()) {
                panic!("get_as() called with invalid generic type");
            }

            if downcast.is_none() {
                panic!(
                    "attempt to read page {page_number} of type {:?} as type {}",
                    &self.cache[index],
                    types.get(&TypeId::of::<P>()).unwrap()
                );
            }
        }

        Ok(downcast.expect("page type error"))
    }

    /// Sames as [`Self::get_as`] but returns a mutable reference.
    pub fn get_mut_as<P: Any + 'static + Into<MemPage> + InitEmptyPage + AsMut<[u8]>>(
        &mut self,
        page_number: PageNumber,
    ) -> io::Result<&mut P> {
        self.dirty_pages.insert(page_number);

        let index = self.lookup::<P>(page_number)?;

        // It's easier to verify the type using read-only references. We'll just
        // call get_as until we find some better solution to deal with page
        // types.
        if cfg!(debug_assertions) {
            self.get_as::<P>(page_number)?;
        }

        let downcast = <dyn Any>::downcast_mut(match &mut self.cache[index] {
            MemPage::Btree(page) => page,
            MemPage::Overflow(overflow_page) => overflow_page,
            MemPage::Zero(page_zero) => {
                if TypeId::of::<P>() == TypeId::of::<Page>() {
                    page_zero.as_btree_page_mut()
                } else {
                    page_zero
                }
            }
        });

        Ok(downcast.expect("page type error"))
    }

    /// Returns a read-only reference to a BTree page.
    ///
    /// BTree page are the ones used most frequently, so we'll consider this
    /// function the default of [`Self::get_as`].
    pub fn get(&mut self, page_number: PageNumber) -> io::Result<&Page> {
        self.get_as(page_number)
    }

    /// Default return type for [`Self::get_mut_as`].
    pub fn get_mut(&mut self, page_number: PageNumber) -> io::Result<&mut Page> {
        self.get_mut_as(page_number)
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
    fn load_from_disk<P: Into<MemPage> + InitEmptyPage + AsMut<[u8]>>(
        &mut self,
        page_number: PageNumber,
    ) -> io::Result<()> {
        let mut page = P::init(page_number, self.page_size);
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
        self.cache.mark_dirty(page_number);

        Ok(())
    }

    pub fn init_page<P: Into<MemPage> + InitEmptyPage + AsMut<[u8]>>(
        &mut self,
        page_number: PageNumber,
    ) -> io::Result<()> {
        self.replace_page(page_number, P::init(page_number, self.page_size))
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

    /// Replaces the page at `page_number` with the given page.
    ///
    /// If the page to replace is already cached, then this is almost a no-op
    /// as we only have to assign the value. If the page is on disk then we
    /// load the replacement into the cache (evicting another page if necessary)
    /// and mark it as dirty, and at some point in the future it will be written
    /// to disk.
    fn replace_page<P: Into<MemPage> + InitEmptyPage + AsMut<[u8]>>(
        &mut self,
        page_number: PageNumber,
        with: P,
    ) -> io::Result<()> {
        if let Some(index) = self.cache.get(page_number) {
            self.cache[index] = with.into();
        } else {
            self.load_from_mem(with)?;
        }

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
    /// Important: do not use the page after calling this function, since it
    /// will be replaced by a free page and all the data will be lost. Consider
    /// that a "use after free" bug.
    pub fn free_page(&mut self, page_number: PageNumber) -> io::Result<()> {
        // Initialize last free page.
        self.init_page::<FreePage>(page_number)?;

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
        paging::io::MemBuf,
        storage::page::{Cell, InitEmptyPage, Page},
    };

    fn init_pager() -> io::Result<Pager<MemBuf>> {
        let mut pager = Pager::new(io::Cursor::new(Vec::new()), 256, 256);
        pager.init()?;

        Ok(pager)
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
            let mut page = Page::init(pager.alloc_page()?, pager.page_size);
            page.push(Cell::new(vec![
                i;
                Page::ideal_max_payload_size(pager.page_size)
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
            let mut expected = Page::init(i, pager.page_size);
            expected.push(Cell::new(vec![
                if update_pages.contains(&i) {
                    10 + i as u8
                } else {
                    i as u8
                };
                Page::ideal_max_payload_size(pager.page_size)
                    as usize
            ]));

            let mut page = Page::init(i, pager.page_size);
            pager.read(page.number, page.as_mut())?;

            assert_eq!(page, expected);
        }

        assert!(!pager.cache.must_evict_dirty_page());

        Ok(())
    }
}
