//! Pager implementation.
//!
//! This module contains the public API (for the crate) to access pages on disk.
//! Pages are also cached in memory and the implementation takes advantage of
//! the Rust type system to automatically send pages that are acquired using
//! `&mut` to a write queue.

use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashSet},
    io::{self, Read, Seek, Write},
    ptr,
};

use super::{cache::Cache, io::BlockIo};
use crate::storage::{
    page::{FreePage, Page},
    Header, MAGIC,
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

    pub fn read_header(&mut self) -> io::Result<Header> {
        // TODO: Cache header.
        let mut buf = vec![0; self.page_size];
        self.io.read(0, &mut buf)?;

        // SAFETY: Unless somebody manually touched the DB file this should be
        // safe.
        unsafe { Ok(ptr::read(buf.as_ptr().cast())) }
    }
}

impl<I: Seek + Write> Pager<I> {
    /// Manually write a page to disk.
    pub fn write(&mut self, page_number: PageNumber, buf: &[u8]) -> io::Result<usize> {
        self.io.write(page_number, buf)
    }

    pub fn write_header(&mut self, header: Header) -> io::Result<()> {
        // TODO: Cache header.
        let mut buf = Vec::with_capacity(self.page_size);
        buf.resize(self.page_size, 0);

        // SAFETY: Buffer has enough space, page_size should always be greater
        // than size_of<Header>()
        unsafe { ptr::write(buf.as_mut_ptr().cast(), header) };

        self.io.write(0, &buf).map(|_| ())
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
            self.io.write(page_number, page.buffer())?;
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
    pub fn init(&mut self) -> io::Result<()> {
        // TODO: Recreate the pager if the file exists and the header page size
        // is different than the default.
        if self.read_header()?.magic != MAGIC {
            self.write_header(Header {
                magic: MAGIC,
                page_size: self.page_size as _,
                total_pages: 1,
                free_pages: 0,
                first_free_page: 0,
                last_free_page: 0,
            })?;
        }

        Ok(())
    }

    /// Returns a read-only reference to a page.
    ///
    /// Best case scenario is when the page is already cached in memory. Worst
    /// case scenario is when we have to evict a dirty page in order to load
    /// the new one, which requires at least two IO operations.
    pub fn get(&mut self, page_number: PageNumber) -> io::Result<&Page> {
        if let Some(index) = self.cache.get(page_number) {
            return Ok(&self.cache[index]);
        }

        self.load_from_disk(page_number)?;
        let index = self.cache.get(page_number).unwrap();
        Ok(&self.cache[index])
    }

    /// Same as [`Self::get`] but marks the page as dirty and sends it to a
    /// write queue.
    ///
    /// This function should not be used unless the page is actually about to
    /// be modified, since otherwise the page will be written to disk
    /// unnecessarily at some point.
    pub fn get_mut(&mut self, page_number: PageNumber) -> io::Result<&mut Page> {
        self.dirty_pages.insert(page_number);

        if let Some(index) = self.cache.get_mut(page_number) {
            return Ok(&mut self.cache[index]);
        }

        self.load_from_disk(page_number)?;
        let index = self.cache.get_mut(page_number).unwrap();
        Ok(&mut self.cache[index])
    }

    /// Loads a page from disk into the cache.
    fn load_from_disk(&mut self, page_number: PageNumber) -> io::Result<()> {
        let mut page = Page::new(page_number, self.page_size as _);
        self.io.read(page_number, page.buffer_mut())?;
        self.load_from_mem(page)
    }

    /// Loads a page created in memory into the cache.
    pub fn load_from_mem(&mut self, page: Page) -> io::Result<()> {
        if self.cache.must_evict_dirty_page() {
            self.write_dirty_pages()?;
        }

        self.cache.load(page);

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
            let mut page = FreePage::new(header.first_free_page, self.page_size as _);
            self.io.read(page.number, page.buffer_mut())?;
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
    pub fn free_page(&mut self, page_number: PageNumber) -> io::Result<()> {
        // Initialize last free page.
        let new_last_free = FreePage::new(page_number, self.page_size as _);
        self.io.write(page_number, new_last_free.buffer())?;

        let mut header = self.read_header()?;

        if header.first_free_page == 0 {
            // No previous free pages, initialize freelist.
            header.first_free_page = page_number;
        } else {
            // Grab the last free and make it point to the new last free.
            let mut last_free = FreePage::new(header.last_free_page, self.page_size as _);
            self.io.read(last_free.number, last_free.buffer_mut())?;

            last_free.header_mut().next = page_number;
            self.io.write(last_free.number, last_free.buffer())?;
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
        storage::page::{Cell, Page},
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
            let mut page = Page::new(pager.alloc_page()?, pager.page_size as _);
            page.push(Cell::new(vec![
                i;
                Page::max_payload_size(pager.page_size as _)
                    as usize
            ]));
            pager.write(page.number, page.buffer())?;
        }

        let update_pages = [5, 7, 9];

        for p in &update_pages {
            pager.get_mut(*p)?.cell_mut(0).content.fill(10 + *p as u8);
        }

        pager.write_dirty_pages()?;
        pager.flush()?;
        pager.sync()?;

        for i in 1..=10 {
            let mut expected = Page::new(i, pager.page_size as _);
            expected.push(Cell::new(vec![
                if update_pages.contains(&i) {
                    10 + i as u8
                } else {
                    i as u8
                };
                Page::max_payload_size(pager.page_size as _)
                    as usize
            ]));

            let mut page = Page::new(i, pager.page_size as _);
            pager.read(page.number, page.buffer_mut())?;

            assert_eq!(page, expected);
        }

        assert!(!pager.cache.must_evict_dirty_page());

        Ok(())
    }
}
