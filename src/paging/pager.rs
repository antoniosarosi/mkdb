//! IO pager implementation.

use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashSet},
    io::{self, Read, Seek, Write},
    ptr,
};

use super::{
    cache::{Cache, EvictedPage},
    io::BlockIo,
};
use crate::{
    database::Header,
    storage::page::{FreePage, Page},
};

/// Are we gonna have more than 4 billion pages? Probably not ¯\_(ツ)_/¯
pub(crate) type PageNumber = u32;

/// IO block device page manager.
pub(crate) struct Pager<I> {
    /// Underlying IO resource handle.
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

impl<F: Seek + Read + Write> Pager<F> {
    pub fn write(&mut self, page_number: PageNumber, buf: &[u8]) -> io::Result<usize> {
        self.io.write(page_number, buf)
    }

    pub fn read(&mut self, page_number: PageNumber, buf: &mut [u8]) -> io::Result<usize> {
        self.io.read(page_number, buf)
    }

    pub fn get(&mut self, page_number: PageNumber) -> io::Result<&Page> {
        if let Some(index) = self.cache.get(page_number) {
            return Ok(&self.cache[index]);
        }

        self.load_from_disk(page_number)?;
        let index = self.cache.get(page_number).unwrap();
        Ok(&self.cache[index])
    }

    pub fn get_mut(&mut self, page_number: PageNumber) -> io::Result<&mut Page> {
        self.dirty_pages.insert(page_number);

        if let Some(index) = self.cache.get_mut(page_number) {
            return Ok(&mut self.cache[index]);
        }

        self.load_from_disk(page_number)?;
        let index = self.cache.get_mut(page_number).unwrap();
        Ok(&mut self.cache[index])
    }

    fn load_from_disk(&mut self, page_number: PageNumber) -> io::Result<()> {
        let mut page = Page::new(page_number, self.page_size as _);
        self.io.read(page_number, page.buffer_mut())?;
        self.load_from_mem(page)
    }

    pub fn load_from_mem(&mut self, page: Page) -> io::Result<()> {
        if let Some(EvictedPage { dirty: true, page }) = self.cache.load(page) {
            self.write_dirty_pages()?;
            self.io.write(page.number, page.buffer())?;
            self.flush()?;
        }

        Ok(())
    }

    pub fn write_dirty_pages(&mut self) -> io::Result<()> {
        let page_numbers = BinaryHeap::from_iter(self.dirty_pages.iter().copied().map(Reverse));

        for Reverse(page_number) in page_numbers {
            let index = self.cache.get(page_number).unwrap();
            let page = &self.cache[index];
            self.io.write(page_number, page.buffer())?;
            self.cache.mark_clean(page_number);
        }

        self.dirty_pages.clear();

        Ok(())
    }

    pub fn read_header(&mut self) -> io::Result<Header> {
        // TODO: Cache header.
        let mut buf = vec![0; self.page_size];
        self.io.read(0, &mut buf)?;

        // SAFETY: Unless somebody manually touched the DB file this should be
        // safe.
        unsafe { Ok(ptr::read(buf.as_ptr().cast())) }
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
            page.number
        };

        self.write_header(header)?;

        Ok(free_page)
    }

    pub fn free_page(&mut self, page_number: PageNumber) -> io::Result<()> {
        // Initialize last free page.
        let last_free = FreePage::new(page_number, self.page_size as _);
        self.io.write(last_free.number, last_free.buffer())?;

        let mut header = self.read_header()?;

        if header.first_free_page == 0 {
            header.first_free_page = page_number;
            return self.write_header(header);
        }

        let mut free_page = FreePage::new(header.first_free_page, self.page_size as _);
        self.io.read(free_page.number, free_page.buffer_mut())?;

        while free_page.header().next != 0 {
            free_page = FreePage::new(free_page.header().next, self.page_size as _);
            self.io.read(free_page.number, free_page.buffer_mut())?;
        }

        free_page.header_mut().next = page_number;
        self.io.write(free_page.number, free_page.buffer())?;

        Ok(())
    }

    pub fn flush(&mut self) -> io::Result<()> {
        self.io.flush()
    }
}

#[cfg(test)]
mod tests {}
