//! IO pager implementation.

use std::{
    io::{self, Read, Seek, SeekFrom, Write},
    ptr,
};

use crate::{database::Header, storage::page::FreePage};

/// Are we gonna have more than 4 billion pages? Probably not ¯\_(ツ)_/¯
pub(crate) type PageNumber = u32;

/// IO block device page manager.
pub(crate) struct Pager<I> {
    /// Underlying IO resource handle.
    io: I,
    /// Hardware block size or prefered IO read/write buffer size.
    pub block_size: usize,
    /// High level page size.
    pub page_size: usize,
}

impl<I> Pager<I> {
    /// Creates a new pager on top of `io`. `block_size` should evenly divide
    /// `page_size` or viceversa. Ideally, both should be powers of 2, but it's
    /// convinient to support any size for testing.
    pub fn new(io: I, page_size: usize, block_size: usize) -> Self {
        Self {
            io,
            block_size,
            page_size,
        }
    }

    /// Some sanity checks for development.
    fn assert_args_are_correct(&self, page_number: PageNumber, buf: &[u8]) {
        debug_assert!(
            buf.len() == self.page_size,
            "Buffer of incorrect length {} given for page size {}",
            buf.len(),
            self.page_size
        );

        // Used for development/debugging in case we mess up. Don't wanna create
        // giant Gigabyte sized files all of a sudden.
        debug_assert!(page_number < 1000, "Page number too high: {page_number}");
    }
}

impl<F: Seek + Read> Pager<F> {
    /// Reads the raw bytes of the page in memory without doing anything else.
    /// Simplest case is when [`Self::page_size`] >= [`Self::block_size`]. For
    /// example, suppose `block_size = 512` and `page_size = 1024`:
    ///
    /// ```text
    /// OFFSET      BLOCKS
    ///         +-------------+
    ///       0 | +---------+ |
    ///         | | BLOCK 0 | |
    ///         | +---------+ | PAGE 0
    ///     512 | +---------+ |
    ///         | | BLOCK 1 | |
    ///         | +---------+ |
    ///         +-------------+
    ///
    ///         +-------------+
    ///    1024 | +---------+ |
    ///         | | BLOCK 2 | |
    ///         | +---------+ | PAGE 1
    ///    1536 | +---------+ |
    ///         | | BLOCK 3 | |
    ///         | +---------+ |
    ///         +-------------+
    /// ```
    ///
    /// Finding a page is as simple as computing `page_number * page_size`. The
    /// second, less trivial case, is when `block_size > page_size`, because
    /// many pages can fit into one block. Suppose `block_size = 4096` and
    /// `page_size = 1024`.
    ///
    /// ```text
    /// BLOCK     PAGE
    /// OFFSET    OFFSET    BLOCKS
    ///
    ///   0 ------->   +-------------+
    ///              0 | +---------+ |
    ///                | | PAGE  0 | |
    ///                | +---------+ |
    ///           1024 | +---------+ |
    ///                | | PAGE  1 | |
    ///                | +---------+ | BLOCK 0
    ///           2048 | +---------+ |
    ///                | | PAGE  2 | |
    ///                | +---------+ |
    ///           3072 | +---------+ |
    ///                | | PAGE  3 | |
    ///                | +---------+ |
    ///                +-------------+
    ///
    /// 4096 ------>   +-------------+
    ///           4096 | +---------+ |
    ///                | | PAGE  4 | |
    ///                | +---------+ |
    ///           5120 | +---------+ |
    ///                | | PAGE  5 | |
    ///                | +---------+ | BLOCK 0
    ///           6144 | +---------+ |
    ///                | | PAGE  6 | |
    ///                | +---------+ |
    ///           7168 | +---------+ |
    ///                | | PAGE  7 | |
    ///                | +---------+ |
    ///                +-------------+
    /// ```
    ///
    /// In this case we'll have to compute the page offset and align downwards
    /// to block size. For example, if we want page 6, first we compute
    /// `1024 * 6 = 6144` and then align `6144` downwards to `4096` to read the
    /// block in memory. Alignment on powers of two can be computed using XOR
    /// and a bitmask. Check [address alignment] for more details.
    ///
    /// [address alignment]: https://os.phil-opp.com/allocator-designs/#address-alignment
    pub fn read(&mut self, page_number: PageNumber, buf: &mut [u8]) -> io::Result<usize> {
        self.assert_args_are_correct(page_number, buf);

        // Compute block offset and inner page offset.
        let (capacity, block_offset, inner_offset) = {
            let page_number = page_number as usize;
            let Self {
                page_size,
                block_size,
                ..
            } = *self;

            if page_size >= block_size {
                (page_size, page_size * page_number, 0)
            } else {
                let offset = (page_number * page_size) & !(block_size - 1);
                (block_size, offset, page_number * page_size - offset)
            }
        };

        // Spin the disk... or let SSD transistors go brrr.
        self.io.seek(SeekFrom::Start(block_offset as u64))?;

        // Read page into memory.
        if self.page_size >= self.block_size {
            return self.io.read(buf);
        }

        // If the block size is greater than page size, we're reading multiple
        // pages in one call. TODO: Find a way to cache all the pages, not just
        // one.
        let mut block = vec![0; capacity];
        let _ = self.io.read(&mut block)?;
        buf.copy_from_slice(&block[inner_offset..inner_offset + self.page_size]);

        Ok(self.page_size)
    }
}

impl<F: Seek + Write> Pager<F> {
    /// Writes the page to disk. See also [`Self::read`] for more details.
    pub fn write(&mut self, page_number: PageNumber, buf: &[u8]) -> io::Result<usize> {
        self.assert_args_are_correct(page_number, buf);

        // TODO: Just like [`Self::read`], when the block size is greater than
        // the page size we should be writing multiple pages at once.
        let offset = self.page_size * page_number as usize;
        self.io.seek(SeekFrom::Start(offset as u64))?;

        // TODO: If page_size > block_size check if all blocks need to be written
        self.io.write(buf)
    }
}

impl<F: Seek + Read + Write> Pager<F> {
    pub fn read_header(&mut self) -> io::Result<Header> {
        // TODO: Cache header.
        let mut buf = vec![0; self.page_size];
        self.read(0, &mut buf)?;

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

        self.write(0, &buf).map(|_| ())
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
            self.read(page.number, page.buffer_mut())?;
            header.first_free_page = page.header().next;
            page.number
        };

        self.write_header(header)?;

        Ok(free_page)
    }

    pub fn free_page(&mut self, page_number: PageNumber) -> io::Result<()> {
        // Initialize last free page.
        let last_free = FreePage::new(page_number, self.page_size as _);
        self.write(last_free.number, last_free.buffer())?;

        let mut header = self.read_header()?;

        if header.first_free_page == 0 {
            header.first_free_page = page_number;
            return self.write_header(header);
        }

        let mut free_page = FreePage::new(header.first_free_page, self.page_size as _);
        self.read(free_page.number, free_page.buffer_mut())?;

        while free_page.header().next != 0 {
            free_page = FreePage::new(free_page.header().next, self.page_size as _);
            self.read(free_page.number, free_page.buffer_mut())?;
        }

        free_page.header_mut().next = page_number;
        self.write(free_page.number, free_page.buffer())?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::io;

    use super::Pager;

    #[test]
    fn pager_write_read() -> io::Result<()> {
        let sizes = [(4, 4), (4, 16), (16, 4)];

        for (page_size, block_size) in sizes {
            let max_pages = 10;

            let buf = io::Cursor::new(Vec::new());
            let mut pager = Pager::new(buf, page_size, block_size);

            for i in 1..=max_pages {
                let expected = vec![i as u8; page_size];
                let mut buf = vec![0; page_size];

                assert_eq!(pager.write(i - 1, &expected)?, page_size);
                assert_eq!(pager.read(i - 1, &mut buf)?, buf.len());
                assert_eq!(buf, expected);
            }
        }

        Ok(())
    }
}
