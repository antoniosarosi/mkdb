use std::io::{self, Read, Seek, SeekFrom, Write};

use super::pager::PageNumber;

pub(super) struct BlockIo<I> {
    io: I,
    pub block_size: usize,
    pub page_size: usize,
}

impl<I> BlockIo<I> {
    pub fn new(io: I, page_size: usize, block_size: usize) -> Self {
        Self {
            io,
            block_size,
            page_size,
        }
    }

    /// Some sanity checks for development.
    fn debug_assert_args_are_correct(&self, page_number: PageNumber, buf: &[u8]) {
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

impl<I: Seek + Read> BlockIo<I> {
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
        self.debug_assert_args_are_correct(page_number, buf);

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

impl<I: Seek + Write> BlockIo<I> {
    /// Writes the page to disk. See also [`Self::read`] for more details.
    pub fn write(&mut self, page_number: PageNumber, buf: &[u8]) -> io::Result<usize> {
        self.debug_assert_args_are_correct(page_number, buf);

        // TODO: Just like [`Self::read`], when the block size is greater than
        // the page size we should be writing multiple pages at once.
        let offset = self.page_size * page_number as usize;
        self.io.seek(SeekFrom::Start(offset as u64))?;

        // TODO: If page_size > block_size check if all blocks need to be written
        self.io.write(buf)
    }
}

impl<I: Write> BlockIo<I> {
    pub fn flush(&mut self) -> io::Result<()> {
        self.io.flush()
    }
}

#[cfg(test)]
mod tests {
    use std::io;

    use super::BlockIo;

    #[test]
    fn block_io() -> io::Result<()> {
        let sizes = [(4, 4), (4, 16), (16, 4)];

        for (page_size, block_size) in sizes {
            let max_pages = 10;

            let buf = io::Cursor::new(Vec::new());

            let mut io = BlockIo {
                io: buf,
                page_size,
                block_size,
            };

            for i in 1..=max_pages {
                let expected = vec![i as u8; page_size];
                let mut buf = vec![0; page_size];

                assert_eq!(io.write(i - 1, &expected)?, page_size);
                assert_eq!(io.read(i - 1, &mut buf)?, buf.len());
                assert_eq!(buf, expected);
            }
        }

        Ok(())
    }
}
