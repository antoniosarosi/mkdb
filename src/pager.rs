use std::io::{self, Read, Seek, SeekFrom, Write};

/// Are we gonna have more than 4 billion pages? Probably not ¯\_(ツ)_/¯
pub(crate) type PageNumber = u32;

/// Raw binary page as read from the disk.
#[derive(PartialEq, Debug)]
pub(crate) struct Page {
    /// Page number.
    pub number: PageNumber,
    /// Raw bytes.
    pub content: Vec<u8>,
}

/// IO device page manager.
pub(crate) struct Pager<I> {
    /// Underlying IO device handle.
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
    /// and a bitmask. Check [address alignment] for more details:
    ///
    /// [address alignment]: https://os.phil-opp.com/allocator-designs/#address-alignment
    pub fn read_page(&mut self, page_number: PageNumber) -> io::Result<Page> {
        // Used for development/debugging in case we mess up. Remove later.
        if page_number > 1000 {
            panic!("Page number too high: {page_number}");
        }

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
                let offset = page_number * page_size & !(block_size - 1);
                (block_size, offset, page_number * page_size - offset)
            }
        };

        // Spin the disk... or let SSD transistors go brrr.
        self.io.seek(SeekFrom::Start(block_offset as u64))?;

        // Read page into memory.
        let mut buf = vec![0; capacity];
        self.io.read(&mut buf[..])?;

        // If page_size < block_size remove the other pages from the buffer.
        // TODO: Cache or do something with the pages that were not read here.
        if self.page_size < self.block_size {
            buf = Vec::from(&buf[inner_offset..inner_offset + self.page_size])
        }

        Ok(Page {
            number: page_number,
            content: buf,
        })
    }

    /// Reads the raw page from disk and parses it into any type `T` that
    /// implements [`From<Page>`].
    pub fn read<T: From<Page>>(&mut self, page_number: u32) -> io::Result<T> {
        self.read_page(page_number).map(From::from)
    }
}

impl<F: Seek + Write> Pager<F> {
    /// Writes the page to disk. See also [`Self::read_page`] for more details.
    pub fn write_page(&mut self, page: Page) -> io::Result<usize> {
        // TODO: Used for development/debugging in case we mess up. Remove later.
        if page.number > 1000 {
            panic!("Page number too high: {}", page.number);
        }

        let offset = self.page_size * page.number as usize;
        self.io.seek(SeekFrom::Start(offset as u64))?;

        // TODO: If page_size > block_size check if all blocks need to be written
        self.io.write(&page.content)
    }

    /// Writes the in-memory representation of a page to disk. The generic type
    /// `T` needs to implement [`Into<Page>`].
    pub fn write<T: Into<Page>>(&mut self, mem_page: T) -> io::Result<usize> {
        self.write_page(mem_page.into())
    }
}

#[cfg(test)]
mod tests {
    use std::{error::Error, io};

    use super::Pager;
    use crate::pager::Page;

    #[test]
    fn pager_write_read() -> Result<(), Box<dyn Error>> {
        let sizes = [(4, 4), (4, 16), (16, 4)];

        for (page_size, block_size) in sizes {
            let max_pages = 10;

            let buf = io::Cursor::new(Vec::new());
            let mut pager = Pager::new(buf, page_size, block_size);

            for i in 1..=max_pages {
                let page = Page {
                    number: i - 1,
                    content: vec![i as u8; page_size],
                };
                assert_eq!(pager.write_page(page)?, page_size);
            }

            for i in 1..=max_pages {
                let expected = vec![i as u8; page_size];
                assert_eq!(
                    pager.read_page((i - 1) as u32)?,
                    Page {
                        number: i - 1,
                        content: vec![i as u8; page_size],
                    }
                );
            }
        }

        Ok(())
    }
}
