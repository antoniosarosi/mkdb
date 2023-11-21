use std::io;

pub(crate) struct Pager<F> {
    file: F,
    block_size: usize,
    page_size: usize,
}

impl<F> Pager<F> {
    pub fn new(file: F, page_size: usize, block_size: usize) -> Self {
        Self {
            file,
            block_size,
            page_size,
        }
    }

    #[inline]
    pub fn page_size(&self) -> usize {
        self.page_size
    }
}

impl<F: io::Seek> Pager<F> {
    pub fn read_page(&mut self, page_number: usize) -> io::Result<Vec<u8>>
    where
        F: io::Read,
    {
        // page_size > block_size, compute offset normally
        let mut capacity = self.page_size;
        let mut offset = self.page_size * page_number;
        let mut inner_offset = 0;

        // Used for development/debugging in case we mess up. Remove later.
        if page_number > 1000 {
            panic!("Page number too high: {page_number}");
        }
        if self.page_size < self.block_size {
            capacity = self.block_size;
            // Each block contains multiple pages, so align the offset downwards
            // to block alignment
            offset = page_number * self.page_size & !(self.block_size - 1);
            // Compute the page offset within the block
            inner_offset = page_number * self.page_size - offset;
        };

        self.file.seek(io::SeekFrom::Start(offset as u64))?;

        let mut buf = vec![0; capacity];

        self.file.read(&mut buf[..])?;

        // TODO: Cache the pages that were not read in this call
        if self.page_size < self.block_size {
            Ok(Vec::from(&buf[inner_offset..inner_offset + self.page_size]))
        } else {
            Ok(buf)
        }
    }

    pub fn write_page(&mut self, page_number: usize, buf: &Vec<u8>) -> io::Result<usize>
    where
        F: io::Write,
    {
        // TODO: Used for development/debugging in case we mess up. Remove later.
        if page_number > 1000 {
            panic!("Page number too high: {page_number}");
        }
        let offset = self.page_size * page_number;
        self.file.seek(io::SeekFrom::Start(offset as u64))?;

        // TODO: If page_size > block_size check if all blocks need to be written
        self.file.write(buf)
    }
}

#[cfg(test)]
mod tests {
    use std::{alloc::Layout, cmp::max, error::Error, io};

    use super::Pager;

    #[test]
    fn pager_write_read() -> Result<(), Box<dyn Error>> {
        let sizes = [(4, 4), (4, 16), (16, 4)];

        for (page_size, block_size) in sizes {
            let max_pages = 10;

            let size = page_size * max_pages;
            let align = max(page_size, block_size);
            let layout = Layout::from_size_align(size, align)?.pad_to_align();

            let buf = io::Cursor::new(vec![0; layout.size()]);
            let mut pager = Pager::new(buf, page_size, block_size);

            for i in 1..=max_pages {
                let page = vec![i as u8; page_size];
                assert_eq!(pager.write_page(i - 1, &page)?, page_size);
            }

            for i in 1..=max_pages {
                let expected = vec![i as u8; page_size];
                assert_eq!(pager.read_page(i - 1)?, expected);
            }
        }

        Ok(())
    }
}
