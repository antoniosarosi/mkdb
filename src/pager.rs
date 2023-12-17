use std::io;

use crate::node::{Entry, Node};

pub(crate) struct Pager<F> {
    file: F,
    pub block_size: usize,
    pub page_size: usize,
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

impl<F: io::Seek + io::Read> Pager<F> {
    pub fn read_page(&mut self, page_number: u32) -> io::Result<Vec<u8>> {
        let page_number = page_number as usize;

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

    pub fn read_node(&mut self, page: u32) -> io::Result<Node> {
        let buf = self.read_page(page)?;

        let mut node = Node::new_at(page);

        let mut i = 4;

        for _ in 0..u16::from_be_bytes(buf[..2].try_into().unwrap()) {
            let key = u32::from_be_bytes(buf[i..i + 4].try_into().unwrap());
            let value = u32::from_be_bytes(buf[i + 4..i + 8].try_into().unwrap());
            node.entries.push(Entry { key, value });
            i += 8;
        }

        for _ in 0..u16::from_be_bytes(buf[2..4].try_into().unwrap()) {
            node.children
                .push(u32::from_be_bytes(buf[i..i + 4].try_into().unwrap()));
            i += 4;
        }

        Ok(node)
    }
}

impl<F: io::Seek + io::Write> Pager<F> {
    pub fn write_page(&mut self, page_number: usize, buf: &Vec<u8>) -> io::Result<usize> {
        // TODO: Used for development/debugging in case we mess up. Remove later.
        if page_number > 1000 {
            panic!("Page number too high: {page_number}");
        }
        let offset = self.page_size * page_number;
        self.file.seek(io::SeekFrom::Start(offset as u64))?;

        // TODO: If page_size > block_size check if all blocks need to be written
        self.file.write(buf)
    }

    pub fn write_node(&mut self, node: &Node) -> io::Result<()> {
        let mut page = vec![0u8; self.page_size()];

        page[..2].copy_from_slice(&(node.entries.len() as u16).to_be_bytes());
        page[2..4].copy_from_slice(&(node.children.len() as u16).to_be_bytes());

        let mut i = 4;

        for entry in &node.entries {
            page[i..i + 4].copy_from_slice(&entry.key.to_be_bytes());
            page[i + 4..i + 8].copy_from_slice(&entry.value.to_be_bytes());
            i += 8;
        }

        for child in &node.children {
            page[i..i + 4].copy_from_slice(&(*child as u32).to_be_bytes());
            i += 4;
        }

        self.write_page(node.page as usize, &page)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::{error::Error, io};

    use super::Pager;

    #[test]
    fn pager_write_read() -> Result<(), Box<dyn Error>> {
        let sizes = [(4, 4), (4, 16), (16, 4)];

        for (page_size, block_size) in sizes {
            let max_pages = 10;

            let buf = io::Cursor::new(Vec::new());
            let mut pager = Pager::new(buf, page_size, block_size);

            for i in 1..=max_pages {
                let page = vec![i as u8; page_size];
                assert_eq!(pager.write_page(i - 1, &page)?, page_size);
            }

            for i in 1..=max_pages {
                let expected = vec![i as u8; page_size];
                assert_eq!(pager.read_page((i - 1) as u32)?, expected);
            }
        }

        Ok(())
    }
}
