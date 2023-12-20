use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashMap},
    io::{self, Read, Seek, Write},
    mem,
};

/// Default value for [`Cache::max_size`].
const MAX_CACHE_SIZE: usize = 1024;

/// Minimum allowed cache size.
const MIN_CACHE_SIZE: usize = 3;

use crate::{node::Node, pager::Pager};

#[derive(Debug, PartialEq)]
struct Frame {
    node: Node,
    reference: bool,
    dirty: bool,
}

impl Frame {
    fn new_unreferenced(page: Node) -> Self {
        Self {
            node: page,
            reference: false,
            dirty: false,
        }
    }

    fn new_referenced(page: Node) -> Self {
        Self {
            node: page,
            reference: true,
            dirty: false,
        }
    }

    fn unreference(&mut self) {
        self.reference = false;
    }

    fn reference(&mut self) {
        self.reference = true;
    }

    fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    fn mark_clean(&mut self) {
        self.dirty = false;
    }
}

pub(crate) struct Cache<F> {
    /// IO device pager.
    pub pager: Pager<F>,

    /// BTree order. Used to calculate overflows on nodes.
    order: usize,

    /// Maximum number of pages that can be stored in memory.
    max_size: usize,

    /// Buffer pool.
    buffer: Vec<Frame>,

    /// Page table. Maps page numbers to frame indexes in the buffer pool.
    pages: HashMap<u32, usize>,

    /// Clock pointer. Keeps cycling around the buffer pool.
    clock: usize,

    /// Writes are sent to the disk in sequential order.
    write_queue: BinaryHeap<Reverse<usize>>,
}

impl<F> Cache<F> {
    pub fn new(pager: Pager<F>) -> Self {
        Self {
            pager,
            clock: 0,
            order: 0,
            max_size: MAX_CACHE_SIZE,
            pages: HashMap::new(),
            buffer: Vec::new(),
            write_queue: BinaryHeap::new(),
        }
    }

    /// Max size 3: Root, overflow node, cache overflow. Explain.
    pub fn with_max_size(mut self, max_size: usize) -> Self {
        if max_size < MIN_CACHE_SIZE {
            panic!("Buffer pool size must be at least {MIN_CACHE_SIZE}");
        }

        self.max_size = max_size;
        self
    }

    pub fn with_order(mut self, order: usize) -> Self {
        self.order = order;
        self
    }

    pub fn push_to_write_queue(&mut self, page: u32) {
        let index = self.pages[&page];
        let frame = &mut self.buffer[index];
        frame.mark_dirty();

        self.write_queue.push(Reverse(index));
    }

    fn tick(&mut self) {
        self.clock += 1;
        if self.clock >= self.buffer.len() {
            self.clock = 0;
        }
    }
}

impl<F: Seek + Read + Write> Cache<F> {
    pub fn get(&mut self, page: u32) -> io::Result<&Node> {
        self.load_disk_page_into_mem_frame(page)
            .map(|index| &self.buffer[index].node)
    }

    pub fn get_mut(&mut self, page: u32) -> io::Result<&mut Node> {
        let index = self.load_disk_page_into_mem_frame(page)?;

        let frame = &mut self.buffer[index];
        frame.mark_dirty();

        self.write_queue.push(Reverse(index));

        Ok(&mut frame.node)
    }

    pub fn invalidate(&mut self, page: u32) {
        if let Some(index) = self.pages.remove(&page) {
            let frame = &mut self.buffer[index];
            frame.unreference();
            frame.mark_clean();

            frame.node.entries.clear();
            frame.node.children.clear();
        }
    }

    fn is_evictable(&self, frame: &Frame) -> bool {
        let is_overflow = self.order > 0 && frame.node.entries.len() > self.order - 1;

        !frame.reference && frame.node.page != 0 && !is_overflow
    }

    fn load_node(&mut self, node: Node) -> io::Result<usize> {
        // Buffer is not full, push the page and return.
        if self.buffer.len() < self.max_size {
            let index = self.buffer.len();
            self.pages.insert(node.page, index);
            self.buffer.push(Frame::new_unreferenced(node));

            return Ok(index);
        }

        // Buffer is full, evict using clock algorithm.
        while !self.is_evictable(&self.buffer[self.clock]) {
            self.buffer[self.clock].unreference();
            self.tick();
        }

        // Can't evict if dirty. Write to disk first.
        // TODO: Better algorithm for deciding which pages are safe to write.
        if self.buffer[self.clock].dirty {
            self.write_frame(self.clock)?;
        }

        self.pages.insert(node.page, self.clock);

        let evict = mem::replace(&mut self.buffer[self.clock], Frame::new_referenced(node));
        self.pages.remove(&evict.node.page);

        Ok(self.clock)
    }

    fn try_reference_page(&mut self, page: u32) -> Result<usize, ()> {
        if let Some(index) = self.pages.get(&page) {
            self.buffer[*index].reference();
            return Ok(*index);
        }

        Err(())
    }

    fn load_disk_page_into_mem_frame(&mut self, page: u32) -> io::Result<usize> {
        self.try_reference_page(page).or_else(|_| {
            let node = self.pager.read_node(page)?;
            self.load_node(node)
        })
    }

    pub fn load_from_mem(&mut self, node: Node) -> io::Result<usize> {
        let index = self
            .try_reference_page(node.page)
            .or_else(|_| self.load_node(node))?;

        self.buffer[index].mark_dirty();
        self.write_queue.push(Reverse(index));

        Ok(index)
    }

    pub fn flush_write_queue_to_disk(&mut self) -> io::Result<()> {
        while let Some(Reverse(index)) = self.write_queue.pop() {
            self.write_frame(index)?;
        }

        Ok(())
    }

    fn write_frame(&mut self, index: usize) -> io::Result<()> {
        let frame = &mut self.buffer[index];

        if frame.dirty {
            self.pager.write_node(&frame.node)?;
            frame.mark_clean();
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::io;

    use super::Cache;
    use crate::{
        node::{Entry, Node},
        pager::Pager,
    };

    impl Node {
        fn with_entries(mut self, entries: Vec<Entry>) -> Self {
            self.entries = entries;
            self
        }
    }

    type MemBuf = io::Cursor<Vec<u8>>;

    enum Prefetch {
        AllNodes,
        UntilBufferIsFull,
        None,
    }

    struct Builder {
        number_of_nodes: usize,
        include_root_node: bool,
        max_size: usize,
        prefetch: Prefetch,
    }

    impl Builder {
        fn new() -> Self {
            Self {
                number_of_nodes: 3,
                max_size: 3,
                prefetch: Prefetch::None,
                include_root_node: false,
            }
        }

        fn total_nodes(mut self, number_of_nodes: usize) -> Self {
            self.number_of_nodes = number_of_nodes;
            self
        }

        fn max_size(mut self, max_size: usize) -> Self {
            self.max_size = max_size;
            self
        }

        fn prefetch(mut self, prefetch: Prefetch) -> Self {
            self.prefetch = prefetch;
            self
        }

        fn include_root_node(mut self, include_root_node: bool) -> Self {
            self.include_root_node = include_root_node;
            self
        }

        fn prefetch_all_nodes(self) -> Self {
            self.prefetch(Prefetch::AllNodes)
        }

        fn prefetch_until_buffer_is_full(self) -> Self {
            self.prefetch(Prefetch::UntilBufferIsFull)
        }

        fn build(self) -> io::Result<(Cache<MemBuf>, Vec<Node>)> {
            let range = if self.include_root_node {
                0..self.number_of_nodes as u32
            } else {
                1..self.number_of_nodes as u32 + 1
            };

            let nodes: Vec<Node> = range
                .map(|i| Node::new_at(i).with_entries(vec![Entry { key: i, value: i }]))
                .collect();

            let page_size = 32;
            let mut pager = Pager::new(io::Cursor::new(Vec::new()), page_size, page_size);

            for node in &nodes {
                pager.write_node(node)?;
            }

            let mut cache = Cache::new(pager).with_max_size(self.max_size);

            let pages = nodes.iter().map(|n| n.page);

            cache.load(match self.prefetch {
                Prefetch::AllNodes => pages.take(self.number_of_nodes),
                Prefetch::UntilBufferIsFull => pages.take(self.max_size),
                Prefetch::None => pages.take(0),
            })?;

            Ok((cache, nodes))
        }
    }

    impl Cache<MemBuf> {
        fn load<P: IntoIterator<Item = u32>>(&mut self, pages: P) -> io::Result<()> {
            for page in pages {
                self.get(page)?;
            }

            Ok(())
        }

        fn builder() -> Builder {
            Builder::new()
        }
    }

    #[test]
    fn get_mut_ref_to_node() -> io::Result<()> {
        let (mut cache, mut nodes) = Cache::builder()
            .total_nodes(3)
            .max_size(3)
            .prefetch_until_buffer_is_full()
            .build()?;

        for node in &mut nodes {
            assert_eq!(node, cache.get_mut(node.page)?);
        }

        Ok(())
    }

    #[test]
    fn fill_buffer() -> io::Result<()> {
        let (cache, nodes) = Cache::builder()
            .total_nodes(3)
            .max_size(3)
            .prefetch_until_buffer_is_full()
            .build()?;

        assert_eq!(cache.buffer.len(), nodes.len());
        assert_eq!(cache.clock, 0);

        for (i, node) in nodes.iter().enumerate() {
            assert_eq!(*node, cache.buffer[i].node);
            assert_eq!(cache.pages[&node.page], i);
            assert!(!cache.buffer[i].reference);
        }

        Ok(())
    }

    #[test]
    fn start_clock_when_buffer_is_full() -> io::Result<()> {
        let (cache, nodes) = Cache::builder()
            .total_nodes(6)
            .max_size(3)
            .prefetch_all_nodes()
            .build()?;

        assert_eq!(cache.buffer.len(), cache.max_size);
        assert_eq!(cache.clock, cache.max_size - 1);

        for (i, node) in nodes[cache.max_size..].iter().enumerate() {
            assert_eq!(*node, cache.buffer[i].node);
            assert_eq!(cache.pages[&node.page], i);
        }

        Ok(())
    }

    #[test]
    fn set_reference_bit_to_one() -> io::Result<()> {
        let (mut cache, nodes) = Cache::builder()
            .total_nodes(3)
            .max_size(3)
            .prefetch_until_buffer_is_full()
            .build()?;

        // Query pages again, this should set reference = true.
        cache.load(1..=nodes.len() as u32)?;

        for (i, node) in nodes.iter().enumerate() {
            assert_eq!(*node, cache.buffer[i].node);
            assert!(cache.buffer[i].reference);
        }

        Ok(())
    }

    #[test]
    fn evict_first_unreferenced_page() -> io::Result<()> {
        let (mut cache, nodes) = Cache::builder()
            .total_nodes(4)
            .max_size(3)
            .prefetch_until_buffer_is_full()
            .build()?;

        // Reference all pages except last one.
        cache.load(nodes[..cache.max_size - 1].iter().map(|n| n.page))?;

        // Should evict page 3 and replace it with page 4.
        cache.get(nodes.last().unwrap().page)?;

        assert_eq!(cache.clock, cache.max_size - 1);

        assert_eq!(
            cache.buffer[cache.max_size - 1].node,
            nodes[nodes.len() - 1]
        );

        assert_eq!(
            cache.pages[&nodes[nodes.len() - 1].page],
            cache.max_size - 1
        );

        for (i, node) in nodes[..cache.max_size - 1].iter().enumerate() {
            assert_eq!(*node, cache.buffer[i].node);
            assert_eq!(cache.pages[&node.page], i);
        }

        Ok(())
    }

    #[test]
    fn dont_evict_root_page() -> io::Result<()> {
        let (cache, nodes) = Cache::builder()
            .total_nodes(4)
            .include_root_node(true)
            .max_size(3)
            .prefetch_all_nodes()
            .build()?;

        assert_eq!(cache.clock, 1);
        assert_eq!(cache.buffer[0].node, nodes[0]);
        assert_eq!(cache.buffer[1].node, nodes[3]);
        assert_eq!(cache.buffer[2].node, nodes[2]);

        Ok(())
    }

    #[test]
    fn mark_dirty_pages() -> io::Result<()> {
        let (mut cache, _) = Cache::builder()
            .total_nodes(3)
            .max_size(3)
            .prefetch_all_nodes()
            .build()?;

        cache.get_mut(1)?;
        cache.get_mut(2)?;

        assert_eq!(cache.write_queue.len(), 2);
        assert!(cache.buffer[0].dirty);
        assert!(cache.buffer[1].dirty);

        Ok(())
    }

    #[test]
    fn flush_writes_to_disk() -> io::Result<()> {
        let (mut cache, _) = Cache::builder()
            .total_nodes(3)
            .max_size(3)
            .prefetch_all_nodes()
            .build()?;

        for page in [1, 2] {
            let node = cache.get_mut(page)?;
            node.entries.push(Entry::new(10, 10));
            cache.get_mut(page)?;
        }

        cache.flush_write_queue_to_disk()?;

        assert!(!cache.buffer[0].dirty);
        assert!(!cache.buffer[1].dirty);

        for page in [1, 2] {
            let node = cache.pager.read_node(page)?;

            assert_eq!(
                node.entries,
                vec![Entry::new(page, page), Entry::new(10, 10)]
            );
        }

        Ok(())
    }

    #[test]
    fn flush_to_disk_if_can_only_evict_dirty_page() -> io::Result<()> {
        let (mut cache, nodes) = Cache::builder()
            .total_nodes(4)
            .max_size(3)
            .prefetch_until_buffer_is_full()
            .build()?;

        // Reference and modify page 1
        cache.get_mut(1)?.entries.push(Entry::new(10, 10));

        // Reference pages 2 and 3
        cache.get(2)?;
        cache.get(3)?;

        // Loading page 4 should evict page 1 and write it to disk as well.
        cache.get(4)?;

        assert_eq!(cache.clock, 0);
        assert_eq!(cache.write_queue.len(), 1);
        assert_eq!(cache.buffer[0].node, nodes[3]);

        let evicted_page = cache.pager.read_node(1)?;

        assert_eq!(
            evicted_page.entries,
            vec![Entry::new(1, 1), Entry::new(10, 10)]
        );

        Ok(())
    }
}