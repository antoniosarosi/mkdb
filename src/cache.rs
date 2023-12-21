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

/// The buffer pool is made of a list of frames. Each frame holds a node, a
/// reference bit a dirty bit.
#[derive(Debug, PartialEq)]
struct Frame {
    /// In-memory representation of a page.
    node: Node,
    /// Reference bit. It's set to 1 every time this frame is accessed.
    reference: bool,
    /// Dirty bit. Set to 1 every time the node is modified.
    dirty: bool,
}

impl Frame {
    /// Builds a new frame with [`Frame::reference`] set to 0.
    fn new_unreferenced(node: Node) -> Self {
        Self {
            node,
            reference: false,
            dirty: false,
        }
    }

    /// Builds a new frame with [`Frame::reference`] set to 1.
    fn new_referenced(node: Node) -> Self {
        Self {
            node,
            reference: true,
            dirty: false,
        }
    }

    /// Set reference bit to 0.
    fn unreference(&mut self) {
        self.reference = false;
    }

    /// Set reference bit to 1.
    fn reference(&mut self) {
        self.reference = true;
    }

    /// Set dirty flag to 1.
    fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    /// Set dirty flag to 0.
    fn mark_clean(&mut self) {
        self.dirty = false;
    }
}

/// Read-Write page cache with clock eviction policy. Pages are loaded into
/// a buffer pool until the buffer is full, then the clock algorithm kicks in.
/// Each page is stored in a [`Frame`] inside the buffer pool, which holds
/// additinal metadata such as reference bit and dirty flag.
///
/// A *page table* keeps track of the frame where each page is stored by mapping
/// page numbers to frame indexes in a [`HashMap`].
///
/// For example, when [`Cache::max_size`] equals 4, this is how the structure
/// would look like right after filling the buffer pool with pages `[1,2,3,4]`
/// in sequential order:
///
/// ```text
///    PAGE TABLE                              BUFFER POOL
/// +------+-------+                        +---------------+
/// | PAGE | FRAME |                        | PAGE 1        |
/// +------+-------+                        | ref: 0 dty: 0 |
/// |  1   |   0   |                        +---------------+
/// +------+-------+                                ^
/// |  2   |   1   |                                |
/// +------+-------+     +---------------+    +----------+    +---------------+
/// |  3   |   2   |     | PAGE 4        |    | CLOCK: 0 |    | PAGE 2        |
/// +------+-------+     | ref: 0 dty: 0 |    +----------+    | ref: 0 dty: 0 |
/// |  4   |   3   |     +---------------+                    +---------------+
/// +------+-------+
///                                         +---------------+
///                                         | PAGE 3        |
///                                         | ref: 0 dty: 0 |
///                                         +---------------+
/// ```
///
/// # Eviction Policy
///
/// All pages have their reference bit set to 0 on initial loads until the
/// buffer reaches max size. Once the buffer is full, loading additional pages
/// will execute the eviction policy, which consists in incrementing the clock
/// pointer until it points to a page that is not currently referenced.
///
/// If the clock finds pages that are referenced in the process, it resets their
/// reference bit back to 0, so that they can be evicted the next time the clock
/// passes by. This algorithm is an approximation of LRU (Least Recently Used)
/// that does not require timestamps.
///
/// Following the example above, the clock pointer already points at page 1,
/// which is not currently referenced. So if we want to load page 5, we can
/// evict page 1:
///
/// ```text
///    PAGE TABLE                              BUFFER POOL
/// +------+-------+                        +---------------+
/// | PAGE | FRAME |                        | PAGE 5        |
/// +------+-------+                        | ref: 1 dty: 0 |
/// |  5   |   0   |                        +---------------+
/// +------+-------+                                ^
/// |  2   |   1   |                                |
/// +------+-------+     +---------------+    +----------+    +---------------+
/// |  3   |   2   |     | PAGE 4        |    | CLOCK: 0 |    | PAGE 2        |
/// +------+-------+     | ref: 0 dty: 0 |    +----------+    | ref: 0 dty: 0 |
/// |  4   |   3   |     +---------------+                    +---------------+
/// +------+-------+
///                                         +---------------+
///                                         | PAGE 3        |
///                                         | ref: 0 dty: 0 |
///                                         +---------------+
/// ```
///
/// Page loads on a full buffer will automatically set the reference bit to 1.
/// Now suppose the user requests pages 2 and 3 again, this will set their
/// respective reference bit to 1:
///
/// ```text
///    PAGE TABLE                              BUFFER POOL
/// +------+-------+                        +---------------+
/// | PAGE | FRAME |                        | PAGE 5        |
/// +------+-------+                        | ref: 1 dty: 0 |
/// |  5   |   0   |                        +---------------+
/// +------+-------+                                ^
/// |  2   |   1   |                                |
/// +------+-------+     +---------------+    +----------+    +---------------+
/// |  3   |   2   |     | PAGE 4        |    | CLOCK: 0 |    | PAGE 2        |
/// +------+-------+     | ref: 0 dty: 0 |    +----------+    | ref: 1 dty: 0 |
/// |  4   |   3   |     +---------------+                    +---------------+
/// +------+-------+
///                                         +---------------+
///                                         | PAGE 3        |
///                                         | ref: 1 dty: 0 |
///                                         +---------------+
/// ```
///
/// Loading page 6 will cause the clock to cycle until it reaches page 4, which
/// is the first unreferenced page. The reference bit of pages 2 and 3 is set
/// back to 0 in the process:
///
/// ```text
///    PAGE TABLE                              BUFFER POOL
/// +------+-------+                        +---------------+
/// | PAGE | FRAME |                        | PAGE 5        |
/// +------+-------+                        | ref: 0 dty: 0 |
/// |  5   |   0   |                        +---------------+
/// +------+-------+
/// |  2   |   1   |
/// +------+-------+     +---------------+    +----------+    +---------------+
/// |  3   |   2   |     | PAGE 6        | <--| CLOCK: 3 |    | PAGE 2        |
/// +------+-------+     | ref: 1 dty: 0 |    +----------+    | ref: 0 dty: 0 |
/// |  6   |   3   |     +---------------+                    +---------------+
/// +------+-------+
///                                         +---------------+
///                                         | PAGE 3        |
///                                         | ref: 0 dty: 0 |
///                                         +---------------+
/// ```
///
/// Acquiring mutable references to pages automatically sets their `dirty` flag
/// to 1 and pushes the frame index to the write queue. Calling
/// [`Self::flush_write_queue_to_disk`] will persist the changes on disk. Pages
/// can also be written to disk if they are evicted while being dirty.
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
    /// Creates a new default cache.
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

    /// Sets the value of [`Self::max_size`].
    ///
    /// # Panics
    ///
    /// This function panics if `max_size` < [`MIN_CACHE_SIZE`].
    pub fn with_max_size(mut self, max_size: usize) -> Self {
        if max_size < MIN_CACHE_SIZE {
            panic!("Buffer pool size must be at least {MIN_CACHE_SIZE}");
        }

        self.max_size = max_size;
        self
    }

    /// Sets [`Self::order`].
    pub fn with_order(mut self, order: usize) -> Self {
        self.order = order;
        self
    }

    /// Sets dirty bit to 1 and pushes frame to write queue.
    pub fn mark_dirty_frame(&mut self, index: usize) {
        self.buffer[index].mark_dirty();
        self.write_queue.push(Reverse(index));
    }

    /// Cycles the clock.
    fn tick(&mut self) {
        self.clock += 1;
        if self.clock >= self.buffer.len() {
            self.clock = 0;
        }
    }
}

impl<F: Seek + Read + Write> Cache<F> {
    /// Returns a read only reference to a node in memory.
    pub fn get(&mut self, page: u32) -> io::Result<&Node> {
        self.load_from_disk(page)
            .map(|index| &self.buffer[index].node)
    }

    /// Returns a mutable reference to a node in memory, automatically adding
    /// the node to the write queue. Changes might not be saved to disk unless
    /// [`Self::flush_write_queue_to_disk`] is called.
    pub fn get_mut(&mut self, page: u32) -> io::Result<&mut Node> {
        self.load_from_disk(page).map(|index| {
            self.mark_dirty_frame(index);
            &mut self.buffer[index].node
        })
    }

    /// Returns the frame index of the given page. If the page is not in memory
    /// at this point, it will be loaded from disk.
    fn load_from_disk(&mut self, page: u32) -> io::Result<usize> {
        self.try_reference_page(page).or_else(|_| {
            let node = self.pager.read_node(page)?;
            self.load_node(node)
        })
    }

    /// Loads a node created in memory into the cache. This automatically adds
    /// the node to the write queue.
    pub fn load_from_mem(&mut self, node: Node) -> io::Result<usize> {
        self.try_reference_page(node.page)
            .or_else(|_| self.load_node(node))
            .inspect(|index| self.mark_dirty_frame(*index))
    }

    /// If the given page is cached its reference bit will be set to 1 and the
    /// frame index where the page is located will be returned. Otherwise
    /// nothing happens.
    fn try_reference_page(&mut self, page: u32) -> Result<usize, ()> {
        self.pages.get(&page).map_or(Err(()), |index| {
            self.buffer[*index].reference();
            Ok(*index)
        })
    }

    /// Loads a node into the buffer pool. Doesn't matter where the node comes
    /// from, it could have been created in memory or read from disk.
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

    /// The eviction policy has some special cases. First of all, the root page
    /// is never evicted, because we'll be reading it all the time. Second and
    /// more important, overflow nodes can not be evicted because they are
    /// invalid and cannot be stored on disk.
    ///
    /// This requires the minimum cache size to equal 3 for single threaded
    /// programs, because in the worst case we'll have the root and an overflow
    /// node cached, needing one additional frame for storing other pages. The
    /// tree can never have more than one overflow node. If a node overflows it
    /// will be split, at which point the node is no longer in overflow mode.
    /// The parent might have overflown because of that, but at any given moment
    /// there is only one overflow node.
    fn is_evictable(&self, frame: &Frame) -> bool {
        let is_overflow = self.order > 0 && frame.node.entries.len() > self.order - 1;

        !frame.reference && frame.node.page != 0 && !is_overflow
    }

    /// Invalidates a cached page. Requesting this page again will force a read
    /// from disk.
    pub fn invalidate(&mut self, page: u32) {
        if let Some(index) = self.pages.remove(&page) {
            let frame = &mut self.buffer[index];
            frame.unreference();
            frame.mark_clean();

            frame.node.entries.clear();
            frame.node.children.clear();
        }
    }

    /// Sends all writes to disk in sequential order to avoid random IO.
    pub fn flush_write_queue_to_disk(&mut self) -> io::Result<()> {
        while let Some(Reverse(index)) = self.write_queue.pop() {
            self.write_frame(index)?;
        }

        Ok(())
    }

    /// Writes the frame to disk if dirty. Otherwise it does nothing.
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
