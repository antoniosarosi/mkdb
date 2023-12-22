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

use crate::pager::{Page, PageNumber, Pager};

/// The buffer pool is made of a list of frames. Each frame holds a parsed page,
/// a reference bit a dirty bit.
#[derive(Debug, PartialEq)]
struct Frame<M> {
    /// In-memory representation of a page.
    mem_page: M,
    /// Reference bit. It's set to 1 every time this frame is accessed.
    reference: bool,
    /// Dirty bit. Set to 1 every time the page is modified.
    dirty: bool,
}

/// Frames are identified by their index in [`Cache::buffer`].
type FrameId = usize;

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
pub(crate) struct Cache<I, M> {
    /// IO device pager.
    pub pager: Pager<I>,

    /// Buffer pool.
    buffer: Vec<Frame<M>>,

    /// Maximum number of pages that can be stored in memory.
    max_size: usize,

    /// Page table. Maps page numbers to frame indexes in the buffer pool.
    pages: HashMap<PageNumber, FrameId>,

    /// Clock pointer. Keeps cycling around the buffer pool.
    clock: FrameId,

    /// Pinned pages. These pages cannot be evicted.
    pinned: Vec<PageNumber>,

    /// Writes are sent to the disk in sequential order.
    write_queue: BinaryHeap<Reverse<FrameId>>,
}

/// The cache stores parsed in-memory representation of disk pages. This allows
/// us to mutate and write back to disk easily, without having to constantly
/// parse and serialize bytes. We need to know some information about how the
/// page looks like on disk though.
pub(crate) trait MemPage {
    /// In-memory pages will likely store the page number in their own
    /// structure, so we'll avoid storing it on the [`Frame`].
    fn disk_page_number(&self) -> PageNumber;

    /// Returns the number of bytes that `self` would take on disk after
    /// serializing. We need this to determine whether a page is safe to write,
    /// since some pages (like [`crate::node::Node`]) can overflow past the
    /// [`Pager::page_size`].
    fn size_on_disk(&self) -> usize;
}

impl<M> Frame<M> {
    /// Builds a new frame with [`Frame::reference`] set to 0.
    fn new_unreferenced(mem_page: M) -> Self {
        Self {
            mem_page,
            reference: false,
            dirty: false,
        }
    }

    /// Builds a new frame with [`Frame::reference`] set to 1.
    fn new_referenced(mem_page: M) -> Self {
        Self {
            mem_page,
            reference: true,
            dirty: false,
        }
    }

    /// Returns `true` if this frame is currently referenced.
    #[inline]
    fn is_referenced(&self) -> bool {
        self.reference
    }

    /// Returns `true` if this frame is marked dirty.
    #[inline]
    fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Set reference bit to 0.
    #[inline]
    fn unreference(&mut self) {
        self.reference = false;
    }

    /// Set reference bit to 1.
    #[inline]
    fn reference(&mut self) {
        self.reference = true;
    }

    /// Set dirty flag to 1.
    #[inline]
    fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    /// Set dirty flag to 0.
    #[inline]
    fn mark_clean(&mut self) {
        self.dirty = false;
    }
}

impl<I, M> Cache<I, M> {
    /// Creates a new default cache. Max size is set to [`MAX_CACHE_SIZE`].
    pub fn new(pager: Pager<I>) -> Self {
        Self {
            pager,
            clock: 0,
            max_size: MAX_CACHE_SIZE,
            buffer: Vec::new(),
            pages: HashMap::new(),
            pinned: Vec::new(),
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

    /// Mark a page as unevictable. Pinned page checking algorithm is O(n), so
    /// there should be many pinned pages at the same time.
    pub fn pin(&mut self, page: PageNumber) {
        if !self.pinned.contains(&page) {
            self.pinned.push(page);
        }
    }

    /// Marks the `page` as evictable again.
    pub fn unpin(&mut self, page: PageNumber) {
        if let Some(index) = self.pinned.iter().position(|p| *p == page) {
            self.pinned.remove(index);
        }
    }

    /// Sets dirty bit to 1 and pushes frame to write queue.
    fn mark_dirty_frame(&mut self, frame_id: usize) {
        self.buffer[frame_id].mark_dirty();
        self.write_queue.push(Reverse(frame_id));
    }

    /// Cycles the clock.
    fn tick(&mut self) {
        self.clock += 1;
        if self.clock >= self.buffer.len() {
            self.clock = 0;
        }
    }
}

impl<I, M> Cache<I, M>
where
    I: Seek + Read + Write,
    M: MemPage + From<Page>,
    for<'m> &'m M: Into<Page>,
{
    /// Returns a read only reference to a page in memory.
    pub fn get(&mut self, page_number: PageNumber) -> io::Result<&M> {
        self.load_from_disk(page_number)
            .map(|frame_id| &self.buffer[frame_id].mem_page)
    }

    /// Returns a mutable reference to a page in memory, automatically adding
    /// the page to the write queue. Changes might not be saved to disk unless
    /// [`Self::flush_write_queue_to_disk`] is called.
    pub fn get_mut(&mut self, page_number: PageNumber) -> io::Result<&mut M> {
        self.load_from_disk(page_number).map(|frame_id| {
            self.mark_dirty_frame(frame_id);
            &mut self.buffer[frame_id].mem_page
        })
    }

    /// Returns the frame index of the given page. If the page is not in memory
    /// at this point, it will be loaded from disk.
    fn load_from_disk(&mut self, page_number: PageNumber) -> io::Result<usize> {
        self.try_reference_page(page_number).or_else(|_| {
            let mem_page = self.pager.read(page_number)?;
            self.load_page(mem_page)
        })
    }

    /// Loads a page created in memory into the cache. This automatically adds
    /// the page to the write queue.
    pub fn load_from_mem(&mut self, mem_page: M) -> io::Result<usize> {
        self.try_reference_page(mem_page.disk_page_number())
            .or_else(|_| self.load_page(mem_page))
            .inspect(|frame_id| self.mark_dirty_frame(*frame_id))
    }

    /// If the given page is cached its reference bit will be set to 1 and the
    /// frame index where the page is located will be returned. Otherwise
    /// nothing happens.
    fn try_reference_page(&mut self, page_number: PageNumber) -> Result<usize, ()> {
        self.pages.get(&page_number).map_or(Err(()), |frame_id| {
            self.buffer[*frame_id].reference();
            Ok(*frame_id)
        })
    }

    /// Loads a page into the buffer pool. Doesn't matter where the page comes
    /// from, it could have been created in memory or read from disk.
    fn load_page(&mut self, mem_page: M) -> io::Result<usize> {
        // Buffer is not full, push the page and return.
        if self.buffer.len() < self.max_size {
            let frame_id = self.buffer.len();
            self.pages.insert(mem_page.disk_page_number(), frame_id);
            self.buffer.push(Frame::new_unreferenced(mem_page));

            return Ok(frame_id);
        }

        // Buffer is full, evict using clock algorithm.
        while !self.is_evictable(self.clock) {
            self.buffer[self.clock].unreference();
            self.tick();
        }

        // Can't evict if dirty. Write to disk first.
        // TODO: Better algorithm for deciding which pages are safe to write.
        if self.buffer[self.clock].is_dirty() {
            self.write_frame(self.clock)?;
        }

        self.pages.insert(mem_page.disk_page_number(), self.clock);

        let evict = mem::replace(
            &mut self.buffer[self.clock],
            Frame::new_referenced(mem_page),
        );

        self.pages.remove(&evict.mem_page.disk_page_number());

        Ok(self.clock)
    }

    /// The eviction policy has some special cases. Pinned pages are never
    /// evicted and pages that have overflown past the page size cannot be
    /// evicted safely, because in case they are dirty they cannot be written
    /// to disk.
    fn is_evictable(&self, frame_id: FrameId) -> bool {
        let frame = &self.buffer[frame_id];

        !frame.is_referenced()
            && !self.pinned.contains(&frame.mem_page.disk_page_number())
            && !frame.mem_page.size_on_disk() > self.pager.page_size
    }

    /// Invalidates a cached page. Requesting this page again will force a read
    /// from disk.
    pub fn invalidate(&mut self, page_number: PageNumber) {
        if let Some(frame_id) = self.pages.remove(&page_number) {
            let frame = &mut self.buffer[frame_id];
            frame.unreference();
            frame.mark_clean();
        }
    }

    /// Sends all writes to disk in sequential order to avoid random IO.
    pub fn flush_write_queue_to_disk(&mut self) -> io::Result<()> {
        while let Some(Reverse(frame_id)) = self.write_queue.pop() {
            self.write_frame(frame_id)?;
        }

        Ok(())
    }

    /// Writes the frame to disk if dirty. Otherwise it does nothing.
    fn write_frame(&mut self, frame_id: usize) -> io::Result<()> {
        let frame = &mut self.buffer[frame_id];

        if frame.is_dirty() {
            self.pager.write(&frame.mem_page)?;
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
        pager::{PageNumber, Pager},
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
        pinned_pages: Vec<PageNumber>,
    }

    impl Builder {
        fn new() -> Self {
            Self {
                number_of_nodes: 3,
                max_size: 3,
                prefetch: Prefetch::None,
                include_root_node: false,
                pinned_pages: Vec::new(),
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

        fn pin_pages<P: IntoIterator<Item = PageNumber>>(mut self, pages: P) -> Self {
            self.pinned_pages = pages.into_iter().collect();
            self
        }

        fn prefetch_all_nodes(self) -> Self {
            self.prefetch(Prefetch::AllNodes)
        }

        fn prefetch_until_buffer_is_full(self) -> Self {
            self.prefetch(Prefetch::UntilBufferIsFull)
        }

        fn build(self) -> io::Result<(Cache<MemBuf, Node>, Vec<Node>)> {
            let range = if self.include_root_node {
                0..self.number_of_nodes as PageNumber
            } else {
                1..self.number_of_nodes as PageNumber + 1
            };

            let nodes: Vec<Node> = range
                .map(|i| Node::new_at(i).with_entries(vec![Entry { key: i, value: i }]))
                .collect();

            let page_size = 32;
            let mut pager = Pager::new(io::Cursor::new(Vec::new()), page_size, page_size);

            for node in &nodes {
                pager.write(node)?;
            }

            let mut cache = Cache::new(pager).with_max_size(self.max_size);

            for pinned in self.pinned_pages {
                cache.pin(pinned);
            }

            let pages = nodes.iter().map(|n| n.page);

            cache.load(match self.prefetch {
                Prefetch::AllNodes => pages.take(self.number_of_nodes),
                Prefetch::UntilBufferIsFull => pages.take(self.max_size),
                Prefetch::None => pages.take(0),
            })?;

            Ok((cache, nodes))
        }
    }

    impl Cache<MemBuf, Node> {
        fn load<P: IntoIterator<Item = PageNumber>>(&mut self, pages: P) -> io::Result<()> {
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
            assert_eq!(*node, cache.buffer[i].mem_page);
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
            assert_eq!(*node, cache.buffer[i].mem_page);
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
            assert_eq!(*node, cache.buffer[i].mem_page);
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
            cache.buffer[cache.max_size - 1].mem_page,
            nodes[nodes.len() - 1]
        );

        assert_eq!(
            cache.pages[&nodes[nodes.len() - 1].page],
            cache.max_size - 1
        );

        for (i, node) in nodes[..cache.max_size - 1].iter().enumerate() {
            assert_eq!(*node, cache.buffer[i].mem_page);
            assert_eq!(cache.pages[&node.page], i);
        }

        Ok(())
    }

    #[test]
    fn dont_evict_pinned_page() -> io::Result<()> {
        let (cache, nodes) = Cache::builder()
            .total_nodes(4)
            .max_size(3)
            .include_root_node(true)
            .pin_pages([0])
            .prefetch_all_nodes()
            .build()?;

        assert_eq!(cache.clock, 1);
        assert_eq!(cache.buffer[0].mem_page, nodes[0]);
        assert_eq!(cache.buffer[1].mem_page, nodes[3]);
        assert_eq!(cache.buffer[2].mem_page, nodes[2]);

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
            let node = cache.pager.read::<Node>(page)?;

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
        assert_eq!(cache.buffer[0].mem_page, nodes[3]);

        let evicted_page = cache.pager.read::<Node>(1)?;

        assert_eq!(
            evicted_page.entries,
            vec![Entry::new(1, 1), Entry::new(10, 10)]
        );

        Ok(())
    }
}
