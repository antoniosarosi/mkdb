//! Page read-write cache implementation.

use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashMap},
    io::{self, Read, Seek, Write},
    mem,
};

/// Default value for [`Cache::max_size`].
const DEFAULT_MAX_CACHE_SIZE: usize = 1024;

/// Minimum allowed cache size.
const MIN_CACHE_SIZE: usize = 3;

use super::pager::{PageNumber, Pager};
use crate::storage::page::Page;

/// The buffer pool is made of a list of frames. Each frame holds a parsed page,
/// a reference bit a dirty bit.
#[derive(Debug, PartialEq)]
struct Frame<Page> {
    /// In-memory representation of a page.
    page: Page,
    /// Reference bit. It's set to 1 every time this frame is accessed.
    reference: bool,
    /// Dirty bit. Set to 1 every time the page is modified.
    dirty: bool,
    /// Whether this page is currently pinned.
    pinned: bool,
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
pub(crate) struct Cache<I> {
    /// IO device pager.
    pub pager: Pager<I>,

    /// Buffer pool.
    buffer: Vec<Frame<Page>>,

    /// Maximum number of pages that can be stored in memory.
    max_size: usize,

    /// Page table. Maps page numbers to frame indexes in the buffer pool.
    pages: HashMap<PageNumber, FrameId>,

    /// Clock pointer. Keeps cycling around the buffer pool.
    clock: FrameId,

    /// Writes are sent to the disk in sequential order.
    write_queue: BinaryHeap<Reverse<FrameId>>,
}

impl<Page> Frame<Page> {
    /// Builds a new frame with [`Frame::reference`] set to 0.
    fn new_unreferenced(page: Page) -> Self {
        Self {
            page,
            reference: false,
            dirty: false,
            pinned: false,
        }
    }

    /// Builds a new frame with [`Frame::reference`] set to 1.
    fn new_referenced(page: Page) -> Self {
        Self {
            page,
            reference: true,
            dirty: false,
            pinned: false,
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

    /// Returns `true` if this page is currently pinned.
    #[inline]
    fn is_pinned(&self) -> bool {
        self.pinned
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

    /// Pin this page.
    #[inline]
    fn pin(&mut self) {
        self.pinned = true;
    }

    /// Unpin this page.
    #[inline]
    fn unpin(&mut self) {
        self.pinned = false;
    }
}

impl<I> Cache<I> {
    /// Creates a new default cache. Max size is set to [`MAX_CACHE_SIZE`].
    pub fn new(pager: Pager<I>) -> Self {
        Self {
            pager,
            clock: 0,
            max_size: DEFAULT_MAX_CACHE_SIZE,
            buffer: Vec::new(),
            pages: HashMap::new(),
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
    pub fn pin(&mut self, page: PageNumber) -> bool {
        let Some(frame_id) = self.pages.get(&page) else {
            return false;
        };

        self.buffer[*frame_id].pin();
        true
    }

    /// Marks the `page` as evictable again.
    pub fn unpin(&mut self, page: PageNumber) -> bool {
        let Some(frame_id) = self.pages.get(&page) else {
            return false;
        };

        self.buffer[*frame_id].unpin();
        true
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

impl<I: Seek + Read + Write> Cache<I> {
    /// Returns a read only reference to a page in memory.
    pub fn get(&mut self, page_number: PageNumber) -> io::Result<&Page> {
        self.load_from_disk(page_number)
            .map(|frame_id| &self.buffer[frame_id].page)
    }

    /// Returns a mutable reference to a page in memory, automatically adding
    /// the page to the write queue. Changes might not be saved to disk unless
    /// [`Self::flush_write_queue_to_disk`] is called.
    pub fn get_mut(&mut self, page_number: PageNumber) -> io::Result<&mut Page> {
        self.load_from_disk(page_number).map(|frame_id| {
            self.mark_dirty_frame(frame_id);
            &mut self.buffer[frame_id].page
        })
    }

    /// Returns the frame index of the given page. If the page is not in memory
    /// at this point, it will be loaded from disk.
    fn load_from_disk(&mut self, page_number: PageNumber) -> io::Result<usize> {
        self.try_reference_page(page_number).or_else(|_| {
            let mut page = Page::new(page_number, self.pager.page_size as u16);
            self.pager.read(page_number, page.buffer_mut())?;
            self.load_page(page)
        })
    }

    /// Loads a page created in memory into the cache. This automatically adds
    /// the page to the write queue.
    pub fn load_from_mem(&mut self, page: Page) -> io::Result<usize> {
        self.try_reference_page(page.number)
            .or_else(|_| self.load_page(page))
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
    fn load_page(&mut self, mut page: Page) -> io::Result<FrameId> {
        page.init();
        // Buffer is not full, push the page and return.
        if self.buffer.len() < self.max_size {
            let frame_id = self.buffer.len();
            self.pages.insert(page.number, frame_id);
            self.buffer.push(Frame::new_unreferenced(page));

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

        self.pages.insert(page.number, self.clock);

        let evict = mem::replace(&mut self.buffer[self.clock], Frame::new_referenced(page));

        self.pages.remove(&evict.page.number);

        Ok(self.clock)
    }

    /// The eviction policy has some special cases. Pinned pages are never
    /// evicted and pages that have overflown past the page size cannot be
    /// evicted safely, because in case they are dirty they cannot be written
    /// to disk.
    fn is_evictable(&self, frame_id: FrameId) -> bool {
        let frame = &self.buffer[frame_id];

        !frame.is_referenced() && !frame.is_pinned()
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
            self.pager.write(frame.page.number, frame.page.buffer())?;
            frame.mark_clean();
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::io;

    use super::{
        super::pager::{PageNumber, Pager},
        Cache,
    };
    use crate::storage::page::{Cell, Page};

    type MemBuf = io::Cursor<Vec<u8>>;

    enum Prefetch {
        AllPages,
        UntilBufferIsFull,
        None,
    }

    struct Builder {
        number_of_pages: usize,
        max_size: usize,
        prefetch: Prefetch,
    }

    impl Builder {
        fn new() -> Self {
            Self {
                number_of_pages: 3,
                max_size: 3,
                prefetch: Prefetch::None,
            }
        }

        fn total_pages(mut self, number_of_pages: usize) -> Self {
            self.number_of_pages = number_of_pages;
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

        fn prefetch_all_pages(self) -> Self {
            self.prefetch(Prefetch::AllPages)
        }

        fn prefetch_until_buffer_is_full(self) -> Self {
            self.prefetch(Prefetch::UntilBufferIsFull)
        }

        fn build(self) -> io::Result<(Cache<MemBuf>, Vec<Page>)> {
            const PAGE_SIZE: usize = 256;

            let mut pager = Pager::new(io::Cursor::new(Vec::new()), PAGE_SIZE, PAGE_SIZE);

            let pages: Vec<Page> = (0..self.number_of_pages as PageNumber)
                .map(|i| {
                    let mut page = Page::new(i, PAGE_SIZE as _);
                    page.push(Cell::new(&vec![i as u8; PAGE_SIZE / 2]));
                    page
                })
                .collect();

            for page in &pages {
                pager.write(page.number, page.buffer())?;
            }

            let mut cache = Cache::new(pager).with_max_size(self.max_size);

            let page_numbers = pages.iter().map(|p| p.number);

            cache.load(match self.prefetch {
                Prefetch::AllPages => page_numbers.take(self.number_of_pages),
                Prefetch::UntilBufferIsFull => page_numbers.take(self.max_size),
                Prefetch::None => page_numbers.take(0),
            })?;

            Ok((cache, pages))
        }
    }

    impl Cache<MemBuf> {
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
    fn get_mut_ref_to_page() -> io::Result<()> {
        let (mut cache, mut pages) = Cache::builder()
            .total_pages(3)
            .max_size(3)
            .prefetch_until_buffer_is_full()
            .build()?;

        for page in &mut pages {
            assert_eq!(page, cache.get_mut(page.number)?);
        }

        Ok(())
    }

    #[test]
    fn fill_buffer() -> io::Result<()> {
        let (cache, pages) = Cache::builder()
            .total_pages(3)
            .max_size(3)
            .prefetch_until_buffer_is_full()
            .build()?;

        assert_eq!(cache.buffer.len(), pages.len());
        assert_eq!(cache.clock, 0);

        for (i, page) in pages.iter().enumerate() {
            assert_eq!(*page, cache.buffer[i].page);
            assert_eq!(cache.pages[&page.number], i);
            assert!(!cache.buffer[i].reference);
        }

        Ok(())
    }

    #[test]
    fn start_clock_when_buffer_is_full() -> io::Result<()> {
        let (cache, pages) = Cache::builder()
            .total_pages(6)
            .max_size(3)
            .prefetch_all_pages()
            .build()?;

        assert_eq!(cache.buffer.len(), cache.max_size);
        assert_eq!(cache.clock, cache.max_size - 1);

        for (i, page) in pages[cache.max_size..].iter().enumerate() {
            assert_eq!(*page, cache.buffer[i].page);
            assert_eq!(cache.pages[&page.number], i);
        }

        Ok(())
    }

    #[test]
    fn set_reference_bit_to_one() -> io::Result<()> {
        let (mut cache, pages) = Cache::builder()
            .total_pages(3)
            .max_size(3)
            .prefetch_until_buffer_is_full()
            .build()?;

        // Query pages again, this should set reference = true.
        cache.load(pages.iter().map(|p| p.number))?;

        for (i, page) in pages.iter().enumerate() {
            assert_eq!(*page, cache.buffer[i].page);
            assert!(cache.buffer[i].reference);
        }

        Ok(())
    }

    #[test]
    fn evict_first_unreferenced_page() -> io::Result<()> {
        let (mut cache, pages) = Cache::builder()
            .total_pages(4)
            .max_size(3)
            .prefetch_until_buffer_is_full()
            .build()?;

        // Reference all pages except last one.
        cache.load(pages[..cache.max_size - 1].iter().map(|p| p.number))?;

        // Should evict page 3 and replace it with page 4.
        cache.get(pages.last().unwrap().number)?;

        assert_eq!(cache.clock, cache.max_size - 1);

        assert_eq!(
            cache.buffer[cache.max_size - 1].page,
            pages[pages.len() - 1]
        );

        assert_eq!(
            cache.pages[&pages[pages.len() - 1].number],
            cache.max_size - 1
        );

        for (i, page) in pages[..cache.max_size - 1].iter().enumerate() {
            assert_eq!(*page, cache.buffer[i].page);
            assert_eq!(cache.pages[&page.number], i);
        }

        Ok(())
    }

    #[test]
    fn dont_evict_pinned_page() -> io::Result<()> {
        let (mut cache, pages) = Cache::builder()
            .total_pages(4)
            .max_size(3)
            .prefetch_until_buffer_is_full()
            .build()?;

        let pinned = cache.pin(0);

        // Should not evict page 0 because it's pinned.
        cache.get(pages[3].number)?;

        assert!(pinned);
        assert_eq!(cache.clock, 1);
        assert_eq!(cache.buffer[0].page, pages[0]);
        assert_eq!(cache.buffer[1].page, pages[3]);
        assert_eq!(cache.buffer[2].page, pages[2]);

        Ok(())
    }

    #[test]
    fn mark_dirty_pages() -> io::Result<()> {
        let (mut cache, _) = Cache::builder()
            .total_pages(3)
            .max_size(3)
            .prefetch_all_pages()
            .build()?;

        cache.get_mut(0)?;
        cache.get_mut(1)?;

        assert_eq!(cache.write_queue.len(), 2);
        assert!(cache.buffer[0].dirty);
        assert!(cache.buffer[1].dirty);

        Ok(())
    }

    #[test]
    fn flush_writes_to_disk() -> io::Result<()> {
        let (mut cache, _) = Cache::builder()
            .total_pages(3)
            .max_size(3)
            .prefetch_all_pages()
            .build()?;

        let mut expected = Vec::new();

        for i in [0, 1] {
            let page = cache.get_mut(i)?;
            page.cell_mut(0)
                .content
                .iter_mut()
                .for_each(|byte| *byte += 10);
            expected.push(page.clone());
        }

        cache.flush_write_queue_to_disk()?;

        assert!(!cache.buffer[0].dirty);
        assert!(!cache.buffer[1].dirty);

        for i in [0, 1] {
            let mut page = Page::new(i as PageNumber, cache.pager.page_size as u16);
            cache.pager.read(i as PageNumber, page.buffer_mut())?;

            assert_eq!(page, expected[i]);
        }

        Ok(())
    }

    #[test]
    fn flush_to_disk_if_can_only_evict_dirty_page() -> io::Result<()> {
        let (mut cache, pages) = Cache::builder()
            .total_pages(4)
            .max_size(3)
            .prefetch_until_buffer_is_full()
            .build()?;

        // Reference and modify page 0
        let page = cache.get_mut(0)?;
        page.cell_mut(0)
            .content
            .iter_mut()
            .for_each(|byte| *byte += 1);
        let expected = page.clone();

        // Reference pages 1 and 2
        cache.get(1)?;
        cache.get(2)?;

        // Loading page 3 should evict page 1 and write it to disk as well.
        cache.get(3)?;

        assert_eq!(cache.clock, 0);
        assert_eq!(cache.write_queue.len(), 1);
        assert_eq!(cache.buffer[0].page, pages[3]);

        let mut evicted_page = Page::new(0, cache.pager.page_size as _);
        cache
            .pager
            .read(evicted_page.number, evicted_page.buffer_mut())?;

        assert_eq!(evicted_page, expected);

        Ok(())
    }
}
