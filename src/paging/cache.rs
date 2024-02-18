//! Page cache implementation.
//!
//! At this layer there should be no disk operations, cache instances should
//! only contain an in-memory buffer pool and an eviction policy algorithm.
//! There is no trait yet for different types of algorithms, but if needed one
//! can be created easily based on this concrete implementation.
//!
//! Another important detail is that in order to comply with the Rust borrow
//! rules the cache itself should not return references to pages directly.
//! Instead it should return some owned value that can be used to "index" the
//! cache and obtain a reference whenever the cache user decides so.
//!
//! Doing this saves us from some "can't borrow as mutable more than once" and
//! "can't borrow as mutable because also borrowed as immutable" errors. See
//! the [`super::pager::Pager`] code for details.

use std::{
    collections::HashMap,
    mem,
    ops::{Index, IndexMut},
};

use super::pager::PageNumber;
use crate::storage::page::MemPage;

/// Default value for [`Cache::max_size`].
const DEFAULT_MAX_CACHE_SIZE: usize = 1024;

/// Minimum allowed cache size.
///
/// Due to how the [`crate::storage::BTree`] structure works, we need to be able
/// to store at least 2 pages in the cache because at any given moment one of
/// the pages could be in "overflow" state, which causes the page to become
/// unevictable. If the cache can only hold one page and the page cannot be
/// evicted, then we won't be able to read more pages. See
/// [`Cache::is_evictable`] for more details on this.
const MIN_CACHE_SIZE: usize = 2;

/// The buffer pool is made of a list of frames. Each frame holds a page, a
/// reference bit and a dirty bit.
#[derive(Debug, PartialEq)]
struct Frame {
    /// In-memory representation of a page.
    page: MemPage,
    /// Reference bit. It's set to 1 every time this frame is accessed.
    reference: bool,
    /// Dirty bit. Set to 1 every time the page is modified.
    dirty: bool,
    /// Whether this page is currently pinned.
    pinned: bool,
}

/// Frames are identified by their index in [`Cache::buffer`].
type FrameId = usize;

/// Page read cache with clock eviction policy.
///
/// Pages are loaded into a buffer pool until the buffer is full, then the clock
/// algorithm kicks in. Each page is stored in a [`Frame`] inside the buffer
/// pool, which holds additinal metadata such as reference bit and dirty flag.
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
/// to 1, so that the cache client, [`super::pager::Pager`] in this case, can
/// determine whether an evicted page should be written to disk.
pub(super) struct Cache {
    /// Buffer pool.
    buffer: Vec<Frame>,
    /// Maximum number of pages that can be stored in memory.
    max_size: usize,
    /// Page table. Maps page numbers to frame indexes in the buffer pool.
    pages: HashMap<PageNumber, FrameId>,
    /// Clock pointer. Keeps cycling around the buffer pool.
    clock: FrameId,
}

impl Frame {
    /// Builds a new frame with [`Frame::reference`] set to 0.
    fn new_unreferenced(page: MemPage) -> Self {
        Self {
            page,
            reference: false,
            dirty: false,
            pinned: false,
        }
    }

    /// Builds a new frame with [`Frame::reference`] set to 1.
    fn new_referenced(page: MemPage) -> Self {
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
    pub fn is_dirty(&self) -> bool {
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

impl Default for Cache {
    fn default() -> Self {
        Self {
            clock: 0,
            max_size: DEFAULT_MAX_CACHE_SIZE,
            buffer: Vec::with_capacity(DEFAULT_MAX_CACHE_SIZE),
            pages: HashMap::with_capacity(DEFAULT_MAX_CACHE_SIZE),
        }
    }
}

impl Index<FrameId> for Cache {
    type Output = MemPage;

    fn index(&self, frame_id: FrameId) -> &Self::Output {
        &self.buffer[frame_id].page
    }
}

impl IndexMut<FrameId> for Cache {
    fn index_mut(&mut self, frame_id: FrameId) -> &mut Self::Output {
        &mut self.buffer[frame_id].page
    }
}

impl Cache {
    /// Creates a new default cache. Max size is set to [`DEFAULT_MAX_CACHE_SIZE`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the value of [`Self::max_size`].
    ///
    /// # Panics
    ///
    /// This function panics if `max_size` < [`MIN_CACHE_SIZE`].
    pub fn with_max_size(max_size: usize) -> Self {
        assert!(
            max_size >= MIN_CACHE_SIZE,
            "buffer pool size must be at least {MIN_CACHE_SIZE}"
        );

        Self {
            clock: 0,
            max_size,
            buffer: Vec::with_capacity(max_size),
            pages: HashMap::with_capacity(max_size),
        }
    }

    pub fn contains(&self, page_number: PageNumber) -> bool {
        self.pages.contains_key(&page_number)
    }

    /// Returns a frame ID that can be used to access the in-memory page.
    ///
    /// If the page is not cached or has been invalidated by calling
    /// [`Self::invalidate`], then this function returns [`None`].
    pub fn get(&mut self, page_number: PageNumber) -> Option<FrameId> {
        self.try_reference_page(page_number)
    }

    /// Same as [`Self::get`] but marks the page as dirty.
    pub fn get_mut(&mut self, page_number: PageNumber) -> Option<FrameId> {
        self.try_reference_page(page_number)
            .inspect(|frame_id| self.buffer[*frame_id].mark_dirty())
    }

    /// Returns the frame ID of `page_number` and sets if ref bit to 1.
    ///
    /// If the page can't be found then nothing happens and [`None`] is
    /// returned.
    fn try_reference_page(&mut self, page_number: PageNumber) -> Option<usize> {
        self.pages.get(&page_number).map(|frame_id| {
            self.buffer[*frame_id].reference();
            *frame_id
        })
    }

    /// Loads a page into the buffer pool.
    ///
    /// Doesn't matter where the page comes from, it could have been created in
    /// memory or read from disk. At this level we need an owned version of the
    /// page.
    pub fn load(&mut self, page: MemPage) -> Option<MemPage> {
        // Buffer is not full, push the page and return.
        if self.buffer.len() < self.max_size {
            self.pages.insert(page.number(), self.buffer.len());
            self.buffer.push(Frame::new_unreferenced(page));

            return None;
        }

        // Buffer is full, evict some page and load the new one.
        self.cycle_clock();
        self.pages.insert(page.number(), self.clock);

        let evict = mem::replace(&mut self.buffer[self.clock], Frame::new_referenced(page));
        self.pages.remove(&evict.page.number());

        Some(evict.page)
    }

    /// Returns `true` if the next page to be evicted is dirty.
    ///
    /// Notice that this function **does not actually evict any page**. The
    /// eviction will take place when [`Self::load`] is called.
    pub fn must_evict_dirty_page(&mut self) -> bool {
        if self.buffer.len() < self.max_size {
            return false;
        }

        self.cycle_clock();
        self.buffer[self.clock].is_dirty()
    }

    /// Cycles the clock until it points to a page that can be safely evicted.
    ///
    /// Pages that could not be evicted in the process are unreferenced.
    fn cycle_clock(&mut self) {
        let initial_location = self.clock;

        while !self.is_evictable(self.clock) {
            self.buffer[self.clock].unreference();
            self.tick();

            // This could end up in an infinite loop in some situations:
            //
            // - All pages are pinned.
            // - Clock unsets references but pages are re-referenced again.
            //
            // None of these scenarios should actually happen because callers
            // should unpin pages when they're done with them and at some point
            // in time somebody should stop reading and referencing pages so
            // that at least one can be evicted. The clock can cycle through the
            // buffer multiple times, that's not a bug per se, it's just slow.
            #[cfg(debug_assertions)]
            if self.clock == initial_location {
                todo!("clock has gone full circle without evicting any page");
            }
        }
    }

    /// Moves the clock to the next frame in the buffer.
    fn tick(&mut self) {
        // There are different ways to do wrapping addition. We could use
        // modulus but a single branch instruction is probably faster, although
        // there are no benchmarks to prove this and it probably doesn't even
        // matter in practice. Still, since the clock could be cycling through
        // a huge buffer multiple times, this should be investigated.
        self.clock += 1;
        if self.clock >= self.buffer.len() {
            self.clock = 0;
        }
    }

    /// Returns `true` if the page stored at the given frame can be safely
    /// evicted.
    fn is_evictable(&self, frame_id: FrameId) -> bool {
        let frame = &self.buffer[frame_id];

        // TODO: Use the pin/unpin mechanism in the BTree balance algorithm to
        // maintain overflow pages in memory. The cache should know as little as
        // possible about pages. Ideally the cache should be generic, it's just
        // a replacement algorithm, doesn't matter exaclty "what" is being
        // replaced.
        //
        // Or maybe once we add transactions and stuff we come to the conclusion
        // that the replacement algorithm should be smart about what pages can
        // and cannot be replaced based on what the DB is doing, who knows ¯\_(ツ)_/¯
        !frame.is_referenced() && !frame.is_pinned() && !frame.page.is_overflow()
    }

    /// Sets the dirty flag of the given page back to 0.
    pub fn mark_clean(&mut self, page_number: PageNumber) {
        if let Some(frame_id) = self.get(page_number) {
            self.buffer[frame_id].mark_clean();
        }
    }

    pub fn mark_dirty(&mut self, page_number: PageNumber) {
        if let Some(frame_id) = self.get(page_number) {
            self.buffer[frame_id].mark_dirty();
        }
    }

    /// Marks a page as unevictable.
    ///
    /// Returns `true` if the page was present and pinned.
    pub fn pin(&mut self, page: PageNumber) -> bool {
        self.pages.get(&page).map_or(false, |frame_id| {
            self.buffer[*frame_id].pin();
            true
        })
    }

    /// Marks the `page` as evictable again.
    ///
    /// Returns true if the page was present and upinned.
    pub fn unpin(&mut self, page: PageNumber) -> bool {
        self.pages.get(&page).map_or(false, |frame_id| {
            self.buffer[*frame_id].unpin();
            true
        })
    }

    /// Invalidates a cached page. Requesting this page again will yield
    /// [`None`].
    pub fn invalidate(&mut self, page_number: PageNumber) {
        if let Some(frame_id) = self.pages.remove(&page_number) {
            let frame = &mut self.buffer[frame_id];
            frame.unreference();
            frame.mark_clean();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{super::pager::PageNumber, Cache};
    use crate::storage::page::{Cell, InitEmptyPage, MemPage, Page};

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

        fn build(self) -> (Cache, Vec<MemPage>) {
            let page_size = 256;

            let pages = (0..self.number_of_pages as PageNumber).map(|i| {
                let mut page = Page::init(i, page_size as _);
                page.push(Cell::new(vec![
                    i as u8;
                    Page::max_payload_size(page_size as u16)
                        as usize
                ]));
                MemPage::Btree(page)
            });

            let mut cache = Cache::with_max_size(self.max_size);

            cache.load_many(match self.prefetch {
                Prefetch::AllPages => pages.clone().take(self.number_of_pages),
                Prefetch::UntilBufferIsFull => pages.clone().take(self.max_size),
                Prefetch::None => pages.clone().take(0),
            });

            (cache, pages.collect())
        }
    }

    impl Cache {
        fn load_many<P: IntoIterator<Item = MemPage>>(&mut self, pages: P) {
            for page in pages {
                self.load(page.into());
            }
        }

        fn builder() -> Builder {
            Builder::new()
        }
    }

    #[test]
    fn fill_buffer() {
        let (cache, pages) = Cache::builder()
            .total_pages(3)
            .max_size(3)
            .prefetch_until_buffer_is_full()
            .build();

        assert_eq!(cache.buffer.len(), pages.len());
        assert_eq!(cache.clock, 0);

        for (i, page) in pages.into_iter().enumerate() {
            assert_eq!(page, cache.buffer[i].page);
            assert_eq!(cache.pages[&page.number()], i);
            assert!(!cache.buffer[i].reference);
        }
    }

    #[test]
    fn start_clock_when_buffer_is_full() {
        let (cache, pages) = Cache::builder()
            .total_pages(6)
            .max_size(3)
            .prefetch_all_pages()
            .build();

        assert_eq!(cache.buffer.len(), cache.max_size);
        assert_eq!(cache.clock, cache.max_size - 1);

        for (i, page) in pages[cache.max_size..].iter().enumerate() {
            assert_eq!(*page, cache.buffer[i].page);
            assert_eq!(cache.pages[&page.number()], i);
        }
    }

    #[test]
    fn reference_page() {
        let (mut cache, pages) = Cache::builder()
            .total_pages(3)
            .max_size(3)
            .prefetch_until_buffer_is_full()
            .build();

        for (i, page) in pages.into_iter().enumerate() {
            let index = cache.get(page.number());
            assert_eq!(Some(i), index);
            assert_eq!(page, cache.buffer[i].page);
            assert!(cache.buffer[i].reference);
            assert!(!cache.buffer[i].dirty);
        }
    }

    #[test]
    fn mark_dirty() {
        let (mut cache, pages) = Cache::builder()
            .total_pages(3)
            .max_size(3)
            .prefetch_until_buffer_is_full()
            .build();

        for (i, page) in pages.into_iter().enumerate() {
            let index = cache.get_mut(page.number());
            assert_eq!(Some(i), index);
            assert_eq!(page, cache.buffer[i].page);
            assert!(cache.buffer[i].reference);
            assert!(cache.buffer[i].dirty);
        }
    }

    #[test]
    fn evict_first_unreferenced_page() {
        let (mut cache, pages) = Cache::builder()
            .total_pages(4)
            .max_size(3)
            .prefetch_until_buffer_is_full()
            .build();

        // Reference all pages except last one.
        for page in &pages[..cache.max_size - 1] {
            cache.get(page.number());
        }

        // Should evict page 2 and replace it with page 3.
        let evicted = cache.load(pages[3].clone());

        assert_eq!(evicted, Some(pages[2].clone()));

        assert_eq!(cache.clock, cache.max_size - 1);

        assert_eq!(
            cache.buffer[cache.max_size - 1].page,
            pages[pages.len() - 1]
        );

        assert_eq!(
            cache.pages[&pages[pages.len() - 1].number()],
            cache.max_size - 1
        );

        for (i, page) in pages[..cache.max_size - 1].iter().enumerate() {
            assert_eq!(*page, cache.buffer[i].page);
            assert_eq!(cache.pages[&page.number()], i);
        }
    }

    #[test]
    fn dont_evict_pinned_page() {
        let (mut cache, pages) = Cache::builder()
            .total_pages(4)
            .max_size(3)
            .prefetch_until_buffer_is_full()
            .build();

        let pinned = cache.pin(0);

        // Should not evict page 0 because it's pinned.
        cache.load(pages[3].clone());

        assert_eq!(pinned, true);
        assert_eq!(cache.clock, 1);
        assert_eq!(cache.buffer[0].page, pages[0]);
        assert_eq!(cache.buffer[1].page, pages[3]);
        assert_eq!(cache.buffer[2].page, pages[2]);
    }
}
