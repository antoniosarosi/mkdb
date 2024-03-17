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
//!
//! It's also important to provide an API for "allocating a page" in the cache.
//! In other words, "give me a page where I can write stuff without having to
//! allocate anything". We do this through [`Cache::map`] and that allows us to
//! reuse all the allocated page buffers when reading from disk.

use std::{
    collections::HashMap,
    mem,
    ops::{Index, IndexMut},
};

use super::pager::PageNumber;
use crate::{db::DEFAULT_PAGE_SIZE, storage::page::MemPage};

/// Default value for [`Cache::max_size`].
pub(crate) const DEFAULT_MAX_CACHE_SIZE: usize = 1024;

/// Minimum allowed cache size.
///
/// Due to how the [`crate::storage::BTree`] structure works, we need to be able
/// to store at least 2 pages in the cache because at any given moment one of
/// the pages could be in "overflow" state, which causes the page to become
/// unevictable. If the cache can only hold one page and that page cannot be
/// evicted, then we won't be able to read more pages. See
/// [`Cache::is_evictable`] for more details on this.
///
/// In theory, a cache of size 1 could work because when
/// [`crate::storage::BTree::insert`] overflows a page it immediately calls
/// [`crate::storage::BTree::balance`], which moves all the cells out of the
/// page leaving it empty without reading other pages in the process.
///
/// But if we change the code in the balancing algorithm and we read another
/// page before clearing overflow buffers then that bug is gonna be hard to
/// spot.
pub(crate) const MIN_CACHE_SIZE: usize = 2;

/// Maximum percentage of pages that can be pinned at any given moment.
///
/// Once this percentage is reached the cache will refuse to pin more pages
/// until the percentage goes back below the limit.
///
/// **This number must be a value between 0 and 100**. Ideally not 0, because
/// that's useless.
const DEFAULT_PIN_PERCENTAGE_LIMIT: f32 = 60.0;

/// Reference bit. It's set to 1 every time a [`Frame`] is accessed.
const REF_FLAG: u8 = 0b001;

/// Dirty bit. Set to 1 every time a cached [`MemPage`] is modified.
const DIRTY_FLAG: u8 = 0b010;

/// Whether the [`MemPage`] is currently pinned.
const PINNED_FLAG: u8 = 0b100;

/// The buffer pool is made of a list of frames.
///
/// Each frame holds a page and a collection of bit flags. Possible flags are
/// [`REF_FLAG`], [`DIRTY_FLAG`] and [`PINNED_FLAG`].
#[derive(Debug, PartialEq)]
struct Frame {
    /// In-memory representation of a page.
    page: MemPage,
    /// The page number stored in this [`Frame`]. Inverse of [`Cache::pages`].
    page_number: PageNumber,
    /// Bit flags.
    flags: u8,
}

impl Frame {
    /// Builds a new frame with [`REF_FLAG`] set to 0.
    fn new(page_number: PageNumber, page: MemPage) -> Self {
        Self {
            page,
            page_number,
            flags: 0,
        }
    }

    /// Sets the given flag(s) to 1.
    fn set(&mut self, flags: u8) {
        self.flags |= flags
    }

    /// Sets the given flag(s) to 0.
    fn unset(&mut self, flags: u8) {
        self.flags &= !flags
    }

    /// Returns `true` if all the given flags are set.
    fn is_set(&self, flags: u8) -> bool {
        self.flags & flags == flags
    }
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
/// If the clock finds pages that are referenced in the process it resets their
/// reference bit back to 0 so that they can be evicted the next time the clock
/// passes by. This algorithm is an approximation of LRU (Least Recently Used)
/// that does not require timestamps.
///
/// Following the example above, the clock pointer already points at page 1,
/// which is not currently referenced. So if we want to load page 5 we can
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
/// to 1 so that the cache client, [`super::pager::Pager`] in this case, can
/// determine whether an evicted page should be written to disk.
pub(crate) struct Cache {
    /// Buffer pool.
    buffer: Vec<Frame>,
    /// Page table. Maps page numbers to frame indexes in the buffer pool.
    pages: HashMap<PageNumber, FrameId>,
    /// Clock pointer. Keeps cycling around the buffer pool.
    clock: FrameId,
    /// Maximum number of pages that can be stored in memory.
    max_size: usize,
    /// Size of each page.
    page_size: usize,
    /// Maximum percentage of pages that can be pinned at once.
    pin_percentage_limit: f32,
    /// Number of pinned pages.
    pinned_pages: usize,
}

/// Cache builder.
pub(crate) struct Builder {
    max_size: usize,
    page_size: usize,
    pin_percentage_limit: f32,
}

impl Builder {
    /// Creates a new builder. Everything is set to default values.
    pub fn new() -> Self {
        Self {
            max_size: DEFAULT_MAX_CACHE_SIZE,
            page_size: DEFAULT_PAGE_SIZE,
            pin_percentage_limit: DEFAULT_PIN_PERCENTAGE_LIMIT,
        }
    }

    /// Sets the maximum size of the cache.
    ///
    /// `max_size` must be greater than [`MIN_CACHE_SIZE`] for the cache to work
    /// properly.
    pub fn max_size(&mut self, max_size: usize) -> &mut Self {
        assert!(
            max_size >= MIN_CACHE_SIZE,
            "buffer pool size must be at least {MIN_CACHE_SIZE}"
        );

        self.max_size = max_size;
        self
    }

    /// Sets the maximum percentage of pages that can be pinned at once.
    ///
    /// [`Cache::pin`] will return `false` once this percentage is reached.
    pub fn pin_percentage_limit(&mut self, pin_percentage_limit: f32) -> &mut Self {
        assert!(
            (0.0..=100.0).contains(&pin_percentage_limit),
            "pin_percentage_limit must a percentage (0..=100) and not {pin_percentage_limit}, in case that's not clear enough"
        );

        self.pin_percentage_limit = pin_percentage_limit;
        self
    }

    /// Page size used for the initial page allocations.
    ///
    /// In theory, the cache can store pages of different sizes, but it should
    /// not.
    pub fn page_size(&mut self, page_size: usize) -> &mut Self {
        self.page_size = page_size;
        self
    }

    /// Builds the cache and pre-allocates the buffer pool and page table.
    pub fn build(&self) -> Cache {
        Cache {
            clock: 0,
            pinned_pages: 0,
            pin_percentage_limit: self.pin_percentage_limit,
            max_size: self.max_size,
            page_size: self.page_size,
            buffer: Vec::with_capacity(self.max_size),
            pages: HashMap::with_capacity(self.max_size),
        }
    }
}

impl Default for Cache {
    fn default() -> Self {
        Self::builder().build()
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

    /// Build it as you wish if defaults aren't good enough :)
    pub fn builder() -> Builder {
        Builder::new()
    }

    /// Shortcut for `Cache::builder().max_size(max_size).build()`.
    pub fn with_max_size(max_size: usize) -> Self {
        Self::builder().max_size(max_size).build()
    }

    /// Shortcut for `Cache::builder().page_size(page_size).build()`.
    pub fn with_page_size(page_size: usize) -> Self {
        Self::builder().page_size(page_size).build()
    }

    /// Max number of pages that can be stored in this cache.
    pub fn max_size(&self) -> usize {
        self.max_size
    }

    /// Page size used to allocate pages when calling [`Self::map`].
    pub fn page_size(&self) -> usize {
        self.page_size
    }

    /// Returns `true` if the given page is cached.
    pub fn contains(&self, page_number: &PageNumber) -> bool {
        self.pages.contains_key(page_number)
    }

    /// Returns a [`FrameId`] that can be used to access the in-memory page.
    ///
    /// If the page is not cached or has been invalidated by calling
    /// [`Self::invalidate`], then this function returns [`None`].
    pub fn get(&mut self, page_number: PageNumber) -> Option<FrameId> {
        self.ref_page(page_number)
    }

    /// Same as [`Self::get`] but marks the page as dirty.
    pub fn get_mut(&mut self, page_number: PageNumber) -> Option<FrameId> {
        self.ref_page(page_number)
            .inspect(|frame_id| self.buffer[*frame_id].set(DIRTY_FLAG))
    }

    /// Returns many mutable references at once if possible.
    ///
    /// Otherwise [`None`] is returned.
    pub fn get_many_mut<const N: usize>(
        &mut self,
        pages: [PageNumber; N],
    ) -> Option<[&mut MemPage; N]> {
        self.pages.get_many_mut(pages.each_ref()).map(|frame_ids| {
            self.buffer
                .get_many_mut(frame_ids.map(|frame_id| *frame_id))
                .unwrap()
                .map(|frame| {
                    frame.set(REF_FLAG | DIRTY_FLAG);
                    &mut frame.page
                })
        })
    }

    /// Returns the [`FrameId`] of `page_number` and automatically sets its ref
    /// bit to 1.
    ///
    /// If the page can't be found then nothing happens and [`None`] is
    /// returned.
    fn ref_page(&mut self, page_number: PageNumber) -> Option<usize> {
        self.pages.get(&page_number).map(|frame_id| {
            self.buffer[*frame_id].set(REF_FLAG);
            *frame_id
        })
    }

    /// Maps a [`PageNumber`] to a [`FrameId`].
    ///
    /// There are 3 possible cases:
    ///
    /// 1. The [`PageNumber`] is already cached and we know the [`FrameId`].
    /// Pretty much a no-op.
    ///
    /// 2. The cache buffer is not full yet, so we can obtain frames without
    /// evicting anything. Again, almost a no-op except for the allocation.
    ///
    /// 3. We must evict a page in order to load a new one. This is when the
    /// clock algorithm runs. Worst case is O(n), but figuring out exaclty how
    /// bad "n" is... definitely not easy.
    ///
    /// In any case, there is no IO at the cache level, all of this is done in
    /// memory. IO is controlled mostly by the [`super::pager`] module.
    ///
    /// There's an important detail for layers on top of this one: [`Self::map`]
    /// simply gives you a [`FrameId`]. It doesn't change the content of the
    /// page that was originally stored in that frame.
    pub fn map(&mut self, page_number: PageNumber) -> FrameId {
        // Already cached.
        if let Some(frame_id) = self.pages.get(&page_number) {
            return *frame_id;
        }

        // Buffer is not full. Allocate the new page and return the frame ID.
        //
        // TODO: We should pre-allocate all the pages at the beginning and
        // ideally using a single giant allocation for the buffers in order to
        // improve locality. But that requires more unsafe code and dealing with
        // drops and ManuallyDrop and bla bla. Already wrote like 2000 lines of
        // unsafe code for the slotted page so there's no motivation :)
        if self.buffer.len() < self.max_size {
            let frame_id = self.buffer.len();
            self.pages.insert(page_number, frame_id);

            self.buffer
                .push(Frame::new(page_number, MemPage::alloc(self.page_size)));

            return frame_id;
        }

        // Buffer is full, find a page that we can evict.
        self.cycle_clock();

        // This page doesn't exist in the cache anymore.
        let frame = &mut self.buffer[self.clock];
        self.pages.remove(&frame.page_number);

        // Now the frame holds the new page.
        frame.page_number = page_number;
        frame.set(REF_FLAG);
        self.pages.insert(page_number, self.clock);

        self.clock
    }

    /// Loads a page from memory into the cache.
    ///
    /// The evicted page is returned. This function shouldn't be used much since
    /// it requires allocating a page in order to call this function and
    /// dropping the evicted one. Ideally [`Self::map`] should be called in
    /// order to reuse allocations.
    pub fn load(&mut self, page_number: PageNumber, page: MemPage) -> MemPage {
        let frame_id = self.map(page_number);
        mem::replace(&mut self.buffer[frame_id].page, page)
    }

    /// Returns `true` if the next page to be evicted is dirty.
    ///
    /// Notice that this function **does not actually evict any page**. The
    /// eviction will take place when [`Self::map`] is called.
    pub fn must_evict_dirty_page(&mut self) -> bool {
        if self.buffer.len() < self.max_size {
            return false;
        }

        self.cycle_clock();
        self.buffer[self.clock].is_set(DIRTY_FLAG)
    }

    /// Cycles the clock until it points to a page that can be safely evicted.
    ///
    /// Pages that could not be evicted in the process are unreferenced.
    fn cycle_clock(&mut self) {
        let initial_location = self.clock;
        let mut rounds = 0;

        while !self.is_evictable(self.clock) {
            self.buffer[self.clock].unset(REF_FLAG);
            self.tick();

            // This technically looks like it could end up in an infinite loop
            // in some situations but it's almost impossible.
            //
            // 1. Infinite loop if all pages are pinned. Can only happen if
            // `Cache::pin_percentage_limit == 0.0` and callers don't unpin the
            // pages when they are done with them.
            //
            // 2. Clock unsets references as it cycles through the buffer but
            // other threads acquire references again. This thing is not even
            // multi-threaded yet so we'll consider this problem later.
            //
            // Regardless, the clock can cycle through the buffer multiple
            // times, that's not a bug per se, it's just slow. We only need to
            // make sure that it doesn't enter an infinite loop.
            #[cfg(debug_assertions)]
            {
                if rounds == 3 {
                    todo!("clock has gone full circle {rounds} times without evicting any page");
                }

                if self.clock == initial_location {
                    rounds += 1;
                }
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
        // Researching useless micro-optimizations is definitely fun :)
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
        !frame.is_set(REF_FLAG) && !frame.is_set(PINNED_FLAG) && !frame.page.is_overflow()
    }

    /// Sets the given flags of a page to 1.
    ///
    /// Returns `true` if the page was present in the cache and its flags were
    /// updated.
    fn set_flags(&mut self, page_number: PageNumber, flags: u8) -> bool {
        self.pages.get(&page_number).map_or(false, |frame_id| {
            self.buffer[*frame_id].set(flags);
            true
        })
    }

    /// Sets the flags of a page back to 0.
    ///
    /// Returns `true` if the page was present in the cache and its flags were
    /// updated.
    fn unset_flags(&mut self, page_number: PageNumber, flags: u8) -> bool {
        self.pages.get(&page_number).map_or(false, |frame_id| {
            self.buffer[*frame_id].unset(flags);
            true
        })
    }

    /// Manually marks the page as dirty.
    ///
    /// Returns `true` if the page was present and marked as dirty.
    pub fn mark_dirty(&mut self, page_number: PageNumber) -> bool {
        self.set_flags(page_number, DIRTY_FLAG)
    }

    /// Sets the dirty flag of the given page back to 0.
    ///
    /// Returns `true` if the page was present and marked as clean.
    pub fn mark_clean(&mut self, page_number: PageNumber) -> bool {
        self.unset_flags(page_number, DIRTY_FLAG)
    }

    /// Marks a page as unevictable.
    ///
    /// Returns `true` if the page was present and pinned.
    pub fn pin(&mut self, page_number: PageNumber) -> bool {
        let pinned_percentage = self.pinned_pages as f32 / self.max_size as f32 * 100.0;

        if pinned_percentage >= self.pin_percentage_limit {
            return false;
        }

        let pinned = self.set_flags(page_number, PINNED_FLAG);

        if pinned {
            self.pinned_pages += 1;
        }

        pinned
    }

    /// Marks the `page` as evictable again.
    ///
    /// Returns true if the page was present and upinned.
    pub fn unpin(&mut self, page_number: PageNumber) -> bool {
        let unpinned = self.unset_flags(page_number, PINNED_FLAG);

        if unpinned {
            self.pinned_pages -= 1;
        }

        unpinned
    }

    /// Invalidates a cached page. Requesting this page again will yield
    /// [`None`].
    pub fn invalidate(&mut self, page_number: PageNumber) {
        if let Some(frame_id) = self.pages.remove(&page_number) {
            self.buffer[frame_id].flags = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{super::pager::PageNumber, Builder, Cache};
    use crate::{
        paging::cache::{DIRTY_FLAG, PINNED_FLAG, REF_FLAG},
        storage::page::{Cell, MemPage, Page},
    };

    /// Allows us to prepare the cache and then just [`assert_eq!`] stuff.
    struct TestBuilder {
        builder: Builder,
        number_of_pages: usize,
        prefetch: Prefetch,
    }

    /// How many pages to load into the cache before returning it.
    enum Prefetch {
        AllPages,
        UntilBufferIsFull,
        None,
    }

    impl TestBuilder {
        fn new() -> Self {
            Self {
                number_of_pages: 3,
                prefetch: Prefetch::None,
                builder: Cache::builder(),
            }
        }

        /// Total number of pages that should be created and returned.
        ///
        /// Don't confuse this with [`Self::max_size`], which is the size of
        /// the cache.
        fn total_pages(&mut self, number_of_pages: usize) -> &mut Self {
            self.number_of_pages = number_of_pages;
            self
        }

        fn max_size(&mut self, max_size: usize) -> &mut Self {
            self.builder.max_size(max_size);
            self
        }

        fn pin_percentage_limit(&mut self, pin_percentage_limit: f32) -> &mut Self {
            self.builder.pin_percentage_limit(pin_percentage_limit);
            self
        }

        fn prefetch(&mut self, prefetch: Prefetch) -> &mut Self {
            self.prefetch = prefetch;
            self
        }

        fn prefetch_all_pages(&mut self) -> &mut Self {
            self.prefetch(Prefetch::AllPages)
        }

        fn prefetch_until_buffer_is_full(&mut self) -> &mut Self {
            self.prefetch(Prefetch::UntilBufferIsFull)
        }

        fn build(&self) -> (Cache, Vec<MemPage>) {
            let page_size = 256;

            let pages = (0..self.number_of_pages as PageNumber).map(|i| {
                let mut page = Page::alloc(page_size);
                page.push(Cell::new(vec![
                    i as u8;
                    Page::ideal_max_payload_size(page_size, 1)
                        as usize
                ]));
                MemPage::Btree(page)
            });

            let mut cache = self.builder.build();

            let prefetch = (0..pages.len() as PageNumber).zip(pages.clone());

            cache.load_many(match self.prefetch {
                Prefetch::AllPages => prefetch.take(self.number_of_pages),
                Prefetch::UntilBufferIsFull => prefetch.take(cache.max_size()),
                Prefetch::None => prefetch.take(0),
            });

            (cache, pages.collect())
        }
    }

    impl Cache {
        fn load_many<P: IntoIterator<Item = (PageNumber, MemPage)>>(&mut self, pages: P) {
            for (page_number, page) in pages {
                let frame_id = self.map(page_number);
                self.buffer[frame_id].page = page;
            }
        }

        fn test() -> TestBuilder {
            TestBuilder::new()
        }
    }

    #[test]
    fn fill_buffer() {
        let (cache, pages) = Cache::test()
            .total_pages(3)
            .max_size(3)
            .prefetch_until_buffer_is_full()
            .build();

        assert_eq!(cache.buffer.len(), pages.len());
        assert_eq!(cache.clock, 0);

        for (i, page) in pages.into_iter().enumerate() {
            assert_eq!(cache.buffer[i].page, page);
            assert_eq!(cache.pages[&(i as PageNumber)], i);
        }
    }

    #[test]
    fn start_clock_when_buffer_is_full() {
        let (cache, pages) = Cache::test()
            .total_pages(6)
            .max_size(3)
            .prefetch_all_pages()
            .build();

        assert_eq!(cache.clock, cache.max_size - 1);

        for ((i, page), page_number) in pages[cache.max_size..].iter().enumerate().zip(3..=5) {
            assert_eq!(*page, cache.buffer[i].page);
            assert_eq!(cache.pages[&page_number], i);
        }
    }

    #[test]
    fn reference_page() {
        let (mut cache, pages) = Cache::test()
            .total_pages(3)
            .max_size(3)
            .prefetch_until_buffer_is_full()
            .build();

        for (i, page) in pages.into_iter().enumerate() {
            let index = cache.get(i as PageNumber);
            assert_eq!(Some(i), index);
            assert_eq!(page, cache.buffer[i].page);
            assert_eq!(cache.buffer[i].flags, REF_FLAG);
        }
    }

    #[test]
    fn mark_dirty() {
        let (mut cache, pages) = Cache::test()
            .total_pages(3)
            .max_size(3)
            .prefetch_until_buffer_is_full()
            .build();

        for (i, page) in pages.into_iter().enumerate() {
            let index = cache.get_mut(i as PageNumber);
            assert_eq!(Some(i), index);
            assert_eq!(page, cache.buffer[i].page);
            assert_eq!(cache.buffer[i].flags, REF_FLAG | DIRTY_FLAG);
        }
    }

    #[test]
    fn evict_first_unreferenced_page() {
        let (mut cache, pages) = Cache::test()
            .total_pages(4)
            .max_size(3)
            .prefetch_until_buffer_is_full()
            .build();

        // Reference all pages except last one.
        for i in 0..2 {
            cache.get(i);
        }

        // Should evict page 2 and replace it with page 3.
        let evicted = cache.load(3, pages[3].clone());

        assert_eq!(evicted, pages[2].clone());

        assert_eq!(cache.clock, cache.max_size - 1);

        assert_eq!(
            cache.buffer[cache.max_size - 1].page,
            pages[pages.len() - 1]
        );

        assert_eq!(cache.pages[&3], cache.max_size - 1);

        for (i, page) in pages[..cache.max_size - 1].iter().enumerate() {
            assert_eq!(*page, cache.buffer[i].page);
            assert_eq!(cache.pages[&(i as PageNumber)], i);
        }
    }

    #[test]
    fn dont_evict_pinned_page() {
        let (mut cache, pages) = Cache::test()
            .total_pages(4)
            .max_size(3)
            .prefetch_until_buffer_is_full()
            .build();

        let pinned = cache.pin(0);

        // Should not evict page 0 because it's pinned.
        cache.load(3, pages[3].clone());

        assert!(pinned);
        assert_eq!(cache.buffer[0].flags, PINNED_FLAG);
        assert_eq!(cache.clock, 1);
        assert_eq!(cache.buffer[0].page, pages[0]);
        assert_eq!(cache.buffer[1].page, pages[3]);
        assert_eq!(cache.buffer[2].page, pages[2]);
    }

    #[test]
    fn dont_pin_page_if_pin_percentage_limit_reached() {
        let (mut cache, _) = Cache::test()
            .total_pages(10)
            .max_size(10)
            .pin_percentage_limit(30.0)
            .prefetch_until_buffer_is_full()
            .build();

        for page in 0..=2 {
            cache.pin(page);
        }

        let pinned = cache.pin(3);

        assert!(!pinned);
        assert_eq!(cache.buffer[3].flags, 0);
        assert_eq!(cache.pinned_pages, 3);
    }

    #[test]
    fn pin_again_if_pin_percentage_goes_below_limit() {
        let (mut cache, _) = Cache::test()
            .total_pages(10)
            .max_size(10)
            .pin_percentage_limit(50.0)
            .prefetch_until_buffer_is_full()
            .build();

        for page in 0..=4 {
            cache.pin(page);
        }

        let unpinned = cache.unpin(0);
        let pinned = cache.pin(5);

        assert!(unpinned);
        assert!(pinned);
        assert_eq!(cache.buffer[0].flags, 0);
        assert_eq!(cache.buffer[5].flags, PINNED_FLAG);
        assert_eq!(cache.pinned_pages, 5);
    }
}
