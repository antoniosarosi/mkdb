//! Slotted page implementation tuned for index organized storage. Initially
//! this was going to be a generic slotted page implementation that would serve
//! as a building block for both index organized storage (like SQLite) and tuple
//! oriented storage (like Postgres), but the idea has been dropped in favor of
//! simplicity.
//!
//! The main difference is that when using index organized storage we never
//! point to slot indexes from outside the page, so there's no need to attempt
//! to maintain their current position for as long as possible. This [commit]
//! contains the last version that used to do so. Another benefit of the current
//! aproach is that BTree nodes and disk pages are the same thing, because
//! everything is stored in a BTree, so we need less generics, less types and
//! less code. Take a look at the [btree.c] file from SQLite 2.X.X for the
//! inspiration.
//!
//! Check these lectures for some background on slotted pages and data storage:
//!
//! - [F2023 #03 - Database Storage Part 1 (CMU Intro to Database Systems)](https://www.youtube.com/watch?v=DJ5u5HrbcMk)
//! - [F2023 #04 - Database Storage Part 2 (CMU Intro to Database Systems)](https://www.youtube.com/watch?v=Ra50bFHkeM8)
//!
//! [commit]: https://github.com/antoniosarosi/mkdb/blob/3011003170f02d337f62cdd9f5af0f3b63786144/src/paging/page.rs
//! [btree.c]: https://github.com/antoniosarosi/sqlite2-btree-visualizer/blob/master/src/btree.c

use std::{
    alloc::{self, Layout},
    cmp::Ordering,
    collections::BinaryHeap,
    fmt::Debug,
    mem,
    ptr::NonNull,
};

use super::pager::PageNumber;

pub const BLOCK_HEADER_SIZE: u16 = mem::size_of::<BlockHeader>() as _;
pub const SLOT_SIZE: u16 = mem::size_of::<u16>() as _;

/// BTree slotted page header. Located at the beginning of each page and does
/// not contain variable length data:
///
/// ```text
///                           HEADER                             CONTENT
/// +-----------------------------------------------------+--------------------+
/// | +------------+-------------+---------------+------+ |                    |
/// | | free_space | total_slots | last_used_blk | meta | |                    |
/// | +------------+-------------+---------------+------+ |                    |
/// +-----------------------------------------------------+--------------------+
///                                    PAGE
/// ```
///
/// The alignment and size of this struct doesn't matter. Actual data is
/// appended towards to end of the page, and the page should be aligned to 8
/// bytes. See [`Page`] for more details.
#[derive(Debug)]
#[repr(C)]
pub(crate) struct PageHeader {
    /// Amount of free bytes in this page.
    free_space: u16,
    /// Length of the slot array.
    total_slots: u16,
    /// Offset of the last used block.
    last_used_block: u16,
    /// Last child of this page.
    pub right_child: PageNumber,
}

/// Located at the beginning of each used block. For now we only need to store
/// the block size.
///
/// ```text
///   HEADER        CONTENT
///  +------+--------------------+
///  | size |                    |
///  +------+--------------------+
///         ^                    ^
///         |                    |
///         +--------------------+
///               size bytes
/// ```
///
/// # Alignment
///
/// Blocks are 64 bit aligned. See [`Page`] for more details.
#[repr(C, align(8))]
struct BlockHeader {
    /// Size of this block excluding the header (data only).
    size: u16,
    ///
    pub left_child: PageNumber,
    /// First overflow page. This is 0 if the block did not overflow.
    pub overflow: PageNumber,
}

impl BlockHeader {
    fn content_of(block: NonNull<Self>) -> NonNull<[u8]> {
        // SAFETY: This is not like a memory allocator where we receive addresses
        // from the user and substract the header size to get the actual header.
        // We control all the addresses ourselves, so we never pass invalid
        // pointers to this function.
        let (content, size) = unsafe { (block.add(1).cast::<u8>(), block.as_ref().size as _) };
        NonNull::slice_from_raw_parts(content, size)
    }

    /// Total size of the block including the header.
    fn total_size(&self) -> u16 {
        self.size + BLOCK_HEADER_SIZE
    }
}

/// Slotted page. The page maintains a "slot array" located after the header
/// that grows towards the right. On the opposite side is where used blocks are
/// located, leaving free space in the middle of the page. Each item in the
/// slot array points to one of the used blocks using the block offset,
/// calculated from the start of the page (before the header).
///
/// ```text
///   HEADER   SLOT ARRAY        FREE SPACE             USED BLOCKS
///  +------+----+----+----+------------------+---------+---------+---------+
///  |      | O1 | O2 | O3 | ->            <- | BLOCK 3 | BLOCK 2 | BLOCK 1 |
///  +------+----+----+----+------------------+---------+---------+---------+
///           |    |    |                     ^         ^         ^
///           |    |    |                     |         |         |
///           +----|----|---------------------|---------|---------+
///                |    |                     |         |
///                +----|---------------------|---------+
///                     |                     |
///                     +---------------------+
/// ```
///
/// This is useful for 2 reasons, described below.
///
/// # Cheap Sorting or Reordering
///
/// First, we can rearrange blocks without changing their physical location. We
/// only need to change the offsets in the slot array. For example, if we wanted
/// to reverse the order of the blocks in the figure above, this is all we need
/// to do:
///
/// ```text
///   HEADER   SLOT ARRAY        FREE SPACE             USED BLOCKS
///  +------+----+----+----+------------------+---------+---------+---------+
///  |      | O3 | O2 | O1 | ->            <- | BLOCK 3 | BLOCK 2 | BLOCK 1 |
///  +------+----+----+----+------------------+---------+---------+---------+
///           |    |    |                     ^         ^         ^
///           |    |    |                     |         |         |
///           +----|----|---------------------+         |         |
///                |    |                               |         |
///                +----|--------------------------------         |
///                     |                                         |
///                     +-----------------------------------------+
/// ```
///
/// Now the third block becomes the first one in the list, second block stays
/// the same and the first block becomes the last. Changing offsets in the slot
/// array is much cheaper than moving the blocks around the page. This is needed
/// for BTree pages to sort entries, and this is the only feature implemented
/// here.
///
/// # Slot Positions After Delete Operations
///
/// The second reason this is useful is because we could maintain the same slot
/// number for the rest of the blocks when one of them is deleted. For example,
/// if we deleted block 2 in the figure above, we don't have to compact the slot
/// array like this:
///
/// ```text
///   HEADER SLOT ARRAY      FREE SPACE                USED BLOCKS
///  +------+----+----+-----------------------+---------+---------+---------+
///  |      | O3 | O1 | ->                 <- | BLOCK 3 |   DEL   | BLOCK 1 |
///  +------+----+----+-----------------------+---------+---------+---------+
///           |    |                          ^                   ^
///           |    |                          |                   |
///           +----|--------------------------+                   |
///                |                                              |
///                +----------------------------------------------+
/// ```
///
/// If we did this, then the slot ID of block 1 would change from 2 to 1. But
/// instead we could simply set the slot to NULL or zero:
///
/// ```text
///   HEADER   SLOT ARRAY        FREE SPACE             USED BLOCKS
///  +------+----+----+----+------------------+---------+---------+---------+
///  |      | O3 |NULL| O1 | ->            <- | BLOCK 3 |   DEL   | BLOCK 1 |
///  +------+----+----+----+------------------+---------+---------+---------+
///           |         |                     ^                   ^
///           |         |                     |                   |
///           +---------|---------------------+                   |
///                     |                                         |
///                     +-----------------------------------------+
/// ```
///
/// The remaining slot IDs do not change. This is useful for deleting tuples
/// and delaying index updates as much as possible if using tuple oriented
/// storage, which we are not doing, so you'll never see NULL slots.
/// Check this [commit](https://github.com/antoniosarosi/mkdb/blob/3011003170f02d337f62cdd9f5af0f3b63786144/src/paging/page.rs)
/// for the initial implementation which used to manage both null and non-null
/// slot indexes.
///
/// # Alignment
///
/// In-memory pages should be 64 bit aligned. This will allow the
/// [`BlockHeader`] instances to be 64 bit aligned as well, which in turn will
/// make the first content byte of each block 64 bit aligned. Further alignment
/// within the content should be controlled on upper layers. For example, if we
/// have some schema like `[Int, Varchar(255), Bool]`, we can't worry about
/// individual column alignment here because we're only storing binary data, we
/// know nothing about what's inside.
pub(crate) struct Page {
    /// Page number on disk.
    pub number: PageNumber,
    /// We need to know where the page ends.
    size: u16,
    /// Data that comes from memory and doesn't fit in this page.
    overflow: Vec<(BlockHeader, Vec<u8>, u16)>,
    /// Page header and pointer to the main memory buffer of this page.
    header: NonNull<PageHeader>,
}

impl Page {
    /// Allocates a new empty [`Page`] of the given `size`.
    pub fn new(number: PageNumber, size: u16) -> Self {
        let header = unsafe {
            let header = NonNull::new_unchecked(alloc::alloc_zeroed(Self::layout(size))).cast();

            header.write(PageHeader {
                total_slots: 0,
                last_used_block: size,
                free_space: Self::usable_space(size),
                right_child: 0,
            });

            header
        };

        Self {
            number,
            size,
            overflow: Vec::new(),
            header,
        }
    }

    /// Returns `true` if this page can fit `data` somehow. We don't know if we
    /// need to remove fragmentation yet, but we know the data will fit.
    pub fn can_fit(&self, data: &[u8]) -> bool {
        // The total size includes the 2 byte slot pointer.
        let total_size = Self::total_block_size_for(data) + SLOT_SIZE;

        self.header().free_space >= total_size
    }

    /// Attempts to insert the given `data` in this page with all the available
    /// methods that we have. If possible, data will be inserted without moving
    /// blocks arount, otherwise optimizations have to be made.
    pub fn try_insert(&mut self, data: &[u8]) -> Result<(), ()> {
        self.try_insert_at(self.header().total_slots, data)
    }

    /// Same as [`Self::try_insert`] but receives the slot index that should
    /// point to the data once inserted in the page.
    pub fn try_insert_at(&mut self, index: u16, data: &[u8]) -> Result<(), ()> {
        // TODO: Optimize
        if !self.can_fit(data) {
            return Err(());
        }

        self.try_insert_without_optimizations(index, data)
            .or_else(|_| self.try_insert_compacting_blocks(index, data))
    }

    /// Attempts to replace the bytes stored in the block pointed by the given
    /// slot `index` with the given `data` buffer.
    pub fn try_replace(&mut self, index: u16, data: &[u8]) -> Result<(), ()> {
        let mut block = self.block_at_slot_index(index);

        let new_data_size = Self::aligned_size_of(data);
        let current_data_size = unsafe { block.as_ref().size };
        let total_block_size = unsafe { block.as_ref().total_size() };

        if current_data_size <= new_data_size {
            // Case 1: Given data fits in the same block that was already
            // allocated. Copy the given bytes and update free space counter.
            let free_bytes = current_data_size - new_data_size;
            unsafe {
                BlockHeader::content_of(block).as_mut()[..data.len()].copy_from_slice(data);
                block.as_mut().size -= free_bytes;
            }
            self.header_mut().free_space += free_bytes;
            Ok(())
        } else if self.header().free_space + total_block_size >= Self::total_block_size_for(data) {
            // Case 2: Given data fits in the page but we have to remove the
            // current block.
            // TODO: Optimize this, slot array is shifted left when we remove
            // and then shifted right when we insert.
            self.remove(index);
            self.try_insert_at(index, data)?;
            Ok(())
        } else {
            // No luck.
            Err(())
        }
    }

    /// Attempts to insert data without doing any work at all. This is the Best
    /// case scenario, because data will fit in the "free space":
    /// ```text
    ///   HEADER SLOT ARRAY       FREE SPACE                   BLOCKS
    ///  +------+----+----+-----------------------+---------+---------+---------+
    ///  |      | O3 | 01 | ->                 <- | BLOCK 3 |   DEL   | BLOCK 1 |
    ///  +------+----+----+-----------------------+---------+---------+---------+
    ///                               ^
    ///                               |
    ///        Both the new block plus the slot index fit here
    /// ```
    pub fn try_insert_without_optimizations(&mut self, index: u16, data: &[u8]) -> Result<(), ()> {
        let block_size = Self::total_block_size_for(data);

        // Space in the the middle of the page, not counting fragmentation.
        let available_space = self.header().last_used_block - self.end_of_slot_array_offset();

        // There's no way we can fit the block without doing anything.
        if available_space < (block_size + SLOT_SIZE) {
            return Err(());
        }

        let offset = self.header().last_used_block - block_size;

        // Add block. SAFETY: The offset has been correctly computed above. We
        // are pointing to a location that doesn't contain any block but has
        // enough space to initialize the new one.
        unsafe {
            let mut block = self.block_at_offset(offset);
            block.as_mut().size = Self::aligned_size_of(data);
            block.as_mut().left_child = 0;
            block.as_mut().overflow = 0;
            BlockHeader::content_of(block).as_mut()[..data.len()].copy_from_slice(data);
        }

        // Update header.
        self.header_mut().last_used_block = offset;
        self.header_mut().free_space -= block_size + SLOT_SIZE;

        // If the index is not the last one, shift slots to the right.
        if index < self.header().total_slots {
            let end = self.header().total_slots as usize - 1;
            self.slot_array_mut()
                .copy_within(index as usize..end, index as usize + 1);
        }

        // Add new slot.
        self.header_mut().total_slots += 1;

        self.slot_array_mut()[index as usize] = offset;

        Ok(())
    }

    /// Attempts to insert data compacting the blocks. Given this state:
    ///
    /// ```text
    ///   HEADER SLOT ARRAY     FREE SPACE                BLOCKS
    ///  +------+----+----+------------------+---------+---------+---------+
    ///  |      | O3 | O1 | ->            <- | BLOCK 3 |   DEL   | BLOCK 1 |
    ///  +------+----+----+------------------+---------+---------+---------+
    /// ```
    ///
    /// we can try to make enough room for the new block by sliding BLOCK 3 to
    /// the end:
    ///
    /// ```text
    ///   HEADER SLOT ARRAY         FREE SPACE              USED BLOCKS
    ///  +------+----+----+----------------------------+---------+---------+
    ///  |      | O3 | O1 | ->                      <- | BLOCK 3 | BLOCK 1 |
    ///  +------+----+----+----------------------------+---------+---------+
    /// ```
    ///
    /// This function is used in a chain, first we call
    /// [`Self::try_insert_without_optimizations`] and then we call this one.
    /// See [`Self::try_insert`] for details.
    pub fn try_insert_compacting_blocks(&mut self, index: u16, data: &[u8]) -> Result<(), ()> {
        self.compact_blocks();
        self.try_insert_without_optimizations(index, data)
    }

    /// Removes the given slot from this page.
    ///
    /// # Panics
    ///
    /// Panics if the `index` is out of bounds. Check the slot array length
    /// first before calling this.
    pub fn remove(&mut self, index: u16) {
        // TODO: Clean all overflow related hacks
        if let Some(p) = self.overflow.iter().position(|o| o.2 == index) {
            self.overflow.remove(p);
            return;
        } else {
            self.overflow.iter_mut().for_each(|o| o.2 -= 1);
        }

        let total_slots = self.header().total_slots;

        if index >= total_slots {
            panic!("Out of bounds: index {index} out of range for length {total_slots}");
        }

        // Remove the index as if we removed from a Vec.
        self.slot_array_mut()
            .copy_within(index as usize + 1..total_slots as usize, index as usize);

        // Add new free space SAFETY: See [`Self::block_at_slot_index`].
        self.header_mut().free_space +=
            unsafe { self.block_at_slot_index(index).as_ref().total_size() };

        // Decrease length.
        self.header_mut().total_slots -= 1;

        // Removed one slot, gained 2 extra bytes.
        self.header_mut().free_space += SLOT_SIZE;
    }

    /// Slides blocks towards the right to eliminate fragmentation. For example:
    ///
    /// ```text
    ///   HEADER SLOT ARRAY       FREE SPACE                      BLOCKS
    ///  +------+----+----+----+------------+---------+---------+---------+---------+---------+
    ///  |      | O1 | O2 | O3 | ->      <- | BLOCK 3 |   DEL   | BLOCK 2 |   DEL   | BLOCK 1 |
    ///  +------+----+----+----+------------+---------+---------+---------+---------+---------+
    /// ```
    ///
    /// turns into:
    ///
    /// ```text
    ///   HEADER   SLOT ARRAY              FREE SPACE                     USED BLOCKS
    ///  +------+----+----+----+--------------------------------+---------+---------+---------+
    ///  |      | O1 | O2 | O3 | ->                          <- | BLOCK 3 | BLOCK 2 | BLOCK 1 |
    ///  +------+----+----+----+--------------------------------+---------+---------+---------+
    /// ```
    ///
    /// # Algorithm
    ///
    /// We can eliminate fragmentation in-place by simply moving the blocks
    /// that have the largest offset first. In the figures above, we would move
    /// BLOCK 1, then BLOCK 2 and finally BLOCK 3. This makes sure that we don't
    /// write one block on top of another or we corrupt the data otherwise.
    /// Since the slots are unsorted we'll use a priority queue to find the
    /// blocks we need to move first.
    pub fn compact_blocks(&mut self) {
        let mut offsets = BinaryHeap::from_iter(
            self.slot_array()
                .iter()
                .enumerate()
                .map(|(i, offset)| (*offset, i)),
        );

        let mut destination_offset = self.size;

        while let Some((offset, i)) = offsets.pop() {
            // SAFETY: Offset is valid because we're just reading whatever we
            // stored in the slot array. The slot array doesn't contain invalid
            // offsets.
            let block = unsafe { self.block_at_offset(offset) };
            unsafe {
                destination_offset -= block.as_ref().total_size();
                block.cast::<u8>().copy_to(
                    self.header
                        .byte_add(destination_offset as usize)
                        .cast::<u8>(),
                    block.as_ref().total_size() as usize,
                );
            }
            self.slot_array_mut()[i] = destination_offset;
        }

        self.header_mut().last_used_block = destination_offset;
        self.header_mut().free_space = destination_offset - self.end_of_slot_array_offset();
    }

    /// Read only reference to the content of the given slot `index`.
    pub fn slot(&self, index: u16) -> &[u8] {
        match self.overflow.iter().position(|o| o.2 == index) {
            None => unsafe { self.slot_non_null(index).as_ref() },
            Some(p) => &self.overflow[p].1,
        }

        // SAFETY: Accessing blocks through slot indexes should always be safe,
        // that's the point of this struct.
    }

    /// Mutable reference to the content of the given slot `index`.
    pub fn slot_mut(&mut self, index: u16) -> &mut [u8] {
        // SAFETY: See [`Self::slot`].
        unsafe { self.slot_non_null(index).as_mut() }
    }

    pub fn buffer(&self) -> &[u8] {
        unsafe { self.buffer_non_null().as_ref() }
    }

    pub fn buffer_mut(&mut self) -> &mut [u8] {
        unsafe { self.buffer_non_null().as_mut() }
    }

    fn buffer_non_null(&self) -> NonNull<[u8]> {
        NonNull::slice_from_raw_parts(self.header.cast::<u8>(), self.size as usize)
    }

    /// Returns a pointer to the content of the block referenced in the given
    /// slot. This is the base of the public API, because the caller can only
    /// use slot indexes, offset values are controlled internally.
    fn slot_non_null(&self, index: u16) -> NonNull<[u8]> {
        BlockHeader::content_of(self.block_at_slot_index(index))
    }

    /// Read-only reference to the entire page header.
    pub fn header(&self) -> &PageHeader {
        // SAFETY: `self.header` points to the beginning of the page, where
        // the [`PageHeader`] struct is located.
        unsafe { self.header.as_ref() }
    }

    /// Mutable reference to the page header.
    pub fn header_mut(&mut self) -> &mut PageHeader {
        // SAFETY: Same as [`Self::header`].
        unsafe { self.header.as_mut() }
    }

    /// Pointer to the slot array.
    fn slot_array_non_null(&self) -> NonNull<[u16]> {
        // SAFETY: `self.header` is a valid pointer and `total_slots` stores the
        // correct length of the slot array.
        NonNull::slice_from_raw_parts(
            unsafe { self.header.add(1).cast() },
            self.header().total_slots as usize,
        )
    }

    // Read-only reference to the slot array.
    fn slot_array(&self) -> &[u16] {
        // SAFETY: See [`Self::slot_array_non_null`].
        unsafe { self.slot_array_non_null().as_ref() }
    }

    // Mutable reference to the slot array.
    fn slot_array_mut(&mut self) -> &mut [u16] {
        // SAFETY: See [`Self::slot_array_non_null`].
        unsafe { self.slot_array_non_null().as_mut() }
    }

    /// Returns a pointer to the block located at the given offset.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the given offset is valid and points to an
    /// initialized block or free space with enough room to initialize a new
    /// block.
    unsafe fn block_at_offset(&self, offset: u16) -> NonNull<BlockHeader> {
        self.header.byte_add(offset as usize).cast()
    }

    /// Returns an actual pointer to block referenced in the given slot `index`.
    fn block_at_slot_index(&self, index: u16) -> NonNull<BlockHeader> {
        let offset = self.slot_array()[index as usize];
        // SAFETY: The slot array always contains valid offsets.
        unsafe { self.block_at_offset(offset) }
    }

    /// Beginning of the free space.
    ///
    ///
    /// ```text
    ///   HEADER   SLOT ARRAY        FREE SPACE             USED BLOCKS
    ///  +------+----+----+----+------------------+---------+---------+---------+
    ///  |      | O1 | O2 | O3 | ->            <- | BLOCK 3 | BLOCK 2 | BLOCK 1 |
    ///  +------+----+----+----+------------------+---------+---------+---------+
    ///                        ^
    ///                        |
    ///                        +------ The returned value points here.
    /// ```
    fn end_of_slot_array_offset(&self) -> u16 {
        let total_slots = self.header().total_slots as usize;
        let offset = mem::size_of::<PageHeader>() + total_slots * mem::size_of::<u16>();

        offset as _
    }

    /// If data is not aligned to 64 bits then add padding.
    pub fn aligned_size_of(data: &[u8]) -> u16 {
        Layout::from_size_align(data.len(), mem::align_of::<BlockHeader>())
            .unwrap()
            .pad_to_align()
            .size() as _
    }

    /// Total size needed to store a valid block of at least `data.len()` size.
    fn total_block_size_for(data: &[u8]) -> u16 {
        BLOCK_HEADER_SIZE + Self::aligned_size_of(data)
    }

    pub fn total_size_needed_to_store(data: &[u8]) -> u16 {
        SLOT_SIZE + Self::total_block_size_for(data)
    }

    /// Available space in the page without counting the header.
    pub fn usable_space(page_size: u16) -> u16 {
        page_size - mem::size_of::<PageHeader>() as u16
    }

    fn layout(page_size: u16) -> Layout {
        // SAFETY:
        // - `size` is u16, it cannot overflow isize.
        // - `align_of::<BlockHeader>()` is a non-zero power of two.
        unsafe {
            Layout::from_size_align_unchecked(page_size as usize, mem::align_of::<BlockHeader>())
        }
    }
}

// TODO: Node functions. Extract this to its own type or something.
impl Page {
    pub fn binary_search_by(&self, mut f: impl FnMut(&[u8]) -> Ordering) -> Result<usize, usize> {
        self.slot_array().binary_search_by(|offset| unsafe {
            let block = self.block_at_offset(*offset);
            f(BlockHeader::content_of(block).as_ref())
        })
    }

    pub fn remove_and_return(&mut self, index: usize) -> Box<[u8]> {
        let index = index as u16;

        let owned = Vec::from(self.entry(index as usize)).into_boxed_slice();

        self.remove(index);

        owned
    }

    pub fn drain_data(&mut self) -> impl Iterator<Item = Box<[u8]>> + '_ {
        std::iter::from_fn(|| {
            if self.entries_len() == 0 {
                None
            } else {
                Some(self.remove_and_return(0))
            }
        })
    }

    pub fn entry(&self, index: usize) -> &[u8] {
        // TODO: Do something about overflow entries
        match self.overflow.iter().position(|o| o.2 == index as u16) {
            None => &self.slot(index as u16),

            Some(p) => self.overflow[p].1.as_ref(),
        }
    }

    pub fn replace_entry_at(&mut self, index: usize, entry: &[u8]) {
        if let Err(_) = self.try_replace(index as u16, entry) {
            let blk = BlockHeader {
                size: Self::aligned_size_of(entry),
                left_child: 0,
                overflow: 0,
            };
            let mut data = Vec::from(entry);
            while data.len() < Self::aligned_size_of(entry) as usize {
                data.push(0);
            }
            self.overflow.push((blk, data, index as u16));
        }
    }

    pub fn replace_and_ret(&mut self, index: usize, entry: &[u8]) -> Box<[u8]> {
        let r = Vec::from(self.entry(index)).into_boxed_slice();
        self.replace_entry_at(index, entry);
        r
    }

    pub fn insert_entry_at(&mut self, index: usize, entry: &[u8]) {
        if let Err(_) = self.try_insert_at(index as u16, entry) {
            let blk = BlockHeader {
                size: Self::aligned_size_of(entry),
                left_child: 0,
                overflow: 0,
            };
            let mut data = Vec::from(entry);
            while data.len() < Self::aligned_size_of(entry) as usize {
                data.push(0);
            }
            self.overflow.push((blk, data, index as u16));
        }
    }

    pub fn push_entry(&mut self, entry: &[u8]) {
        self.insert_entry_at(self.entries_len(), entry)
    }

    pub fn set_right_child(&mut self, child: PageNumber) {
        self.header_mut().right_child = child;
    }

    pub fn child(&self, index: usize) -> PageNumber {
        // TODO: Do something about overflow entries
        match self.overflow.iter().position(|o| o.2 == index as _) {
            None => {
                if self.header().total_slots == 0
                    || index as u16 == self.header().total_slots + self.overflow.len() as u16
                {
                    self.header().right_child
                } else {
                    unsafe { self.block_at_slot_index(index as _).as_ref().left_child }
                }
            }

            Some(p) => self.overflow[p].0.left_child,
        }
    }

    pub fn set_child(&mut self, index: usize, value: PageNumber) {
        match self.overflow.iter().position(|o| o.2 == index as _) {
            None => {
                if self.header().total_slots == 0 || index as u16 == self.header().total_slots {
                    self.header_mut().right_child = value;
                } else {
                    unsafe { self.block_at_slot_index(index as _).as_mut().left_child = value }
                }
            }

            Some(p) => self.overflow[p].0.left_child = value,
        }
    }

    pub fn children_len(&self) -> usize {
        if self.header().right_child == 0 {
            0
        } else {
            1 + self.header().total_slots as usize + self.overflow.len()
        }
    }

    pub fn entries_len(&self) -> usize {
        self.header().total_slots as usize + self.overflow.len()
    }

    pub fn children_iter(&self) -> impl Iterator<Item = PageNumber> + '_ {
        (0..self.children_len()).map(|i| self.child(i))
    }

    pub fn is_leaf(&self) -> bool {
        self.children_len() == 0
    }

    pub fn is_root(&self) -> bool {
        self.number == 0
    }

    pub fn append(&mut self, other: &mut Self) {
        let mut i = 0;
        while i < other.entries_len() {
            self.push_entry(other.slot(i as _));
            self.set_child(i, other.child(i));
            i += 1;
        }
        self.header_mut().right_child = other.header().right_child;

        while other.entries_len() > 0 {
            other.remove(0);
        }

        other.header_mut().right_child = 0;
    }

    pub fn is_overflow(&self) -> bool {
        !self.overflow.is_empty()
    }

    pub fn is_underflow(&self) -> bool {
        self.entries_len() == 0
            || !self.is_root() && self.header().free_space > Self::usable_space(self.size) / 2
    }
}

impl Drop for Page {
    fn drop(&mut self) {
        unsafe { alloc::dealloc(self.header.as_ptr().cast(), Self::layout(self.size)) }
    }
}

impl Debug for Page {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Page")
            .field("header", self.header())
            .field("number", &self.number)
            .field("size", &self.size)
            .field("slots", &self.slot_array())
            .field_with("blocks", |f| {
                let mut list = f.debug_list();
                self.slot_array().iter().for_each(|offset| {
                    let block = unsafe { self.block_at_offset(*offset as _).as_ref() };
                    list.entry_with(|f| {
                        f.debug_struct("Block")
                            .field("start", &offset)
                            .field("end", &(*offset + block.total_size()))
                            .field("size", &block.size)
                            .finish()
                    });
                });

                list.finish()
            })
            .finish()
    }
}

impl PartialEq for Page {
    fn eq(&self, other: &Self) -> bool {
        self.number == other.number && self.size == other.size && self.buffer().eq(other.buffer())
    }
}

impl Clone for Page {
    fn clone(&self) -> Self {
        let mut page = Page::new(self.number, self.size);
        page.buffer_mut().copy_from_slice(self.buffer());
        page
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_page(size: u16) -> Page {
        Page::new(0, size)
    }

    fn make_variable_size_blocks(sizes: &[usize]) -> Vec<Vec<u8>> {
        sizes
            .iter()
            .enumerate()
            .map(|(i, size)| vec![i as u8 + 1; *size])
            .collect()
    }

    fn make_fixed_size_blocks(size: usize, amount: usize) -> Vec<Vec<u8>> {
        make_variable_size_blocks(&vec![size; amount])
    }

    impl Page {
        fn insert_all(&mut self, blocks: &Vec<Vec<u8>>) {
            for block in blocks {
                self.try_insert(block).unwrap()
            }
        }
    }

    /// # Arguments
    ///
    /// * `page` - [`Page`] instance.
    ///
    /// * `blocks` - List of blocks that have been inserted in the page
    /// (in order). Necessary for checking data corruption.
    fn compare_consecutive_offsets(page: &Page, blocks: &Vec<Vec<u8>>) {
        let mut expected_offset = page.size;
        for (i, block) in blocks.iter().enumerate() {
            expected_offset -= Page::total_block_size_for(block);
            assert_eq!(page.slot_array()[i], expected_offset);
            assert_eq!(&page.slot(i as u16)[..block.len()], block);
        }
    }

    #[test]
    fn add_fixed_size_blocks() {
        let mut page = make_page(512);
        let blocks = make_fixed_size_blocks(32, 3);
        page.insert_all(&blocks);

        assert_eq!(page.header().total_slots, blocks.len() as u16);
        compare_consecutive_offsets(&page, &blocks);
    }

    #[test]
    fn add_variable_size_blocks() {
        let mut page = make_page(512);
        let blocks = make_variable_size_blocks(&[64, 32, 128]);
        page.insert_all(&blocks);

        assert_eq!(page.header().total_slots, blocks.len() as u16);
        compare_consecutive_offsets(&page, &blocks);
    }

    #[test]
    fn delete_slot() {
        let mut page = make_page(512);
        let blocks = make_fixed_size_blocks(32, 3);
        page.insert_all(&blocks);

        let expected_offsets = [page.slot_array()[0], page.slot_array()[2]];

        page.remove(1);

        assert_eq!(page.header().total_slots, 2);
        assert_eq!(page.slot_array(), expected_offsets);
    }

    #[test]
    fn compact_blocks() {
        let mut page = make_page(512);
        let mut blocks = make_variable_size_blocks(&[24, 64, 32, 128, 8]);
        page.insert_all(&blocks);

        for i in [1, 2] {
            page.remove(i as u16);
            blocks.remove(i);
        }

        page.compact_blocks();

        assert_eq!(page.header().total_slots, 3);
        compare_consecutive_offsets(&page, &blocks);
    }

    #[test]
    fn unaligned_blocks() {
        let mut page = make_page(512);
        let blocks = make_variable_size_blocks(&[7, 19, 20]);
        page.insert_all(&blocks);

        compare_consecutive_offsets(&page, &blocks);

        // Check padding
        for i in 0..blocks.len() {
            assert_eq!(
                page.slot(i as u16).len(),
                Page::aligned_size_of(&blocks[i]) as usize
            );
        }
    }

    #[test]
    fn insert_compacting_blocks() {
        let mut page = make_page(512);
        let mut blocks = make_variable_size_blocks(&[64, 32, 128]);
        page.insert_all(&blocks);

        page.remove(1);
        blocks.remove(1);

        let available_space = page.header().last_used_block - page.end_of_slot_array_offset();

        blocks.push(vec![4; available_space as usize]);
        page.try_insert(&blocks[2]).unwrap();

        compare_consecutive_offsets(&page, &blocks);
    }
}
