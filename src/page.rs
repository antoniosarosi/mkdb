//! Generic slotted page implementation. At this layer we don't care whether
//! pages are used for BTree nodes or tuple storage, we only deal with binary
//! data. This allows us to implement either index organized storage like
//! SQLite or tuple oriented storage like Postgres on top of the [`Page`]
//! struct.
//!
//! Check these lectures for some background on slotted pages and data storage:
//!
//! - [F2023 #03 - Database Storage Part 1 (CMU Intro to Database Systems)](https://www.youtube.com/watch?v=DJ5u5HrbcMk)
//! - [F2023 #04 - Database Storage Part 2 (CMU Intro to Database Systems)](https://www.youtube.com/watch?v=Ra50bFHkeM8)

use std::{alloc::Layout, collections::BinaryHeap, fmt::Debug, mem, ptr::NonNull};

/// Generic slotted page header. Located at the beginning of each page and does
/// not contain variable length data:
///
/// ```text
///                                 HEADER                                       CONTENT
/// +------------------------------------------------------------------+-----------------------+
/// | +------------+-------------+------------+---------------+------+ |                       |
/// | | free_space | total_slots | null_slots | last_used_blk | meta | |                       |
/// | +------------+-------------+------------+---------------+------+ |                       |
/// +------------------------------------------------------------------+-----------------------+
///                                         PAGE
/// ```
///
/// The alignment and size of this struct doesn't matter. Actual data is
/// appended towards to end of the page, and the page should be aligned to 8
/// bytes. See [`Page`] for more details.
#[derive(Debug)]
#[repr(C)]
struct PageHeader<H> {
    /// Amount of free bytes in this page.
    free_space: u16,
    /// Length of the slot array.
    total_slots: u16,
    /// Number of Slots that aren't pointing to anything.
    null_slots: u16,
    /// Offset of the last used block.
    last_used_block: u16,
    /// Additional page metadata defined by the user. This must not allocate
    /// anything, it should be a self contained struct.
    meta: H,
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
        self.size + mem::size_of::<Self>() as u16
    }
}

/// Decides what to do with the slot when it's removed. See [`Page`] for more
/// details.
enum RemoveSlotStrategy {
    /// Sets the slot value to 0. All of the others slots do not change. Useful
    /// for delaying BTree index updates as much as possible when using tuple
    /// oriented storage.
    SetNull,
    /// Compacts the slot array as if we removed from a [`Vec`]. In the worst
    /// case all the remaining slots change their position and all pointers to
    /// them must be updated.
    Compact,
}

/// The result of an insert operation in the page.
struct Insert {
    /// The newly created slot index.
    new_slot: u16,
    /// Changes made to previous slots to accommodate data.
    /// Format is `(previous_location, new_location)`.
    changes: Vec<(u16, u16)>,
}

/// For convenience.
type InsertResult = Result<Insert, ()>;

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
/// This is useful for 2 reasons:
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
/// for BTree pages to sort entries.
///
/// The second reason this is useful is because we can maintain the same slot
/// number for the rest of the blocks when one of them is deleted. For example,
/// if we deleted block 2, we don't have to compact the slot array like this:
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
/// instead we can just set the slot to NULL or zero:
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
/// storage (we are not doing so currently).
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
struct Page<H> {
    /// Page number on disk.
    number: u32,
    /// We need to know where the page ends.
    size: u16,
    /// Page header.
    header: NonNull<PageHeader<H>>,
}

impl<H> Page<H> {
    pub fn new(number: u32, buf: NonNull<[u8]>) -> Self {
        let size = buf.len() as u16;
        let mut header = buf.cast::<PageHeader<H>>();

        if !buf.is_aligned_to(mem::align_of::<BlockHeader>()) {
            panic!(
                "In-memory pages must me aligned to {}",
                mem::align_of::<BlockHeader>()
            );
        }

        // SAFETY: `buf` must be a valid pointer.
        unsafe {
            let header = header.as_mut();
            header.total_slots = 0;
            header.null_slots = 0;
            header.last_used_block = size;
            header.free_space = Self::usable_space(size as usize) as u16;
        }

        Self {
            number,
            size,
            header,
        }
    }

    /// Removes all null slots from the slot array. For example:
    ///
    /// ```text
    ///   HEADER         SLOT ARRAY          FREE SPACE           USED BLOCKS
    ///  +------+----+----+----+----+----+--------------+---------+---------+---------+
    ///  |      | O1 |NULL| O2 |NULL| O3 |  ->       <- | BLOCK 3 | BLOCK 2 | BLOCK 1 |
    ///  +------+----+----+----+----+----+--------------+---------+---------+---------+
    /// ```
    ///
    /// would turn into:
    ///
    /// ```text
    ///   HEADER   SLOT ARRAY        FREE SPACE             USED BLOCKS
    ///  +------+----+----+----+------------------+---------+---------+---------+
    ///  |      | O1 | O2 | O3 | ->            <- | BLOCK 3 | BLOCK 2 | BLOCK 1 |
    ///  +------+----+----+----+------------------+---------+---------+---------+
    /// ```
    ///
    /// The changes made to each slot are returned in a list of two-tuples with
    /// the format `(previous_index, new_index)`.
    pub fn compact_slots(&mut self) -> Vec<(u16, u16)> {
        let slots = self.slot_array_mut();

        // Find first null slot.
        let Some(mut dest) = slots.iter().position(|offset| *offset == 0) else {
            return vec![];
        };

        let mut changes = Vec::new();

        // We'll use two pointers, dest and source. We copy from source to dest.
        let mut source = dest + 1;

        while source < slots.len() {
            if slots[source] != 0 {
                slots[dest] = slots[source];
                changes.push((source as u16, dest as u16));
                dest += 1;
            }
            source += 1;
        }

        // NOTE: Null slots already count as free space, so this shouldn't run.
        // self.header_mut().free_space += ((slots.len() - dest) * mem::size_of::<u16>()) as u16;
        self.header_mut().total_slots = dest as u16;
        self.header_mut().null_slots = 0;

        changes
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
    /// Note that the slot array is not touched. Call [`Self::compact_slots`] if
    /// you need to optimize the slots themselves.
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
                .filter(|offset| **offset != 0)
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
            self.slot_array_mut()[i] = destination_offset as u16;
        }

        self.header_mut().last_used_block = destination_offset;
        self.header_mut().free_space = destination_offset - self.end_of_slot_array_offset();
    }

    /// Returns `true` if this page can fit `data` somehow. We don't know how,
    /// but it sure will fit.
    pub fn can_fit(&self, data: &[u8]) -> bool {
        let mut total_size = Self::total_block_size_for(data);
        if self.header().null_slots == 0 {
            total_size += mem::size_of::<u16>() as u16;
        }

        self.header().free_space >= total_size
    }

    /// Attempts to insert data without doing any work at all. This is the Best
    /// case scenario, because data will fit in the "free space":
    /// ```text
    ///   HEADER   SLOT ARRAY        FREE SPACE             USED BLOCKS
    ///  +------+----+----+----+------------------+---------+---------+---------+
    ///  |      | O3 |NULL| O1 | ->            <- | BLOCK 3 |   DEL   | BLOCK 1 |
    ///  +------+----+----+----+------------------+---------+---------+---------+
    ///                                  ^
    ///                                  |
    ///           Both the new block plus the slot index fit here
    /// ```
    ///
    /// If for some reason the new block fits in the free space but the slot
    /// pointer doesn't, we'll try to reuse a null slot. Null slot search is
    /// O(n), but it's still much faster than rearranging blocks to eliminate
    /// fragmentation.
    pub fn try_insert_without_optimizations(&mut self, data: &[u8]) -> InsertResult {
        let block_size = Self::total_block_size_for(data);
        let slot_size = mem::size_of::<u16>() as u16;

        // Space in the the middle of the page, not counting fragmentation.
        let available_space = self.header().last_used_block - self.end_of_slot_array_offset();

        let can_fit_both_slot_and_block = available_space > block_size + slot_size;

        // There's no way we can fit the block without doing anything.
        if !can_fit_both_slot_and_block && (available_space < block_size) {
            return Err(());
        }

        let offset = self.header().last_used_block - block_size;

        // Add block. SAFETY: The offset has been correctly computed above. We
        // are pointing to a location that doesn't contain any block but has
        // enough space to initialize the new one.
        unsafe {
            let mut block = self.block_at_offset(offset);
            block.as_mut().size = Self::aligned_size_of(data) as u16;
            BlockHeader::content_of(block).as_mut()[..data.len()].copy_from_slice(data);
        }

        // Update header.
        self.header_mut().last_used_block = offset;
        self.header_mut().free_space -= block_size + slot_size;

        // Find available slot or add a new one.
        let index = if can_fit_both_slot_and_block {
            let index = self.header().total_slots;
            self.header_mut().total_slots += 1;
            index
        } else {
            self.header_mut().null_slots -= 1;
            self.slot_array()
                .iter()
                .position(|value| *value == 0)
                .unwrap() as u16
        };

        self.slot_array_mut()[index as usize] = offset;

        Ok(Insert {
            new_slot: index,
            changes: vec![],
        })
    }

    /// Attempts to insert data compacting only the blocks and not the slots.
    /// Given this state:
    ///
    /// ```text
    ///   HEADER   SLOT ARRAY        FREE SPACE             USED BLOCKS
    ///  +------+----+----+----+------------------+---------+---------+---------+
    ///  |      | O3 |NULL| O1 | ->            <- | BLOCK 3 |   DEL   | BLOCK 1 |
    ///  +------+----+----+----+------------------+---------+---------+---------+
    /// ```
    ///
    /// we can try to make enough room for the new block by sliding BLOCK 3 to
    /// the end:
    ///
    /// ```text
    ///   HEADER   SLOT ARRAY        FREE SPACE             USED BLOCKS
    ///  +------+----+----+----+----------------------------+---------+---------+
    ///  |      | O3 |NULL| O1 | ->                      <- | BLOCK 3 | BLOCK 1 |
    ///  +------+----+----+----+----------------------------+---------+---------+
    /// ```
    ///
    /// This function is used in a chain, first we call
    /// [`Self::try_insert_without_optimizations`] and then we call this one.
    /// See [`Self::try_insert`] for details.
    pub fn try_insert_compacting_blocks(&mut self, data: &[u8]) -> InsertResult {
        self.compact_blocks();
        self.try_insert_without_optimizations(data)
    }

    /// Worst case scenario, we'll have to compact the slots and report the
    /// changes to the user. Last step in the insert chain, see
    /// [`Self::try_insert`] for details.
    pub fn try_insert_compacting_slots(&mut self, data: &[u8]) -> InsertResult {
        let changes = self.compact_slots();
        self.try_insert_without_optimizations(data)
            .map(|Insert { new_slot, .. }| Insert { new_slot, changes })
    }

    /// Attempts to insert the given `data` in this page with all the available
    /// methods that we have. If possible, data will be inserted without moving
    /// blocks and slots, otherwise optimizations have to be made. In case slots
    /// are moved, they are returned in [`Insert::changes`] and pointers to them
    /// in indexes or other places can be updated.
    pub fn try_insert(&mut self, data: &[u8]) -> InsertResult {
        // There's no way we can fit the given data in this page.
        if !self.can_fit(data) {
            return Err(());
        }

        self.try_insert_without_optimizations(data)
            .or_else(|_| self.try_insert_compacting_blocks(data))
            .or_else(|_| self.try_insert_compacting_slots(data))
    }

    /// Removes the given slot from this page. Caller can choose whether the
    /// rest of slots maintain their position or not.
    pub fn remove(&mut self, index: u16, strategy: RemoveSlotStrategy) -> Vec<(u16, u16)> {
        let total_slots = self.header().total_slots;

        if total_slots == 0 {
            return vec![];
        }

        if index >= total_slots {
            panic!("Out of bounds: index {index} out of range for length {total_slots}");
        }

        // Set the slots to null first. We'll check later if we need to compact.
        self.slot_array_mut()[index as usize] = 0;
        self.header_mut().null_slots += 1;
        // SAFETY: See [`Self::block_at_slot_index`].
        self.header_mut().free_space +=
            unsafe { self.block_at_slot_index(index).as_ref().total_size() };
        // Removed one slot, gained 2 bytes.
        self.header_mut().free_space += mem::size_of::<u16>() as u16;

        match strategy {
            RemoveSlotStrategy::Compact => self.compact_slots(),

            // No changes.
            _ => vec![],
        }
    }

    /// Read-only reference to header metadata.
    pub fn meta(&self) -> &H {
        &self.header().meta
    }

    /// Mutable reference to header metadata.
    pub fn meta_mut(&mut self) -> &mut H {
        &mut self.header_mut().meta
    }

    /// Read only reference to the content of the given slot `index`.
    pub fn slot(&self, index: u16) -> &[u8] {
        // SAFETY: Accessing blocks through slot indexes should always be safe,
        // that's the point of this struct.
        unsafe { self.slot_non_null(index).as_ref() }
    }

    /// Mutable reference to the content of the given slot `index`.
    pub fn slot_mut(&mut self, index: u16) -> &mut [u8] {
        // SAFETY: See [`Self::slot`].
        unsafe { self.slot_non_null(index).as_mut() }
    }

    /// Returns a pointer to the content of the block referenced in the given
    /// slot. This is the base of the public API, because the caller can only
    /// use slot indexes, offset values are controlled internally.
    fn slot_non_null(&self, index: u16) -> NonNull<[u8]> {
        BlockHeader::content_of(self.block_at_slot_index(index))
    }

    /// Read-only reference to the entire page header.
    fn header(&self) -> &PageHeader<H> {
        // SAFETY: `self.header` points to the beginning of the page, where
        // the [`PageHeader`] struct is located.
        unsafe { &self.header.as_ref() }
    }

    /// Mutable reference to the page header.
    fn header_mut(&mut self) -> &mut PageHeader<H> {
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

    /// Available space in the page without counting the header.
    fn usable_space(page_size: usize) -> usize {
        page_size - mem::size_of::<PageHeader<H>>()
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
    ///
    /// Note that null slots don't mean anything in this context. They are
    /// treated as normal used slots.
    fn end_of_slot_array_offset(&self) -> u16 {
        let total_slots = self.header().total_slots as usize;
        let offset = mem::size_of::<PageHeader<H>>() + total_slots * mem::size_of::<u16>();

        offset as _
    }

    /// If data is not aligned to 64 bits then add padding.
    fn aligned_size_of(data: &[u8]) -> u16 {
        Layout::from_size_align(data.len(), mem::align_of::<BlockHeader>())
            .unwrap()
            .pad_to_align()
            .size() as _
    }

    /// Total size needed to store a valid block of at least `data.len()` size.
    fn total_block_size_for(data: &[u8]) -> u16 {
        mem::size_of::<BlockHeader>() as u16 + Self::aligned_size_of(data)
    }
}

impl<H> Drop for Page<H> {
    fn drop(&mut self) {
        unsafe {
            std::alloc::dealloc(
                self.header.as_ptr().cast(),
                Layout::from_size_align(self.size as usize, 8).unwrap(),
            )
        }
    }
}

impl<H: Debug> Debug for Page<H> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Page")
            .field("header", unsafe { self.header.as_ref() })
            .field("number", &self.number)
            .field("size", &self.size)
            .field("slots", unsafe { &self.slot_array_non_null().as_ref() })
            .field_with("blocks", |f| {
                let mut list = f.debug_list();
                unsafe {
                    self.slot_array_non_null()
                        .as_ref()
                        .iter()
                        .for_each(|offset| {
                            let block = self.block_at_offset(*offset as _).as_ref();
                            list.entry_with(|f| {
                                f.debug_struct("Block")
                                    .field("start", &offset)
                                    .field("end", &(*offset + block.total_size()))
                                    .field("size", &block.size)
                                    .finish()
                            });
                        })
                }

                list.finish()
            })
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_page(size: usize) -> Page<()> {
        let buf = unsafe {
            NonNull::new_unchecked(std::alloc::alloc(
                Layout::from_size_align(size, 8).unwrap().pad_to_align(),
            ))
        };
        Page::new(0, NonNull::slice_from_raw_parts(buf, size))
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

    impl<H> Page<H> {
        fn insert_all(&mut self, blocks: &Vec<Vec<u8>>) -> Vec<u16> {
            let mut slots = Vec::new();
            for block in blocks {
                slots.push(self.try_insert(block).unwrap().new_slot);
            }
            slots
        }
    }

    /// # Arguments
    ///
    /// * `blocks` - List of blocks that have been inserted in the page.
    /// Necessary for checking data corruption.
    ///
    /// * `slots` - Slot index of each given block, in order.
    ///
    /// * `page` - [`Page`] instance.
    fn compare_consecutive_offsets<H>(blocks: &Vec<Vec<u8>>, slots: &Vec<u16>, page: &Page<H>) {
        let mut expected_offset = page.size;
        for (i, slot_index) in slots.iter().enumerate() {
            expected_offset -= Page::<H>::total_block_size_for(&blocks[i]);
            assert_eq!(page.slot_array()[i], expected_offset);
            assert_eq!(&page.slot(*slot_index)[..blocks[i].len()], &blocks[i]);
        }
    }

    #[test]
    fn add_fixed_size_blocks() {
        let mut page = make_page(512);
        let blocks = make_fixed_size_blocks(32, 3);
        let slots = page.insert_all(&blocks);

        assert_eq!(slots, vec![0, 1, 2]);
        compare_consecutive_offsets(&blocks, &slots, &page);
    }

    #[test]
    fn add_variable_size_blocks() {
        let mut page = make_page(512);
        let blocks = make_variable_size_blocks(&[64, 32, 128]);
        let slots = page.insert_all(&blocks);

        assert_eq!(slots, vec![0, 1, 2]);
        compare_consecutive_offsets(&blocks, &slots, &page);
    }

    #[test]
    fn delete_slot_by_compacting() {
        let mut page = make_page(512);
        let blocks = make_fixed_size_blocks(32, 3);
        page.insert_all(&blocks);

        let expected_offsets = [page.slot_array()[0], page.slot_array()[2]];

        page.remove(1, RemoveSlotStrategy::Compact);

        assert_eq!(page.header().total_slots, 2);
        assert_eq!(page.slot_array(), expected_offsets);
    }

    #[test]
    fn delete_slot_by_set_null() {
        let mut page = make_page(512);
        let blocks = make_fixed_size_blocks(32, 3);
        page.insert_all(&blocks);

        let expected_offsets = [page.slot_array()[0], 0, page.slot_array()[2]];

        page.remove(1, RemoveSlotStrategy::SetNull);

        assert_eq!(page.header().total_slots, 3);
        assert_eq!(page.slot_array()[1], 0);
        assert_eq!(page.slot_array(), expected_offsets);
    }

    #[test]
    fn compact_blocks() {
        let mut page = make_page(512);
        let mut blocks = make_variable_size_blocks(&[24, 64, 32, 128, 8]);
        page.insert_all(&blocks);

        for i in [1, 2] {
            page.remove(i as u16, RemoveSlotStrategy::Compact);
            blocks.remove(i);
        }

        page.compact_blocks();

        assert_eq!(page.header().total_slots, 3);
        compare_consecutive_offsets(&blocks, &vec![0, 1, 2], &page);
    }

    #[test]
    fn compact_slots() {
        let mut page = make_page(512);
        let blocks = make_variable_size_blocks(&[64, 32, 128, 16, 24, 8]);
        page.insert_all(&blocks);

        let offsets = page.slot_array();
        let expected = [offsets[0], offsets[2], offsets[5]];

        page.remove(1, RemoveSlotStrategy::SetNull);
        page.remove(3, RemoveSlotStrategy::SetNull);
        page.remove(4, RemoveSlotStrategy::SetNull);
        page.compact_slots();

        assert_eq!(page.header().total_slots as usize, expected.len());
        assert_eq!(page.slot_array(), expected);
    }

    #[test]
    fn unaligned_blocks() {
        let mut page = make_page(512);
        let blocks = make_variable_size_blocks(&[7, 19, 20]);
        let slots = page.insert_all(&blocks);

        compare_consecutive_offsets(&blocks, &slots, &page);

        // Check padding
        for i in 0..blocks.len() {
            assert_eq!(
                page.slot(i as u16).len(),
                Page::<()>::aligned_size_of(&blocks[i]) as usize
            );
        }
    }

    #[test]
    fn insert_compacting_blocks() {
        let mut page = make_page(512);
        let mut blocks = make_variable_size_blocks(&[64, 32, 128]);
        page.insert_all(&blocks);

        page.remove(1, RemoveSlotStrategy::Compact);
        blocks.remove(1);

        let available_space = page.header().last_used_block - page.end_of_slot_array_offset();

        blocks.push(vec![(blocks.len() + 1) as u8; available_space as usize]);
        page.try_insert(&blocks[2]).unwrap();

        compare_consecutive_offsets(&blocks, &vec![0, 1, 2], &page);
    }
}
