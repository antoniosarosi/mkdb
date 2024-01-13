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
    alloc::{self, Allocator, Layout},
    cmp::Ordering,
    collections::{BinaryHeap, VecDeque},
    fmt::Debug,
    iter, mem,
    ops::{Bound, RangeBounds},
    ptr::NonNull,
};

use crate::paging::pager::PageNumber;

/// Maximum page size is 64 KiB, which is the maximum number that we can
/// represent using 2 bytes.
pub const MAX_PAGE_SIZE: u16 = u16::MAX;

/// Size of the [`Page`] header. See [`PageHeader`] for details.
pub const PAGE_HEADER_SIZE: u16 = mem::size_of::<PageHeader>() as _;

/// Size of [`CellHeader`].
pub const CELL_HEADER_SIZE: u16 = mem::size_of::<CellHeader>() as _;

/// Size of an individual slot (offset pointer).
pub const SLOT_SIZE: u16 = mem::size_of::<u16>() as _;

/// See [`Page`] for alignment details.
pub const CELL_ALIGNMENT: usize = mem::align_of::<CellHeader>() as _;

/// Type alias to make it clear when we're dealing with cell offsets. A 2 byte
/// offset can point to the end of [`MAX_PAGE_SIZE`].
type Offset = u16;

/// The slot array can be indexed using 2 bytes, since it will never be bigger
/// than [`MAX_PAGE_SIZE`].
type SlotId = u16;

/// Slotted page header. It is located at the beginning of each page and it does
/// not contain variable length data:
///
/// ```text
///                           HEADER                                          CONTENT
/// +-------------------------------------------------------------+----------------------------+
/// | +------------+-----------+------------------+-------------+ |                            |
/// | | free_space | num_slots | last_used_offset | right_child | |                            |
/// | +------------+-----------+------------------+-------------+ |                            |
/// +-------------------------------------------------------------+----------------------------+
///                                          PAGE
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
    num_slots: u16,
    /// Offset of the last cell.
    last_used_offset: Offset,
    /// Last child of this page.
    pub right_child: PageNumber,
}

impl PageHeader {
    /// Total free space that can be used to store [`Cell`] instances.
    pub fn free_space(&self) -> u16 {
        self.free_space
    }
}

/// Located at the beginning of each cell. The header stores the size of the
/// cell without including its own size and it also stores a pointer to the
/// BTree page the contains entries "less than" this one.
///
/// ```text
///           HEADER                               CONTENT
/// +-----------------------+-----------------------------------------------+
/// | +------+------------+ |                                               |
/// | | size | left_child | |                                               |
/// | +------+------------+ |                                               |
/// +-----------------------+-----------------------------------------------+
///                         ^                                               ^
///                         |                                               |
///                         +-----------------------------------------------+
///                                            size bytes
/// ```
///
/// # Alignment
///
/// cells are 64 bit aligned. See [`Page`] for more details.
#[derive(Debug, PartialEq, Clone)]
#[repr(C, align(8))]
pub struct CellHeader {
    /// Size of the cell content.
    size: u16,
    /// ID of the BTree page that contains values less than this cell.
    pub left_child: PageNumber,
}

impl CellHeader {
    fn total_size(&self) -> u16 {
        CELL_HEADER_SIZE + self.size
    }

    pub fn storage_size(&self) -> u16 {
        self.total_size() + SLOT_SIZE
    }

    fn content_of(cell: NonNull<Self>) -> NonNull<[u8]> {
        unsafe { NonNull::slice_from_raw_parts(cell.add(1).cast(), cell.as_ref().size as _) }
    }
}

/// Owned version of a cell. The [`crate::storage::BTree`] structure reorders
/// cells around different sibling pages when an overflow or underflow occurs,
/// so instead of hiding the low level details we provide some API that can be
/// used by upper levels.
#[derive(Debug, PartialEq, Clone)]
pub struct Cell {
    pub header: CellHeader,
    pub content: Box<[u8]>,
}

/// Read-only reference to a cell. The cell could be located either in a page
/// that comes from disk or in memory if the page overflowed. See [`MemPage`]
/// for more details.
pub struct CellRef<'a> {
    pub header: &'a CellHeader,
    pub content: &'a [u8],
}

impl<'a> CellRef<'a> {
    pub fn size(&self) -> u16 {
        self.header.size
    }

    pub fn total_size(&self) -> u16 {
        self.header.total_size()
    }

    pub fn storage_size(&self) -> u16 {
        self.header.storage_size()
    }
}

/// Same as [`CellRef`] but mutable.
pub struct CellRefMut<'a> {
    pub header: &'a mut CellHeader,
    pub content: &'a mut [u8],
}

impl Cell {
    pub fn new(data: &[u8]) -> Self {
        let mut data = Vec::from(data);
        let size = Self::aligned_size_of(&data);
        data.resize(size as _, 0);

        Self {
            header: CellHeader {
                size,
                left_child: 0,
            },
            content: data.into_boxed_slice(),
        }
    }

    pub fn size(&self) -> u16 {
        self.header.size
    }

    pub fn total_size(&self) -> u16 {
        self.header.total_size()
    }

    pub fn storage_size(&self) -> u16 {
        self.header.storage_size()
    }

    /// See [`Page`] for details.
    pub fn aligned_size_of(data: &[u8]) -> u16 {
        Layout::from_size_align(data.len(), CELL_ALIGNMENT)
            .unwrap()
            .pad_to_align()
            .size() as _
    }
}

struct OverflowCell {
    cell: Cell,
    index: SlotId,
}

/// Fixed size slotted page. This is what we store on disk. The page maintains a
/// "slot array" located after the header that grows towards the right. On the
/// opposite side is where used cells are located, leaving free space in the
/// middle of the page. Each item in the slot array points to one of the used
/// cells through its offset, calculated from the start of the page (before the
/// header).
///
/// ```text
///   HEADER   SLOT ARRAY       FREE SPACE             USED CELLS
///  +------+----+----+----+------------------+--------+--------+--------+
///  |      | O1 | O2 | O3 | ->            <- | CELL 3 | CELL 2 | CELL 1 |
///  +------+----+----+----+------------------+--------+--------+--------+
///           |    |    |                     ^        ^        ^
///           |    |    |                     |        |        |
///           +----|----|---------------------|--------|--------+
///                |    |                     |        |
///                +----|---------------------|--------+
///                     |                     |
///                     +---------------------+
/// ```
///
/// This is useful for 2 reasons, described below.
///
/// # Cheap Sorting or Reordering
///
/// First, we can rearrange cells without changing their physical location. We
/// only need to change the offsets in the slot array. For example, if we wanted
/// to reverse the order of the cells in the figure above, this is all we need
/// to do:
///
/// ```text
///   HEADER   SLOT ARRAY       FREE SPACE             USED CELLS
///  +------+----+----+----+------------------+--------+--------+--------+
///  |      | O3 | O2 | O1 | ->            <- | CELL 3 | CELL 2 | CELL 1 |
///  +------+----+----+----+------------------+--------+--------+--------+
///           |    |    |                     ^        ^        ^
///           |    |    |                     |        |        |
///           +----|----|---------------------+        |        |
///                |    |                              |        |
///                +----|-------------------------------        |
///                     |                                       |
///                     +---------------------------------------+
/// ```
///
/// Now the third cell becomes the first one in the list, second cell stays
/// the same and the first cell becomes the last. Changing offsets in the slot
/// array is much cheaper than moving the cells around the page. This is needed
/// for BTree pages to sort entries, and this is the only feature implemented
/// here.
///
/// # Maintaining Slot Positions After Delete Operations
///
/// The second reason this is useful is because we could maintain the same slot
/// number for the rest of the cells when one of them is deleted. For example,
/// if we deleted CELL 2 in the figure above, we wouldn't have to compact the
/// slot array like this:
///
/// ```text
///   HEADER SLOT ARRAY       FREE SPACE              USED CELLS
///  +------+----+----+----------------------+--------+--------+--------+
///  |      | O3 | O1 | ->                <- | CELL 3 |   DEL  | CELL 1 |
///  +------+----+----+----------------------+--------+--------+--------+
///           |    |                         ^                 ^
///           |    |                         |                 |
///           +----|-------------------------+                 |
///                |                                           |
///                +-------------------------------------------+
/// ```
///
/// If we did this, then the slot ID of CELL 1 would change from 2 to 1. But
/// instead we could simply set the slot to NULL or zero:
///
/// ```text
///   HEADER SLOT ARRAY       FREE SPACE              USED CELLS
///  +------+----+----+----+------------------+--------+--------+--------+
///  |      | O3 |NULL| O1 | ->            <- | CELL 3 |   DEL  | CELL 1 |
///  +------+----+----+----+------------------+--------+--------+--------+
///           |         |                     ^                 ^
///           |         |                     |                 |
///           +---------|---------------------+                 |
///                     |                                       |
///                     +---------------------------------------+
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
/// In-memory pages should be 64 bit aligned. This will allow the [`CellHeader`]
/// instances to be 64 bit aligned as well, which in turn will make the first
/// content byte of each cell 64 bit aligned. Further alignment within the
/// content should be controlled on upper layers. For example, if we have some
/// schema like `[Int, Varchar(255), Bool]`, we can't worry about individual
/// column alignment here because we're only storing binary data, we know
/// nothing about what's inside.
///
/// # Overflow
///
/// To deal with overflow we maintain a list of cells that didn't fit in the
/// slotted page and we store the index where they should have been inserted
/// in the slotted array. Not all methods take overflow into account, most of
/// them don't care because once the page overflows the BTree balancing
/// algorithm will move all the cells out of the page and reorganize them across
/// siblings. The only method that needs to work correctly is [`Page::drain`] as
/// that's the one used to move the cells out.
pub(crate) struct Page {
    /// Page number on disk.
    pub number: PageNumber,
    /// Fixed size in-memory buffer that contains the data read from disk.
    buffer: NonNull<[u8]>,
    /// Overflow list.
    overflow: VecDeque<OverflowCell>,
}

impl Page {
    pub fn new(number: PageNumber, size: u16) -> Self {
        let buffer = alloc::Global
            .allocate_zeroed(Self::layout(size))
            .expect("could not allocate page");

        unsafe {
            buffer.cast().write(PageHeader {
                num_slots: 0,
                last_used_offset: size,
                free_space: Self::usable_space(size),
                right_child: 0,
            });
        }

        Self {
            number,
            buffer,
            overflow: VecDeque::new(),
        }
    }

    pub fn size(&self) -> u16 {
        self.buffer.len() as _
    }

    pub fn len(&self) -> u16 {
        self.header().num_slots + self.overflow.len() as u16
    }

    pub fn buffer(&self) -> &[u8] {
        unsafe { self.buffer.as_ref() }
    }

    pub fn buffer_mut(&mut self) -> &mut [u8] {
        unsafe { self.buffer.as_mut() }
    }

    pub fn header(&self) -> &PageHeader {
        unsafe { self.buffer.cast().as_ref() }
    }

    /// Mutable reference to the page header.
    pub fn header_mut(&mut self) -> &mut PageHeader {
        unsafe { self.buffer.cast().as_mut() }
    }

    fn slot_array_non_null(&self) -> NonNull<[u16]> {
        unsafe {
            NonNull::slice_from_raw_parts(
                self.buffer.byte_add(PAGE_HEADER_SIZE as _).cast(),
                self.header().num_slots as _,
            )
        }
    }

    fn slot_array(&self) -> &[u16] {
        unsafe { self.slot_array_non_null().as_ref() }
    }

    fn slot_array_mut(&mut self) -> &mut [u16] {
        unsafe { self.slot_array_non_null().as_mut() }
    }

    unsafe fn cell_header_at_offset(&self, offset: Offset) -> NonNull<CellHeader> {
        self.buffer.byte_add(offset as _).cast()
    }

    fn cell_header_at_slot_index(&self, index: SlotId) -> NonNull<CellHeader> {
        unsafe { self.cell_header_at_offset(self.slot_array()[index as usize]) }
    }

    pub fn usable_space(page_size: u16) -> u16 {
        page_size - PAGE_HEADER_SIZE
    }

    pub fn layout(page_size: u16) -> Layout {
        unsafe { Layout::from_size_align_unchecked(page_size as usize, CELL_ALIGNMENT) }
    }

    pub fn cell(&self, index: SlotId) -> CellRef {
        let header = self.cell_header_at_slot_index(index);
        unsafe {
            CellRef {
                header: header.as_ref(),
                content: CellHeader::content_of(header).as_ref(),
            }
        }
    }

    pub fn cell_mut(&mut self, index: SlotId) -> CellRefMut {
        let mut header = self.cell_header_at_slot_index(index);
        unsafe {
            CellRefMut {
                header: header.as_mut(),
                content: CellHeader::content_of(header).as_mut(),
            }
        }
    }

    pub fn owned_cell(&mut self, index: SlotId) -> Cell {
        let header = self.cell_header_at_slot_index(index);
        unsafe {
            Cell {
                header: header.read(),
                content: Vec::from(CellHeader::content_of(header).as_ref()).into(),
            }
        }
    }

    pub fn child(&self, index: SlotId) -> PageNumber {
        if index == self.len() {
            self.header().right_child
        } else {
            self.cell(index).header.left_child
        }
    }

    pub fn is_underflow(&self) -> bool {
        self.len() == 0
            || !self.is_root() && self.header().free_space > Self::usable_space(self.size()) / 2
    }

    pub fn iter_children(&self) -> impl Iterator<Item = PageNumber> + '_ {
        let len = if self.is_leaf() { 0 } else { self.len() + 1 };

        (0..len).map(|i| self.child(i))
    }

    pub fn append(&mut self, other: &mut Self) {
        for cell in other.drain(..) {
            self.push(cell);
        }

        self.header_mut().right_child = other.header().right_child;
    }

    pub fn binary_search_by(&self, mut f: impl FnMut(&[u8]) -> Ordering) -> Result<SlotId, SlotId> {
        self.slot_array()
            .binary_search_by(|offset| unsafe {
                let cell = self.cell_header_at_offset(*offset);
                f(CellHeader::content_of(cell).as_ref())
            })
            .map(|index| index as _)
            .map_err(|index| index as _)
    }

    pub fn is_overflow(&self) -> bool {
        !self.overflow.is_empty()
    }

    pub fn is_leaf(&self) -> bool {
        self.header().right_child == 0
    }

    pub fn is_root(&self) -> bool {
        self.number == 0
    }

    pub fn push(&mut self, cell: Cell) {
        self.insert(self.len(), cell);
    }

    pub fn insert(&mut self, index: SlotId, cell: Cell) {
        if self.is_overflow() {
            return self.overflow.push_back(OverflowCell { cell, index });
        }

        if let Err(cell) = self.try_insert(index, cell) {
            self.overflow.push_back(OverflowCell { cell, index });
        }
    }

    pub fn replace(&mut self, index: SlotId, new_cell: Cell) -> Cell {
        match self.try_replace(index, new_cell) {
            Ok(old_cell) => old_cell,

            Err(new_cell) => {
                self.overflow.push_back(OverflowCell {
                    cell: new_cell,
                    index,
                });
                self.remove(index)
            }
        }
    }

    /// Attempts to insert the given `cell` in this page. There are two possible
    /// cases:
    ///
    /// - Case 1: the cell fits in the "free space" between the slot array and
    /// the closest cell. This is the best case scenario since we can just write
    /// the new cell without doing anything else.
    ///
    /// ```text
    ///   HEADER SLOT ARRAY       FREE SPACE              USED CELLS
    ///  +------+----+----+----------------------+--------+--------+--------+
    ///  |      | O3 | O1 | ->                <- | CELL 3 |   DEL  | CELL 1 |
    ///  +------+----+----+----------------------+--------+--------+--------+
    ///                               ^
    ///                               |
    ///           Both the new cell plus the slot fit here
    /// ```
    ///
    /// - Case 2: the cell fits in this page but we have to defragment first.
    /// This is the worst case scenario, but at least it does some garbage
    /// collection. See [`Self::defragment`] for details.
    ///
    /// ```text
    ///   HEADER SLOT ARRAY       FREE SPACE              USED CELLS
    ///  +------+----+----+----------------------+--------+--------+--------+
    ///  |      | O3 | O1 | ->                <- | CELL 3 |   DEL  | CELL 1 |
    ///  +------+----+----+----------------------+--------+--------+--------+
    ///                               ^                        ^
    ///                               |                        |
    ///                               +------------+-----------+
    ///                                            |
    ///                            We can fit the cell plus the slot
    ///                            if we join all the free space into
    ///                            one contiguous block
    /// ```
    ///
    /// There is no free list, we don't search for deleted blocks that can fit
    /// the new cell.
    pub fn try_insert(&mut self, index: SlotId, cell: Cell) -> Result<SlotId, Cell> {
        let total_size = cell.storage_size();

        // There's no way we can fit the cell in this page.
        if self.header().free_space < total_size {
            return Err(cell);
        }

        // Space between the end of the slot array and the closest cell.
        let available_space = {
            let end = self.header().last_used_offset;
            let start = PAGE_HEADER_SIZE + self.header().num_slots * SLOT_SIZE;

            end - start
        };

        // We can fit the new cell but we have to defragment the page first.
        if available_space < total_size {
            self.defragment();
        }

        let offset = self.header().last_used_offset - cell.total_size();

        // Write new cell.
        // SAFETY: `last_used_offset` keeps track of where the last cell was
        // written. By substracting the total size of the new cell to
        // `last_used_offset` we get a valid pointer within the page where we
        // write the new cell.
        unsafe {
            let header = self.cell_header_at_offset(offset);
            header.write(cell.header);

            CellHeader::content_of(header)
                .as_mut()
                .copy_from_slice(&cell.content);
        }

        // Update header.
        self.header_mut().last_used_offset = offset;
        self.header_mut().free_space -= total_size;

        // If the index is not the last one, shift slots to the right.
        if index < self.header().num_slots {
            let end = self.header().num_slots as usize - 1;
            self.slot_array_mut()
                .copy_within(index as usize..end, index as usize + 1);
        }

        // Add new slot.
        self.header_mut().num_slots += 1;

        // Set offset.
        self.slot_array_mut()[index as usize] = offset;

        Ok(index)
    }

    /// Tries to replace the cell pointed by the given slot `index` with the
    /// `new_cell`. Similar to [`Self::try_insert`] there are 2 main cases:
    ///
    /// - Case 1: The new cell fits in the same place as the old cell:
    ///
    /// ```text
    ///                               The size of the new cell is less or
    ///                                  equal to that of the old cell
    ///
    ///                                         +----------+
    ///                                         | NEW CELL |
    ///                                         +----------+
    ///                                               |
    ///                                               v
    ///  +------+----+----+------------------+----------------+--------+--------+
    ///  |      | O3 | O1 | ->            <- |     CELL 3     |   DEL  | CELL 1 |
    ///  +------+----+----+------------------+----------------+--------+--------+
    /// ```
    ///
    /// - Case 2: The new cell does not fit in the same place, but it does fit
    /// in the page if we remove the old cell.
    ///     - Case A: The new cell fits in the "free space".
    ///     - Case B: We have to defragment the page to fit the new cell.
    ///
    /// ```text
    ///                                                    The size of the new cell is greater
    ///                                                         than that of the old cell
    ///
    ///                                                          +--------------------+
    ///                                                          |      NEW CELL      |
    ///                Case A: new cell fits here                +--------------------+
    ///                             |                                       |
    ///                             v                                       v
    ///  +------+----+----+------------------+----------------+--------+--------+
    ///  |      | O3 | O1 | ->            <- |      CELL 3    |   DEL  | CELL 1 |
    ///  +------+----+----+------------------+----------------+--------+--------+
    ///                             ^                              ^        ^
    ///                             |                              |        |
    ///                             +------------------------------+--------+
    ///                                                  |
    ///                                   Case B: new cell fits in the page
    ///                                   using all the available free space
    ///                                   including the deleted cell
    /// ```
    pub fn try_replace(&mut self, index: SlotId, new_cell: Cell) -> Result<Cell, Cell> {
        let old_cell = self.cell(index);

        // There's no way we can fit the new cell in this page, even if we
        // remove the one that has to be replaced.
        if self.header().free_space + old_cell.header.total_size() < new_cell.total_size() {
            return Err(new_cell);
        }

        // Case 1: The new cell is smaller than the old cell. This is the best
        // case scenario because we can simply overwrite the contents without
        // doing much else.
        if old_cell.header.size <= new_cell.header.size {
            // If new_cell is smaller we gain some extra bytes.
            let free_bytes = old_cell.header.size - new_cell.header.size;

            // Copy the old cell to return it.
            let owned_cell = self.owned_cell(index);

            // Overwrite the contents of the old cell.
            let old_cell = self.cell_mut(index);
            old_cell.content[..new_cell.content.len()].copy_from_slice(&new_cell.content);
            *old_cell.header = new_cell.header;

            self.header_mut().free_space += free_bytes;

            return Ok(owned_cell);
        }

        // Case 2: The new cell fits in this page but we have to remove the old
        // one and potentially defragment the page. Worst case scenario.
        let old = self.remove(index);

        self.try_insert(index, new_cell)
            .expect("we made room for the new cell, it should fit in the page");

        Ok(old)
    }

    /// Removes the cell pointed by the given slot `index`. Unlike
    /// [`Self::try_insert`] and [`Self::try_replace`], this function cannot
    /// fail. However, it does panic if the given `index` is out of bounds.
    pub fn remove(&mut self, index: SlotId) -> Cell {
        let len = self.header().num_slots;

        assert!(index < len, "index {index} out of range for length {len}");

        let cell = self.owned_cell(index);

        // Remove the index as if we removed from a Vec.
        self.slot_array_mut()
            .copy_within(index as usize + 1..len as usize, index as usize);

        // Add new free space.
        self.header_mut().free_space += cell.total_size();

        // Decrease length.
        self.header_mut().num_slots -= 1;

        // Removed one slot, gained 2 extra bytes.
        self.header_mut().free_space += SLOT_SIZE;

        cell
    }

    /// Slides cells towards the right to eliminate fragmentation. For example:
    ///
    /// ```text
    ///   HEADER   SLOT ARRAY    FREE SPACE                      CELLS
    ///  +------+----+----+----+------------+--------+---------+--------+---------+--------+
    ///  |      | O1 | O2 | O3 | ->      <- | CELL 3 |   DEL   | CELL 2 |   DEL   | CELL 1 |
    ///  +------+----+----+----+------------+--------+---------+--------+---------+--------+
    /// ```
    ///
    /// turns into:
    ///
    /// ```text
    ///   HEADER   SLOT ARRAY              FREE SPACE                       CELLS
    ///  +------+----+----+----+--------------------------------+--------+--------+--------+
    ///  |      | O1 | O2 | O3 | ->                          <- | CELL 3 | CELL 2 | CELL 1 |
    ///  +------+----+----+----+--------------------------------+--------+--------+--------+
    /// ```
    ///
    /// Note that we the total amount of "free space" does not change. Instead,
    /// all the free space is now contiguous, but we do not gain any bytes from
    /// this.
    ///
    /// # Algorithm
    ///
    /// We can eliminate fragmentation in-place (without copying the page) by
    /// simply moving the cells that have the largest offset first. In the
    /// figures above, we would move CELL 1, then CELL 2 and finally CELL 3.
    /// This makes sure that we don't write one cell on top of another or we
    /// corrupt the data otherwise.
    fn defragment(&mut self) {
        let mut offsets = BinaryHeap::from_iter(
            self.slot_array()
                .iter()
                .enumerate()
                .map(|(i, offset)| (*offset, i)),
        );

        let mut destination_offset = self.size();

        while let Some((offset, i)) = offsets.pop() {
            unsafe {
                let cell = self.cell_header_at_offset(offset);
                let size = cell.as_ref().total_size();

                destination_offset -= size;

                cell.cast::<u8>().copy_to(
                    self.buffer
                        .byte_add(destination_offset as usize)
                        .cast::<u8>(),
                    size as usize,
                );
            }
            self.slot_array_mut()[i] = destination_offset;
        }

        self.header_mut().last_used_offset = destination_offset;
    }

    pub fn drain(&mut self, range: impl RangeBounds<usize>) -> impl Iterator<Item = Cell> + '_ {
        let start = match range.start_bound() {
            Bound::Unbounded => 0,
            Bound::Excluded(i) => i + 1,
            Bound::Included(i) => *i,
        };

        let end = match range.end_bound() {
            Bound::Unbounded => self.len() as usize,
            Bound::Excluded(i) => *i,
            Bound::Included(i) => i + 1,
        };

        let mut drain_index = start;
        let mut slot_index = start;

        self.overflow
            .make_contiguous()
            .sort_by_key(|overflow| overflow.index);

        iter::from_fn(move || {
            // Copy cells until we reach the end.
            if drain_index < end {
                let cell = if self
                    .overflow
                    .front()
                    .is_some_and(|overflow| overflow.index as usize == drain_index)
                {
                    self.overflow.pop_front().unwrap().cell
                } else {
                    let cell = self.owned_cell(slot_index as _);
                    slot_index += 1;
                    cell
                };

                drain_index += 1;

                Some(cell)
            } else {
                // Now compute gained space and shift slots towards the left.
                self.header_mut().free_space += (start..slot_index)
                    .map(|slot| self.cell(slot as _).storage_size())
                    .sum::<u16>();

                self.slot_array_mut().copy_within(start..slot_index, 0);

                self.header_mut().num_slots -= (slot_index - start) as u16;

                None
            }
        })
    }
}

impl Drop for Page {
    fn drop(&mut self) {
        unsafe {
            alloc::Global.deallocate(self.buffer.cast(), Self::layout(self.buffer.len() as _))
        }
    }
}

impl PartialEq for Page {
    fn eq(&self, other: &Self) -> bool {
        self.number == other.number
            && self.buffer.len() == other.buffer.len()
            && self.buffer().eq(other.buffer())
    }
}

impl Clone for Page {
    fn clone(&self) -> Self {
        let mut page = Page::new(self.number, self.buffer.len() as _);
        page.buffer_mut().copy_from_slice(self.buffer());
        page
    }
}

impl Debug for Page {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Page")
            .field("header", self.header())
            .field("number", &self.number)
            .field("size", &self.size())
            .field("slots", &self.slot_array())
            .field_with("cells", |f| {
                let mut list = f.debug_list();
                (0..self.len()).for_each(|slot| {
                    let cell = self.cell(slot);
                    let offset = self.slot_array()[slot as usize];
                    list.entry_with(|f| {
                        f.debug_struct("Cell")
                            .field("header", &cell.header)
                            .field("start", &offset)
                            .field("end", &(offset + cell.total_size()))
                            .field("size", &cell.size())
                            .finish()
                    });
                });

                list.finish()
            })
            .finish()
    }
}