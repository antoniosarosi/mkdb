use std::{
    alloc::{self, Allocator, Layout},
    cmp::Ordering,
    collections::BinaryHeap,
    mem,
    ops::Index,
    ptr::NonNull,
};

use super::pager::PageNumber;

pub const PAGE_HEADER_SIZE: u16 = mem::size_of::<PageHeader>() as _;
pub const CELL_HEADER_SIZE: u16 = mem::size_of::<CellHeader>() as _;
pub const SLOT_SIZE: u16 = mem::size_of::<u16>() as _;

const CELL_ALIGNMENT: usize = mem::align_of::<CellHeader>() as _;

type Offset = u16;
type SlotId = u16;

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

#[derive(Debug)]
#[repr(C, align(8))]
pub struct CellHeader {
    size: u16,
    ///
    pub left_child: PageNumber,
}

impl CellHeader {
    fn total_size(&self) -> u16 {
        CELL_HEADER_SIZE + self.size
    }

    fn content_of(cell: NonNull<Self>) -> NonNull<[u8]> {
        unsafe { NonNull::slice_from_raw_parts(cell.add(1).cast(), cell.as_ref().size as _) }
    }
}

#[derive(Debug)]
pub struct Cell {
    pub header: CellHeader,
    pub content: Box<[u8]>,
}

pub struct CellRef<'a> {
    pub header: &'a CellHeader,
    pub content: &'a [u8],
}

pub struct CellRefMut<'a> {
    pub header: &'a mut CellHeader,
    pub content: &'a mut [u8],
}

impl Cell {
    pub fn new(mut data: &[u8]) -> Self {
        let mut data = Vec::from(data);
        let size = Page::aligned_size_of(&data);
        data.resize(size as _, 0);

        Self {
            header: CellHeader {
                size,
                left_child: 0,
            },
            content: data.into_boxed_slice(),
        }
    }

    pub fn total_size(&self) -> u16 {
        self.header.total_size()
    }

    pub fn size(&self) -> u16 {
        self.header.size
    }

    pub fn storage_size(&self) -> u16 {
        self.total_size() + SLOT_SIZE
    }
}

#[derive(Debug)]
enum CellLocation {
    Disk(u16),
    Mem(Cell),
}

#[derive(Debug)]
pub(crate) struct Page {
    pub number: PageNumber,
    buffer: NonNull<[u8]>,
    cells: Vec<CellLocation>,
    mem_cells: usize,
}

impl Page {
    /// Allocates a new empty [`Page`] of the given `size`.
    pub fn new(number: PageNumber, size: u16) -> Self {
        // TODO: Handle alloc error.
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
            mem_cells: 0,
            cells: Vec::new(),
        }
    }

    pub fn init(&mut self) {
        if self.cells.is_empty() {
            for slot in 0..self.header().num_slots {
                self.cells.push(CellLocation::Disk(slot));
            }
        }
    }

    pub fn cell(&self, index: u16) -> CellRef {
        match &self.cells[index as usize] {
            CellLocation::Mem(cell) => CellRef {
                header: &cell.header,
                content: &cell.content,
            },

            CellLocation::Disk(slot) => unsafe {
                let offset = self.slots()[*slot as usize];

                let header = self.cell_at_offset(offset);
                CellRef {
                    header: header.as_ref(),
                    content: CellHeader::content_of(header).as_ref(),
                }
            },
        }
    }

    pub fn cell_mut(&mut self, index: u16) -> CellRefMut {
        match self.cells[index as usize] {
            CellLocation::Mem(ref mut cell) => CellRefMut {
                header: &mut cell.header,
                content: &mut cell.content,
            },

            CellLocation::Disk(slot) => unsafe {
                let offset = self
                    .buffer
                    .byte_add((PAGE_HEADER_SIZE + SLOT_SIZE * slot) as usize)
                    .cast::<u16>()
                    .read();

                let mut header = self.buffer.byte_add(offset as usize).cast::<CellHeader>();

                CellRefMut {
                    header: header.as_mut(),
                    content: CellHeader::content_of(header).as_mut(),
                }
            },
        }
    }

    pub fn size(&self) -> u16 {
        self.buffer.len() as _
    }

    pub fn is_overflow(&self) -> bool {
        self.mem_cells > 0
    }

    pub fn len(&self) -> u16 {
        self.cells.len() as _
    }

    pub fn push(&mut self, cell: Cell) {
        self.insert(self.len(), cell)
    }

    pub fn append(&mut self, other: &mut Self) {
        for cell in other.drain() {
            self.push(cell);
        }

        self.header_mut().right_child = other.header().right_child;
    }

    pub fn drain(&mut self) -> impl Iterator<Item = Cell> + '_ {
        self.init();
        // TODO: Optimize this, there's no need to remove anything until the
        // end.
        std::iter::from_fn(|| {
            if self.cells.is_empty() {
                return None;
            } else {
                Some(self.remove(0))
            }
        })
    }

    pub fn insert(&mut self, index: u16, cell: Cell) {
        match self.try_insert(index, cell) {
            Ok(slot) => self.cells.insert(index as _, CellLocation::Disk(slot)),

            Err(cell) => {
                self.cells.insert(index as _, CellLocation::Mem(cell));
                self.mem_cells += 1;
            }
        }
    }

    pub fn remove(&mut self, index: u16) -> Cell {
        let location = self.cells.remove(index as _);

        // Cell is not written in the page, not much to do here.
        if let CellLocation::Mem(cell) = location {
            self.mem_cells -= 1;
            return cell;
        }

        // Grab the slot id and delete the cell from the page.
        let slot = match location {
            CellLocation::Disk(slot) => slot,
            _ => unreachable!(),
        };

        let len = self.header().num_slots;

        if slot >= len {
            panic!("index {slot} out of range for length {len}");
        }

        let cell = {
            unsafe {
                let offset = self.slots()[slot as usize];

                let header = self.cell_at_offset(offset);
                Cell {
                    header: header.read(),
                    content: Vec::from(CellHeader::content_of(header).as_ref()).into(),
                }
            }
        };

        // Remove the index as if we removed from a Vec.
        self.slots_mut()
            .copy_within(slot as usize + 1..len as usize, slot as usize);

        self.cells[index as _..].iter_mut().for_each(|location| {
            if let CellLocation::Disk(slot) = location {
                *slot -= 1
            }
        });

        // Add new free space SAFETY: See [`Self::block_at_slot_index`].
        self.header_mut().free_space += cell.total_size();

        // Decrease length.
        self.header_mut().num_slots -= 1;

        // Removed one slot, gained 2 extra bytes.
        self.header_mut().free_space += SLOT_SIZE;

        cell
    }

    pub fn is_leaf(&self) -> bool {
        self.header().right_child == 0
    }

    pub fn replace(&mut self, index: u16, cell: Cell) -> Cell {
        match self.try_replace(index, cell) {
            Ok(old) => old,

            Err(cell) => {
                let old = self.remove(index);
                self.insert(index, cell);
                old
            }
        }
    }

    pub fn binary_search_by(&self, mut f: impl FnMut(&[u8]) -> Ordering) -> Result<u16, u16> {
        self.slots()
            .binary_search_by(|offset| unsafe {
                let cell = self.cell_at_offset(*offset);
                f(CellHeader::content_of(cell).as_ref())
            })
            .map(|index| index as _)
            .map_err(|index| index as _)
    }

    pub fn child(&self, index: u16) -> PageNumber {
        if index == self.len() {
            self.header().right_child
        } else {
            self.cell(index).header.left_child
        }
    }

    fn try_insert(&mut self, index: u16, cell: Cell) -> Result<u16, Cell> {
        let total_size = SLOT_SIZE + cell.total_size();

        // There's no way we can fit the block in the page without overflowing.
        if self.header().free_space < total_size {
            return Err(cell);
        }

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
        unsafe {
            let header = self.cell_at_offset(offset);
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
            self.slots_mut()
                .copy_within(index as usize..end, index as usize + 1);

            self.cells[index as _..].iter_mut().for_each(|location| {
                if let CellLocation::Disk(slot) = location {
                    *slot += 1
                }
            });
        }

        // Add new slot.
        self.header_mut().num_slots += 1;

        self.slots_mut()[index as usize] = offset;

        Ok(index)
    }

    fn try_replace(&mut self, index: u16, new_cell: Cell) -> Result<Cell, Cell> {
        let current_cell = self.cell(index);

        if current_cell.header.size <= new_cell.header.size {
            let free_bytes = current_cell.header.size - new_cell.header.size;

            let old = Cell {
                header: CellHeader {
                    size: current_cell.header.size,
                    left_child: current_cell.header.left_child,
                },
                content: Vec::from(current_cell.content.as_ref()).into_boxed_slice(),
            };

            let current_cell = self.cell_mut(index);

            current_cell.content[..new_cell.content.len()].copy_from_slice(&new_cell.content);
            current_cell.header.size = new_cell.header.size;

            self.header_mut().free_space += free_bytes;
            Ok(old)
        } else if self.header().free_space + current_cell.header.total_size()
            >= new_cell.total_size()
        {
            let old = self.remove(index);

            self.try_insert(index, new_cell)
                .expect("we made room for the new cell");

            Ok(old)
        } else {
            Err(new_cell)
        }
    }

    fn defragment(&mut self) {
        let mut offsets = BinaryHeap::from_iter(
            self.slots()
                .iter()
                .enumerate()
                .map(|(i, offset)| (*offset, i)),
        );

        let mut destination_offset = self.size();

        while let Some((offset, i)) = offsets.pop() {
            let cell = unsafe { self.cell_at_offset(offset) };
            unsafe {
                let cell_size = cell.as_ref().total_size();
                destination_offset -= cell_size;
                cell.cast::<u8>().copy_to(
                    self.buffer
                        .byte_add(destination_offset as usize)
                        .cast::<u8>(),
                    cell_size as usize,
                );
            }
            self.slots_mut()[i] = destination_offset;
        }

        self.header_mut().last_used_offset = destination_offset;
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

    fn slots_non_null(&self) -> NonNull<[u16]> {
        unsafe {
            NonNull::slice_from_raw_parts(
                self.buffer.byte_add(PAGE_HEADER_SIZE as _).cast(),
                self.header().num_slots as _,
            )
        }
    }

    fn slots(&self) -> &[u16] {
        unsafe { self.slots_non_null().as_ref() }
    }

    fn slots_mut(&mut self) -> &mut [u16] {
        unsafe { self.slots_non_null().as_mut() }
    }

    unsafe fn cell_at_offset(&self, offset: u16) -> NonNull<CellHeader> {
        self.buffer.byte_add(offset as _).cast()
    }

    /// If data is not aligned to 64 bits then add padding.
    pub fn aligned_size_of(data: &[u8]) -> u16 {
        Layout::from_size_align(data.len(), CELL_ALIGNMENT)
            .unwrap()
            .pad_to_align()
            .size() as _
    }

    pub fn usable_space(page_size: u16) -> u16 {
        page_size - PAGE_HEADER_SIZE
    }

    fn layout(page_size: u16) -> Layout {
        unsafe { Layout::from_size_align_unchecked(page_size as usize, CELL_ALIGNMENT) }
    }

    pub fn is_underflow(&self) -> bool {
        self.len() == 0
            || !self.is_root() && self.header().free_space > Self::usable_space(self.size()) / 2
    }

    pub fn is_root(&self) -> bool {
        self.number == 0
    }

    pub fn iter_children(&self) -> impl Iterator<Item = PageNumber> + '_ {
        let len = if self.is_leaf() { 0 } else { self.len() + 1 };

        (0..len).map(|i| self.child(i))
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
