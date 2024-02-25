//! Implementations of different disk page types.
//!
//! Currently there are 3 types of pages:
//!
//! - [`Page`]: Used for BTree nodes. This is the most common type of page,
//! that's why it's called literally "Page".
//!
//! - [`PageZero`]: Special subtype of [`Page`] which holds additional metadata
//! about the database file.
//!
//! - [`OverflowPage`]: Used when variable size data exceeds the maximum amount
//! of bytes that can be stored in a single page. [`OverflowPage`] instances are
//! also reused as free pages, because they use the same headers.
//!
//! Each database is stored in a single file, and the data structures in this
//! module operate on pages of such file. The DB file looks roughly like this:
//!
//! ```text
//! +--------------------------+ <----+
//! |       File Metadata      |      |
//! +--------------------------+      |
//! | +----------------------+ |      |
//! | | Btree Page 0 Header  | |      | PAGE 0     struct PageZero
//! | +----------------------+ |      |
//! | | Btree page 0 Content | |      |
//! | +----------------------+ |      |
//! +--------------------------+ <----+
//! | +----------------------+ |      |
//! | | Btree Page 1 Header  | |      |
//! | |                      | |      |
//! | +----------------------+ |      | PAGE 1       struct Page
//! | | Btree page 1 Content | |      |
//! | |                      | |      |
//! | +----------------------+ |      |
//! +--------------------------+ <----+
//! | +----------------------+ |      |
//! | | Btree Page 2 Header  | |      |
//! | |                      | |      |
//! | +----------------------+ |      | PAGE 2       struct Page
//! | | Btree page 2 Content | |      |
//! | |                      | |      |
//! | +----------------------+ |      |
//! +--------------------------+ <----+
//! | +----------------------+ |      |
//! | | Overflow Page Header | |      |
//! | |                      | |      |
//! | +----------------------+ |      | PAGE 3     struct OverflowPage
//! | |   Overflow Content   | |      |
//! | |                      | |      |
//! | +----------------------+ |      |
//! +--------------------------+ <----+
//! | +----------------------+ |      |
//! | |    Fee Page Header   | |      |
//! | |                      | |      |
//! | +----------------------+ |      | PAGE 4  struct OverflowPage (AKA FreePage)
//! | |        UNUSED        | |      |
//! | |                      | |      |
//! | +----------------------+ |      |
//! +--------------------------+ <----+
//!             ...
//! ```
//!
//! Pages link to other pages using their [`PageNumber`], which is just a 32 bit
//! offset that can be used to jump from the beginning of the file to a
//! concrete page.
//!
//! All the datastructures in this module offer APIs to operate on a single
//! page. For multi-page operations see the [`super::btree`] module.

use std::{
    alloc::{self, Allocator, Layout},
    cmp::Ordering,
    collections::BinaryHeap,
    fmt::Debug,
    iter,
    mem::{self, ManuallyDrop},
    ops::{Bound, RangeBounds},
    ptr::NonNull,
};

use crate::paging::pager::PageNumber;

/// Magic number at the beginning of the database file.
///
/// `0xB74EE` is supposed to stand for "BTree" and also serves as endianess
/// check, since the big endian and little endian representations are different.
pub(crate) const MAGIC: u32 = 0xB74EE;

/// Maximum page size is 64 KiB.
pub(crate) const MAX_PAGE_SIZE: usize = 64 * 1024;

/// Minimum acceptable page size.
///
/// When in debug mode the minimum value of the page size is calculated so that
/// a normal [`Page`] instance can store at least one valid [`Cell`], which is
/// an aligned cell that can fit [`CELL_ALIGNMENT`] bytes.
///
/// In numbers, at the moment of writing this the page header is 12 bytes, a
/// slot pointer is 2 bytes, and the cell header is 8 bytes. This gives us a
/// total of 22 bytes of metadata for one single payload.
///
/// [`CELL_ALIGNMENT`] is 8 bytes, so we'll consider that to be the minimum
/// payload. Essentially what we do is add 8 bytes to 22 bytes, giving us 30
/// bytes, and then we align upwards to 32 bytes. So the minimum page size in
/// debug mode can store only 8 bytes worth of data, but we allow this because
/// most tests use really small page sizes for simplicity and because it's
/// easier to debug.
///
/// In release mode we'll consider the minimum size to be 512 bytes.
pub(crate) const MIN_PAGE_SIZE: usize = if cfg!(debug_assertions) {
    (PAGE_HEADER_SIZE + SLOT_SIZE + CELL_HEADER_SIZE + 2 * CELL_ALIGNMENT as u16 - 1) as usize
        & !(CELL_ALIGNMENT - 1)
} else {
    512
};

/// Size of the [`Page`] header. See [`PageHeader`] for details.
pub(crate) const PAGE_HEADER_SIZE: u16 = mem::size_of::<PageHeader>() as _;

/// Size of the database file header.
pub(crate) const DB_HEADER_SIZE: u16 = mem::size_of::<DbHeader>() as _;

/// Size of [`CellHeader`].
pub(crate) const CELL_HEADER_SIZE: u16 = mem::size_of::<CellHeader>() as _;

/// Size of an individual slot (offset pointer).
pub(crate) const SLOT_SIZE: u16 = mem::size_of::<u16>() as _;

/// See [`Page`] for alignment details.
pub(crate) const CELL_ALIGNMENT: usize = mem::align_of::<CellHeader>();

/// The slot array can be indexed using 2 bytes, since it will never be bigger
/// than [`MAX_PAGE_SIZE`].
pub(crate) type SlotId = u16;

/// In-memory binary buffer split into header and content.
///
/// This struct is the basic building block for all the pages and contains a
/// single memory buffer that looks like this:
///
/// ```text
/// +----- Buffer address
/// |
/// |        +----- Content offset  (Buffer address + size of header)
/// |        |
/// V        V
/// +--------+--------------------------+
/// | HEADER |         CONTENT          |
/// +--------+--------------------------+
/// ```
///
/// The struct provides methods for directly reading or writing both the header
/// and the content. The size in bytes of the header is determined by the
/// generic type `H` using [`mem::size_of`].
///
/// References to the entire memory buffer (including both the header and the
/// content) can be obtained using [`AsRef::as_ref`] and [`AsMut::as_mut`].
#[derive(Debug)]
pub(crate) struct BufferWithHeader<H> {
    /// Pointer to the header located at the beginning of the buffer.
    header: NonNull<H>,
    /// Pointer to the content located right after the header.
    content: NonNull<[u8]>,
    /// Total size of the buffer (size of header + size of content).
    size: usize,
}

impl<H> BufferWithHeader<H> {
    /// Calculates the content address of the given buffer and returns a pointer
    /// to it.
    ///
    /// # Safety
    ///
    /// Caller must ensure that the buffer has enough size to contain at least
    /// `mem::size_of::<H>() + 1` bytes.
    unsafe fn content_of(buffer: NonNull<[u8]>) -> NonNull<[u8]> {
        NonNull::slice_from_raw_parts(
            buffer.byte_add(mem::size_of::<H>()).cast::<u8>(),
            Self::usable_space(buffer.len()) as usize,
        )
    }

    /// Returns a pointer to the in-memory buffer.
    ///
    /// This is needed for reading and writing pages to/from disk using
    /// [`AsRef`] and [`AsMut`].
    fn pointer(&self) -> NonNull<[u8]> {
        NonNull::slice_from_raw_parts(self.header.cast::<u8>(), self.size)
    }

    /// Allocates a new buffer setting all the bytes to 0.
    ///
    /// If all the fields in the header should be set to 0 then this is good
    /// enough, otherwise use [`Self::new`] and pass the header manually.
    pub fn alloc(size: usize) -> Self {
        debug_assert!(
            (MIN_PAGE_SIZE..=MAX_PAGE_SIZE).contains(&size),
            "size {size} is not a value between {MIN_PAGE_SIZE} and {MAX_PAGE_SIZE}"
        );
        debug_assert!(
            size as usize > mem::size_of::<H>(),
            "allocated buffer only has space for the header, which makes the content pointer invalid"
        );

        // TODO: We can probably handle the alloc error by rolling back the
        // database and returning an error to the client.
        let buffer = alloc::Global
            .allocate_zeroed(alloc::Layout::from_size_align(size as _, CELL_ALIGNMENT).unwrap())
            .expect("could not allocate page buffer");

        // SAFETY: If the assertions above were met then the content pointer
        // should always be valid.
        let content = unsafe { Self::content_of(buffer) };

        Self {
            header: buffer.cast(),
            content,
            size: buffer.len(),
        }
    }

    /// Allocates a new buffer writing the given header at the beginning.
    pub fn new(size: usize, header: H) -> Self {
        let mut buffer = Self::alloc(size);
        *buffer.header_mut() = header;

        buffer
    }

    /// Sames as [`Self::new`] but does not allocate, it uses the given wrapper
    /// instead.
    ///
    /// # Safety
    ///
    /// This function is marked as unsafe because the caller must make two
    /// guarantees to prevent use after free bugs:
    ///
    /// 1. The wrapped buffer is never used after the parent buffer is dropped.
    /// 2. The wrapped buffer is never dropped itself.
    ///
    /// Ensuring such behaviours should make this safe. See the implementation
    /// of [`Drop`] for [`BufferWithHeader`] for more details.
    pub unsafe fn new_in<W>(wrapper: &mut BufferWithHeader<W>, header: H) -> Self {
        let buffer = wrapper.content_non_null();

        debug_assert!(
            buffer.len() > mem::size_of::<H>(),
            "wrapped buffer only has space for the header, which makes the content pointer invalid"
        );

        let content = unsafe { Self::content_of(buffer) };

        let mut buffer = Self {
            header: buffer.cast(),
            content,
            size: buffer.len(),
        };

        *buffer.header_mut() = header;

        buffer
    }

    /// Number of bytes that can be used to store content.
    ///
    /// ```text
    ///                 usable_space()
    ///          +--------------------------+
    ///          |                          |
    ///          V                          V
    /// +--------+--------------------------+
    /// | HEADER |         CONTENT          |
    /// +--------+--------------------------+
    /// ```
    pub fn usable_space(page_size: usize) -> u16 {
        (page_size - mem::size_of::<H>()) as u16
    }

    /// Full size of the buffer in bytes.
    ///
    /// ```text
    ///                size()
    /// +-----------------------------------+
    /// |                                   |
    /// V                                   V
    /// +--------+--------------------------+
    /// | HEADER |         CONTENT          |
    /// +--------+--------------------------+
    /// ```
    pub fn size(&self) -> usize {
        self.size
    }

    /// Returns a read-only reference to the header.
    pub fn header(&self) -> &H {
        // SAFETY: The underlying [`NonNull`] pointer should always be valid and
        // aligned since we allocate it ourselves or obtain it from another
        // buffer that has already been allocated when using [`Self::new_in`].
        // Unless we manually build an instance of this struct giving it an
        // incorrect pointer this should be safe.
        unsafe { self.header.as_ref() }
    }

    /// Returns a mutable reference to the header.
    pub fn header_mut(&mut self) -> &mut H {
        // SAFETY: Same as [`Self::header`].
        unsafe { self.header.as_mut() }
    }

    /// Returns a pointer to the content of this buffer.
    fn content_non_null(&self) -> NonNull<[u8]> {
        self.content
    }

    /// Returns a read-only reference to the content part of this buffer.
    pub fn content(&self) -> &[u8] {
        // SAFETY: The assertions in [`Self::alloc`] verify that the content
        // pointer is valid.
        unsafe { self.content_non_null().as_ref() }
    }

    /// Returns a mutable reference to the content part of this buffer.
    pub fn content_mut(&mut self) -> &mut [u8] {
        // SAFETY: Same as [`Self::content`].
        unsafe { self.content_non_null().as_mut() }
    }
}

impl<H> AsRef<[u8]> for BufferWithHeader<H> {
    fn as_ref(&self) -> &[u8] {
        // SAFETY: We allocate the buffer so we know the pointer is valid and
        // well aligned.
        unsafe { self.pointer().as_ref() }
    }
}

impl<H> AsMut<[u8]> for BufferWithHeader<H> {
    fn as_mut(&mut self) -> &mut [u8] {
        // SAFETY: Same as [`AsRef::as_ref`].
        unsafe { self.pointer().as_mut() }
    }
}

impl<H> PartialEq for BufferWithHeader<H> {
    fn eq(&self, other: &Self) -> bool {
        self.as_ref().eq(other.as_ref())
    }
}

impl<H> Clone for BufferWithHeader<H> {
    fn clone(&self) -> Self {
        let mut buffer = Self::alloc(self.size());
        buffer.as_mut().copy_from_slice(self.as_ref());

        buffer
    }
}

impl<H> Drop for BufferWithHeader<H> {
    fn drop(&mut self) {
        // NOTE: See [`Self::new_in`]. It is possible to have buffers contained
        // within other buffers, in which case this code should not run. We
        // could add a guard to each buffer, something like `if self.is_wrapped`
        // but as of right now there is only one case of wrapped buffer (page
        // zero, which contains two headers). So we'll just leave it as is and
        // delegate the responsibilities to the caller (which unfortunatley
        // happens to be the same person who wrote this...)
        unsafe {
            alloc::Global.deallocate(
                self.header.cast(),
                alloc::Layout::from_size_align(self.size, CELL_ALIGNMENT).unwrap(),
            )
        }
    }
}

/// A trait similar to [`Default`] but for initializing empty pages in memory.
pub(crate) trait InitEmptyPage {
    /// Initializes an empty page of the given size.
    fn init(number: PageNumber, size: usize) -> Self;
}

/// Slotted page header.
///
/// It is located at the beginning of each page and it does not contain variable
/// length data:
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

    /// Offset of the last inserted cell.
    ///
    /// ```text
    ///                         last_used_offset
    ///                               |
    ///                               V
    /// +--------+-------+------------+------+-----+------+
    /// | HEADER | SLOTS | FREE SPACE | CELL | DEL | CELL |
    /// +--------+-------+------------+------+-----+------+
    /// ```
    ///
    /// For empty pages with no cells, this value is equal to the page size:
    ///
    /// ```text
    ///                                             last_used_offset
    ///                                                   |
    ///                                                   V
    /// +--------+----------------------------------------+
    /// | HEADER |              EMPTY PAGE                |
    /// +--------+----------------------------------------+
    /// ```
    ///
    /// This makes it so that there is no distinction between empty pages and
    /// pages with cells, we can always substract the size of a cell from
    /// `last_used_offset` to obtain the next `last_used_offset`.
    ///
    /// Note that this value is of type [`u32`] even though a [`Cell`] can never
    /// have an offset greater than [`u16::MAX`]. However, [`u16::MAX`] is equal
    /// to 65535 (2^16 - 1) while [`MAX_PAGE_SIZE`] is 65536 (2^16), so the
    /// trick mentioned above to reduce the algorithm to one single case would
    /// not work if we stored this value in [`u16`], we'd have to use if
    /// statements when inserting cells to account for 64 KiB pages.
    ///
    /// Storing this as [`u16`] would require 2 bytes of padding to align the
    /// [`PageNumber`] below anyway, so we might as well use those bytes for
    /// something.
    ///
    /// This has its own problems though, having to deal with `u16` and `u32`
    /// arithmetic requires a bunch of casts. So... TODO.
    last_used_offset: u32,

    /// Last child of this page.
    pub right_child: PageNumber,
}

impl PageHeader {
    /// Initializes a new header for empty pages.
    fn new(size: usize) -> Self {
        Self {
            num_slots: 0,
            last_used_offset: size as _,
            free_space: Page::usable_space(size),
            right_child: 0,
        }
    }
}

/// Cell header located at the beginning of each cell.
///
/// The header stores the size of the cell without including its own size and it
/// also stores a pointer to the BTree page the contains entries "less than"
/// this one.
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
/// Cells are 64 bit aligned. See [`Page`] for more details.
///
/// # Overflow
///
/// The maximum size of a cell is about one fourth of the page size (substract
/// the cell header size and slot pointer size). This allows to store a minimum
/// of 4 keys per page. If a cell needs to hold more data than we can fit in a
/// single page, then we'll set [`CellHeader::is_overflow`] to `true` and make
/// the last 4 bytes of the content point to an overflow page.
#[derive(Debug, PartialEq, Clone)]
#[repr(C, align(8))]
pub(crate) struct CellHeader {
    /// Size of the cell content.
    size: u16,

    /// True if this cell points to an overflow page.
    ///
    /// Since we need to add
    /// padding to align the fields of this struct anyway, we're not wasting
    /// space here.
    pub is_overflow: bool,

    /// Add padding manually to avoid uninitialized bytes.
    ///
    /// The [`PartialEq`] implementation for [`Page`] relies on comparing the
    /// memory buffers, but Rust does not make any guarantees about the values
    /// of padding bytes. See here:
    ///
    /// <https://github.com/rust-lang/unsafe-code-guidelines/issues/174>
    padding: u8,

    /// Page number of the BTree page that contains values less than this cell.
    pub left_child: PageNumber,
}

impl CellHeader {
    /// Total size of the cell including the header.
    fn total_size(&self) -> u16 {
        CELL_HEADER_SIZE + self.size
    }

    /// Total size of the cell including the header and the slot array pointer
    /// needed to store the offset.
    pub fn storage_size(&self) -> u16 {
        self.total_size() + SLOT_SIZE
    }

    /// Returns a pointer to the content of the given cell header.
    ///
    /// # Safety
    ///
    /// Caller must guarantee that the given pointer is valid and points to a
    /// cell within the page.
    unsafe fn content_of(header: NonNull<Self>) -> NonNull<[u8]> {
        NonNull::slice_from_raw_parts(header.add(1).cast(), header.as_ref().size as _)
    }
}

/// Owned version of a cell.
///
/// The [`crate::storage::BTree`] structure reorders cells around different
/// sibling pages when an overflow or underflow occurs, so instead of hiding the
/// low level details we provide some API that can be used by upper levels.
#[derive(Debug, PartialEq, Clone)]
pub(crate) struct Cell {
    pub header: CellHeader,
    pub content: Box<[u8]>,
}

/// Read-only reference to a cell.
///
/// The cell could be located either in a page that comes from disk or in memory
/// if the page overflowed.
#[derive(Debug, PartialEq)]
pub(crate) struct CellRef<'a> {
    pub header: &'a CellHeader,
    pub content: &'a [u8],
}

impl CellRef<'_> {
    pub fn overflow_page(&self) -> PageNumber {
        if !self.header.is_overflow {
            return 0;
        }

        PageNumber::from_le_bytes(
            self.content[self.content.len() - mem::size_of::<PageNumber>()..]
                .try_into()
                .expect("failed parsing overflow page number"),
        )
    }
}

/// Same as [`CellRef`] but mutable.
pub(crate) struct CellRefMut<'a> {
    pub header: &'a mut CellHeader,
    pub content: &'a mut [u8],
}

impl<'a> From<&'a Cell> for CellRef<'a> {
    fn from(cell: &'a Cell) -> Self {
        Self {
            header: &cell.header,
            content: &cell.content,
        }
    }
}

impl PartialEq<&Cell> for CellRef<'_> {
    fn eq(&self, other: &&Cell) -> bool {
        self.header.eq(&other.header) && self.content.eq(other.content.as_ref())
    }
}

impl Cell {
    /// Creates a new cell allocated in memory.
    pub fn new(mut payload: Vec<u8>) -> Self {
        let size = Self::aligned_size_of(&payload);
        // Add padding.
        payload.resize(size as _, 0);

        Self {
            header: CellHeader {
                size,
                left_child: 0,
                is_overflow: false,
                padding: 0,
            },
            content: payload.into_boxed_slice(),
        }
    }

    /// Creates a new overflow cell by extending the `payload` buffer with the
    /// `overflow_page` number.
    pub fn new_overflow(mut payload: Vec<u8>, overflow_page: PageNumber) -> Self {
        payload.extend_from_slice(&overflow_page.to_le_bytes());

        let mut cell = Self::new(payload);
        cell.header.is_overflow = true;

        cell
    }

    /// Shorthand for [`CellHeader::total_size`].
    pub fn total_size(&self) -> u16 {
        self.header.total_size()
    }

    /// Shorthand for [`CellHeader::storage_size`].
    pub fn storage_size(&self) -> u16 {
        self.header.storage_size()
    }

    pub fn overflow_page(&self) -> PageNumber {
        CellRef::from(self).overflow_page()
    }

    /// See [`Page`] for details.
    pub fn aligned_size_of(data: &[u8]) -> u16 {
        Layout::from_size_align(data.len(), CELL_ALIGNMENT)
            .unwrap()
            .pad_to_align()
            .size() as _
    }
}

/// Cell that didn't fit in the slotted page. This is stored in-memory and
/// cannot be written to disk.
#[derive(Debug, Clone)]
struct OverflowCell {
    /// Owned cell.
    cell: Cell,
    /// Index in the slot array where the cell should have been inserted.
    index: SlotId,
}

impl PartialEq for OverflowCell {
    fn eq(&self, other: &Self) -> bool {
        self.index.eq(&other.index)
    }
}

impl Eq for OverflowCell {}

impl PartialOrd for OverflowCell {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OverflowCell {
    fn cmp(&self, other: &Self) -> Ordering {
        // BinaryHeap is max-heap by default, make it min-heap.
        self.index.cmp(&other.index).reverse()
    }
}

/// Fixed size slotted page.
///
/// This is what we store on disk. The page maintains a "slot array" located
/// after the header that grows towards the right. On the opposite side is where
/// used cells are located, leaving free space in the middle of the page. Each
/// item in the slot array points to one of the used cells through its offset,
/// calculated from the start of the page (before the header).
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
///
/// # Null Slots
///
/// Initially this was going to be a generic slotted page implementation that
/// would serve as a building block for both index organized storage (like
/// SQLite) and tuple oriented storage (like Postgres), but the idea has been
/// dropped in favor of simplicity.
///
/// The main difference is that when using index organized storage we never
/// point to slot indexes from outside the page, so there's no need to attempt
/// to maintain their current position for as long as possible. This [commit]
/// contains the last version that used to do so. Another benefit of the current
/// aproach is that BTree nodes and disk pages are the same thing, because
/// everything is stored in a BTree, so we need less generics, less types and
/// less code. Take a look at the [btree.c] file from SQLite 2.X.X for the
/// inspiration.
///
/// Check these lectures for some background on slotted pages and data storage:
///
/// - [F2023 #03 - Database Storage Part 1 (CMU Intro to Database Systems)](https://www.youtube.com/watch?v=DJ5u5HrbcMk)
/// - [F2023 #04 - Database Storage Part 2 (CMU Intro to Database Systems)](https://www.youtube.com/watch?v=Ra50bFHkeM8)
///
/// [commit]: https://github.com/antoniosarosi/mkdb/blob/3011003170f02d337f62cdd9f5af0f3b63786144/src/paging/page.rs
/// [btree.c]: https://github.com/antoniosarosi/sqlite2-btree-visualizer/blob/master/src/btree.c
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
#[derive(Clone)]
pub(crate) struct Page {
    /// Page number on disk.
    pub number: PageNumber,
    /// Fixed size in-memory buffer that contains the data read from disk.
    buffer: BufferWithHeader<PageHeader>,
    /// Overflow list.
    overflow: BinaryHeap<OverflowCell>,
}

impl InitEmptyPage for Page {
    fn init(number: PageNumber, size: usize) -> Self {
        let buffer = BufferWithHeader::new(size, PageHeader::new(size));

        Self {
            number,
            buffer,
            overflow: BinaryHeap::new(),
        }
    }
}

// TODO: Technically two pages could have the exact same content but different
// internal structure due to calls to defragment or different insertion order.
// However figuring out if the cells are actually equal probably requires an
// O(n^2) algorithm. This is good enough for tests anyway, as the BTree always
// inserts data in the same order.
impl PartialEq for Page {
    fn eq(&self, other: &Self) -> bool {
        self.number == other.number && self.buffer == other.buffer
    }
}

impl AsRef<[u8]> for Page {
    fn as_ref(&self) -> &[u8] {
        self.buffer.as_ref()
    }
}

impl AsMut<[u8]> for Page {
    fn as_mut(&mut self) -> &mut [u8] {
        self.buffer.as_mut()
    }
}

impl Page {
    /// Amount of space that can be used in a page to store [`Cell`] instances.
    ///
    /// Since [`MAX_PAGE_SIZE`] is 64 KiB, the usable space should be a value
    /// less than [`u16::MAX`] unless the header of the page is zero sized,
    /// which doesn't make much sense anyway.
    pub fn usable_space(page_size: usize) -> u16 {
        BufferWithHeader::<PageHeader>::usable_space(page_size)
    }

    /// The maximum payload size that can be stored in the given usable space.
    ///
    /// It's calculated by substracting the cell header size and the slot
    /// pointer size from the usable space and then aligning the result
    /// downwards to [`CELL_ALIGNMENT`]. This makes sure that at least one cell
    /// can successfuly fit in the given space.
    fn max_payload_size_in(usable_space: u16) -> u16 {
        (usable_space - CELL_HEADER_SIZE - SLOT_SIZE) & !(CELL_ALIGNMENT as u16 - 1)
    }

    /// The maximum size that the payload of a single cell "should" take on
    /// the page.
    ///
    /// We hardcoded the number 4 here so that the BTree can always store at
    /// least 4 cells in every page, but this should probably be configurable.
    pub fn ideal_max_payload_size(page_size: usize) -> u16 {
        let usable_space = Self::usable_space(page_size);

        let max_size = Self::max_payload_size_in(usable_space / 4);

        // When the page size is too small we can't fit 4 keys. This is mostly
        // for tests, since we use small page sizes for simplicity.
        #[cfg(debug_assertions)]
        if max_size == 0 {
            return Self::max_payload_size_in(usable_space);
        }

        max_size
    }

    /// Similar to [`Self::ideal_max_payload_size`] but allows a cell to occupy
    /// as much as possible. See [`Self::insert`] for more details.
    fn max_allowed_payload_size(&self) -> u16 {
        Self::max_payload_size_in(Self::usable_space(self.size()))
    }

    /// Size in bytes of the page.
    pub fn size(&self) -> usize {
        self.buffer.size()
    }

    /// Number of cells in the page.
    pub fn len(&self) -> u16 {
        self.header().num_slots + self.overflow.len() as u16
    }

    /// Reference to the page header.
    pub fn header(&self) -> &PageHeader {
        self.buffer.header()
    }

    /// Mutable reference to the page header.
    pub fn header_mut(&mut self) -> &mut PageHeader {
        self.buffer.header_mut()
    }

    /// Pointer to the slot array.
    fn slot_array_non_null(&self) -> NonNull<[u16]> {
        NonNull::slice_from_raw_parts(
            self.buffer.content_non_null().cast(),
            self.header().num_slots as _,
        )
    }

    // Slotted array as a slice.
    fn slot_array(&self) -> &[u16] {
        // SAFETY: See [`BufferWithHeader::content_non_null`].
        unsafe { self.slot_array_non_null().as_ref() }
    }

    // Slotted array as a mutable slice.
    fn slot_array_mut(&mut self) -> &mut [u16] {
        // SAFETY: See [`BufferWithHeader::content_non_null`].
        unsafe { self.slot_array_non_null().as_mut() }
    }

    /// Returns a pointer to the [`CellHeader`] located at the given `offset`.
    ///
    /// # Safety
    ///
    /// This is the only function marked as `unsafe` because we can't guarantee
    /// the offset is valid within the function, so the caller is responsible
    /// for that.
    unsafe fn cell_header_at_offset(&self, offset: u16) -> NonNull<CellHeader> {
        self.buffer.header.byte_add(offset as _).cast()
    }

    /// Returns a pointer to the [`CellHeader`] pointer by the given slot.
    fn cell_header_at_slot_index(&self, index: SlotId) -> NonNull<CellHeader> {
        // SAFETY: The slot array always stores valid offsets within the page
        // that point to actual cells.
        unsafe { self.cell_header_at_offset(self.slot_array()[index as usize]) }
    }

    /// Read-only reference to a cell.
    pub fn cell(&self, index: SlotId) -> CellRef {
        let header = self.cell_header_at_slot_index(index);
        // SAFETY: The slot array stores a correct offset to a cell. Once we
        // have the pointer to the cell, we know the header and content are
        // contiguous, so we can take the two references.
        unsafe {
            CellRef {
                header: header.as_ref(),
                content: CellHeader::content_of(header).as_ref(),
            }
        }
    }

    /// Mutable reference to a cell.
    pub fn cell_mut(&mut self, index: SlotId) -> CellRefMut {
        let mut header = self.cell_header_at_slot_index(index);
        // SAFETY: This one is not as easy as getting a [`CellRef`] because we
        // can't get a mutable reference to the header and _then_ a mutable
        // reference to the content, since in order to know how long the content
        // is we have to borrow the header as immutable to read the `size`
        // property. Since we can't borrow as mutable and immutable at the same
        // time we'll just flip the order. Grab the content first and then the
        // header.
        unsafe {
            CellRefMut {
                content: CellHeader::content_of(header).as_mut(),
                header: header.as_mut(),
            }
        }
    }

    /// Returns an owned cell by cloning it.
    pub fn owned_cell(&self, index: SlotId) -> Cell {
        let header = self.cell_header_at_slot_index(index);
        unsafe {
            Cell {
                header: header.read(),
                content: Vec::from(CellHeader::content_of(header).as_ref()).into(),
            }
        }
    }

    /// Returns the child at the given `index`.
    pub fn child(&self, index: SlotId) -> PageNumber {
        if index == self.len() {
            self.header().right_child
        } else {
            self.cell(index).header.left_child
        }
    }

    /// Iterates over all the children pointers in this page.
    pub fn iter_children(&self) -> impl Iterator<Item = PageNumber> + '_ {
        let len = if self.is_leaf() { 0 } else { self.len() + 1 };

        (0..len).map(|i| self.child(i))
    }

    /// Returns `true` if this page is underflow.
    ///
    /// An ordinary page is in "underflow" condition when less than half of
    /// its usable space is occupied. Special cases must be handled at other
    /// layers.
    pub fn is_underflow(&self) -> bool {
        self.header().free_space > Self::usable_space(self.size()) / 2
    }

    /// Returns `true` if this page is overflow.
    pub fn is_overflow(&self) -> bool {
        !self.overflow.is_empty()
    }

    pub fn can_consume_without_overflow(&self, other: &Self) -> bool {
        if other.is_overflow() {
            return false;
        }

        let used_bytes = Self::usable_space(other.size()) - other.header().free_space;

        self.header().free_space >= used_bytes
    }

    /// Just like [`Vec::append`], this function removes all the cells in
    /// `other` and adds them to `self`.
    pub fn append(&mut self, other: &mut Self) {
        other.drain(..).for_each(|cell| self.push(cell));
        self.header_mut().right_child = other.header().right_child;
    }

    /// Returns `true` if this page has no children.
    pub fn is_leaf(&self) -> bool {
        self.header().right_child == 0
    }

    /// Adds `cell` to this page, possibly overflowing the page.
    pub fn push(&mut self, cell: Cell) {
        self.insert(self.len(), cell);
    }

    /// Inserts the `cell` at `index`, possibly overflowing.
    pub fn insert(&mut self, index: SlotId, cell: Cell) {
        debug_assert!(
            cell.content.len() <= self.max_allowed_payload_size() as usize,
            "attempt to store payload of size {} when max allowed payload size is {}",
            cell.content.len(),
            self.max_allowed_payload_size()
        );

        if self.is_overflow() {
            return self.overflow.push(OverflowCell { cell, index });
        }

        if let Err(cell) = self.try_insert(index, cell) {
            self.overflow.push(OverflowCell { cell, index });
        }
    }

    /// Attempts to replace the cell at `index` with `new_cell`.
    ///
    /// It causes the page to overflow if it's not possible. After that, this
    /// function should not be called anymore.
    pub fn replace(&mut self, index: SlotId, new_cell: Cell) -> Cell {
        debug_assert!(
            !self.is_overflow(),
            "overflow cells are not replaced so replace() should not run on overflow pages"
        );

        match self.try_replace(index, new_cell) {
            Ok(old_cell) => old_cell,

            Err(new_cell) => {
                let old_cell = self.remove(index);

                self.overflow.push(OverflowCell {
                    cell: new_cell,
                    index,
                });

                old_cell
            }
        }
    }

    /// Attempts to insert the given `cell` in this page.
    ///
    /// There are two possible cases:
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
    fn try_insert(&mut self, index: SlotId, cell: Cell) -> Result<SlotId, Cell> {
        let cell_storage_size = cell.storage_size();

        // There's no way we can fit the cell in this page.
        if self.header().free_space < cell_storage_size {
            return Err(cell);
        }

        // Space between the end of the slot array and the closest cell.
        let available_space = {
            let end = self.header().last_used_offset;
            let start = PAGE_HEADER_SIZE + self.header().num_slots * SLOT_SIZE;

            // This value fits in u16 even if the page is 64 KiB.
            (end - start as u32) as u16
        };

        // We can fit the new cell but we have to defragment the page first.
        if available_space < cell_storage_size {
            self.defragment();
        }

        // Same as above, this offset always fits in u16.
        let offset = self.header().last_used_offset - cell.total_size() as u32;

        // Write new cell.
        // SAFETY: `last_used_offset` keeps track of where the last cell was
        // written. By substracting the total size of the new cell to
        // `last_used_offset` we get a valid pointer within the page where we
        // write the new cell.
        unsafe {
            let header = self.cell_header_at_offset(offset as u16);
            header.write(cell.header);

            CellHeader::content_of(header)
                .as_mut()
                .copy_from_slice(&cell.content);
        }

        // Update header.
        self.header_mut().last_used_offset = offset;
        self.header_mut().free_space -= cell_storage_size;

        // Add new slot.
        self.header_mut().num_slots += 1;

        // If the index is not the last one, shift slots to the right.
        if index < self.header().num_slots {
            let end = self.header().num_slots as usize - 1;
            self.slot_array_mut()
                .copy_within(index as usize..end, index as usize + 1);
        }

        // Set offset.
        self.slot_array_mut()[index as usize] = offset as u16;

        Ok(index)
    }

    /// Tries to replace the cell pointed by the given slot `index` with the
    /// `new_cell`.
    ///
    /// Similar to [`Self::try_insert`] there are 2 main cases:
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
    fn try_replace(&mut self, index: SlotId, new_cell: Cell) -> Result<Cell, Cell> {
        let old_cell = self.cell(index);

        // There's no way we can fit the new cell in this page, even if we
        // remove the one that has to be replaced.
        if self.header().free_space + old_cell.header.total_size() < new_cell.total_size() {
            return Err(new_cell);
        }

        // Case 1: The new cell is smaller than the old cell. This is the best
        // case scenario because we can simply overwrite the contents without
        // doing much else.
        if new_cell.header.size <= old_cell.header.size {
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

    /// Removes the cell pointed by the given slot `index`.
    ///
    /// Unlike [`Self::try_insert`] and [`Self::try_replace`], this function
    /// cannot fail. However, it does panic if the given `index` is out of
    /// bounds or the page is overflow.
    pub fn remove(&mut self, index: SlotId) -> Cell {
        debug_assert!(
            !self.is_overflow(),
            "remove() does not handle overflow indexes"
        );

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

    /// Slides cells towards the right to eliminate fragmentation.
    ///
    /// For example:
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
            // SAFETY: Calling [`Self::cell_header_at_offset`] is safe here
            // because we obtained all the offsets from the slot array, which
            // should always be in a valid state. If that holds true, then
            // casting the dereferencing the cell pointer should be safe as
            // well.
            unsafe {
                let cell = self.cell_header_at_offset(offset);
                let size = cell.as_ref().total_size() as usize;

                destination_offset -= size;

                cell.cast::<u8>().copy_to(
                    self.buffer.header.byte_add(destination_offset).cast::<u8>(),
                    size,
                );
            }
            self.slot_array_mut()[i] = destination_offset as u16;
        }

        self.header_mut().last_used_offset = destination_offset as u32;
    }

    /// Works just like [`Vec::drain`]. Removes the specified cells from this
    /// page and returns an owned version of them.
    ///
    /// This function does account for [`Self::is_overflow`], so it's safe to
    /// call on overflow pages.
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

        iter::from_fn(move || {
            // Copy cells until we reach the end.
            if drain_index < end {
                let cell = if self
                    .overflow
                    .peek()
                    .is_some_and(|overflow| overflow.index as usize == drain_index)
                {
                    self.overflow.pop().unwrap().cell
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
                    .map(|slot| self.cell(slot as _).header.storage_size())
                    .sum::<u16>();

                self.slot_array_mut().copy_within(slot_index.., start);

                self.header_mut().num_slots -= (slot_index - start) as u16;

                None
            }
        })
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
                            .field("start", &offset)
                            .field("end", &(offset + cell.header.total_size()))
                            .field("size", &cell.header.size)
                            .field("header", &cell.header)
                            .field("content", &cell.content)
                            .finish()
                    });
                });

                list.finish()
            })
            .finish()
    }
}

/// Header of an overflow page.
#[derive(Debug)]
#[repr(C)]
pub(crate) struct OverflowPageHeader {
    /// Next overflow page.
    pub next: PageNumber,
    /// Number of bytes stored in this page. Not to be confused with the page
    /// size.
    pub num_bytes: u16,
}

/// Cell overflow page.
///
/// Cells have a maximum size that depends on the page size. See
/// [`Page::max_payload_size`] for details. If the page size is big enough we
/// attempt to fit at least 4 cells in each page. However, when the payload
/// size of a cell exceeds the maximum size, we need to allocate extra pages
/// to store the contents of that cell. The cell then points to the first
/// overflow page, and that page points to the next and so on.
///
/// ```text
/// PAGE       SLOT          FREE
/// HEADER     ARRAY         SPACE         CELLS
/// +----------------------------------------------------------------+
/// | PAGE   | +---+                      +--------+---------+-----+ |
/// | HEADER | | 1 | ->                <- | HEADER | PAYLOAD | OVF | |
/// |        | +---+                      +--------+---------+--|--+ |
/// +------------|----------------------------------------------|----+
///              |                        ^                     |
///              |                        |                     |
///              +------------------------+                     |
///                                                             |
///      +------------------------------------------------------+
///      |
///      V
/// +--------------------+-------------------------------------------+
/// | +----------------+ | +---------------------------------------+ |
/// | | next | n_bytes | | |    OVERFLOW PAYLOAD (FULL PAGE)       | |
/// | +--|---+---------+ | +---------------------------------------+ |
/// +----|---------------+-------------------------------------------+
///      |
///      V
/// +--------------------+-------------------------------------------+
/// | +----------------+ | +------------------------+                |
/// | | next | n_bytes | | | REMAINING OVF PAYLOAD  |                |
/// | +------+---------+ | +------------------------+                |
/// +--------------------+-------------------------------------------+
/// OVERFLOW PAGE                  OVERFLOW PAGE CONTENT
/// HEADER                    (stores as many bytes as needed)
/// ```
///
/// When a [`CellHeader::is_overflow`] equals `true` then the last 4 bytes of
/// the [`Cell`] payload point to the first overflow page.
#[derive(Debug, PartialEq, Clone)]
pub(crate) struct OverflowPage {
    /// Page number.
    pub number: PageNumber,
    /// In-memory page buffer.
    buffer: BufferWithHeader<OverflowPageHeader>,
}

/// We can reuse the overflow page to represent free pages since we only need
/// a linked list of pages.
pub(crate) type FreePage = OverflowPage;

impl InitEmptyPage for OverflowPage {
    fn init(number: PageNumber, size: usize) -> Self {
        Self {
            number,
            buffer: BufferWithHeader::alloc(size),
        }
    }
}

impl AsRef<[u8]> for OverflowPage {
    fn as_ref(&self) -> &[u8] {
        self.buffer.as_ref()
    }
}

impl AsMut<[u8]> for OverflowPage {
    fn as_mut(&mut self) -> &mut [u8] {
        self.buffer.as_mut()
    }
}

impl OverflowPage {
    /// Total space that can be used for overflow payloads.
    pub fn usable_space(page_size: usize) -> u16 {
        BufferWithHeader::<OverflowPageHeader>::usable_space(page_size)
    }

    /// Returns a reference to the header.
    pub fn header(&self) -> &OverflowPageHeader {
        self.buffer.header()
    }

    /// Returns a mutable reference to the header.
    pub fn header_mut(&mut self) -> &mut OverflowPageHeader {
        self.buffer.header_mut()
    }

    /// Returns a mutable reference to the entire content buffer.
    pub fn content_mut(&mut self) -> &mut [u8] {
        self.buffer.content_mut()
    }

    /// Returns a read-only reference to the payload (not the entire content).
    pub fn payload(&self) -> &[u8] {
        &self.buffer.content()[..self.header().num_bytes as usize]
    }
}

/// Database file header.
///
/// This is located at the beginning of the DB file and is used by the pager to
/// keep track of free pages and other metadata.
#[derive(Debug, Clone, Copy)]
#[repr(C, align(8))]
pub(crate) struct DbHeader {
    /// Magic number at the beginning of the file.
    pub magic: u32,
    /// Page size used for this DB file.
    pub page_size: u32,
    /// Number of pages in the file (both free and used).
    pub total_pages: u32,
    /// Number of free pages.
    pub free_pages: u32,
    /// First free page in the freelist.
    pub first_free_page: PageNumber,
    /// Last free page in the freelist.
    pub last_free_page: PageNumber,
}

/// The first page of the DB file (offset 0) is a special case since it contains
/// an additional header with metadata.
///
/// As such, this page has less usable space for [`Cell`] instances. Note that
/// the BTree balancing algorithm should always operate on pages of the same
/// size, but since root nodes are special cases this particular page doesn't
/// introduce any bugs. The alternative would be to use an entire page to store
/// just the [`DbHeader`], which doesn't make much sense if we peek a reasonable
/// page size of 4096 or above.
#[derive(Debug, PartialEq, Clone)]
pub(crate) struct PageZero {
    /// Page buffer.
    buffer: BufferWithHeader<DbHeader>,
    /// Inner BTree slotted page. Marked as manually drop to ensure the safety
    /// conditions of [`BufferWithHeader::new_in`].
    page: ManuallyDrop<Page>,
}

impl InitEmptyPage for PageZero {
    fn init(number: PageNumber, size: usize) -> Self {
        let mut buffer = BufferWithHeader::new(
            size,
            DbHeader {
                magic: MAGIC,
                page_size: size as _,
                total_pages: 1,
                free_pages: 0,
                first_free_page: 0,
                last_free_page: 0,
            },
        );

        // SAFETY: `new_in` requires two guarantees which we meet as follows:
        //
        // 1. There is no way to use the wrapped buffer after the main one is
        // dropped because we own both of them and they're gonna live as long
        // as this struct.
        //
        // 2. The wrapped buffer can never be dropped because we're using
        // ManuallyDrop to store the Page instance.
        let page_buffer = unsafe {
            BufferWithHeader::<PageHeader>::new_in(
                &mut buffer,
                PageHeader::new(size - mem::size_of::<DbHeader>()),
            )
        };

        let page = ManuallyDrop::new(Page {
            number,
            buffer: page_buffer,
            overflow: BinaryHeap::new(),
        });

        Self { buffer, page }
    }
}

impl PageZero {
    /// Read-only reference to the database file header.
    pub fn header(&self) -> &DbHeader {
        self.buffer.header()
    }

    /// Mutable reference to the database file header.
    pub fn header_mut(&mut self) -> &mut DbHeader {
        self.buffer.header_mut()
    }

    /// Read-only references to the contained slotted page.
    pub fn as_btree_page(&self) -> &Page {
        &self.page
    }

    /// Mutable reference to the inner slotted page.
    pub fn as_btree_page_mut(&mut self) -> &mut Page {
        &mut self.page
    }
}

impl AsRef<[u8]> for PageZero {
    fn as_ref(&self) -> &[u8] {
        self.buffer.as_ref()
    }
}

impl AsMut<[u8]> for PageZero {
    fn as_mut(&mut self) -> &mut [u8] {
        self.buffer.as_mut()
    }
}

/// Serves as a wrapper to hold multiple types of pages.
///
/// See [`crate::paging::pager::Pager::get_as`] for details as to why we need
/// this.
#[derive(Debug, PartialEq, Clone)]
pub(crate) enum MemPage {
    Zero(PageZero),
    Overflow(OverflowPage),
    Btree(Page),
}

impl MemPage {
    /// Returns the disk page number.
    pub fn number(&self) -> PageNumber {
        match self {
            Self::Zero(_) => 0,
            Self::Overflow(page) => page.number,
            Self::Btree(page) => page.number,
        }
    }

    /// Returns `true` if the page is in overflow state.
    pub fn is_overflow(&self) -> bool {
        match self {
            Self::Btree(page) => page.is_overflow(),
            Self::Zero(page_zero) => page_zero.as_btree_page().is_overflow(),
            _ => false,
        }
    }
}

impl AsRef<[u8]> for MemPage {
    fn as_ref(&self) -> &[u8] {
        match self {
            Self::Zero(page) => page.as_ref(),
            Self::Overflow(page) => page.as_ref(),
            Self::Btree(page) => page.as_ref(),
        }
    }
}

impl From<Page> for MemPage {
    fn from(page: Page) -> MemPage {
        MemPage::Btree(page)
    }
}

impl From<OverflowPage> for MemPage {
    fn from(page: OverflowPage) -> MemPage {
        MemPage::Overflow(page)
    }
}

impl From<PageZero> for MemPage {
    fn from(page: PageZero) -> MemPage {
        MemPage::Zero(page)
    }
}

/// WARNING: Verbose duplicated code ahead :(
/// See [`crate::paging::pager::Pager::get_as`] for details.
/// We should probably use a macro to generate this but there are only 3 types
/// of pages for now so it's not that bad.

impl<'p> TryFrom<&'p MemPage> for &'p Page {
    type Error = String;

    fn try_from(mem_page: &'p MemPage) -> Result<Self, Self::Error> {
        match mem_page {
            MemPage::Btree(page) => Ok(page),
            MemPage::Zero(page_zero) => Ok(page_zero.as_btree_page()),
            other => Err(format!("attempt to convert {other:?} into Page")),
        }
    }
}

impl<'p> TryFrom<&'p mut MemPage> for &'p mut Page {
    type Error = String;

    fn try_from(mem_page: &'p mut MemPage) -> Result<Self, Self::Error> {
        match mem_page {
            MemPage::Btree(page) => Ok(page),
            MemPage::Zero(page_zero) => Ok(page_zero.as_btree_page_mut()),
            other => Err(format!("attempt to convert {other:?} into Page")),
        }
    }
}

impl<'p> TryFrom<&'p MemPage> for &'p PageZero {
    type Error = String;

    fn try_from(mem_page: &'p MemPage) -> Result<Self, Self::Error> {
        match mem_page {
            MemPage::Zero(page_zero) => Ok(page_zero),
            other => Err(format!("attempt to convert {other:?} into PageZero")),
        }
    }
}

impl<'p> TryFrom<&'p mut MemPage> for &'p mut PageZero {
    type Error = String;

    fn try_from(mem_page: &'p mut MemPage) -> Result<Self, Self::Error> {
        match mem_page {
            MemPage::Zero(page_zero) => Ok(page_zero),
            other => Err(format!("attempt to convert {other:?} into PageZero")),
        }
    }
}

impl<'p> TryFrom<&'p MemPage> for &'p OverflowPage {
    type Error = String;

    fn try_from(mem_page: &'p MemPage) -> Result<Self, Self::Error> {
        match mem_page {
            MemPage::Overflow(page) => Ok(page),
            other => Err(format!("attempt to convert {other:?} into OverflowPage")),
        }
    }
}

impl<'p> TryFrom<&'p mut MemPage> for &'p mut OverflowPage {
    type Error = String;

    fn try_from(mem_page: &'p mut MemPage) -> Result<Self, Self::Error> {
        match mem_page {
            MemPage::Overflow(page) => Ok(page),
            other => Err(format!("attempt to convert {other:?} into OverflowPage")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn variable_size_cells(sizes: &[usize]) -> Vec<Cell> {
        sizes
            .iter()
            .enumerate()
            .map(|(i, size)| Cell::new(vec![i as u8 + 1; *size]))
            .collect()
    }

    fn fixed_size_cells(size: usize, amount: usize) -> Vec<Cell> {
        variable_size_cells(&vec![size; amount])
    }

    struct Builder {
        size: usize,
        cells: Vec<Cell>,
    }

    impl Builder {
        fn new() -> Self {
            Self {
                size: MIN_PAGE_SIZE,
                cells: Vec::new(),
            }
        }

        fn size(mut self, size: usize) -> Self {
            self.size = size;
            self
        }

        fn cells(mut self, cells: Vec<Cell>) -> Self {
            self.cells = cells;
            self
        }

        fn build(self) -> (Page, Vec<Cell>) {
            let mut page = Page::init(0, self.size);
            page.push_all(self.cells.clone());

            (page, self.cells)
        }
    }

    impl Page {
        fn push_all(&mut self, cells: Vec<Cell>) {
            cells.into_iter().for_each(|cell| self.push(cell));
        }

        fn builder() -> Builder {
            Builder::new()
        }
    }

    /// # Arguments
    ///
    /// * `page` - [`Page`] instance.
    ///
    /// * `cells` - List of cells that have been inserted in the page
    /// (in order). Necessary for checking data corruption.
    fn compare_consecutive_offsets(page: &Page, cells: &Vec<Cell>) {
        let mut expected_offset = page.size();
        for (i, cell) in cells.iter().enumerate() {
            expected_offset -= cell.total_size() as usize;
            assert_eq!(page.slot_array()[i], expected_offset as u16);
            assert_eq!(page.cell(i as u16), cell);
        }
    }

    /// Same as [`compare_consecutive_offsets`] but only checks the cell values,
    /// not the offsets.
    fn compare_cells(page: &Page, cells: &Vec<Cell>) {
        for (i, cell) in cells.iter().enumerate() {
            assert_eq!(page.cell(i as _), cell);
        }
    }

    #[test]
    fn buffer_with_header() {
        const CONTENT_SIZE: usize = 24;

        let mut buf = BufferWithHeader::new(
            mem::size_of::<OverflowPageHeader>() + CONTENT_SIZE,
            OverflowPageHeader {
                next: 0,
                num_bytes: 0,
            },
        );

        buf.header_mut().num_bytes = CONTENT_SIZE as u16;
        buf.content_mut().fill(8);

        assert_eq!(buf.header().num_bytes, CONTENT_SIZE as u16);
        assert_eq!(buf.content(), &[8; CONTENT_SIZE]);
    }

    // Some important notes to keep in mind before you read the code below:
    // - For pages of size 512, Page::max_payload_size() equals 112.

    #[test]
    fn push_fixed_size_cells() {
        let (page, cells) = Page::builder()
            .size(512)
            .cells(fixed_size_cells(32, 3))
            .build();

        assert_eq!(page.header().num_slots, cells.len() as u16);
        compare_consecutive_offsets(&page, &cells);
    }

    #[test]
    fn push_variable_size_cells() {
        let (page, cells) = Page::builder()
            .size(512)
            .cells(variable_size_cells(&[64, 32, 80]))
            .build();

        assert_eq!(page.header().num_slots, cells.len() as u16);
        compare_consecutive_offsets(&page, &cells);
    }

    #[test]
    fn delete_slot() {
        let (mut page, mut cells) = Page::builder()
            .size(512)
            .cells(fixed_size_cells(32, 3))
            .build();

        let expected_offsets = [page.slot_array()[0], page.slot_array()[2]];

        page.remove(1);
        cells.remove(1);

        assert_eq!(page.header().num_slots, 2);
        assert_eq!(page.slot_array(), expected_offsets);
        compare_cells(&page, &cells);
    }

    #[test]
    fn defragment() {
        let (mut page, mut cells) = Page::builder()
            .size(512)
            .cells(variable_size_cells(&[24, 64, 32, 72, 8]))
            .build();

        for i in [1, 2] {
            page.remove(i as u16);
            cells.remove(i);
        }

        page.defragment();

        assert_eq!(page.header().num_slots, 3);
        compare_consecutive_offsets(&page, &cells);
    }

    #[test]
    fn unaligned_content() {
        let (page, cells) = Page::builder()
            .size(512)
            .cells(variable_size_cells(&[7, 19, 20]))
            .build();

        compare_consecutive_offsets(&page, &cells);

        // Check padding
        for i in 0..cells.len() {
            assert_eq!(
                page.cell(i as u16).content.len(),
                Cell::aligned_size_of(&cells[i].content) as usize
            );
        }
    }

    #[test]
    fn insert_defragmenting() {
        let (mut page, mut cells) = Page::builder()
            .size(512)
            .cells(variable_size_cells(&[64, 112, 112, 112]))
            .build();

        page.remove(1);
        cells.remove(1);

        let new_cell = Cell::new(vec![4; 112]);

        cells.push(new_cell.clone());
        page.push(new_cell);

        compare_consecutive_offsets(&page, &cells);
    }

    #[test]
    fn replace_cell_in_place() {
        let (mut page, mut cells) = Page::builder()
            .size(512)
            .cells(variable_size_cells(&[64, 32, 112]))
            .build();

        let new_cell = Cell::new(vec![4; 32]);
        cells[1] = new_cell.clone();

        page.replace(1, new_cell);

        compare_consecutive_offsets(&page, &cells);
    }

    #[test]
    fn replace_cell_removing_previous() {
        let (mut page, mut cells) = Page::builder()
            .size(512)
            .cells(variable_size_cells(&[64, 96]))
            .build();

        let new_cell = Cell::new(vec![4; 96]);

        let expected_offset = page.size()
            - (cells.iter().map(Cell::total_size).sum::<u16>() + new_cell.total_size()) as usize;

        cells[0] = new_cell.clone();

        page.replace(0, new_cell);

        assert_eq!(page.header().num_slots, 2);
        assert_eq!(page.slot_array()[0], expected_offset as u16);
        compare_cells(&page, &cells);
    }

    #[test]
    fn drain() {
        let (mut page, mut cells) = Page::builder()
            .size(512)
            .cells(fixed_size_cells(32, 4))
            .build();

        let expected_offsets = [page.slot_array()[0], page.slot_array()[3]];

        assert_eq!(
            page.drain(1..=2).collect::<Vec<_>>(),
            cells.drain(1..=2).collect::<Vec<_>>()
        );

        assert_eq!(page.header().num_slots, 2);
        assert_eq!(page.slot_array(), expected_offsets);
        compare_cells(&page, &cells);
    }

    #[test]
    fn insert_with_overflow() {
        let cell_size = 32;
        let page_size = 512;

        let num_cells_in_page =
            Page::usable_space(page_size) / Cell::new(vec![0; cell_size]).storage_size();

        let (mut page, mut cells) = Page::builder()
            .size(512)
            .cells(fixed_size_cells(cell_size, num_cells_in_page as usize))
            .build();

        let mut cell_data = num_cells_in_page + 1;
        for i in [1, 3] {
            let new_cell = Cell::new(vec![cell_data as u8; cell_size]);
            cells.insert(i, new_cell.clone());
            page.insert(i as _, new_cell);
            cell_data += 1;
        }

        assert_eq!(page.overflow.len(), 2);
        assert_eq!(page.len(), num_cells_in_page + 2);
        assert_eq!(page.drain(..).collect::<Vec<_>>(), cells);
    }
}
