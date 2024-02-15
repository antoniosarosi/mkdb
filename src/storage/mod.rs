//! Disk storage data structures.

mod btree;
pub(crate) mod page;

pub(crate) use btree::{BTree, FixedSizeMemCmp, DEFAULT_BALANCE_SIBLINGS_PER_SIDE};

use crate::paging::pager::PageNumber;

/// Magic number at the beginning of the database file.
pub(crate) const MAGIC: u32 = 0xB74EE;

/// Database file header.
///
/// This is located at the beginning of the DB file and is used by the pager to
/// keep track of free pages and other metadata.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub(crate) struct Header {
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
