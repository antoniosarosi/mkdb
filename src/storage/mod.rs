//! Disk storage data structures.

mod btree;
pub(crate) mod page;

pub(crate) use btree::{BTree, BytesCmp, FixedSizeMemCmp, DEFAULT_BALANCE_SIBLINGS_PER_SIDE};
