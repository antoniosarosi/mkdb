//! Paging implementation.
//!
//! This module contains all the data structures that deal with reading and
//! writing pages from/to disk and caching pages in memory. The high level
//! public API is [`pager::Pager`], which abstracts away both the disk and the
//! cache subsystem.

pub(super) mod cache;

pub(crate) mod io;
pub(crate) mod pager;
