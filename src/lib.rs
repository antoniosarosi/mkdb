//! MKDB toy database.

#![feature(non_null_convenience)]
#![feature(debug_closure_helpers)]
#![feature(allocator_api)]
#![feature(map_try_insert)]
#![feature(vec_into_raw_parts)]
#![feature(set_ptr_value)]
#![feature(slice_ptr_get)]
#![feature(get_many_mut)]
#![feature(map_many_mut)]
#![feature(trait_alias)]
#![feature(pointer_is_aligned)]
#![feature(buf_read_has_data_left)]

mod db;
mod os;
mod paging;
mod pool;
mod query;
mod sql;
mod storage;
mod vm;

pub mod tcp;

pub use db::{DbError, QuerySet};
pub use storage::tuple::deserialize;

pub type Result<T> = std::result::Result<T, DbError>;
