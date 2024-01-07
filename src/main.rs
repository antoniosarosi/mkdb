#![feature(non_null_convenience)]
#![feature(ptr_metadata)]
#![feature(debug_closure_helpers)]
#![feature(pointer_is_aligned)]

mod btree;
mod os;
mod paging;
mod sql;

use std::io;

fn main() -> io::Result<()> {
    Ok(())
}
