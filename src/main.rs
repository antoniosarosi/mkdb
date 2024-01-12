#![feature(non_null_convenience)]
#![feature(debug_closure_helpers)]
#![feature(allocator_api)]

mod os;
mod paging;
mod sql;
mod storage;

use std::io;

fn main() -> io::Result<()> {
    // let mut btree = storage::BTree::new_at_path("btree.bin", 72)?;
    // for i in 1_u32..=46 {
    //     btree.insert(&i.to_be_bytes())?;
    // }

    // eprintln!(
    //     "Get check: btree.get({}) = {:?}",
    //     5,
    //     btree.get(&5_u32.to_be_bytes())?
    // );

    // eprintln!(
    //     "Remove check: btree.remove({}) = {:?}",
    //     46,
    //     btree.remove(&46_u32.to_be_bytes())?
    // );
    // eprintln!("BTree in JSON format goes to STDOUT\n");

    // println!("{}", btree.json()?);

    Ok(())
}
