mod btree;
mod pager;

use std::io;

use btree::BTree;

fn main() -> io::Result<()> {
    let mut btree = BTree::new_at("btree.bin", 48)?;
    for i in 1..=46 {
        btree.insert(i, 1000 + i)?;
        // println!("{}", btree.json()?);
    }

    eprintln!("BTree order: {}", btree.order());
    eprintln!("Number of nodes: {}", btree.len());
    eprintln!("Sanity check: btree.get({}) = {}", 5, btree.get(5)?);
    eprintln!("BTree in JSON format goes to STDOUT\n");

    println!("{}", btree.json()?);

    Ok(())
}
