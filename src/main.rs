mod btree;
mod pager;

use std::io;

use btree::BTree;

fn main() -> io::Result<()> {
    let mut btree = BTree::new_at("btree.bin", 48)?;
    for i in 1..=12 {
        btree.insert(i, 1000 + i)?;
    }

    eprintln!("BTree degree: {}", btree.degree());
    eprintln!("Number of nodes: {}", btree.len());
    eprintln!("Sanity check: btree.get({}) = {}", 5, btree.get(5)?);
    eprintln!("BTree in JSON format goes to STDOUT\n");

    println!("{}", btree.json()?);

    Ok(())
}
