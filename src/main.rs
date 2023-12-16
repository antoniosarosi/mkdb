mod btree;
mod node;
mod pager;

use std::io;

use btree::BTree;

fn main() -> io::Result<()> {
    let mut btree = BTree::new_at("btree.bin", 48)?;
    for i in 1..=35 {
        btree.insert(i, 1000 + i)?;
    }

    eprintln!("BTree order: {}", btree.order());
    eprintln!("Number of nodes: {}", btree.len());
    eprintln!("Get check: btree.get({}) = {}", 5, btree.get(5)?.unwrap());

    eprintln!(
        "Remove check: btree.remove({}) = {}",
        46,
        btree.remove(46)?.unwrap()
    );
    eprintln!("BTree in JSON format goes to STDOUT\n");

    println!("{}", btree.json()?);

    Ok(())
}
