use std::{
    cmp::Ordering,
    fs::File,
    io,
    io::{Read, Seek, Write},
    mem,
    os::unix::prelude::MetadataExt,
};

#[derive(Debug, Eq)]
struct Entry {
    key: u32,
    value: u32,
}

impl PartialEq for Entry {
    fn eq(&self, other: &Self) -> bool {
        self.key.eq(&other.key)
    }
}

impl PartialOrd for Entry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Entry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.key.cmp(&other.key)
    }
}

#[derive(Debug)]
struct Node {
    block: u32,
    entries: Vec<Entry>,
    children: Vec<u32>,
}

impl Node {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            children: Vec::new(),
            block: 0,
        }
    }
}

/// `degree`: Minimum number of children per node (except root).
///
/// # Properties
///
/// - Min keys:     `degree - 1`
/// - Min children: `degree`
/// - Max keys:     `2 * degree - 1`
/// - Max children: `2 * degree`
///
/// File format (suppose block size = 4096):
///
/// +---------+--------+--------+--------+--------+     +--------+--------+
/// | KL | CL |   K1   |   V1   |   K2   |   V2   | ... |   C1   |   C2   | ...
/// +---------+--------+--------+--------+--------+     +--------+--------+
///   2    2      4        4        4        4              4        4
///
/// - KL: Keys Length
/// - CL: Children Length
/// - K: Key
/// - V: Value
/// - C: Child
///
/// Children pointers start at 4 + 8 * (2 * degree - 1).
#[derive(Debug)]
struct BTree {
    file: File,
    block_size: u32,
    num_blocks: u32,
    degree: usize,
}

impl BTree {
    pub fn new(file_name: String) -> io::Result<Self> {
        let file = File::options().read(true).write(true).open(file_name)?;
        let metadata = file.metadata()?;

        let block_size = 48; // Makes degree = 2 for debugging.
                             // let block_size = metadata.blksize() as usize;
        let max_children = block_size / (mem::size_of::<Entry>() + 4);
        let degree = max_children / 2;

        let num_blocks = metadata.len() as usize / block_size;

        Ok(BTree {
            file,
            degree,
            block_size: block_size as u32,
            num_blocks: num_blocks as u32,
        })
    }

    pub fn insert(&mut self, key: u32, value: u32) -> io::Result<()> {
        let mut root = self.read_node(0)?;

        if root.entries.len() == (2 * self.degree) - 1 {
            let mut old_root = Node::new();
            old_root.entries.extend(root.entries.drain(0..));
            old_root.children.extend(root.children.drain(0..));

            old_root.block = self.free_block();
            root.children.push(old_root.block);

            self.write_node(&mut old_root)?;
        }

        self.insert_into(&mut root, Entry { key, value })
    }

    fn insert_into(&mut self, node: &mut Node, entry: Entry) -> io::Result<()> {
        let search = node.entries.binary_search(&entry);

        if let Ok(index) = search {
            node.entries[index].value = entry.value;
            return self.write_node(node);
        }

        let index = search.unwrap_err();

        if node.children.is_empty() {
            node.entries.insert(index, entry);
            return self.write_node(node);
        }

        let mut next_node = self.read_node(node.children[index])?;
        if next_node.entries.len() == (2 * self.degree) - 1 {
            self.split_child(node, index)?;
            if entry.key > node.entries[index].key {
                next_node = self.read_node(node.children[index + 1])?;
            }
        }

        self.insert_into(&mut next_node, entry)
    }

    fn split_child(&mut self, parent: &mut Node, index: usize) -> io::Result<()> {
        let mut target_node = self.read_node(parent.children[index])?;

        let mut new_node = Node::new();
        new_node.block = self.free_block();

        // Move keys greater than the median into the new node.
        new_node.entries.extend(
            target_node
                .entries
                .drain(self.degree..(2 * self.degree - 1)),
        );

        // Move median key into parent.
        parent
            .entries
            .insert(index, target_node.entries.remove(self.degree - 1));

        // If the target node is not a leaf node, update children pointers.
        if !target_node.children.is_empty() {
            new_node
                .children
                .extend(target_node.children.drain(self.degree..2 * self.degree));
        }

        // Insert new node pointer into parent.
        parent.children.insert(index + 1, new_node.block);

        self.write_node(parent)?;
        self.write_node(&mut target_node)?;
        self.write_node(&mut new_node)?;

        Ok(())
    }

    fn free_block(&self) -> u32 {
        self.num_blocks as u32
    }

    fn seek(&mut self, block: u32) -> io::Result<u64> {
        let offset = (self.block_size * block) as u64;
        self.file.seek(io::SeekFrom::Start(offset))
    }

    fn read_node(&mut self, block: u32) -> io::Result<Node> {
        self.seek(block)?;
        let mut buf = Vec::with_capacity(self.block_size as usize);
        buf.resize(buf.capacity(), 0);

        self.file.read(&mut buf[..])?;

        let mut node = Node::new();
        node.block = block;

        let mut i = 4;

        for _ in 0..u16::from_be_bytes(buf[..2].try_into().unwrap()) {
            let key = u32::from_be_bytes(buf[i..i + 4].try_into().unwrap());
            let value = u32::from_be_bytes(buf[i + 4..i + 8].try_into().unwrap());
            node.entries.push(Entry { key, value });
            i += 8;
        }

        i = mem::size_of::<u16>() * 2 + mem::size_of::<Entry>() * (2 * self.degree - 1);

        for _ in 0..u16::from_be_bytes(buf[2..4].try_into().unwrap()) {
            node.children
                .push(u32::from_be_bytes(buf[i..i + 4].try_into().unwrap()));
            i += 4;
        }

        Ok(node)
    }

    fn write_node(&mut self, node: &mut Node) -> io::Result<()> {
        self.seek(node.block)?;

        let mut block = Vec::<u8>::with_capacity(self.block_size as usize);
        block.resize(block.capacity(), 0);

        block[..2].copy_from_slice(&(node.entries.len() as u16).to_be_bytes());
        block[2..4].copy_from_slice(&(node.children.len() as u16).to_be_bytes());

        let mut i = 4;

        for entry in &node.entries {
            block[i..i + 4].copy_from_slice(&entry.key.to_be_bytes());
            block[i + 4..i + 8].copy_from_slice(&entry.value.to_be_bytes());
            i += 8;
        }

        i = mem::size_of::<u16>() * 2 + mem::size_of::<Entry>() * (2 * self.degree - 1);

        for child in &node.children {
            block[i..i + 4].copy_from_slice(&(*child as u32).to_be_bytes());
            i += 4;
        }

        self.file.write(&block[..])?;

        if node.block == self.num_blocks {
            self.num_blocks += 1;
        }

        Ok(())
    }
}

impl BTree {
    pub fn json(&mut self, node: &Node) -> io::Result<String> {
        let mut string = format!("{{\"block\":{},\"entries\":[", node.block);

        if node.entries.len() >= 1 {
            // string.push_str(&format!("{}", node.entries[0].key));
            // let last = node.entries.last().unwrap();
            // if last.key != node.entries[0].key {
            //     string.push_str(&format!(",{}", last.key));
            // }
            let Entry { key, value } = node.entries[0];
            string.push_str(&format!("{{\"key\":{key},\"value\":{value}}}"));

            for Entry { key, value } in &node.entries[1..] {
                string.push(',');
                string.push_str(&format!("{{\"key\":{key},\"value\":{value}}}"));
            }
        }

        string.push_str("],\"children\":[");

        if node.children.len() >= 1 {
            let subtree = self.read_node(node.children[0])?;
            string.push_str(&self.json(&subtree)?);

            for child in &node.children[1..] {
                string.push(',');
                let subtree = self.read_node(*child)?;
                string.push_str(&self.json(&subtree)?);
            }
        }

        string.push(']');
        string.push('}');

        Ok(string)
    }
}

fn main() -> io::Result<()> {
    let mut btree = BTree::new("btree.bin".into())?;
    for i in 1..=12 {
        btree.insert(i, 1000 + i)?;
    }
    let root = btree.read_node(0)?;
    println!("{}", btree.json(&root)?);

    Ok(())
}
