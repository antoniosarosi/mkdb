use std::{
    io,
    io::{Read, Seek, Write},
    os::unix::prelude::MetadataExt,
};

#[derive(Debug)]
struct Node {
    block: usize,
    keys: Vec<usize>,
    children: Vec<usize>,
}

impl Node {
    pub fn new() -> Self {
        Self {
            keys: Vec::new(),
            children: Vec::new(),
            block: 0,
        }
    }

    pub fn with_keys(keys: Vec<usize>) -> Self {
        Self {
            keys,
            children: Vec::new(),
            block: 0,
        }
    }
}

/// Serialize 4 byte unsigned integer (endianness is same as machine).
fn serialize_number(number: usize, buf: &mut [u8]) {
    buf[0] = (number & 0xFF) as u8;
    buf[1] = (number >> 8 & 0xFF) as u8;
    buf[2] = (number >> 16 & 0xFF) as u8;
    buf[3] = (number >> 24 & 0xFF) as u8;
}

/// Deserialize unsigned integeres serialized with [`serialize_number`].
/// TODO: doesn't work if endianness is different than current machine.
fn deserialize_number(buf: &[u8]) -> usize {
    let mut number: usize = 0;
    number |= buf[0] as usize;
    number |= (buf[1] as usize) << 8;
    number |= (buf[2] as usize) << 16;
    number |= (buf[3] as usize) << 24;

    number
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
/// +---------+--------+--------+     +--------+--------+
/// | KL | CL |   K1   |   K2   | ... |   C1   |   C2   | ...
/// +---------+--------+--------+     +--------+--------+
///   2    2      4        4              4        4
///
/// KL: Keys Length
/// CL: Children Length
///
/// Children pointers start at 4096 / 2.
#[derive(Debug)]
struct BTree {
    root: Option<Node>,
    file: std::fs::File,
    block_size: usize,
    num_blocks: usize,
    degree: usize,
}

impl BTree {
    pub fn new(file_name: String) -> io::Result<Self> {
        let mut file = std::fs::File::options()
            .read(true)
            .write(true)
            .open(file_name)?;

        file.write(&[0, 0, 0, 0])?;

        let metadata = file.metadata()?;

        let block_size = metadata.blksize() as usize;
        let capacity = block_size / 4;

        let max_keys = capacity / 2 - 1;
        let degree = (max_keys + 1) / 2;

        let num_blocks = metadata.len() as usize / block_size;

        Ok(BTree {
            root: None,
            file,
            degree,
            block_size,
            num_blocks,
        })
    }

    fn split_child(&mut self, parent: &mut Node, index: usize) -> io::Result<()> {
        let mut target_node = self.read_node(parent.children[index])?;

        let mut new_node = Node::new();
        new_node.block = self.free_block();

        // Move keys greater than the median into the new node.
        new_node
            .keys
            .extend(target_node.keys.drain(self.degree..(2 * self.degree - 1)));

        // Move median key into parent.
        parent
            .keys
            .insert(index, target_node.keys.remove(self.degree - 1));

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

    fn insert_into(&mut self, node: &mut Node, key: usize) -> io::Result<()> {
        if node.children.is_empty() {
            node.keys
                .insert(node.keys.binary_search(&key).unwrap_err(), key);
            self.write_node(node)?;
            return Ok(());
        }

        let index = node.keys.binary_search(&key).unwrap_err();
        let mut next_node = self.read_node(node.children[index])?;
        if next_node.keys.len() == (2 * self.degree) - 1 {
            self.split_child(node, index)?;
            if key > node.keys[index] {
                next_node = self.read_node(node.children[index + 1])?;
            }
        }
        self.insert_into(&mut next_node, key)
    }

    fn free_block(&self) -> usize {
        self.num_blocks
    }

    fn seek(&mut self, block: usize) -> io::Result<u64> {
        let offset = (self.block_size * block) as u64;
        self.file.seek(io::SeekFrom::Start(offset))
    }

    fn read_node(&mut self, block: usize) -> io::Result<Node> {
        self.seek(block)?;
        let mut buf = Vec::with_capacity(self.block_size);
        buf.resize(buf.capacity(), 0);

        self.file.read(&mut buf[..])?;

        let mut node = Node::new();
        node.block = block;

        let mut keys_len = 0;
        keys_len |= buf[0] as usize;
        keys_len |= (buf[1] as usize) << 8;

        let mut children_len = 0;
        children_len |= buf[2] as usize;
        children_len |= (buf[3] as usize) << 8;

        let mut i = 4;

        for _ in 0..keys_len {
            node.keys.push(deserialize_number(&buf[i..i + 4]));
            i += 4;
        }

        i = self.block_size / 2;

        for _ in 0..children_len {
            node.children.push(deserialize_number(&buf[i..i + 4]));
            i += 4;
        }

        Ok(node)
    }

    fn write_node(&mut self, node: &mut Node) -> io::Result<()> {
        self.seek(node.block)?;

        let mut block = Vec::<u8>::with_capacity(self.block_size);
        block.resize(block.capacity(), 0);

        block[0] = (node.keys.len() & 0xFF) as u8;
        block[1] = (node.keys.len() >> 8 & 0xFF) as u8;

        block[2] = (node.children.len() & 0xFF) as u8;
        block[3] = (node.children.len() >> 8 & 0xFF) as u8;

        let mut i = 4;

        for key in &node.keys {
            serialize_number(*key, &mut block[i..i + 4]);
            i += 4;
        }

        i = self.block_size / 2;

        for child in &node.children {
            serialize_number(*child, &mut block[i..i + 4]);
            i += 4;
        }

        self.file.write(&block[..])?;

        if node.block == self.num_blocks {
            self.num_blocks += 1;
        }

        Ok(())
    }

    pub fn insert(&mut self, key: usize) -> io::Result<()> {
        let Some(mut root) = self.root.take() else {
            let mut node = Node::with_keys(vec![key]);
            self.write_node(&mut node)?;
            self.root = Some(node);
            return Ok(());
        };

        if root.keys.len() == (2 * self.degree) - 1 {
            let mut old_root = Node::new();
            old_root.keys.extend(root.keys.drain(0..));
            old_root.children.extend(root.children.drain(0..));

            old_root.block = self.free_block();
            root.children.push(old_root.block);

            self.write_node(&mut old_root)?;
            self.split_child(&mut root, 0)?;
        }

        self.insert_into(&mut root, key)?;
        self.root = Some(root);

        Ok(())
    }
}

impl BTree {
    pub fn json(&mut self, node: &Node) -> io::Result<String> {
        let mut string = format!(
            "{{\"block\":{},\"keys\":[{}, {}],\"children\":[",
            node.block,
            node.keys[0],
            node.keys.last().unwrap()
        );

        if node.children.len() >= 1 {
            let subtree = self.read_node(node.children[0])?;
            string.extend(self.json(&subtree));

            for child in &node.children[1..] {
                string.push(',');
                let subtree = self.read_node(*child)?;
                string.extend(self.json(&subtree));
            }
        }

        string.push(']');
        string.push('}');

        Ok(string)
    }
}

fn main() -> io::Result<()> {
    let mut btree = BTree::new("btree.bin".into())?;
    for i in 1..=1024 {
        btree.insert(i)?;
    }
    let root = btree.read_node(0)?;
    println!("{}", btree.json(&root)?);

    Ok(())
}
