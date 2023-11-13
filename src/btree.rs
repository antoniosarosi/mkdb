use std::{
    cmp::Ordering,
    fs::File,
    io,
    io::{Read, Seek, Write},
    mem,
    os::unix::prelude::MetadataExt,
    path::Path,
};

use crate::pager::Pager;

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
    page: u32,
    entries: Vec<Entry>,
    children: Vec<u32>,
}

impl Node {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            children: Vec::new(),
            page: 0,
        }
    }

    pub fn new_at(page: usize) -> Self {
        let mut node = Self::new();
        node.page = page as u32;
        node
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
/// Page format (suppose page size = 4096):
///
/// ```text
/// +---------+--------+--------+--------+--------+     +--------+--------+
/// | EL | CL |   K1   |   V1   |   K2   |   V2   | ... |   C1   |   C2   | ...
/// +---------+--------+--------+--------+--------+     +--------+--------+
///   2    2      4        4        4        4              4        4
/// ````
///
/// - `KL`: Keys Length
/// - `CL`: Children Length
/// - `K`: Key
/// - `V`: Value
/// - `C`: Child Pointer
///
/// Children pointers start at `4 + 8 * (2 * degree - 1)`. See
/// [`optimal_degree_for`] for degree computations.
pub(crate) struct BTree<F> {
    pager: Pager<F>,
    degree: usize,
    len: usize,
}

/// See [`BTree`].
fn optimal_degree_for(page_size: usize) -> usize {
    let max_children = page_size / (mem::size_of::<Entry>() + mem::size_of::<u32>());

    max_children / 2
}

impl<F> BTree<F> {
    #[allow(dead_code)]
    pub fn new(file: F, page_size: usize, block_size: usize) -> io::Result<Self> {
        let pager = Pager::new(file, page_size, block_size);
        let degree = optimal_degree_for(page_size);
        let len = 0;

        Ok(Self { pager, degree, len })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn degree(&self) -> usize {
        self.degree
    }

    fn free_page(&self) -> usize {
        self.len
    }

    fn max_keys(&self) -> usize {
        (2 * self.degree) - 1
    }
}

impl BTree<File> {
    pub fn new_at<P: AsRef<Path>>(path: P, page_size: usize) -> io::Result<Self> {
        let file = File::options().read(true).write(true).open(path)?;
        let metadata = file.metadata()?;

        if !metadata.is_file() {
            return Err(io::Error::new(io::ErrorKind::Unsupported, "Not a file"));
        }

        // TODO: Windows
        let block_size = metadata.blksize() as usize;

        // TODO: Add magic number & header to file.
        let pager = Pager::new(file, page_size, block_size);
        let degree = optimal_degree_for(page_size);
        let len = metadata.len() as usize / page_size;

        Ok(Self { pager, degree, len })
    }
}

impl<F: Seek + Read + Write> BTree<F> {
    pub fn insert(&mut self, key: u32, value: u32) -> io::Result<()> {
        let mut root = self.read_node(0)?;

        if root.entries.len() == self.max_keys() {
            let mut old_root = Node::new_at(self.free_page());
            old_root.entries.extend(root.entries.drain(0..));
            old_root.children.extend(root.children.drain(0..));

            root.children.push(old_root.page);

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
        if next_node.entries.len() == self.max_keys() {
            self.split_child(node, index)?;
            if entry.key > node.entries[index].key {
                next_node = self.read_node(node.children[index + 1])?;
            }
        }

        self.insert_into(&mut next_node, entry)
    }

    fn split_child(&mut self, parent: &mut Node, index: usize) -> io::Result<()> {
        let mut target_node = self.read_node(parent.children[index])?;

        let mut new_node = Node::new_at(self.free_page());

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
        parent.children.insert(index + 1, new_node.page);

        self.write_node(parent)?;
        self.write_node(&mut target_node)?;
        self.write_node(&mut new_node)?;

        Ok(())
    }

    pub fn get(&mut self, key: u32) -> io::Result<u32> {
        let root = self.read_node(0)?;
        self.find(&root, key)
    }

    fn find(&mut self, node: &Node, key: u32) -> io::Result<u32> {
        match node.entries.binary_search(&Entry { key, value: 0 }) {
            Ok(index) => Ok(node.entries[index].value),

            Err(index) => {
                let next_node = self.read_node(node.children[index])?;
                self.find(&next_node, key)
            }
        }
    }

    fn read_node(&mut self, page: u32) -> io::Result<Node> {
        let buf = self.pager.read_page(page as usize)?;

        let mut node = Node::new_at(page as usize);

        let mut i = 4;

        for _ in 0..u16::from_be_bytes(buf[..2].try_into().unwrap()) {
            let key = u32::from_be_bytes(buf[i..i + 4].try_into().unwrap());
            let value = u32::from_be_bytes(buf[i + 4..i + 8].try_into().unwrap());
            node.entries.push(Entry { key, value });
            i += 8;
        }

        i = mem::size_of::<u16>() * 2 + mem::size_of::<Entry>() * self.max_keys();

        for _ in 0..u16::from_be_bytes(buf[2..4].try_into().unwrap()) {
            node.children
                .push(u32::from_be_bytes(buf[i..i + 4].try_into().unwrap()));
            i += 4;
        }

        Ok(node)
    }

    fn write_node(&mut self, node: &Node) -> io::Result<()> {
        let mut page = vec![0u8; self.pager.page_size()];

        page[..2].copy_from_slice(&(node.entries.len() as u16).to_be_bytes());
        page[2..4].copy_from_slice(&(node.children.len() as u16).to_be_bytes());

        let mut i = 4;

        for entry in &node.entries {
            page[i..i + 4].copy_from_slice(&entry.key.to_be_bytes());
            page[i + 4..i + 8].copy_from_slice(&entry.value.to_be_bytes());
            i += 8;
        }

        i = mem::size_of::<u16>() * 2 + mem::size_of::<Entry>() * self.max_keys();

        for child in &node.children {
            page[i..i + 4].copy_from_slice(&(*child as u32).to_be_bytes());
            i += 4;
        }

        self.pager.write_page(node.page as usize, &page)?;

        if node.page as usize == self.len {
            self.len += 1;
        }

        Ok(())
    }

    pub fn json(&mut self) -> io::Result<String> {
        let root = self.read_node(0)?;
        self.to_json(&root)
    }

    fn to_json(&mut self, node: &Node) -> io::Result<String> {
        let mut string = format!("{{\"page\":{},\"entries\":[", node.page);

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
            string.push_str(&self.to_json(&subtree)?);

            for child in &node.children[1..] {
                string.push(',');
                let subtree = self.read_node(*child)?;
                string.push_str(&self.to_json(&subtree)?);
            }
        }

        string.push(']');
        string.push('}');

        Ok(string)
    }
}

#[cfg(test)]
mod tests {
    use std::{error::Error, io};

    use super::BTree;

    /// This is how a Btree of `degree = 2` should look like if we insert values
    /// from 1 to 12 included sequentially:
    ///
    /// ```text
    ///                   +---+
    ///           +-------| 4 |-------+
    ///         /         +---+        \
    ///        /                        \
    ///     +---+                   +--------+
    ///     | 2 |              +--- | 6,8,10 |----+
    ///     +---+             /     +--------+     \
    ///   /       \          /       /      \       \
    /// +---+   +---+     +---+   +---+   +---+   +-------+
    /// | 1 |   | 3 |     | 5 |   | 7 |   | 9 |   | 11,12 |
    /// +---+   +---+     +---+   +---+   +---+   +-------+
    /// ```
    #[test]
    fn insertion() -> Result<(), Box<dyn Error>> {
        let page_size = 48;
        let max_nodes = 9;
        let keys: Vec<u32> = (1..=12).collect();

        let buf = io::Cursor::new(vec![0; page_size * max_nodes]);
        let mut btree = BTree::new(buf, page_size, page_size)?;

        for key in &keys {
            btree.insert(*key, 100 + key)?;
        }

        assert_eq!(btree.degree(), 2);
        assert_eq!(btree.len(), 9);

        for key in &keys {
            assert_eq!(btree.get(*key)?, 100 + key);
        }

        Ok(())
    }
}
