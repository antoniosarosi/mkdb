use std::{
    cmp::{min, Ordering},
    fs::File,
    io,
    io::{Read, Seek, Write},
    mem,
    path::Path,
};

use crate::pager::Pager;

/// Number of siblings per side to examine when balancing.
const BALANCE_SIBLINGS_PER_SIDE: usize = 1;

#[derive(Eq, Copy, Clone)]
struct Entry {
    key: u32,
    value: u32,
}

impl std::fmt::Debug for Entry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.key)
    }
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

/// B*-Tree implementation inspired by "Art of Computer Programming Volume 3:
/// Sorting and Searching" and SQLite 2.X.X
///
/// Terminology
///
/// `order`: Maximum number of children per node (except root).
///
/// # Properties
///
/// - Max children: `order`
/// - Max keys:     `order - 1`
/// - Min children: `(order - 1) * 2 / 3 + 1` (except root)
/// - Min keys:     `(order - 1) * 2 / 3` (except root)
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
/// Children pointers start at `4 + 8 * (order - 1)`. See
/// [`optimal_degree_for`] for degree computations.
pub(crate) struct BTree<F> {
    pager: Pager<F>,
    order: usize,
    len: usize,
}

/// See [`BTree`].
fn optimal_order_for(page_size: usize) -> usize {
    let total_size = mem::size_of::<Entry>() + mem::size_of::<u32>();

    // Calculate how many entries + pointers we can fit.
    let mut order = page_size / total_size;
    let remainder = page_size % total_size;

    // The number of children is always one more than the number of keys, so
    // see if we can fit an extra child in the remaining space.
    if remainder >= mem::size_of::<u32>() {
        order += 1;
    }

    order
}

impl<F> BTree<F> {
    #[allow(dead_code)]
    pub fn new(file: F, page_size: usize, block_size: usize) -> io::Result<Self> {
        let pager = Pager::new(file, page_size, block_size);
        let order = optimal_order_for(page_size);
        let len = 0;

        Ok(Self { pager, order, len })
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn order(&self) -> usize {
        self.order
    }

    #[inline]
    fn max_keys(&self) -> usize {
        self.order() - 1
    }

    fn allocate_page(&mut self) -> usize {
        // TODO: Free list.
        let page = self.len;
        self.len += 1;

        page
    }
}

impl BTree<File> {
    pub fn new_at<P: AsRef<Path>>(path: P, page_size: usize) -> io::Result<Self> {
        let file = File::options()
            .create(true)
            .read(true)
            .write(true)
            .open(&path)?;

        let metadata = file.metadata()?;

        if !metadata.is_file() {
            return Err(io::Error::new(io::ErrorKind::Unsupported, "Not a file"));
        }

        #[cfg(unix)]
        let block_size = {
            use std::os::unix::prelude::MetadataExt;
            metadata.blksize() as usize
        };

        #[cfg(windows)]
        let block_size = unsafe {
            use std::os::windows::ffi::OsStrExt;

            use windows::{
                core::PCWSTR,
                Win32::{Foundation::MAX_PATH, Storage::FileSystem},
            };

            let mut volume = [0u16; MAX_PATH as usize];

            let mut win_file_path = path
                .as_ref()
                .as_os_str()
                .encode_wide()
                .collect::<Vec<u16>>();

            win_file_path.push(0);

            FileSystem::GetVolumePathNameW(PCWSTR::from_raw(win_file_path.as_ptr()), &mut volume)?;

            let mut bytes_per_sector: u32 = 0;
            let mut sectors_per_cluster: u32 = 0;

            FileSystem::GetDiskFreeSpaceW(
                PCWSTR::from_raw(volume.as_ptr()),
                Some(&mut bytes_per_sector),
                Some(&mut sectors_per_cluster),
                None,
                None,
            )?;

            (bytes_per_sector * sectors_per_cluster) as usize
        };

        // TODO: Add magic number & header to file.
        let pager = Pager::new(file, page_size, block_size);
        let order = optimal_order_for(page_size);
        let len = metadata.len() as usize / page_size;

        Ok(Self { pager, order, len })
    }
}

impl<F: Seek + Read + Write> BTree<F> {
    pub fn insert(&mut self, key: u32, value: u32) -> io::Result<()> {
        let root = self.read_node(0)?;

        let mut parents = Vec::new();
        let node = self.insert_into(root, Entry { key, value }, &mut parents)?;

        self.balance(node, parents)
    }

    fn insert_into(
        &mut self,
        mut node: Node,
        entry: Entry,
        parents: &mut Vec<(Node, usize)>,
    ) -> io::Result<Node> {
        let search = node.entries.binary_search(&entry);

        // Key found, swap value and return.
        if let Ok(index) = search {
            node.entries[index].value = entry.value;
            self.write_node(&node)?;
            return Ok(node);
        }

        // Key is not present in current node, get the index where it should be.
        let index = search.unwrap_err();

        // If this node is a leaf node we're done.
        if node.children.is_empty() {
            node.entries.insert(index, entry);
            if node.entries.len() <= self.max_keys() {
                self.write_node(&node)?;
            }
            return Ok(node);
        }

        // If it's not a leaf node, we have to go one level below.
        let next_node = self.read_node(node.children[index])?;
        parents.push((node, index));

        self.insert_into(next_node, entry, parents)
    }

    /// B*-Tree balancing algorithm. This algorithm attempts to delay node
    /// splitting as much as possible and keep all nodes at least 2/3 full
    /// except the root.
    ///
    /// TODO: Explain how.
    fn balance(&mut self, mut node: Node, mut parents: Vec<(Node, usize)>) -> io::Result<()> {
        // Done, this node didn't overflow.
        if node.entries.len() <= self.max_keys() {
            return Ok(());
        }

        // The root overflowed, so it must be split.
        if parents.is_empty() {
            let mut old_root = Node::new_at(self.allocate_page());
            old_root.entries.extend(node.entries.drain(..));
            old_root.children.extend(node.children.drain(..));

            node.children.push(old_root.page);

            parents.push((node, 0));
            node = old_root;
        }

        let (mut parent, index) = parents.pop().unwrap();

        // Find all siblings involved in the balancing algorithm.
        let range = if index == 0 {
            0..=min(BALANCE_SIBLINGS_PER_SIDE * 2, parent.children.len() - 1)
        } else if index == parent.children.len() - 1 {
            index
                .checked_sub(BALANCE_SIBLINGS_PER_SIDE * 2)
                .unwrap_or(0)..=index
        } else {
            index.checked_sub(BALANCE_SIBLINGS_PER_SIDE).unwrap_or(0)
                ..=min(index + BALANCE_SIBLINGS_PER_SIDE, parent.children.len() - 1)
        };

        let mut siblings = Vec::new();
        for i in range {
            if parent.children[i] != node.page {
                siblings.push((self.read_node(parent.children[i])?, i));
            }
        }

        // TODO: Optimize this. Insert nodes at left, then current node, then right.
        siblings.insert(
            siblings
                .binary_search_by_key(&index, |(_, i)| *i)
                .unwrap_err(),
            (node, index),
        );

        // All siblings are full, must split.
        if siblings
            .iter()
            .all(|(sibling, _)| sibling.entries.len() >= self.max_keys())
        {
            // Only two nodes needed for splitting.
            if siblings.len() > 2 {
                if siblings[0].1 == index {
                    siblings.drain(2..);
                } else if siblings.last().unwrap().1 == index {
                    siblings.drain(..siblings.len() - 2);
                } else {
                    siblings.retain(|(_, i)| [index, index + 1].contains(i));
                }
            }

            // Allocate new node.
            let new_node = Node::new_at(self.allocate_page());

            // Prepare terrain for balancing.
            let rightmost_sibling = siblings.last_mut().unwrap();
            let new_node_parent_index = rightmost_sibling.1 + 1;
            parent.entries.insert(
                rightmost_sibling.1,
                rightmost_sibling.0.entries.pop().unwrap(),
            );
            parent.children.insert(new_node_parent_index, new_node.page);

            // Add new node to balancing list.
            siblings.push((new_node, new_node_parent_index));
        }

        // Redistribute entries and children evenly.
        let mut entries_to_balance = Vec::new();
        let mut children_to_balance = Vec::new();

        for (node, _) in &mut siblings {
            entries_to_balance.extend(node.entries.drain(..));
            children_to_balance.extend(node.children.drain(..));
        }

        let chunk_size = entries_to_balance.len() / siblings.len();
        let remainder = entries_to_balance.len() % siblings.len();

        // Swap keys with parent.
        let mut split_point = 0;
        for (current_iter, (_, divider_idx)) in siblings[..siblings.len() - 1].iter().enumerate() {
            let mut balanced_chunk_size = chunk_size;
            if current_iter < remainder {
                balanced_chunk_size += 1;
            }

            split_point += balanced_chunk_size;

            let mut swap_key_idx = split_point;
            if parent.entries[*divider_idx] < entries_to_balance[split_point] {
                swap_key_idx -= 1;
            }

            mem::swap(
                &mut parent.entries[*divider_idx],
                &mut entries_to_balance[swap_key_idx],
            );

            // Sort demoted keys
            while swap_key_idx > 0
                && entries_to_balance[swap_key_idx - 1] > entries_to_balance[swap_key_idx]
            {
                entries_to_balance.swap(swap_key_idx - 1, swap_key_idx);
            }
            while swap_key_idx < entries_to_balance.len() - 1
                && entries_to_balance[swap_key_idx + 1] < entries_to_balance[swap_key_idx]
            {
                entries_to_balance.swap(swap_key_idx + 1, swap_key_idx);
            }
        }

        // Write keys and children into nodes.
        // TODO: Reuse this algorithm for both entris and children somehow.
        let mut start = 0;
        let mut chld_start = 0;
        for (current_iter, sibling) in siblings.iter_mut().enumerate() {
            let mut balanced_chunk_size = chunk_size;
            if current_iter < remainder {
                balanced_chunk_size += 1;
            }

            let end = min(start + balanced_chunk_size, entries_to_balance.len());
            sibling.0.entries.extend(&entries_to_balance[start..end]);

            if !children_to_balance.is_empty() {
                // there's always one more child than keys.
                let chld_chunk = balanced_chunk_size + 1;
                let end = min(chld_start + chld_chunk, children_to_balance.len());
                sibling
                    .0
                    .children
                    .extend(&children_to_balance[chld_start..end]);
                chld_start += chld_chunk;
            }

            start += balanced_chunk_size;
        }

        // Write to disk.
        // TODO: Sequential write queue, cache and stuff.
        for (node, _) in &siblings {
            self.write_node(node)?;
        }

        // If the parent didn't overflow we can terminate here. Otherwise
        // recurse upwards.
        if parent.entries.len() <= self.max_keys() {
            return self.write_node(&parent);
        }

        self.balance(parent, parents)
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
        let mut string = String::from('[');

        let root = self.read_node(0)?;
        string.push_str(&self.to_json(&root)?);

        for page in 1..self.len() {
            let next_node = self.read_node(page as u32)?;
            string.push(',');
            string.push_str(&self.to_json(&next_node)?);
        }

        string.push(']');

        Ok(string)
    }

    fn to_json(&mut self, node: &Node) -> io::Result<String> {
        let mut string = format!("{{\"page\":{},\"entries\":[", node.page);

        if node.entries.len() >= 1 {
            #[allow(unused_variables)]
            let Entry { key, value } = node.entries[0];
            // string.push_str(&format!("{{\"key\":{key},\"value\":{value}}}"));
            string.push_str(&format!("{}", key));

            #[allow(unused_variables)]
            for Entry { key, value } in &node.entries[1..] {
                string.push(',');
                // string.push_str(&format!("{{\"key\":{key},\"value\":{value}}}"));
                string.push_str(&format!("{}", key));
            }
        }

        string.push_str("],\"children\":[");

        if node.children.len() >= 1 {
            string.push_str(&format!("{}", node.children[0]));

            for child in &node.children[1..] {
                string.push_str(&format!(",{child}"));
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

    /// This is how a B*-Tree of `order = 4` should look like if we insert
    /// values from 1 to 15 sequentially:
    ///
    /// ```text
    ///                 +--------+
    ///         +-------| 4,8,12 |--------+
    ///       /         +--------+         \
    ///     /          /         \          \
    /// +-------+  +-------+  +---------+  +----------+
    /// | 1,2,3 |  | 5,6,7 |  | 9,10,11 |  | 13,14,15 |
    /// +-------+  +-------+  +---------+  +----------+
    /// ```
    #[test]
    fn basic_insertion() -> Result<(), Box<dyn Error>> {
        let page_size = 48;
        let max_nodes = 5;
        let keys: Vec<u32> = (1..=12).collect();

        let buf = io::Cursor::new(vec![0; page_size * max_nodes]);
        let mut btree = BTree::new(buf, page_size, page_size)?;

        for key in &keys {
            btree.insert(*key, 100 + key)?;
        }

        assert_eq!(btree.order(), 4);
        assert_eq!(btree.len(), max_nodes);

        for key in &keys {
            assert_eq!(btree.get(*key)?, 100 + key);
        }

        Ok(())
    }
}
