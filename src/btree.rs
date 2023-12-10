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
    pub fn new(file: F, page_size: usize, block_size: usize) -> Self {
        let pager = Pager::new(file, page_size, block_size);
        let order = optimal_order_for(page_size);
        let len = 1;

        Self { pager, order, len }
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

        // TODO: Take free pages into account.
        let len = metadata.len() as usize / page_size;

        Ok(Self { pager, order, len })
    }
}

impl<F: Seek + Read + Write> BTree<F> {
    pub fn get(&mut self, key: u32) -> io::Result<Option<u32>> {
        let root = self.read_node(0)?;
        self.find(&root, key)
    }

    pub fn insert(&mut self, key: u32, value: u32) -> io::Result<()> {
        let root = self.read_node(0)?;

        let mut parents = Vec::new();
        let node = self.insert_into(root, Entry { key, value }, &mut parents)?;

        self.balance(node, parents)
    }

    pub fn remove(&mut self, key: u32) -> io::Result<Option<u32>> {
        let root = self.read_node(0)?;

        let mut parents = Vec::new();

        let Some((entry, leaf_node)) = self.remove_from(root, key, &mut parents)? else {
            return Ok(None);
        };

        self.balance(leaf_node, parents)?;

        return Ok(Some(entry.value));
    }

    fn allocate_page(&mut self) -> usize {
        // TODO: Free list.
        let page = self.len;
        self.len += 1;

        page
    }

    fn free_page(&mut self, page: u32) {
        // TODO: Free list.
        self.pager
            .write_page(page as usize, &Vec::from("FREE PAGE".as_bytes()))
            .unwrap();

        self.len -= 1;
    }

    fn remove_from(
        &mut self,
        mut node: Node,
        key: u32,
        parents: &mut Vec<(Node, usize)>,
    ) -> io::Result<Option<(Entry, Node)>> {
        let search = node.entries.binary_search(&Entry { key, value: 0 });

        // If the key is not present in the current node, recurse downwards.
        if let Err(index) = search {
            let child = self.read_node(node.children[index])?;
            parents.push((node, index));
            return self.remove_from(child, key, parents);
        }

        let index = search.unwrap();

        // If this node is a leaf node, we're done with deleting.
        if node.children.is_empty() {
            let entry = node.entries.remove(index);
            self.write_node(&node)?;
            return Ok(Some((entry, node)));
        }

        parents.push((node, index));

        let curr_node_idx = parents.len() - 1;

        // If it's not a leaf node, then find a suitable substitute in a leaf
        // node (either predecessor in the left subtree or successor in the
        // right subtree) and return the entire path for balancing purposes.
        let left_child = self.read_node(parents[curr_node_idx].0.children[index])?;
        let right_child = self.read_node(parents[curr_node_idx].0.children[index + 1])?;

        let (substitute, leaf_node) = if left_child.entries.len() >= right_child.entries.len() {
            let mut predecesor = self.find_max(left_child, parents)?;
            (predecesor.entries.pop().unwrap(), predecesor)
        } else {
            let mut successor = self.find_min(right_child, parents)?;
            (successor.entries.remove(0), successor)
        };

        let value = mem::replace(&mut parents[curr_node_idx].0.entries[index], substitute);

        self.write_node(&parents[curr_node_idx].0)?;
        self.write_node(&leaf_node)?;

        return Ok(Some((value, leaf_node)));
    }

    fn find_max(&mut self, node: Node, parents: &mut Vec<(Node, usize)>) -> io::Result<Node> {
        if node.children.is_empty() {
            return Ok(node);
        }

        let index = node.children.len() - 1;

        let child = self.read_node(node.children[index])?;
        parents.push((node, index));

        self.find_max(child, parents)
    }

    fn find_min(&mut self, node: Node, parents: &mut Vec<(Node, usize)>) -> io::Result<Node> {
        if node.children.is_empty() {
            return Ok(node);
        }

        let child = self.read_node(node.children[0])?;
        parents.push((node, 0));

        self.find_min(child, parents)
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
    /// splitting and merging as much as possible while keeping all nodes at
    /// least 2/3 full except the root.
    ///
    /// TODO: Explain how.
    fn balance(&mut self, mut node: Node, mut parents: Vec<(Node, usize)>) -> io::Result<()> {
        // Done, this node didn't overflow or underflow.
        if (self.max_keys() * 2 / 3 <= node.entries.len() || parents.is_empty())
            && node.entries.len() <= self.max_keys()
            && !node.entries.is_empty()
        {
            return Ok(());
        }

        if parents.is_empty() {
            // The root overflowed, so it must be split.
            if node.entries.len() > self.max_keys() {
                let mut old_root = Node::new_at(self.allocate_page());
                old_root.entries.extend(node.entries.drain(..));
                old_root.children.extend(node.children.drain(..));

                node.children.push(old_root.page);

                parents.push((node, 0));
                node = old_root;
            }
            // Merging internal nodes has consumed the entire root. Decrease
            // tree height.
            else if node.entries.is_empty() {
                let mut root_child = self.read_node(node.children[0])?;
                node.children.remove(0);
                node.entries.extend(root_child.entries.drain(..));
                node.children.extend(root_child.children.drain(..));
                self.free_page(root_child.page);
                return self.write_node(&node);
            }
        }

        let (mut parent, index) = parents.pop().unwrap();

        // Find all siblings involved in the balancing algorithm, including
        // the given node.
        let mut siblings = {
            let mut num_siblings_per_side = BALANCE_SIBLINGS_PER_SIDE;

            if index == 0 || index == parent.children.len() - 1 {
                num_siblings_per_side *= 2;
            };

            let left_siblings = index.checked_sub(num_siblings_per_side).unwrap_or(0)..index;

            let right_siblings =
                (index + 1)..min(index + num_siblings_per_side + 1, parent.children.len());

            let mut siblings =
                Vec::with_capacity(left_siblings.size_hint().0 + 1 + right_siblings.size_hint().0);

            for i in left_siblings {
                siblings.push((self.read_node(parent.children[i])?, i));
            }

            siblings.push((node, index));

            for i in right_siblings {
                siblings.push((self.read_node(parent.children[i])?, i));
            }

            siblings
        };

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
        // The node has underflowed and none of the siblings can lend a key to
        // keep it at 66% full. Merge two nodes together to keep balance.
        else if siblings
            .iter()
            .map(|(sibling, _)| sibling.entries.len())
            .sum::<usize>()
            < (siblings.len() - 1) * self.max_keys() - (self.max_keys() * 2 % 3)
        {
            let (mut deleted_node, _) = siblings.remove(1);
            let (merge_node, divider_idx) = &mut siblings[0];

            merge_node.entries.push(parent.entries.remove(*divider_idx));
            merge_node.entries.extend(deleted_node.entries.drain(..));
            merge_node.children.extend(deleted_node.children.drain(..));

            parent.children.remove(*divider_idx + 1);
            self.free_page(deleted_node.page);
        }

        self.redistribute_entries_and_children(&mut parent, &mut siblings);

        // Write to disk.
        // TODO: Sequential write queue, cache and stuff.
        for (node, _) in &siblings {
            self.write_node(node)?;
        }

        // If the parent didn't overflow or underflow we can terminate here.
        if (self.max_keys() * 2 / 3 <= parent.entries.len() || parents.is_empty())
            && parent.entries.len() <= self.max_keys()
            && !parent.entries.is_empty()
        {
            return self.write_node(&parent);
        }

        // Otherwise recurse upwards and propagate splits/merges.
        self.balance(parent, parents)
    }

    /// Redistribute entries and children evenly. TODO: Explain algorithm.
    fn redistribute_entries_and_children(
        &self,
        parent: &mut Node,
        siblings: &mut Vec<(Node, usize)>,
    ) {
        let mut entries_to_balance = Vec::new();
        let mut children_to_balance = Vec::new();

        for (node, _) in siblings.iter_mut() {
            entries_to_balance.extend(node.entries.drain(..));
            children_to_balance.extend(node.children.drain(..));
        }

        let num_siblings = siblings.len();

        let chunk_size = entries_to_balance.len() / num_siblings;
        let remainder = entries_to_balance.len() % num_siblings;

        let mut entries_start = 0;
        let mut children_start = 0;

        for (i, (sibling, divider_idx)) in siblings.iter_mut().enumerate() {
            let mut balanced_chunk_size = chunk_size;
            if i < remainder {
                balanced_chunk_size += 1;
            }

            // Swap keys with parent.
            if i < num_siblings - 1 {
                let swap_key_idx = entries_start + balanced_chunk_size;

                let mut keys = [
                    entries_to_balance[swap_key_idx - 1],
                    parent.entries[*divider_idx],
                    entries_to_balance[swap_key_idx],
                ];

                keys.sort();

                let maybe_swap = if keys[1] == entries_to_balance[swap_key_idx] {
                    Some(swap_key_idx)
                } else if keys[1] == entries_to_balance[swap_key_idx - 1] {
                    Some(swap_key_idx - 1)
                } else {
                    None
                };

                if let Some(swap_key_idx) = maybe_swap {
                    mem::swap(
                        &mut parent.entries[*divider_idx],
                        &mut entries_to_balance[swap_key_idx],
                    );

                    // Sort demoted keys.
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
            }

            // Distribute entries.
            let entries_end = min(
                entries_start + balanced_chunk_size,
                entries_to_balance.len(),
            );
            sibling
                .entries
                .extend(&entries_to_balance[entries_start..entries_end]);
            entries_start += balanced_chunk_size;

            // Distribute children.
            if !children_to_balance.is_empty() {
                // There's always one more child than keys.
                balanced_chunk_size += 1;
                let children_end = min(
                    children_start + balanced_chunk_size,
                    children_to_balance.len(),
                );
                sibling
                    .children
                    .extend(&children_to_balance[children_start..children_end]);
                children_start += balanced_chunk_size;
            }
        }
    }

    fn find(&mut self, node: &Node, key: u32) -> io::Result<Option<u32>> {
        let search = node.entries.binary_search(&Entry { key, value: 0 });

        // If the key is here, bail out.
        if let Ok(index) = search {
            return Ok(Some(node.entries[index].value));
        }

        let index = search.unwrap_err();

        // If the key is not in this leaf node, bad luck.
        if node.children.is_empty() {
            return Ok(None);
        }

        // Otherwise recurse downwards.
        let next_node = self.read_node(node.children[index])?;
        self.find(&next_node, key)
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

        // TODO: Length increment is confirmed when writing, but one node can
        // be written multiple times from different functions. Do something
        // about it.
        // if node.page as usize == self.len {
        //     self.len += 1;
        // }

        Ok(())
    }

    // Testing/Debugging only.
    fn read_into_mem(&mut self, node: Node, buf: &mut Vec<Node>) -> io::Result<()> {
        for page in &node.children {
            let child = self.read_node(*page)?;
            self.read_into_mem(child, buf)?;
        }

        buf.push(node);

        Ok(())
    }

    pub fn json(&mut self) -> io::Result<String> {
        let root = self.read_node(0)?;

        let mut nodes = Vec::new();
        self.read_into_mem(root, &mut nodes)?;

        nodes.sort_by(|n1, n2| n1.page.cmp(&n2.page));

        let mut string = String::from('[');

        string.push_str(&self.to_json(&nodes[0])?);

        for node in &nodes[1..] {
            string.push(',');
            string.push_str(&self.to_json(&node)?);
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
    use std::{
        error::Error,
        io::{self, Cursor},
        mem,
    };

    use super::{BTree, Entry};

    fn optimal_page_size_for(order: usize) -> usize {
        mem::size_of::<u32>() * 2
            + mem::size_of::<u32>() * order
            + mem::size_of::<Entry>() * (order - 1)
    }

    impl BTree<Cursor<Vec<u8>>> {
        fn with_order(order: usize, max_nodes: usize) -> Self {
            let page_size = optimal_page_size_for(order);
            let buf = io::Cursor::new(vec![0; page_size * max_nodes]);

            BTree::new(buf, page_size, page_size)
        }
    }

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
        let max_nodes = 5;
        let keys: Vec<u32> = (1..=12).collect();

        let mut btree = BTree::with_order(4, max_nodes);

        for key in &keys {
            btree.insert(*key, 100 + key)?;
        }

        assert_eq!(btree.len(), max_nodes);

        for key in &keys {
            assert_eq!(btree.get(*key)?.unwrap(), 100 + key);
        }

        Ok(())
    }

    // TODO: Clean/refactor.
    #[test]
    fn remove() -> Result<(), Box<dyn Error>> {
        let max_nodes = 16;
        let keys: Vec<u32> = (1..=46).collect();

        let mut btree = BTree::with_order(4, max_nodes);

        for key in keys {
            btree.insert(key, key)?;
        }

        println!("INITIAL LEN {}", btree.len());

        // Delete from leaf node without overflow
        assert_eq!(btree.remove(46)?, Some(46));
        assert_eq!(btree.len(), max_nodes);

        // Delete from another leaf node without overflow
        assert_eq!(btree.remove(1)?, Some(1));
        assert_eq!(btree.len(), max_nodes);

        // Delete from leaf node causing underflow. Siblings can lend keys.
        assert_eq!(btree.remove(2)?, Some(2));
        assert_eq!(btree.len(), max_nodes);

        // Delete from leaf node causing underflow. This time must merge.
        assert_eq!(btree.remove(3)?, Some(3));
        assert_eq!(btree.len(), max_nodes - 1);

        // Delete from internal node.
        assert_eq!(btree.remove(7)?, Some(7));
        assert_eq!(btree.len(), max_nodes - 1);

        // Delete from internal node.
        assert_eq!(btree.remove(6)?, Some(6));
        assert_eq!(btree.len(), max_nodes - 1);

        // Delete from internal node.
        assert_eq!(btree.remove(8)?, Some(8));
        assert_eq!(btree.len(), max_nodes - 1);

        // Delete from internal node. Merge & propagate lending/redistribution.
        assert_eq!(btree.remove(11)?, Some(11));
        assert_eq!(btree.len(), max_nodes - 2);

        assert_eq!(btree.remove(4)?, Some(4));
        assert_eq!(btree.remove(5)?, Some(5));
        assert_eq!(btree.remove(9)?, Some(9));
        assert_eq!(btree.remove(10)?, Some(10));
        assert_eq!(btree.len(), max_nodes - 3);

        assert_eq!(btree.remove(12)?, Some(12));
        assert_eq!(btree.remove(13)?, Some(13));
        assert_eq!(btree.remove(14)?, Some(14));
        assert_eq!(btree.len(), max_nodes - 3);

        // Merge propagates twice. Loose 2 nodes.
        assert_eq!(btree.remove(15)?, Some(15));
        assert_eq!(btree.len(), max_nodes - 5);

        // Merge decreases tree height (root is consumed).
        for k in 16..=32 {
            assert_eq!(btree.remove(k)?, Some(k));
        }

        assert_eq!(btree.len(), 5);

        // println!("{}", btree.json()?);

        Ok(())
    }
}
