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
    fn new() -> Self {
        Self {
            entries: Vec::new(),
            children: Vec::new(),
            page: 0,
        }
    }

    fn new_at(page: usize) -> Self {
        let mut node = Self::new();
        node.page = page as u32;
        node
    }

    // Consumes all the entries and children in `other` and appends them to
    // `self`.
    fn extend_by_draining(&mut self, other: &mut Node) {
        self.entries.extend(other.entries.drain(..));
        self.children.extend(other.children.drain(..));
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
        let len = std::cmp::max(metadata.len() as usize / page_size, 1);

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

        // If the key is not present in the current node, recurse downwards or
        // return None if we are already at a leaf node.
        if let Err(index) = search {
            if node.children.is_empty() {
                return Ok(None);
            } else {
                let child = self.read_node(node.children[index])?;
                parents.push((node, index));
                return self.remove_from(child, key, parents);
            }
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
        // Started from the bottom now we're here :)
        let is_root = parents.is_empty();

        // A node can only overflow if the number of keys is greater than the
        // maximum allowed.
        let is_overflow = node.entries.len() > self.max_keys();

        // A node has underflowed if it contains less than 2/3 of keys. The root
        // is allowed to underflow past that point, but if it reaches 0 keys it
        // has completely underflowed and tree height must be decreased.
        let is_underflow =
            (node.entries.len() < self.max_keys() * 2 / 3 && !is_root) || node.entries.is_empty();

        // Done, this node didn't overflow or underflow.
        if !is_overflow && !is_underflow {
            return self.write_node(&node);
        }

        // Root underflow. Merging internal nodes has consumed the entire root.
        // Decrease tree height and return.
        if is_root && is_underflow {
            // Grab the only child remaining.
            let mut direct_child = self.read_node(node.children.remove(0))?;

            // Make the child the new root.
            node.extend_by_draining(&mut direct_child);

            // Done, no need to fall through the code below. Redistribution is
            // not necessary.
            self.free_page(direct_child.page);
            return self.write_node(&node);
        }

        // The root overflowed, so it must be split. We're not going to split
        // it here, we will only prepare the terrain for the splitting algorithm
        // below.
        if is_root && is_overflow {
            // The actual tree root always stays at the same page, it does not
            // move. First, copy the contents of the root to a new node and
            // leave the root empty.
            let mut old_root = Node::new_at(self.allocate_page());
            old_root.extend_by_draining(&mut node);

            // Now make the new node a child of the empty root.
            node.children.push(old_root.page);

            // Turn the new node into the balance target. Since this node has
            // overflowed the code below will split it accordingly.
            parents.push((node, 0));
            node = old_root;
        }

        let (mut parent, index) = parents.pop().unwrap();

        // Find all siblings involved in the balancing algorithm, including
        // the given node (balance target).
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

        let total_entries = siblings
            .iter()
            .map(|(sibling, _)| sibling.entries.len())
            .sum::<usize>();

        // We only split nodes if absolutely necessary. Otherwise we can just
        // move keys around siblings to keep balance. We reach the "must split"
        // point when all siblings are full and one of them has overflowed.
        let must_split = total_entries > self.max_keys() * siblings.len();

        // Same as before, merge only if there's no other way to keep balance.
        // This happens when all the entries can fit into one less node than we
        // have and one of the nodes has underflowed below 2/3 (integer divsion).
        // If we can avoid merging while still keeping siblings at 66%, we will.
        let must_merge =
            total_entries < (siblings.len() - 1) * self.max_keys() - (self.max_keys() * 2 % 3);

        if must_split {
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

            // Prepare terrain for the redistribution algorithm below by moving
            // the greatest key into the parent and adding the new empty child
            // to the parent.
            let rightmost_sibling = siblings.last_mut().unwrap();
            let new_node_parent_index = rightmost_sibling.1 + 1;
            parent.entries.insert(
                rightmost_sibling.1,
                rightmost_sibling.0.entries.pop().unwrap(),
            );
            parent.children.insert(new_node_parent_index, new_node.page);

            // Add new node to balancing list.
            siblings.push((new_node, new_node_parent_index));
        } else if must_merge {
            // Merge the first two nodes together, demote the first key in
            // the parent and let the redistribution algorithm below do its job.
            let (mut deleted_node, _) = siblings.remove(1);
            let (merge_node, divider_idx) = &mut siblings[0];

            merge_node.entries.push(parent.entries.remove(*divider_idx));
            merge_node.extend_by_draining(&mut deleted_node);

            parent.children.remove(*divider_idx + 1);
            self.free_page(deleted_node.page);
        }

        // This algorithm does most of the magic here. It prevents us from
        // splitting and merging by reordering keys around the nodes.
        self.redistribute_entries_and_children(&mut parent, &mut siblings);

        // Write to disk. Parent is not written yet because it might be in an
        // overflow or underflow state.
        // TODO: Sequential write queue, cache and stuff.
        for (node, _) in &siblings {
            self.write_node(node)?;
        }

        // Recurse upwards and propagate redistribution/merging/splitting.
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

    fn try_drain<'k, K: IntoIterator<Item = u32> + 'k>(
        &'k mut self,
        keys: K,
    ) -> impl Iterator<Item = io::Result<Option<u32>>> + 'k {
        keys.into_iter().map(|k| self.remove(k))
    }

    fn try_remove_all<K: IntoIterator<Item = u32>>(
        &mut self,
        keys: K,
    ) -> io::Result<Vec<Option<u32>>> {
        self.try_drain(keys).collect()
    }
}

impl<F: Seek + Read + Write> Extend<(u32, u32)> for BTree<F> {
    fn extend<T: IntoIterator<Item = (u32, u32)>>(&mut self, iter: T) {
        for (key, value) in iter {
            self.insert(key, value).unwrap();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{io, mem};

    /// Allows us to build an entire tree manually and then compare it to an
    /// actual [`BTree`] structure. See tests below for examples.
    #[derive(Debug, PartialEq)]
    struct Node {
        keys: Vec<u32>,
        children: Vec<Self>,
    }

    impl Node {
        /// Leaf nodes have no children. This method saves us some unecessary
        /// typing and makes the tree structure more readable.
        fn leaf<K: IntoIterator<Item = u32>>(keys: K) -> Self {
            Self {
                keys: keys.into_iter().collect(),
                children: Vec::new(),
            }
        }
    }

    /// Inverse of [`super::optimal_order_for`].
    fn optimal_page_size_for(order: usize) -> usize {
        mem::size_of::<u16>() * 2
            + mem::size_of::<u32>() * order
            + mem::size_of::<super::Entry>() * (order - 1)
    }

    /// Config/Builder for [`MemBufBTree`]. Can be used with
    /// [`TryFrom::try_from`] or [`MemBufBTree::builder`].
    struct Config {
        keys: Vec<u32>,
        order: usize,
        max_nodes: usize,
    }

    impl Default for Config {
        fn default() -> Self {
            Config {
                keys: vec![],
                order: 4,
                max_nodes: 1,
            }
        }
    }

    impl Config {
        fn keys<K: IntoIterator<Item = u32>>(mut self, keys: K) -> Self {
            self.keys = keys.into_iter().collect();
            self
        }

        fn max_nodes(mut self, max_nodes: usize) -> Self {
            self.max_nodes = max_nodes;
            self
        }

        fn order(mut self, order: usize) -> Self {
            self.order = order;
            self
        }

        fn try_build(self) -> io::Result<MemBufBTree> {
            MemBufBTree::try_from(self)
        }
    }

    /// We use in-memory buffers instead of disk files for testing. This speeds
    /// up tests as it avoids disk IO and system calls.
    type MemBufBTree = super::BTree<io::Cursor<Vec<u8>>>;

    impl MemBufBTree {
        fn into_test_nodes(&mut self, node: &super::Node) -> io::Result<Node> {
            let mut test_node = Node {
                keys: node.entries.iter().map(|e| e.key).collect(),
                children: vec![],
            };

            for page in &node.children {
                let child = self.read_node(*page)?;
                test_node.children.push(self.into_test_nodes(&child)?);
            }

            Ok(test_node)
        }

        fn extend_from_keys_only<K: IntoIterator<Item = u32>>(
            &mut self,
            keys: K,
        ) -> io::Result<()> {
            for key in keys {
                self.insert(key, key)?;
            }

            Ok(())
        }

        fn builder() -> Config {
            Config::default()
        }
    }

    impl TryFrom<Config> for MemBufBTree {
        type Error = io::Error;

        fn try_from(config: Config) -> Result<Self, Self::Error> {
            let page_size = optimal_page_size_for(config.order);
            let buf = io::Cursor::new(vec![0; page_size * config.max_nodes]);

            let mut btree = MemBufBTree::new(buf, page_size, page_size);
            btree.extend_from_keys_only(config.keys)?;

            Ok(btree)
        }
    }

    impl TryFrom<MemBufBTree> for Node {
        type Error = io::Error;

        fn try_from(mut btree: MemBufBTree) -> Result<Self, Self::Error> {
            let root = btree.read_node(0)?;

            btree.into_test_nodes(&root)
        }
    }

    /// When `order = 4` the root should be able to sustain 3 keys without
    /// splitting.
    ///
    /// ```text
    ///                  ROOT
    ///                    |
    ///                    V
    ///                +-------+
    ///                | 1,2,3 |
    ///                +-------+
    /// ```
    #[test]
    fn fill_root() -> io::Result<()> {
        let btree = MemBufBTree::builder()
            .keys(1..=3)
            .max_nodes(1)
            .try_build()?;

        assert_eq!(Node::try_from(btree)?, Node::leaf([1, 2, 3]));

        Ok(())
    }

    /// When `order = 4` and the root gets filled with 3 keys, inserting an
    /// additional key should cause it to split.
    ///
    /// ```text
    ///                INSERT 4
    ///                    |
    ///                    v
    ///                +-------+
    ///                | 1,2,3 |
    ///                +-------+
    ///
    ///                 RESULT
    ///                    |
    ///                    v
    ///                +-------+
    ///                |   3   |
    ///                +-------+
    ///               /         \
    ///           +-------+  +-------+
    ///           |  1,2  |  |   4   |
    ///           +-------+  +-------+
    /// ```
    #[test]
    fn split_root() -> io::Result<()> {
        let btree = MemBufBTree::builder()
            .keys(1..=4)
            .max_nodes(3)
            .try_build()?;

        assert_eq!(
            Node::try_from(btree)?,
            Node {
                keys: vec![3],
                children: vec![Node::leaf([1, 2]), Node::leaf([4])]
            }
        );

        Ok(())
    }

    /// Non-full sibling nodes can borrow keys from overflow nodes. See this
    /// example where `order = 4`:
    ///
    /// ```text
    ///                INSERT 7
    ///                    |
    ///                    v
    ///                +-------+
    ///                |   3   |
    ///                +-------+
    ///               /         \
    ///           +-------+  +-------+
    ///           |  1,2  |  | 4,5,6 |
    ///           +-------+  +-------+
    ///
    ///                 RESULT
    ///                    |
    ///                    v
    ///                +-------+
    ///                |   4   |
    ///                +-------+
    ///               /         \
    ///           +-------+  +-------+
    ///           | 1,2,3 |  | 5,6,7 |
    ///           +-------+  +-------+
    /// ```
    #[test]
    fn delay_leaf_node_split() -> io::Result<()> {
        let btree = MemBufBTree::builder()
            .keys(1..=7)
            .max_nodes(3)
            .try_build()?;

        assert_eq!(
            Node::try_from(btree)?,
            Node {
                keys: vec![4],
                children: vec![Node::leaf([1, 2, 3]), Node::leaf([5, 6, 7])]
            }
        );

        Ok(())
    }

    /// Leaf node should split when keys can't be moved around siblings. See
    /// this example where `order = 4`:
    ///
    /// ```text
    ///                INSERT 8
    ///                    |
    ///                    v
    ///                +-------+
    ///                |   4   |
    ///                +-------+
    ///               /         \
    ///           +-------+  +-------+
    ///           | 1,2,3 |  | 5,6,7 |
    ///           +-------+  +-------+
    ///
    ///                 RESULT
    ///                    |
    ///                    v
    ///                +-------+
    ///            +---|  3,6  |---+
    ///           /    +-------+    \
    ///          /         |         \
    ///     +-------+  +-------+  +-------+
    ///     |  1,2  |  |  4,5  |  |  7,8  |
    ///     +-------+  +-------+  +-------+
    /// ```
    #[test]
    fn split_leaf_node() -> io::Result<()> {
        let btree = MemBufBTree::builder()
            .keys(1..=8)
            .max_nodes(4)
            .try_build()?;

        assert_eq!(
            Node::try_from(btree)?,
            Node {
                keys: vec![3, 6],
                children: vec![Node::leaf([1, 2]), Node::leaf([4, 5]), Node::leaf([7, 8])]
            }
        );

        Ok(())
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
    fn basic_insertion() -> io::Result<()> {
        let btree = MemBufBTree::builder()
            .keys(1..=15)
            .max_nodes(5)
            .try_build()?;

        assert_eq!(
            Node::try_from(btree)?,
            Node {
                keys: vec![4, 8, 12],
                children: vec![
                    Node::leaf([1, 2, 3]),
                    Node::leaf([5, 6, 7]),
                    Node::leaf([9, 10, 11]),
                    Node::leaf([13, 14, 15]),
                ]
            }
        );

        Ok(())
    }

    /// When a node splits and causes the parent to overflow, the parent should
    /// split as well.
    ///
    /// ```text
    ///                           INSERT 16
    ///                               |
    ///                               V
    ///                           +--------+
    ///                   +-------| 4,8,12 |--------+
    ///                 /         +--------+         \
    ///               /          /         \          \
    ///           +-------+  +-------+  +---------+  +----------+
    ///           | 1,2,3 |  | 5,6,7 |  | 9,10,11 |  | 13,14,15 |
    ///           +-------+  +-------+  +---------+  +----------+
    ///
    ///                            RESULT
    ///                               |
    ///                               V
    ///                           +--------+
    ///                   +-------|   11   |-------+
    ///                  /        +--------+        \
    ///                 /                            \
    ///            +-------+                     +--------+
    ///       +----|  4,8  |----+                |   14   |
    ///      /     +-------+     \               +--------+
    ///     /          |          \               /      \
    /// +-------+  +-------+  +-------+     +-------+  +-------+
    /// | 1,2,3 |  | 5,6,7 |  | 9,10  |     | 12,13 |  | 15,16 |
    /// +-------+  +-------+  +-------+     +-------+  +-------+
    /// ```
    #[test]
    fn propagate_split_to_root() -> io::Result<()> {
        let btree = MemBufBTree::builder()
            .keys(1..=16)
            .max_nodes(8)
            .try_build()?;

        assert_eq!(
            Node::try_from(btree)?,
            Node {
                keys: vec![11],
                children: vec![
                    Node {
                        keys: vec![4, 8],
                        children: vec![
                            Node::leaf([1, 2, 3]),
                            Node::leaf([5, 6, 7]),
                            Node::leaf([9, 10]),
                        ]
                    },
                    Node {
                        keys: vec![14],
                        children: vec![Node::leaf([12, 13]), Node::leaf([15, 16])]
                    }
                ]
            }
        );

        Ok(())
    }

    /// Same as [`delay_leaf_node_split`] but with internal nodes. In this
    /// example, the second child of the root should not split because its
    /// left sibling can take one additional key and child.
    ///
    /// ```text
    ///                                    INSERT 27
    ///                                        |
    ///                                        V
    ///                                    +--------+
    ///                   +----------------|   11   |---------------+
    ///                  /                 +--------+                \
    ///                 /                                             \
    ///            +-------+                                     +----------+
    ///       +----|  4,8  |----+                   +------------| 15,19,23 |------------+
    ///      /     +-------+     \                 /             +----------+              \
    ///     /          |          \               /               /        \                \
    /// +-------+  +-------+  +-------+     +----------+  +----------+  +----------+  +----------+
    /// | 1,2,3 |  | 5,6,7 |  | 9,10  |     | 12,13,14 |  | 16,17,18 |  | 20,21,22 |  | 24,25,26 |
    /// +-------+  +-------+  +-------+     +----------+  +----------+  +----------+  +----------+
    ///
    ///                                              RESULT
    ///                                                |
    ///                                                V
    ///                                            +--------+
    ///                         +------------------|   15   |-----------------+
    ///                        /                   +--------+                  \
    ///                       /                                                 \
    ///                  +--------+                                         +----------+
    ///       +----------| 4,8,11 |---------+                     +---------| 19,22,25 |--------+
    ///      /           +--------+          \                   /          +----------+         \
    ///     /             /     \             \                 /             /      \            \
    /// +-------+  +-------+  +-------+  +----------+     +----------+  +-------+  +-------+  +-------+
    /// | 1,2,3 |  | 5,6,7 |  | 9,10  |  | 12,13,14 |     | 16,17,18 |  | 20,21 |  | 23,24 |  | 26,27 |
    /// +-------+  +-------+  +-------+  +----------+     +----------+  +-------+  +-------+  +-------+
    /// ```
    #[test]
    fn delay_internal_node_split() -> io::Result<()> {
        let btree = MemBufBTree::builder()
            .keys(1..=27)
            .max_nodes(11)
            .try_build()?;

        assert_eq!(
            Node::try_from(btree)?,
            Node {
                keys: vec![15],
                children: vec![
                    Node {
                        keys: vec![4, 8, 11],
                        children: vec![
                            Node::leaf([1, 2, 3]),
                            Node::leaf([5, 6, 7]),
                            Node::leaf([9, 10]),
                            Node::leaf([12, 13, 14]),
                        ]
                    },
                    Node {
                        keys: vec![19, 22, 25],
                        children: vec![
                            Node::leaf([16, 17, 18]),
                            Node::leaf([20, 21]),
                            Node::leaf([23, 24]),
                            Node::leaf([26, 27]),
                        ]
                    },
                ]
            }
        );

        Ok(())
    }

    /// When internal nodes can't move keys around siblings they should split as
    /// well.
    ///
    /// ```text
    ///                                             INSERT 31
    ///                                                |
    ///                                                V
    ///                                            +--------+
    ///                         +------------------|   15   |-----------------+
    ///                        /                   +--------+                  \
    ///                       /                                                 \
    ///                  +--------+                                            +----------+
    ///       +----------| 4,8,11 |---------+                     +------------| 19,23,27 |-----------+
    ///      /           +--------+          \                   /             +----------+            \
    ///     /             /     \             \                 /               /        \              \
    /// +-------+  +-------+  +-------+  +----------+     +----------+  +----------+  +----------+  +----------+
    /// | 1,2,3 |  | 5,6,7 |  | 9,10  |  | 12,13,14 |     | 16,17,18 |  | 20,21,22 |  | 24,25,26 |  | 28,29,30 |
    /// +-------+  +-------+  +-------+  +----------+     +----------+  +----------+  +----------+  +----------+
    ///
    ///                                                     RESULT
    ///                                                        |
    ///                                                        V
    ///                                                    +-------+
    ///                  +---------------------------------| 11,23 |--------------------------------+
    ///                 /                                  +-------+                                 \
    ///                /                                       |                                      \
    ///             +-----+                                +-------+                              +-------+
    ///        +----| 4,8 |-----+                  +-------| 15,19 |-------+                 +----| 26,29 |----+
    ///       /     +-----+      \                /        +-------+        \               /     +-------+     \
    ///      /         |          \              /             |             \             /          |          \
    /// +-------+  +-------+  +-------+    +----------+  +----------+  +----------+    +-------+  +-------+  +-------+
    /// | 1,2,3 |  | 5,6,7 |  | 9,10  |    | 12,13,14 |  | 16,17,18 |  | 20,21,22 |    | 24,25 |  | 27,28 |  | 30,31 |
    /// +-------+  +-------+  +-------+    +----------+  +----------+  +----------+    +-------+  +-------+  +-------+
    /// ```
    #[test]
    fn propagate_split_to_internal_nodes() -> io::Result<()> {
        let btree = MemBufBTree::builder()
            .keys(1..=31)
            .max_nodes(13)
            .try_build()?;

        assert_eq!(
            Node::try_from(btree)?,
            Node {
                keys: vec![11, 23],
                children: vec![
                    Node {
                        keys: vec![4, 8],
                        children: vec![
                            Node::leaf([1, 2, 3]),
                            Node::leaf([5, 6, 7]),
                            Node::leaf([9, 10]),
                        ]
                    },
                    Node {
                        keys: vec![15, 19],
                        children: vec![
                            Node::leaf([12, 13, 14]),
                            Node::leaf([16, 17, 18]),
                            Node::leaf([20, 21, 22]),
                        ]
                    },
                    Node {
                        keys: vec![26, 29],
                        children: vec![
                            Node::leaf([24, 25]),
                            Node::leaf([27, 28]),
                            Node::leaf([30, 31]),
                        ]
                    },
                ]
            }
        );

        Ok(())
    }

    /// This is how a [`super::BTree`] of `order = 4` should look like after
    /// inserting values 1 to 46 included sequentially:
    ///
    /// ```text
    ///                                                                        +-------+
    ///                        +-----------------------------------------------| 15,31 |---------------------------------------------------+
    ///                       /                                                +-------+                                                    \
    ///                      /                                                     |                                                         \
    ///                 +--------+                                            +----------+                                              +----------+
    ///        +--------| 4,8,11 |---------+                     +------------| 19,23,27 |-----------+                     +------------| 35,39,43 |-----------+
    ///       /         +--------+          \                   /             +----------+            \                   /             +----------+            \
    ///      /           /     \             \                 /               /       \               \                 /               /       \               \
    /// +-------+  +-------+  +-------+  +----------+    +----------+  +----------+  +----------+  +----------+    +----------+  +----------+  +----------+  +----------+
    /// | 1,2,3 |  | 5,6,7 |  | 9,10  |  | 12,13,14 |    | 16,17,18 |  | 20,21,22 |  | 24,25,26 |  | 28,29,30 |    | 32,33,34 |  | 36,37,38 |  | 40,41,42 |  | 44,45,46 |
    /// +-------+  +-------+  +-------+  +----------+    +----------+  +----------+  +----------+  +----------+    +----------+  +----------+  +----------+  +----------+
    /// ```
    #[test]
    fn sequential_insertion() -> io::Result<()> {
        let btree = MemBufBTree::builder()
            .keys(1..=46)
            .max_nodes(16)
            .try_build()?;

        assert_eq!(
            Node::try_from(btree)?,
            Node {
                keys: vec![15, 31],
                children: vec![
                    Node {
                        keys: vec![4, 8, 11],
                        children: vec![
                            Node::leaf([1, 2, 3]),
                            Node::leaf([5, 6, 7]),
                            Node::leaf([9, 10]),
                            Node::leaf([12, 13, 14]),
                        ]
                    },
                    Node {
                        keys: vec![19, 23, 27],
                        children: vec![
                            Node::leaf([16, 17, 18]),
                            Node::leaf([20, 21, 22]),
                            Node::leaf([24, 25, 26]),
                            Node::leaf([28, 29, 30]),
                        ]
                    },
                    Node {
                        keys: vec![35, 39, 43],
                        children: vec![
                            Node::leaf([32, 33, 34]),
                            Node::leaf([36, 37, 38]),
                            Node::leaf([40, 41, 42]),
                            Node::leaf([44, 45, 46]),
                        ]
                    },
                ]
            }
        );

        Ok(())
    }

    /// Deleting from leaf nodes is the simplest case.
    ///
    /// ```text
    ///                           DELETE 13
    ///                               |
    ///                               V
    ///                           +--------+
    ///                   +-------| 4,8,12 |--------+
    ///                 /         +--------+         \
    ///               /           /       \           \
    ///           +-------+  +-------+  +---------+  +----------+
    ///           | 1,2,3 |  | 5,6,7 |  | 9,10,11 |  | 13,14,15 |
    ///           +-------+  +-------+  +---------+  +----------+
    ///
    ///                             RESULT
    ///                               |
    ///                               V
    ///                           +--------+
    ///                   +-------| 4,8,12 |--------+
    ///                 /         +--------+         \
    ///               /            /      \           \
    ///           +-------+  +-------+  +---------+  +----------+
    ///           | 1,2,3 |  | 5,6,7 |  | 9,10,11 |  | 14,15    |
    ///           +-------+  +-------+  +---------+  +----------+
    /// ```
    #[test]
    fn delete_from_leaf_node() -> io::Result<()> {
        let mut btree = MemBufBTree::builder()
            .keys(1..=15)
            .max_nodes(5)
            .try_build()?;

        btree.remove(13)?;

        assert_eq!(
            Node::try_from(btree)?,
            Node {
                keys: vec![4, 8, 12],
                children: vec![
                    Node::leaf([1, 2, 3]),
                    Node::leaf([5, 6, 7]),
                    Node::leaf([9, 10, 11]),
                    Node::leaf([14, 15]),
                ]
            }
        );

        Ok(())
    }

    /// When a leaf node falls under 66% capacity it should not be merged with
    /// one of its siblings if the siblings can lend keys without underflowing.
    ///
    /// ```text
    ///
    ///                           DELETE 15
    ///                               |
    ///                               V
    ///                           +--------+
    ///                   +-------| 4,8,12 |--------+
    ///                 /         +--------+         \
    ///               /            /      \           \
    ///           +-------+  +-------+  +---------+  +----------+
    ///           | 1,2,3 |  | 5,6,7 |  | 9,10,11 |  | 13,14,15 |
    ///           +-------+  +-------+  +---------+  +----------+
    ///
    ///                           DELETE 14
    ///                               |
    ///                               V
    ///                           +--------+
    ///                   +-------| 4,8,12 |--------+
    ///                 /         +--------+         \
    ///               /            /      \           \
    ///           +-------+  +-------+  +---------+  +----------+
    ///           | 1,2,3 |  | 5,6,7 |  | 9,10,11 |  | 13,14    |
    ///           +-------+  +-------+  +---------+  +----------+
    ///
    ///                          FINAL RESULT
    ///                               |
    ///                               V
    ///                           +--------+
    ///                   +-------| 4,8,11 |--------+
    ///                 /         +--------+         \
    ///               /            /      \           \
    ///           +-------+  +-------+  +---------+  +----------+
    ///           | 1,2,3 |  | 5,6,7 |  | 9,10    |  | 12,13    |
    ///           +-------+  +-------+  +---------+  +----------+
    /// ```
    #[test]
    fn delay_leaf_node_merge() -> io::Result<()> {
        let mut btree = MemBufBTree::builder()
            .keys(1..=15)
            .max_nodes(5)
            .try_build()?;

        btree.try_remove_all((14..=15).rev())?;

        assert_eq!(
            Node::try_from(btree)?,
            Node {
                keys: vec![4, 8, 11],
                children: vec![
                    Node::leaf([1, 2, 3]),
                    Node::leaf([5, 6, 7]),
                    Node::leaf([9, 10]),
                    Node::leaf([12, 13]),
                ]
            }
        );

        Ok(())
    }

    /// Leaf nodes should be merged when keys can't be reordered across
    /// siblings.
    ///
    /// ```text
    ///                     DELETE (15,14,13) -> No Merge
    ///                               |
    ///                               V
    ///                           +--------+
    ///                   +-------| 4,8,12 |--------+
    ///                 /         +--------+         \
    ///               /            /      \           \
    ///           +-------+  +-------+  +---------+  +----------+
    ///           | 1,2,3 |  | 5,6,7 |  | 9,10,11 |  | 13,14,15 |
    ///           +-------+  +-------+  +---------+  +----------+
    ///
    ///                           DELETE 12
    ///                               |
    ///                               V
    ///                           +--------+
    ///                   +-------| 4,7,10 |------+
    ///                  /        +--------+       \
    ///                 /          /      \         \
    ///            +-------+  +-----+  +-----+  +-------+
    ///            | 1,2,3 |  | 5,6 |  | 8,9 |  | 11,12 |
    ///            +-------+  +-----+  +-----+  +-------+
    ///
    ///                          FINAL RESULT
    ///                               |
    ///                               V
    ///                            +-----+
    ///                       +----| 4,8 |-----+
    ///                      /     +-----+      \  
    ///                     /         |          \    
    ///                +-------+  +-------+  +---------+  
    ///                | 1,2,3 |  | 5,6,7 |  | 9,10,11 |  
    ///                +-------+  +-------+  +---------+  
    /// ```
    #[test]
    fn merge_leaf_node() -> io::Result<()> {
        let mut btree = MemBufBTree::builder()
            .keys(1..=15)
            .max_nodes(5)
            .try_build()?;

        btree.try_remove_all((12..=15).rev())?;

        assert_eq!(
            Node::try_from(btree)?,
            Node {
                keys: vec![4, 8],
                children: vec![
                    Node::leaf([1, 2, 3]),
                    Node::leaf([5, 6, 7]),
                    Node::leaf([9, 10, 11]),
                ]
            }
        );

        Ok(())
    }
}
