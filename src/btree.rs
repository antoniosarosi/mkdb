use std::{
    cmp::min,
    fs::File,
    io,
    io::{Read, Seek, Write},
    mem,
    path::Path,
};

use crate::{
    cache::Cache,
    node::{Entry, Node},
    pager::Pager,
};

/// Default value for [`BTree::balance_siblings_per_side`].
const BALANCE_SIBLINGS_PER_SIDE: usize = 1;

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
/// [`optimal_order_for`] for degree computations.
pub(crate) struct BTree<F> {
    cache: Cache<F>,

    /// Maximum number of children per node.
    order: usize,

    /// Number of siblings to examine at each side when balancing a node.
    balance_siblings_per_side: usize,

    /// Number of nodes (or pages) in the tree.
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
    pub fn new(cache: Cache<F>, balance_siblings_per_side: usize) -> Self {
        let order = optimal_order_for(cache.pager.page_size);
        let len = 1;

        Self {
            cache,
            order,
            len,
            balance_siblings_per_side,
        }
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
        let cache = Cache::new(Pager::new(file, page_size, block_size));
        let order = optimal_order_for(page_size);
        let balance_siblings_per_side = BALANCE_SIBLINGS_PER_SIDE;

        // TODO: Take free pages into account.
        let len = std::cmp::max(metadata.len() as usize / page_size, 1);

        Ok(Self {
            cache,
            order,
            len,
            balance_siblings_per_side,
        })
    }
}

impl<F: Seek + Read + Write> BTree<F> {
    pub fn get(&mut self, key: u32) -> io::Result<Option<u32>> {
        self.find(0, key)
    }

    pub fn insert(&mut self, key: u32, value: u32) -> io::Result<()> {
        let mut parents = Vec::new();
        let node = self.insert_into(0, Entry { key, value }, &mut parents)?;

        self.balance(node, parents)?;
        self.cache.flush_write_queue_to_disk()?;

        Ok(())
    }

    pub fn remove(&mut self, key: u32) -> io::Result<Option<u32>> {
        let mut parents = Vec::new();

        let Some((entry, leaf_node)) = self.remove_from(0, key, &mut parents)? else {
            return Ok(None);
        };

        self.balance(leaf_node, parents)?;
        self.cache.flush_write_queue_to_disk()?;

        return Ok(Some(entry.value));
    }

    fn node(&mut self, page: u32) -> io::Result<&Node> {
        self.cache.get(page)
    }

    fn node_mut(&mut self, page: u32) -> io::Result<&mut Node> {
        self.cache.get_mut(page)
    }

    fn append_node(&mut self, base: u32, target: u32) -> io::Result<()> {
        let target = self.node_mut(target)?;
        let entries: Vec<Entry> = target.entries.drain(..).collect();
        let children: Vec<u32> = target.children.drain(..).collect();

        let base = self.node_mut(base)?;
        base.entries.extend(entries);
        base.children.extend(children);

        Ok(())
    }

    fn find(&mut self, page: u32, key: u32) -> io::Result<Option<u32>> {
        let node = self.cache.get(page)?;
        let search = node.entries.binary_search_by_key(&key, |e| e.key);

        // If the key is here, bail out.
        if let Ok(index) = search {
            return Ok(Some(node.entries[index].value));
        }

        let index = search.unwrap_err();

        // If the key is not in this leaf node, bad luck.
        if node.is_leaf() {
            return Ok(None);
        }

        // Otherwise recurse downwards.
        let next_node = node.children[index];
        self.find(next_node, key)
    }

    fn insert_into(&mut self, page: u32, entry: Entry, parents: &mut Vec<u32>) -> io::Result<u32> {
        let node = self.cache.get(page)?;
        let search = node.entries.binary_search(&entry);

        // Key found, swap value and return.
        if let Ok(index) = search {
            self.node_mut(page)?.entries[index].value = entry.value;
            return Ok(page);
        }

        // Key is not present in current node, get the index where it should be.
        let index = search.unwrap_err();

        // If this node is a leaf node we're done.
        if node.is_leaf() {
            self.node_mut(page)?.entries.insert(index, entry);
            return Ok(page);
        }

        // If it's not a leaf node, we have to go one level below.
        parents.push(page);
        let next_node = node.children[index];

        self.insert_into(next_node, entry, parents)
    }

    fn remove_from(
        &mut self,
        page: u32,
        key: u32,
        parents: &mut Vec<u32>,
    ) -> io::Result<Option<(Entry, u32)>> {
        let node = self.cache.get(page)?;

        let search = node.entries.binary_search_by_key(&key, |e| e.key);

        // If the key is not present in the current node, recurse downwards or
        // return None if we are already at a leaf node.
        if let Err(index) = search {
            if node.is_leaf() {
                return Ok(None);
            } else {
                let child = node.children[index];
                parents.push(page);
                return self.remove_from(child, key, parents);
            }
        }

        let index = search.unwrap();

        // If this node is a leaf node, we're done with deleting.
        if node.is_leaf() {
            let entry = self.node_mut(page)?.entries.remove(index);
            return Ok(Some((entry, page)));
        }

        // If it's not a leaf node, then find a suitable substitute in a leaf
        // node (either predecessor in the left subtree or successor in the
        // right subtree) and return the entire path for balancing purposes.
        let left_child = node.children[index];
        let right_child = node.children[index + 1];

        parents.push(page);

        let (substitute, leaf_node) =
            if self.node(left_child)?.entries.len() >= self.node(right_child)?.entries.len() {
                let predecesor = self.find_max(left_child, parents)?;
                let substitute = self.node_mut(predecesor)?.entries.pop().unwrap();
                (substitute, predecesor)
            } else {
                let successor = self.find_min(right_child, parents)?;
                let substitute = self.node_mut(successor)?.entries.remove(0);
                (substitute, successor)
            };

        let value = mem::replace(&mut self.node_mut(page)?.entries[index], substitute);

        return Ok(Some((value, leaf_node)));
    }

    fn find_max(&mut self, page: u32, parents: &mut Vec<u32>) -> io::Result<u32> {
        let node = self.cache.get(page)?;

        if node.is_leaf() {
            return Ok(page);
        }

        let index = node.children.len() - 1;

        let child = node.children[index];
        parents.push(page);

        self.find_max(child, parents)
    }

    fn find_min(&mut self, page: u32, parents: &mut Vec<u32>) -> io::Result<u32> {
        let node = self.cache.get(page)?;

        if node.is_leaf() {
            return Ok(page);
        }

        let child = node.children[0];
        parents.push(page);

        self.find_min(child, parents)
    }

    /// B*-Tree balancing algorithm. This algorithm attempts to delay node
    /// splitting and merging as much as possible while keeping all nodes at
    /// least 2/3 full except the root.
    ///
    /// TODO: Explain how.
    fn balance(&mut self, mut page: u32, mut parents: Vec<u32>) -> io::Result<()> {
        let max_keys = self.max_keys();

        let (is_root, node_entries_len) = {
            let node = self.node(page)?;
            (node.is_root(), node.entries.len())
        };

        // A node can only overflow if the number of keys is greater than the
        // maximum allowed.
        let is_overflow = node_entries_len > max_keys;

        // A node has underflowed if it contains less than 2/3 of keys. The root
        // is allowed to underflow past that point, but if it reaches 0 keys it
        // has completely underflowed and tree height must be decreased.
        let is_underflow =
            (node_entries_len < max_keys * 2 / 3 && !is_root) || node_entries_len == 0;

        // Done, this node didn't overflow or underflow.
        if !is_overflow && !is_underflow {
            return Ok(());
        }

        // Root underflow. Merging internal nodes has consumed the entire root.
        // Decrease tree height and return.
        if is_root && is_underflow {
            if let Some(child) = self.node_mut(page)?.children.pop() {
                self.append_node(page, child)?;

                // Free child page.
                self.free_page(child);
            }

            return Ok(());
        }

        // The root overflowed, so it must be split. We're not going to split
        // it here, we will only prepare the terrain for the splitting algorithm
        // below.
        if is_root && is_overflow {
            // The actual tree root always stays at the same page, it does not
            // move. First, copy the contents of the root to a new node and
            // leave the root empty.
            let mut old_root = Node::new_at(self.allocate_page());
            let root = self.node_mut(page)?;

            old_root.extend_by_draining(root);

            // Now make the new node a child of the empty root.
            root.children.push(old_root.page);

            // Turn the new node into the balance target. Since this node has
            // overflowed the code below will split it accordingly.
            parents.push(page);
            page = old_root.page;
            self.cache.load_from_mem(old_root)?;
        }

        // Find all siblings involved in the balancing algorithm, including
        // the given node (balance target).
        let (mut siblings, index, parent_page) = {
            let mut num_siblings_per_side = self.balance_siblings_per_side;

            let parent = self.node(parents.pop().unwrap())?;

            // TODO: Store this somewhere somehow.
            let index = parent.children.iter().position(|p| *p == page).unwrap();

            if index == 0 || index == parent.children.len() - 1 {
                num_siblings_per_side *= 2;
            };

            let left_siblings = index.checked_sub(num_siblings_per_side).unwrap_or(0)..index;

            let right_siblings =
                (index + 1)..min(index + num_siblings_per_side + 1, parent.children.len());

            let mut siblings =
                Vec::with_capacity(left_siblings.size_hint().0 + 1 + right_siblings.size_hint().0);

            for i in left_siblings {
                siblings.push((parent.children[i], i));
            }

            siblings.push((page, index));

            for i in right_siblings {
                siblings.push((parent.children[i], i));
            }

            (siblings, index, parent.page)
        };

        let total_entries = {
            let mut sum = 0;
            for sibling in &siblings {
                sum += self.node(sibling.0)?.entries.len();
            }
            sum
        };

        // We only split nodes if absolutely necessary. Otherwise we can just
        // move keys around siblings to keep balance. We reach the "must split"
        // point when all siblings are full and one of them has overflowed.
        let must_split = total_entries > max_keys * siblings.len();

        // Same as before, merge only if there's no other way to keep balance.
        // This happens when the node has underflowed below 66% and none of the
        // siblings can lend a key. When the root has only 2 children those are
        // considered an exception, they'll be merged as soon as possible.
        let must_merge = if siblings.len() == 2 {
            total_entries < max_keys
        } else {
            total_entries < max_keys * 2 / 3 * siblings.len()
        };

        if must_split {
            // Only two nodes needed for splitting.
            if siblings.len() > 2 {
                if siblings[0].1 == index {
                    siblings.drain(2..);
                } else if siblings.last().unwrap().1 == index {
                    siblings.drain(..siblings.len() - 2);
                } else {
                    siblings.retain(|s| [index, index + 1].contains(&s.1));
                }
            }

            // The last node is the one that splits.
            let rightmost_sibling = siblings.last_mut().unwrap();

            // Allocate new node.
            let new_page = self.allocate_page();
            let new_node_parent_index = rightmost_sibling.1 + 1;
            self.cache.load_from_mem(Node::new_at(new_page))?;

            let new_entry = self.node_mut(rightmost_sibling.0)?.entries.pop().unwrap();

            let parent = self.node_mut(parent_page)?;

            // Prepare terrain for the redistribution algorithm below by moving
            // the greatest key into the parent and adding the new empty child
            // to the parent.
            parent.entries.insert(rightmost_sibling.1, new_entry);
            parent.children.insert(new_node_parent_index, new_page);

            // Add new node to balancing list.
            siblings.push((new_page, new_node_parent_index));
        } else if must_merge {
            // Merge the first two nodes together, demote the first key in
            // the parent and let the redistribution algorithm below do its job.
            let (deleted_node, _) = siblings.remove(1);
            let (merge_node, divider_idx) = siblings[0];

            let new_entry = self.node_mut(parent_page)?.entries.remove(divider_idx);

            self.node_mut(merge_node)?.entries.push(new_entry);

            self.append_node(merge_node, deleted_node)?;

            self.node_mut(parent_page)?.children.remove(divider_idx + 1);
            self.free_page(deleted_node);
        }

        // This algorithm does most of the magic here. It prevents us from
        // splitting and merging by reordering keys around the nodes.
        self.redistribute_entries_and_children(parent_page, &mut siblings)?;

        // Recurse upwards and propagate redistribution/merging/splitting.
        self.balance(parent_page, parents)
    }

    /// Redistribute entries and children evenly. TODO: Explain algorithm.
    fn redistribute_entries_and_children(
        &mut self,
        parent_page: u32,
        siblings: &mut Vec<(u32, usize)>,
    ) -> io::Result<()> {
        let mut entries_to_balance = Vec::new();
        let mut children_to_balance = Vec::new();

        for node in siblings.iter_mut() {
            let node = self.cache.get_mut(node.0)?;
            entries_to_balance.extend(node.entries.drain(..));
            children_to_balance.extend(node.children.drain(..));
        }

        let num_siblings = siblings.len();

        let chunk_size = entries_to_balance.len() / num_siblings;
        let remainder = entries_to_balance.len() % num_siblings;

        let mut entries_start = 0;
        let mut children_start = 0;

        for (i, sibling) in siblings.iter_mut().enumerate() {
            let mut balanced_chunk_size = chunk_size;
            if i < remainder {
                balanced_chunk_size += 1;
            }

            // Swap keys with parent.
            if i < num_siblings - 1 {
                let swap_key_idx = entries_start + balanced_chunk_size;

                let mut keys = [
                    entries_to_balance[swap_key_idx - 1],
                    self.node(parent_page)?.entries[sibling.1],
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

                if let Some(mut swap_key_idx) = maybe_swap {
                    let parent = self.node_mut(parent_page)?;

                    mem::swap(
                        &mut parent.entries[sibling.1],
                        &mut entries_to_balance[swap_key_idx],
                    );

                    // Sort demoted keys.
                    while swap_key_idx > 0
                        && entries_to_balance[swap_key_idx - 1] > entries_to_balance[swap_key_idx]
                    {
                        entries_to_balance.swap(swap_key_idx - 1, swap_key_idx);
                        swap_key_idx -= 1;
                    }
                    while swap_key_idx < entries_to_balance.len() - 1
                        && entries_to_balance[swap_key_idx + 1] < entries_to_balance[swap_key_idx]
                    {
                        entries_to_balance.swap(swap_key_idx + 1, swap_key_idx);
                        swap_key_idx += 1;
                    }
                }
            }

            let sibling = self.node_mut(sibling.0)?;

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

        Ok(())
    }

    fn allocate_page(&mut self) -> u32 {
        // TODO: Free list.
        let page = self.len;
        self.len += 1;

        page as u32
    }

    fn free_page(&mut self, page: u32) {
        self.cache.invalidate(page);

        // TODO: Free list.
        self.cache
            .pager
            .write_page(page as usize, &Vec::from("FREE PAGE".as_bytes()))
            .unwrap();

        self.len -= 1;
    }

    // Testing/Debugging only.
    fn read_into_mem(&mut self, node: Node, buf: &mut Vec<Node>) -> io::Result<()> {
        for page in &node.children {
            let child = self.cache.pager.read_node(*page)?;
            self.read_into_mem(child, buf)?;
        }

        buf.push(node);

        Ok(())
    }

    pub fn json(&mut self) -> io::Result<String> {
        let root = self.cache.pager.read_node(0)?;

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

    use super::{BTree, BALANCE_SIBLINGS_PER_SIDE};
    use crate::{cache::Cache, pager::Pager};

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

    /// Config/Builder for [`BTree<MemBuf>`]. Can be used with
    /// [`TryFrom::try_from`] or [`BTree::builder`].
    struct Config {
        keys: Vec<u32>,
        order: usize,
        balance_siblings_per_side: usize,
    }

    impl Default for Config {
        fn default() -> Self {
            Config {
                keys: vec![],
                order: 4,
                balance_siblings_per_side: BALANCE_SIBLINGS_PER_SIDE,
            }
        }
    }

    impl Config {
        fn keys<K: IntoIterator<Item = u32>>(mut self, keys: K) -> Self {
            self.keys = keys.into_iter().collect();
            self
        }

        fn order(mut self, order: usize) -> Self {
            self.order = order;
            self
        }

        fn balance_siblings_per_side(mut self, balance_siblings_per_side: usize) -> Self {
            self.balance_siblings_per_side = balance_siblings_per_side;
            self
        }

        fn try_build(self) -> io::Result<BTree<MemBuf>> {
            BTree::try_from(self)
        }
    }

    /// We use in-memory buffers instead of disk files for testing. This speeds
    /// up tests as it avoids disk IO and system calls.
    type MemBuf = io::Cursor<Vec<u8>>;

    impl BTree<MemBuf> {
        fn into_test_nodes(&mut self, root: u32) -> io::Result<Node> {
            let node = self.cache.pager.read_node(root)?;
            let mut test_node = Node {
                keys: node.entries.iter().map(|e| e.key).collect(),
                children: vec![],
            };

            for page in &node.children {
                let child = self.cache.pager.read_node(*page)?;
                test_node.children.push(self.into_test_nodes(child.page)?);
            }

            Ok(test_node)
        }

        fn builder() -> Config {
            Config::default()
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
    }

    impl TryFrom<Config> for BTree<MemBuf> {
        type Error = io::Error;

        fn try_from(config: Config) -> Result<Self, Self::Error> {
            let page_size = optimal_page_size_for(config.order);
            let buf = io::Cursor::new(Vec::new());

            let mut btree = BTree::new(
                Cache::new(Pager::new(buf, page_size, page_size)).with_order(config.order),
                config.balance_siblings_per_side,
            );
            btree.extend_from_keys_only(config.keys)?;

            Ok(btree)
        }
    }

    impl TryFrom<BTree<MemBuf>> for Node {
        type Error = io::Error;

        fn try_from(mut btree: BTree<MemBuf>) -> Result<Self, Self::Error> {
            btree.into_test_nodes(0)
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
        let btree = BTree::builder().keys(1..=3).try_build()?;

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
        let btree = BTree::builder().keys(1..=4).try_build()?;

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
        let btree = BTree::builder().keys(1..=7).try_build()?;

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
        let btree = BTree::builder().keys(1..=8).try_build()?;

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
        let btree = BTree::builder().keys(1..=15).try_build()?;

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
        let btree = BTree::builder().keys(1..=16).try_build()?;

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
        let btree = BTree::builder().keys(1..=27).try_build()?;

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
        let btree = BTree::builder().keys(1..=31).try_build()?;

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

    /// This is how a [`BTree`] structure of `order = 4` should look like after
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
        let btree = BTree::builder().keys(1..=46).try_build()?;

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
        let mut btree = BTree::builder().keys(1..=15).try_build()?;

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

    /// Deleting from internal nodes should substitute the key with either the
    /// predecessor in the left subtree or the successor in the right subtree.
    ///
    /// ```text
    ///                           DELETE 8
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
    ///
    ///                            RESULT
    ///                               |
    ///                               V
    ///                           +--------+
    ///                   +-------|   11   |-------+
    ///                  /        +--------+        \
    ///                 /                            \
    ///            +-------+                     +--------+
    ///       +----|  4,7  |----+                |   14   |
    ///      /     +-------+     \               +--------+
    ///     /          |          \               /      \
    /// +-------+  +-------+  +-------+     +-------+  +-------+
    /// | 1,2,3 |  | 5,6   |  | 9,10  |     | 12,13 |  | 15,16 |
    /// +-------+  +-------+  +-------+     +-------+  +-------+
    /// ```
    #[test]
    fn delete_from_internal_node() -> io::Result<()> {
        let mut btree = BTree::builder().keys(1..=16).try_build()?;

        btree.remove(8)?;

        assert_eq!(
            Node::try_from(btree)?,
            Node {
                keys: vec![11],
                children: vec![
                    Node {
                        keys: vec![4, 7],
                        children: vec![
                            Node::leaf([1, 2, 3]),
                            Node::leaf([5, 6]),
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

    /// Deleting from root is the same as deleting from internal nodes.
    ///
    /// ```text
    ///                           DELETE 11
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
    ///
    ///                             RESULT
    ///                               |
    ///                               V
    ///                           +--------+
    ///                   +-------|   10   |-------+
    ///                  /        +--------+        \
    ///                 /                            \
    ///            +-------+                     +--------+
    ///       +----|  4,7  |----+                |   14   |
    ///      /     +-------+     \               +--------+
    ///     /          |          \               /      \
    /// +-------+  +-------+  +-------+     +-------+  +-------+
    /// | 1,2,3 |  | 5,6   |  | 8,9   |     | 12,13 |  | 15,16 |
    /// +-------+  +-------+  +-------+     +-------+  +-------+
    /// ```
    #[test]
    fn delete_from_root() -> io::Result<()> {
        let mut btree = BTree::builder().keys(1..=16).try_build()?;

        btree.remove(11)?;

        assert_eq!(
            Node::try_from(btree)?,
            Node {
                keys: vec![10],
                children: vec![
                    Node {
                        keys: vec![4, 7],
                        children: vec![
                            Node::leaf([1, 2, 3]),
                            Node::leaf([5, 6]),
                            Node::leaf([8, 9]),
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

    /// When applicable, the successor is used instead of the predecessor for
    /// delete operations (when the right child has more keys than the left
    /// child).
    ///
    /// ```text
    ///                                     DELETE 11
    ///                                         |
    ///                                         V
    ///                                     +--------+
    ///                   +-----------------|   11   |--------------+
    ///                  /                  +--------+               \
    ///                 /                                             \
    ///            +-------+                                     +----------+
    ///       +----|  4,8  |----+                   +------------| 15,19,23 |------------+
    ///      /     +-------+     \                 /             +----------+             \
    ///     /          |          \               /                /      \                \
    /// +-------+  +-------+  +-------+     +----------+  +----------+  +----------+  +----------+
    /// | 1,2,3 |  | 5,6,7 |  | 9,10  |     | 12,13,14 |  | 16,17,18 |  | 20,21,22 |  | 24,25,26 |
    /// +-------+  +-------+  +-------+     +----------+  +----------+  +----------+  +----------+
    ///
    ///                                       RESULT
    ///                                         |
    ///                                         V
    ///                                     +--------+
    ///                   +-----------------|   12   |--------------+
    ///                  /                  +--------+               \
    ///                 /                                             \
    ///            +-------+                                     +----------+
    ///       +----|  4,8  |----+                   +------------| 15,19,23 |------------+
    ///      /     +-------+     \                 /             +----------+             \
    ///     /          |          \               /                /      \                \
    /// +-------+  +-------+  +-------+     +----------+  +----------+  +----------+  +----------+
    /// | 1,2,3 |  | 5,6,7 |  | 9,10  |     | 13,14    |  | 16,17,18 |  | 20,21,22 |  | 24,25,26 |
    /// +-------+  +-------+  +-------+     +----------+  +----------+  +----------+  +----------+
    /// ```
    #[test]
    fn delete_using_successor_instead_of_predecessor() -> io::Result<()> {
        let mut btree = BTree::builder().keys(1..=26).try_build()?;

        btree.remove(11)?;

        assert_eq!(
            Node::try_from(btree)?,
            Node {
                keys: vec![12],
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
                        keys: vec![15, 19, 23],
                        children: vec![
                            Node::leaf([13, 14]),
                            Node::leaf([16, 17, 18]),
                            Node::leaf([20, 21, 22]),
                            Node::leaf([24, 25, 26]),
                        ]
                    }
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
        let mut btree = BTree::builder().keys(1..=15).try_build()?;

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
        let mut btree = BTree::builder().keys(1..=15).try_build()?;

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

    /// When the root has two children and they get merged, then the merged node
    /// should become the new root, decreasing tree height by one.
    ///
    /// ```text
    ///                 DELETE 4
    ///                    |
    ///                    v
    ///                +-------+
    ///                |   3   |
    ///                +-------+
    ///               /         \
    ///           +-------+  +-------+
    ///           |  1,2  |  |   4   |
    ///           +-------+  +-------+
    ///
    ///                 RESULT
    ///                    |
    ///                    v
    ///                +-------+
    ///                | 1,2,3 |
    ///                +-------+
    /// ```
    #[test]
    fn merge_root() -> io::Result<()> {
        let mut btree = BTree::builder().keys(1..=4).try_build()?;

        btree.remove(4)?;

        assert_eq!(Node::try_from(btree)?, Node::leaf([1, 2, 3]));

        Ok(())
    }

    /// Internal nodes won't be merged if keys can be reordered around.
    ///
    /// ```text
    ///                                    DELETE (1,2,3) -> Merge leftmost leaf nodes
    ///                                                        |
    ///                                                        V
    ///                                                    +-------+
    ///                  +---------------------------------| 11,23 |----------------------------------------+
    ///                 /                                  +-------+                                         \
    ///                /                                       |                                              \
    ///             +-----+                                +-------+                                     +----------+
    ///        +----| 4,8 |-----+                  +-------| 15,19 |-------+                   +---------| 27,30,33 |-----+
    ///       /     +-----+      \                /        +-------+        \                 /          +----------+      \
    ///      /         |          \              /             |             \               /             /      \         \
    /// +-------+  +-------+  +-------+    +----------+  +----------+  +----------+    +----------+  +-------+  +-------+  +-------+
    /// | 1,2,3 |  | 5,6,7 |  | 9,10  |    | 12,13,14 |  | 16,17,18 |  | 20,21,22 |    | 24,25,26 |  | 28,29 |  | 31,32 |  | 34,35 |
    /// +-------+  +-------+  +-------+    +----------+  +----------+  +----------+    +----------+  +-------+  +-------+  +-------+
    ///
    ///                                                         RESULT
    ///                                                            |
    ///                                                            V
    ///                                                        +-------+
    ///                   +------------------------------------| 15,27 |---------------------------------+
    ///                  /                                     +-------+                                  \
    ///                 /                                          |                                       \
    ///             +------+                                   +-------+                               +-------+
    ///        +----| 7,11 |-----+                      +------| 19,23 |-----+                     +---| 30,33 |----+
    ///       /     +------+      \                    /       +-------+      \                   /    +-------+     \
    ///      /         |           \                  /            |           \                 /         |          \
    /// +-------+  +--------+   +----------+    +----------+  +----------+  +----------+    +-------+  +-------+  +-------+
    /// | 4,5,6 |  | 8,9,10 |   | 12,13,14 |    | 16,17,18 |  | 20,21,22 |  | 24,25,26 |    | 28,29 |  | 31,32 |  | 34,35 |
    /// +-------+  +--------+   +----------+    +----------+  +----------+  +----------+    +-------+  +-------+  +-------+
    /// ```
    #[test]
    fn delay_internal_node_merge() -> io::Result<()> {
        let mut btree = BTree::builder().keys(1..=35).try_build()?;

        btree.try_remove_all(1..=3)?;

        assert_eq!(
            Node::try_from(btree)?,
            Node {
                keys: vec![15, 27],
                children: vec![
                    Node {
                        keys: vec![7, 11],
                        children: vec![
                            Node::leaf([4, 5, 6]),
                            Node::leaf([8, 9, 10]),
                            Node::leaf([12, 13, 14]),
                        ]
                    },
                    Node {
                        keys: vec![19, 23],
                        children: vec![
                            Node::leaf([16, 17, 18]),
                            Node::leaf([20, 21, 22]),
                            Node::leaf([24, 25, 26]),
                        ]
                    },
                    Node {
                        keys: vec![30, 33],
                        children: vec![
                            Node::leaf([28, 29]),
                            Node::leaf([31, 32]),
                            Node::leaf([34, 35]),
                        ]
                    }
                ]
            }
        );

        Ok(())
    }

    /// When merging is unavoidable... then merge.
    ///
    /// ```text
    ///                                    DELETE (1,2,3) -> Merge leftmost leaf nodes
    ///                                                        |
    ///                                                        V
    ///                                                    +-------+
    ///                  +---------------------------------| 11,23 |----------------------------------------+
    ///                 /                                  +-------+                                         \
    ///                /                                       |                                              \
    ///             +-----+                                +-------+                                     +----------+
    ///        +----| 4,8 |-----+                  +-------| 15,19 |-------+                   +---------| 27,30,33 |-----+
    ///       /     +-----+      \                /        +-------+        \                 /          +----------+      \
    ///      /         |          \              /             |             \               /             /      \         \
    /// +-------+  +-------+  +-------+    +----------+  +----------+  +----------+    +----------+  +-------+  +-------+  +-------+
    /// | 1,2,3 |  | 5,6,7 |  | 9,10  |    | 12,13,14 |  | 16,17,18 |  | 20,21,22 |    | 24,25,26 |  | 28,29 |  | 31,32 |  | 34,35 |
    /// +-------+  +-------+  +-------+    +----------+  +----------+  +----------+    +----------+  +-------+  +-------+  +-------+
    ///
    ///                                           DELETE 35 -> Merge Two Internal Nodes
    ///                                                            |
    ///                                                            V
    ///                                                        +-------+
    ///                   +------------------------------------| 15,27 |---------------------------------+
    ///                  /                                     +-------+                                  \
    ///                 /                                          |                                       \
    ///             +------+                                   +-------+                               +-------+
    ///        +----| 7,11 |-----+                      +------| 19,23 |-----+                     +---| 30,33 |----+
    ///       /     +------+      \                    /       +-------+      \                   /    +-------+     \
    ///      /         |           \                  /            |           \                 /         |          \
    /// +-------+  +--------+   +----------+    +----------+  +----------+  +----------+    +-------+  +-------+  +-------+
    /// | 4,5,6 |  | 8,9,10 |   | 12,13,14 |    | 16,17,18 |  | 20,21,22 |  | 24,25,26 |    | 28,29 |  | 31,32 |  | 34,35 |
    /// +-------+  +--------+   +----------+    +----------+  +----------+  +----------+    +-------+  +-------+  +-------+
    ///
    ///                                               FINAL RESULT
    ///                                                     |
    ///                                                     V
    ///                                                  +----+
    ///                         +------------------------| 19 |----------------------+
    ///                        /                         +----+                       \
    ///                       /                                                        \
    ///                  +---------+                                              +----------+
    ///        +---------| 7,11,15 |----------+                      +------------| 23,27,31 |------------+
    ///       /          +---------+           \                    /             +----------+             \
    ///      /            /      \              \                  /                /      \                \
    /// +-------+  +--------+  +----------+  +----------+    +----------+  +----------+  +----------+  +----------+
    /// | 4,5,6 |  | 8,9,10 |  | 12,13,14 |  | 16,17,18 |    | 20,21,22 |  | 24,25,26 |  | 28,29,30 |  | 32,33,34 |
    /// +-------+  +--------+  +----------+  +----------+    +----------+  +----------+  +----------+  +----------+
    /// ```
    #[test]
    fn merge_internal_node() -> io::Result<()> {
        let mut btree = BTree::builder().keys(1..=35).try_build()?;

        btree.try_remove_all(1..=3).and_then(|_| btree.remove(35))?;

        assert_eq!(
            Node::try_from(btree)?,
            Node {
                keys: vec![19],
                children: vec![
                    Node {
                        keys: vec![7, 11, 15],
                        children: vec![
                            Node::leaf([4, 5, 6]),
                            Node::leaf([8, 9, 10]),
                            Node::leaf([12, 13, 14]),
                            Node::leaf([16, 17, 18]),
                        ]
                    },
                    Node {
                        keys: vec![23, 27, 31],
                        children: vec![
                            Node::leaf([20, 21, 22]),
                            Node::leaf([24, 25, 26]),
                            Node::leaf([28, 29, 30]),
                            Node::leaf([32, 33, 34]),
                        ]
                    },
                ]
            }
        );

        Ok(())
    }

    fn check_return_value(
        method: fn(btree: &mut BTree<MemBuf>, key: u32) -> io::Result<Option<u32>>,
    ) -> io::Result<()> {
        let compute_value = |key| key + 1000;
        let keys = 1..=46;

        let mut btree = BTree::builder().try_build()?;
        btree.extend(keys.clone().map(|key| (key, compute_value(key))));

        for key in keys {
            assert_eq!(method(&mut btree, key)?, Some(compute_value(key)));
        }

        assert_eq!(method(&mut btree, 1000)?, None);

        Ok(())
    }

    #[test]
    fn get_value() -> io::Result<()> {
        check_return_value(BTree::get)
    }

    #[test]
    fn remove_value() -> io::Result<()> {
        check_return_value(BTree::remove)
    }

    /// Most tests use `order = 4` for simplicty. This one uses `order = 6` to
    /// check if everything still works. This is what we're going to build:
    ///
    /// ```text
    ///
    ///                                                INSERT 36
    ///                                                    |
    ///                                                    v
    ///                                            +---------------+
    ///           +--------------------------------| 6,12,18,24,30 |------------------------------------+
    ///          /                                 +---------------+                                     \
    ///         /                                    |  |     |  |                                        \
    ///        /                +--------------------+  |     |  +----------------------+                  \
    ///       /                /                        |     |                          \                  \
    /// +-----------+  +-------------+  +----------------+  +----------------+  +----------------+  +----------------+
    /// | 1,2,3,4,5 |  | 7,8,9,10,11 |  | 13,14,15,16,17 |  | 19,20,21,22,23 |  | 25,26,27,28,29 |  | 31,32,33,34,35 |
    /// +-----------+  +-------------+  +----------------+  +----------------+  +----------------+  +----------------+
    ///
    ///                                                           RESULT
    ///                                                              |
    ///                                                              V
    ///                                                           +----+
    ///                                  +------------------------| 24 |---------------------------+
    ///                                 /                         +----+                            \
    ///                                /                                                             \
    ///                           +---------+                                                     +-------+
    ///          +----------------| 6,12,18 |---------------------+                         +-----| 29,33 |-----+
    ///         /                 +---------+                      \                       /      +-------+      \
    ///        /                   /      \                         \                     /           |           \
    /// +-----------+  +-------------+  +----------------+  +----------------+  +-------------+  +----------+  +----------+
    /// | 1,2,3,4,5 |  | 7,8,9,10,11 |  | 13,14,15,16,17 |  | 19,20,21,22,23 |  | 25,26,27,28 |  | 30,31,32 |  | 34,35,36 |
    /// +-----------+  +-------------+  +----------------+  +----------------+  +-------------+  +----------+  +----------+
    /// ```
    #[test]
    fn greater_order_insertion() -> io::Result<()> {
        let btree = BTree::builder().order(6).keys(1..=36).try_build()?;

        assert_eq!(
            Node::try_from(btree)?,
            Node {
                keys: vec![24],
                children: vec![
                    Node {
                        keys: vec![6, 12, 18],
                        children: vec![
                            Node::leaf([1, 2, 3, 4, 5]),
                            Node::leaf([7, 8, 9, 10, 11]),
                            Node::leaf([13, 14, 15, 16, 17]),
                            Node::leaf([19, 20, 21, 22, 23]),
                        ]
                    },
                    Node {
                        keys: vec![29, 33],
                        children: vec![
                            Node::leaf([25, 26, 27, 28]),
                            Node::leaf([30, 31, 32]),
                            Node::leaf([34, 35, 36]),
                        ]
                    },
                ]
            }
        );

        Ok(())
    }

    /// Delete on `order = 6`.
    ///
    /// ```text
    ///
    ///                                                      DELETE (34,35,36)
    ///                                                              |
    ///                                                              V
    ///                                                           +----+
    ///                                  +------------------------| 24 |---------------------------+
    ///                                 /                         +----+                            \
    ///                                /                                                             \
    ///                           +---------+                                                     +-------+
    ///          +----------------| 6,12,18 |---------------------+                         +-----| 29,33 |-----+
    ///         /                 +---------+                      \                       /      +-------+      \
    ///        /                   /      \                         \                     /           |           \
    /// +-----------+  +-------------+  +----------------+  +----------------+  +-------------+  +----------+  +----------+
    /// | 1,2,3,4,5 |  | 7,8,9,10,11 |  | 13,14,15,16,17 |  | 19,20,21,22,23 |  | 25,26,27,28 |  | 30,31,32 |  | 34,35,36 |
    /// +-----------+  +-------------+  +----------------+  +----------------+  +-------------+  +----------+  +----------+
    ///
    ///                                                 RESULT
    ///                                                    |
    ///                                                    V
    ///                                            +---------------+
    ///           +--------------------------------| 6,12,18,24,30 |---------------------------------+
    ///          /                                 +---------------+                                  \
    ///         /                                    |  |     |  |                                     \
    ///        /                +--------------------+  |     |  +----------------------+               \
    ///       /                /                        |     |                          \               \
    /// +-----------+  +-------------+  +----------------+  +----------------+  +----------------+  +----------+
    /// | 1,2,3,4,5 |  | 7,8,9,10,11 |  | 13,14,15,16,17 |  | 19,20,21,22,23 |  | 25,26,27,28,29 |  | 31,32,33 |
    /// +-----------+  +-------------+  +----------------+  +----------------+  +----------------+  +----------+
    /// ```
    #[test]
    fn greater_order_deletion() -> io::Result<()> {
        let mut btree = BTree::builder().order(6).keys(1..=36).try_build()?;

        btree.try_remove_all(34..=36)?;

        assert_eq!(
            Node::try_from(btree)?,
            Node {
                keys: vec![6, 12, 18, 24, 30],
                children: vec![
                    Node::leaf([1, 2, 3, 4, 5]),
                    Node::leaf([7, 8, 9, 10, 11]),
                    Node::leaf([13, 14, 15, 16, 17]),
                    Node::leaf([19, 20, 21, 22, 23]),
                    Node::leaf([25, 26, 27, 28, 29]),
                    Node::leaf([31, 32, 33]),
                ]
            }
        );

        Ok(())
    }

    /// See [`merge_leaf_node`]. In that test `balance_siblings_per_side = 1`,
    /// in this case `balance_siblings_per_side = 2` which will delay merging
    /// and splitting even further (increasing IO, so it's a tradeoff).
    ///
    /// ```text
    ///                   DELETE (15,14,13,12) -> No Merge
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
    ///                           FINAL RESULT
    ///                                |
    ///                                V
    ///                            +-------+
    ///                      +-----| 3,6,9 |------+
    ///                     /      +-------+       \
    ///                    /        /     \         \
    ///               +-----+  +-----+  +-----+  +-------+
    ///               | 1,2 |  | 4,5 |  | 7,8 |  | 10,11 |
    ///               +-----+  +-----+  +-----+  +-------+
    /// ```
    #[test]
    fn increased_balance_siblings_per_side() -> io::Result<()> {
        let mut btree = BTree::builder()
            .balance_siblings_per_side(2)
            .keys(1..=15)
            .try_build()?;

        btree.try_remove_all((12..=15).rev())?;

        assert_eq!(
            Node::try_from(btree)?,
            Node {
                keys: vec![3, 6, 9],
                children: vec![
                    Node::leaf([1, 2]),
                    Node::leaf([4, 5]),
                    Node::leaf([7, 8]),
                    Node::leaf([10, 11]),
                ]
            }
        );

        Ok(())
    }
}
