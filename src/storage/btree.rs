//! Disk BTree data structure. See [`BTree`] for details.

use std::{
    cmp::{min, Ordering},
    fs::File,
    io,
    io::{Read, Seek, Write},
    path::Path,
};

use super::page::{Cell, Page};
use crate::{
    os::{Disk, HardwareBlockSize},
    paging::{
        cache::Cache,
        pager::{PageNumber, Pager},
    },
};

pub(crate) trait BytesCmp {
    fn bytes_cmp(&self, a: &[u8], b: &[u8]) -> Ordering;
}

pub struct MemCmp;

impl BytesCmp for MemCmp {
    fn bytes_cmp(&self, a: &[u8], b: &[u8]) -> Ordering {
        let size = std::cmp::min(a.len(), b.len());
        a[..size].cmp(&b[..size])
    }
}

/// B*-Tree implementation inspired by "Art of Computer Programming Volume 3:
/// Sorting and Searching" and SQLite 2.X.X
pub(crate) struct BTree<F, C: BytesCmp = MemCmp> {
    /// Read-Write page cache.
    cache: Cache<F>,

    /// Bytes comparator used to obtain [`Ordering`] instances from binary data.
    comparator: C,

    /// Number of siblings to examine at each side when balancing a node.
    /// See [`Self::load_siblings`].
    balance_siblings_per_side: usize,

    len: usize, // TODO: Free list
}

/// Default value for [`BTree::balance_siblings_per_side`].
const BALANCE_SIBLINGS_PER_SIDE: usize = 1;

/// The result of a search in the [`BTree`] structure.
struct Search {
    /// Page number of the node where the search ended.
    page: PageNumber,

    /// If the search was successful, this stores a copy of [`Entry::value`].
    index: Result<u16, u16>,
}

impl<F> BTree<F> {
    #[allow(dead_code)]
    pub fn new(cache: Cache<F>, balance_siblings_per_side: usize) -> Self {
        Self {
            cache,
            comparator: MemCmp,
            balance_siblings_per_side,
            len: 1,
        }
    }
}

impl<F, C: BytesCmp> BTree<F, C> {
    #[allow(dead_code)]
    pub fn new_with_comparator(
        cache: Cache<F>,
        balance_siblings_per_side: usize,
        comparator: C,
    ) -> Self {
        Self {
            cache,
            comparator,
            balance_siblings_per_side,
            len: 1,
        }
    }
}

impl<C: BytesCmp> BTree<File, C> {
    /// Creates a new [`BTree`] at the given `path` in the file system.
    pub fn new_at_path_with_comparator<P: AsRef<Path>>(
        path: P,
        page_size: usize,
        comparator: C,
    ) -> io::Result<Self> {
        let file = File::options()
            .create(true)
            .read(true)
            .write(true)
            .open(&path)?;

        let metadata = file.metadata()?;

        if !metadata.is_file() {
            return Err(io::Error::new(io::ErrorKind::Unsupported, "Not a file"));
        }

        let block_size = Disk::from(&path).block_size()?;

        // TODO: Add magic number & header to file.
        let mut cache = Cache::new(Pager::new(file, page_size, block_size));
        let balance_siblings_per_side = BALANCE_SIBLINGS_PER_SIDE;

        // TODO: Init root while we don't have a header.
        let mut current_root = Page::new(0, page_size as _);
        cache.pager.read(0, current_root.buffer_mut())?;
        if current_root.header().free_space() == 0 {
            let new_root = Page::new(0, page_size as _);
            cache.pager.write(new_root.number, new_root.buffer())?;
        }

        // TODO: Take free pages into account.
        let len = std::cmp::max(metadata.len() as usize / page_size, 1);

        Ok(Self {
            cache,
            len,
            comparator,
            balance_siblings_per_side,
        })
    }
}

impl BTree<File> {
    /// Creates a new [`BTree`] at the given `path` in the file system.
    pub fn new_at_path<P: AsRef<Path>>(path: P, page_size: usize) -> io::Result<Self> {
        Self::new_at_path_with_comparator(path, page_size, MemCmp)
    }
}

enum LeafKeyType {
    /// Maximum key in a leaf node. Located at the last index of [`Node::entries`].
    Max,
    /// Minimum key in a leaf node. Located at the first index of [`Node::entries`].
    Min,
}

struct Sibling {
    page: PageNumber,
    index: u16,
}

impl Sibling {
    fn new(page: PageNumber, index: u16) -> Self {
        Self { page, index }
    }
}

impl<F: Seek + Read + Write, C: BytesCmp> BTree<F, C> {
    /// Returns the value corresponding to the key. See [`Self::search`] for
    /// details.
    pub fn get(&mut self, entry: &[u8]) -> io::Result<Option<&[u8]>> {
        let search = self.search(0, entry, &mut Vec::new())?;

        match search.index {
            Err(_) => Ok(None),
            Ok(index) => Ok(Some(self.cache.get(search.page)?.cell(index).content)),
        }
    }

    fn search(
        &mut self,
        page: PageNumber,
        entry: &[u8],
        parents: &mut Vec<PageNumber>,
    ) -> io::Result<Search> {
        let node = self.cache.get(page)?;

        // Search key in this node.
        let index = node.binary_search_by(|buf| self.comparator.bytes_cmp(buf, entry));

        // We found the key or we're already at the bottom, stop recursion.
        if index.is_ok() || node.is_leaf() {
            return Ok(Search { page, index });
        }

        // No luck, keep recursing downwards.
        parents.push(page);
        let next_node = node.child(index.unwrap_err());

        self.search(next_node, entry, parents)
    }

    pub fn insert(&mut self, entry: &[u8]) -> io::Result<()> {
        let mut parents = Vec::new();
        let search = self.search(0, entry, &mut parents)?;
        let node = self.cache.get_mut(search.page)?;

        match search.index {
            // Key found, swap value.
            Ok(index) => {
                let mut cell = Cell::new(entry);
                cell.header.left_child = node.child(index);
                node.replace(index, Cell::new(entry));
            }
            // Key not found, insert new entry.
            Err(index) => node.insert(index, Cell::new(entry)),
        };

        self.balance(search.page, parents)?;
        self.cache.flush_write_queue_to_disk()
    }

    pub fn remove(&mut self, entry: &[u8]) -> io::Result<Option<Box<[u8]>>> {
        let mut parents = Vec::new();
        let Some((entry, leaf_node)) = self.remove_entry(entry, &mut parents)? else {
            return Ok(None);
        };

        self.balance(leaf_node, parents)?;
        self.cache.flush_write_queue_to_disk()?;

        Ok(Some(entry))
    }

    /// Finds the node where `key` is located and removes it from the entry
    /// list, replacing it with either its predecessor or successor in the case
    /// of internal nodes. The returned tuple contains the removed entry and the
    /// page number of the leaf node that was used to substitute the key.
    /// [`Self::balance`] must be called on the leaf node after this operation.
    /// See [`Self::remove`] for more details.
    fn remove_entry(
        &mut self,
        entry: &[u8],
        parents: &mut Vec<u32>,
    ) -> io::Result<Option<(Box<[u8]>, u32)>> {
        let search = self.search(0, entry, parents)?;
        let node = self.cache.get(search.page)?;

        // Can't remove entry, key not found.
        if search.index.is_err() {
            return Ok(None);
        }

        let index = search.index.unwrap();

        // Leaf node is the simplest case, remove key and pop off the stack.
        if node.is_leaf() {
            let cell = self.cache.get_mut(search.page)?.remove(index);
            return Ok(Some((cell.content, search.page)));
        }

        // Root or internal nodes require additional work. We need to find a
        // suitable substitute for the key in one of the leaf nodes. We'll
        // pick either the predecessor (max key in the left subtree) or the
        // successor (min key in the right subtree) of the key we want to
        // delete.
        let left_child = node.child(index);
        let right_child = node.child(index + 1);

        parents.push(search.page);
        let (leaf_node, key_idx) =
            if self.cache.get(left_child)?.len() >= self.cache.get(right_child)?.len() {
                self.search_max_key(left_child, parents)?
            } else {
                self.search_min_key(right_child, parents)?
            };

        let mut substitute = self.cache.get_mut(leaf_node)?.remove(key_idx);

        let node = self.cache.get_mut(search.page)?;

        substitute.header.left_child = node.child(index);
        let entry = node.replace(index, substitute);

        Ok(Some((entry.content, leaf_node)))
    }

    /// Traverses the tree all the way down to the leaf nodes, following the
    /// path specified by [`LeafKeyType`]. [`LeafKeyType::Max`] will always
    /// choose the last child for recursion, while [`LeafKeyType::Min`] will
    /// always choose the first child. This function is used to find successors
    /// or predecessors of keys in internal nodes.
    fn search_leaf_key(
        &mut self,
        page: PageNumber,
        parents: &mut Vec<PageNumber>,
        leaf_key_type: LeafKeyType,
    ) -> io::Result<(PageNumber, u16)> {
        let node = self.cache.get(page)?;

        let (key_idx, child_idx) = match leaf_key_type {
            LeafKeyType::Min => (0, 0),
            LeafKeyType::Max => (node.len() - 1, node.len()),
        };

        if node.is_leaf() {
            return Ok((page, key_idx));
        }

        parents.push(page);
        let child = node.child(child_idx);

        self.search_leaf_key(child, parents, leaf_key_type)
    }

    /// Returns the page and index in [`Node::entries`] where the greatest key
    /// of the given subtree is located.
    fn search_max_key(
        &mut self,
        page: PageNumber,
        parents: &mut Vec<PageNumber>,
    ) -> io::Result<(PageNumber, u16)> {
        self.search_leaf_key(page, parents, LeafKeyType::Max)
    }

    /// Returns the page and index in [`Node::entries`] where the smallest key
    /// of the given subtree is located.
    fn search_min_key(
        &mut self,
        page: PageNumber,
        parents: &mut Vec<PageNumber>,
    ) -> io::Result<(PageNumber, u16)> {
        self.search_leaf_key(page, parents, LeafKeyType::Min)
    }

    /// Drops a currently allocated page.
    fn free_page(&mut self, page: PageNumber) {
        self.cache.invalidate(page);

        let mut buf = vec![0; self.cache.pager.page_size];
        let s = "FREE PAGE".as_bytes();
        buf[..s.len()].copy_from_slice(s);

        // TODO: Free list.
        self.cache.pager.write(page, &buf).unwrap();

        // self.len -= 1; // TODO: Next free page is grabed from here
    }

    /// B*-Tree balancing algorithm inspired by (or rather stolen from) SQLite
    /// 2.X.X. Take a look at the original source code here:
    ///
    /// https://github.com/antoniosarosi/sqlite2-btree-visualizer/blob/master/src/btree.c#L2171
    ///
    /// This is how it works:
    ///
    /// 1. Check for root overflow or underflow. The root is only considered
    /// underflow when it contains 0 cells and has one child:
    ///
    /// ```text
    ///                      +--------+
    ///                      |        | Empty root
    ///                      +--------+
    ///                           |
    ///                           v
    ///                      +--------+ Direct root child
    ///                  +---| 4,8,11 |----+
    ///                 /    +--------+     \
    ///                /          |          \
    ///            +-------+  +-------+  +----------+
    ///            | 1,2,3 |  | 5,6,7 |  | 12,13,14 | Other pages below
    ///            +-------+  +-------+  +----------+
    /// ```
    ///
    /// In that case, move the child into the root decreasing tree height by
    /// one, free the child page and return:
    ///
    /// ```text
    ///                       New root
    ///                      +--------+
    ///                  +---| 4,8,11 |----+
    ///                 /    +--------+     \
    ///                /          |          \
    ///            +-------+  +-------+  +----------+
    ///            | 1,2,3 |  | 5,6,7 |  | 12,13,14 | Other pages below
    ///            +-------+  +-------+  +----------+
    /// ```
    ///
    /// TODO
    fn balance(&mut self, mut page: PageNumber, mut parents: Vec<PageNumber>) -> io::Result<()> {
        let node = self.cache.get(page)?;

        if !node.is_overflow() && !node.is_underflow() {
            return Ok(());
        }

        // Root underflow
        if node.is_root() && node.is_underflow() {
            let child_page = node.header().right_child;
            let mut child_node = self.cache.get(child_page)?.clone();
            self.cache.get_mut(page)?.append(&mut child_node);
            self.free_page(child_page);

            return Ok(());
        }

        // Root overflow
        if node.is_root() && node.is_overflow() {
            let mut old_root = Page::new(self.allocate_page(), self.cache.pager.page_size as u16);

            let root = self.cache.get_mut(page)?;
            old_root.append(root);

            root.header_mut().right_child = old_root.number;

            parents.push(page);
            page = old_root.number;

            self.cache.load_from_mem(old_root)?;
        }

        let parent_page = parents.remove(parents.len() - 1);
        let mut siblings = self.load_siblings(page, parent_page)?;

        let mut cells = Vec::new();
        let mut divider_idx = siblings[0].index;

        // Make copies of data in order.
        for (i, sibling) in siblings.iter().enumerate() {
            cells.extend(self.cache.get_mut(sibling.page)?.drain(..));
            if i < siblings.len() - 1 {
                let mut divider = self.cache.get_mut(parent_page)?.remove(divider_idx);
                divider.header.left_child = self.cache.get(sibling.page)?.header().right_child;
                cells.push(divider);
            }
        }

        let usable_space = Page::usable_space(self.cache.pager.page_size as _);

        let mut total_size_in_each_page = vec![0];
        let mut number_of_cells_per_page = vec![0];

        // Precompute left biased distribution
        for cell in &cells {
            let i = number_of_cells_per_page.len() - 1;
            if total_size_in_each_page[i] + cell.storage_size() <= usable_space {
                number_of_cells_per_page[i] += 1;
                total_size_in_each_page[i] += cell.storage_size();
            } else {
                number_of_cells_per_page.push(0);
                total_size_in_each_page.push(0);
            }
        }

        // Account for underflow towards the right
        if number_of_cells_per_page.len() >= 2 {
            let mut i = cells.len() - number_of_cells_per_page.last().unwrap() - 1;

            for j in (1..=(total_size_in_each_page.len() - 1)).rev() {
                while total_size_in_each_page[j] < usable_space / 2 {
                    number_of_cells_per_page[j] += 1;
                    total_size_in_each_page[j] += &cells[i].storage_size();

                    number_of_cells_per_page[j - 1] -= 1;
                    total_size_in_each_page[j - 1] -= &cells[i - 1].storage_size();
                    i -= 1;
                }
            }

            // Second page has more data than the first one, make a little
            // adjustment to keep it left biased.
            if total_size_in_each_page[0] < usable_space / 2 {
                number_of_cells_per_page[0] += 1;
                number_of_cells_per_page[1] -= 1;
            }
        }

        let old_right_child = self
            .cache
            .get(siblings[siblings.len() - 1].page)?
            .header()
            .right_child;

        // Allocate missing pages.
        while siblings.len() < number_of_cells_per_page.len() {
            let new_page = Page::new(self.allocate_page() as _, self.cache.pager.page_size as _);
            siblings.push(Sibling::new(
                new_page.number,
                siblings[siblings.len() - 1].index + 1,
            ));
            self.cache.load_from_mem(new_page)?;
        }

        // Free unused pages.
        while number_of_cells_per_page.len() < siblings.len() {
            let unused_page = siblings.pop().unwrap().page;
            self.free_page(unused_page);
        }

        // Begin distribution
        for (i, n) in number_of_cells_per_page.iter().enumerate() {
            let page = self.cache.get_mut(siblings[i].page)?;
            for _ in 0..*n {
                page.push(cells.remove(0));
            }

            if i < siblings.len() - 1 {
                let mut divider = cells.remove(0);
                page.header_mut().right_child = divider.header.left_child;
                divider.header.left_child = siblings[i].page;
                self.cache
                    .get_mut(parent_page)?
                    .insert(divider_idx as _, divider);
                divider_idx += 1;
            }
        }

        // Fix children pointers.
        self.cache
            .get_mut(siblings[siblings.len() - 1].page)?
            .header_mut()
            .right_child = old_right_child;

        if divider_idx == self.cache.get(parent_page)?.len() {
            self.cache.get_mut(parent_page)?.header_mut().right_child =
                siblings[siblings.len() - 1].page;
        } else {
            self.cache
                .get_mut(parent_page)?
                .cell_mut(divider_idx)
                .header
                .left_child = siblings[siblings.len() - 1].page;
        }

        self.balance(parent_page, parents)?;

        Ok(())
    }

    fn load_siblings(
        &mut self,
        page: PageNumber,
        parent_page: PageNumber,
    ) -> io::Result<Vec<Sibling>> {
        let mut num_siblings_per_side = self.balance_siblings_per_side as u16;

        let parent = self.cache.get(parent_page)?;

        // TODO: Store this somewhere somehow.
        let index = parent.iter_children().position(|p| p == page).unwrap() as u16;

        if index == 0 || index == parent.len() {
            num_siblings_per_side *= 2;
        };

        let left_siblings = index.saturating_sub(num_siblings_per_side)..index;
        let right_siblings = (index + 1)..min(index + num_siblings_per_side + 1, parent.len() + 1);

        let read_sibling = |index| Sibling::new(parent.child(index), index);

        Ok(left_siblings
            .map(read_sibling)
            .chain(std::iter::once(Sibling::new(page, index)))
            .chain(right_siblings.map(read_sibling))
            .collect())
    }

    /// Returns a free page.
    fn allocate_page(&mut self) -> u32 {
        // TODO: Free list.
        let page = self.len;
        self.len += 1;

        page as u32
    }

    // Testing/Debugging only.
    fn read_into_mem(&mut self, root: PageNumber, buf: &mut Vec<Page>) -> io::Result<()> {
        for page in self.cache.get(root)?.iter_children().collect::<Vec<_>>() {
            self.read_into_mem(page, buf)?;
        }

        let node = self.cache.get(root)?.clone();
        buf.push(node);

        Ok(())
    }

    pub fn json(&mut self) -> io::Result<String> {
        let mut nodes = Vec::new();
        self.read_into_mem(0, &mut nodes)?;

        nodes.sort_by(|n1, n2| n1.number.cmp(&n2.number));

        let mut string = String::from('[');

        string.push_str(&self.to_json(&nodes[0])?);

        for node in &nodes[1..] {
            string.push(',');
            string.push_str(&self.to_json(&node)?);
        }

        string.push(']');

        Ok(string)
    }

    fn to_json(&mut self, page: &Page) -> io::Result<String> {
        let mut string = format!("{{\"page\":{},\"entries\":[", page.number);

        if page.len() >= 1 {
            let key = page.cell(0).content;
            string.push_str(&format!("{:?}", key));

            for i in 1..page.len() {
                string.push(',');
                string.push_str(&format!("{:?}", page.cell(i).content));
            }
        }

        string.push_str("],\"children\":[");

        if page.header().right_child != 0 {
            string.push_str(&format!("{}", page.child(0)));

            for child in page.iter_children().skip(1) {
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
    //! BTree testing module. Most of the tests use fixed size 32 bit keys
    //! to easily test whether the BTree balancing algorithm does the correct
    //! thing in each situation. To easily compare the state of the [`BTree`]
    //! structure with something human readable we use [`Node`] instances, which
    //! allow us to define a tree as if we used some JSON-like syntax. There's
    //! also a [`Builder`] struct that can be used to insert many 32 bit keys at
    //! once into the tree and also tune parameters such as
    //! [`BTree::balance_siblings_per_side`]. See the tests for more details and
    //! examples. Remember that `order` means the maximum number of children in
    //! a BTree that stores fixed size keys.

    use std::{io, mem};

    use super::{BTree, BALANCE_SIBLINGS_PER_SIDE};
    use crate::{
        paging::{
            cache::Cache,
            pager::{PageNumber, Pager},
        },
        storage::page::{
            Cell, Page, CELL_ALIGNMENT, CELL_HEADER_SIZE, PAGE_HEADER_SIZE, SLOT_SIZE,
        },
    };

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
        fn leaf(keys: impl IntoIterator<Item = u32>) -> Self {
            Self {
                keys: keys.into_iter().collect(),
                children: Vec::new(),
            }
        }
    }

    /// Builder for [`BTree<MemBuf>`].
    struct Builder {
        keys: Vec<u32>,
        order: usize,
        balance_siblings_per_side: usize,
    }

    impl Default for Builder {
        fn default() -> Self {
            Builder {
                keys: vec![],
                order: 4,
                balance_siblings_per_side: BALANCE_SIBLINGS_PER_SIDE,
            }
        }
    }

    impl Builder {
        fn keys(mut self, keys: impl IntoIterator<Item = u32>) -> Self {
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
            let page_size = optimal_page_size_for(self.order);
            let buf = io::Cursor::new(Vec::new());

            let mut btree = BTree::new(
                Cache::new(Pager::new(buf, page_size, page_size)),
                self.balance_siblings_per_side,
            );

            let root = Page::new(0, page_size as _);
            // Init root.
            btree.cache.pager.write(root.number, root.buffer())?;

            btree.extend_from_keys(self.keys)?;

            Ok(btree)
        }
    }

    /// We use in-memory buffers instead of disk files for testing. This speeds
    /// up tests as it avoids disk IO and system calls.
    type MemBuf = io::Cursor<Vec<u8>>;

    impl BTree<MemBuf> {
        fn into_test_nodes(&mut self, root: PageNumber) -> io::Result<Node> {
            let mut page = Page::new(root, self.cache.pager.page_size as _);
            self.cache.pager.read(root, page.buffer_mut())?;

            let mut node = Node {
                keys: (0..page.len())
                    .map(|i| u32::from_be_bytes(page.cell(i).content[..4].try_into().unwrap()))
                    .collect(),
                children: vec![],
            };

            for page in page.iter_children() {
                node.children.push(self.into_test_nodes(page)?);
            }

            Ok(node)
        }

        fn builder() -> Builder {
            Builder::default()
        }

        fn insert_key(&mut self, key: u32) -> io::Result<()> {
            self.insert(&key.to_be_bytes())
        }

        fn remove_key(&mut self, key: u32) -> io::Result<Option<Box<[u8]>>> {
            self.remove(&key.to_be_bytes())
        }

        fn extend_from_keys(&mut self, keys: impl IntoIterator<Item = u32>) -> io::Result<()> {
            for key in keys {
                self.insert(&key.to_be_bytes())?;
            }

            Ok(())
        }

        fn try_remove_all_keys(
            &mut self,
            keys: impl IntoIterator<Item = u32>,
        ) -> io::Result<Vec<Option<Box<[u8]>>>> {
            let mut results = Vec::new();
            for k in keys {
                results.push(self.remove(&k.to_be_bytes())?);
            }
            Ok(results)
        }
    }

    fn optimal_page_size_for(order: usize) -> usize {
        let key_size = Cell::new(&[0; mem::size_of::<u32>()]).storage_size();
        let total_space_needed = PAGE_HEADER_SIZE + key_size * (order as u16 - 1);

        std::alloc::Layout::from_size_align(total_space_needed as usize, CELL_ALIGNMENT)
            .unwrap()
            .pad_to_align()
            .size() as _
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
    ///     /           /        \          \
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
    ///               /           /        \          \
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
    ///               /            /      \           \
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

        btree.remove_key(13)?;

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

        btree.remove_key(8)?;

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

        btree.remove_key(11)?;

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

        btree.remove_key(11)?;

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

    /// When a leaf node falls under 50% capacity it should not be merged with
    /// one of its siblings if the siblings can lend keys without underflowing.
    ///
    /// ```text
    /// 
    ///                           DELETE 15
    ///                               |
    ///                               V
    ///                           +--------+
    ///                  +--------| 4,8,12 |--------+
    ///                 /         +--------+         \
    ///                /           /      \           \
    ///           +-------+  +-------+  +---------+  +----------+
    ///           | 1,2,3 |  | 5,6,7 |  | 9,10,11 |  | 13,14,15 |
    ///           +-------+  +-------+  +---------+  +----------+
    ///
    ///                           DELETE 14
    ///                               |
    ///                               V
    ///                           +--------+
    ///                  +--------| 4,8,12 |--------+
    ///                 /         +--------+         \
    ///                /           /      \           \
    ///           +-------+  +-------+  +---------+  +----------+
    ///           | 1,2,3 |  | 5,6,7 |  | 9,10,11 |  | 13,14    |
    ///           +-------+  +-------+  +---------+  +----------+
    ///
    ///                          FINAL RESULT
    ///                               |
    ///                               V
    ///                           +--------+
    ///                  +--------| 4,8,11 |--------+
    ///                 /         +--------+         \
    ///                /           /      \           \
    ///           +-------+  +-------+  +---------+  +----------+
    ///           | 1,2,3 |  | 5,6,7 |  | 9,10    |  | 12,13    |
    ///           +-------+  +-------+  +---------+  +----------+
    /// ```
    #[test]
    fn delay_leaf_node_merge() -> io::Result<()> {
        let mut btree = BTree::builder().keys(1..=15).try_build()?;

        btree.try_remove_all_keys((14..=15).rev())?;

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

        btree.try_remove_all_keys((12..=15).rev())?;

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

        btree.remove_key(4)?;

        assert_eq!(Node::try_from(btree)?, Node::leaf([1, 2, 3]));

        Ok(())
    }

    /// Same as [`merge_root`] but the root children have their own children.
    ///
    /// ```text
    ///                          DELETE 15,16
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
    ///                                |
    ///                                V
    ///                           +--------+
    ///                   +-------| 4,8,11 |--------+
    ///                 /         +--------+         \
    ///               /            /      \           \
    ///           +-------+  +-------+  +------+  +----------+
    ///           | 1,2,3 |  | 5,6,7 |  | 9,10 |  | 12,13,14 |
    ///           +-------+  +-------+  +------+  +----------+
    /// ```
    #[test]
    fn decrease_tree_height() -> io::Result<()> {
        let mut btree = BTree::builder().keys(1..=16).try_build()?;

        btree.try_remove_all_keys(15..=16)?;

        assert_eq!(
            Node::try_from(btree)?,
            Node {
                keys: vec![4, 8, 11],
                children: vec![
                    Node::leaf([1, 2, 3]),
                    Node::leaf([5, 6, 7]),
                    Node::leaf([9, 10]),
                    Node::leaf([12, 13, 14]),
                ]
            }
        );

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

        btree.try_remove_all_keys(1..=3)?;

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

        btree
            .try_remove_all_keys(1..=3)
            .and_then(|_| btree.remove_key(35))?;

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

        btree.try_remove_all_keys(34..=36)?;

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
    ///                  DELETE 3, INSERT 16 -> No Split
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
    ///                           +--------+
    ///                   +-------| 5,9,13 |--------+
    ///                 /         +--------+         \
    ///               /            /      \           \
    ///           +-------+  +-------+  +----------+  +----------+
    ///           | 1,2,4 |  | 6,7,8 |  | 10,11,12 |  | 14,15,16 |
    ///           +-------+  +-------+  +----------+  +----------+
    /// ```
    #[test]
    fn increased_balance_siblings_per_side() -> io::Result<()> {
        let mut btree = BTree::builder()
            .balance_siblings_per_side(2)
            .keys(1..=15)
            .try_build()?;

        btree.remove_key(3)?;
        btree.insert_key(16)?;

        assert_eq!(
            Node::try_from(btree)?,
            Node {
                keys: vec![5, 9, 13],
                children: vec![
                    Node::leaf([1, 2, 4]),
                    Node::leaf([6, 7, 8]),
                    Node::leaf([10, 11, 12]),
                    Node::leaf([14, 15, 16]),
                ]
            }
        );

        Ok(())
    }
}
