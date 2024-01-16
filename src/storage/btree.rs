//! Disk BTree data structure. See [`BTree`] for details.

use std::{
    cmp::{min, Ordering},
    collections::VecDeque,
    fs::File,
    io,
    io::{Read, Seek, Write},
    os::unix::fs::MetadataExt,
    path::Path,
};

use super::page::{Cell, Page, SlotId};
use crate::{
    os::{Disk, HardwareBlockSize},
    paging::{
        cache::Cache,
        pager::{PageNumber, Pager},
    },
};

/// [`BTree`] key comparator. Keys are stored in binary, so we need a way to
/// determine the correct [`Ordering`].
pub(crate) trait BytesCmp {
    /// Compares two byte arrays and returns the corresponding [`Ordering`].
    /// At the [`BTree`] level we don't care how that is done. Upper levels can
    /// parse the binary to obtain an [`Ordering`] instance, they can store
    /// the entry in such a format that they can tell the [`Ordering`] by
    /// looking at the bytes, or they can do anything else, we don't care here.
    fn bytes_cmp(&self, a: &[u8], b: &[u8]) -> Ordering;
}

/// Default byte comparator for [`BTree`]. It does the good old `memcmp`. This
/// is handy for integer keys, since storing them in big endian will yield the
/// correct results.
pub struct MemCmp;

impl BytesCmp for MemCmp {
    fn bytes_cmp(&self, a: &[u8], b: &[u8]) -> Ordering {
        let size = std::cmp::min(a.len(), b.len());
        a[..size].cmp(&b[..size])
    }
}

/// The result of a search in the [`BTree`] structure.
struct Search {
    /// Page number of the node where the search ended.
    page: PageNumber,

    /// Contains an [`Ok`] value with the index where the search matched or an
    /// [`Err`] value with the index where the searched key should be located.
    index: Result<u16, u16>,
}

/// Stores the page number and the index in the parent page entries that points
/// back to this node. See [`BTree::load_siblings`] and [`BTree::balance`] for
/// details.
struct Sibling {
    /// Page number of this sibling node.
    page: PageNumber,
    /// Index of the cell that points to this node in the parent.
    index: SlotId,
}

impl Sibling {
    fn new(page: PageNumber, index: u16) -> Self {
        Self { page, index }
    }
}

/// Used to search either the minimum or maximum key of a subtree. See
/// [`BTree::search_leaf_key`] for details.
enum LeafKeySearch {
    /// Maximum key in a leaf node.
    Max,
    /// Minimum key in a leaf node.
    Min,
}

/// B*-Tree implementation inspired by "Art of Computer Programming Volume 3:
/// Sorting and Searching" and SQLite 2.X.X
///
/// # About BTrees
///
/// BTrees are a family of tree datastructures that always maintain their
/// balance. Unlike binary trees, which can be turned into a linked list by
/// inserting keys in sequential order like this:
///
/// ```text
///                      +---+
///                      | 1 |
///                      +---+
///                           \
///                          +---+
///                          | 2 |
///                          +---+
///                               \
///                              +---+
///                              | 3 |
///                              +---+
/// ```
///
/// BTrees will never lose their O(log n) search time, because they cannot
/// be turned into linked lists. This is how a BTree looks like:
///
/// ```text
///                                     +--------+ Root Node
///                   +-----------------|   11   |--------------+
///                  /                  +--------+               \
///                 /                                             \
///            +-------+                                     +----------+ Internal Node
///       +----|  4,8  |----+                   +------------| 15,19,23 |------------+
///      /     +-------+     \                 /             +----------+             \
///     /          |          \               /                /      \                \
/// +-------+  +-------+  +-------+     +----------+  +----------+  +----------+  +----------+
/// | 1,2,3 |  | 5,6,7 |  | 9,10  |     | 12,13,14 |  | 16,17,18 |  | 20,21,22 |  | 24,25,26 |
/// +-------+  +-------+  +-------+     +----------+  +----------+  +----------+  +----------+
///                                                                                 Leaf Node
/// ```
///
/// The first difference that can be spotted is the number of keys stored in
/// each node. BTrees are generalization of binary trees, so they can store N
/// keys per node. This greatly reduces the overall height of the tree. The
/// more keys we store, the slower the tree grows in size. Second difference you
/// can spot here is that even though the tree is storing sequential keys, it's
/// still a tree, not a linked list. All this is thanks to [`BTree::balance`].
///
/// Another important distinction is that BTrees are better suited for disk
/// storage, since storage hardware is usually divided in blocks or pages of
/// N bytes each, and we can essentially store one BTree node per page.
///
/// The BTree stores variable size binary keys, but if all the keys have the
/// same size then some nice properties apply.
///
/// **Terminology**:
///
/// `order`: Maximum number of children per node (except root).
///
/// **Properties**:
///
/// - Max children: `order`
/// - Max keys:     `order - 1`
///
/// The lower bounds depend on the type of BTree. For normal BTrees, the
/// formulas are:
///
/// - Min children: `(order / 2)` (except root)
/// - Min keys:     `(order / 2) - 1` (except root)
///
/// This would keep nodes at least 50% full. There's a variant of the BTree
/// called B*-Tree which attempts to keep nodes 66% (two thirds) full. In that
/// case, the lower bounds would be:
///
/// - Min children: `order * 2 / 3` (except root)
/// - Min keys:     `order * 2 / 3 - 1` (except root)
///
/// The algorithms for this kind of BTree are described in "Art of Computer
/// Programming Volume 3: Sorting and Searching". However, since we're storing
/// variable length data, none of this applies. It's important to know for the
/// tests because most of the tests use fixed size data, but databases usually
/// store variable length data.
///
/// In this implementation of the BTree, we attempt to keep nodes above 50%
/// capacity, but we don't know exactly how much "above" 50% because, as
/// mentioned earlier, we store variable length data.
///
/// See the documentation of the [`BTree`] struct methods for more details on
/// the algorithms used.
///
/// # Structure
///
/// The [`BTree`] is made of [`Page`] instances that are stored on disk and
/// cached in memory as required. Each [`Page`] stores a list of [`Cell`]
/// instances using a slot array. Cells are the smallest unit of data that we
/// work with, each cell stores a key (or key-value pair, we don't care) and a
/// pointer to the node that contains keys less than the one stored in the
/// cell. Overall, this is how the BTree really looks like:
///
/// ```text
///                                                              +---------------+
///                                                              |               |
///                                           +-------------+----|---------------V----------+
///                                           | +----+----+ | +----+             +----+---+ |
///                                           | | RC | NS | | | O1 | ->  FS   <- | LC | D | |
///                                           | +----+----+ | +----+             +----+---+ |
///                                           +---|---------+----------------------|--------+
///                                               |                                |
///                               +---------------|--------------------------------+
///                               |               |
///                               |               +-----------------------------------------------+
///                               |                                                               |
///  PAGE HEADER    SLOT ARRAY    V                 CELLS                                         V
/// +-------------+-----------------------------------------------+      +-------------+-------------------------------+
/// | +----+----+ | +----+----+             +----+---+ +----+---+ |      | +----+----+ | +----+             +----+---+ |
/// | | RC | NS | | | O1 | O2 | ->   FS  <- | LC | D | | LC | D | |      | | RC | NS | | | O1 | ->   FS  <- | LC | D | |
/// | +----+----+ | +----+----+             +----+---+ +----+---+ |      | +----+----+ | +----+             +----+---+ |
/// +-------------+---|----|--------------------------------------+      +-------------+---|---------------------------+
///                   |    |                ^          ^                                   |                ^
///                   |    |                |          |                                   |                |
///                   |    +----------------+          |                                   +----------------+
///                   +--------------------------------+
/// ```
///
/// Here's what everything stands for:
///
/// - RC: Right Child
/// - NS: Number of Slots
/// - ON: Offset N
/// - LC: Left Child
/// - FS: Free Space
/// -  D: Data
///
/// Right child is stored in the page header and is always the "last child" of
/// a node. Or in other words, the child that has keys grater than any key in
/// the current node. Left child is stored in the cell header and points to
/// the child that has keys less than the one in the cell. For the slotted page
/// details and omitted fields see the documentation of [`super::page`] module.
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
        if metadata.size() == 0 {
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

/// Default value for [`BTree::balance_siblings_per_side`].
const BALANCE_SIBLINGS_PER_SIDE: usize = 1;

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

    /// # BTree Search Algorithm
    ///
    /// 1. Read the subtree root node into memory.
    /// 2. Run a binary search on the entries to find the given key.
    /// 3. If successful, return the [`Search`] result.
    /// 4. If not, the binary search result will tell us which child to pick for
    /// recursion.
    ///
    /// # Example
    ///
    /// Find key 9 in this tree, located at page 5:
    ///
    /// ```text
    ///                             PAGE 0
    ///                           +--------+
    ///                   +-------|   11   |-------+
    ///                  /        +--------+        \
    ///                 /                            \
    ///            +-------+ PAGE 1              +--------+
    ///       +----|  4,8  |----+                |   14   |
    ///      /     +-------+     \               +--------+
    ///     /          |          \               /      \
    /// +-------+  +-------+  +-------+     +-------+  +-------+
    /// | 1,2,3 |  | 5,6,7 |  | 9,10  |     | 12,13 |  | 15,16 |
    /// +-------+  +-------+  +-------+     +-------+  +-------+
    ///  PAGE 3     PAGE 4      PAGE 5
    /// ```
    /// ## First iteration
    ///
    /// 1. Read page 0 into memory.
    /// 2. Binary search on [`Page`] results in [`Err(0)`].
    /// 3. Read index 0 using [`Page::child`] and recurse into the result.
    ///
    /// ## Second iteration
    ///
    /// 1. Read page 1 into memory.
    /// 2. Binary search results in [`Err(3)`].
    /// 3. Read child pointer at index 3 and recurse again.
    ///
    /// ## Final iteration.
    ///
    /// 1. Read page 5 into memory.
    /// 2. Binary search results in [`Ok(0)`].
    /// 3. Done, construct the [`Search`] result and return.
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

    /// Inserts a new entry into the tree or updates the value if the key
    /// already exists.
    ///
    /// # Insertion algorithm
    ///
    /// Entries are always inserted at leaf nodes. Internal nodes and the root
    /// node can only grow in size when leaf nodes overflow and siblings can't
    /// take any load to keep the leaves balanced, causing a split.
    ///
    /// Let's walk through an example. Suppose we have the following [`BTree`]
    /// of `order = 4`, which means each node can hold at maximum 3 keys and 4
    /// children.
    ///
    /// ```text
    ///                             PAGE 0 (ROOT)
    ///                              +-------+
    ///                          +---|  3,6  |---+
    ///                         /    +-------+    \
    ///                        /         |         \
    ///                   +-------+  +-------+  +-------+
    ///                   |  1,2  |  |  4,5  |  |  7,8  |
    ///                   +-------+  +-------+  +-------+
    ///                     PAGE 1     PAGE 2     PAGE 3
    /// ```
    ///
    /// Now let's say we want to insert key `9`. The insertion algorithm will
    /// call [`Self::search`] to find the page and index where the new key
    /// should be added, and it will simply insert the key:
    ///
    /// ```text
    ///                             PAGE 0 (ROOT)
    ///                              +-------+
    ///                          +---|  3,6  |---+
    ///                         /    +-------+    \
    ///                        /         |         \
    ///                   +-------+  +-------+  +---------+
    ///                   |  1,2  |  |  4,5  |  |  7,8,9  |
    ///                   +-------+  +-------+  +---------+
    ///                     PAGE 1     PAGE 2     PAGE 3
    /// ```
    ///
    /// Now page 3 has reached the maximum number of keys allowed. We are not
    /// going to do anything special yet, the insertion algorithm per se does
    /// not care about overflowing, it will always insert keys in the node where
    /// they should belong as returned by [`Self::search`].
    ///
    /// Therefore, when the caller wants to insert key `10` the algorithm will
    /// do this:
    ///
    /// ```text
    ///                             PAGE 0 (ROOT)
    ///                              +-------+
    ///                          +---|  3,6  |---+
    ///                         /    +-------+    \
    ///                        /         |         \
    ///                   +-------+  +-------+  +----------+
    ///                   |  1,2  |  |  4,5  |  | 7,8,9,10 |
    ///                   +-------+  +-------+  +----------+
    ///                     PAGE 1     PAGE 2      PAGE 3
    /// ```
    ///
    /// Note that this leaves the tree in an unbalanced state and since page 3
    /// has overflowed it cannot be saved to disk yet, because the bytes may not
    /// fit in the current page size.
    ///
    /// However, before finishing and returning an [`Ok`] result, we call
    /// [`Self::balance`], which does all the work needed to maintain the tree
    /// in a correct and balanced state. See [`Self::balance`] for details on
    /// rearranging keys and splitting nodes after inserts.
    pub fn insert(&mut self, entry: &[u8]) -> io::Result<()> {
        let mut parents = Vec::new();
        let search = self.search(0, entry, &mut parents)?;
        let node = self.cache.get_mut(search.page)?;

        match search.index {
            // Key found, swap value.
            Ok(index) => {
                let mut cell = Cell::new(entry);
                cell.header.left_child = node.cell(index).header.left_child;
                node.replace(index, cell);
            }
            // Key not found, insert new entry.
            Err(index) => node.insert(index, Cell::new(entry)),
        };

        self.balance(search.page, parents)?;
        self.cache.flush_write_queue_to_disk()
    }

    /// Removes the entry corresponding to the given key if it exists.
    ///
    /// # Deletion algorithm
    ///
    /// There are 2 major cases to consider. First, the simplest case, deleting
    /// from leaf nodes. Suppose we have the following [`BTree`] of order 4 and
    /// we want to delete key `13`:
    ///
    /// ```text
    ///                             PAGE 0
    ///                           +--------+
    ///                   +-------| 4,8,12 |----------+
    ///                 /         +--------+           \
    ///               /           /       \             \
    ///           +-------+  +-------+  +---------+  +----------+
    ///           | 1,2,3 |  | 5,6,7 |  | 9,10,11 |  | 13,14,15 |
    ///           +-------+  +-------+  +---------+  +----------+
    ///            PAGE 1     PAGE 2      PAGE 3        PAGE 4
    /// ```
    ///
    /// The deletion algorithm will simply remove key 13 from page 4 and it
    /// won't care whether the node has underflowed or not:
    ///
    /// ```text
    ///                             PAGE 0
    ///                           +--------+
    ///                   +-------| 4,8,12 |----------+
    ///                 /         +--------+           \
    ///               /           /       \             \
    ///           +-------+  +-------+  +---------+  +----------+
    ///           | 1,2,3 |  | 5,6,7 |  | 9,10,11 |  | 14,15    |
    ///           +-------+  +-------+  +---------+  +----------+
    ///            PAGE 1     PAGE 2      PAGE 3        PAGE 4
    /// ```
    ///
    /// Later, when calling [`Self::balance`] before returning, underflows will
    /// be handled properly.
    ///
    /// The second case to consider is deleting from internal nodes or the root
    /// node. The root node is not a special case, it is treated just like any
    /// other internal node.
    ///
    /// Suppose we have the following [`BTree`] of order 4 and we want to delete
    /// key `11` located at page 0:
    ///
    /// ```text
    ///                             PAGE 0
    ///                           +--------+
    ///                   +-------|   11   |-------+
    ///                  /        +--------+        \
    ///                 /                            \
    ///            +-------+ PAGE 1              +--------+
    ///       +----|  4,8  |----+                |   14   |
    ///      /     +-------+     \               +--------+
    ///     /          |          \               /      \
    /// +-------+  +-------+  +-------+     +-------+  +-------+
    /// | 1,2,3 |  | 5,6,7 |  | 9,10  |     | 12,13 |  | 15,16 |
    /// +-------+  +-------+  +-------+     +-------+  +-------+
    ///  PAGE 3      PAGE 4     PAGE 5
    /// ```
    ///
    /// Deleting keys from internal nodes is not allowed because each key needs
    /// a corresponding child to the left and one to the right. Therefore, we
    /// have to find a key in one of the leaf nodes that can substitute the key
    /// we want to delete.
    ///
    /// There are two possible options for that:
    /// - *Predecessor*: greatest key in the left subtree.
    /// - *Successor*: smallest key in the right subtree.
    ///
    /// Locate either the predecessor or successor and move it up to the current
    /// node. The condition for choosing one or another is based on how many
    /// children the left child has VS how many children the right child has.
    ///
    /// We'll choose the option with more children since there's less
    /// probabilities of merging nodes. See [`Self::balance`] for details on
    /// merging. In this case, the left child of page 0 has 2 children while
    /// the right child has only 1, so we'll pick the predecessor of key 11,
    /// which is key 10 located at page 5. The end result is this:
    ///
    /// ```text
    ///                             PAGE 0
    ///                           +--------+
    ///                   +-------|   10   |-------+
    ///                  /        +--------+        \
    ///                 /                            \
    ///            +-------+ PAGE 1              +--------+
    ///       +----|  4,8  |----+                |   14   |
    ///      /     +-------+     \               +--------+
    ///     /          |          \               /      \
    /// +-------+  +-------+  +-------+     +-------+  +-------+
    /// | 1,2,3 |  | 5,6,7 |  | 9     |     | 12,13 |  | 15,16 |
    /// +-------+  +-------+  +-------+     +-------+  +-------+
    ///  PAGE 3      PAGE 4     PAGE 5
    /// ```
    ///
    /// As you can see, page 5 has underflowed below 50%. Underflows are
    /// handled by [`Self::balance`]. See also [`Self::remove_entry`] for the
    /// actual deletion code, this function is a wrapper that provides a public
    /// API and calls [`Self::balance`] at the end.
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
        leaf_key_search: LeafKeySearch,
    ) -> io::Result<(PageNumber, u16)> {
        let node = self.cache.get(page)?;

        let (key_idx, child_idx) = match leaf_key_search {
            LeafKeySearch::Min => (0, 0),
            LeafKeySearch::Max => (node.len() - 1, node.len()),
        };

        if node.is_leaf() {
            return Ok((page, key_idx));
        }

        parents.push(page);
        let child = node.child(child_idx);

        self.search_leaf_key(child, parents, leaf_key_search)
    }

    /// Returns the page number and slot index in [`Page`] where the greatest
    /// key of the given subtree is located.
    fn search_max_key(
        &mut self,
        page: PageNumber,
        parents: &mut Vec<PageNumber>,
    ) -> io::Result<(PageNumber, u16)> {
        self.search_leaf_key(page, parents, LeafKeySearch::Max)
    }

    /// Returns the page number and slot index in [`Page`] where the smallest
    /// key of the given subtree is located.
    fn search_min_key(
        &mut self,
        page: PageNumber,
        parents: &mut Vec<PageNumber>,
    ) -> io::Result<(PageNumber, u16)> {
        self.search_leaf_key(page, parents, LeafKeySearch::Min)
    }

    /// B*-Tree balancing algorithm inspired by (or rather stolen from) SQLite
    /// 2.X.X. Take a look at the original source code here:
    ///
    /// https://github.com/antoniosarosi/sqlite2-btree-visualizer/blob/master/src/btree.c#L2171
    ///
    /// # Algorithm Steps
    ///
    /// This is how it works:
    ///
    /// 1. Check if the current node has overflown or underflown. If not, early
    /// return without doing anything.
    ///
    /// 2. Check if the node is the root and has underflown. The root is only
    /// considered underflow when it contains 0 cells and has one child:
    ///
    /// ```text
    ///                       +-------+
    ///                       |       | Empty root
    ///                       +-------+
    ///                           |
    ///                           v
    ///                       +-------+ Direct root child
    ///                  +----|  4,8  |----+
    ///                 /     +-------+     \
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
    ///                       +-------+
    ///                  +----|  4,8  |----+
    ///                 /     +-------+     \
    ///                /          |          \
    ///            +-------+  +-------+  +----------+
    ///            | 1,2,3 |  | 5,6,7 |  | 12,13,14 | Other pages below
    ///            +-------+  +-------+  +----------+
    /// ```
    ///
    /// 3. Check if the node is the root and is overflow. The overflow condition
    /// is simpler than underflow, as we only need to check if the page could
    /// not fit the last inserted [`Cell`]. See [`Page`] for details. For this
    /// simple example, suppose a page can only fit 3 keys. If it contains more
    /// than that then it is considered overflow.
    ///
    /// ```text
    ///               Overflow root (can only fit 3 keys max)
    ///                          +-----------+
    ///             +------------| 4,8,12,16 |----------------+
    ///            /             +-----------+                 \
    ///           /                |   |   |                    \
    ///          /          +------+   |   +-------+             \
    ///         /          /           |            \             \
    ///     +-------+  +-------+  +---------+  +----------+  +----------+
    ///     | 1,2,3 |  | 5,6,7 |  | 9,10,11 |  | 13,14,15 |  | 17,18,19 |
    ///     +-------+  +-------+  +---------+  +----------+  +----------+
    /// ```
    ///
    /// If the root has overflown, then execute the reverse of the underflow
    /// algorithm, that is, create a new empty child below the root, move the
    /// content of the root into that child increasing tree height by one and
    /// leave the root empty:
    ///
    /// ```text
    ///                          +-----------+
    ///                          |           | Empty root
    ///                          +-----------+
    ///                                |
    ///                          +-----------+ Overflow child node
    ///             +------------| 4,8,12,16 |----------------+
    ///            /             +-----------+                 \
    ///           /                |   |   |                    \
    ///          /          +------+   |   +-------+             \
    ///         /          /           |            \             \
    ///     +-------+  +-------+  +---------+  +----------+  +----------+
    ///     | 1,2,3 |  | 5,6,7 |  | 9,10,11 |  | 13,14,15 |  | 17,18,19 |
    ///     +-------+  +-------+  +---------+  +----------+  +----------+
    /// ```
    ///
    /// Note that the root page never changes. New pages can be allocated below
    /// it or they can be freed, but the root does not move, the root always
    /// stays at the same place.
    ///
    /// Now follow the steps below to split the overflow child.
    ///
    /// 4. If we managed to get here, we are either working with an overflow
    /// or underflow internal/leaf node. The root should have been handled in
    /// the steps above. What we have to do now is redistribute all the cells
    /// in the current node and surrounding siblings. The amount of siblings
    /// that we take from each side is dictated by
    /// [`Self::balance_siblings_per_side`].
    ///
    /// If there are no siblings on any of the two sides then we will simply
    /// redistribute the keys in the current node. Let's start precisely with
    /// such case, following the figure above that contains an empty root and an
    /// overflow child. First step is moving all the keys out and leaving the
    /// overflow page empty:
    ///
    /// ```text
    ///                                    +---+ +---+ +----+ +----+
    ///     In-memory copies of each cell: | 4 | | 8 | | 12 | | 16 |
    ///                                    +---+ +---+ +----+ +----+
    ///
    ///                          +-----------+
    ///                          |           | Empty root
    ///                          +-----------+
    ///                                |
    ///                          +-----------+ Overflow page
    ///             +------------|           |----------------+
    ///            /             +-----------+                 \
    ///           /                |   |   |                    \
    ///          /          +------+   |   +-------+             \
    ///         /          /           |            \             \
    ///     +-------+  +-------+  +---------+  +----------+  +----------+
    ///     | 1,2,3 |  | 5,6,7 |  | 9,10,11 |  | 13,14,15 |  | 17,18,19 |
    ///     +-------+  +-------+  +---------+  +----------+  +----------+
    /// ```
    ///
    /// 5. Once we have local copies of every cell, we need to compute how many
    /// pages we need to fit all the cells. If we need more than we previously
    /// had, it means that the node was "overflow". If we need less than we
    /// previously had, the node was "underflow". If the number of pages does
    /// not change, then that means one of the siblinigs can take a key from
    /// the overflow node without becoming overflow itself or it can lend a
    /// key to the underflow node without becoming underflow itself. That would
    /// be the best case scenario since it does not require extra IO.
    ///
    /// Going back to the example in the figure above, we know that a page can
    /// only fit 3 cells of the same size but we have 4 keys in total, so we
    /// will need to allocate an extra page.
    ///
    /// ```text
    /// 
    ///                                    +---+ +---+ +----+ +----+
    ///     In-memory copies of each cell: | 4 | | 8 | | 12 | | 16 |
    ///                                    +---+ +---+ +----+ +----+
    ///
    ///                          +-----------+
    ///                          |           | Empty root
    ///                          +-----------+
    ///
    ///                  +-----------+  +-----------+
    ///    Previous page |           |  |           | New page
    ///                  +-----------+  +-----------+
    ///
    ///     +-------+  +-------+  +---------+  +----------+  +----------+
    ///     | 1,2,3 |  | 5,6,7 |  | 9,10,11 |  | 13,14,15 |  | 17,18,19 |
    ///     +-------+  +-------+  +---------+  +----------+  +----------+
    /// ```
    ///
    /// 6. Once we have all the pages we need, we can start distributing the
    /// cells, inserting cells into the parent as needed. The exact amount of
    /// cells in each page is computed in step 5 and should follow a left biased
    /// distribution where nodes towards the left contain more cells than nodes
    /// towards the right. If the number of cells is even then each node will
    /// have the same amount of cells. For example, if we have to distribute 8
    /// cells between 4 nodes, then each node will have 2 cells. On the other
    /// hand, if we have to distribute 9 cells between 4 nodes, then the first
    /// node will have 3 cells and the rest only 2 (left biased).
    ///
    /// Also, keep in mind that we're working with fixed size cells in this
    /// example, but in reality the data that we store will be variable size.
    /// That means it would be more correct to speak of "total size in bytes"
    /// instead of "number of cells", but let's keep it simple for now.
    ///
    /// This is how the distribution would look like once we finished step 6:
    ///
    /// ```text
    ///                                +--------+ Root page
    ///                   +------------|   12   |-------+
    ///                  /             +--------+        \
    ///                 /                                 \
    ///            +-------+ Previous overflow page   +--------+ Newly allocated page
    ///       +----|  4,8  |----+                     |   16   |
    ///      /     +-------+     \                    +--------+
    ///     /          |          \                    /      \
    /// +-------+  +-------+  +---------+     +----------+  +----------+
    /// | 1,2,3 |  | 5,6,7 |  | 9,10,11 |     | 13,14,15 |  | 17,18,19 |
    /// +-------+  +-------+  +---------+     +----------+  +----------+
    /// ```
    ///
    /// 7. Final step, call [`Self::balance`] on the parent node and propagate
    /// the algorithm upwards towards the root node. Once the root node is
    /// reached, the entire tree is guaranteed to be valid (no overflows or
    /// underflows) and balanced (searching any key is O(log n)).
    ///
    /// That's the whole algorithm, but let's see more examples where we can
    /// work with siblings.
    ///
    /// # Overflow Redistribution (No Split)
    ///
    /// Let's take a look at an overflow that does not require splitting (AKA
    /// allocating extra pages). Picture this tree where each page can hold a
    /// maximum number of 3 keys:
    ///
    /// ```text
    ///                  Root
    ///                +-------+
    ///                |   3   |
    ///                +-------+
    ///               /         \
    ///           +-------+  +-------+
    ///           |  1,2  |  | 4,5,6 |
    ///           +-------+  +-------+
    /// ```
    ///
    /// Suppose we want to insert key 7 into the tree above. It would end up
    /// in the second leaf node:
    ///
    /// ```text
    ///                  Root
    ///                +-------+
    ///                |   3   |
    ///                +-------+
    ///               /         \
    ///           +-------+  +---------+
    ///           |  1,2  |  | 4,5,6,7 | Key 7 is stored here
    ///           +-------+  +---------+
    /// ```
    ///
    /// Now the second leaf node is considered "overflow". However, we don't
    /// have to split it because the sibling at the left can take an extra
    /// key:
    ///
    /// ```text
    ///                  Root
    ///                +-------+
    ///                |   4   |
    ///                +-------+
    ///               /         \
    ///           +-------+  +-------+
    ///           | 1,2,3 |  | 5,6,7 | Key 7 is stored here
    ///           +-------+  +-------+
    /// ```
    ///
    /// In this case, the subalgorithm that allocates extra pages did not run
    /// because it determined it wasn't necessary. So we jump straight into
    /// the redistribution subalgorithm, which inserts keys and dividers cells
    /// into the parent as needed, causing the tree to look like the figure
    /// above.
    ///
    /// # Underflow Redistribution (No Merge)
    ///
    /// Now let's look at the "underflow" counterpart. The correct definition
    /// of "underflow" taking variable size data into account is as follows:
    ///
    /// > A page is considered "underflow" when less than half of its usable
    /// space in bytes is occupied.
    ///
    /// If the keys are fixed size then basically a page is underflow when it
    /// stores less than half the amount of keys it can store (aproximately, we
    /// have to consider the remainder). Going back to our example where each
    /// page stores 3 keys max, a page would be "underflow" if it stores only
    /// one key or less (the root is an exception). Let's use the same starting
    /// point as the previous example:
    ///
    /// ```text
    ///                  Root
    ///                +-------+
    ///                |   3   |
    ///                +-------+
    ///               /         \
    ///           +-------+  +-------+
    ///           |  1,2  |  | 4,5,6 |
    ///           +-------+  +-------+
    /// ```
    ///
    /// Now let's say we want to delete key 2:
    ///
    /// ```text
    ///                  Root
    ///                +-------+
    ///                |   3   |
    ///                +-------+
    ///               /         \
    ///           +-------+  +-------+
    ///           |   1   |  | 4,5,6 |
    ///           +-------+  +-------+
    /// ```
    ///
    /// The left leaf node has gone "underflow". But again, we can keep the
    /// tree balanced by shuffling keys around:
    ///
    /// ```text
    ///                 Root
    ///               +-----+
    ///               |  4  |
    ///               +-----+
    ///                /   \
    ///           +-----+  +-----+
    ///           | 1,3 |  | 5,6 |
    ///           +-----+  +-----+
    /// ```
    ///
    /// You can understand this as "the right sibling has lended a key to the
    /// left sibling so that it is not underflow anymore".
    ///
    /// # Overflow Split
    ///
    /// We've talked about splitting nodes before, but let's take a look at a
    /// more complicated example:
    ///
    /// ```text
    ///                                              PAGE 0
    ///                                            +--------+
    ///                         +------------------|   15   |-----------------+
    ///                        /                   +--------+                  \
    ///                       /                                                 \ PAGE 2
    ///                  +--------+                                            +----------+
    ///       +----------| 4,8,11 |---------+                     +------------| 19,23,27 |-----------+
    ///      /           +--------+          \                   /             +----------+            \
    ///     /             /     \             \                 /               /        \              \ PAGE 10
    /// +-------+  +-------+  +-------+  +----------+     +----------+  +----------+  +----------+  +----------+
    /// | 1,2,3 |  | 5,6,7 |  | 9,10  |  | 12,13,14 |     | 16,17,18 |  | 20,21,22 |  | 24,25,26 |  | 28,29,30 |
    /// +-------+  +-------+  +-------+  +----------+     +----------+  +----------+  +----------+  +----------+
    /// ```
    ///
    /// Almost all of the nodes in the tree above are full. What happens when
    /// we insert key 31?
    ///
    /// ```text
    ///                                              PAGE 0
    ///                                            +--------+
    ///                         +------------------|   15   |-----------------+
    ///                        /                   +--------+                  \
    ///                       /                                                 \ PAGE 2
    ///                  +--------+                                            +----------+
    ///       +----------| 4,8,11 |---------+                     +------------| 19,23,27 |-----------+
    ///      /           +--------+          \                   /             +----------+            \
    ///     /             /     \             \                 /               /        \              \ PAGE 10
    /// +-------+  +-------+  +-------+  +----------+     +----------+  +----------+  +----------+  +-------------+
    /// | 1,2,3 |  | 5,6,7 |  | 9,10  |  | 12,13,14 |     | 16,17,18 |  | 20,21,22 |  | 24,25,26 |  | 28,29,30,31 |
    /// +-------+  +-------+  +-------+  +----------+     +----------+  +----------+  +----------+  +-------------+
    ///                                                                                                         ^
    ///                                                                                                         |
    ///                                                                                              Key 31 was stored here
    /// ```
    ///
    /// As you can see, page 10 is now "overflow" and none of the siblings can
    /// take more keys because they are already full themselves. So let's
    /// execute the balancing algorithm on page 10. Suppose
    /// [`Self::balance_siblings_per_side`] has a value of 1, which means we
    /// have to take one sibling from each side. Since the overflow node does
    /// not have siblings at the right side, we will take an extra sibling from
    /// the left side:
    ///
    /// ```text
    ///                                              PAGE 0
    ///                                            +--------+
    ///                         +------------------|   15   |-----------------+
    ///                        /                   +--------+                  \
    ///                       /                                                 \ PAGE 2
    ///                  +--------+                                            +----------+
    ///       +----------| 4,8,11 |---------+                     +------------| 19,23,27 |-----------+
    ///      /           +--------+          \                   /             +----------+            \
    ///     /             /     \             \                 /               /        \              \ PAGE 10
    /// +-------+  +-------+  +-------+  +----------+     +----------+  +----------+  +----------+  +-------------+
    /// | 1,2,3 |  | 5,6,7 |  | 9,10  |  | 12,13,14 |     | 16,17,18 |  | 20,21,22 |  | 24,25,26 |  | 28,29,30,31 |
    /// +-------+  +-------+  +-------+  +----------+     +----------+  +----------+  +----------+  +-------------+
    ///                                                                       ^            ^               ^
    ///                                                                       |            |               |
    ///                                                                       +------------+---------------+
    ///                                                                                    |
    ///                                                                 Redistribute cells in all this siblings
    /// ```
    ///
    /// Next step is moving all the keys in sibling pages and the dividers in
    /// the parent page out:
    ///
    ///
    /// ```text
    /// In-memory copies of each cell (they come already sorted):
    /// +----+ +----+ +----+ +----+ +----+ +----+ +----+ +----+ +----+ +----+ +----+ +----+
    /// | 20 | | 21 | | 22 | | 23 | | 24 | | 25 | | 26 | | 27 | | 28 | | 29 | | 30 | | 31 |
    /// +----+ +----+ +----+ +----+ +----+ +----+ +----+ +----+ +----+ +----+ +----+ +----+
    ///
    /// State of the tree after taking keys out:
    ///                                              PAGE 0
    ///                                            +--------+
    ///                         +------------------|   15   |-----------------+
    ///                        /                   +--------+                  \
    ///                       /                                                 \ PAGE 2
    ///                  +--------+                                            +----------+
    ///       +----------| 4,8,11 |---------+                     +------------| 19       |-----------+
    ///      /           +--------+          \                   /             +----------+            \
    ///     /             /     \             \                 /               /        \              \ PAGE 10
    /// +-------+  +-------+  +-------+  +----------+     +----------+  +----------+  +----------+  +----------+
    /// | 1,2,3 |  | 5,6,7 |  | 9,10  |  | 12,13,14 |     | 16,17,18 |  |          |  |          |  |          |
    /// +-------+  +-------+  +-------+  +----------+     +----------+  +----------+  +----------+  +----------+
    /// ```
    ///
    /// The allocation subalgorithm will determine that we need an extra page
    /// to store all the cells so it will allocate it:
    ///
    /// ```text
    /// In-memory copies of each cell (they come already sorted):
    /// +----+ +----+ +----+ +----+ +----+ +----+ +----+ +----+ +----+ +----+ +----+ +----+
    /// | 20 | | 21 | | 22 | | 23 | | 24 | | 25 | | 26 | | 27 | | 28 | | 29 | | 30 | | 31 |
    /// +----+ +----+ +----+ +----+ +----+ +----+ +----+ +----+ +----+ +----+ +----+ +----+
    ///
    /// State of the tree after taking keys out:
    ///                                              PAGE 0
    ///                                            +--------+
    ///                         +------------------|   15   |-----------------+
    ///                        /                   +--------+                  \
    ///                       /                                                 \ PAGE 2
    ///                  +--------+                                            +----------+
    ///       +----------| 4,8,11 |---------+                     +------------| 19       |-----------+
    ///      /           +--------+          \                   /             +----------+            \
    ///     /             /     \             \                 /               /        \              \ PAGE 10     PAGE 11
    /// +-------+  +-------+  +-------+  +----------+     +----------+  +----------+  +----------+  +----------+  +----------+
    /// | 1,2,3 |  | 5,6,7 |  | 9,10  |  | 12,13,14 |     | 16,17,18 |  |          |  |          |  |          |  |          |
    /// +-------+  +-------+  +-------+  +----------+     +----------+  +----------+  +----------+  +----------+  +----------+
    ///                                                                                                             New Page
    /// ```
    ///
    /// No we run the redisribution subalgorithm:
    ///
    /// ```text
    ///                                              PAGE 0
    ///                                            +--------+
    ///                         +------------------|   15   |--------------------+
    ///                        /                   +--------+                     \
    ///                       /                                                    \ PAGE 2
    ///                  +--------+                                               +-------------+
    ///       +----------| 4,8,11 |---------+                     +---------------| 19,23,26,29 |-----------+
    ///      /           +--------+          \                   /                +-------------+            \
    ///     /             /     \             \                 /                 /       |      \ PAGE 10    \ PAGE 11
    /// +-------+  +-------+  +-------+  +----------+     +----------+  +----------+  +-------+  +-------+  +-------+
    /// | 1,2,3 |  | 5,6,7 |  | 9,10  |  | 12,13,14 |     | 16,17,18 |  | 20,21,22 |  | 24,25 |  | 27,28 |  | 30,31 |
    /// +-------+  +-------+  +-------+  +----------+     +----------+  +----------+  +-------+  +-------+  +-------+
    /// ```
    ///
    /// As you can see, the leaf nodes are all balanced now, but PAGE 2 is
    /// overflow and its left sibling cannot take more keys because it's full.
    /// So... repeat the same algorithm on the parent page. Take keys out and
    /// allocate extra pages:
    ///
    /// ```text
    /// In-memory copies of each cell (they come already sorted):
    /// +----+ +----+ +----+ +----+ +----+ +----+ +----+ +----+
    /// |  4 | |  8 | | 11 | | 15 | | 19 | | 23 | | 26 | | 29 |
    /// +----+ +----+ +----+ +----+ +----+ +----+ +----+ +----+
    ///
    ///                                              PAGE 0
    ///                                            +--------+
    ///                         +------------------|        |--------------------+
    ///                        /                   +--------+                     \
    ///                       /                                                    \ PAGE 2                                    PAGE 12
    ///                  +--------+                                               +-------------+                          +-------------+
    ///       +----------|        |---------+                     +---------------|             |-----------+              |             |
    ///      /           +--------+          \                   /                +-------------+            \             +-------------+
    ///     /             /     \             \                 /                 /       |      \ PAGE 10    \ PAGE 11        New Page
    /// +-------+  +-------+  +-------+  +----------+     +----------+  +----------+  +-------+  +-------+  +-------+
    /// | 1,2,3 |  | 5,6,7 |  | 9,10  |  | 12,13,14 |     | 16,17,18 |  | 20,21,22 |  | 24,25 |  | 27,28 |  | 30,31 |
    /// +-------+  +-------+  +-------+  +----------+     +----------+  +----------+  +-------+  +-------+  +-------+
    /// ```
    ///
    /// Redistribute:
    ///
    /// ```text
    ///                                                      PAGE 0
    ///                                                    +-------+
    ///                  +---------------------------------| 11,23 |--------------------------------+
    ///                 /                                  +-------+                                 \
    ///                /                                       |                                      \
    ///             +-----+                                +-------+ PAGE 2                       +-------+ PAGE 12
    ///        +----| 4,8 |-----+                  +-------| 15,19 |-------+                 +----| 26,29 |----+
    ///       /     +-----+      \                /        +-------+        \               /     +-------+     \
    ///      /         |          \              /             |             \             /          | PAGE 10  \ PAGE 11
    /// +-------+  +-------+  +-------+    +----------+  +----------+  +----------+    +-------+  +-------+  +-------+
    /// | 1,2,3 |  | 5,6,7 |  | 9,10  |    | 12,13,14 |  | 16,17,18 |  | 20,21,22 |    | 24,25 |  | 27,28 |  | 30,31 |
    /// +-------+  +-------+  +-------+    +----------+  +----------+  +----------+    +-------+  +-------+  +-------+
    /// ```
    ///
    /// And now the tree is completely balanced again. Nice work! If the root
    /// went overflow in this process, then we'd have to execute step 3
    /// discussed above.
    ///
    /// # Underflow Merge
    ///
    /// The exact same code that runs the "splits" we've talked about before
    /// can also run node "merges". Let's start from where we've left:
    ///
    /// ```text
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
    ///
    /// Deleting key 1 will not do anything to the tree:
    ///
    /// ```text
    ///                                                    +-------+
    ///                  +---------------------------------| 11,23 |--------------------------------+
    ///                 /                                  +-------+                                 \
    ///                /                                       |                                      \
    ///             +-----+                                +-------+                              +-------+
    ///        +----| 4,8 |-----+                  +-------| 15,19 |-------+                 +----| 26,29 |----+
    ///       /     +-----+      \                /        +-------+        \               /     +-------+     \
    ///      /         |          \              /             |             \             /          |          \
    /// +-------+  +-------+  +-------+    +----------+  +----------+  +----------+    +-------+  +-------+  +-------+
    /// |  2,3  |  | 5,6,7 |  | 9,10  |    | 12,13,14 |  | 16,17,18 |  | 20,21,22 |    | 24,25 |  | 27,28 |  | 30,31 |
    /// +-------+  +-------+  +-------+    +----------+  +----------+  +----------+    +-------+  +-------+  +-------+
    ///  ^
    ///  |
    ///  Key 1 no longer here
    /// ```
    ///
    /// Deleting key 2 now will just reorganize the keys in the leftmost leaf
    /// nodes:
    ///
    /// ```text
    ///                                                    +-------+
    ///                  +---------------------------------| 11,23 |--------------------------------+
    ///                 /                                  +-------+                                 \
    ///                /                                       |                                      \
    ///             +-----+                                +-------+                              +-------+
    ///        +----| 5,8 |-----+                  +-------| 15,19 |-------+                 +----| 26,29 |----+
    ///       /     +-----+      \                /        +-------+        \               /     +-------+     \
    ///      /         |          \              /             |             \             /          |          \
    /// +-------+  +-------+  +-------+    +----------+  +----------+  +----------+    +-------+  +-------+  +-------+
    /// |  3,4  |  |  6,7  |  | 9,10  |    | 12,13,14 |  | 16,17,18 |  | 20,21,22 |    | 24,25 |  | 27,28 |  | 30,31 |
    /// +-------+  +-------+  +-------+    +----------+  +----------+  +----------+    +-------+  +-------+  +-------+
    ///     ^          ^
    ///     |          |
    ///     +----+------
    ///          |
    ///   Reorganize keys
    /// ```
    ///
    /// Now here comes the magic. Deleting key 3 or any of the keys in the
    /// leftmost leaf nodes will cause an underflow, and the balancing algorithm
    /// will do its job again:
    ///
    /// ```text
    ///                                                    +-------+
    ///                  +---------------------------------| 11,23 |--------------------------------+
    ///                 /                                  +-------+                                 \
    ///                /                                       |                                      \
    ///             +-----+                                +-------+                              +-------+
    ///        +----| 5,8 |-----+                  +-------| 15,19 |-------+                 +----| 26,29 |----+
    ///       /     +-----+      \                /        +-------+        \               /     +-------+     \
    ///      /         |          \              /             |             \             /          |          \
    /// +-------+  +-------+  +-------+    +----------+  +----------+  +----------+    +-------+  +-------+  +-------+
    /// |   4   |  |  6,7  |  | 9,10  |    | 12,13,14 |  | 16,17,18 |  | 20,21,22 |    | 24,25 |  | 27,28 |  | 30,31 |
    /// +-------+  +-------+  +-------+    +----------+  +----------+  +----------+    +-------+  +-------+  +-------+
    ///     ^
    ///     |
    /// Key 3 deleted
    /// ```
    ///
    /// So, first step, take keys out and leave pagees empty:
    ///
    /// ```text
    /// In-memory copies of each cell:
    /// +---+ +---+ +---+ +---+ +---+ +---+ +----+
    /// | 4 | | 5 | | 6 | | 7 | | 8 | | 9 | | 10 |
    /// +---+ +---+ +---+ +---+ +---+ +---+ +----+
    ///                                                    +-------+
    ///                  +---------------------------------| 11,23 |--------------------------------+
    ///                 /                                  +-------+                                 \
    ///                /                                       |                                      \
    ///             +-----+                                +-------+                              +-------+
    ///        +----|     |-----+                  +-------| 15,19 |-------+                 +----| 26,29 |----+
    ///       /     +-----+      \                /        +-------+        \               /     +-------+     \
    ///      /         |          \              /             |             \             /          |          \
    /// +-------+  +-------+  +-------+    +----------+  +----------+  +----------+    +-------+  +-------+  +-------+
    /// |       |  |       |  |       |    | 12,13,14 |  | 16,17,18 |  | 20,21,22 |    | 24,25 |  | 27,28 |  | 30,31 |
    /// +-------+  +-------+  +-------+    +----------+  +----------+  +----------+    +-------+  +-------+  +-------+
    /// ```
    ///
    /// We have a total of seven keys. Since each node can fit 3 keys, we can
    /// use 2 nodes to fit 6 keys total and one divider in the parent. Which
    /// means one of the pages that we previously had is no longer needed:
    ///
    /// ```text
    /// In-memory copies of each cell:
    /// +---+ +---+ +---+ +---+ +---+ +---+ +----+
    /// | 4 | | 5 | | 6 | | 7 | | 8 | | 9 | | 10 |
    /// +---+ +---+ +---+ +---+ +---+ +---+ +----+
    ///                                                    +-------+
    ///                  +---------------------------------| 11,23 |--------------------------------+
    ///                 /                                  +-------+                                 \
    ///                /                                       |                                      \
    ///             +-----+                                +-------+                              +-------+
    ///        +----|     |                        +-------| 15,19 |-------+                 +----| 26,29 |----+
    ///       /     +-----+                       /        +-------+        \               /     +-------+     \
    ///      /         |                         /             |             \             /          |          \
    /// +-------+  +-------+               +----------+  +----------+  +----------+    +-------+  +-------+  +-------+
    /// |       |  |       |    X          | 12,13,14 |  | 16,17,18 |  | 20,21,22 |    | 24,25 |  | 27,28 |  | 30,31 |
    /// +-------+  +-------+               +----------+  +----------+  +----------+    +-------+  +-------+  +-------+
    ///                         ^
    ///                         |
    ///         The page we had here was dropped
    /// ```
    ///
    /// Now redistribute:
    ///
    /// ```text
    ///                                       +-------+
    ///              +------------------------| 11,23 |--------------------------------+
    ///             /                         +-------+                                 \
    ///            /                              |                                      \
    ///        +-----+                        +-------+                              +-------+
    ///        |  7  |                +-------| 15,19 |-------+                 +----| 26,29 |----+
    ///        +-----+               /        +-------+        \               /     +-------+     \
    ///        /    \               /             |             \             /          |          \
    /// +-------+  +--------+  +----------+  +----------+  +----------+    +-------+  +-------+  +-------+
    /// | 4,5,6 |  | 8,9,10 |  | 12,13,14 |  | 16,17,18 |  | 20,21,22 |    | 24,25 |  | 27,28 |  | 30,31 |
    /// +-------+  +--------+  +----------+  +----------+  +----------+    +-------+  +-------+  +-------+
    /// ```
    ///
    /// Leaf nodes are correct but the parent page has gone "underflow". There
    /// are a total of 5 keys in the internal nodes and 2 in the parent page.
    /// So again we have 7 keys to distribute, which means we can drop another
    /// page because we only need two nodes and a divider in the parent:
    ///
    /// ```text
    /// In-memory copies of each cell:
    /// +---+ +----+ +----+ +----+ +----+ +----+ +----+
    /// | 7 | | 11 | | 15 | | 19 | | 23 | | 26 | | 29 |
    /// +---+ +----+ +----+ +----+ +----+ +----+ +----+
    ///
    ///                                       +-------+
    ///              +------------------------|       |                              Page dropped
    ///             /                         +-------+                                   |
    ///            /                              |                                       v
    ///        +-----+                        +-------+
    ///        |     |                +-------|       |-------+                           X
    ///        +-----+               /        +-------+        \
    ///        /    \               /             |             \
    /// +-------+  +--------+  +----------+  +----------+  +----------+    +-------+  +-------+  +-------+
    /// | 4,5,6 |  | 8,9,10 |  | 12,13,14 |  | 16,17,18 |  | 20,21,22 |    | 24,25 |  | 27,28 |  | 30,31 |
    /// +-------+  +--------+  +----------+  +----------+  +----------+    +-------+  +-------+  +-------+
    /// ```
    ///
    /// Redistribute again:
    ///
    ///
    /// ```text
    ///                                             +------+
    ///                          +------------------|  19  |----------------------+
    ///                         /                   +------+                       \
    ///                        /                                                    \
    ///                   +---------+                                         +----------+
    ///        +----------| 7,11,15 |-----------+                    +--------| 23,26,29 |---------+
    ///       /           +---------+            \                  /         +----------+          \
    ///      /             /      \               \                /           /        \            \
    /// +-------+  +--------+  +----------+  +----------+    +----------+  +-------+  +-------+  +-------+
    /// | 4,5,6 |  | 8,9,10 |  | 12,13,14 |  | 16,17,18 |    | 20,21,22 |  | 24,25 |  | 27,28 |  | 30,31 |
    /// +-------+  +--------+  +----------+  +----------+    +----------+  +-------+  +-------+  +-------+
    /// ```
    ///
    /// Now the entire tree is balanced again. If the root went underflow in
    /// this process, we would execute step 2 discussed at the beginning of
    /// this doc comment. With all this information in mind, you can now attempt
    /// to understand the source code.
    fn balance(&mut self, mut page: PageNumber, mut parents: Vec<PageNumber>) -> io::Result<()> {
        let node = self.cache.get(page)?;

        // Nothing to do, the node is balanced.
        if !node.is_overflow() && !node.is_underflow() {
            return Ok(());
        }

        // Root underflow.
        if node.is_root() && node.is_underflow() {
            let child_page = node.header().right_child;
            let mut child_node = self.cache.get(child_page)?.clone();
            self.cache.get_mut(page)?.append(&mut child_node);
            self.free_page(child_page);

            return Ok(());
        }

        // Root overflow.
        if node.is_root() && node.is_overflow() {
            let mut old_root = Page::new(self.allocate_page(), self.cache.pager.page_size as u16);

            let root = self.cache.get_mut(page)?;
            old_root.append(root);

            root.header_mut().right_child = old_root.number;

            parents.push(page);
            page = old_root.number;

            self.cache.load_from_mem(old_root)?;
        }

        // Internal/Leaf node Overflow/Underlow.
        let parent_page = parents.remove(parents.len() - 1);
        let mut siblings = self.load_siblings(page, parent_page)?;

        let mut cells = VecDeque::new();
        let divider_idx = siblings[0].index;

        // Make copies of cells in order.
        for (i, sibling) in siblings.iter().enumerate() {
            cells.extend(self.cache.get_mut(sibling.page)?.drain(..));
            if i < siblings.len() - 1 {
                let mut divider = self.cache.get_mut(parent_page)?.remove(divider_idx);
                divider.header.left_child = self.cache.get(sibling.page)?.header().right_child;
                cells.push_back(divider);
            }
        }

        let usable_space = Page::usable_space(self.cache.pager.page_size as _);

        let mut total_size_in_each_page = vec![0];
        let mut number_of_cells_per_page = vec![0];

        // Precompute left biased distribution.
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

        // Account for underflow towards the right.
        if number_of_cells_per_page.len() >= 2 {
            let mut div_cell = cells.len() - number_of_cells_per_page.last().unwrap() - 1;

            for i in (1..=(total_size_in_each_page.len() - 1)).rev() {
                while total_size_in_each_page[i] < usable_space / 2 {
                    number_of_cells_per_page[i] += 1;
                    total_size_in_each_page[i] += &cells[div_cell].storage_size();

                    number_of_cells_per_page[i - 1] -= 1;
                    total_size_in_each_page[i - 1] -= &cells[div_cell - 1].storage_size();
                    div_cell -= 1;
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
            .get(siblings.last().unwrap().page)?
            .header()
            .right_child;

        // Allocate missing pages.
        while siblings.len() < number_of_cells_per_page.len() {
            let new_page = Page::new(self.allocate_page(), self.cache.pager.page_size as _);
            let parent_index = siblings.last().unwrap().index + 1;
            siblings.push(Sibling::new(new_page.number, parent_index));
            self.cache.load_from_mem(new_page)?;
        }

        // Free unused pages.
        while number_of_cells_per_page.len() < siblings.len() {
            self.free_page(siblings.pop().unwrap().page);
        }

        // Begin redistribution.
        for (i, n) in number_of_cells_per_page.iter().enumerate() {
            let page = self.cache.get_mut(siblings[i].page)?;
            for _ in 0..*n {
                page.push(cells.pop_front().unwrap());
            }

            if i < siblings.len() - 1 {
                let mut divider = cells.pop_front().unwrap();
                page.header_mut().right_child = divider.header.left_child;
                divider.header.left_child = siblings[i].page;
                self.cache
                    .get_mut(parent_page)?
                    .insert(siblings[i].index, divider);
            }
        }

        let last_sibling = siblings.last().unwrap();

        // Fix children pointers.
        self.cache
            .get_mut(last_sibling.page)?
            .header_mut()
            .right_child = old_right_child;

        if last_sibling.index == self.cache.get(parent_page)?.len() {
            self.cache.get_mut(parent_page)?.header_mut().right_child = last_sibling.page;
        } else {
            self.cache
                .get_mut(parent_page)?
                .cell_mut(divider_idx)
                .header
                .left_child = last_sibling.page;
        }

        // Done, propagate upwards.
        self.balance(parent_page, parents)?;

        Ok(())
    }

    /// Given a page number and its parent page number, this function attemps to
    /// load a list of  `B * 2 + 1` sibling pages, where
    /// B = [`Self::balance_siblings_per_side`]. For example, suppose we have
    /// the following [`BTree`] of order 4 where
    /// [`BTree::balance_siblings_per_side`] equals 1:
    ///
    /// ```text
    ///                             PAGE 0
    ///                           +--------+
    ///                   +-------| 4,8,12 |--------+
    ///                 /         +--------+         \
    ///               /            /      \           \
    ///           +-------+  +-------+  +---------+  +----------+
    ///           | 1,2,3 |  | 5,6,7 |  | 9,10,11 |  | 13,14,15 |
    ///           +-------+  +-------+  +---------+  +----------+
    ///            PAGE 1     PAGE 2       PAGE 3       PAGE 4
    /// ```
    ///
    /// Given page 2 and parent page 0, this function will return `[1,2,3]`.
    /// When the given page is the first one or the last one in the parent
    /// children pointers, only one side will be available for loading pages,
    /// but the returned number of pages will be the same.
    ///
    /// For instance, given page 4, the returned list will be `[2,3,4]` whereas
    /// given page 1 the returned list will be `[1,2,3]`. If the cache size is
    /// big enough, all siblings can be loaded into memory at once.
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

        page as PageNumber
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

    // Testing/Debugging only.
    fn read_into_mem(&mut self, root: PageNumber, buf: &mut Vec<Page>) -> io::Result<()> {
        for page in self.cache.get(root)?.iter_children().collect::<Vec<_>>() {
            self.read_into_mem(page, buf)?;
        }

        let node = self.cache.get(root)?.clone();
        buf.push(node);

        Ok(())
    }

    // Testing/Debugging only.
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

    // Testing/Debugging only.
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
        storage::page::{Cell, Page, CELL_ALIGNMENT, PAGE_HEADER_SIZE},
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
