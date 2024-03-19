//! Disk BTree data structure implementation.
//!
//! See [`BTree`] for details.

use std::{
    cmp::{min, Ordering, Reverse},
    collections::{BinaryHeap, HashSet, VecDeque},
    io::{self, Read, Seek, Write},
    mem,
};

use super::page::{Cell, OverflowPage, Page, SlotId};
use crate::{
    paging::pager::{PageNumber, Pager},
    sql::statement::DataType,
};

/// [`BTree`] key comparator. Entries are stored in binary, so we need a way to
/// determine the correct [`Ordering`]. Whenever two entries need to be
/// compared, the [`BTree`] will call [`BytesCmp::bytes_cmp`] passing both
/// binary slices as parameters. For example, suppose we need to compare the
/// following two slices:
///
/// ```no_run
/// let A = [1, 0, 0, 0];
/// let B = [2, 0, 0, 0, 0, 0, 0, 0];
/// ```
///
/// The [`BTree`] will just call the comparator function with `A` and `B`, it
/// is up to the implementer to determine the [`Ordering`]. The format of `A`
/// and `B` is also defined by the [`BTree`] user, since the [`BTree`] only
/// receives binary buffers as parameters.
///
/// We provide a couple of reusable comparators in this module:
/// - [`FixedSizeMemCmp`]
/// - [`StringCmp`]
pub(crate) trait BytesCmp {
    /// Compares two byte arrays and returns the corresponding [`Ordering`].
    ///
    /// At the [`BTree`] level we don't care how that is done. Upper levels can
    /// parse the binary to obtain an [`Ordering`] instance, they can store
    /// the entry in such a format that they can tell the [`Ordering`] by
    /// looking at the bytes, or they can do anything else, we don't care here.
    fn bytes_cmp(&self, a: &[u8], b: &[u8]) -> Ordering;
}

/// Compares the first `self.0` number of bytes using the good old `memcmp`.
/// This is more useful than it seems at first glance because if we store
/// integer keys at the beginning of the binary buffer in big endian format,
/// then this is all we need to successfuly determine the [`Ordering`].
pub(crate) struct FixedSizeMemCmp(pub usize);

impl FixedSizeMemCmp {
    /// Creates a comparator for a certain type.
    ///
    /// ```no_run
    /// let comparator = FixedSizeMemCmp::for_type::<u32>();
    ///
    /// let a = [1, 0, 0, 0, 0, 0, 0]
    /// let b = [2, 0, 0, 0]
    ///
    /// assert_eq!(comparator.bytes_cmp(a, b), std::cmp::Ordering::Less);
    /// ```
    ///
    /// Using this with structs or other complex data types probably doesn't
    /// make sense.
    pub fn for_type<T>() -> Self {
        Self(mem::size_of::<T>())
    }
}

impl BytesCmp for FixedSizeMemCmp {
    fn bytes_cmp(&self, a: &[u8], b: &[u8]) -> Ordering {
        a[..self.0].cmp(&b[..self.0])
    }
}

/// Compares UTF-8 strings.
///
/// Assumes that the buffers have this format:
///
/// ```text
/// self.0 bytes
/// +---------+
/// V         V
/// +---------+--------+--------+--------+-------+
/// | STR LEN | BYTE 0 | BYTE 1 | BYTE 2 |  ...  |
/// +---------+--------+--------+--------+-------+
///           ^                          ^
///           +--------------------------+
///             STR LEN of string bytes
/// ```
///
/// Then computes the total length of the string in bytes by reading the first
/// `self.0` bytes as a little endian integer and once the total length is known
/// [`str`] instances can be created.
pub(crate) struct StringCmp(pub usize);

impl BytesCmp for StringCmp {
    fn bytes_cmp(&self, a: &[u8], b: &[u8]) -> Ordering {
        debug_assert!(
            self.0 <= 2,
            "currently strings longer than 65535 bytes are not supported"
        );

        let mut buf = [0; std::mem::size_of::<usize>()];

        buf[..self.0].copy_from_slice(&a[..self.0]);

        let len_a = usize::from_le_bytes(buf);

        buf.fill(0);
        buf[..self.0].copy_from_slice(&b[..self.0]);

        let len_b = usize::from_le_bytes(buf);

        // TODO: Not sure if unwrap() can actually panic here. When we insert
        // data we have a valid [`String`] instance and we call String::as_bytes()
        // to serialize it into binary. If unwrap() can't panic then we should
        // use the unchecked version of from_utf8 that doesn't loop through the
        // entire string to check that all bytes are valid UTF-8.
        std::str::from_utf8(&a[self.0..self.0 + len_a])
            .unwrap()
            .cmp(std::str::from_utf8(&b[self.0..self.0 + len_b]).unwrap())
    }
}

impl BytesCmp for Box<dyn BytesCmp> {
    fn bytes_cmp(&self, a: &[u8], b: &[u8]) -> Ordering {
        self.as_ref().bytes_cmp(a, b)
    }
}

impl From<&DataType> for Box<dyn BytesCmp> {
    /// Easy way to obtain a [`BytesCmp`] impl at runtime based on SQL data
    /// types.
    fn from(data_type: &DataType) -> Self {
        match data_type {
            DataType::Varchar(max) => {
                let length_bytes = if *max <= u8::MAX as usize { 1 } else { 2 };
                Box::new(StringCmp(length_bytes))
            }

            fixed => {
                let size = match fixed {
                    DataType::BigInt | DataType::UnsignedBigInt => mem::size_of::<i64>(),
                    DataType::Int | DataType::UnsignedInt => mem::size_of::<i32>(),
                    _ => unreachable!(),
                };

                Box::new(FixedSizeMemCmp(size))
            }
        }
    }
}

/// The result of a search in the [`BTree`] structure.
pub(crate) struct Search {
    /// Page number of the node where the search ended.
    pub page: PageNumber,

    /// Contains an [`Ok`] value with the index where the search matched or an
    /// [`Err`] value with the index where the searched key should be located.
    pub index: Result<u16, u16>,
}

/// The result of a remove operation on the BTree.
struct Removal {
    /// The removed cell.
    cell: Box<Cell>,
    /// Page number of the leaf node where a substitute was found.
    leaf_node: PageNumber,
    /// If the removal ocurred on an internal node, then this is its page number.
    internal_node: Option<PageNumber>,
}

/// Stores the page number and the index in the parent page entries that points
/// back to this node.
///
/// See [`BTree::load_siblings`] and [`BTree::balance`] for details.
#[derive(Debug, Clone, Copy)]
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

/// Used to search either the minimum or maximum key of a subtree.
///
/// See [`BTree::search_leaf_key`] for details.
enum LeafKeySearch {
    /// Maximum key in a leaf node.
    Max,
    /// Minimum key in a leaf node.
    Min,
}

/// Used for reading payloads stored in the BTree.
///
/// Most of the times the BTree can simply return a reference, but large
/// payloads require reassembly to construct a buffer with contiguous data.
#[derive(Debug, PartialEq)]
pub(crate) enum Payload<'s> {
    /// Payload did not need reassembly, so we returned a reference to the
    /// slotted page buffer.
    PageRef(&'s [u8]),

    /// Payload was too large and needed reassembly.
    Reassembled(Box<[u8]>),
}

impl<'s> AsRef<[u8]> for Payload<'s> {
    fn as_ref(&self) -> &[u8] {
        match self {
            Self::PageRef(reference) => reference,
            Self::Reassembled(boxed) => boxed,
        }
    }
}

/// B*-Tree implementation inspired by "Art of Computer Programming Volume 3:
/// Sorting and Searching" and SQLite 2.X.X
///
/// # About BTrees
///
/// BTrees are a family of tree data structures that always maintain their
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
/// cell. When a [`Cell`] is too big it points to a linked list of
/// [`OverflowPage`] instances. Overall, this is how the BTree really looks
/// like:
///
/// ```text
///                                              BTREE           +---------------+
///                                              PAGE            |               |
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
/// +-------------+-----------------------------------------------+      +-------------+-------------------------------------+
/// | +----+----+ | +----+----+             +----+---+ +----+---+ |      | +----+----+ | +----+             +----+---+-----+ |
/// | | RC | NS | | | O1 | O2 | ->   FS  <- | LC | D | | LC | D | |      | | RC | NS | | | O1 | ->   FS  <- | LC | D | OVF | |
/// | +----+----+ | +----+----+             +----+---+ +----+---+ |      | +----+----+ | +----+             +----+---+-----+ |
/// +-------------+---|----|--------------------------------------+      +-------------+---|----------------------------|----+
///                   |    |                ^          ^                                   |                ^           |
///                   |    |                |          |                                   |                |           |
///                   |    +----------------+          |                                   +----------------+           |
///                   +--------------------------------+                                                                |
///                                                                          +------------------------------------------+
///                                                                          |
///              OVERFLOW PAGE                                               V
///            +------+-------------------------------------------+      +------+-------------------------------------------+
///            | NEXT | OVERFLOW PAYLOAD                          |      | NEXT | OVERFLOW PAYLOAD                          |
///            +------+-------------------------------------------+      +------+-------------------------------------------+
///               ^                                                         |
///               |                                                         |
///               +---------------------------------------------------------+
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
/// - OVF: Overflow
///
/// Right child is stored in the page header and is always the "last child" of
/// a node. Or in other words, the child that has keys grater than any key in
/// the current node. Left child is stored in the cell header and points to
/// the child that has keys less than the one in the cell. For the slotted page
/// details and omitted fields see the documentation of [`super::page`] module.
pub(crate) struct BTree<'c, F, C> {
    /// Root page.
    root: PageNumber,

    /// Pager instance.
    pager: &'c mut Pager<F>,

    /// Bytes comparator used to obtain [`Ordering`] instances from binary data.
    comparator: C,

    /// Number of siblings to examine at each side when balancing a node.
    ///
    /// See [`Self::load_siblings`] and [`Self::balance`].
    balance_siblings_per_side: usize,

    /// Forces pages to store at least this number of [`Cell`] instances.
    minimum_keys: usize,
}

/// Default value for [`BTree::balance_siblings_per_side`].
pub(crate) const DEFAULT_BALANCE_SIBLINGS_PER_SIDE: usize = 1;

/// Default value for [`BTree::minimum_keys`].
pub(crate) const DEFAULT_MINIMUM_KEYS: usize = 4;

impl<'p, I, C: BytesCmp> BTree<'p, I, C> {
    pub fn new(pager: &'p mut Pager<I>, root: PageNumber, comparator: C) -> Self {
        Self {
            pager,
            root,
            comparator,
            balance_siblings_per_side: DEFAULT_BALANCE_SIBLINGS_PER_SIDE,
            minimum_keys: DEFAULT_MINIMUM_KEYS,
        }
    }
}

impl<'p, I: Seek + Read + Write, C: BytesCmp> BTree<'p, I, C> {
    /// Returns the value corresponding to the key.
    ///
    /// See [`Self::search`] for details.
    pub fn get(&mut self, entry: &[u8]) -> io::Result<Option<Payload>> {
        let search = self.search(self.root, entry, &mut Vec::new())?;

        match search.index {
            Err(_) => Ok(None),
            Ok(index) => Ok(Some(reassemble_payload(self.pager, search.page, index)?)),
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
    pub fn search(
        &mut self,
        page: PageNumber,
        entry: &[u8],
        parents: &mut Vec<PageNumber>,
    ) -> io::Result<Search> {
        // Search key in this node.
        let index = self.binary_search(page, entry)?;

        let node = self.pager.get(page)?;

        // We found the key or we're already at the bottom, stop recursion.
        if index.is_ok() || node.is_leaf() {
            return Ok(Search { page, index });
        }

        // No luck, keep recursing downwards.
        parents.push(page);
        let next_node = node.child(index.unwrap_err());

        self.search(next_node, entry, parents)
    }

    /// Binary search with support for overflow data.
    ///
    /// Returns an [`Ok`] result containing the index where `entry` was found or
    /// an [`Err`] result containing the index where `entry` should have been
    /// found if present.
    fn binary_search(&mut self, page: PageNumber, entry: &[u8]) -> io::Result<Result<u16, u16>> {
        let mut size = self.pager.get(page)?.len();

        let mut left = 0;
        let mut right = size;
        while left < right {
            let mid = left + size / 2;

            let cell = self.pager.get(page)?.cell(mid);

            let overflow_buf: Box<[u8]>;

            // TODO: Figure out if we actually need to reassemble the payload.
            // We could ask the comparator through the [`BytesCmp`] trait if
            // it needs the entire buffer or not. When comparing strings for
            // example, "a" is always less than "abcdefg", so there's no point
            // in reassembling the entire string.
            let payload = if cell.header.is_overflow {
                match reassemble_payload(self.pager, page, mid)? {
                    Payload::Reassembled(buf) => overflow_buf = buf,
                    _ => unreachable!(),
                }
                &overflow_buf
            } else {
                &cell.content
            };

            match self.comparator.bytes_cmp(payload, entry) {
                Ordering::Equal => return Ok(Ok(mid)),
                Ordering::Greater => right = mid,
                Ordering::Less => left = mid + 1,
            }

            size = right - left;
        }

        Ok(Err(left))
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
    pub fn insert(&mut self, entry: Vec<u8>) -> io::Result<()> {
        let mut parents = Vec::new();
        let search = self.search(self.root, &entry, &mut parents)?;

        let mut new_cell = self.alloc_cell(entry)?;
        let node = self.pager.get_mut(search.page)?;

        match search.index {
            // Key found, swap value.
            Ok(index) => {
                new_cell.header.left_child = node.cell(index).header.left_child;
                let old_cell = node.replace(index, new_cell);
                self.free_cell(old_cell)?;
            }
            // Key not found, insert new entry.
            Err(index) => node.insert(index, new_cell),
        };

        self.balance(search.page, &mut parents)
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
    pub fn remove(&mut self, entry: &[u8]) -> io::Result<Option<Box<Cell>>> {
        let mut parents = Vec::new();
        let Some(Removal {
            cell,
            leaf_node,
            internal_node,
        }) = self.remove_entry(entry, &mut parents)?
        else {
            return Ok(None);
        };

        self.balance(leaf_node, &mut parents)?;

        // Since data is variable size it could happen that we peek a large
        // substitute from a leaf node to replace a small key in an internal
        // node, leaving the leaf node balanced and the internal node overflow.
        if let Some(node) = internal_node {
            // This algorithm is O(n) but the height of the tree grows
            // logarithmically so there shoudn't be that many elements to search
            // here.
            if let Some(index) = parents.iter().position(|n| n == &node) {
                parents.drain(index..);
                self.balance(node, &mut parents)?;
            }
        }

        Ok(Some(cell))
    }

    /// Finds the node where `key` is located and removes it from the page.
    ///
    /// The removes [`Cell`] is replaced with either its predecessor or
    /// successor in the case of internal nodes. [`Self::balance`] must be
    /// called on the leaf node after this operation, and possibly on the
    /// internal node if the leaf was balanced. See [`Self::remove`] for more
    /// details.
    fn remove_entry(
        &mut self,
        entry: &[u8],
        parents: &mut Vec<PageNumber>,
    ) -> io::Result<Option<Removal>> {
        let search = self.search(self.root, entry, parents)?;
        let node = self.pager.get(search.page)?;

        // Can't remove entry, key not found.
        if search.index.is_err() {
            return Ok(None);
        }

        let index = search.index.unwrap();

        // Leaf node is the simplest case, remove key and pop off the stack.
        if node.is_leaf() {
            let cell = self.pager.get_mut(search.page)?.remove(index);
            return Ok(Some(Removal {
                cell,
                leaf_node: search.page,
                internal_node: None,
            }));
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
            if self.pager.get(left_child)?.len() >= self.pager.get(right_child)?.len() {
                self.search_max_key(left_child, parents)?
            } else {
                self.search_min_key(right_child, parents)?
            };

        let mut substitute = self.pager.get_mut(leaf_node)?.remove(key_idx);

        let node = self.pager.get_mut(search.page)?;

        substitute.header.left_child = node.child(index);
        let cell = node.replace(index, substitute);

        Ok(Some(Removal {
            cell,
            leaf_node,
            internal_node: Some(search.page),
        }))
    }

    /// Traverses the tree all the way down to the leaf nodes, following the
    /// path specified by [`LeafKeySearch`].
    ///
    /// [`LeafKeySearch::Max`] will always choose the last child for recursion,
    /// while [`LeafKeySearch::Min`] will always choose the first child. This
    /// function is used to find successors or predecessors of keys in internal
    /// nodes.
    fn search_leaf_key(
        &mut self,
        page: PageNumber,
        parents: &mut Vec<PageNumber>,
        leaf_key_search: LeafKeySearch,
    ) -> io::Result<(PageNumber, u16)> {
        let node = self.pager.get(page)?;

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

    /// Returns the greatest entry in this tree.
    pub fn max(&mut self) -> io::Result<Option<Payload>> {
        if self.pager.get(self.root)?.is_empty() {
            return Ok(None);
        }

        let (page, slot) = self.search_max_key(self.root, &mut Vec::new())?;

        if self.pager.get(page)?.is_empty() {
            return Ok(None);
        }

        reassemble_payload(self.pager, page, slot).map(Some)
    }

    /// B*-Tree balancing algorithm inspired by (or rather stolen from) SQLite
    /// 2.X.X. Take a look at the original source code here:
    ///
    /// <https://github.com/antoniosarosi/sqlite2-btree-visualizer/blob/master/src/btree.c#L2171>
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
    fn balance(&mut self, mut page: PageNumber, parents: &mut Vec<PageNumber>) -> io::Result<()> {
        let node = self.pager.get(page)?;

        // Started from the bottom now we're here :)
        let is_root = parents.is_empty();

        // The root is a special case because it can be fully empty.
        let is_underflow = node.is_empty() || !is_root && node.is_underflow();

        // Nothing to do, the node is balanced.
        if !node.is_overflow() && !is_underflow {
            return Ok(());
        }

        // Root underflow.
        if is_root && is_underflow {
            // Root doesn't have children, can't do anything.
            if node.is_leaf() {
                return Ok(());
            }

            let child_page = node.header().right_child;
            let necessary_space = self.pager.get(child_page)?.used_bytes();

            // Account for page zero having less space than the rest of pages.
            // If page zero can't consume its children then we'll leave it
            // empty temporarily until it can do so safely.
            if self.pager.get(page)?.free_space() < necessary_space {
                return Ok(());
            }

            let child = self.pager.get_mut(child_page)?;
            let grandchild = child.header().right_child;
            let cells = child.drain(..).collect::<Vec<_>>();

            self.pager.free_page(child_page)?;

            let root = self.pager.get_mut(page)?;
            cells.into_iter().for_each(|cell| root.push(cell));
            root.header_mut().right_child = grandchild;

            return Ok(());
        }

        // Root overflow.
        if is_root && node.is_overflow() {
            let new_page = self.pager.alloc_page::<Page>()?;

            let root = self.pager.get_mut(page)?;
            let grandchild = mem::replace(&mut root.header_mut().right_child, new_page);
            let cells = root.drain(..).collect::<Vec<_>>();

            let new_child = self.pager.get_mut(new_page)?;
            cells.into_iter().for_each(|cell| new_child.push(cell));
            new_child.header_mut().right_child = grandchild;

            parents.push(page);
            page = new_page;
        }

        // Internal/Leaf node Overflow/Underlow.
        let parent_page = parents.remove(parents.len() - 1);
        let mut siblings = self.load_siblings(page, parent_page)?;

        // Run into some nasty bug because of this and it was hard to spot so...
        debug_assert_eq!(
            HashSet::<PageNumber>::from_iter(siblings.iter().map(|s| s.page)).len(),
            siblings.len(),
            "siblings array contains duplicated pages: {siblings:?}"
        );

        let mut cells = VecDeque::new();
        let divider_idx = siblings[0].index;

        // Make copies of cells in order.
        for (i, sibling) in siblings.iter().enumerate() {
            cells.extend(self.pager.get_mut(sibling.page)?.drain(..));
            if i < siblings.len() - 1 {
                let mut divider = self.pager.get_mut(parent_page)?.remove(divider_idx);
                divider.header.left_child = self.pager.get(sibling.page)?.header().right_child;
                cells.push_back(divider);
            }
        }

        let usable_space = Page::usable_space(self.pager.page_size);

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
            .pager
            .get(siblings.last().unwrap().page)?
            .header()
            .right_child;

        // Allocate missing pages.
        while siblings.len() < number_of_cells_per_page.len() {
            let new_page = self.pager.alloc_page::<Page>()?;
            let parent_index = siblings.last().unwrap().index + 1;
            siblings.push(Sibling::new(new_page, parent_index));
        }

        // Free unused pages.
        while number_of_cells_per_page.len() < siblings.len() {
            self.pager.free_page(siblings.pop().unwrap().page)?;
        }

        // Put pages in ascending order to favor sequential IO where possible.
        BinaryHeap::from_iter(siblings.iter().map(|s| Reverse(s.page)))
            .iter()
            .enumerate()
            .for_each(|(i, Reverse(page))| siblings[i].page = *page);

        // Fix children pointers.
        let last_sibling = siblings[siblings.len() - 1];
        self.pager
            .get_mut(last_sibling.page)?
            .header_mut()
            .right_child = old_right_child;

        let parent_node = self.pager.get_mut(parent_page)?;
        if divider_idx == parent_node.len() {
            parent_node.header_mut().right_child = last_sibling.page;
        } else {
            parent_node.cell_mut(divider_idx).header.left_child = last_sibling.page;
        }

        // Begin redistribution.
        for (i, n) in number_of_cells_per_page.iter().enumerate() {
            let page = self.pager.get_mut(siblings[i].page)?;
            for _ in 0..*n {
                page.push(cells.pop_front().unwrap());
            }

            if i < siblings.len() - 1 {
                let mut divider = cells.pop_front().unwrap();
                page.header_mut().right_child = divider.header.left_child;
                divider.header.left_child = siblings[i].page;
                self.pager
                    .get_mut(parent_page)?
                    .insert(siblings[i].index, divider);
            }
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

        let parent = self.pager.get(parent_page)?;

        // TODO: Store this somewhere somehow, probably in the "parents" Vec that
        // we keep passing around. It's not a big deal anyway except for large
        // pages with many cells.
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

    /// Allocates a cell that can fit the entire given `payload`.
    ///
    /// Overflow pages are used if necessary. See [`OverflowPage`] for details.
    fn alloc_cell(&mut self, payload: Vec<u8>) -> io::Result<Box<Cell>> {
        let max_payload_size =
            Page::ideal_max_payload_size(self.pager.page_size, self.minimum_keys) as usize;

        // No overflow needed.
        if payload.len() <= max_payload_size {
            return Ok(Cell::new(payload));
        }

        // Store payload in chunks, link overflow pages and return the cell.
        let first_cell_payload_size = max_payload_size - mem::size_of::<PageNumber>();
        let mut overflow_page_number = self.pager.alloc_page::<OverflowPage>()?;

        // TODO: We're making a copy of the vec and the Cell::new makes another one...
        let cell = Cell::new_overflow(
            Vec::from(&payload[..first_cell_payload_size]),
            overflow_page_number,
        );

        let mut stored_bytes = first_cell_payload_size;

        loop {
            let overflow_bytes = min(
                OverflowPage::usable_space(self.pager.page_size) as usize,
                payload[stored_bytes..].len(),
            );

            let overflow_page = self
                .pager
                .get_mut_as::<OverflowPage>(overflow_page_number)?;

            overflow_page.content_mut()[..overflow_bytes]
                .copy_from_slice(&payload[stored_bytes..stored_bytes + overflow_bytes]);

            overflow_page.header_mut().num_bytes = overflow_bytes as _;

            stored_bytes += overflow_bytes;

            if stored_bytes >= payload.len() {
                break;
            }

            let next_overflow_page = self.pager.alloc_page::<OverflowPage>()?;

            self.pager
                .get_mut_as::<OverflowPage>(overflow_page_number)?
                .header_mut()
                .next = next_overflow_page;

            overflow_page_number = next_overflow_page;
        }

        Ok(cell)
    }

    /// Frees the pages occupied by the given `cell`.
    ///
    /// Pretty much a no-op if the cell is not "overflow".
    fn free_cell(&mut self, cell: Box<Cell>) -> io::Result<()> {
        if !cell.header.is_overflow {
            return Ok(());
        }

        let mut overflow_page = cell.overflow_page();

        while overflow_page != 0 {
            let unused_page = self.pager.get_as::<OverflowPage>(overflow_page)?;
            let next = unused_page.header().next;
            self.pager.free_page(overflow_page)?;
            overflow_page = next;
        }

        Ok(())
    }

    #[cfg(debug_assertions)]
    fn read_into_mem(
        &mut self,
        root: PageNumber,
        buf: &mut Vec<(PageNumber, Page)>,
    ) -> io::Result<()> {
        for page in self.pager.get(root)?.iter_children().collect::<Vec<_>>() {
            self.read_into_mem(page, buf)?;
        }

        let node = self.pager.get(root)?.clone();
        buf.push((root, node));

        Ok(())
    }

    #[cfg(debug_assertions)]
    pub fn json(&mut self) -> io::Result<String> {
        let mut nodes = Vec::new();
        self.read_into_mem(self.root, &mut nodes)?;

        nodes.sort_by(|(page_num1, _), (page_num2, _)| page_num1.cmp(page_num2));

        let mut string = String::from('[');

        string.push_str(&self.node_json(nodes[0].0, &nodes[0].1)?);

        for (page_num, node) in &nodes[1..] {
            string.push(',');
            string.push_str(&self.node_json(*page_num, node)?);
        }

        string.push(']');

        Ok(string)
    }

    #[cfg(debug_assertions)]
    fn node_json(&mut self, number: PageNumber, page: &Page) -> io::Result<String> {
        let mut string = format!("{{\"page\":{},\"entries\":[", number);

        if !page.is_empty() {
            let key = &page.cell(0).content;
            string.push_str(&format!("{:?}", key));

            assert!(
                !page.is_overflow(),
                "page {number} is overflow and indexing overflow pages is not supported"
            );

            for i in 1..page.len() {
                string.push(',');
                string.push_str(&format!("{:?}", &page.cell(i).content));
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

/// Joins a split payload back into one contiguous memory region.
///
/// If the cell at the given slot is not "overflow" then this simply returns a
/// reference to its content.
pub(crate) fn reassemble_payload<I: Seek + Read + Write>(
    pager: &mut Pager<I>,
    page: PageNumber,
    slot: SlotId,
) -> io::Result<Payload> {
    let cell = pager.get(page)?.cell(slot);

    // TODO: Check if we can circumvent Rust borrowing rules to use just
    // `cell.content` here.
    if !cell.header.is_overflow {
        return Ok(Payload::PageRef(&pager.get(page)?.cell(slot).content));
    }

    let mut overflow_page = cell.overflow_page();

    let mut payload = Vec::from(&cell.content[..cell.content.len() - mem::size_of::<PageNumber>()]);

    while overflow_page != 0 {
        let page = pager.get_as::<OverflowPage>(overflow_page)?;
        payload.extend_from_slice(page.payload());
        overflow_page = page.header().next;
    }

    Ok(Payload::Reassembled(payload.into()))
}

/// BTree cursor.
#[derive(Debug)]
pub(crate) struct Cursor {
    /// The page that the cursor points at currently.
    page: PageNumber,
    /// Current slot in [`Self::page`].
    slot: SlotId,
    /// Stack of parents. See the documentation of [`Self::try_next`].
    descent: Vec<PageNumber>,
    /// Lazy initialization.
    init: bool,
    /// `true` if there are no more elements to return.
    done: bool,
}

impl Cursor {
    pub fn new(page: PageNumber, slot: SlotId) -> Self {
        Self {
            page,
            slot,
            descent: vec![],
            init: false,
            done: false,
        }
    }

    pub fn initialized(page: PageNumber, slot: SlotId, descent: Vec<PageNumber>) -> Self {
        Self {
            page,
            slot,
            descent,
            init: true,
            done: false,
        }
    }

    pub fn done() -> Self {
        Self {
            page: 0,
            slot: 0,
            descent: vec![],
            init: true,
            done: true,
        }
    }

    /// Moves the cursor to the leftmost node in the current subtree.
    ///
    /// ```text
    ///                          self.page starts here
    ///                                    |
    ///                                    V
    ///                               +--------+ PAGE 1
    ///                       +-------|   11   |-------+
    ///                      /        +--------+        \
    ///                     /                            \
    ///                +-------+ PAGE 7              +--------+ PAGE 8
    ///           +----|  4,8  |----+                |   14   |
    ///          /     +-------+     \               +--------+
    ///         /          |          \               /      \
    ///     +-------+  +-------+  +-------+     +-------+  +-------+
    ///     | 1,2,3 |  | 5,6,7 |  | 9,10  |     | 12,13 |  | 15,16 |
    ///     +-------+  +-------+  +-------+     +-------+  +-------+
    ///      PAGE 2      PAGE 3     PAGE 4        PAGE 5     PAGE 6
    ///         ^
    ///         |
    ///     self.page
    ///     ends here
    /// ```
    fn move_to_leftmost<I: Seek + Read + Write>(&mut self, pager: &mut Pager<I>) -> io::Result<()> {
        let mut node = pager.get(self.page)?;

        while !node.is_leaf() {
            self.descent.push(self.page);
            self.page = node.child(0);
            node = pager.get(self.page)?;
        }

        self.slot = 0;

        Ok(())
    }

    /// Disk BTree traversal algorithm.
    ///
    /// Traversing a disk BTree with an iterator that maintains its state (which
    /// means no recursion) without blowing up the memory by storing a billion
    /// page numbers in a queue while maximizing sequential IO is not easy at
    /// all.
    ///
    /// There are two main ways to traverse a tree (or graph), discussed below.
    ///
    /// # Breadth-First Search (BFS)
    ///
    /// Ideally, this is how BFS should look like:
    ///
    /// ```text
    ///                                     +--------+ PAGE 1
    /// DEPTH 0                     +-------|   11   |-------+
    ///                            /        +--------+        \
    ///                           /                            \
    ///                      +-------+ PAGE 7              +--------+ PAGE 8
    /// DEPTH 1         +----|  4,8  |----+                |   14   |
    ///                /     +-------+     \               +--------+
    ///               /          |          \               /      \
    ///           +-------+  +-------+  +-------+     +-------+  +-------+
    /// DEPTH 2   | 1,2,3 |  | 5,6,7 |  | 9,10  |     | 12,13 |  | 15,16 |
    ///           +-------+  +-------+  +-------+     +-------+  +-------+
    ///            PAGE 2      PAGE 3     PAGE 4        PAGE 5     PAGE 6
    ///
    /// Key Order:     [11, 4, 8, 14, 1, 2, 3, 5, 6, 7, 9, 10, 12, 13, 15, 16]
    /// Visited Pages: [1, 7, 8, 2, 3, 4, 5, 6]
    /// ```
    ///
    /// If it's not clear from the diagram, in words it would be: return all
    /// the keys from all the pages at depth 0, then move to depth 1 and return
    /// all the keys from all the pages at that level, and so on until we're
    /// done.
    ///
    /// Normal BFS requires a FIFO queue where we push the children page numbers
    /// of every page we visit and then pop the next page from the front of the
    /// queue. Now that's nice and all until you have to deal with disk BTrees
    /// that could potentially contain billions of nodes.
    ///
    /// By the time we reach the leaf nodes of a balanced BTree of order 4 and
    /// depth 15 we have to push 4^15 page numbers into the queue, more than 1
    /// billion 4-byte integers which add up to roughly 4 GiB of memory. As
    /// always, a database system can never assume that something "fits in
    /// memory". This algorithm has exponential space complexity, or more
    /// specifically O(b^d) where b is the branching factor (number of children
    /// per node, also known as order in our case) and d is the maximum depth of
    /// the tree.
    ///
    /// So we can't use normal BFS, but there are variants of BFS that don't
    /// require so much memory. We'll discuss them later as they are related to
    /// depth-first search.
    ///
    /// # Depth-First Search (DFS)
    ///
    /// Depth-first search traverses the tree vertically instead of
    /// horizontally.
    ///
    /// This is how a conventional non-recursive depth-first traversal would
    /// look like:
    ///
    /// ```text
    ///                               +--------+ PAGE 1
    ///                       +-------|   11   |-------+
    ///                      /        +--------+        \
    ///                     /                            \
    ///                +-------+ PAGE 7              +--------+ PAGE 8
    ///           +----|  4,8  |----+                |   14   |
    ///          /     +-------+     \               +--------+
    ///         /          |          \               /      \
    ///     +-------+  +-------+  +-------+     +-------+  +-------+
    ///     | 1,2,3 |  | 5,6,7 |  | 9,10  |     | 12,13 |  | 15,16 |
    ///     +-------+  +-------+  +-------+     +-------+  +-------+
    ///      PAGE 2      PAGE 3     PAGE 4        PAGE 5     PAGE 6
    ///
    /// Key Order:     [11, 14, 15, 16, 12, 13, 4, 8, 9, 10, 5, 6, 7, 1, 2, 3]
    /// Visited Pages: [1, 8, 6, 5, 7, 4, 3, 2]
    /// ```
    ///
    /// Again, in words: return all the keys in the current page, move to one
    /// of its children, return all the keys in that child, move to one of the
    /// child's children, return all the keys and repeat until the bottom is
    /// reached. Then take another branch of the tree and repeat the process.
    ///
    /// Depth-first doesn't use a FIFO queue, it uses a stack or LIFO queue
    /// instead where we push children page numbers and pop them off the stack.
    /// The advantage of DFS over BFS is that since it goes down vertically the
    /// memory usage is limited because the tree depth grows exponentially, the
    /// bigger the tree is the less likely it is to grow another level.
    ///
    /// Using the same example from before, a BTree of order 4 and depth 15
    /// would only require the stack to store 4 * 15 page numbers at once, since
    /// the space complexity is O(b * d). That's only a few bytes. The time
    /// complexity is the same as BFS, no change there.
    ///
    /// # Memory Efficient BFS
    ///
    /// At this point it's pretty clear that we have to go "depth" first because
    /// of memory constraints. The problem is dealing with sequential IO. Since
    /// we don't have a B+Tree where leaf nodes contain all the data and are
    /// linked together, we can't do 100% sequential IO.
    ///
    /// The LIFO depth-first approach will basically send us at the end of the
    /// file first, then it will do some reversed sequential IO, then send us at
    /// the middle of the file and do more reversed sequential IO, and so on.
    /// Doesn't seem ideal.
    ///
    /// The best we can do is chunks of sequential IO, because the way the BTree
    /// is build and balanced causes all the pages at the same depth level to be
    /// sequential, although reusing freed pages might mess that up a little. So
    /// we could do a breadth-first traversal that's a little smarter and
    /// doesn't store all the children in a queue but uses a stack instead where
    /// it maintains the parents of the current node so that it can backtrack
    /// when needed. Something like this:
    ///
    /// ```text
    ///                               +--------+ PAGE 1
    ///                       +-------|   11   |-------+
    ///                      /        +--------+        \
    ///                     /                            \
    ///                +-------+ PAGE 7              +--------+ PAGE 8
    ///           +----|  4,8  |----+                |   14   |
    ///          /     +-------+     \               +--------+
    ///         /          |          \               /      \
    ///     +-------+  +-------+  +-------+     +-------+  +-------+
    ///     | 1,2,3 |  | 5,6,7 |  | 9,10  |     | 12,13 |  | 15,16 |
    ///     +-------+  +-------+  +-------+     +-------+  +-------+
    ///      PAGE 2      PAGE 3     PAGE 4        PAGE 5     PAGE 6
    ///
    /// Key Order:     [11, 4, 8, 14, 1, 2, 3, 5, 6, 7, 9, 10, 12, 13, 15, 16]
    /// Visited Pages: [1, 7, 1, 8, 1, 7, 2, 7, 3, 7, 4, 7, 1, 8, 5, 8, 6, 14, 11]
    /// ```
    ///
    /// When the algorithm is done with one node it has to go back to the parent
    /// to grab the next sibling. If there's no next sibling, then go to the
    /// grandparent and back down through the uncle to find the cousin, and so
    /// on. When it's done with one entire depth level it has to move to the
    /// next one by going back to the root and following the leftmost pointers.
    /// That's why you can see it moving upwards so many times in the visited
    /// pages array. This [commit] implements such solution.
    ///
    /// [commit]: https://github.com/antoniosarosi/mkdb/blob/653a0628f4f4a91ac1ae0e1d6237811b5abdf449/src/storage/btree.rs#L1923-L2014
    ///
    /// The "moving upwards" part is not as bad as it seems, since we can pin
    /// the parent pages in the cache or store only their children numbers in a
    /// local cache. The problem is when we are located at the maximum depth
    /// level and we have to move from a leaf of one subtree to the leaf of
    /// another subtree. For example, going from page 4 to page 5 in the figure
    /// above requires moving both upwards towards the root and then downwards
    /// through another branch towards the leaf node. This process has to be
    /// repeated for every branch of the root node.
    ///
    /// # In-Order Depth-First Search
    ///
    /// Regardless, there's another similar solution that visits less pages
    /// overall. We can basically do a sophisticated in-order depth first
    /// traversal that works with BTrees:
    ///
    /// ```text
    ///                               +--------+ PAGE 1
    ///                       +-------|   11   |-------+
    ///                      /        +--------+        \
    ///                     /                            \
    ///                +-------+ PAGE 7              +--------+ PAGE 8
    ///           +----|  4,8  |----+                |   14   |
    ///          /     +-------+     \               +--------+
    ///         /          |          \               /      \
    ///     +-------+  +-------+  +-------+     +-------+  +-------+
    ///     | 1,2,3 |  | 5,6,7 |  | 9,10  |     | 12,13 |  | 15,16 |
    ///     +-------+  +-------+  +-------+     +-------+  +-------+
    ///      PAGE 2      PAGE 3     PAGE 4        PAGE 5     PAGE 6
    ///
    /// Key Order:     [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    /// Visited Pages: [1, 7, 2, 7, 3, 7, 4, 7, 1, 8, 5, 8, 6, 8, 1]
    /// ```
    ///
    /// It uses the same backtracking technique with the stack that we mentioned
    /// before but it takes advantage of the fact that it has to move upwards
    /// by returning keys when it does so. That in turn reduces the number of
    /// times we have to go back upwards, because once we are done with a
    /// subtree we'll never visit any of its pages again.
    ///
    /// Of course, just like [`BTree::balance`], this algorithm is inspired by
    /// SQLite 2. Take a look at the [original].
    ///
    /// [original]: https://github.com/antoniosarosi/sqlite2-btree-visualizer/blob/master/src/btree.c#L1630-L1677
    pub fn try_next<I: Seek + Read + Write>(
        &mut self,
        pager: &mut Pager<I>,
    ) -> io::Result<Option<(PageNumber, SlotId)>> {
        // Wish everything was as easy as this.
        if self.done {
            return Ok(None);
        }

        // This only runs on the first call to `try_next`. Lazy initialization
        // for the iterator.
        if !self.init {
            self.move_to_leftmost(pager)?;
            self.init = true;
        }

        let node = pager.get(self.page)?;

        // The only page that can be empty is the root. The BTree will not allow
        // the rest of pages to stay empty.
        if node.is_empty() && node.is_leaf() {
            self.done = true;
            return Ok(None);
        }

        // We return the "current" position and prepare the next one on every call.
        let position = Ok(Some((self.page, self.slot)));

        // We are currently returning keys from a leaf node and we're not done
        // yet, so simply move to the next key (or cell in this case).
        if node.is_leaf() && self.slot + 1 < node.len() {
            self.slot += 1;
            return position;
        }

        // The last position we returned was pointing at an internal node and
        // the node has more children. Move to the bottom of the next subtree to
        // maintain order.
        if !node.is_leaf() && self.slot < node.len() {
            self.descent.push(self.page);
            self.page = node.child(self.slot + 1);
            self.move_to_leftmost(pager)?;

            return position;
        }

        // Now we know for sure we have to move upwards because we are done with
        // the current subtree. We'll go back the same path we came from until
        // we find the next branch that we have to take.
        let mut found_branch = false;

        while !self.descent.is_empty() && !found_branch {
            let parent_page = self.descent.pop().unwrap();
            let parent = pager.get(parent_page)?;
            // TODO: We can get rid of this O(n) by storing the index in the stack.
            let index = parent.iter_children().position(|c| c == self.page).unwrap() as u16;
            self.page = parent_page;

            if index < parent.len() {
                self.slot = index;
                found_branch = true;
            }
        }

        // We went all the way back to the root and didn't find any branch. We
        // are done.
        if self.descent.is_empty() && !found_branch {
            self.done = true;
        }

        position
    }

    /// Returns the next position in the BTree.
    ///
    /// See [`Self::try_next`] for the actual code. This one just flips
    /// [`Result<Option>`] to make it [`Option<Result>`].
    pub fn next<I: Seek + Read + Write>(
        &mut self,
        pager: &mut Pager<I>,
    ) -> Option<io::Result<(PageNumber, SlotId)>> {
        self.try_next(pager).transpose()
    }
}

/// BTree testing framework.
///
/// Most of the tests use fixed size 64 bit keys to easily test whether the
/// BTree balancing algorithm does the correct thing in each situation. To
/// easily compare the state of the [`BTree`] structure with something human
/// readable we use [`Node`] instances, which allow us to define a tree as if we
/// used some JSON-like syntax. There's also a [`Builder`] struct that can be
/// used to insert many 64 bit keys at once into the tree and also tune
/// parameters such as [`BTree::balance_siblings_per_side`]. See the tests for
/// more details and examples. Remember that `order` means the maximum number of
/// children in a BTree that stores fixed size keys.
#[cfg(test)]
mod tests {
    use std::{alloc::Layout, io, mem};

    use super::{BTree, Cursor, FixedSizeMemCmp, DEFAULT_BALANCE_SIBLINGS_PER_SIDE};
    use crate::{
        paging::{
            cache::{Cache, DEFAULT_MAX_CACHE_SIZE, MIN_CACHE_SIZE},
            io::MemBuf,
            pager::{PageNumber, Pager},
        },
        storage::{
            btree::Payload,
            page::{Cell, Page, CELL_HEADER_SIZE, MEM_ALIGNMENT, PAGE_HEADER_SIZE, SLOT_SIZE},
        },
    };

    /// Fixed size key used for most of the tests.
    type Key = u64;

    /// Serialize into big endian and use [`FixedSizeMemCmp`] for comparisons.
    fn serialize_key(key: Key) -> [u8; mem::size_of::<Key>()] {
        key.to_be_bytes()
    }

    /// Same as [`serialize_key`] but allows the caller to specify the length.
    fn serialize_key_of_size(key: Key, len: usize) -> Vec<u8> {
        let mut buf = Vec::from(key.to_be_bytes());
        buf.append(&mut vec![0; len - mem::size_of::<Key>()]);

        buf
    }

    fn deserialize_key(buf: &[u8]) -> Key {
        Key::from_be_bytes(buf[..mem::size_of::<Key>()].try_into().unwrap())
    }

    /// Allows us to build an entire tree manually and then compare it to an
    /// actual [`BTree`] structure. See tests below for examples.
    #[derive(Debug, PartialEq)]
    struct Node {
        keys: Vec<Key>,
        children: Vec<Self>,
    }

    impl Node {
        /// Leaf nodes have no children.
        ///
        /// This method saves us some unecessary typing and makes the tree
        /// structure more readable.
        fn leaf(keys: impl IntoIterator<Item = Key>) -> Self {
            Self {
                keys: keys.into_iter().collect(),
                children: Vec::new(),
            }
        }
    }

    /// Test builder for [`BTree<'c, MemBuf, FixedSizeCmp>`].
    struct Builder {
        keys: Vec<Key>,
        order: usize,
        page_size: Option<usize>,
        root_at_zero: bool,
        balance_siblings_per_side: usize,
        cache_size: usize,
        minimum_keys: usize,
    }

    impl Default for Builder {
        fn default() -> Self {
            Builder {
                keys: vec![],
                order: 4,
                page_size: None,
                root_at_zero: false,
                balance_siblings_per_side: DEFAULT_BALANCE_SIBLINGS_PER_SIDE,
                cache_size: DEFAULT_MAX_CACHE_SIZE,
                // We use small page sizes for tests which don't allow many keys
                minimum_keys: 1,
            }
        }
    }

    impl Builder {
        fn keys(mut self, keys: impl IntoIterator<Item = Key>) -> Self {
            self.keys = keys.into_iter().collect();
            self
        }

        fn order(mut self, order: usize) -> Self {
            self.order = order;
            self
        }

        fn page_size(mut self, page_size: usize) -> Self {
            self.page_size = Some(page_size);
            self
        }

        fn balance_siblings_per_side(mut self, balance_siblings_per_side: usize) -> Self {
            self.balance_siblings_per_side = balance_siblings_per_side;
            self
        }

        fn root_at_zero(mut self, root_at_zero: bool) -> Self {
            self.root_at_zero = root_at_zero;
            self
        }

        fn cache_size(mut self, cache_size: usize) -> Self {
            self.cache_size = cache_size;
            self
        }

        fn try_build<'c>(self) -> io::Result<BTree<'c, MemBuf, FixedSizeMemCmp>> {
            let page_size = self
                .page_size
                .unwrap_or(optimal_page_size_for_order(self.order));
            let buf = io::Cursor::new(Vec::new());

            let mut pager = Pager::with_cache(
                buf,
                page_size,
                page_size,
                Cache::builder()
                    .page_size(page_size)
                    .max_size(self.cache_size)
                    .build(),
            );
            pager.init()?;

            let root = if self.root_at_zero {
                0
            } else {
                pager.alloc_page::<Page>()?
            };

            // TODO: Do something about leaking, we shouldn't need that here.
            let mut btree = BTree {
                pager: Box::leak(Box::new(pager)),
                root,
                balance_siblings_per_side: self.balance_siblings_per_side,
                comparator: FixedSizeMemCmp::for_type::<Key>(),
                minimum_keys: self.minimum_keys,
            };

            btree.try_insert_all_keys(self.keys)?;

            Ok(btree)
        }
    }

    impl<'p> BTree<'p, MemBuf, FixedSizeMemCmp> {
        fn into_test_nodes(&mut self, root: PageNumber) -> io::Result<Node> {
            let page = self.pager.get(root)?;

            let mut node = Node {
                keys: (0..page.len())
                    .map(|i| deserialize_key(&page.cell(i).content))
                    .collect(),
                children: vec![],
            };

            let children = page.iter_children().collect::<Vec<_>>();

            for page in children {
                node.children.push(self.into_test_nodes(page)?);
            }

            Ok(node)
        }

        fn builder() -> Builder {
            Builder::default()
        }

        fn insert_key(&mut self, key: Key) -> io::Result<()> {
            self.insert(Vec::from(serialize_key(key)))
        }

        fn remove_key(&mut self, key: Key) -> io::Result<Option<Box<Cell>>> {
            self.remove(&serialize_key(key))
        }

        fn try_insert_all_keys(&mut self, keys: impl IntoIterator<Item = Key>) -> io::Result<()> {
            for key in keys {
                self.insert(Vec::from(serialize_key(key)))?;
            }

            Ok(())
        }

        fn try_remove_all_keys(
            &mut self,
            keys: impl IntoIterator<Item = Key>,
        ) -> io::Result<Vec<Option<Box<Cell>>>> {
            keys.into_iter()
                .map(|key| self.remove(&serialize_key(key)))
                .collect()
        }
    }

    fn align_upwards(size: usize, alignment: usize) -> usize {
        Layout::from_size_align(size, alignment)
            .unwrap()
            .pad_to_align()
            .size()
    }

    /// Computes the page size needed to store `order - 1` keys of type [`Key`]
    /// in one page.
    fn optimal_page_size_for_order(order: usize) -> usize {
        optimal_page_size_for_max_payload(mem::size_of::<Key>(), order - 1)
    }

    /// Computes the page size needed to store at least `min_keys` each carrying
    /// `max` amount of payload.
    fn optimal_page_size_for_max_payload(max: usize, min_keys: usize) -> usize {
        let cell_storage_size =
            CELL_HEADER_SIZE + SLOT_SIZE + align_upwards(max, MEM_ALIGNMENT) as u16;

        let total_size = PAGE_HEADER_SIZE + cell_storage_size * min_keys as u16;

        align_upwards(total_size as usize, MEM_ALIGNMENT)
    }

    impl<'c> TryFrom<BTree<'c, MemBuf, FixedSizeMemCmp>> for Node {
        type Error = io::Error;

        fn try_from(mut btree: BTree<MemBuf, FixedSizeMemCmp>) -> Result<Self, Self::Error> {
            btree.into_test_nodes(btree.root)
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

        assert_eq!(Node::try_from(btree)?, Node {
            keys: vec![3],
            children: vec![Node::leaf([1, 2]), Node::leaf([4])]
        });

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

        assert_eq!(Node::try_from(btree)?, Node {
            keys: vec![4],
            children: vec![Node::leaf([1, 2, 3]), Node::leaf([5, 6, 7])]
        });

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

        assert_eq!(Node::try_from(btree)?, Node {
            keys: vec![3, 6],
            children: vec![Node::leaf([1, 2]), Node::leaf([4, 5]), Node::leaf([7, 8])]
        });

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

        assert_eq!(Node::try_from(btree)?, Node {
            keys: vec![4, 8, 12],
            children: vec![
                Node::leaf([1, 2, 3]),
                Node::leaf([5, 6, 7]),
                Node::leaf([9, 10, 11]),
                Node::leaf([13, 14, 15]),
            ]
        });

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

        assert_eq!(Node::try_from(btree)?, Node {
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
        });

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

        assert_eq!(Node::try_from(btree)?, Node {
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
        });

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

        assert_eq!(Node::try_from(btree)?, Node {
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
        });

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

        assert_eq!(Node::try_from(btree)?, Node {
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
        });

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

        assert_eq!(Node::try_from(btree)?, Node {
            keys: vec![4, 8, 12],
            children: vec![
                Node::leaf([1, 2, 3]),
                Node::leaf([5, 6, 7]),
                Node::leaf([9, 10, 11]),
                Node::leaf([14, 15]),
            ]
        });

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

        assert_eq!(Node::try_from(btree)?, Node {
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
        });

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

        assert_eq!(Node::try_from(btree)?, Node {
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
        });

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

        assert_eq!(Node::try_from(btree)?, Node {
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
        });

        Ok(())
    }

    /// When a leaf node falls under 50% capacity it should not be merged with
    /// one of its siblings if the siblings can lend keys without underflowing.
    ///
    /// ```text
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

        assert_eq!(Node::try_from(btree)?, Node {
            keys: vec![4, 8, 11],
            children: vec![
                Node::leaf([1, 2, 3]),
                Node::leaf([5, 6, 7]),
                Node::leaf([9, 10]),
                Node::leaf([12, 13]),
            ]
        });

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

        assert_eq!(Node::try_from(btree)?, Node {
            keys: vec![4, 8],
            children: vec![
                Node::leaf([1, 2, 3]),
                Node::leaf([5, 6, 7]),
                Node::leaf([9, 10, 11]),
            ]
        });

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

        assert_eq!(Node::try_from(btree)?, Node {
            keys: vec![4, 8, 11],
            children: vec![
                Node::leaf([1, 2, 3]),
                Node::leaf([5, 6, 7]),
                Node::leaf([9, 10]),
                Node::leaf([12, 13, 14]),
            ]
        });

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

        assert_eq!(Node::try_from(btree)?, Node {
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
        });

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

        assert_eq!(Node::try_from(btree)?, Node {
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
        });

        Ok(())
    }

    /// Most tests use `order = 4` for simplicty. This one uses `order = 6` to
    /// check if everything still works. This is what we're going to build:
    ///
    /// ```text
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

        assert_eq!(Node::try_from(btree)?, Node {
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
        });

        Ok(())
    }

    /// Delete on `order = 6`.
    ///
    /// ```text
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

        assert_eq!(Node::try_from(btree)?, Node {
            keys: vec![6, 12, 18, 24, 30],
            children: vec![
                Node::leaf([1, 2, 3, 4, 5]),
                Node::leaf([7, 8, 9, 10, 11]),
                Node::leaf([13, 14, 15, 16, 17]),
                Node::leaf([19, 20, 21, 22, 23]),
                Node::leaf([25, 26, 27, 28, 29]),
                Node::leaf([31, 32, 33]),
            ]
        });

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

        assert_eq!(Node::try_from(btree)?, Node {
            keys: vec![5, 9, 13],
            children: vec![
                Node::leaf([1, 2, 4]),
                Node::leaf([6, 7, 8]),
                Node::leaf([10, 11, 12]),
                Node::leaf([14, 15, 16]),
            ]
        });

        Ok(())
    }

    #[test]
    fn variable_length_data() -> io::Result<()> {
        // Payload size of each cell in bytes.
        let payload_sizes = [64, 32, 48, 16, 8, 8];

        // TODO: This does the same thing as
        // [`super::page::tests::optimal_page_size_to_fit`]. Make a module for
        // test utilities or something.
        let page_size = {
            let space_needed = PAGE_HEADER_SIZE
                + payload_sizes
                    .iter()
                    .map(|size| size + CELL_HEADER_SIZE + SLOT_SIZE)
                    .sum::<u16>();

            align_upwards(space_needed as usize, MEM_ALIGNMENT)
        };

        let mut btree = BTree::builder().page_size(page_size).try_build()?;

        // This should fill the root page.
        for (i, size) in payload_sizes.iter().enumerate() {
            btree.insert(serialize_key_of_size(i as Key + 1, *size as usize))?;
        }

        // Now adding another 48 byte key should cause it to split.
        btree.insert(serialize_key_of_size(payload_sizes.len() as Key, 48))?;

        // Considering the size of each cell, this is how the BTree should
        // end up looking:
        //
        //         48 bytes
        //          +---+
        //          | 3 |
        //          +---+
        //         /     \
        // +------+       +---------+
        // | 1, 2 |       | 4, 5, 6 |
        // +------+       +---------+
        //  64b 32b        16b 8b 48b
        //
        // 96 bytes         72 bytes
        //  total            total
        assert_eq!(Node::try_from(btree)?, Node {
            keys: vec![3],
            children: vec![Node::leaf([1, 2]), Node::leaf([4, 5, 6])]
        });

        Ok(())
    }

    #[test]
    fn reassemble_overflow_cell() -> io::Result<()> {
        let page_size = 256;
        let mut large_key = Vec::with_capacity(1024);

        for _ in 0..4 {
            large_key.extend(&(0..=255).collect::<Vec<u8>>());
        }

        let mut btree = BTree::builder().page_size(page_size).try_build()?;

        btree.insert(large_key.clone())?;

        assert_eq!(
            btree.get(&large_key)?,
            Some(Payload::Reassembled(large_key.into_boxed_slice()))
        );

        Ok(())
    }

    /// Some obscure edge case when deleting. See the source of
    /// [`BTree::remove`].
    ///
    /// ```text
    ///                                      Remove 11 and 13 then double the size of 6, 18, 10 and 14
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
    ///                                      Removing 12 should cause the internal node to rebalance
    ///                                                              |
    ///                                                              V
    ///                                                           +----+
    ///                                  +------------------------| 24 |---------------------------+
    ///                                 /                         +----+                            \
    ///                                /                                                             \
    ///                       +---------------+                                                   +-------+
    ///          +------------| 6-XX,12,18-XX |-------------------+                         +-----| 29,33 |-----+
    ///         /             +---------------+                    \                       /      +-------+      \
    ///        /                   /      \                         \                     /           |           \
    /// +-----------+  +-------------+  +----------------+  +----------------+  +-------------+  +----------+  +----------+
    /// | 1,2,3,4,5 |  | 7,8,9,10-XX |  | 14-XX,15,16,17 |  | 19,20,21,22,23 |  | 25,26,27,28 |  | 30,31,32 |  | 34,35,36 |
    /// +-----------+  +-------------+  +----------------+  +----------------+  +-------------+  +----------+  +----------+
    ///
    ///                                            Final Result
    ///                                                  |
    ///                                                  V
    ///                                              +-------+
    ///                        +---------------------| 18-XX |------------------------+
    ///                       /                      +-------+                         \
    ///                      /                                                          \
    ///              +------------+                                                 +----------+
    ///          +---| 6-XX,10-XX |---+                          +------------------| 24,29,33 |------------+
    ///         /    +------------+    \                        /                   +----------+             \
    ///        /           |            \                      /                      /      \                \
    /// +-----------+  +-------+  +----------------+  +----------------+  +-------------+  +----------+  +----------+
    /// | 1,2,3,4,5 |  | 7,8,9 |  | 14-XX,15,16,17 |  | 19,20,21,22,23 |  | 25,26,27,28 |  | 30,31,32 |  | 34,35,36 |
    /// +-----------+  +-------+  +----------------+  +----------------+  +-------------+  +----------+  +----------+
    /// ```
    #[test]
    fn delete_leaving_leaf_balanced_and_internal_unbalanced() -> io::Result<()> {
        // We need to make the cells big enough to make sure alignment and
        // header sizes don't mess up what we're trying to do here.
        let payload_size = 32;

        let mut btree = BTree::builder()
            .page_size(optimal_page_size_for_max_payload(payload_size, 5))
            .try_build()?;

        for key in 1..=36 {
            btree.insert(serialize_key_of_size(key, payload_size))?;
        }

        btree.try_remove_all_keys([11, 13])?;

        for key in [6, 18, 10, 14] {
            btree.insert(serialize_key_of_size(key, payload_size * 2))?;
        }

        btree.remove_key(12)?;

        assert_eq!(Node::try_from(btree)?, Node {
            keys: vec![18],
            children: vec![
                Node {
                    keys: vec![6, 10],
                    children: vec![
                        Node::leaf([1, 2, 3, 4, 5]),
                        Node::leaf([7, 8, 9]),
                        Node::leaf([14, 15, 16, 17]),
                    ]
                },
                Node {
                    keys: vec![24, 29, 33],
                    children: vec![
                        Node::leaf([19, 20, 21, 22, 23]),
                        Node::leaf([25, 26, 27, 28]),
                        Node::leaf([30, 31, 32]),
                        Node::leaf([34, 35, 36]),
                    ]
                },
            ]
        });

        Ok(())
    }

    /// Page zero has less space than the rest of pages and does some weird
    /// things but it still keeps the BTree in a correct state. Sometimes the
    /// root stays empty because it becomes "overflow", then it is moved into
    /// a new page of greater size that is not "overflow" anymore, so there's
    /// no split, which means there's no divider key.
    ///
    /// But the BTree still works because if there's no divider then the
    /// binary search fails with index 0, and the child 0 is whichever comes
    /// beneath the root.
    ///
    /// When the root child finally splits the root starts collecting dividers
    /// making the tree look normal, but this cycle can repeat itself.
    ///
    /// This test is the same as [`basic_insertion`] but with the root set at
    /// page 0. In this case, every node can fit 3 keys except the root, which
    /// can only fit 2 keys.
    ///
    /// Due to the "empty root not empty root cycle", the tree ends up like
    /// this:
    ///
    /// ```text
    ///                 +--------+
    ///                 |        | Empty Root
    ///                 +--------+
    ///                     |
    ///                 +--------+
    ///         +-------| 4,8,12 |--------+
    ///       /         +--------+         \
    ///     /           /        \          \
    /// +-------+  +-------+  +---------+  +----------+
    /// | 1,2,3 |  | 5,6,7 |  | 9,10,11 |  | 13,14,15 |
    /// +-------+  +-------+  +---------+  +----------+
    /// ```
    #[test]
    fn page_zero_root_basic_insertion() -> io::Result<()> {
        let btree = BTree::builder()
            .root_at_zero(true)
            .keys(1..=15)
            .try_build()?;

        assert_eq!(Node::try_from(btree)?, Node {
            keys: vec![],
            children: vec![Node {
                keys: vec![4, 8, 12],
                children: vec![
                    Node::leaf([1, 2, 3]),
                    Node::leaf([5, 6, 7]),
                    Node::leaf([9, 10, 11]),
                    Node::leaf([13, 14, 15]),
                ]
            }]
        });

        Ok(())
    }

    /// Right after [`page_zero_root_basic_insertion`] this is what should
    /// happen:
    ///
    /// ```text
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
    ///
    /// As you can see the tree behaves normally again. The test case is the
    /// same as [`propagate_split_to_root`].
    #[test]
    fn page_zero_insert_divider_in_empty_root() -> io::Result<()> {
        let btree = BTree::builder()
            .root_at_zero(true)
            .keys(1..=16)
            .try_build()?;

        assert_eq!(Node::try_from(btree)?, Node {
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
        },);

        Ok(())
    }

    #[test]
    fn page_zero_merge_all_nodes() -> io::Result<()> {
        let mut btree = BTree::builder()
            .root_at_zero(true)
            .keys(1..=16)
            .try_build()?;

        btree.try_remove_all_keys(2..=16)?;

        assert_eq!(Node::try_from(btree)?, Node::leaf([1]));

        Ok(())
    }

    fn assert_cursor_traversal_matches(
        btree: &mut BTree<'_, MemBuf, FixedSizeMemCmp>,
        keys: impl Iterator<Item = Key>,
    ) -> io::Result<()> {
        let mut cursor = Cursor::new(btree.root, 0);

        for expected_key in keys {
            let (page, slot) = cursor.next(btree.pager).unwrap_or_else(|| {
                panic!("cursor should return the position of key {expected_key} but returns None")
            })?;

            let key = deserialize_key(&btree.pager.get(page)?.cell(slot).content);

            assert_eq!(
                key,
                expected_key,
                "cursor at position ({page}, {slot}) should return {expected_key} but returns {key}"
            );
        }

        assert!(cursor.next(btree.pager).is_none());

        Ok(())
    }

    /// Traverse this tree:
    ///
    /// ```text
    ///                                                 +----+ P1
    ///                            +--------------------| 18 |---------------------+
    ///                           /                     +----+                      \
    ///                        +---+ P16                                          +----+ P17
    ///              +---------| 9 |---------+                             +------| 24 |-------+
    ///             /          +---+          \                           /       +----+        \
    ///         +-----+                    +-------+                   +----+                +-------+
    ///         | 3,6 | P6                 | 12,15 | P7                | 21 | P11            | 27,29 | P15
    ///         +-----+                    +-------+                   +----+                +-------+
    ///        /   |   \                  /    |    \                   /  \                 /   |   \
    /// +-----+ +-----+ +-----+  +-------+ +-------+ +-------+  +-------+ +-------+  +-------+ +----+ +----+
    /// | 1,2 | | 4,5 | | 7,8 |  | 10,11 | | 13,14 | | 16,17 |  | 19,20 | | 22,23 |  | 25,26 | | 28 | | 30 |
    /// +-----+ +-----+ +-----+  +-------+ +-------+ +-------+  +-------+ +-------+  +-------+ +----+ +----+
    ///   P2      P3      P4       P5          P8        P9        P10       P12        P13      P14   P18
    /// ```
    #[test]
    fn basic_cursor() -> io::Result<()> {
        let keys = 1..=30;
        let mut btree = BTree::builder().order(3).keys(keys.clone()).try_build()?;

        assert_cursor_traversal_matches(&mut btree, keys)
    }

    /// Not gonna draw this one but you get the idea.
    #[cfg(not(miri))]
    #[test]
    fn cursor_with_more_depth_and_keys() -> io::Result<()> {
        let keys = 1..=400;
        let mut btree = BTree::builder().order(6).keys(keys.clone()).try_build()?;

        assert_cursor_traversal_matches(&mut btree, keys)
    }

    /// Should work with this tree.
    ///
    /// ```text
    ///                 +--------+
    ///                 |        | Empty Root
    ///                 +--------+
    ///                     |
    ///                 +--------+
    ///         +-------| 4,8,12 |--------+
    ///       /         +--------+         \
    ///     /           /        \          \
    /// +-------+  +-------+  +---------+  +----------+
    /// | 1,2,3 |  | 5,6,7 |  | 9,10,11 |  | 13,14,15 |
    /// +-------+  +-------+  +---------+  +----------+
    /// ```
    #[test]
    fn cursor_on_empty_root_with_children() -> io::Result<()> {
        let keys = 1..=15;

        let mut btree = BTree::builder()
            .root_at_zero(true)
            .keys(keys.clone())
            .try_build()?;

        assert_cursor_traversal_matches(&mut btree, keys)
    }

    #[test]
    fn cursor_on_empty_root_with_no_children() -> io::Result<()> {
        let btree = BTree::builder().try_build()?;
        let mut cursor = Cursor::new(btree.root, 0);

        assert!(cursor.next(btree.pager).is_none());

        Ok(())
    }

    /// Make the cache small and put some pressure on it to see if everything
    /// still works in practice.
    #[test]
    fn cache_pressure() -> io::Result<()> {
        let mut btree = BTree::builder().cache_size(MIN_CACHE_SIZE).try_build()?;

        btree.try_insert_all_keys(1..=31)?;
        btree.try_remove_all_keys(2..=31)?;

        assert_eq!(Node::try_from(btree)?, Node::leaf([1]));

        Ok(())
    }

    #[test]
    fn cache_pressure_with_page_zero() -> io::Result<()> {
        let mut btree = BTree::builder()
            .root_at_zero(true)
            .cache_size(MIN_CACHE_SIZE)
            .try_build()?;

        btree.try_insert_all_keys(1..=31)?;
        btree.try_remove_all_keys(2..=31)?;

        assert_eq!(Node::try_from(btree)?, Node::leaf([1]));

        Ok(())
    }

    /// Same as [`basic_insertion`] but inserting in reverse order.
    #[test]
    fn reverse_order_basic_insertion() -> io::Result<()> {
        let btree = BTree::builder().keys(1..=15).try_build()?;

        assert_eq!(Node::try_from(btree)?, Node {
            keys: vec![4, 8, 12],
            children: vec![
                Node::leaf([1, 2, 3]),
                Node::leaf([5, 6, 7]),
                Node::leaf([9, 10, 11]),
                Node::leaf([13, 14, 15]),
            ]
        });

        Ok(())
    }

    /// Same as [`propagate_split_to_root`] but reversing the order.
    #[test]
    fn reverse_order_insertion() -> io::Result<()> {
        let btree = BTree::builder().keys((1..=16).rev()).try_build()?;

        assert_eq!(Node::try_from(btree)?, Node {
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
        });

        Ok(())
    }

    /// Same as [`basic_insertion`] but inserting more "randomly".
    #[test]
    fn random_order_basic_insertion() -> io::Result<()> {
        let mut btree = BTree::builder().try_build()?;

        btree.try_insert_all_keys([4, 2, 1, 3, 5])?;
        btree.try_insert_all_keys([11, 15, 13, 12, 14])?;
        btree.try_insert_all_keys([8, 6, 9, 7, 10])?;

        assert_eq!(Node::try_from(btree)?, Node {
            keys: vec![4, 8, 12],
            children: vec![
                Node::leaf([1, 2, 3]),
                Node::leaf([5, 6, 7]),
                Node::leaf([9, 10, 11]),
                Node::leaf([13, 14, 15]),
            ]
        });

        Ok(())
    }
}
