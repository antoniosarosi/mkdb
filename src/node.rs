use std::cmp::Ordering;

/// Key-Value pairs stored in [`Node::entries`].
#[derive(Eq, Copy, Clone, Debug)]
pub(crate) struct Entry {
    pub key: u32,
    pub value: u32,
}

impl Entry {
    pub fn new(key: u32, value: u32) -> Self {
        Self { key, value }
    }
}

impl PartialEq for Entry {
    fn eq(&self, other: &Self) -> bool {
        self.key.eq(&other.key)
    }
}

impl PartialEq<u32> for Entry {
    fn eq(&self, other: &u32) -> bool {
        self.key == *other
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

/// We know the type of a node based on whether it has children and whether it
/// has a parent.
#[derive(PartialEq)]
pub(crate) enum NodeKind {
    /// The root node may or may not have children but it never has a parent.
    Root,
    /// Internal nodes have both children and a parent.
    Internal,
    /// Leaf nodes have no children and have a parent.
    Leaf,
}

/// Each of the nodes that compose the [`crate::btree::BTree`] structure. One
/// [`Node`] should always map to a single page in the disk.
#[derive(Debug, PartialEq, Clone)]
pub(crate) struct Node {
    /// The number of the page where this node is stored in disk.
    pub page: u32,

    /// Key-Value pairs stored by this node.
    pub entries: Vec<Entry>,

    /// Children pointers. Each child has its own page number.
    pub children: Vec<u32>,

    /// The index of this node in the parent's children list.
    pub parent_index: usize,
}

impl Default for Node {
    fn default() -> Self {
        Self::new()
    }
}

impl Node {
    /// Creates an empty default [`Node`] with [`Self::page`] and
    /// [`Self::parent_index`] set to 0.
    pub fn new() -> Self {
        Self {
            page: 0,
            parent_index: 0,
            entries: Vec::new(),
            children: Vec::new(),
        }
    }

    /// Automatically sets the page number when creating the node.
    pub fn new_at(page: u32) -> Self {
        let mut node = Self::new();
        node.page = page;
        node
    }

    /// Sets the value of [`Self::parent_index`].
    pub fn with_parent_index(mut self, parent_index: usize) -> Self {
        self.parent_index = parent_index;
        self
    }

    /// Consumes all the entries and children in `other` and appends them to
    /// `self`.
    pub fn extend_by_draining(&mut self, other: &mut Node) {
        self.entries.extend(other.entries.drain(..));
        self.children.extend(other.children.drain(..));
    }

    pub fn kind(&self) -> NodeKind {
        if self.is_root() {
            NodeKind::Root
        } else if self.is_leaf() {
            NodeKind::Leaf
        } else {
            NodeKind::Internal
        }
    }

    /// Returns `true` if this node is the root node.
    pub fn is_root(&self) -> bool {
        self.page == 0
    }

    /// Returns `true` if this node is a leaf node. Note that the root node can
    /// also be a leaf node if it has no children.
    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }
}
