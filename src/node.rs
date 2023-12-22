use std::{cmp::Ordering, mem};

use crate::{
    cache::MemPage,
    pager::{Page, PageNumber},
};

/// Key-Value pairs stored in [`Node::entries`].
#[derive(Eq, Copy, Clone, Debug)]
pub(crate) struct Entry {
    pub key: u32,
    pub value: u32,
}

/// Each of the nodes that compose the [`crate::btree::BTree`] structure. One
/// [`Node`] should always map to a single page in the disk.
///
/// Page format:
///
/// ```text
/// +---------+--------+--------+--------+--------+     +--------+--------+
/// | EL | CL |   K1   |   V1   |   K2   |   V2   | ... |   C1   |   C2   | ...
/// +---------+--------+--------+--------+--------+     +--------+--------+
///   2    2      4        4        4        4              4        4
/// ```
///
/// - `KL`: Keys Length
/// - `CL`: Children Length
/// - `K`: Key
/// - `V`: Value
/// - `C`: Child Pointer
#[derive(Debug, PartialEq)]
pub(crate) struct Node {
    /// Disk page number of this node. Stored only in memory.
    pub page: PageNumber,

    /// Key-Value pairs stored by this node.
    pub entries: Vec<Entry>,

    /// Children pointers. Each child has its own page number.
    pub children: Vec<PageNumber>,
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

impl Default for Node {
    fn default() -> Self {
        Self::new()
    }
}

impl Node {
    /// Creates an empty default [`Node`].
    pub fn new() -> Self {
        Self {
            page: 0,
            entries: Vec::new(),
            children: Vec::new(),
        }
    }

    /// Automatically sets the page number when creating the node.
    pub fn new_at(page: PageNumber) -> Self {
        let mut node = Self::new();
        node.page = page;
        node
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

impl From<&Page> for Node {
    fn from(page: &Page) -> Self {
        let mut node = Node::new_at(page.number);

        let mut i = 4;

        for _ in 0..u16::from_le_bytes(page.content[..2].try_into().unwrap()) {
            let key = u32::from_le_bytes(page.content[i..i + 4].try_into().unwrap());
            let value = u32::from_le_bytes(page.content[i + 4..i + 8].try_into().unwrap());
            node.entries.push(Entry { key, value });
            i += 8;
        }

        for _ in 0..u16::from_le_bytes(page.content[2..4].try_into().unwrap()) {
            node.children.push(u32::from_le_bytes(
                page.content[i..i + 4].try_into().unwrap(),
            ));
            i += 4;
        }

        node
    }
}

impl From<Page> for Node {
    fn from(page: Page) -> Self {
        Node::from(&page)
    }
}

impl From<&Node> for Page {
    fn from(node: &Node) -> Self {
        let mut page = Page {
            number: node.page,
            content: Vec::new(),
        };

        page.content
            .extend(&(node.entries.len() as u16).to_le_bytes());
        page.content
            .extend(&(node.children.len() as u16).to_le_bytes());

        for entry in &node.entries {
            page.content.extend(&entry.key.to_le_bytes());
            page.content.extend(&entry.value.to_le_bytes());
        }

        for child in &node.children {
            page.content.extend(&child.to_le_bytes());
        }

        page
    }
}

impl From<Node> for Page {
    fn from(node: Node) -> Self {
        Page::from(&node)
    }
}

impl MemPage for Node {
    fn size_on_disk(&self) -> usize {
        2 * mem::size_of::<u16>()
            + mem::size_of::<Entry>() * self.entries.len()
            + mem::size_of::<PageNumber>() * self.children.len()
    }

    fn disk_page_number(&self) -> PageNumber {
        self.page
    }
}
