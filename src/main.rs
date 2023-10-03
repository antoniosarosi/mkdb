#[derive(Debug)]
struct Node<T> {
    keys: Vec<T>,
    children: Vec<Box<Node<T>>>,
}

impl<T> Node<T> {
    pub fn new() -> Self {
        Self {
            keys: Vec::new(),
            children: Vec::new(),
        }
    }

    pub fn with_keys(keys: Vec<T>) -> Self {
        Self {
            keys,
            children: Vec::new(),
        }
    }

    pub fn with_children(children: Vec<Box<Node<T>>>) -> Self {
        Self {
            keys: Vec::new(),
            children,
        }
    }
}

/// `degree`: Minimum number of children.
#[derive(Debug)]
struct BTree<T> {
    root: Option<Box<Node<T>>>,
    degree: usize,
}

impl<T: Ord> BTree<T> {
    pub fn new(degree: usize) -> Self {
        BTree { degree, root: None }
    }

    fn split_child(&self, parent: &mut Box<Node<T>>, index: usize) {
        let target_node = &mut parent.children[index];

        let mut new_node = Box::new(Node::new());

        // Move keys greater than the median into the new node.
        new_node
            .keys
            .extend(target_node.keys.drain(self.degree..(2 * self.degree - 1)));

        // Move median key into parent.
        parent
            .keys
            .insert(index, target_node.keys.remove(self.degree - 1));

        // If the target node is not a leaf node, update children pointers.
        if !target_node.children.is_empty() {
            new_node
                .children
                .extend(target_node.children.drain(self.degree..2 * self.degree));
        }

        // Insert new node pointer into parent.
        parent.children.insert(index + 1, new_node);
    }

    fn insert_into(&self, node: &mut Box<Node<T>>, key: T) {
        if node.children.is_empty() {
            // TODO: Binary search is not necessary, but we can't manually
            // move values inside the vector without unsafe.
            node.keys
                .insert(node.keys.binary_search(&key).unwrap_err(), key);
        } else {
            let mut index = node.keys.binary_search(&key).unwrap_err();
            if node.children[index].keys.len() == (2 * self.degree) - 1 {
                self.split_child(node, index);
                if key > node.keys[index] {
                    index += 1;
                }
            }
            self.insert_into(&mut node.children[index], key);
        }
    }

    pub fn insert(&mut self, key: T) {
        let Some(mut root) = self.root.take() else {
            self.root = Some(Box::new(Node::with_keys(vec![key])));
            return;
        };

        if root.keys.len() == (2 * self.degree) - 1 {
            root = Box::new(Node::with_children(vec![root]));
            self.split_child(&mut root, 0);
        }

        self.insert_into(&mut root, key);
        self.root = Some(root);
    }
}

fn main() {
    let mut btree = BTree::<usize>::new(2);
    btree.insert(1);
    btree.insert(2);
    btree.insert(3);
    btree.insert(4);
    btree.insert(5);
    btree.insert(6);
    btree.insert(7);
    btree.insert(8);
    btree.insert(9);
    btree.insert(10);
    println!("{btree:?}");
}
