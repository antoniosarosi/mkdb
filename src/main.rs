#[derive(Debug)]
struct Node<T> {
    keys: Vec<T>,
    children: Vec<Box<Node<T>>>,
}

impl<T> Node<T> {
    pub fn new(keys: Vec<T>, children: Vec<Box<Self>>) -> Self {
        Self { keys, children }
    }

    pub fn new_empty() -> Self {
        Self::new(vec![], vec![])
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

    fn split_child(&mut self, parent: &mut Box<Node<T>>, child: usize) {
        let mut new_node = Box::new(Node::new_empty());

        // TODO: Optimize
        for i in self.degree..(2 * self.degree - 1) {
            new_node.keys.push(parent.children[child].keys.remove(i));
        }

        parent.keys.insert(child, parent.children[child].keys.remove(self.degree - 1));

        parent.children.insert(child + 1, new_node);

        // TODO: Non-leaf nodes
    }

    pub fn insert(&mut self, key: T) {
        match self.root {
            None => {
                self.root = Some(Box::new(Node::new(vec![key], vec![])));
            }

            Some(ref mut root) => {
                // Create new node
                if root.keys.len() == (2 * self.degree) - 1 {
                    let old_root = self.root.take().unwrap();

                    let mut new_root = Box::new(Node::new(vec![], vec![old_root]));

                    self.split_child(&mut new_root, 0);

                    self.root = Some(new_root);
                } else {
                    // Insert in root
                    root.keys.push(key);
                    root.keys.sort(); // TODO: Binary Search
                }
            }
        }
    }
}

fn main() {
    let mut btree = BTree::<usize>::new(2);
    btree.insert(1);
    btree.insert(2);
    btree.insert(3);
    btree.insert(4);
    println!("{btree:?}");
}
