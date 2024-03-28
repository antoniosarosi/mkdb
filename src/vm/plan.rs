//! Code that executes [`Plan`] trees. This is where real stuff happens :)
//!
//! The basic idea for the implementation of this module comes from this
//! [lecture].
//!
//! [lecture]: https://youtu.be/vmI72W-vgYI?si=xhO1CbX-DJZ5mkjb&t=377
//!
//! # Iterator Model
//!
//! The idea is to build some sort of "pipeline" where each [`Plan`] node is an
//! iterator that returns tuples and has an optional "source". The source is
//! just another [`Plan`] that does its own processing to the tuples before
//! returning them. For example, picture this query:
//!
//! ```sql
//! SELECT name, age FROM users WHERE age > 20 ORDER BY age;
//! ```
//!
//! The [`crate::query::planner`] would generate a [`Plan`] that looks like
//! this:
//!
//! ```text
//! Table Scan on users -> Filter by age -> Sort by age -> Project (name, age)
//! ```
//!
//! That way we can process tuples one at a time and each plan does one thing
//! only, which makes it easier to reason about the code. If some plan has
//! multiple sources then instead of a simple pipeline we'd have a tree. A
//! basic example is the `JOIN` statement which is not yet implemented.
//!
//! Another important details which makes the code here more complicated is that
//! some plans cannot work with a single tuple, they need all the tuples in
//! order to execute their code. One example is the [`Sort`] plan which needs
//! all the tuples before it can sort them. Other examples are the [`Update`] or
//! [`Delete`] plans which cannot make any changes to the underlying BTree until
//! the scan plan that is reading tuples from the BTree has finished reading.
//! That's because the scan plan holds an internal cursor and updating or
//! deleting from the BTree would invalidate that cursor.
//!
//! So, in order to deal with such cases, there's a special type of iterator
//! which is the [`BufferedIter`]. The [`BufferedIter`] contains an in-memory
//! buffer of configurable size that is written to a file once it fills up.
//! That way the [`BufferedIter`] can collect as many tuples as necessary
//! without memory concerns. Once all the tuples are collected, they are
//! returned one by one just like any other normal iterator would return them.
use std::{
    cell::RefCell,
    cmp::Ordering,
    collections::{HashMap, VecDeque},
    io::{self, BufRead, BufReader, Read, Seek, Write},
    mem,
    ops::Index,
    path::PathBuf,
    rc::Rc,
};

use crate::{
    db::{DbError, IndexMetadata, RowId, Schema, SqlError, TableMetadata},
    paging::{io::FileOps, pager::Pager},
    sql::statement::{Assignment, BinaryOperator, Expression, Value},
    storage::{reassemble_payload, tuple, BTree, BytesCmp, Cursor, FixedSizeMemCmp},
    vm,
};

pub(crate) type Tuple = Vec<Value>;

/// Plan node.
///
/// Each plan contains a tag (type of plan) and the structure that runs the plan
/// code. This is inspired by Postgres' query planner. See [createplan.c].
///
/// [createplan.c]: https://github.com/postgres/postgres/blob/master/src/backend/optimizer/plan/createplan.c
///
/// Technically we could make this work with traits and [`Box<dyn Trait>`] but
/// no clear benefit was found on previous attempts. See the comment below on
/// the [`Self::try_next`] impl block.
pub(crate) enum Plan<F> {
    /// Runs a sequential scan on a table and returns the rows one by one.
    SeqScan(SeqScan<F>),
    /// Uses an index to scan the table instead of doing it sequentially.
    IndexScan(IndexScan<F>),
    /// Returns raw values from `INSERT INTO` statements.
    Values(Values),
    /// Executes `WHERE` clauses and filters rows.
    Filter(Filter<F>),
    /// Final projection of a plan. Usually the columns of `SELECT` statements.
    Project(Project<F>),
    /// Inserts data into tables.
    Insert(Insert<F>),
    /// Executes assignment expressions from `UPDATE` statements.
    Update(Update<F>),
    /// Deletes data from tables.
    Delete(Delete<F>),
    /// Executes `ORDER BY` clauses.
    Sort(Sort<F>),
    /// Helper for the main [`Plan::Sort`] plan.
    SortKeysGen(SortKeysGen<F>),
}

// TODO: As mentioned at [`crate::paging::pager::get_as`], we could also use
// [`enum_dispatch`](https://docs.rs/enum_dispatch/) here to automate the match
// statement or switch to Box<dyn Iterator<Item = Result<Projection, DbError>>>
// but that's even more verbose than this and requires F: 'static everywhere. We
// also woudn't know the type of a plan because dyn Trait doesn't have a tag. So
// match it for now :)
impl<F: Seek + Read + Write + FileOps> Plan<F> {
    pub fn try_next(&mut self) -> Result<Option<Tuple>, DbError> {
        match self {
            Self::SeqScan(seq_scan) => seq_scan.try_next(),
            Self::IndexScan(index_scan) => index_scan.try_next(),
            Self::Values(values) => values.try_next(),
            Self::Filter(filter) => filter.try_next(),
            Self::Project(project) => project.try_next(),
            Self::Insert(insert) => insert.try_next(),
            Self::Update(update) => update.try_next(),
            Self::Delete(delete) => delete.try_next(),
            Self::Sort(sort) => sort.try_next(),
            Self::SortKeysGen(sort_keys_gen) => sort_keys_gen.try_next(),
        }
    }
}

impl<F: Seek + Read + Write + FileOps> Iterator for Plan<F> {
    type Item = Result<Tuple, DbError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.try_next().transpose()
    }
}

impl<F> Plan<F> {
    /// Returns the final schema of this plan.
    ///
    /// For now this is always going to be the columns of `SELECT` statements.
    pub fn schema(&self) -> Option<Schema> {
        match self {
            Self::Project(project) => Some(project.output_schema.clone()),
            _ => None,
        }
    }
}

/// Sequential scan plan.
///
/// This is not 100% sequential because we're not using a B+Tree but a normal
/// BTree.
///
/// See [`Cursor::try_next`] for details.
pub(crate) struct SeqScan<F> {
    pub table: TableMetadata,
    pub pager: Rc<RefCell<Pager<F>>>,
    pub cursor: Cursor,
}

impl<F: Seek + Read + Write + FileOps> SeqScan<F> {
    fn try_next(&mut self) -> Result<Option<Tuple>, DbError> {
        let mut pager = self.pager.borrow_mut();

        let Some((page, slot)) = self.cursor.try_next(&mut pager)? else {
            return Ok(None);
        };

        Ok(Some(tuple::deserialize(
            reassemble_payload(&mut pager, page, slot)?.as_ref(),
            &self.table.schema,
        )))
    }
}

/// Index scan uses an indexed column to retrieve data from a table.
///
/// An index is a BTree that maps a key to a [`RowId`], so it's easy to find
/// all the row IDs we need without scanning sequentially.
pub(crate) struct IndexScan<F> {
    pub index: IndexMetadata,
    pub table: TableMetadata,
    pub stop_when: Option<(Vec<u8>, BinaryOperator, Box<dyn BytesCmp>)>,
    pub done: bool,
    pub pager: Rc<RefCell<Pager<F>>>,
    pub cursor: Cursor,
}

// TODO: Sort index entries by row_id to improve sequential IO.
impl<F: Seek + Read + Write + FileOps> IndexScan<F> {
    fn try_next(&mut self) -> Result<Option<Tuple>, DbError> {
        if self.done {
            return Ok(None);
        };

        let mut pager = self.pager.borrow_mut();

        let Some((page, slot)) = self.cursor.try_next(&mut pager)? else {
            self.done = true;
            return Ok(None);
        };

        if let Some((key, operator, cmp)) = &self.stop_when {
            let entry = &pager.get(page)?.cell(slot).content;

            let ordering = cmp.bytes_cmp(entry, key);

            let stop = match operator {
                BinaryOperator::Gt => ordering == Ordering::Greater,
                BinaryOperator::GtEq => matches!(ordering, Ordering::Greater | Ordering::Equal),
                BinaryOperator::Lt => ordering == Ordering::Less,
                BinaryOperator::LtEq => matches!(ordering, Ordering::Less | Ordering::Equal),
                _ => unreachable!(),
            };

            if stop {
                self.done = true;
                return Ok(None);
            }
        }

        let index_entry =
            tuple::deserialize(&pager.get(page)?.cell(slot).content, &self.index.schema());

        let Value::Number(row_id) = index_entry[1] else {
            panic!("indexes should always map to row IDs but this one doesn't: {index_entry:?}");
        };

        let mut btree = BTree::new(
            &mut pager,
            self.table.root,
            FixedSizeMemCmp::for_type::<RowId>(),
        );

        let table_entry = btree
            .get(&tuple::serialize_row_id(row_id as RowId))?
            .unwrap_or_else(|| {
                panic!(
                    "index at root {} maps to row ID {} that doesn't exist in table at root {}",
                    self.index.root, row_id, self.table.root
                )
            });

        Ok(Some(tuple::deserialize(
            table_entry.as_ref(),
            &self.table.schema,
        )))
    }
}

/// Raw values from `INSERT INTO table (c1, c2) VALUES (v1, v2)`.
///
/// This supports multiple values but the parser does not currently parse
/// `INSERT` statements with multiple values.
pub(crate) struct Values {
    pub values: VecDeque<Vec<Expression>>,
}

impl Values {
    fn try_next(&mut self) -> Result<Option<Tuple>, DbError> {
        let Some(mut values) = self.values.pop_front() else {
            return Ok(None);
        };

        Ok(Some(
            values
                .drain(..)
                .map(|expr| vm::resolve_literal_expression(&expr))
                .collect::<Result<Vec<Value>, SqlError>>()?,
        ))
    }
}

/// Applies a filter [`Expression`] to its source returning only tuples that
/// evaluate to `true`.
///
/// Used for `WHERE` clauses in `SELECT`, `DELETE` and `UPDATE` statements.
pub(crate) struct Filter<F> {
    pub source: Box<Plan<F>>,
    pub schema: Schema,
    pub filter: Expression,
}

impl<F: Seek + Read + Write + FileOps> Filter<F> {
    fn try_next(&mut self) -> Result<Option<Tuple>, DbError> {
        while let Some(tuple) = self.source.try_next()? {
            if vm::eval_where(&self.schema, &tuple, &self.filter)? {
                return Ok(Some(tuple));
            }
        }

        Ok(None)
    }
}

/// Applies a projection to a tuple.
///
/// In simple words, it "selects" columns from a row. For example:
///
/// ```sql
/// CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(255), age INT);
///
/// SELECT id, age FROM users;
/// ```
///
/// The projection in this case would be the "id" and "age columns". The name
/// is discarded.
pub(crate) struct Project<F> {
    pub source: Box<Plan<F>>,
    pub input_schema: Schema,
    pub output_schema: Schema,
    pub projection: Vec<Expression>,
}

impl<F: Seek + Read + Write + FileOps> Project<F> {
    fn try_next(&mut self) -> Result<Option<Tuple>, DbError> {
        let Some(tuple) = self.source.try_next()? else {
            return Ok(None);
        };

        Ok(Some(
            self.projection
                .iter()
                .map(|expr| vm::resolve_expression(&tuple, &self.input_schema, expr))
                .collect::<Result<Tuple, _>>()?,
        ))
    }
}

/// Inserts data into a table and upates indexes.
pub(crate) struct Insert<F> {
    pub pager: Rc<RefCell<Pager<F>>>,
    pub source: Box<Plan<F>>,
    pub table: TableMetadata,
}

impl<F: Seek + Read + Write + FileOps> Insert<F> {
    fn try_next(&mut self) -> Result<Option<Tuple>, DbError> {
        let Some(mut tuple) = self.source.try_next()? else {
            return Ok(None);
        };

        let mut pager = self.pager.borrow_mut();

        let mut btree = BTree::new(
            &mut pager,
            self.table.root,
            FixedSizeMemCmp::for_type::<RowId>(),
        );

        btree.insert(tuple::serialize(&self.table.schema, &tuple))?;

        for index in &self.table.indexes {
            let col = self.table.schema.index_of(&index.column.name).unwrap();
            assert_eq!(self.table.schema.columns[0].name, "row_id");

            let key = tuple[col].clone();
            let row_id = tuple[0].clone();

            let entry = tuple::serialize(&index.schema(), &[key, row_id]);

            let mut btree = BTree::new(
                &mut pager,
                index.root,
                Box::<dyn BytesCmp>::from(&index.column.data_type),
            );

            btree
                .try_insert(entry)?
                .map_err(|_| SqlError::DuplicatedKey(tuple.swap_remove(col)))?;
        }

        Ok(Some(vec![]))
    }
}

/// Assigns values to columns.
pub(crate) struct Update<F> {
    pub table: TableMetadata,
    pub assignments: Vec<Assignment>,
    pub pager: Rc<RefCell<Pager<F>>>,
    pub source: BufferedIter<F>,
}

impl<F: Seek + Read + Write + FileOps> Update<F> {
    fn try_next(&mut self) -> Result<Option<Tuple>, DbError> {
        let Some(mut tuple) = self.source.try_next()? else {
            return Ok(None);
        };

        let mut updated_index_cols = HashMap::new();

        for assignment in &self.assignments {
            let col = self.table.schema.index_of(&assignment.identifier).unwrap();
            let new_value = vm::resolve_expression(&tuple, &self.table.schema, &assignment.value)?;
            let old_value = tuple[col].clone();
            tuple[col] = new_value.clone();

            updated_index_cols.insert(assignment.identifier.clone(), (old_value, new_value));
        }

        let mut pager = self.pager.borrow_mut();

        let mut btree = BTree::new(
            &mut pager,
            self.table.root,
            FixedSizeMemCmp::for_type::<RowId>(),
        );

        btree.insert(tuple::serialize(&self.table.schema, &tuple))?;

        for index in &self.table.indexes {
            if let Some((old_key, new_key)) = updated_index_cols.remove(&index.column.name) {
                let mut btree = BTree::new(
                    &mut pager,
                    index.root,
                    Box::<dyn BytesCmp>::from(&index.column.data_type),
                );

                btree.remove(&tuple::serialize(
                    &Schema::from(vec![index.column.clone()]),
                    &[old_key],
                ))?;

                btree
                    .try_insert(tuple::serialize(&index.schema(), &[
                        new_key.clone(),
                        tuple[0].clone(),
                    ]))?
                    .map_err(|_| SqlError::DuplicatedKey(new_key))?;
            }
        }

        Ok(Some(vec![]))
    }
}

/// Removes values from a table BTree and from all the necessary index BTrees.
pub(crate) struct Delete<F> {
    pub table: TableMetadata,
    pub pager: Rc<RefCell<Pager<F>>>,
    pub source: BufferedIter<F>,
}

impl<F: Seek + Read + Write + FileOps> Delete<F> {
    fn try_next(&mut self) -> Result<Option<Tuple>, DbError> {
        let Some(tuple) = self.source.try_next()? else {
            return Ok(None);
        };

        let mut pager = self.pager.borrow_mut();
        let mut btree = BTree::new(
            &mut pager,
            self.table.root,
            FixedSizeMemCmp::for_type::<RowId>(),
        );

        btree.remove(&tuple::serialize(&self.table.schema, &tuple))?;

        for index in &self.table.indexes {
            let col_idx = self.table.schema.index_of(&index.column.name).unwrap();
            let key = tuple[col_idx].clone();

            let entry = tuple::serialize(&Schema::from(vec![index.column.clone()]), &[key]);

            let mut btree = BTree::new(
                &mut pager,
                index.root,
                Box::<dyn BytesCmp>::from(&index.column.data_type),
            );

            btree.remove(&entry)?;
        }

        Ok(Some(vec![]))
    }
}

/// In-memory tuple buffer.
///
/// This structure stores [`Tuple`] instances (AKA rows) and can be written to
/// a file once it reaches a certain preconfigured threshold. The
/// [`TupleBuffer`] can be used in two different modes: packed and not packed.
///
/// Packing affects how the buffer is written to files. If packed, then
/// in-memory tuples will be written to the file in one single continuous
/// sequence of bytes that follows this format:
///
/// ```text
/// +-----------------+ <---+
/// |   Tuple 0 Size  |     |
/// +-----------------+     |
/// |       ...       |     |
/// | Tuple 0 Content |     |
/// |       ...       |     |
/// +-----------------+     | VARIABLE SIZE
/// |   Tuple 1 Size  |     |
/// +-----------------+     |
/// |       ...       |     |
/// | Tuple 1 Content |     |
/// |       ...       |     |
/// +-----------------+ <---+
/// ```
///
/// This is useful for simply collecting a huge amount of tuples and then
/// returning them back, which is what the [`BufferedIter`] does. However, the
/// [`Sort`] plan needs to work with fixed size pages, so it will use the
/// [`TupleBuffer`] without packing. In that case, only one page at a time is
/// written using this format:
///
/// ```text
/// +-----------------+ <---+
/// |    Num Tuples   |     |
/// +-----------------+     |
/// |   Tuple 0 Size  |     |
/// +-----------------+     |
/// |       ...       |     |
/// | Tuple 0 Content |     |
/// |       ...       |     |
/// +-----------------+     |
/// |   Tuple 1 Size  |     | PAGE SIZE
/// +-----------------+     |
/// |       ...       |     |
/// | Tuple 1 Content |     |
/// |       ...       |     |
/// +-----------------+     |
/// |       ...       |     |
/// |     Padding     |     |
/// |       ...       |     |
/// +-----------------+ <---+
/// ```
///
/// Both the number of tuples and the tuple sizes are stored using 4 byte little
/// endian integers.
#[derive(Debug)]
pub(crate) struct TupleBuffer {
    /// Maximum size of this buffer in bytes.
    page_size: usize,

    /// Current size of the buffer in bytes.
    current_size: usize,

    /// Size in bytes of the largest tuple that has ever been stored in this
    /// buffer.
    ///
    /// This number is not updated if tuples are removed from the buffer, it
    /// simply stores the maximum size that has been recorded.
    largest_tuple_size: usize,

    /// See the [`TupleBuffer`] documentation.
    packed: bool,

    /// Schema of the tuples in this buffer.
    schema: Schema,

    /// Tuple FIFO queue.
    tuples: VecDeque<Tuple>,
}

impl Index<usize> for TupleBuffer {
    type Output = Tuple;

    fn index(&self, index: usize) -> &Self::Output {
        &self.tuples[index]
    }
}

impl TupleBuffer {
    /// Creates an empty buffer that doesn't serve any purpose.
    ///
    /// Used to move buffers out using [`mem::replace`] just like
    /// [`Option::take`] moves values out.
    pub fn empty() -> Self {
        Self {
            page_size: 0,
            schema: Schema::empty(),
            packed: false,
            current_size: 0,
            largest_tuple_size: 0,
            tuples: VecDeque::new(),
        }
    }

    /// Creates a new buffer. Doesn't allocate anything yet.
    pub fn new(page_size: usize, schema: Schema, packed: bool) -> Self {
        Self {
            page_size,
            schema,
            packed,
            current_size: if packed { 0 } else { mem::size_of::<u32>() },
            largest_tuple_size: 0,
            tuples: VecDeque::new(),
        }
    }

    /// Returns `true` if the given `tuple` can be appended to this buffer
    /// without incrementing its size past [`Self::page_size`].
    pub fn can_fit(&self, tuple: &Tuple) -> bool {
        let tuple_size = tuple::size_of(tuple, &self.schema);
        let total_tuple_size = mem::size_of::<u32>() + tuple_size;

        self.current_size + total_tuple_size <= self.page_size
    }

    /// Appends the given `tuple` to the buffer.
    ///
    /// It doesn't panic or return any error if the buffer overflows
    /// [`Self::page_size`], this function won't fail. This is useful because
    /// the [`BufferedIter`] needs to be able to process tuples of any size even
    /// if they are larger than the maximum buffer size. The [`Sort`] plan has
    /// its own tricks to avoid working with tuples that wouldn't fit in the
    /// buffer.
    pub fn push(&mut self, tuple: Tuple) {
        let tuple_size = tuple::size_of(&tuple, &self.schema);
        let total_tuple_size = mem::size_of::<u32>() + tuple_size;

        if tuple_size > self.largest_tuple_size {
            self.largest_tuple_size = tuple_size;
        }

        self.current_size += total_tuple_size;
        self.tuples.push_back(tuple);
    }

    /// Removes the first tuple in this buffer and returns it.
    pub fn pop_front(&mut self) -> Option<Tuple> {
        self.tuples.pop_front().inspect(|tuple| {
            self.current_size -= mem::size_of::<u32>() + tuple::size_of(tuple, &self.schema);
        })
    }

    /// `true` if there are no tuples stored in this buffer.
    pub fn is_empty(&self) -> bool {
        self.tuples.is_empty()
    }

    /// Resets the state of the buffer to "empty".
    pub fn clear(&mut self) {
        self.tuples.clear();
        self.current_size = if self.packed {
            0
        } else {
            mem::size_of::<u32>()
        };
    }

    /// Serializes this buffer into a byte array that can be written to a file.
    pub fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(self.page_size);

        if !self.packed {
            buf.extend_from_slice(&(self.tuples.len() as u32).to_le_bytes());
        }

        for tuple in &self.tuples {
            let serialized = tuple::serialize(&self.schema, tuple);
            buf.extend_from_slice(&(serialized.len() as u32).to_le_bytes());
            buf.extend_from_slice(&serialized);
        }

        if !self.packed {
            buf.resize(self.page_size, 0);
        }

        buf
    }

    /// Writes the contents of this buffer to the given `file`.
    ///
    /// The buffer is not modified in any way, call [`Self::clear`] to delete
    /// the tuples from it.
    pub fn write_to(&self, file: &mut impl Write) -> io::Result<()> {
        file.write_all(&self.serialize())
    }

    /// Reads one page from the given file into memory.
    ///
    /// Only works with fixed size buffers (not packed) that are already empty
    /// and the underlying file cursor **must already be positioned** at the
    /// beginning of a page.
    pub fn read_from(&mut self, file: &mut impl Read) -> io::Result<()> {
        assert!(
            self.is_empty() && !self.packed,
            "read_from() only works with fixed size empty buffers"
        );

        let mut buf = vec![0; self.page_size];
        file.read_exact(&mut buf)?;

        let number_of_tuples = u32::from_le_bytes(buf[..mem::size_of::<u32>()].try_into().unwrap());
        let mut index = mem::size_of::<u32>();

        for _ in 0..number_of_tuples {
            let size = u32::from_le_bytes(
                buf[index..index + mem::size_of::<u32>()]
                    .try_into()
                    .unwrap(),
            ) as usize;

            index += mem::size_of::<u32>();

            self.push(tuple::deserialize(&buf[index..index + size], &self.schema));
            index += size;
        }

        Ok(())
    }

    /// Same as [`Self::read_from`] but positions the file cursor at the
    /// beginning of the given page number.
    pub fn read_page(&mut self, file: &mut (impl Seek + Read), page: usize) -> io::Result<()> {
        file.seek(io::SeekFrom::Start((self.page_size * page) as u64))?;
        self.read_from(file)
    }

    /// Returns the minimum power of two that could fit at least one tuple of
    /// the given size.
    ///
    /// This is computed for fixed size buffers (not packed).
    pub fn page_size_needed_for(tuple_size: usize) -> usize {
        // Bit hack that computes the next power of two for 32 bit integers
        // (although we're using usize for convinience to avoid casting).
        // See here:
        // https://stackoverflow.com/questions/466204/rounding-up-to-next-power-of-2
        let mut page_size = mem::size_of::<u32>() * 2 + tuple_size;
        page_size -= 1;
        page_size |= page_size >> 1;
        page_size |= page_size >> 2;
        page_size |= page_size >> 4;
        page_size |= page_size >> 8;
        page_size |= page_size >> 16;
        page_size += 1;

        page_size
    }
}

/// Similar to [`io::BufReader`] and [`io::BufWriter`].
///
/// This structure consumes all the tuples from its source through the
/// [`Self::collect`] operation and writes them to a "collection" file if they
/// don't fit in memory. Once tuples have been successfully collected,
/// [`BufferedIter`] acts like a normal iterator that returns tuples one at a
/// time. If the collected tuples fit in memory there is no IO, so that's the
/// best case scenario.
pub(crate) struct BufferedIter<F> {
    /// Tuple source. This is where we collect from.
    pub source: Box<Plan<F>>,
    /// Tuple schema.
    pub schema: Schema,
    /// `true` if [`Self::collect`] completed successfully.
    pub collected: bool,
    /// In-memory buffer that stores tuples from the source.
    pub mem_buf: TupleBuffer,
    /// File handle/descriptor in case we had to create the collection file.
    pub file: Option<F>,
    /// Buffered reader in case we created the file and have to read from it.
    pub reader: Option<BufReader<F>>,
    /// Path of the collection file.
    pub file_path: PathBuf,
}

impl<F: FileOps> BufferedIter<F> {
    /// Drops the IO resource and deletes it from the file system.
    fn drop_file(&mut self) -> io::Result<()> {
        drop(self.file.take());
        drop(self.reader.take());
        F::remove(&self.file_path)
    }
}

// TODO: Requires defining the struct as BufferdIter<F: FileOps>
// impl<F: FileOps> Drop for BufferedIter<F> {
//     fn drop(&mut self) {
//         if self.file.is_some() {
//             self.drop_file();
//         }
//     }
// }

impl<F: Seek + Read + Write + FileOps> BufferedIter<F> {
    // TODO: Create a builder or something.
    pub fn new(source: Box<Plan<F>>, work_dir: PathBuf, schema: Schema) -> Self {
        // TODO: Use uuid or tempfile or something. This is poor man's random
        // file name.
        let file_path = work_dir.join(format!("{}.mkdb.query", &*source as *const _ as usize));

        Self {
            source,
            mem_buf: TupleBuffer::new(256, schema.clone(), true),
            schema,
            collected: false,
            file_path,
            file: None,
            reader: None,
        }
    }

    /// Collects all the tuples from [`Self::source`].
    fn collect(&mut self) -> Result<(), DbError> {
        // Buffer tuples in-memory until we have no space left. At that point
        // create the file if it doesn't exist, write the buffer to disk and
        // repeat until there are no more tuples.
        while let Some(tuple) = self.source.try_next()? {
            if !self.mem_buf.can_fit(&tuple) {
                if self.file.is_none() {
                    self.file = Some(F::create(&self.file_path)?);
                }
                self.mem_buf.write_to(self.file.as_mut().unwrap())?;
                self.mem_buf.clear();
            }

            self.mem_buf.push(tuple);
        }

        // If we ended up creating a file and writing to it we must set the
        // cursor position back to the first byte in order to read from it
        // later.
        if let Some(mut file) = self.file.take() {
            file.rewind()?;
            self.reader = Some(BufReader::new(file));
        }

        Ok(())
    }

    pub fn try_next(&mut self) -> Result<Option<Tuple>, DbError> {
        if !self.collected {
            self.collect()?;
            self.collected = true;
        }

        // While there's stuff written to the file return from there.
        if let Some(reader) = self.reader.as_mut() {
            if reader.has_data_left()? {
                let mut size = [0; mem::size_of::<u32>()];
                reader.read_exact(&mut size)?;

                let mut serialized = vec![0; u32::from_le_bytes(size) as usize];
                reader.read_exact(&mut serialized)?;

                return Ok(Some(tuple::deserialize(&serialized, &self.schema)));
            }

            // Reader is done, drop the file.
            self.drop_file()?;
        }

        // If there's no file or the file has been consumed return from memory.
        // Tuples that were not written to the file because it wasn't necessary
        // are also returned here.
        Ok(self.mem_buf.pop_front())
    }
}

/// Generates sort keys.
///
/// This is a helper for the main [`Sort`] plan that basically evaluates the
/// `ORDER BY` expressions and appends the results to each tuple.
///
/// See the documentation of [`Sort`] for more details.
pub(crate) struct SortKeysGen<F> {
    pub source: Box<Plan<F>>,
    pub schema: Schema,
    pub order_by: Vec<Expression>,
}

impl<F: Seek + Read + Write + FileOps> SortKeysGen<F> {
    pub fn try_next(&mut self) -> Result<Option<Tuple>, DbError> {
        let Some(mut tuple) = self.source.try_next()? else {
            return Ok(None);
        };

        for expr in &self.order_by {
            tuple.push(vm::resolve_expression(&tuple, &self.schema, expr)?);
        }

        Ok(Some(tuple))
    }
}

/// External K-way merge sort implementation.
///
/// Check this [lecture] for the basic idea:
///
/// [lecture]: https://youtu.be/DOu7SVUbuuM?si=gQM_rf1BESUmSdLo&t=1517
///
/// # Algorithm
///
/// Variable length data makes this algorithm a little bit more complicated than
/// the lecture suggests but the core concepts are the same. The first thing we
/// do is we generate the "sort keys" for all the tuples and collect them into
/// a file or in-memory buffer if we're lucky enough and they fit. Sort keys are
/// basically the results of the `ORDER BY` expressions. For example:
///
/// ```sql
/// CREATE TABLE products (id INT PRIMARY KEY, name VARCHAR(255), price INT, discount INT);
///
/// SELECT * FROM products ORDER BY price * discount, price;
/// ```
///
/// The "sort keys" for each tuple in the query above would be the values
/// generated for the expressions `price * discount` and `price`. These sort
/// keys are computed by [`SortKeysGen`] before the [`BufferedIter`] writes the
/// tuple to a file or an in-memory buffer and they are appended to the end of
/// the tuple after all its other columns.
///
/// The reason we need to generate the sort keys so early is because they change
/// the length in bytes of the tuple and we need to know the length of the
/// largest tuple that we are going to work with in order to compute the exact
/// page size that we need for the K-way external merge sort.
///
/// Note that there is no overflow at this level, tuples that are distributed
/// across overflow pages are already merged together back into one contiguous
/// buffer by the scan plan. So we only work with complete data here.
///
/// Once the [`BufferedIter`] has successfully collected all the tuples modified
/// by [`SortKeysGen`] we can finally start sorting.
///
/// There are two main cases:
///
/// 1. The [`BufferedIter`] did not use any files because all the tuples fit
/// in its in-memory buffer. In that case, move the buffer out of the
/// [`BufferedIter`] into the [`Sort`] plan and just do an in-memory sorting.
/// No IO required.
///
/// 2. The [`BufferedIter`] had to create a file in order to collect all
/// tuples. This is the complicated case.
///
/// # External 2-Way Merge Sort With Variable Length Data
///
/// As mentioned earlier, we keep track of the largest tuple in order to compute
/// the page size that we need for sorting. The page size is either that of the
/// [`Pager`] or the closest power of two that can fit the largest tuple.
///
/// Once we know the exact page size we can start the "pass 0" or "1 page runs".
/// In this step we collect tuples into one page, sort them in-memory, write the
/// page to a file, fill the page again, sort it, write it to the file again and
/// so on until there are no more tuples. The file ends up looking roughly like
/// this:
///
/// ```text
///       40     20        20    20    20         30      30               60
///     bytes   bytes     bytes bytes bytes      bytes   bytes            bytes
/// +-------------------+--------------------+--------------------+--------------------+
/// | +--------+----+   | +----+----+----+   | +-------+------+   | +--------------+   |
/// | |   T3   | T5 |   | | T1 | T7 | T8 |   | |  T4   |  T6  |   | |      T2      |   |
/// | +--------+----+   | +----+----+----+   | +-------+------+   | +--------------+   |
/// +-------------------+--------------------+--------------------+--------------------+
///        PAGE 0               PAGE 1               PAGE 2               PAGE 3
///       64 bytes             64 bytes             64 bytes             64 bytes
/// ```
///
/// As you can see, tuples are sorted within every page but they are not sorted
/// as a whole, so there is still a lot of work to do. After pass 0 is done we
/// don't actually need to do any more "sorting" per se, we only need to "merge"
/// pages together.
///
/// ## The Merge Sub-algorithm
///
/// Once we've completed the "1 page runs" we do the "2 page runs". Basically
/// we load two pages into memory, and since they are already sorted on their
/// own we can create a new set of pages that are fully sorted from start to
/// finish. Continuing with the example above, we start by loading page 0 and
/// page 1 into memory and merging them together:
///
/// ```text
///       40     20
///     bytes   bytes
/// +--------------------+
/// | +--------+----+    |
/// | |   T3   | T5 |    | ------+
/// | +--------+----+    |       |            20       40
/// +--------------------+       |           bytes    bytes
///        PAGE 0                |         +--------------------+
///       64 bytes               |         | +----+---------+   |
///                              +-------> | | T1 |    T3   |   |
///    20    20    20            |         | +----+---------+   |
///   bytes bytes bytes          |         +--------------------+
/// +--------------------+       |              MERGED PAGE 0
/// | +----+----+----+   |       |                64 bytes
/// | | T1 | T7 | T8 |   | ------+
/// | +----+----+----+   |
/// +--------------------+
///         PAGE 1
///        64 bytes
/// ```
///
/// Now that the first merged page is full, we write it to a new file and
/// continue merging the rest of tuples:
///
/// ```text
///     20
///   bytes
/// +--------------------+
/// | +----+             |
/// | | T5 |             | ------+
/// | +----+             |       |            20    20    20
/// +--------------------+       |           bytes bytes bytes
///        PAGE 0                |         +--------------------+
///       64 bytes               |         | +----+----+----+   |
///                              +-------> | | T5 | T7 | T8 |   |
///    20    20                  |         | +----+----+----+   |
///   bytes bytes                |         +--------------------+
/// +--------------------+       |              MERGED PAGE 1
/// | +----+----+        |       |                64 bytes
/// | | T7 | T8 |        | ------+
/// | +----+----+        |
/// +--------------------+
///         PAGE 1
///        64 bytes
/// ```
///
/// Pages 0 and 1 are now completely merged into two sorted pages. This is how
/// the new file looks like so far:
///
/// ```text
///     20       40          20    20    20
///    bytes    bytes       bytes bytes bytes
///  +--------------------+--------------------+
///  | +----+---------+   | +----+----+----+   |
///  | | T1 |    T3   |   | | T5 | T7 | T8 |   |
///  | +----+---------+   | +----+----+----+   |
///  +--------------------+--------------------+
///       MERGED PAGE 0        MERGED PAGE 1
///         64 bytes             64 bytes
/// ```
///
/// As you can see, we have 2 pages that are fully sorted from start to finish.
/// That's why this step is called a "2 page run". However, merging 2 pages
/// won't always result in exactly 2 new pages due to the nature of variable
/// length data, but we'll discuss that later. Now we load the next two pages in
/// memory and repeat the process. We end up with this new file:
///
/// ```text
///     20       40          20    20    20             60               30      30
///    bytes    bytes       bytes bytes bytes          bytes            bytes   bytes
///  +--------------------+--------------------+--------------------+--------------------+
///  | +----+---------+   | +----+----+----+   | +--------------+   | +-------+------+   |
///  | | T1 |    T3   |   | | T5 | T7 | T8 |   | |      T2      |   | |  T4   |  T6  |   |
///  | +----+---------+   | +----+----+----+   | +--------------+   | +-------+------+   |
///  +--------------------+--------------------+--------------------+--------------------+
///       MERGED PAGE 0        MERGED PAGE 1       MERGED PAGE 2         MERGED PAGE 3
///         64 bytes             64 bytes             64 bytes             64 bytes
/// ```
///
/// ## Dealing With Variable Length Data
///
/// The new file has the same number of pages as the original "1 page runs"
/// file. However, that is not always the case due to how variable length data
/// fits in fixed size pages. There's a clear example in the next step of the
/// algorithm: the "4 page runs". Now instead of taking 2 pages and merging
/// them together we take 4 pages. We still load 2 pages at a time in memory,
/// but we do so until we've merged 4 pages instead of 2. So we'll use two
/// cursors for this task:
///
/// ```text
///     20       40          20    20    20             60               30      30
///    bytes    bytes       bytes bytes bytes          bytes            bytes   bytes
///  +--------------------+--------------------+--------------------+--------------------+
///  | +----+---------+   | +----+----+----+   | +--------------+   | +-------+------+   |
///  | | T1 |    T3   |   | | T5 | T7 | T8 |   | |      T2      |   | |  T4   |  T6  |   |
///  | +----+---------+   | +----+----+----+   | +--------------+   | +-------+------+   |
///  +--------------------+--------------------+--------------------+--------------------+
///       MERGED PAGE 0        MERGED PAGE 1       MERGED PAGE 2         MERGED PAGE 3
///         64 bytes             64 bytes             64 bytes             64 bytes
///
///            ^                                         ^
///            |                                         |
///         CURSOR 1                                 CURSOR 2
/// ```
///
/// Cursor one starts pointing at 0 and cursor two points at 2. We start by
/// merging pages 0 and 2 and then we move to pages 1 and 3. So, 0 and 2 first:
///
/// ```text
///    20       40
///   bytes    bytes
/// +--------------------+
/// | +----+---------+   |
/// | | T1 |    T3   |   | ------+
/// | +----+---------+   |       |            20
/// +--------------------+       |           bytes
///      MERGED PAGE 0           |         +--------------------+
///        64 bytes              |         | +----+             |
///                              +-------> | | T1 |             |
///          60                  |         | +----+             |
///         bytes                |         +--------------------+
/// +--------------------+       |              MERGED PAGE 0
/// | +--------------+   |       |                64 bytes
/// | |      T2      |   | ------+
/// | +--------------+   |
/// +--------------------+
///     MERGED PAGE 2
///        64 bytes
/// ```
///
/// Notice how the first page that we produce can only fit tuple 1 because
/// tuple 2 is 60 bytes in size and won't fit in the 64 byte page. So we write
/// page 0 as is, with only one tuple, and produce the next one:
///
/// ```text
///       40
///      bytes
/// +--------------------+
/// | +---------+        |
/// | |    T3   |        | ------+
/// | +---------+        |       |                  60
/// +--------------------+       |                 bytes
///      MERGED PAGE 0           |         +--------------------+
///        64 bytes              |         | +--------------+   |
///                              +-------> | |      T2      |   |
///          60                  |         | +--------------+   |
///         bytes                |         +--------------------+
/// +--------------------+       |              MERGED PAGE 1
/// | +--------------+   |       |                64 bytes
/// | |      T2      |   | ------+
/// | +--------------+   |
/// +--------------------+
///     MERGED PAGE 2
///        64 bytes
/// ```
///
/// The second page we've produced is full again. So write it to the output file
/// and produce the next one. Page 2 is also empty, it doesn't have any more
/// tuples, so we'll move cursor two and load page 3 in its place:
///
///
/// ```text
///       40
///      bytes
/// +--------------------+
/// | +---------+        |
/// | |    T3   |        | ------+
/// | +---------+        |       |               40
/// +--------------------+       |              bytes
///      MERGED PAGE 0           |         +--------------------+
///        64 bytes              |         | +---------+        |
///                              +-------> | |    T3   |        |
///      30      30              |         | +---------+        |
///     bytes   bytes            |         +--------------------+
/// +--------------------+       |              MERGED PAGE 2
/// | +-------+------+   |       |                64 bytes
/// | |  T4   |  T6  |   | ------+
/// | +-------+------+   |
/// +--------------------+
///      MERGED PAGE 3
///        64 bytes
/// ```
///
/// See how we're facing the same issue again where the output page won't be
/// able to fit tuple 4. No problem, just write the output page to the output
/// file and continue repeating the process. Once we've written tuple 3 page 0
/// becomes empty, so we'll shift its cursor to page 1:
///
/// ```text
///    20    20    20
///   bytes bytes bytes
/// +--------------------+
/// | +----+----+----+   |
/// | | T5 | T7 | T8 |   | ------+
/// | +----+----+----+   |       |             30      20
/// +--------------------+       |            bytes   bytes
///      MERGED PAGE 1           |         +--------------------+
///        64 bytes              |         | +-------+----+     |
///                              +-------> | |  T4   | T5 |     |
///      30      30              |         | +-------+----+     |
///     bytes   bytes            |         +--------------------+
/// +--------------------+       |              MERGED PAGE 3
/// | +-------+------+   |       |                64 bytes
/// | |  T4   |  T6  |   | ------+
/// | +-------+------+   |
/// +--------------------+
///      MERGED PAGE 3
///        64 bytes
/// ```
///
/// We managed to squeeze tuples 4 and 5 into page 3. Now we still have to merge
/// the rest of them. At the end, we'll end up with this new file:
///
/// ```text
///     20                         60                40                 30      20           30      20          20
///    bytes                      bytes             bytes              bytes   bytes        bytes   bytes       bytes
///  +--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+
///  | +----+             | +--------------+   | +---------+        | +-------+----+     | +-------+----+     | +----+             |
///  | | T1 |             | |      T2      |   | |    T3   |        | |  T4   | T5 |     | |  T6   | T7 |     | | T8 |             |
///  | +----+             | +--------------+   | +---------+        | +-------+----+     | +-------+----+     | +----+             |
///  +--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+
///       MERGED PAGE 0       MERGED PAGE 1         MERGED PAGE 2        MERGED PAGE 3        MERGED PAGE 4        MERGED PAGE 5
///         64 bytes             64 bytes             64 bytes             64 bytes             64 bytes             64 bytes
/// ```
///
/// In theory, we should have produced 4 pages, that's why it'c called a
/// "4 page run". But variable length data makes everything more interesting
/// doesn't it? Anyway, if the original input file had more than 4 pages, we'd
/// continue the 4 page run by shifting the cursor to pages 4-8. We'd merge 4
/// and 6, then 5 and 7. And so on until there are no more pages. After that we
/// duplicate the page runs every time. So after the "4 page runs" we'd do the
/// "8 page runs", which is not necessary in this case because we don't even
/// have 8 pages to begin with and everything is already sorted.
///
/// But that's the idea, we keep duplicating the page runs until we're done. It
/// doesn't matter whether the amount of pages changes or not in the process,
/// the algorithm still works. The only caveat is that we need two files. We
/// have an input file from which we read from and an output file where we write
/// to. Once we've completed an entire pass, we swap the files so that the
/// previous output file becomes the new input file and the previous input file
/// is truncated and becomes the new output file.
///
/// # K-Way Merge
///
/// The algorithm described above is a 2-way merge where we only use two input
/// buffers or input pages and one output buffer. However, we can use more than
/// two input buffers and generalize the algorithm to work for K buffers. In
/// that case, after we do the 1 page runs to sort pages, we don't do a "2 page
/// run", we do a "K page run" because we can load K pages at a time in memory.
///
/// Again, starting from the sorted file after we do the "1 page runs":
///
/// ```text
///       40     20        20    20    20         30      30               60
///     bytes   bytes     bytes bytes bytes      bytes   bytes            bytes
/// +-------------------+--------------------+--------------------+--------------------+
/// | +--------+----+   | +----+----+----+   | +-------+------+   | +--------------+   |
/// | |   T3   | T5 |   | | T1 | T7 | T8 |   | |  T4   |  T6  |   | |      T2      |   |
/// | +--------+----+   | +----+----+----+   | +-------+------+   | +--------------+   |
/// +-------------------+--------------------+--------------------+--------------------+
///        PAGE 0               PAGE 1               PAGE 2               PAGE 3
///       64 bytes             64 bytes             64 bytes             64 bytes
/// ```
///
/// If we have 4 buffers instead of 2 we can load all these 4 pages into memory
/// and merge them:
///
/// ```text
///       40     20
///     bytes   bytes
/// +--------------------+
/// | +--------+----+    |
/// | |   T3   | T5 |    | ------+
/// | +--------+----+    |       |
/// +--------------------+       |
///        PAGE 0                |
///       64 bytes               |
///                              |
///    20    20    20            |
///   bytes bytes bytes          |
/// +--------------------+       |
/// | +----+----+----+   |       |
/// | | T1 | T7 | T8 |   | ------+
/// | +----+----+----+   |       |            20
/// +--------------------+       |           bytes
///         PAGE 1               |         +-------------------+
///        64 bytes              |         | +----+            |
///                              +-------> | | T1 |            |
///      30      30              |         | +----+            |
///     bytes   bytes            |         +-------------------+
/// +--------------------+       |             MERGED PAGE 0
/// | +-------+------+   |       |                64 bytes
/// | |  T4   |  T6  |   | ------+
/// | +-------+------+   |       |
/// +--------------------+       |
///         PAGE 2               |
///        64 bytes              |
///                              |
///          60                  |
///         bytes                |
/// +--------------------+       |
/// | +--------------+   |       |
/// | |      T2      |   | ------+
/// | +--------------+   |
/// +--------------------+
///         PAGE 2
///        64 bytes
/// ```
///
/// And so we'd produce the entire sorted file in one single run. The next run
/// would be able to produce 16 pages because we already have segments of 4
/// pages that are sorted. With a K-way algorithm each run multiplies the number
/// or pages by K instead of doubling it, which results in fewer overall passes
/// through the file and thus less IO. Check the [lecture] mentioned at the
/// beginning for the IO complexity analysis and formulas, but basically the
/// more buffers we have the less IO we have to do. The number of input buffers
/// is configured through [`Self::input_buffers`].
pub(crate) struct Sort<F> {
    /// Tuple input.
    pub source: BufferedIter<F>,
    /// Original schema of the tuples.
    pub schema: Schema,
    /// Schema that includes the sort keys.
    pub sort_schema: Schema,
    /// `true` if we already sorted the tuples.
    pub sorted: bool,
    /// Page size used by the [`Pager`].
    pub page_size: usize,
    /// How many input buffers to use for the K-way algorithm. This is "K".
    pub input_buffers: usize,
    /// K-way output buffer. Used also for sorting in memory and returning tuples.
    pub output_buffer: TupleBuffer,
    /// Working directory to create temporary files.
    pub work_dir: PathBuf,
    /// File used to read tuples.
    pub input_file: Option<F>,
    /// File used to write tuples to.
    pub output_file: Option<F>,
    /// Path of [`Self::input_file`].
    pub input_file_path: PathBuf,
    /// Path of [`Self::output_file`].
    pub output_file_path: PathBuf,
}

/// Compares two tuples and returns the [`Ordering`].
fn cmp_tuples(t1: &[Value], t2: &[Value]) -> Ordering {
    for (a, b) in t1.iter().zip(t2) {
        match a.partial_cmp(b) {
            Some(ordering) => {
                if ordering != Ordering::Equal {
                    return ordering;
                }
            }
            None => {
                if mem::discriminant(a) != mem::discriminant(b) {
                    unreachable!("it should be impossible to run into type errors at this point: cmp() {a} against {b}");
                }
            }
        }
    }

    Ordering::Equal
}

impl<F> Sort<F> {
    /// Sorts the tuples in [`Self::output_page`].
    fn sort_output_buffer(&mut self) {
        self.output_buffer
            .tuples
            .make_contiguous()
            .sort_by(|t1, t2| cmp_tuples(&t1[self.schema.len()..], &t2[self.schema.len()..]));
    }
}

// TODO: Requires defining the struct as Sort<F: FileOps>.
// impl<F: FileOps> Drop for Sort<F> {
//     fn drop(&mut self) {
//         self.drop_files();
//     }
// }

impl<F: FileOps> Sort<F> {
    /// Removes the files used by this [`Sort`] instance.
    fn drop_files(&mut self) -> io::Result<()> {
        if let Some(input_file) = self.input_file.take() {
            drop(input_file);
            F::remove(&self.input_file_path)?;
        }

        if let Some(output_file) = self.output_file.take() {
            drop(output_file);
            F::remove(&self.output_file_path)?;
        }

        Ok(())
    }
}

impl<F: Seek + Read + Write + FileOps> Sort<F> {
    /// Iterative implementation of the K-way external merge sort algorithm
    /// described in the documentation of [`Sort`].
    fn sort(&mut self) -> Result<(), DbError> {
        // Mem only sorting, didn't need files. Early return wishing that
        // everything was as simple as this.
        if self.source.reader.is_none() {
            self.output_buffer = mem::replace(&mut self.source.mem_buf, TupleBuffer::empty());
            self.sort_output_buffer();

            return Ok(());
        }

        // We need files to sort.
        let input_file_name = format!("{}.mkdb.sort", &self.source as *const _ as usize);
        self.input_file_path = self.work_dir.join(input_file_name);
        self.input_file = Some(F::create(&self.input_file_path)?);

        let output_file_name = format!("{}.mkdb.sort", &self.output_buffer as *const _ as usize);
        self.output_file_path = self.work_dir.join(output_file_name);
        self.output_file = Some(F::create(&self.output_file_path)?);

        // Figure out the page size.
        self.page_size = std::cmp::max(
            TupleBuffer::page_size_needed_for(self.source.mem_buf.largest_tuple_size),
            self.page_size,
        );

        // "Pass 0" or "1 page runs". Sort pages in memory and write them to the
        // "input" file.
        //
        // TODO: We don't actually need a "1 page run" here, we can fill all
        // the input buffers, sort them individually and merge them to skip
        // not only "Pass 0" but also "Pass 1". It's not hard to do but requires
        // figuring out exactly how to extract the "merge" behaviour written
        // below into its own function considering that the first pass we would
        // do here doesn't need to fill the input buffers from a file but
        // rather from the buffered iterator source.
        self.output_buffer = TupleBuffer::new(self.page_size, self.sort_schema.clone(), false);

        let mut input_pages = 0;
        while let Some(tuple) = self.source.try_next()? {
            // Write output page if full.
            if !self.output_buffer.can_fit(&tuple) {
                self.sort_output_buffer();
                self.output_buffer
                    .write_to(self.input_file.as_mut().unwrap())?;
                self.output_buffer.clear();
                input_pages += 1;
            }

            self.output_buffer.push(tuple);
        }

        // Last page.
        if !self.output_buffer.is_empty() {
            self.sort_output_buffer();
            self.output_buffer
                .write_to(self.input_file.as_mut().unwrap())?;
            self.output_buffer.clear();
            input_pages += 1;
        }

        // Pass 0 completed, 1 page runs sort algorithm done. Now do the "merge"
        // runs till fully sorted.
        let mut page_runs = self.input_buffers;

        let mut input_buffers = Vec::from_iter(
            std::iter::repeat_with(|| {
                TupleBuffer::new(self.page_size, self.sort_schema.clone(), false)
            })
            .take(self.input_buffers),
        );

        let mut cursors = vec![0; input_buffers.len()];

        // page_runs / input_buffers is the number of pages we processed in the
        // previous iteration. If we processed all the pages then we won't go
        // into another iteration. This is basically a hack for do while loops,
        // it saves us from writing an "if break" block 100 lines later.
        while page_runs / self.input_buffers < input_pages {
            // Number of output pages.
            let mut output_pages = 0;

            // Beginning of the current "chunk" or "segment". If we do 2 page
            // runs then this is gonna be 0, 2, 4... If we do 4 page runs it's
            // gonna be 0, 4, 8... etc
            let mut run = 0;

            // Loops through the page segments until we reach the end.
            while run < input_pages {
                // Initialize the cursors and load the first page for each
                // buffer.
                for (i, (input_buffer, cursor)) in
                    input_buffers.iter_mut().zip(cursors.iter_mut()).enumerate()
                {
                    *cursor = run + page_runs / self.input_buffers * i;
                    if *cursor < input_pages {
                        input_buffer.read_page(self.input_file.as_mut().unwrap(), *cursor)?;
                        *cursor += 1;
                    }
                }

                // Now start merging. When the output page fills up we write it
                // to a file. If one of the input pages is emptied we try to
                // load the next one. We repeat until there are no more pages
                // available in this run.
                while input_buffers.iter().any(|buffer| !buffer.is_empty()) {
                    // Find the input buffer that contains the smallest tuple.
                    // That's gonna be the next tuple that we push into the
                    // output buffer.
                    let mut min = input_buffers
                        .iter()
                        .position(|buffer| !buffer.is_empty())
                        .unwrap();

                    for (i, input_buffer) in (min + 1..).zip(&input_buffers[min + 1..]) {
                        if input_buffer.is_empty() {
                            continue;
                        }

                        let cmp = cmp_tuples(
                            &input_buffers[i][0][self.schema.len()..],
                            &input_buffers[min][0][self.schema.len()..],
                        );

                        if cmp == Ordering::Less {
                            min = i;
                        }
                    }

                    // Remove the tuple from the input page.
                    let tuple = input_buffers[min].pop_front().unwrap();

                    // Now check for empty pages. Load the next one if there
                    // are more pages in the current run.
                    for (i, (input_buffer, cursor)) in
                        input_buffers.iter_mut().zip(cursors.iter_mut()).enumerate()
                    {
                        if input_buffer.is_empty()
                            && *cursor < input_pages
                            && *cursor < run + page_runs / self.input_buffers * (i + 1)
                        {
                            input_buffer.read_page(self.input_file.as_mut().unwrap(), *cursor)?;
                            *cursor += 1;
                        }
                    }

                    // Write the output page if full. Otherwise just push the
                    // tuple.
                    if !self.output_buffer.can_fit(&tuple) {
                        self.output_buffer
                            .write_to(self.output_file.as_mut().unwrap())?;
                        self.output_buffer.clear();
                        output_pages += 1;
                    }

                    self.output_buffer.push(tuple);
                }

                // Write last page.
                if !self.output_buffer.is_empty() {
                    self.output_buffer
                        .write_to(self.output_file.as_mut().unwrap())?;
                    self.output_buffer.clear();
                    output_pages += 1;
                }

                // Now move to the next segment and repeat.
                run += page_runs;
            }

            // Now swap the files. Previous output becomes the input for the
            // next pass and the previous input becomes the output.
            let mut input_file = self.input_file.take().unwrap();
            let output_file = self.output_file.take().unwrap();

            input_file.truncate()?;

            self.input_file = Some(output_file);
            self.output_file = Some(input_file);

            page_runs *= self.input_buffers;
            input_pages = output_pages;
        }

        // Put the cursor back to the beginning for reading.
        self.input_file.as_mut().unwrap().rewind()?;

        // Drop the output file.
        drop(self.output_file.take());
        F::remove(&self.output_file_path)?;

        Ok(())
    }

    fn try_next(&mut self) -> Result<Option<Tuple>, DbError> {
        if !self.sorted {
            self.source.collect()?;
            self.sort()?;
            self.sorted = true;
        }

        if self.output_buffer.is_empty() {
            if let Some(input_file) = self.input_file.as_mut() {
                if let Err(e) = self.output_buffer.read_from(input_file) {
                    if e.kind() == io::ErrorKind::UnexpectedEof {
                        self.drop_files()?;
                    } else {
                        return Err(e.into());
                    }
                }
            }
        }

        // Remove sort keys when returning to the next plan node.
        Ok(self.output_buffer.pop_front().map(|mut tuple| {
            tuple.drain(self.schema.len()..);
            tuple
        }))
    }
}

// TODO: Some tests would be nice here. We can use the [`Values`] plan as a base
// for mocks that return any tuples we want.
