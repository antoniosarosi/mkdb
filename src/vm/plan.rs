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
//! Another important detail which makes the code here more complicated is that
//! some plans cannot work with a single tuple, they need all the tuples in
//! order to execute their code. One example is the [`Sort`] plan which needs
//! all the tuples before it can sort them. Other examples are the [`Update`] or
//! [`Delete`] plans which cannot make any changes to the underlying BTree until
//! the scan plan that is reading tuples from the BTree has finished reading.
//! That's because the scan plan holds an internal cursor and updating or
//! deleting from the BTree would invalidate that cursor.
//!
//! So, in order to deal with such cases, there's a special type of iterator
//! which is the [`Collect`]. The [`Collect`] contains an in-memory
//! buffer of configurable size that is written to a file once it fills up.
//! That way the [`Collect`] can collect as many tuples as necessary
//! without memory concerns. Once all the tuples are collected, they are
//! returned one by one just like any other normal iterator would return them.
use std::{
    cell::RefCell,
    cmp::Ordering,
    collections::{HashMap, VecDeque},
    fmt::{self, Debug, Display},
    io::{self, BufRead, BufReader, Read, Seek, Write},
    mem,
    ops::{Bound, Index, RangeBounds},
    path::{Path, PathBuf},
    rc::Rc,
};

use crate::{
    db::{DbError, Relation, Schema, SqlError, TableMetadata},
    paging::{
        io::FileOps,
        pager::{PageNumber, Pager},
    },
    sql::statement::{join, Assignment, Expression, Value},
    storage::{
        reassemble_payload, tuple, BTree, BTreeKeyComparator, BytesCmp, Cursor, FixedSizeMemCmp,
    },
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
#[derive(Debug, PartialEq)]
pub(crate) enum Plan<F> {
    /// Runs a sequential scan on a table BTree and returns the rows one by one.
    SeqScan(SeqScan<F>),
    /// Exact match for expressions like `SELECT * FROM table WHERE id = 5`.
    ExactMatch(ExactMatch<F>),
    /// Range scan for table BTrees or index BTrees.
    RangeScan(RangeScan<F>),
    /// Uses primary keys or row IDs to scan a table BTree.
    KeyScan(KeyScan<F>),
    /// Multi-index or multi-range scan.
    LogicalOrScan(LogicalOrScan<F>),
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
    /// Executes `ORDER BY` clauses or any other internal sorting.
    Sort(Sort<F>),
    /// Helper for the main [`Plan::Sort`] plan.
    SortKeysGen(SortKeysGen<F>),
    /// Helper for various plans.
    Collect(Collect<F>),
}

// TODO: As mentioned at [`crate::paging::pager::get_as`], we could also use
// [`enum_dispatch`](https://docs.rs/enum_dispatch/) here to automate the match
// statement or switch to Box<dyn Iterator<Item = Result<Tuple, DbError>>> but
// that's even more verbose than this and requires F: 'static everywhere. We
// also woudn't know the type of a plan because dyn Trait doesn't have a tag. So
// match it for now :)
impl<F: Seek + Read + Write + FileOps> Plan<F> {
    pub fn try_next(&mut self) -> Result<Option<Tuple>, DbError> {
        match self {
            Self::SeqScan(seq_scan) => seq_scan.try_next(),
            Self::ExactMatch(exact_match) => exact_match.try_next(),
            Self::RangeScan(range_scan) => range_scan.try_next(),
            Self::KeyScan(index_scan) => index_scan.try_next(),
            Self::LogicalOrScan(or_scan) => or_scan.try_next(),
            Self::Values(values) => values.try_next(),
            Self::Filter(filter) => filter.try_next(),
            Self::Project(project) => project.try_next(),
            Self::Insert(insert) => insert.try_next(),
            Self::Update(update) => update.try_next(),
            Self::Delete(delete) => delete.try_next(),
            Self::Sort(sort) => sort.try_next(),
            Self::SortKeysGen(sort_keys_gen) => sort_keys_gen.try_next(),
            Self::Collect(collect) => collect.try_next(),
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
    /// This is the schema of the top level plan.
    pub fn schema(&self) -> Option<Schema> {
        let schema = match self {
            Self::Project(project) => &project.output_schema,
            Self::KeyScan(index_scan) => &index_scan.table.schema,
            Self::SeqScan(seq_scan) => &seq_scan.table.schema,
            Self::RangeScan(range_scan) => &range_scan.schema,
            Self::ExactMatch(exact_match) => exact_match.relation.schema(),
            Self::Sort(sort) => &sort.collection.schema,
            Self::Filter(filter) => return filter.source.schema(),

            Self::LogicalOrScan(or_scan) => return or_scan.scans[0].schema().to_owned(),
            _ => return None,
        };

        Some(schema.to_owned())
    }

    /// Returns the child node of this plan.
    pub fn child(&self) -> Option<&Self> {
        Some(match self {
            Self::KeyScan(index_scan) => &index_scan.source,
            Self::Filter(filter) => &filter.source,
            Self::Project(project) => &project.source,
            Self::Insert(insert) => &insert.source,
            Self::Update(update) => &update.source,
            Self::Delete(delete) => &delete.source,
            Self::Sort(sort) => &sort.collection.source,
            Self::SortKeysGen(sort_keys_gen) => &sort_keys_gen.source,
            Self::Collect(collect) => &collect.source,
            _ => return None,
        })
    }

    /// String representation of a plan.
    pub fn display(&self) -> String {
        let prefix = "-> ";

        // TODO: Can be optimized with write! macro and fmt::Write. Too lazy to
        // change it, doesn't matter for now.
        let display = match self {
            Self::SeqScan(seq_scan) => format!("{seq_scan}"),
            Self::ExactMatch(exact_match) => format!("{exact_match}"),
            Self::RangeScan(range_scan) => format!("{range_scan}"),
            Self::KeyScan(index_scan) => format!("{index_scan}"),
            Self::LogicalOrScan(or_scan) => format!("{or_scan}"),
            Self::Values(values) => format!("{values}"),
            Self::Filter(filter) => format!("{filter}"),
            Self::Project(project) => format!("{project}"),
            Self::Insert(insert) => format!("{insert}"),
            Self::Update(update) => format!("{update}"),
            Self::Delete(delete) => format!("{delete}"),
            Self::Sort(sort) => format!("{sort}"),
            Self::SortKeysGen(sort_keys_gen) => format!("{sort_keys_gen}"),
            Self::Collect(collect) => format!("{collect}"),
        };

        format!("{prefix}{display}")
    }
}

impl<F> Display for Plan<F> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut plans = vec![self.display()];

        let mut node = self;
        while let Some(child) = node.child() {
            plans.push(child.display());
            node = child;
        }

        writeln!(f, "{}", plans.pop().unwrap())?;
        while let Some(plan) = plans.pop() {
            writeln!(f, "{plan}")?;
        }

        Ok(())
    }
}

/// Sequential scan plan.
///
/// This is not 100% sequential because we're not using a B+Tree but a normal
/// BTree.
///
/// See [`Cursor::try_next`] for details.
#[derive(Debug, PartialEq)]
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

impl<F> Display for SeqScan<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SeqScan on table '{}'", self.table.name)
    }
}

/// Exact key match for expressions like `SELECT * FROM table WHERE id = 5;`.
#[derive(Debug, PartialEq)]
pub(crate) struct ExactMatch<F> {
    pub relation: Relation,
    pub key: Vec<u8>,
    pub expr: Expression,
    pub pager: Rc<RefCell<Pager<F>>>,
    pub done: bool,
    pub emit_table_key_only: bool,
}

impl<F: Seek + Read + Write + FileOps> ExactMatch<F> {
    fn try_next(&mut self) -> Result<Option<Tuple>, DbError> {
        if self.done {
            return Ok(None);
        }

        // Only runs once.
        self.done = true;

        let mut pager = self.pager.borrow_mut();
        let mut btree = BTree::new(&mut pager, self.relation.root(), self.relation.comparator());

        let Some(entry) = btree.get(&self.key)? else {
            return Ok(None);
        };

        let mut tuple = tuple::deserialize(entry.as_ref(), self.relation.schema());

        if self.emit_table_key_only {
            let table_key_index = self.relation.index_of_table_key();
            tuple.drain(table_key_index + 1..);
            tuple.drain(..table_key_index);
        }

        Ok(Some(tuple))
    }
}

impl<F> Display for ExactMatch<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ExactMatch ({}) on {} '{}'",
            self.expr,
            self.relation.kind(),
            self.relation.name()
        )
    }
}

/// Parameters for constructing [`RangeScan`] objects.
pub(crate) struct RangeScanConfig<F> {
    pub relation: Relation,
    pub pager: Rc<RefCell<Pager<F>>>,
    pub range: (Bound<Vec<u8>>, Bound<Vec<u8>>),
    pub expr: Expression,
    pub emit_table_key_only: bool,
}

/// BTree range scan.
///
/// This struct serves 2 purposes:
///
/// 1. Scan auto-indexed BTree tables.
/// 2. Scan external index BTrees.
///
/// It's called a "range" scan because it's used to partially scan BTrees.
/// Queries like the following:
///
/// ```sql
/// SELECT * FROM users WHERE id > 500;
/// ```
///
/// only need to scan the BTree starting from a key >= 500. The [`RangeScan`]
/// positions its cursor at the first key that is greater than 500 and starts
/// returning tuples from there.
///
/// # Range Scan Algorithm
///
/// Each [`RangeScan`] object will receive an implementation of [`RangeBounds`]
/// and will use [`RangeBounds::start_bound`] to position the cursor initially
/// then it will use [`RangeBounds::end_bound`] to know when to stop.
#[derive(Debug, PartialEq)]
pub(crate) struct RangeScan<F> {
    pub emit_table_key_only: bool,
    key_index: usize,
    relation: Relation,
    root: PageNumber,
    schema: Schema,
    pager: Rc<RefCell<Pager<F>>>,
    range: (Bound<Vec<u8>>, Bound<Vec<u8>>),
    comparator: BTreeKeyComparator,
    expr: Expression,
    cursor: Cursor,
    init: bool,
    done: bool,
}

impl<F> From<RangeScanConfig<F>> for RangeScan<F> {
    fn from(
        RangeScanConfig {
            relation,
            emit_table_key_only,
            pager,
            range,
            expr,
        }: RangeScanConfig<F>,
    ) -> Self {
        Self {
            schema: relation.schema().clone(),
            comparator: relation.comparator(),
            root: relation.root(),
            cursor: Cursor::new(relation.root(), 0),
            key_index: relation.index_of_table_key(),
            emit_table_key_only,
            expr,
            pager,
            range,
            relation,
            done: false,
            init: false,
        }
    }
}

impl<F: Seek + Read + Write + FileOps> RangeScan<F> {
    /// Positions the cursor.
    fn init(&mut self) -> io::Result<()> {
        let mut pager = self.pager.borrow_mut();

        let key = match self.range.start_bound() {
            Bound::Unbounded => return Ok(()),
            Bound::Excluded(key) => key,
            Bound::Included(key) => key,
        };

        let mut descent = Vec::new();
        let mut btree = BTree::new(&mut pager, self.root, self.comparator);
        let search = btree.search(self.root, key, &mut descent)?;

        match search.index {
            // Found exact match. Easy case.
            Ok(slot) => {
                self.cursor = Cursor::initialized(search.page, slot, descent);

                // Skip it.
                if let Bound::Excluded(_) = self.range.start_bound() {
                    self.cursor.try_next(&mut pager)?;
                }
            }

            // We didn't find the exact key we were looking for. This index is
            // the index where the key "should" be located. If we were looking
            // for key 2 in this array:
            //
            // [1, 3, 5, 7]
            //
            // "slot" would be 1. Index 1 points to 3 in the array, which means
            // we are already located at a key that is >= 1.
            //
            // On the other hand, if we were looking for key 8, "slot" would be
            // 4 which is out of bounds. That means we have to move to the next
            // page in order to find the first key >= 8. Since that's not easy
            // at all we'll position the cursor at the last key in the page and
            // consume that key, allowing the cursor to compute where the next
            // one is.
            Err(slot) => {
                if slot >= pager.get(search.page)?.len() {
                    self.cursor = Cursor::initialized(search.page, slot.saturating_sub(1), descent);
                    self.cursor.try_next(&mut pager)?;
                } else {
                    self.cursor = Cursor::initialized(search.page, slot, descent);
                }
            }
        };

        Ok(())
    }

    fn try_next(&mut self) -> Result<Option<Tuple>, DbError> {
        if self.done {
            return Ok(None);
        }

        if !self.init {
            self.init()?;
            self.init = true;
        }

        let mut pager = self.pager.borrow_mut();

        let Some((page, slot)) = self.cursor.try_next(&mut pager)? else {
            self.done = true;
            return Ok(None);
        };

        let entry = reassemble_payload(&mut pager, page, slot)?;

        let bound = self.range.end_bound();
        if let Bound::Excluded(key) | Bound::Included(key) = bound {
            let ordering = self.comparator.bytes_cmp(entry.as_ref(), key);
            if let Ordering::Equal | Ordering::Greater = ordering {
                self.done = true;
                if matches!(bound, Bound::Excluded(_))
                    || matches!(bound, Bound::Included(_)) && ordering == Ordering::Greater
                {
                    return Ok(None);
                }
            }
        }

        let mut tuple = tuple::deserialize(entry.as_ref(), &self.schema);

        if self.emit_table_key_only {
            tuple.drain(self.key_index + 1..);
            tuple.drain(..self.key_index);
        }

        Ok(Some(tuple))
    }
}

impl<F> Display for RangeScan<F> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "RangeScan ({}) on {} '{}'",
            self.expr,
            self.relation.kind(),
            self.relation.name()
        )
    }
}

/// [`KeyScan`] uses an indexed column to retrieve data from a table.
///
/// An index is a BTree that maps a column value to the primary key or row ID of
/// the table. It makes it easy to find all the primary keys we need without
/// scanning sequentially.
///
/// # Index Scan Algorithm
///
/// The index BTree maps keys to primary keys / row IDs like this:
///
/// ```text
///                        +--------------+
///                        | "Carla" -> 5 |
///                        +--------------+
///                          /           \
/// +-------------+------------+      +--------------+------------+
/// | "Alex" -> 3 | "Bob" -> 1 |      | "David" -> 4 | "Fia" -> 2 |
/// +-------------+------------+      +--------------+------------+
/// ```
///
/// Then the table BTree stores rows sorted by their primary key / Row ID:
///
/// ```text
///                           +------------+
///                           | 3 -> "Bob" |
///                           +------------+
///                            /          \
/// +------------+--------------+      +--------------+-------------+
/// | 1 -> "Fia" | 2 -> "David" |      | 4 -> "Carla" | 5 -> "Alex" |
/// +------------+--------------+      +--------------+-------------+
/// ```
///
/// The first step in the algorithm is sorting all the BTree entries that
/// we need by Row ID. Imagine the user sent a query like this one:
///
/// ```sql
/// SELECT * FROM users WHERE name <= "Carla";
/// ```
///
/// The [`RangeScan`] helper will return these tuples:
///
/// ```text
/// +--------------+
/// | "Alex" -> 3  |
/// +--------------+
/// | "Bob" -> 1   |
/// +--------------+
/// | "Carla" -> 5 |
/// +--------------+
/// ```
///
/// These tuples will be buffered by [`Collect`] and stored in a file if
/// necessary. Remember that we can never assume that something fits in memory
/// when writing a database.
///
/// After we have all the tuples the [`Sort`] plan takes care of sorting them by
/// Row ID and will start returning the following results:
///
/// ```text
/// +--------------+
/// | "Bob" -> 1   |
/// +--------------+
/// | "Alex" -> 3  |
/// +--------------+
/// | "Carla" -> 5 |
/// +--------------+
/// ```
///
/// Sorting the tuples by primary key saves us from doing completely random IO,
/// so scanning the table BTree will be a little bit more predictable.
#[derive(Debug, PartialEq)]
pub(crate) struct KeyScan<F> {
    pub comparator: FixedSizeMemCmp,
    pub table: TableMetadata,
    pub pager: Rc<RefCell<Pager<F>>>,
    pub source: Box<Plan<F>>,
}

impl<F: Seek + Read + Write + FileOps> KeyScan<F> {
    fn try_next(&mut self) -> Result<Option<Tuple>, DbError> {
        let Some(key_only_tuple) = self.source.try_next()? else {
            return Ok(None);
        };

        debug_assert!(key_only_tuple.len() == 1, "needs key only");

        let mut pager = self.pager.borrow_mut();

        let mut btree = BTree::new(&mut pager, self.table.root, self.comparator);

        let table_entry = btree
            .get(&tuple::serialize_key(
                &self.table.schema.columns[0].data_type,
                &key_only_tuple[0],
            ))?
            .ok_or_else(|| {
                DbError::Corrupted(format!(
                    "attempt to scan key {key_only_tuple:?} that doesn't exist on table {} at root {}",
                    self.table.name, self.table.root,
                ))
            })?;

        Ok(Some(tuple::deserialize(
            table_entry.as_ref(),
            &self.table.schema,
        )))
    }
}

impl<F> Display for KeyScan<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "KeyScan ({}) on table '{}'",
            self.table.schema.columns[0].name, self.table.name
        )
    }
}

/// [`LogicalOrScan`] returns tuples from multiple indexes, multiple ranges
/// within the same index or a combination of both.
///
/// Ranges should be sorted for efficiency.
///
/// TODO: There's an optimization we can do here when scanning by primary key
/// and external index keys at the same time. Instead of collecting all the keys
/// and sorting them, then using [`KeyScan`] (which is what we do now), we could
/// collect only external index keys, sort them, and switch between range key
/// scanning and external index key scanning.
///
/// For example, imagine something like this:
///
/// ```text
/// LogialOrScan
///     -> RangeScan (id < 5) on table "users"
///     -> RangeScan (id > 100 AND id < 200) on table "users"
///
///     -> RangeScan (email < 'test@test.com') on index "email"
///     -> RangeScan (something < 'value') on index "some_index"
/// ```
///
/// We could collect and sort all the keys returned by the email and some_index
/// BTrees, discard all the keys that fall within the current table range, scan
/// the table range until it's done and repeat. When keys returned by external
/// indexes are "less than the starting bound of the current table range" then
/// we scan those until we reach the table range again.
///
/// In the example above, imagine that external indexes returned the keys
/// `[20, 30, 40, 500, 600]`. We start by scanning the range `..5` on the users
/// BTree since the key 20 is after the range. Once we're done with range `..5`
/// we have to scan range `100..200`, but we can pop keys `20, 30, 40` from the
/// index queue and scan that before. Then scan the range `100..200` and finally
/// keys `500, 600`.
///
/// We can use [`Peek`] to check the value of a plan without consuming it. This
/// algorithm probably sounds more complicated than it actually is, it shouldn't
/// be too hard to implement.
#[derive(Debug, PartialEq)]
pub(crate) struct LogicalOrScan<F> {
    pub scans: VecDeque<Plan<F>>,
}

impl<F: Seek + Read + Write + FileOps> LogicalOrScan<F> {
    fn try_next(&mut self) -> Result<Option<Tuple>, DbError> {
        let Some(mut scan) = self.scans.front_mut() else {
            return Ok(None);
        };

        // Optimistic approach, suppose the current scan has more tuples to
        // return.
        let mut tuple = scan.try_next()?;

        // If it's not the case then find the next scan with tuples available.
        while tuple.is_none() {
            self.scans.pop_front();
            let Some(next) = self.scans.front_mut() else {
                return Ok(None);
            };
            scan = next;
            tuple = scan.try_next()?;
        }

        Ok(Some(tuple.unwrap()))
    }
}

impl<F> Display for LogicalOrScan<F> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "LogicalOrScan")?;

        for scan in &self.scans {
            write!(f, "\n    {}", scan.display())?;
        }

        Ok(())
    }
}

/// Raw values from `INSERT INTO table (c1, c2) VALUES (v1, v2)`.
///
/// This supports multiple values but the parser does not currently parse
/// `INSERT` statements with multiple values.
#[derive(Debug, PartialEq)]
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

impl Display for Values {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Values ({})", join(&self.values[0], ", "))
    }
}

/// Applies a filter [`Expression`] to its source returning only tuples that
/// evaluate to `true`.
///
/// Used for `WHERE` clauses in `SELECT`, `DELETE` and `UPDATE` statements.
#[derive(Debug, PartialEq)]
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

impl<F> Display for Filter<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Filter ({})", self.filter)
    }
}

/// Applies a projection to a tuple.
///
/// A "projection" is a relation algebra unary operation which, in simple words,
/// "selects" columns from a row. For example:
///
/// ```sql
/// CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(255), age INT);
///
/// SELECT id, age FROM users;
/// ```
///
/// The projection in this case would be the "id" and "age columns". The name
/// is discarded.
#[derive(Debug, PartialEq)]
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

impl<F> Display for Project<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Project ({})", join(&self.projection, ", "))
    }
}

/// Inserts data into a table and upates indexes.
#[derive(Debug, PartialEq)]
pub(crate) struct Insert<F> {
    pub pager: Rc<RefCell<Pager<F>>>,
    pub source: Box<Plan<F>>,
    pub table: TableMetadata,
    pub comparator: FixedSizeMemCmp,
}

impl<F: Seek + Read + Write + FileOps> Insert<F> {
    fn try_next(&mut self) -> Result<Option<Tuple>, DbError> {
        let Some(mut tuple) = self.source.try_next()? else {
            return Ok(None);
        };

        let mut pager = self.pager.borrow_mut();

        // TODO: We know that all tables use integers as BTree keys whereas
        // indexes can use either strings or integers. Having two types of
        // BTrees introduces code bloat but at the same time using dynamic
        // dispatch for a type that we alrady know doesn't make sense.
        BTree::new(&mut pager, self.table.root, self.comparator)
            .try_insert(tuple::serialize(&self.table.schema, &tuple))?
            .map_err(|_| SqlError::DuplicatedKey(tuple.swap_remove(0)))?;

        for index in &self.table.indexes {
            let col = self
                .table
                .schema
                .index_of(&index.column.name)
                .ok_or(DbError::Corrupted(format!(
                    "index column '{}' not found on table {} schema: {:?}",
                    index.column.name, self.table.name, self.table.schema,
                )))?;

            // This one's dynamic, we can either use Box<dyn BytesCmp> or the
            // BTreeKeyComparator enum which dispatches using jump tables
            // instead of VTables. The enum also doesn't need an additional Box
            // allocation.
            let comparator = BTreeKeyComparator::from(&index.column.data_type);

            BTree::new(&mut pager, index.root, comparator)
                .try_insert(tuple::serialize(&index.schema, [&tuple[col], &tuple[0]]))?
                .map_err(|_| SqlError::DuplicatedKey(tuple.swap_remove(col)))?;
        }

        Ok(Some(vec![]))
    }
}

impl<F> Display for Insert<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Insert on table '{}'", self.table.name)
    }
}

/// Assigns values to columns and updates indexes in the process.
#[derive(Debug, PartialEq)]
pub(crate) struct Update<F> {
    pub table: TableMetadata,
    pub assignments: Vec<Assignment>,
    pub pager: Rc<RefCell<Pager<F>>>,
    pub source: Box<Plan<F>>,
    pub comparator: FixedSizeMemCmp,
}

impl<F: Seek + Read + Write + FileOps> Update<F> {
    fn try_next(&mut self) -> Result<Option<Tuple>, DbError> {
        let Some(mut tuple) = self.source.try_next()? else {
            return Ok(None);
        };

        // Col Name -> (old value, new value index)
        let mut updated_cols = HashMap::new();

        for assignment in &self.assignments {
            let col =
                self.table
                    .schema
                    .index_of(&assignment.identifier)
                    .ok_or(DbError::Corrupted(format!(
                        "column {} not found in table schema {:?}",
                        assignment.identifier, self.table
                    )))?;

            // Compute updated column value.
            let new_value = vm::resolve_expression(&tuple, &self.table.schema, &assignment.value)?;

            // If the value did not change we'll skip this column.
            if new_value != tuple[col] {
                let old_value = mem::replace(&mut tuple[col], new_value);
                updated_cols.insert(assignment.identifier.clone(), (old_value, col));
            }
        }

        let mut pager = self.pager.borrow_mut();
        let mut btree = BTree::new(&mut pager, self.table.root, self.comparator);

        // Updated tuple.
        let updated_entry = tuple::serialize(&self.table.schema, &tuple);

        // If the primary key changes we have to remove the old entry from the
        // BTree. Otherwise we do a normal update where we override the existing
        // entry.
        if let Some((old_pk, new_pk)) = updated_cols.get(&self.table.schema.columns[0].name) {
            btree
                .try_insert(updated_entry)?
                .map_err(|_| SqlError::DuplicatedKey(tuple.swap_remove(0)))?;
            btree.remove(&tuple::serialize_key(
                &self.table.schema.columns[0].data_type,
                old_pk,
            ))?;
        } else {
            btree.insert(updated_entry)?;
        }

        for index in &self.table.indexes {
            let mut btree = BTree::new(
                &mut pager,
                index.root,
                BTreeKeyComparator::from(&index.column.data_type),
            );

            // Three cases to consider:
            //
            // 1. The value of the indexed column has changed. Remove the
            // previous one and insert the new one. If the primary key we're
            // pointing to has changed then this case covers that as well.
            //
            // 2. Only the primary key has changed while the indexed column
            // value remains the same. In that case do a normal update
            // overriding the previous index entry.
            //
            // 3. Nothing has change, move to the next iteration.
            if let Some((old_key, new_key)) = updated_cols.get(&index.column.name) {
                btree
                    .try_insert(tuple::serialize(&index.schema, [
                        &tuple[*new_key],
                        &tuple[0],
                    ]))?
                    .map_err(|_| SqlError::DuplicatedKey(tuple.swap_remove(*new_key)))?;

                btree.remove(&tuple::serialize_key(&index.column.data_type, old_key))?;
            } else if updated_cols.contains_key(&self.table.schema.columns[0].name) {
                let index_col = self.table.schema.index_of(&index.column.name).unwrap();
                btree.insert(tuple::serialize(&index.schema, [
                    &tuple[index_col],
                    &tuple[0],
                ]))?;
            }
        }

        Ok(Some(vec![]))
    }
}

impl<F> Display for Update<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Update ({}) on table '{}'",
            join(&self.assignments, ", "),
            self.table.name
        )
    }
}

/// Removes values from a table BTree and from all the necessary index BTrees.
#[derive(Debug, PartialEq)]
pub(crate) struct Delete<F> {
    pub table: TableMetadata,
    pub comparator: FixedSizeMemCmp,
    pub pager: Rc<RefCell<Pager<F>>>,
    pub source: Box<Plan<F>>,
}

impl<F: Seek + Read + Write + FileOps> Delete<F> {
    fn try_next(&mut self) -> Result<Option<Tuple>, DbError> {
        let Some(tuple) = self.source.try_next()? else {
            return Ok(None);
        };

        let mut pager = self.pager.borrow_mut();
        let mut btree = BTree::new(&mut pager, self.table.root, self.comparator);

        btree.remove(&tuple::serialize_key(
            &self.table.schema.columns[0].data_type,
            &tuple[0],
        ))?;

        for index in &self.table.indexes {
            let col = self.table.schema.index_of(&index.column.name).unwrap();
            let key = tuple::serialize_key(&index.column.data_type, &tuple[col]);

            let mut btree = BTree::new(
                &mut pager,
                index.root,
                BTreeKeyComparator::from(&index.column.data_type),
            );

            btree.remove(&key)?;
        }

        Ok(Some(vec![]))
    }
}

impl<F> Display for Delete<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Delete from table '{}'", self.table.name)
    }
}

/// See [`TupleBuffer`].
const TUPLE_PAGE_HEADER_SIZE: usize = mem::size_of::<u32>();

/// In-memory tuple buffer.
///
/// This structure stores [`Tuple`] instances (AKA rows) and can be written to
/// a file once it reaches a certain preconfigured threshold. The
/// [`TupleBuffer`] is used by the [`Collect`] and [`Sort`] plans to store
/// large query sets in the file system for queries like `SELECT * FROM table`
/// which might not fit in RAM. Depending on the use case the tuple buffer can
/// be configured with two different modes: fixed size and packed.
///
/// # Fixed Size Format
///
/// Setting [`TupleBuffer::packed`] to false will cause the buffer to use a
/// "fixed size" page format when writing its contents to a file. This will
/// produce a file with multiple pages described by the following structure:
///
/// ```text
/// +-----------------+          <---+
/// |   Num Tuples    | 4 bytes      |
/// +-----------------+              |
/// |       ...       |              |
/// | Tuple 0 Content | Var Size     |
/// |       ...       |              |
/// +-----------------+              |
/// |       ...       |              |
/// | Tuple 1 Content | Var Size     |
/// |       ...       |              | PAGE SIZE
/// +-----------------+              |
/// |       ...       |              |
/// | Tuple 2 Content | Var Size     |
/// |       ...       |              |
/// +-----------------+              |
/// |       ...       |              |
/// |     PADDING     | Var Size     |
/// |       ...       |              |
/// +-----------------+          <---+
/// ```
///
/// Each page contains a header that encodes the number of tuples in the page
/// using a 32 bit little endian integer. The header is followed by N tuples
/// stored in the same format that the database uses, see [`tuple`] for details.
/// Storing the size of each individual tuple is not necessary because we
/// already have the schema of the table in memory which can be used to parse
/// variable size tuples. Finally, the page contains padding if necessary to
/// make it fixed size.
///
/// Fixed size pages are necessary for sorting large amounts of data. The page
/// size is dependant on the size of reassembled tuples that are not scattered
/// across overflow pages (see [`reassemble_payload`]), so the page size of the
/// sort file may not be the same as the page size used by the database. See the
/// documentation of [`Sort`] for more details on sorting.
///
/// # Packed Format
///
/// When [`TupleBuffer::packed`] is set to true in-memory tuples will be written
/// to the file in one single continuous sequence of bytes without any headers
/// or padding:
///
/// ```text
/// +-----------------+          <---+
/// |       ...       |              |
/// | Tuple 0 Content | Var Size     |
/// |       ...       |              |
/// +-----------------+              |
/// |       ...       |              |
/// | Tuple 1 Content | Var Size     | VARIABLE SIZE
/// |       ...       |              |
/// +-----------------+              |
/// |       ...       |              |
/// | Tuple 2 Content | Var Size     |
/// |       ...       |              |
/// +-----------------+          <---+
/// ```
///
/// This format is used by the [`Collect`] plan to simply store the
/// results of a large query like `SELECT * FROM table` in a file and then
/// "stream" the rows one by one. Again, we don't need any information about the
/// size of anything because the table [`Schema`] already tells us exactly how
/// to parse tuples back into [`Tuple`] structures.
///
/// The format of this file doesn't need to be complicated because it's just
/// used to temporarily store rows returned by a query while the program is
/// alive, if the program crashes the transaction that opened the query file
/// automatically fails and will be rolled back by the journal system when we
/// reboot. At that point the file no longer serves any purpose, it's not used
/// for recovery.
#[derive(Debug, PartialEq)]
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

    /// Packed or fixed size mode.
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
            current_size: if packed { 0 } else { TUPLE_PAGE_HEADER_SIZE },
            largest_tuple_size: 0,
            tuples: VecDeque::new(),
        }
    }

    /// Returns `true` if the given `tuple` can be appended to this buffer
    /// without incrementing its size past [`Self::page_size`].
    pub fn can_fit(&self, tuple: &Tuple) -> bool {
        self.current_size + tuple::size_of(tuple, &self.schema) <= self.page_size
    }

    /// Appends the given `tuple` to the buffer.
    ///
    /// It doesn't panic or return any error if the buffer overflows
    /// [`Self::page_size`], this function won't fail. This is useful because
    /// the [`Collect`] needs to be able to process tuples of any size even
    /// if they are larger than the maximum buffer size. The [`Sort`] plan has
    /// its own tricks to avoid working with tuples that wouldn't fit in the
    /// buffer.
    pub fn push(&mut self, tuple: Tuple) {
        let tuple_size = tuple::size_of(&tuple, &self.schema);

        if tuple_size > self.largest_tuple_size {
            self.largest_tuple_size = tuple_size;
        }

        self.current_size += tuple_size;
        self.tuples.push_back(tuple);
    }

    /// Removes the first tuple in this buffer and returns it.
    pub fn pop_front(&mut self) -> Option<Tuple> {
        self.tuples.pop_front().inspect(|tuple| {
            self.current_size -= tuple::size_of(tuple, &self.schema);
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
            TUPLE_PAGE_HEADER_SIZE
        };
    }

    /// Serializes this buffer into a byte array that can be written to a file.
    pub fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(self.page_size);

        // Page header.
        if !self.packed {
            buf.extend_from_slice(&(self.tuples.len() as u32).to_le_bytes());
        }

        // Tuples.
        for tuple in &self.tuples {
            buf.extend_from_slice(&tuple::serialize(&self.schema, tuple));
        }

        // Padding.
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
    pub fn read_from(&mut self, file: &mut impl Read) -> Result<(), DbError> {
        debug_assert!(
            self.is_empty() && !self.packed,
            "read_from() only works with fixed size empty buffers"
        );

        let mut buf = vec![0; self.page_size];
        file.read_exact(&mut buf)?;

        // This should only fail due to human introduced errors (wrong
        // paremeter, incorrect seek() calls to the file, etc). We just don't
        // want to panic here and crash the database.
        let number_of_tuples =
            u32::from_le_bytes(buf[..TUPLE_PAGE_HEADER_SIZE].try_into().map_err(|e| {
                DbError::Other(format!("error while reading query file header: {e}"))
            })?);

        let mut cursor = TUPLE_PAGE_HEADER_SIZE;

        for _ in 0..number_of_tuples {
            let tuple = tuple::deserialize(&buf[cursor..], &self.schema);
            cursor += tuple::size_of(&tuple, &self.schema);
            self.push(tuple);
        }

        Ok(())
    }

    /// Same as [`Self::read_from`] but positions the file cursor at the
    /// beginning of the given page number.
    pub fn read_page(&mut self, file: &mut (impl Seek + Read), page: usize) -> Result<(), DbError> {
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
/// [`Collect`] acts like a normal iterator that returns tuples one at a
/// time. If the collected tuples fit in memory there is no IO, so that's the
/// best case scenario.
///
/// # Collection Algorithm
///
/// We use [`TupleBuffer`] to store tuples in memory for as long as we can. Once
/// the [`TupleBuffer`] cannot fit more tuples we write its contents to a file
/// and repeat until the source returns no more tuples. By the end of the
/// operation we might have something like this:
///
/// ```text
///            Memory Buffer        File
///           +----+----+----+     +----+
/// Source -> | T8 | T7 | T6 | ->  | T1 |
///           +----+----+----+     +----+
///                                | T2 |
///                                +----+
///                                | T3 |
///                                +----+
///                                | T4 |
///                                +----+
///                                | T5 |
///                                +----+
/// ```
///
/// Now that we've collected everything we switch our behaviour from a buffered
/// writer to a buffered reader. To ensure that tuples are returned in the same
/// order they come from the source we have to read from the file first and
/// finally return whatever we got left in memory.
///
/// The collection file format is described in the documentation of
/// [`TupleBuffer`], but basically it only contains the tuples just like they
/// are stored in the database without any additional information such as their
/// size. We use [`Schema`] to parse tuples so we don't need to know their size,
/// we can just read sequentially from the file indefinitely until there are no
/// more bytes.
#[derive(Debug)]
pub(crate) struct Collect<F> {
    /// Tuple source. This is where we collect from.
    source: Box<Plan<F>>,
    /// Tuple schema.
    schema: Schema,
    /// `true` if [`Self::collect`] completed successfully.
    collected: bool,
    /// In-memory buffer that stores tuples from the source.
    mem_buf: TupleBuffer,
    /// File handle/descriptor in case we had to create the collection file.
    file: Option<F>,
    /// Buffered reader in case we created the file and have to read from it.
    reader: Option<BufReader<F>>,
    /// Path of the collection file.
    file_path: PathBuf,
    /// Working directory.
    work_dir: PathBuf,
}

impl<F> Display for Collect<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Collect ({})",
            join(self.schema.columns.iter().map(|col| &col.name), ", ")
        )
    }
}

// Can't derive because of the BufReader<F>.
impl<F: PartialEq> PartialEq for Collect<F> {
    fn eq(&self, other: &Self) -> bool {
        self.source == other.source && self.schema == other.schema
    }
}

impl<F: FileOps> Collect<F> {
    /// Drops the IO resource and deletes it from the file system.
    fn drop_file(&mut self) -> io::Result<()> {
        drop(self.file.take());
        drop(self.reader.take());
        F::remove(&self.file_path)
    }
}

// TODO: Requires defining the struct as BufferdIter<F: FileOps>
// impl<F: FileOps> Drop for Collect<F> {
//     fn drop(&mut self) {
//         if self.file.is_some() {
//             self.drop_file();
//         }
//     }
// }

/// Used to build [`Collect`] objects.
pub(crate) struct CollectConfig<F> {
    pub source: Box<Plan<F>>,
    pub schema: Schema,
    pub work_dir: PathBuf,
    pub mem_buf_size: usize,
}

impl<F> From<CollectConfig<F>> for Collect<F> {
    fn from(
        CollectConfig {
            source,
            schema,
            work_dir,
            mem_buf_size,
        }: CollectConfig<F>,
    ) -> Self {
        Self {
            source,
            mem_buf: TupleBuffer::new(mem_buf_size, schema.clone(), true),
            schema,
            collected: false,
            file_path: PathBuf::new(),
            work_dir,
            file: None,
            reader: None,
        }
    }
}

impl<F: Seek + Read + Write + FileOps> Collect<F> {
    /// Collects all the tuples from [`Self::source`].
    fn collect(&mut self) -> Result<(), DbError> {
        // Buffer tuples in-memory until we have no space left. At that point
        // create the file if it doesn't exist, write the buffer to disk and
        // repeat until there are no more tuples.
        while let Some(tuple) = self.source.try_next()? {
            if !self.mem_buf.can_fit(&tuple) {
                if self.file.is_none() {
                    let (file_path, file) = tmp_file(&self.work_dir, "mkdb.query")?;
                    self.file_path = file_path;
                    self.file = Some(file);
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
            self.reader = Some(BufReader::with_capacity(self.mem_buf.page_size, file));
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
                return Ok(Some(tuple::read_from(reader, &self.schema)?));
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

/// Same as [`std::iter::Peekable`] but fallible.
///
/// Maintains an intermediate buffer of capacity 1 that holds a [`Tuple`] until
/// it is consumed through [`Self::try_next`]. It is somewhat similar to
/// [`Collect`] but much simpler.
///
/// Not used anywhere for now but we can use it to implement the optimization
/// described in the documentation of [`LogicalOrScan`].
#[derive(Debug, PartialEq)]
pub(crate) struct Peek<F> {
    source: Box<Plan<F>>,
    tuple: Option<Tuple>,
}

impl<F: Seek + Read + Write + FileOps> Peek<F> {
    fn try_peek(&mut self) -> Result<Option<&Tuple>, DbError> {
        if self.tuple.is_none() {
            self.tuple = self.source.try_next()?;
        }

        Ok(self.tuple.as_ref())
    }

    fn try_next(&mut self) -> Result<Option<Tuple>, DbError> {
        self.try_peek()?;
        Ok(self.tuple.take())
    }
}

/// Generates sort keys.
///
/// This is a helper for the main [`Sort`] plan that basically evaluates the
/// `ORDER BY` expressions and appends the results to each tuple.
///
/// See the documentation of [`Sort`] for more details.
#[derive(Debug, PartialEq)]
pub(crate) struct SortKeysGen<F> {
    pub source: Box<Plan<F>>,
    pub schema: Schema,
    pub gen_exprs: Vec<Expression>,
}

impl<F: Seek + Read + Write + FileOps> SortKeysGen<F> {
    pub fn try_next(&mut self) -> Result<Option<Tuple>, DbError> {
        let Some(mut tuple) = self.source.try_next()? else {
            return Ok(None);
        };

        for expr in &self.gen_exprs {
            debug_assert!(
                !matches!(expr, Expression::Identifier(_)),
                "identifiers are not allowed here"
            );

            tuple.push(vm::resolve_expression(&tuple, &self.schema, expr)?);
        }

        Ok(Some(tuple))
    }
}

impl<F> Display for SortKeysGen<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SortKeysGen ({})", join(&self.gen_exprs, ", "))
    }
}

/// Used to build [`Sort`] objects.
///
/// Building objects ain't easy these days...
pub(crate) struct SortConfig<F> {
    pub page_size: usize,
    pub work_dir: PathBuf,
    pub collection: Collect<F>,
    pub comparator: TuplesComparator,
    pub input_buffers: usize,
}

/// Default value for [`Sort::input_buffers`].
pub const DEFAULT_SORT_INPUT_BUFFERS: usize = 4;

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
/// basically the resolved values of `ORDER BY` expressions. In the case of
/// simple columns we already have the value in the database. However, in the
/// case of more complicated expressions *that are NOT simple columns* we have
/// to compute the value. For example:
///
/// ```sql
/// CREATE TABLE products (id INT PRIMARY KEY, name VARCHAR(255), price INT, discount INT);
///
/// SELECT * FROM products ORDER BY price * discount, price;
/// ```
///
/// In the example above the first sort key is `price * discount`, which is an
/// expression that needs to be evaluated. The second sort key corresponds to
/// the expression `price`, which is a simple column that we already have value
/// for, we don't need to generate it. The sort keys that need to be generated
/// are computed by [`SortKeysGen`] before the [`Collect`] writes the tuple
/// to a file or an in-memory buffer and they are appended to the end of the
/// tuple after all its other columns. That format is then used by
/// [`TuplesComparator`] to determine the final [`Ordering`].
///
/// The reason we need to generate the sort keys so early is because they change
/// the length in bytes of the tuple and we need to know the length of the
/// largest tuple that we are going to work with in order to compute the exact
/// page size that we need for the K-way external merge sort.
///
/// Note that there is no "overflow" at this level, tuples that are distributed
/// across overflow pages are already merged together back into one contiguous
/// buffer by the scan plan. So we only work with complete data here, there are
/// no partial rows.
///
/// Once the [`Collect`] has successfully collected all the tuples modified
/// by [`SortKeysGen`] we can finally start sorting.
///
/// There are two main cases:
///
/// 1. The [`Collect`] did not use any files because all the tuples fit
/// in its in-memory buffer. In that case, move the buffer out of the
/// [`Collect`] into the [`Sort`] plan and just do an in-memory sorting.
/// No IO required.
///
/// 2. The [`Collect`] had to create a file in order to collect all
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
#[derive(Debug, PartialEq)]
pub(crate) struct Sort<F> {
    /// Tuple input.
    collection: Collect<F>,
    /// Tuples comparator used to obtain [`Ordering`] instances.
    comparator: TuplesComparator,
    /// `true` if we already sorted the tuples.
    sorted: bool,
    /// Page size used by the [`Pager`].
    page_size: usize,
    /// How many input buffers to use for the K-way algorithm. This is "K".
    input_buffers: usize,
    /// K-way output buffer. Used also for sorting in memory and returning tuples.
    output_buffer: TupleBuffer,
    /// Working directory to create temporary files.
    work_dir: PathBuf,
    /// File used to read tuples.
    input_file: Option<F>,
    /// File used to write tuples to.
    output_file: Option<F>,
    /// Path of [`Self::input_file`].
    input_file_path: PathBuf,
    /// Path of [`Self::output_file`].
    output_file_path: PathBuf,
}

impl<F> From<SortConfig<F>> for Sort<F> {
    fn from(
        SortConfig {
            page_size,
            work_dir,
            collection,
            comparator,
            input_buffers,
        }: SortConfig<F>,
    ) -> Self {
        Self {
            page_size,
            work_dir,
            collection,
            comparator,
            input_buffers,
            sorted: false,
            input_file: None,
            output_file: None,
            output_buffer: TupleBuffer::empty(),
            input_file_path: PathBuf::new(),
            output_file_path: PathBuf::new(),
        }
    }
}

/// Compares two tuples using their "sort keys" and returns an [`Ordering`].
///
/// See the documentation of [`Sort`] for more details.
#[derive(Debug, PartialEq)]
pub(crate) struct TuplesComparator {
    /// Original schema of the tuples.
    pub schema: Schema,
    /// Schema that includes generated sort keys (expressions like `age + 10`).
    pub sort_schema: Schema,
    /// Index of each sort key in [`Self::sort_schema`].
    pub sort_keys_indexes: Vec<usize>,
}

impl TuplesComparator {
    pub fn cmp(&self, t1: &[Value], t2: &[Value]) -> Ordering {
        debug_assert!(t1.len() == t2.len(), "tuple length mismatch");

        debug_assert!(
            t1.len() == self.sort_schema.len(),
            "tuple length doesn't match sort schema length"
        );

        for index in self.sort_keys_indexes.iter().copied() {
            match t1[index].partial_cmp(&t2[index]) {
                Some(ordering) => {
                    if ordering != Ordering::Equal {
                        return ordering;
                    }
                }
                None => {
                    if mem::discriminant(&t1[index]) != mem::discriminant(&t2[index]) {
                        unreachable!(
                            "it should be impossible to run into type errors at this point: cmp() {} against {}",
                            t1[index],
                            t2[index]
                        );
                    }
                }
            }
        }

        Ordering::Equal
    }
}

impl<F> Sort<F> {
    /// Sorts the tuples in [`Self::output_buffer`].
    fn sort_output_buffer(&mut self) {
        self.output_buffer
            .tuples
            .make_contiguous()
            .sort_by(|t1, t2| self.comparator.cmp(t1, t2));
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
        if self.collection.reader.is_none() {
            self.output_buffer = mem::replace(&mut self.collection.mem_buf, TupleBuffer::empty());
            self.sort_output_buffer();

            return Ok(());
        }

        // We need files to sort.
        let (input_file_path, input_file) = tmp_file::<F>(&self.work_dir, "mkdb.sort.input")?;
        self.input_file = Some(input_file);
        self.input_file_path = input_file_path;

        let (output_file_path, output_file) = tmp_file::<F>(&self.work_dir, "mkdb.sort.output")?;
        self.output_file = Some(output_file);
        self.output_file_path = output_file_path;

        // Figure out the page size.
        self.page_size = std::cmp::max(
            TupleBuffer::page_size_needed_for(self.collection.mem_buf.largest_tuple_size),
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
        self.output_buffer =
            TupleBuffer::new(self.page_size, self.comparator.sort_schema.clone(), false);

        let mut input_pages = 0;
        while let Some(tuple) = self.collection.try_next()? {
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
                TupleBuffer::new(self.page_size, self.comparator.sort_schema.clone(), false)
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

                        let cmp = self
                            .comparator
                            .cmp(&input_buffers[i][0], &input_buffers[min][0]);

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
            self.collection.collect()?;
            self.sort()?;
            self.sorted = true;
        }

        if self.output_buffer.is_empty() {
            if let Some(input_file) = self.input_file.as_mut() {
                if let Err(DbError::Io(e)) = self.output_buffer.read_from(input_file) {
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
            tuple.drain(self.comparator.schema.len()..);
            tuple
        }))
    }
}

impl<F> Display for Sort<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let sort_col_names = self
            .comparator
            .sort_keys_indexes
            .iter()
            .map(|i| &self.comparator.sort_schema.columns[*i].name);

        write!(f, "Sort ({})", join(sort_col_names, ", "))
    }
}

/// Creates a temporary file.
///
/// We should use uuid or tempfile or something. This is poor man's random
/// file name, but since only the client code is allowed to use dependencies
/// we'll just roll Unix Epoch based files.
fn tmp_file<F: FileOps>(work_dir: &Path, extension: &str) -> io::Result<(PathBuf, F)> {
    use std::time::SystemTime;

    let file_name = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_millis();

    let path = work_dir.join(format!("mkdb.tmp/{file_name:x}.{extension}"));

    let file = F::create(&path)?;

    Ok((path, file))
}

// TODO: All the code in this module is indirectly tested by
// [`crate::db::tests`] but some specific tests would be nice here. We can use
// the [`Values`] plan as a base for mocks that return any tuples we want and
// build a little testing framework with that.
