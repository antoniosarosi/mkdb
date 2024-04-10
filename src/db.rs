//! This is where the actual code starts executing.
//!
//! The [`Database`] struct owns everything and delegates work to the other
//! modules.

use std::{
    cell::RefCell,
    collections::{HashMap, VecDeque},
    fmt::Display,
    fs::File,
    io::{self, Read, Seek, Write},
    path::{Path, PathBuf},
    rc::Rc,
};

use crate::{
    os::{FileSystemBlockSize, Open},
    paging::{
        io::FileOps,
        pager::{PageNumber, Pager},
    },
    query,
    sql::{
        self,
        analyzer::AnalyzerError,
        parser::{Parser, ParserError},
        statement::{Column, Constraint, Create, DataType, Statement, Value},
    },
    storage::{tuple, BTree, BTreeKeyComparator, FixedSizeMemCmp},
    vm::{
        self,
        plan::{Plan, Tuple},
        TypeError, VmError,
    },
};

/// Database file default page size.
pub(crate) const DEFAULT_PAGE_SIZE: usize = 4096;

/// Name of the meta-table used to keep track of other tables.
pub(crate) const MKDB_META: &str = "mkdb_meta";

/// Name of the column used to store [`RowId`] values.
pub(crate) const ROW_ID_COL: &str = "row_id";

/// Root page of the meta-table.
pub(crate) const MKDB_META_ROOT: PageNumber = 0;

/// Max size that can be collected in memory for [`QuerySet`] structures. 1 GiB.
///
/// Mostly relevant for network code, as the database can process rows one at
/// a time but we have no implementation of SQL cursors or anything like that.
pub(crate) const MAX_QUERY_SET_SIZE: usize = 1 << 30;

/// Rows are uniquely identified by an 8 byte key stored in big endian at the
/// beginning of each tuple.
///
/// The big endian format allows for fast memcmp() comparisons where we don't
/// need to parse or cast the integer to compare it to another one. This idea
/// was of course stolen from SQLite.
pub(crate) type RowId = u64;

/// Main entry point to everything.
///
/// Provides the high level [`Database::exec`] API that receives SQL text and
/// runs it.
pub(crate) struct Database<F> {
    /// The database owns the pager.
    ///
    /// TODO: [`Rc<Refcell>`] is a temporary solution until we make the pager
    /// multithreaded. The pager should be able to allow multiple readers and
    /// one writer.
    pub pager: Rc<RefCell<Pager<F>>>,
    /// Database context. See [`DatabaseContext`].
    pub context: Context,
    /// Working directory (the directory of the file).
    pub work_dir: PathBuf,
    /// `true` if we are currently in a transaction.
    pub transaction_in_progress: bool,
}

/// Not really "Send" because of the [`Rc<RefCell>`], but we put the entire
/// database behind a mutex when working with it in the "server.rs" file and we
/// take care of not unlocking the database until `transaction_started` is
/// false. We could probably build a specific struct that wraps the Database
/// and does all this, but what we really should do instead is make the program
/// actually multithreaded. We can support multiple readers while only allowing
/// one writer. Of course, easier said than done, that's why we're using a
/// Mutex :)
unsafe impl Send for Database<File> {}

impl Database<File> {
    /// Initializes a [`Database`] instance from the given file.
    pub fn init(path: impl AsRef<Path>) -> Result<Self, DbError> {
        let file = crate::os::Fs::options()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .bypass_cache(true)
            .sync_on_write(false)
            .lock(true)
            .open(&path)?;

        let metadata = file.metadata()?;

        if !metadata.is_file() {
            return Err(io::Error::new(io::ErrorKind::Unsupported, "not a file").into());
        }

        let block_size = crate::os::Fs::block_size(&path)?;

        let full_db_file_path = path.as_ref().canonicalize()?;
        let work_dir = full_db_file_path.parent().unwrap().to_path_buf();

        let mut extension = full_db_file_path.extension().unwrap().to_os_string();
        extension.push(".journal");

        let journal_file_path = full_db_file_path.with_extension(extension);

        let mut pager = Pager::<File>::builder()
            .page_size(DEFAULT_PAGE_SIZE)
            .block_size(block_size)
            .journal_file_path(journal_file_path)
            .wrap(file);

        pager.init()?;

        // Initial rollback on startup if the journal file exists.
        pager.rollback()?;

        Ok(Database::new(Rc::new(RefCell::new(pager)), work_dir))
    }
}

/// Errors somehow related to SQL.
#[derive(Debug, PartialEq)]
pub(crate) enum SqlError {
    /// Database table not found or otherwise not usable.
    InvalidTable(String),
    /// Table column not found or not usable in the context of the error.
    InvalidColumn(String),
    /// Duplicated UNIQUE columns, duplicated PRIMARY KEY columns, etc.
    DuplicatedKey(Value),
    /// Errors caught by the [`sql::analyzer`].
    AnalyzerError(AnalyzerError),
    /// Data type errors. Trying to add numbers to strings, etc.
    TypeError(TypeError),
    /// Errors thrown by the [`vm`] when executing expressions.
    VmError(VmError),
    /// Uncategorized error with custom message.
    Other(String),
}

impl From<TypeError> for SqlError {
    fn from(type_error: TypeError) -> Self {
        SqlError::TypeError(type_error)
    }
}

impl From<AnalyzerError> for SqlError {
    fn from(analyzer_error: AnalyzerError) -> Self {
        SqlError::AnalyzerError(analyzer_error)
    }
}

impl From<VmError> for SqlError {
    fn from(vm_error: VmError) -> Self {
        Self::VmError(vm_error)
    }
}

impl Display for SqlError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidTable(name) => write!(f, "invalid table '{name}'"),
            Self::InvalidColumn(name) => write!(f, "invalid column '{name}'"),
            Self::DuplicatedKey(key) => write!(f, "duplicated key {key}"),
            Self::AnalyzerError(analyzer_error) => write!(f, "{analyzer_error}"),
            Self::VmError(vm_error) => write!(f, "{vm_error}"),
            Self::TypeError(type_error) => write!(f, "{type_error}"),
            Self::Other(message) => f.write_str(message),
        }
    }
}

/// Generic top level error.
#[derive(Debug)]
pub enum DbError {
    /// Files, sockets, etc.
    Io(io::Error),
    /// [`sql::parser`] error.
    Parser(ParserError),
    /// Other SQL error not related to syntax.
    Sql(SqlError),
    /// Something in the database file or journal file is corrupted/unexpected.
    Corrupted(String),
    /// Query too large or out of memory.
    NoMem,
    /// Uncategorized custom error.
    Other(String),
}

impl Display for DbError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "{e}"),
            Self::Parser(e) => write!(f, "{e}"),
            Self::Sql(e) => write!(f, "{e}"),
            Self::Corrupted(message) => f.write_str(message),
            Self::NoMem => f.write_str("our of memory"),
            Self::Other(message) => f.write_str(message),
        }
    }
}

impl<E: Into<SqlError>> From<E> for DbError {
    fn from(err: E) -> Self {
        DbError::Sql(err.into())
    }
}

impl From<io::Error> for DbError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<ParserError> for DbError {
    fn from(e: ParserError) -> Self {
        Self::Parser(e)
    }
}

/// In-memory representation of a table schema.
#[derive(Debug, PartialEq, Clone)]
pub struct Schema {
    /// Column definitions.
    pub columns: Vec<Column>,
    /// Quick index to find column defs based on their name.
    pub index: HashMap<String, usize>,
}

impl Schema {
    /// Create a new schema with the given column definitions.
    pub fn new(columns: Vec<Column>) -> Self {
        let index = columns
            .iter()
            .enumerate()
            .map(|(i, col)| (col.name.clone(), i))
            .collect();

        Self { columns, index }
    }

    /// Creates an empty schema with no columns.
    pub fn empty() -> Self {
        Self::new(vec![])
    }

    /// Returns the index in [`Self::columns`] of `col`.
    pub fn index_of(&self, col: &str) -> Option<usize> {
        self.index.get(col).copied()
    }

    /// Number of columns in this schema.
    pub fn len(&self) -> usize {
        self.columns.len()
    }

    /// Appends a new column to the end of the schema.
    pub fn push(&mut self, column: Column) {
        self.index.insert(column.name.to_owned(), self.len());
        self.columns.push(column);
    }

    /// Prepends the special "row_id" column at the beginning of the schema.
    pub fn prepend_row_id(&mut self) {
        debug_assert!(
            self.columns[0].name != ROW_ID_COL,
            "schema already has {ROW_ID_COL}: {self:?}"
        );

        let col = Column::new(ROW_ID_COL, DataType::UnsignedBigInt);

        self.columns.insert(0, col);
        self.index.values_mut().for_each(|idx| *idx += 1);
        self.index.insert(String::from(ROW_ID_COL), 0);
    }

    /// See [`has_btree_key`].
    pub fn has_btree_key(&self) -> bool {
        has_btree_key(&self.columns)
    }
}

impl<'c, C: IntoIterator<Item = &'c Column>> From<C> for Schema {
    fn from(columns: C) -> Self {
        Self::new(Vec::from_iter(columns.into_iter().cloned()))
    }
}

/// Returns `true` if the primary key of the table can also be used as the BTree
/// key.
///
/// If the table doesn't need a Row ID we won't create an index for the primary
/// key because the table BTree itself will already be an index for the primary
/// key since we're using index organized storage.
///
/// On the other hand if we need a Row ID then the primary key is not usable as
/// a direct table index, so we'll create a separate BTree index instead.
pub fn has_btree_key(columns: &[Column]) -> bool {
    columns[0].constraints.contains(&Constraint::PrimaryKey)
        && !matches!(columns[0].data_type, DataType::Varchar(_) | DataType::Bool)
}

/// This only exists because in earlier development stages the iterator model
/// was not yet implemented.
///
/// Without the iterator model we couldn't process tuples one at a time, we had
/// to collect them all at once and process them after that. So all tests in
/// this module (quite a few at this point) were written using this struct for
/// `assert_eq` comparisons.
///
/// Right now we're only using this struct to collect the results of a query
/// in memory, which we mostly need for tests and also the client package
/// collects all the results to print an ASCII table like the MySQL client.
#[derive(Debug, PartialEq)]
pub struct QuerySet {
    /// Schema of the results.
    pub schema: Schema,
    /// Rows.
    pub tuples: Vec<Vec<Value>>,
}

impl QuerySet {
    /// Creates a new [`QuerySet`].
    pub fn new(schema: Schema, tuples: Vec<Vec<Value>>) -> Self {
        Self { schema, tuples }
    }

    /// Creates a set with no schema and no results.
    pub fn empty() -> Self {
        Self {
            schema: Schema::empty(),
            tuples: Vec::new(),
        }
    }

    /// Returns a concrete value given its column name and row number.
    pub fn get(&self, row: usize, column: &str) -> Option<&Value> {
        self.tuples.get(row)?.get(self.schema.index_of(column)?)
    }

    /// `true` if there are no results.
    pub fn is_empty(&self) -> bool {
        self.tuples.is_empty()
    }
}

/// Schema of the table used to keep track of the database information.
pub(crate) fn mkdb_meta_schema() -> Schema {
    Schema::from(&[
        // Either "index" or "table"
        Column::new("type", DataType::Varchar(255)),
        // Index or table name
        Column::new("name", DataType::Varchar(255)),
        // Root page
        Column::new("root", DataType::UnsignedInt),
        // Table name
        Column::new("table_name", DataType::Varchar(255)),
        // SQL used to create the index or table.
        // TODO: Implement and use some TEXT data type with higher length limits.
        Column::new("sql", DataType::Varchar(65535)),
    ])
}

/// Data that we need to know about an index at runtime.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct IndexMetadata {
    /// Root page of the index-
    pub root: PageNumber,
    /// Index name.
    pub name: String,
    /// Column on which the index was created.
    pub column: Column,
    /// Schema of the index. Always key -> primary key.
    pub schema: Schema,
    /// Always `true` because non-unique indexes are not implemented.
    pub unique: bool,
}

/// Data that we need to know about tables at runtime.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct TableMetadata {
    /// Root page of the table.
    pub root: PageNumber,
    /// Table name.
    pub name: String,
    /// Schema of the table as defined by the `CREATE TABLE` statement.
    pub schema: Schema,
    /// All the indexes associated to this table.
    pub indexes: Vec<IndexMetadata>,
    /// Next [`RowId`] for this table.
    row_id: RowId,
}

/// Dynamic dispatch for relation types.
///
/// Some [`Plan`] types need to work with both table BTrees and Index BTrees but
/// we will only know exactly wich at runtime. The relations share some
/// properties like the root page but differ in others, for example tables have
/// indexes associated to them but indexes don't.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Relation {
    Index(IndexMetadata),
    Table(TableMetadata),
}

impl Relation {
    /// Root page.
    pub fn root(&self) -> PageNumber {
        match self {
            Self::Index(index) => index.root,
            Self::Table(table) => table.root,
        }
    }

    /// Dynamically dispatched key comparator for the BTree.
    pub fn comparator(&self) -> BTreeKeyComparator {
        match self {
            Self::Index(index) => BTreeKeyComparator::from(&index.column.data_type),
            Self::Table(table) => BTreeKeyComparator::from(&table.schema.columns[0].data_type),
        }
    }

    /// Schema of the relation.
    ///
    /// Indexes will always have 2 columns, the index key and the table key.
    pub fn schema(&self) -> &Schema {
        match self {
            Self::Index(index) => &index.schema,
            Self::Table(table) => &table.schema,
        }
    }

    /// Type of relation as a string.
    pub fn kind(&self) -> &str {
        match self {
            Self::Index(_) => "index",
            Self::Table(_) => "table",
        }
    }

    /// Name of the relation.
    ///
    /// This is the name of the index or the table.
    pub fn name(&self) -> &str {
        match self {
            Self::Index(index) => &index.name,
            Self::Table(table) => &table.name,
        }
    }

    /// Returns the array index where the table key is located in the schema.
    ///
    /// For tables the key is always at index 0 while for indexes it's at index
    /// 1.
    pub fn index_of_table_key(&self) -> usize {
        match self {
            Self::Index(_) => 1,
            Self::Table(_) => 0,
        }
    }
}

impl TableMetadata {
    /// Returns the next [`RowId`] that should be used for rows in this table.
    pub fn next_row_id(&mut self) -> RowId {
        let row_id = self.row_id;
        self.row_id += 1;

        row_id
    }

    /// As of right now all tables use integers as real primary keys.
    ///
    /// Varchar primary keys are not used, we use the special "row_id" column
    /// in that case.
    pub(crate) fn comparator(&self) -> Result<FixedSizeMemCmp, DbError> {
        FixedSizeMemCmp::try_from(&self.schema.columns[0].data_type).map_err(|_| {
            DbError::Corrupted(format!(
                "table {} is using a non-integer BTree key of type {}",
                self.name, self.schema.columns[0].data_type
            ))
        })
    }

    /// Generates a new schema that contains only the key of this table.
    ///
    /// This is useful for some query plans that only need to work with keys.
    pub fn key_only_schema(&self) -> Schema {
        Schema::new(vec![self.schema.columns[0].clone()])
    }
}

/// API to obtain data about the database itself.
pub(crate) trait DatabaseContext {
    /// Returns a [`TableMetadata`] object describing `table`.
    fn table_metadata(&mut self, table: &str) -> Result<&mut TableMetadata, DbError>;
}

/// Default value for [`Context::max_size`].
const DEFAULT_RELATION_CACHE_SIZE: usize = 512;

/// Dead simple cache made for storing [`TableMetadata`] instances.
///
/// Unlike [`crate::paging::cache`], this one doesn't need to complicated since
/// [`TableMetadata`] structs are just a handful of bytes depending on the
/// schema and databases usually don't have thousands of tables like they do
/// pages. So the eviction policy is pretty much random, we evict whatever the
/// underlying [`HashMap`] decides that is the first element.
///
/// This struct is also used as some sort of mock for tests in [`crate::sql`].
/// Some components like the analyzer need access to the database context but
/// we don't want to create an entire [`Database`] struct just for that. The
/// [`Database`] struct also need to parse SQL to obtain metadata about tables
/// so it would call the [`crate::sql`] module again, and we don't want so much
/// mutual recursion because it makes test debugging hard.
pub(crate) struct Context {
    /// Maps the table name to its metadata.
    tables: HashMap<String, TableMetadata>,
    /// Maximum size of the cache.
    max_size: Option<usize>,
}

impl Context {
    /// New default [`Context`].
    ///
    /// [`Context::max_size`] is unlimited.
    pub fn new() -> Self {
        Self {
            tables: HashMap::new(),
            max_size: None,
        }
    }

    /// New [`Context`] with fixed max size.
    pub fn with_max_size(max_size: usize) -> Self {
        Self {
            tables: HashMap::with_capacity(max_size),
            max_size: Some(max_size),
        }
    }

    /// `true` if `table` exists in memory.
    pub fn contains(&self, table: &str) -> bool {
        self.tables.contains_key(table)
    }

    /// Adds the given table and metadata to this context.
    pub fn insert(&mut self, metadata: TableMetadata) {
        if self.max_size.is_some_and(|size| self.tables.len() >= size) {
            let evict = self.tables.keys().next().unwrap().clone();
            self.tables.remove(&evict);
        }

        self.tables.insert(metadata.name.clone(), metadata);
    }

    /// Removes the `table` from cache. Next it will be loaded from disk.
    pub fn invalidate(&mut self, table: &str) {
        self.tables.remove(table);
    }
}

// Mainly used for mocks in tests to avoid passing the entire database around
// and debugging mutual recursive calls.
#[cfg(test)]
impl TryFrom<&[&str]> for Context {
    type Error = DbError;

    /// Creates a context from raw SQL statements.
    ///
    /// Used for tests. See [`crate::sql::analyzer`] or [`crate::sql::prepare`].
    fn try_from(statements: &[&str]) -> Result<Self, Self::Error> {
        let mut context = Self::new();
        let mut root = 1;

        for sql in statements {
            let statement = Parser::new(sql).parse_statement()?;

            match statement {
                Statement::Create(Create::Table { name, columns }) => {
                    let mut schema = Schema::from(&columns);
                    schema.prepend_row_id();

                    let mut metadata = TableMetadata {
                        root,
                        name: name.clone(),
                        row_id: 1,
                        schema,
                        indexes: vec![],
                    };
                    root += 1;

                    for column in &columns {
                        for constraint in &column.constraints {
                            let index_name = match constraint {
                                Constraint::PrimaryKey => format!("{name}_pk_index"),
                                Constraint::Unique => format!("{name}_{}_uq_index", column.name),
                            };

                            metadata.indexes.push(IndexMetadata {
                                column: column.clone(),
                                schema: Schema::new(vec![column.clone(), columns[0].clone()]),
                                name: index_name,
                                root,
                                unique: true,
                            });

                            root += 1;
                        }
                    }

                    context.insert(metadata);
                }

                Statement::Create(Create::Index {
                    name,
                    column,
                    unique,
                    ..
                }) if unique => {
                    let table = context.table_metadata(&name)?;
                    let index_col = table.schema.columns[table.schema.index_of(&column).unwrap()].clone();

                    table.indexes.push(IndexMetadata {
                        column: index_col.clone(),
                        schema: Schema::new(vec![index_col, table.schema.columns[0].clone()]),
                        name,
                        root,
                        unique,
                    });
                    root += 1;
                }

                other => return Err(DbError::Sql(SqlError::Other(format!(
                    "only CREATE TABLE and CREATE UNIQUE INDEX should be used to create a mock context but received {other}"
                )))),
            }
        }

        Ok(context)
    }
}

impl DatabaseContext for Context {
    fn table_metadata(&mut self, table: &str) -> Result<&mut TableMetadata, DbError> {
        self.tables
            .get_mut(table)
            .ok_or_else(|| DbError::Sql(SqlError::InvalidTable(table.into())))
    }
}

impl<F> Database<F> {
    /// Creates a new database.
    pub fn new(pager: Rc<RefCell<Pager<F>>>, work_dir: PathBuf) -> Self {
        Self {
            pager,
            work_dir,
            context: Context::with_max_size(DEFAULT_RELATION_CACHE_SIZE),
            transaction_in_progress: false,
        }
    }

    /// Returns `true` if there's a transaction in progress at the moment.
    pub fn transaction_in_progress(&self) -> bool {
        self.transaction_in_progress
    }

    /// Starts a new transaction.
    ///
    /// Transactions can only be terminated by calling [`Database::rollback`]
    /// or [`Database::commit`]. Otherwise the equivalent SQL statements can
    /// be used to terminate transactions.
    pub fn start_transaction(&mut self) {
        self.transaction_in_progress = true;
    }
}

impl<F: Seek + Read + Write + FileOps> DatabaseContext for Database<F> {
    fn table_metadata(&mut self, table: &str) -> Result<&mut TableMetadata, DbError> {
        if !self.context.contains(table) {
            let metadata = self.load_table_metadata(table)?;
            self.context.insert(metadata);
        }

        self.context.table_metadata(table)
    }
}

impl<F: Seek + Read + Write + FileOps> Database<F> {
    /// Loads the next row ID that should be used for the table rooted at
    /// `root`.
    ///
    /// This is not so expensive since it traverses the BTree from the root
    /// straight to a leaf node to find the max row ID, but it should be cached
    /// to avoid IO next time.
    fn load_next_row_id(&mut self, root: PageNumber) -> Result<RowId, DbError> {
        let mut pager = self.pager.borrow_mut();
        let mut btree = BTree::new(&mut pager, root, FixedSizeMemCmp::for_type::<RowId>());

        let row_id = if let Some(max) = btree.max()? {
            tuple::deserialize_row_id(max.as_ref()) + 1
        } else {
            1
        };

        Ok(row_id)
    }

    /// Loads all the metadata that we store about `table`.
    ///
    /// Right now the [`MKDB_META`] table doesn't use any indexes, so this
    /// is basically a sequential scan. The metadata table shouldn't get too
    /// big in most scenarios though.
    fn load_table_metadata(&mut self, table: &str) -> Result<TableMetadata, DbError> {
        if table == MKDB_META {
            let mut schema = mkdb_meta_schema();
            schema.prepend_row_id();

            return Ok(TableMetadata {
                root: MKDB_META_ROOT,
                name: String::from(table),
                row_id: self.load_next_row_id(MKDB_META_ROOT)?,
                schema,
                indexes: vec![],
            });
        }

        let mut metadata = TableMetadata {
            root: 1,
            name: String::from(table),
            row_id: 1,
            schema: Schema::empty(),
            indexes: Vec::new(),
        };

        let mut found_table_definition = false;

        let (schema, mut results) = self.prepare(&format!(
            "SELECT root, sql FROM {MKDB_META} where table_name = '{table}';"
        ))?;

        let corrupted_error = || {
            DbError::Corrupted(format!(
                "{MKDB_META} table is corrupted or contains wrong/unexpected data"
            ))
        };

        while let Some(tuple) = results.try_next()? {
            let Value::Number(root) = &tuple[schema.index_of("root").ok_or(corrupted_error())?]
            else {
                return Err(corrupted_error());
            };

            match &tuple[schema.index_of("sql").ok_or(corrupted_error())?] {
                Value::String(sql) => match Parser::new(sql).parse_statement()? {
                    Statement::Create(Create::Table { columns, .. }) => {
                        assert!(
                            !found_table_definition,
                            "multiple definitions of table '{table}'"
                        );

                        metadata.root = *root as PageNumber;
                        metadata.schema = Schema::new(columns);

                        // Tables tha don't have an integer primary key as the
                        // first field will use a hidden primary key that we
                        // generate ourselves.
                        if !metadata.schema.has_btree_key() {
                            metadata.schema.prepend_row_id();
                        }

                        found_table_definition = true;
                    }

                    Statement::Create(Create::Index {
                        column,
                        name,
                        unique,
                        ..
                    }) => {
                        // The table schema should be loaded by this time
                        // because it's impossible to define an index unless the
                        // table exists and the results are returned sorted by
                        // row_id.
                        let col_idx = metadata.schema.index_of(&column).ok_or(
                            SqlError::Other(format!(
                                "could not find index column {column} in the definition of table {table}"
                            )),
                        )?;

                        let index_col = metadata.schema.columns[col_idx].clone();

                        metadata.indexes.push(IndexMetadata {
                            column: index_col.clone(),
                            schema: Schema::new(vec![
                                index_col,
                                metadata.schema.columns[0].clone(),
                            ]),
                            name,
                            root: *root as PageNumber,
                            unique,
                        });
                    }

                    _ => return Err(corrupted_error()),
                },

                _ => return Err(corrupted_error()),
            };
        }

        if !found_table_definition {
            return Err(DbError::Sql(SqlError::InvalidTable(table.into())));
        }

        if metadata.schema.columns[0].name == ROW_ID_COL {
            metadata.row_id = self.load_next_row_id(metadata.root)?;
        }

        Ok(metadata)
    }

    /// Returns the root page of `index` if it exists.
    fn index_metadata(&mut self, index_name: &str) -> Result<IndexMetadata, DbError> {
        let query = self.exec(&format!(
            "SELECT table_name FROM {MKDB_META} where name = '{index_name}' AND type = 'index';"
        ))?;

        if query.is_empty() {
            return Err(DbError::Sql(SqlError::Other(format!(
                "index {index_name} does not exist"
            ))));
        }

        let table_name = match query.get(0, "table_name") {
            Some(Value::String(name)) => name,
            _ => unreachable!(),
        };

        let table_metadata = self.table_metadata(table_name)?;

        // TODO: Innefficient.
        Ok(table_metadata
            .indexes
            .iter()
            .find(|index| index.name == index_name)
            .unwrap()
            .clone())
    }

    /// Highest level API in the entire system.
    ///
    /// Receives a SQL string and executes it, collecting the results in memory.
    /// Of course this is not ideal for internal work because the results might
    /// not fit in memory, but it's nice for tests and network code that returns
    /// one complete packet.
    ///
    /// Internal APIs must use [`Database::prepare`] instead to obtain an
    /// iterator over the rows produced by the query. That will limit the memory
    /// usage to the size of internal buffers used the [`Plan`] execution engine
    /// at [`vm::plan`].
    pub fn exec(&mut self, input: &str) -> Result<QuerySet, DbError> {
        let (schema, mut preapred_staement) = self.prepare(input)?;

        let mut query_set = QuerySet::new(schema, vec![]);

        let mut total_size = 0;

        while let Some(tuple) = preapred_staement.try_next()? {
            total_size += tuple::size_of(&tuple, &query_set.schema);
            if total_size > MAX_QUERY_SET_SIZE {
                self.rollback()?;
                return Err(DbError::NoMem);
            }

            query_set.tuples.push(tuple);
        }

        Ok(query_set)
    }

    /// Parses the given `sql` and generates an execution plan for it.
    ///
    /// The execution plan is returned and can be iterated tuple by tuple
    /// with fixed memory usage (except for the size of the tuple itself). This
    /// is the API the should be used to process queries as it will not make use
    /// of all the system's RAM.
    pub fn prepare(&mut self, sql: &str) -> Result<(Schema, PreparedStatement<'_, F>), DbError> {
        let statement = sql::pipeline(sql, self)?;

        let mut schema = Schema::empty();

        let exec = match statement {
            Statement::Create(_)
            | Statement::StartTransaction
            | Statement::Commit
            | Statement::Rollback => Exec::Statement(statement),

            Statement::Explain(inner) => match &*inner {
                Statement::Select { .. }
                | Statement::Insert { .. }
                | Statement::Update { .. }
                | Statement::Delete { .. } => {
                    schema = Schema::new(vec![Column::new("Query Plan", DataType::Varchar(255))]);
                    let plan = query::planner::generate_plan(*inner, self)?;
                    Exec::Explain(format!("{plan}").lines().map(String::from).collect())
                }

                _ => {
                    return Err(DbError::Other(String::from(
                        "EXPLAIN only works with SELECT, UPDATE, DELETE and INSERT statements",
                    )));
                }
            },

            _ => {
                let plan = query::planner::generate_plan(statement, self)?;
                if let Some(plan_schema) = plan.schema() {
                    schema = plan_schema;
                }
                Exec::Plan(plan)
            }
        };

        let prepared_statement = PreparedStatement {
            db: self,
            auto_commit: false,
            exec: Some(exec),
        };

        Ok((schema, prepared_statement))
    }

    /// Manually rolls back the database and stops the current transaction.
    pub fn rollback(&mut self) -> Result<usize, DbError> {
        self.transaction_in_progress = false;
        self.pager.borrow_mut().rollback()
    }

    /// Manually commits the changes and stops the current transaction.
    pub fn commit(&mut self) -> io::Result<()> {
        self.transaction_in_progress = false;
        self.pager.borrow_mut().commit()
    }
}

/// Not all statements need [`Plan`] trees for execution.
///
/// See [`vm::statement`].
enum Exec<F> {
    /// Statements that don't need any plans executed by [`vm::statement`].
    Statement(Statement),
    /// Complex statements that require [`Plan`] trees executed by [`vm::plan`].
    Plan(Plan<F>),
    /// Return a string that describes the generated plan.
    Explain(VecDeque<String>),
}

/// A prepared statement is a statement that has been successfully parsed and
/// is ready to execute.
///
/// The prepared statement contains the plan that the virtual machine will run
/// and can be queried like an iterator through the [`Self::try_next`] method.
/// Everything is lazily evaluated, the prepared statement will do absolutely
/// nothing unless consumed.
///
/// It's important to take into account that after the first call to
/// [`Self::try_next`] the database will start a transaction if there isn't one
/// in progress and that transaction *will not end* until either the statement
/// iterator is fully consumed or the transaction is stopped manually with
/// [`Database::rollback`] or [`Database::commit`].
///
/// If the iterator fails or consumes all the tuples then the transaction is
/// automatically closed. Transactions manually started by the client through
/// `START TRANSACTION` are only closed when the client sends `COMMIT` or
/// `ROLLBACK` or there is an error. Whenever there's an error the database
/// always rolls back.
pub(crate) struct PreparedStatement<'d, F> {
    /// Reference to the main databases object.
    db: &'d mut Database<F>,
    /// Execution plan.
    ///
    /// Once this is set to [`None`] the iterator is considered fully consumed
    /// and will no longer produce results.
    exec: Option<Exec<F>>,
    /// `true` if the client did not start a transaction.
    auto_commit: bool,
}

impl<'d, F: Seek + Read + Write + FileOps> PreparedStatement<'d, F> {
    /// Returns the next tuple that the query produces.
    ///
    /// See the documentation of [`PreparedStatement`] for more details.
    pub fn try_next(&mut self) -> Result<Option<Tuple>, DbError> {
        let Some(exec) = self.exec.as_mut() else {
            return Ok(None);
        };

        if let Exec::Statement(Statement::StartTransaction) = exec {
            if self.db.transaction_in_progress() {
                return Err(DbError::Other(String::from(
                    "There is already a transaction in progress",
                )));
            }

            self.db.start_transaction();
            return Ok(None);
        }

        if !self.db.transaction_in_progress() {
            self.db.start_transaction();
            self.auto_commit = true;
        }

        let tuple = match exec {
            Exec::Statement(_) => {
                // Statements only run once, the iterator ends right here.
                let Some(Exec::Statement(statement)) = self.exec.take() else {
                    unreachable!();
                };

                match statement {
                    Statement::Commit => {
                        self.db.commit()?;
                    }
                    Statement::Rollback => {
                        self.db.rollback()?;
                    }
                    Statement::Create(_) => {
                        if let Err(e) = vm::statement::exec(statement, self.db) {
                            self.db.rollback()?;
                            return Err(e);
                        }
                    }
                    _ => unreachable!(),
                };

                None
            }

            Exec::Plan(plan) => match plan.try_next() {
                Ok(tuple) => tuple,

                Err(e) => {
                    // The iterator ends here, rollback and return the error.
                    self.exec.take();
                    self.db.rollback()?;
                    return Err(e);
                }
            },

            Exec::Explain(lines) => {
                let line = lines.pop_front().map(|line| vec![Value::String(line)]);

                if line.is_none() {
                    self.exec.take();
                }

                line
            }
        };

        // If this block runs then everything executed successfully. End the
        // iterator and auto commit if necessary.
        if tuple.is_none() {
            self.exec.take();
            if self.auto_commit {
                self.db.commit()?;
            }
        }

        Ok(tuple)
    }
}

#[cfg(test)]
mod tests {
    use std::{
        cell::RefCell,
        cmp::Ordering,
        collections::HashMap,
        io::{self, Read, Seek, Write},
        path::PathBuf,
        rc::Rc,
    };

    use super::{Database, DbError, DEFAULT_PAGE_SIZE};
    use crate::{
        db::{mkdb_meta_schema, QuerySet, Schema, SqlError, TypeError},
        paging::{
            cache::{Cache, DEFAULT_MAX_CACHE_SIZE},
            io::{FileOps, MemBuf},
            pager::Pager,
        },
        sql::{
            analyzer::AnalyzerError,
            parser::Parser,
            statement::{Column, DataType, Expression, Value},
        },
        storage::{reassemble_payload, tuple, Cursor},
        vm::VmDataType,
    };

    impl PartialEq for DbError {
        fn eq(&self, other: &Self) -> bool {
            match (self, other) {
                (Self::Io(a), Self::Io(b)) => a.kind() == b.kind(),
                (Self::Parser(a), Self::Parser(b)) => a == b,
                (Self::Sql(a), Self::Sql(b)) => a == b,
                _ => false,
            }
        }
    }

    struct DbConf {
        page_size: usize,
        cache_size: usize,
    }

    fn init_database_with(conf: DbConf) -> io::Result<Database<MemBuf>> {
        let mut pager = Pager::<MemBuf>::builder()
            .page_size(conf.page_size)
            .cache(Cache::with_max_size(conf.cache_size))
            .wrap(io::Cursor::new(Vec::<u8>::new()));

        pager.init()?;

        Ok(Database::new(Rc::new(RefCell::new(pager)), PathBuf::new()))
    }

    fn init_database() -> io::Result<Database<MemBuf>> {
        init_database_with(DbConf {
            cache_size: DEFAULT_MAX_CACHE_SIZE,
            page_size: DEFAULT_PAGE_SIZE,
        })
    }

    fn assert_index_contains<F: Seek + Read + Write + FileOps>(
        db: &mut Database<F>,
        name: &str,
        expected_entries: &[Vec<Value>],
    ) -> Result<(), DbError> {
        let index = db.index_metadata(name)?;

        let mut pager = db.pager.borrow_mut();
        let mut cursor = Cursor::new(index.root, 0);

        let mut entries = Vec::new();

        while let Some((page, slot)) = cursor.try_next(&mut pager)? {
            let entry = reassemble_payload(&mut pager, page, slot)?;
            entries.push(tuple::deserialize(entry.as_ref(), &index.schema));
        }

        assert_eq!(entries, expected_entries);

        Ok(())
    }

    #[test]
    fn create_table_auto_index() -> Result<(), DbError> {
        let mut db = init_database()?;

        let sql = "CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(255));";
        db.exec(sql)?;

        let query = db.exec("SELECT * FROM mkdb_meta;")?;

        assert_eq!(
            query,
            QuerySet::new(mkdb_meta_schema(), vec![vec![
                Value::String("table".into()),
                Value::String("users".into()),
                Value::Number(1),
                Value::String("users".into()),
                Value::String(Parser::new(sql).parse_statement()?.to_string())
            ]])
        );

        Ok(())
    }

    #[test]
    fn create_table_with_forced_pk_index() -> Result<(), DbError> {
        let mut db = init_database()?;

        let sql = "CREATE TABLE users (name VARCHAR(255), id INT PRIMARY KEY);";
        db.exec(sql)?;

        let query = db.exec("SELECT * FROM mkdb_meta;")?;

        assert_eq!(
            query,
            QuerySet::new(mkdb_meta_schema(), vec![
                vec![
                    Value::String("table".into()),
                    Value::String("users".into()),
                    Value::Number(1),
                    Value::String("users".into()),
                    Value::String(Parser::new(sql).parse_statement()?.to_string())
                ],
                vec![
                    Value::String("index".into()),
                    Value::String("users_pk_index".into()),
                    Value::Number(2),
                    Value::String("users".into()),
                    Value::String(
                        Parser::new("CREATE UNIQUE INDEX users_pk_index ON users(id);")
                            .parse_statement()?
                            .to_string()
                    )
                ]
            ])
        );

        Ok(())
    }

    // If the first column can be used as a primary key then there will be no
    // separate index BTree for it.
    #[test]
    fn create_multiple_tables_with_auto_index() -> Result<(), DbError> {
        let mut db = init_database()?;

        let t1 = "CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(255));";
        let t2 = "CREATE TABLE tasks (id INT PRIMARY KEY, title VARCHAR(255), description VARCHAR(255));";
        let t3 = "CREATE TABLE products (id INT PRIMARY KEY, price INT);";

        db.exec(t1)?;
        db.exec(t2)?;
        db.exec(t3)?;

        let query = db.exec("SELECT * FROM mkdb_meta;")?;

        assert_eq!(query, QuerySet {
            schema: mkdb_meta_schema(),
            tuples: vec![
                vec![
                    Value::String("table".into()),
                    Value::String("users".into()),
                    Value::Number(1),
                    Value::String("users".into()),
                    Value::String(Parser::new(t1).parse_statement()?.to_string())
                ],
                vec![
                    Value::String("table".into()),
                    Value::String("tasks".into()),
                    Value::Number(2),
                    Value::String("tasks".into()),
                    Value::String(Parser::new(t2).parse_statement()?.to_string())
                ],
                vec![
                    Value::String("table".into()),
                    Value::String("products".into()),
                    Value::Number(3),
                    Value::String("products".into()),
                    Value::String(Parser::new(t3).parse_statement()?.to_string())
                ],
            ]
        });

        Ok(())
    }

    // If the first column can't be used as a primary key then the database
    // always creates a unique index for the PK.
    #[test]
    fn create_multiple_tables_with_forced_pk_index() -> Result<(), DbError> {
        let mut db = init_database()?;

        let t1 = "CREATE TABLE users (name VARCHAR(255), id INT PRIMARY KEY);";
        let i1 = "CREATE UNIQUE INDEX users_pk_index ON users(id);";

        let t2 = "CREATE TABLE tasks (title VARCHAR(255), description VARCHAR(255), id INT PRIMARY KEY);";
        let i2 = "CREATE UNIQUE INDEX tasks_pk_index ON tasks(id);";

        let t3 = "CREATE TABLE products (price INT, id INT PRIMARY KEY);";
        let i3 = "CREATE UNIQUE INDEX products_pk_index ON products(id);";

        db.exec(t1)?;
        db.exec(t2)?;
        db.exec(t3)?;

        let query = db.exec("SELECT * FROM mkdb_meta;")?;

        assert_eq!(query, QuerySet {
            schema: mkdb_meta_schema(),
            tuples: vec![
                vec![
                    Value::String("table".into()),
                    Value::String("users".into()),
                    Value::Number(1),
                    Value::String("users".into()),
                    Value::String(Parser::new(t1).parse_statement()?.to_string())
                ],
                vec![
                    Value::String("index".into()),
                    Value::String("users_pk_index".into()),
                    Value::Number(2),
                    Value::String("users".into()),
                    Value::String(Parser::new(i1).parse_statement()?.to_string())
                ],
                vec![
                    Value::String("table".into()),
                    Value::String("tasks".into()),
                    Value::Number(3),
                    Value::String("tasks".into()),
                    Value::String(Parser::new(t2).parse_statement()?.to_string())
                ],
                vec![
                    Value::String("index".into()),
                    Value::String("tasks_pk_index".into()),
                    Value::Number(4),
                    Value::String("tasks".into()),
                    Value::String(Parser::new(i2).parse_statement()?.to_string())
                ],
                vec![
                    Value::String("table".into()),
                    Value::String("products".into()),
                    Value::Number(5),
                    Value::String("products".into()),
                    Value::String(Parser::new(t3).parse_statement()?.to_string())
                ],
                vec![
                    Value::String("index".into()),
                    Value::String("products_pk_index".into()),
                    Value::Number(6),
                    Value::String("products".into()),
                    Value::String(Parser::new(i3).parse_statement()?.to_string())
                ],
            ]
        });

        Ok(())
    }

    /// Used mostly to test the possibilities of a BTree rooted at page zero,
    /// since that's a special case.
    #[cfg(not(miri))]
    #[test]
    fn create_many_tables_with_forced_pk_index() -> Result<(), DbError> {
        let mut db = init_database()?;

        let mut expected = Vec::new();

        let schema = "(title VARCHAR(255), description VARCHAR(255), id INT PRIMARY KEY)";

        for i in 1..=100 {
            let table_name = format!("table_{i:03}");
            let index_name = format!("{table_name}_pk_index");

            let table_sql = format!("CREATE TABLE {table_name} {schema};");
            let index_sql = format!("CREATE UNIQUE INDEX {index_name} ON {table_name}(id);");

            db.exec(&table_sql)?;

            expected.push(vec![
                Value::String("table".into()),
                Value::String(table_name.clone()),
                Value::Number(0),
                Value::String(table_name.clone()),
                Value::String(Parser::new(&table_sql).parse_statement()?.to_string()),
            ]);

            expected.push(vec![
                Value::String("index".into()),
                Value::String(index_name),
                Value::Number(0),
                Value::String(table_name),
                Value::String(Parser::new(&index_sql).parse_statement()?.to_string()),
            ]);
        }

        let query = db.exec("SELECT * FROM mkdb_meta ORDER BY table_name, name;")?;

        let mut roots = HashMap::new();

        for (i, mut tuple) in expected.into_iter().enumerate() {
            let root = match query.get(i, "root").unwrap() {
                Value::Number(root) => *root,
                other => panic!("root is not a page number: {other:?}"),
            };

            let name = tuple[1].clone();
            tuple[2] = Value::Number(root);

            if let Err(root_used_by) = roots.try_insert(root, name.clone()) {
                panic!("root {root} used for both {root_used_by} and {name}");
            }

            assert_eq!(query.tuples[i], tuple);
        }

        Ok(())
    }

    #[test]
    fn insert_data() -> Result<(), DbError> {
        let mut db = init_database()?;

        db.exec("CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(255));")?;
        db.exec("INSERT INTO users(id, name) VALUES (1, 'John Doe');")?;
        db.exec("INSERT INTO users(id, name) VALUES (2, 'Jane Doe');")?;

        let query = db.exec("SELECT * FROM users;")?;

        assert_eq!(query, QuerySet {
            schema: Schema::new(vec![
                Column::primary_key("id", DataType::Int),
                Column::new("name", DataType::Varchar(255)),
            ]),
            tuples: vec![
                vec![Value::Number(1), Value::String("John Doe".into())],
                vec![Value::Number(2), Value::String("Jane Doe".into())],
            ]
        });

        Ok(())
    }

    #[test]
    fn insert_disordered() -> Result<(), DbError> {
        let mut db = init_database()?;

        db.exec("CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(255), age INT);")?;
        db.exec("INSERT INTO users(name, age, id) VALUES ('John Doe', 18, 1);")?;
        db.exec("INSERT INTO users(name, age, id) VALUES ('Jane Doe', 22, 2);")?;

        let query = db.exec("SELECT * FROM users;")?;

        assert_eq!(query, QuerySet {
            schema: Schema::new(vec![
                Column::primary_key("id", DataType::Int),
                Column::new("name", DataType::Varchar(255)),
                Column::new("age", DataType::Int),
            ]),
            tuples: vec![
                vec![
                    Value::Number(1),
                    Value::String("John Doe".into()),
                    Value::Number(18)
                ],
                vec![
                    Value::Number(2),
                    Value::String("Jane Doe".into()),
                    Value::Number(22)
                ],
            ]
        });

        Ok(())
    }

    #[test]
    fn insert_expressions() -> Result<(), DbError> {
        let mut db = init_database()?;

        db.exec("CREATE TABLE products (id INT PRIMARY KEY, price INT, discount INT);")?;
        db.exec("INSERT INTO products(id, price, discount) VALUES (1, 10*10 + 10, 2+2);")?;
        db.exec("INSERT INTO products(id, price, discount) VALUES (2, 50 - 5*2, 100 / (3+2));")?;

        let query = db.exec("SELECT * FROM products;")?;

        assert_eq!(query, QuerySet {
            schema: Schema::new(vec![
                Column::primary_key("id", DataType::Int),
                Column::new("price", DataType::Int),
                Column::new("discount", DataType::Int),
            ]),
            tuples: vec![
                vec![Value::Number(1), Value::Number(110), Value::Number(4),],
                vec![Value::Number(2), Value::Number(40), Value::Number(20),],
            ]
        });

        Ok(())
    }

    /// This test really "tests" the limits of the underlying BTrees by using a
    /// really small page size and variable length data that's going to force
    /// the BTrees to allocate a bunch of overflow pages and rebalance many
    /// times. The pager will allocate hundreds of both overflow and slotted
    /// pages. On top of that, we set the cache size to a really small number
    /// to force as many evictions as possible.
    ///
    /// If this one works then I guess we can go home...
    #[cfg(not(miri))]
    #[test]
    fn insert_many() -> Result<(), DbError> {
        let mut db = init_database_with(DbConf {
            page_size: 96,
            cache_size: 8,
        })?;

        let create_table = r#"
            CREATE TABLE users (
                id INT PRIMARY KEY,
                name VARCHAR(255),
                email VARCHAR(255) UNIQUE
            );
        "#;

        db.exec(create_table)?;

        let mut expected_table_entries = Vec::new();
        let mut expected_email_uq_index_entries = Vec::new();

        for i in 1..200 {
            let name = format!("User {i}");
            let email = format!("user{i}@test.com");

            expected_table_entries.push(vec![
                Value::Number(i),
                Value::String(name.clone()),
                Value::String(email.clone()),
            ]);

            expected_email_uq_index_entries
                .push(vec![Value::String(email.clone()), Value::Number(i)]);

            db.exec(&format!(
                "INSERT INTO users(id, name, email) VALUES ({i}, '{name}', '{email}');"
            ))?;
        }

        expected_email_uq_index_entries.sort_by(|a, b| {
            let (Value::String(email_a), Value::String(email_b)) = (&a[0], &b[0]) else {
                unreachable!();
            };

            email_a.cmp(email_b)
        });

        let query = db.exec("SELECT * FROM users;")?;

        assert_eq!(&query, &QuerySet {
            schema: Schema::new(vec![
                Column::primary_key("id", DataType::Int),
                Column::new("name", DataType::Varchar(255)),
                Column::unique("email", DataType::Varchar(255)),
            ]),
            tuples: expected_table_entries.clone()
        });

        expected_table_entries.sort_by(|a, b| {
            let (Value::String(email_a), Value::String(email_b)) = (&a[2], &b[2]) else {
                unreachable!()
            };

            email_a.cmp(email_b)
        });

        assert_index_contains(
            &mut db,
            "users_email_uq_index",
            &expected_email_uq_index_entries,
        )?;

        Ok(())
    }

    #[test]
    fn select_where() -> Result<(), DbError> {
        let mut db = init_database()?;

        db.exec("CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(255), age INT);")?;
        db.exec("INSERT INTO users(id, name, age) VALUES (1, 'John Doe', 18);")?;
        db.exec("INSERT INTO users(id, name, age) VALUES (2, 'Jane Doe', 22);")?;
        db.exec("INSERT INTO users(id, name, age) VALUES (3, 'Some Dude', 24);")?;

        let query = db.exec("SELECT * FROM users WHERE age > 18;")?;

        assert_eq!(query, QuerySet {
            schema: Schema::new(vec![
                Column::primary_key("id", DataType::Int),
                Column::new("name", DataType::Varchar(255)),
                Column::new("age", DataType::Int),
            ]),
            tuples: vec![
                vec![
                    Value::Number(2),
                    Value::String("Jane Doe".into()),
                    Value::Number(22)
                ],
                vec![
                    Value::Number(3),
                    Value::String("Some Dude".into()),
                    Value::Number(24)
                ],
            ]
        });

        Ok(())
    }

    #[test]
    fn select_disordered_columns() -> Result<(), DbError> {
        let mut db = init_database()?;

        db.exec(
            "CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(255), age INT, is_admin BOOL);",
        )?;
        db.exec("INSERT INTO users(id, name, age, is_admin) VALUES (1, 'John Doe', 18, TRUE);")?;
        db.exec("INSERT INTO users(id, name, age, is_admin) VALUES (2, 'Jane Doe', 22, FALSE);")?;

        let query = db.exec("SELECT age, name, id, is_admin FROM users;")?;

        assert_eq!(query, QuerySet {
            schema: Schema::new(vec![
                Column::new("age", DataType::Int),
                Column::new("name", DataType::Varchar(255)),
                Column::primary_key("id", DataType::Int),
                Column::new("is_admin", DataType::Bool),
            ]),
            tuples: vec![
                vec![
                    Value::Number(18),
                    Value::String("John Doe".into()),
                    Value::Number(1),
                    Value::Bool(true),
                ],
                vec![
                    Value::Number(22),
                    Value::String("Jane Doe".into()),
                    Value::Number(2),
                    Value::Bool(false),
                ],
            ]
        });

        Ok(())
    }

    #[test]
    fn select_expressions() -> Result<(), DbError> {
        let mut db = init_database()?;

        db.exec("CREATE TABLE products (id INT PRIMARY KEY, price INT, discount INT);")?;
        db.exec("INSERT INTO products(id, price, discount) VALUES (1, 100, 5);")?;
        db.exec("INSERT INTO products(id, price, discount) VALUES (2, 250, 10);")?;

        let query = db.exec("SELECT id, price / 10, discount * 100 FROM products;")?;

        assert_eq!(query, QuerySet {
            schema: Schema::new(vec![
                Column::primary_key("id", DataType::Int),
                Column::new("price / 10", DataType::BigInt),
                Column::new("discount * 100", DataType::BigInt),
            ]),
            tuples: vec![
                vec![Value::Number(1), Value::Number(10), Value::Number(500),],
                vec![Value::Number(2), Value::Number(25), Value::Number(1000),],
            ]
        });

        Ok(())
    }

    #[cfg(not(miri))]
    #[test]
    fn select_many() -> Result<(), DbError> {
        let mut db = init_database_with(DbConf {
            page_size: 96,
            cache_size: 1024,
        })?;

        db.exec("CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(255));")?;

        let mut expected = Vec::new();

        for i in 1..=500 {
            expected.push(vec![Value::Number(i), Value::String(format!("User{i}"))]);
        }

        for user in expected.iter().rev() {
            db.exec(&format!(
                "INSERT INTO users(id, name) VALUES ({}, {});",
                user[0], user[1]
            ))?;
        }

        let query = db.exec("SELECT * FROM users;")?;

        assert_eq!(query, QuerySet {
            schema: Schema::new(vec![
                Column::primary_key("id", DataType::Int),
                Column::new("name", DataType::Varchar(255)),
            ]),
            tuples: expected,
        });

        Ok(())
    }

    #[test]
    fn select_order_by() -> Result<(), DbError> {
        let mut db = init_database()?;

        db.exec("CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(255), age INT);")?;
        db.exec("INSERT INTO users(id, name, age) VALUES (3, 'Some Dude', 24);")?;
        db.exec("INSERT INTO users(id, name, age) VALUES (2, 'John Doe', 22);")?;
        db.exec("INSERT INTO users(id, name, age) VALUES (1, 'John Doe', 18);")?;

        let query = db.exec("SELECT * FROM users ORDER BY name, age;")?;

        assert_eq!(query, QuerySet {
            schema: Schema::new(vec![
                Column::primary_key("id", DataType::Int),
                Column::new("name", DataType::Varchar(255)),
                Column::new("age", DataType::Int),
            ]),
            tuples: vec![
                vec![
                    Value::Number(1),
                    Value::String("John Doe".into()),
                    Value::Number(18)
                ],
                vec![
                    Value::Number(2),
                    Value::String("John Doe".into()),
                    Value::Number(22)
                ],
                vec![
                    Value::Number(3),
                    Value::String("Some Dude".into()),
                    Value::Number(24)
                ]
            ]
        });

        Ok(())
    }

    // Force the external merge sort algorithm to do some real work.
    #[cfg(not(miri))]
    #[test]
    fn select_order_by_many() -> Result<(), DbError> {
        let mut db = init_database_with(DbConf {
            page_size: 96,
            cache_size: 1024,
        })?;

        db.exec("CREATE TABLE users (name VARCHAR(255));")?;

        let mut expected = Vec::new();

        for i in 1..=500 {
            let name = format!("User{i:03}");
            expected.push(vec![Value::String(name)]);
        }

        for user in expected.iter().rev() {
            db.exec(&format!("INSERT INTO users(name) VALUES ({});", user[0]))?;
        }

        let query = db.exec("SELECT * FROM users ORDER BY name;")?;

        assert_eq!(query, QuerySet {
            schema: Schema::new(vec![Column::new("name", DataType::Varchar(255)),]),
            tuples: expected,
        });

        Ok(())
    }

    // The default page size won't be enough to store these tuples.
    #[test]
    fn select_order_by_large_tuples() -> Result<(), DbError> {
        let mut db = init_database_with(DbConf {
            page_size: 96,
            cache_size: 1024,
        })?;

        db.exec("CREATE TABLE users (id BIGINT, name VARCHAR(255), email VARCHAR(255), age INT);")?;

        let mut expected = Vec::new();

        for i in 1..=10 {
            let name = format!("Very Long User{i} Name").repeat(10);
            let mut email = format!("very_long_user_email{i}").repeat(10);
            email.push_str("@email.com");
            expected.push(vec![
                Value::Number(i),
                Value::String(name),
                Value::String(email),
                Value::Number(i + 20),
            ]);
        }

        for user in expected.iter().rev() {
            db.exec(&format!(
                "INSERT INTO users(id, name, email, age) VALUES ({});",
                user.iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            ))?;
        }

        expected.sort_by(|user1, user2| {
            user1[1]
                .partial_cmp(&user2[1])
                .or(user1[2].partial_cmp(&user2[2]).or_else(|| {
                    let (Value::Number(age1), Value::Number(age2)) = (&user1[3], &user2[3]) else {
                        unreachable!();
                    };

                    (age1 + 10).partial_cmp(&(age2 + 10))
                }))
                .unwrap_or(Ordering::Equal)
        });

        let query = db.exec("SELECT * FROM users ORDER BY name, email, age + 10;")?;

        assert_eq!(query, QuerySet {
            schema: Schema::new(vec![
                Column::new("id", DataType::BigInt),
                Column::new("name", DataType::Varchar(255)),
                Column::new("email", DataType::Varchar(255)),
                Column::new("age", DataType::Int),
            ],),
            tuples: expected,
        });

        Ok(())
    }

    #[test]
    fn select_order_by_expressions() -> Result<(), DbError> {
        let mut db = init_database_with(DbConf {
            page_size: 96,
            cache_size: 1024,
        })?;

        db.exec("CREATE TABLE products (id INT, name VARCHAR(255), price INT, discount INT);")?;

        let mut products = vec![
            (1, "mouse", 20, 10),
            (2, "mouse", 20, 20),
            (3, "keyboard", 40, 25),
            (4, "keyboard", 20, 50),
            (5, "speakers", 40, 10),
        ];

        for (id, name, price, discount) in &products {
            db.exec(&format!("INSERT INTO products(id, name, price, discount) VALUES ({id}, '{name}', {price}, {discount});"))?;
        }

        products.sort_by(
            |(_, name1, price1, discount1), (_, name2, price2, discount2)| {
                let mut ordering = name1.cmp(name2);

                if ordering == Ordering::Equal {
                    ordering = (price1 - price1 * discount1 / 100)
                        .cmp(&(price2 - price2 * discount2 / 100));
                }

                if ordering == Ordering::Equal {
                    ordering = price1.cmp(price2);
                }

                ordering
            },
        );

        let query = db
            .exec("SELECT * FROM products ORDER BY name, price - price * discount / 100, price;")?;

        assert_eq!(query, QuerySet {
            schema: Schema::new(vec![
                Column::new("id", DataType::Int),
                Column::new("name", DataType::Varchar(255)),
                Column::new("price", DataType::Int),
                Column::new("discount", DataType::Int),
            ]),
            tuples: products
                .into_iter()
                .map(|(id, name, price, discount)| vec![
                    Value::Number(id),
                    Value::String(name.into()),
                    Value::Number(price),
                    Value::Number(discount)
                ])
                .collect(),
        });

        Ok(())
    }

    #[test]
    fn select_where_auto_index_exact() -> Result<(), DbError> {
        let mut db = init_database()?;

        db.exec("CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(255), age INT);")?;
        db.exec("INSERT INTO users(id, name, age) VALUES (1, 'John Doe', 18);")?;
        db.exec("INSERT INTO users(id, name, age) VALUES (2, 'Jane Doe', 22);")?;
        db.exec("INSERT INTO users(id, name, age) VALUES (3, 'Some Dude', 24);")?;

        let query = db.exec("SELECT * FROM users WHERE id = 2;")?;

        assert_eq!(query, QuerySet {
            schema: Schema::new(vec![
                Column::primary_key("id", DataType::Int),
                Column::new("name", DataType::Varchar(255)),
                Column::new("age", DataType::Int),
            ]),
            tuples: vec![vec![
                Value::Number(2),
                Value::String("Jane Doe".into()),
                Value::Number(22)
            ],]
        });

        Ok(())
    }

    #[test]
    fn select_where_indexed_column_exact() -> Result<(), DbError> {
        let mut db = init_database()?;

        db.exec(
            "CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(255), email VARCHAR(255) UNIQUE);",
        )?;
        db.exec("INSERT INTO users(id, name, email) VALUES (1, 'Alex', 'alex@email.com');")?;
        db.exec("INSERT INTO users(id, name, email) VALUES (2, 'Bob', 'bob@email.com');")?;
        db.exec("INSERT INTO users(id, name, email) VALUES (3, 'David', 'david@email.com');")?;

        let query = db.exec("SELECT * FROM users WHERE email = 'bob@email.com';")?;

        assert_eq!(query, QuerySet {
            schema: Schema::new(vec![
                Column::primary_key("id", DataType::Int),
                Column::new("name", DataType::Varchar(255)),
                Column::unique("email", DataType::Varchar(255)),
            ]),
            tuples: vec![vec![
                Value::Number(2),
                Value::String("Bob".into()),
                Value::String("bob@email.com".into()),
            ],]
        });

        Ok(())
    }

    #[test]
    fn select_where_auto_index_exact_not_found() -> Result<(), DbError> {
        let mut db = init_database()?;

        db.exec("CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(255), age INT);")?;
        db.exec("INSERT INTO users(id, name, age) VALUES (10, 'John Doe', 18);")?;
        db.exec("INSERT INTO users(id, name, age) VALUES (20, 'Jane Doe', 22);")?;
        db.exec("INSERT INTO users(id, name, age) VALUES (30, 'Some Dude', 24);")?;

        let query = db.exec("SELECT * FROM users WHERE id = 15;")?;

        assert!(query.is_empty());

        Ok(())
    }

    #[test]
    fn select_where_indexed_column_exact_not_found() -> Result<(), DbError> {
        let mut db = init_database()?;

        db.exec(
            "CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(255), email VARCHAR(255) UNIQUE);",
        )?;
        db.exec("INSERT INTO users(id, name, email) VALUES (1, 'Alex', 'alex@email.com');")?;
        db.exec("INSERT INTO users(id, name, email) VALUES (2, 'Bob', 'bob@email.com');")?;
        db.exec("INSERT INTO users(id, name, email) VALUES (3, 'David', 'david@email.com');")?;

        let query = db.exec("SELECT * FROM users WHERE email = 'carla@email.com';")?;

        assert!(query.is_empty());

        Ok(())
    }

    #[test]
    fn select_where_auto_index_less_than() -> Result<(), DbError> {
        let mut db = init_database()?;

        db.exec("CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(255), age INT);")?;
        db.exec("INSERT INTO users(id, name, age) VALUES (1, 'John Doe', 18);")?;
        db.exec("INSERT INTO users(id, name, age) VALUES (2, 'Jane Doe', 22);")?;
        db.exec("INSERT INTO users(id, name, age) VALUES (3, 'Some Dude', 24);")?;
        db.exec("INSERT INTO users(id, name, age) VALUES (4, 'Another Dude', 30);")?;

        let query = db.exec("SELECT * FROM users WHERE id < 3;")?;

        assert_eq!(query, QuerySet {
            schema: Schema::new(vec![
                Column::primary_key("id", DataType::Int),
                Column::new("name", DataType::Varchar(255)),
                Column::new("age", DataType::Int),
            ]),
            tuples: vec![
                vec![
                    Value::Number(1),
                    Value::String("John Doe".into()),
                    Value::Number(18)
                ],
                vec![
                    Value::Number(2),
                    Value::String("Jane Doe".into()),
                    Value::Number(22)
                ],
            ]
        });

        Ok(())
    }

    #[test]
    fn select_where_indexed_column_less_than() -> Result<(), DbError> {
        let mut db = init_database()?;

        db.exec(
            "CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(255), email VARCHAR(255) UNIQUE);",
        )?;
        db.exec("INSERT INTO users(id, name, email) VALUES (1, 'Alex', 'alex@email.com');")?;
        db.exec("INSERT INTO users(id, name, email) VALUES (2, 'Bob', 'bob@email.com');")?;
        db.exec("INSERT INTO users(id, name, email) VALUES (3, 'David', 'david@email.com');")?;

        let query = db.exec("SELECT * FROM users WHERE email < 'david@email.com';")?;

        assert_eq!(query, QuerySet {
            schema: Schema::new(vec![
                Column::primary_key("id", DataType::Int),
                Column::new("name", DataType::Varchar(255)),
                Column::unique("email", DataType::Varchar(255)),
            ]),
            tuples: vec![
                vec![
                    Value::Number(1),
                    Value::String("Alex".into()),
                    Value::String("alex@email.com".into()),
                ],
                vec![
                    Value::Number(2),
                    Value::String("Bob".into()),
                    Value::String("bob@email.com".into()),
                ],
            ]
        });

        Ok(())
    }

    #[test]
    fn select_where_auto_index_less_than_not_found() -> Result<(), DbError> {
        let mut db = init_database()?;

        db.exec("CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(255), age INT);")?;
        db.exec("INSERT INTO users(id, name, age) VALUES (10, 'John Doe', 18);")?;
        db.exec("INSERT INTO users(id, name, age) VALUES (20, 'Jane Doe', 22);")?;
        db.exec("INSERT INTO users(id, name, age) VALUES (30, 'Some Dude', 24);")?;
        db.exec("INSERT INTO users(id, name, age) VALUES (40, 'Another Dude', 30);")?;

        let query = db.exec("SELECT * FROM users WHERE id < 35;")?;

        assert_eq!(query, QuerySet {
            schema: Schema::new(vec![
                Column::primary_key("id", DataType::Int),
                Column::new("name", DataType::Varchar(255)),
                Column::new("age", DataType::Int),
            ]),
            tuples: vec![
                vec![
                    Value::Number(10),
                    Value::String("John Doe".into()),
                    Value::Number(18)
                ],
                vec![
                    Value::Number(20),
                    Value::String("Jane Doe".into()),
                    Value::Number(22)
                ],
                vec![
                    Value::Number(30),
                    Value::String("Some Dude".into()),
                    Value::Number(24)
                ],
            ]
        });

        Ok(())
    }

    #[test]
    fn select_where_indexed_column_less_than_not_found() -> Result<(), DbError> {
        let mut db = init_database()?;

        db.exec(
            "CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(255), email VARCHAR(255) UNIQUE);",
        )?;
        db.exec("INSERT INTO users(id, name, email) VALUES (1, 'Alex', 'alex@email.com');")?;
        db.exec("INSERT INTO users(id, name, email) VALUES (2, 'Bob', 'bob@email.com');")?;
        db.exec("INSERT INTO users(id, name, email) VALUES (3, 'David', 'david@email.com');")?;

        let query = db.exec("SELECT * FROM users WHERE email < 'carla@email.com';")?;

        assert_eq!(query, QuerySet {
            schema: Schema::new(vec![
                Column::primary_key("id", DataType::Int),
                Column::new("name", DataType::Varchar(255)),
                Column::unique("email", DataType::Varchar(255)),
            ]),
            tuples: vec![
                vec![
                    Value::Number(1),
                    Value::String("Alex".into()),
                    Value::String("alex@email.com".into()),
                ],
                vec![
                    Value::Number(2),
                    Value::String("Bob".into()),
                    Value::String("bob@email.com".into()),
                ],
            ]
        });

        Ok(())
    }

    #[test]
    fn select_where_auto_index_greather_than() -> Result<(), DbError> {
        let mut db = init_database()?;

        db.exec("CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(255), age INT);")?;
        db.exec("INSERT INTO users(id, name, age) VALUES (1, 'John Doe', 18);")?;
        db.exec("INSERT INTO users(id, name, age) VALUES (2, 'Jane Doe', 22);")?;
        db.exec("INSERT INTO users(id, name, age) VALUES (3, 'Some Dude', 24);")?;
        db.exec("INSERT INTO users(id, name, age) VALUES (4, 'Another Dude', 30);")?;

        let query = db.exec("SELECT * FROM users WHERE id > 2;")?;

        assert_eq!(query, QuerySet {
            schema: Schema::new(vec![
                Column::primary_key("id", DataType::Int),
                Column::new("name", DataType::Varchar(255)),
                Column::new("age", DataType::Int),
            ]),
            tuples: vec![
                vec![
                    Value::Number(3),
                    Value::String("Some Dude".into()),
                    Value::Number(24)
                ],
                vec![
                    Value::Number(4),
                    Value::String("Another Dude".into()),
                    Value::Number(30)
                ],
            ]
        });

        Ok(())
    }

    #[test]
    fn select_where_auto_index_greater_than_not_found() -> Result<(), DbError> {
        let mut db = init_database()?;

        db.exec("CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(255), age INT);")?;
        db.exec("INSERT INTO users(id, name, age) VALUES (10, 'John Doe', 18);")?;
        db.exec("INSERT INTO users(id, name, age) VALUES (20, 'Jane Doe', 22);")?;
        db.exec("INSERT INTO users(id, name, age) VALUES (30, 'Some Dude', 24);")?;
        db.exec("INSERT INTO users(id, name, age) VALUES (40, 'Another Dude', 30);")?;

        let query = db.exec("SELECT * FROM users WHERE id > 25;")?;

        assert_eq!(query, QuerySet {
            schema: Schema::new(vec![
                Column::primary_key("id", DataType::Int),
                Column::new("name", DataType::Varchar(255)),
                Column::new("age", DataType::Int),
            ]),
            tuples: vec![
                vec![
                    Value::Number(30),
                    Value::String("Some Dude".into()),
                    Value::Number(24)
                ],
                vec![
                    Value::Number(40),
                    Value::String("Another Dude".into()),
                    Value::Number(30)
                ],
            ]
        });

        Ok(())
    }

    #[test]
    fn select_where_indexed_column_greater_than() -> Result<(), DbError> {
        let mut db = init_database()?;

        db.exec(
            "CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(255), email VARCHAR(255) UNIQUE);",
        )?;
        db.exec("INSERT INTO users(id, name, email) VALUES (1, 'Alex', 'alex@email.com');")?;
        db.exec("INSERT INTO users(id, name, email) VALUES (2, 'Bob', 'bob@email.com');")?;
        db.exec("INSERT INTO users(id, name, email) VALUES (3, 'David', 'david@email.com');")?;

        let query = db.exec("SELECT * FROM users WHERE email > 'bob@email.com';")?;

        assert_eq!(query, QuerySet {
            schema: Schema::new(vec![
                Column::primary_key("id", DataType::Int),
                Column::new("name", DataType::Varchar(255)),
                Column::unique("email", DataType::Varchar(255)),
            ]),
            tuples: vec![vec![
                Value::Number(3),
                Value::String("David".into()),
                Value::String("david@email.com".into()),
            ],]
        });

        Ok(())
    }

    #[test]
    fn select_where_indexed_column_greater_than_not_found() -> Result<(), DbError> {
        let mut db = init_database()?;

        db.exec(
            "CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(255), email VARCHAR(255) UNIQUE);",
        )?;
        db.exec("INSERT INTO users(id, name, email) VALUES (1, 'Alex', 'alex@email.com');")?;
        db.exec("INSERT INTO users(id, name, email) VALUES (2, 'Bob', 'bob@email.com');")?;
        db.exec("INSERT INTO users(id, name, email) VALUES (3, 'David', 'david@email.com');")?;

        let query = db.exec("SELECT * FROM users WHERE email > 'carla@email.com';")?;

        assert_eq!(query, QuerySet {
            schema: Schema::new(vec![
                Column::primary_key("id", DataType::Int),
                Column::new("name", DataType::Varchar(255)),
                Column::unique("email", DataType::Varchar(255)),
            ]),
            tuples: vec![vec![
                Value::Number(3),
                Value::String("David".into()),
                Value::String("david@email.com".into()),
            ]]
        });

        Ok(())
    }

    #[test]
    fn select_where_auto_index_less_than_or_equal() -> Result<(), DbError> {
        let mut db = init_database()?;

        db.exec("CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(255), age INT);")?;
        db.exec("INSERT INTO users(id, name, age) VALUES (1, 'John Doe', 18);")?;
        db.exec("INSERT INTO users(id, name, age) VALUES (2, 'Jane Doe', 22);")?;
        db.exec("INSERT INTO users(id, name, age) VALUES (3, 'Some Dude', 24);")?;
        db.exec("INSERT INTO users(id, name, age) VALUES (4, 'Another Dude', 30);")?;

        let query = db.exec("SELECT * FROM users WHERE id <= 3;")?;

        assert_eq!(query, QuerySet {
            schema: Schema::new(vec![
                Column::primary_key("id", DataType::Int),
                Column::new("name", DataType::Varchar(255)),
                Column::new("age", DataType::Int),
            ]),
            tuples: vec![
                vec![
                    Value::Number(1),
                    Value::String("John Doe".into()),
                    Value::Number(18)
                ],
                vec![
                    Value::Number(2),
                    Value::String("Jane Doe".into()),
                    Value::Number(22)
                ],
                vec![
                    Value::Number(3),
                    Value::String("Some Dude".into()),
                    Value::Number(24)
                ],
            ]
        });

        Ok(())
    }

    #[test]
    fn select_where_auto_index_less_than_or_equal_not_found() -> Result<(), DbError> {
        let mut db = init_database()?;

        db.exec("CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(255), age INT);")?;
        db.exec("INSERT INTO users(id, name, age) VALUES (10, 'John Doe', 18);")?;
        db.exec("INSERT INTO users(id, name, age) VALUES (20, 'Jane Doe', 22);")?;
        db.exec("INSERT INTO users(id, name, age) VALUES (30, 'Some Dude', 24);")?;
        db.exec("INSERT INTO users(id, name, age) VALUES (40, 'Another Dude', 30);")?;

        let query = db.exec("SELECT * FROM users WHERE id <= 25;")?;

        assert_eq!(query, QuerySet {
            schema: Schema::new(vec![
                Column::primary_key("id", DataType::Int),
                Column::new("name", DataType::Varchar(255)),
                Column::new("age", DataType::Int),
            ]),
            tuples: vec![
                vec![
                    Value::Number(10),
                    Value::String("John Doe".into()),
                    Value::Number(18)
                ],
                vec![
                    Value::Number(20),
                    Value::String("Jane Doe".into()),
                    Value::Number(22)
                ],
            ]
        });

        Ok(())
    }

    #[test]
    fn select_where_indexed_column_less_than_or_equal() -> Result<(), DbError> {
        let mut db = init_database()?;

        db.exec(
            "CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(255), email VARCHAR(255) UNIQUE);",
        )?;
        db.exec("INSERT INTO users(id, name, email) VALUES (1, 'Alex', 'alex@email.com');")?;
        db.exec("INSERT INTO users(id, name, email) VALUES (2, 'Bob', 'bob@email.com');")?;
        db.exec("INSERT INTO users(id, name, email) VALUES (3, 'David', 'david@email.com');")?;

        let query = db.exec("SELECT * FROM users WHERE email <= 'bob@email.com';")?;

        assert_eq!(query, QuerySet {
            schema: Schema::new(vec![
                Column::primary_key("id", DataType::Int),
                Column::new("name", DataType::Varchar(255)),
                Column::unique("email", DataType::Varchar(255)),
            ]),
            tuples: vec![
                vec![
                    Value::Number(1),
                    Value::String("Alex".into()),
                    Value::String("alex@email.com".into()),
                ],
                vec![
                    Value::Number(2),
                    Value::String("Bob".into()),
                    Value::String("bob@email.com".into()),
                ]
            ]
        });

        Ok(())
    }

    #[test]
    fn select_where_indexed_column_less_than_or_equal_not_found() -> Result<(), DbError> {
        let mut db = init_database()?;

        db.exec(
            "CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(255), email VARCHAR(255) UNIQUE);",
        )?;
        db.exec("INSERT INTO users(id, name, email) VALUES (1, 'Alex', 'alex@email.com');")?;
        db.exec("INSERT INTO users(id, name, email) VALUES (2, 'Bob', 'bob@email.com');")?;
        db.exec("INSERT INTO users(id, name, email) VALUES (3, 'David', 'david@email.com');")?;

        let query = db.exec("SELECT * FROM users WHERE email <= 'carla@email.com';")?;

        assert_eq!(query, QuerySet {
            schema: Schema::new(vec![
                Column::primary_key("id", DataType::Int),
                Column::new("name", DataType::Varchar(255)),
                Column::unique("email", DataType::Varchar(255)),
            ]),
            tuples: vec![
                vec![
                    Value::Number(1),
                    Value::String("Alex".into()),
                    Value::String("alex@email.com".into()),
                ],
                vec![
                    Value::Number(2),
                    Value::String("Bob".into()),
                    Value::String("bob@email.com".into()),
                ]
            ]
        });

        Ok(())
    }

    #[test]
    fn select_where_auto_index_greather_than_or_equal() -> Result<(), DbError> {
        let mut db = init_database()?;

        db.exec("CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(255), age INT);")?;
        db.exec("INSERT INTO users(id, name, age) VALUES (1, 'John Doe', 18);")?;
        db.exec("INSERT INTO users(id, name, age) VALUES (2, 'Jane Doe', 22);")?;
        db.exec("INSERT INTO users(id, name, age) VALUES (3, 'Some Dude', 24);")?;
        db.exec("INSERT INTO users(id, name, age) VALUES (4, 'Another Dude', 30);")?;

        let query = db.exec("SELECT * FROM users WHERE id >= 3;")?;

        assert_eq!(query, QuerySet {
            schema: Schema::new(vec![
                Column::primary_key("id", DataType::Int),
                Column::new("name", DataType::Varchar(255)),
                Column::new("age", DataType::Int),
            ]),
            tuples: vec![
                vec![
                    Value::Number(3),
                    Value::String("Some Dude".into()),
                    Value::Number(24)
                ],
                vec![
                    Value::Number(4),
                    Value::String("Another Dude".into()),
                    Value::Number(30)
                ],
            ]
        });

        Ok(())
    }

    #[test]
    fn select_where_auto_index_greather_than_or_equal_not_found() -> Result<(), DbError> {
        let mut db = init_database()?;

        db.exec("CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(255), age INT);")?;
        db.exec("INSERT INTO users(id, name, age) VALUES (10, 'John Doe', 18);")?;
        db.exec("INSERT INTO users(id, name, age) VALUES (20, 'Jane Doe', 22);")?;
        db.exec("INSERT INTO users(id, name, age) VALUES (30, 'Some Dude', 24);")?;
        db.exec("INSERT INTO users(id, name, age) VALUES (40, 'Another Dude', 30);")?;

        let query = db.exec("SELECT * FROM users WHERE id >= 15;")?;

        assert_eq!(query, QuerySet {
            schema: Schema::new(vec![
                Column::primary_key("id", DataType::Int),
                Column::new("name", DataType::Varchar(255)),
                Column::new("age", DataType::Int),
            ]),
            tuples: vec![
                vec![
                    Value::Number(20),
                    Value::String("Jane Doe".into()),
                    Value::Number(22)
                ],
                vec![
                    Value::Number(30),
                    Value::String("Some Dude".into()),
                    Value::Number(24)
                ],
                vec![
                    Value::Number(40),
                    Value::String("Another Dude".into()),
                    Value::Number(30)
                ],
            ]
        });

        Ok(())
    }

    #[test]
    fn select_where_indexed_column_greater_than_or_equal() -> Result<(), DbError> {
        let mut db = init_database()?;

        db.exec(
            "CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(255), email VARCHAR(255) UNIQUE);",
        )?;
        db.exec("INSERT INTO users(id, name, email) VALUES (1, 'Alex', 'alex@email.com');")?;
        db.exec("INSERT INTO users(id, name, email) VALUES (2, 'Bob', 'bob@email.com');")?;
        db.exec("INSERT INTO users(id, name, email) VALUES (3, 'David', 'david@email.com');")?;

        let query = db.exec("SELECT * FROM users WHERE email >= 'bob@email.com';")?;

        assert_eq!(query, QuerySet {
            schema: Schema::new(vec![
                Column::primary_key("id", DataType::Int),
                Column::new("name", DataType::Varchar(255)),
                Column::unique("email", DataType::Varchar(255)),
            ]),
            tuples: vec![
                vec![
                    Value::Number(2),
                    Value::String("Bob".into()),
                    Value::String("bob@email.com".into()),
                ],
                vec![
                    Value::Number(3),
                    Value::String("David".into()),
                    Value::String("david@email.com".into()),
                ],
            ]
        });

        Ok(())
    }

    #[test]
    fn select_where_indexed_column_greater_than_or_equal_not_found() -> Result<(), DbError> {
        let mut db = init_database()?;

        db.exec(
            "CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(255), email VARCHAR(255) UNIQUE);",
        )?;
        db.exec("INSERT INTO users(id, name, email) VALUES (1, 'Alex', 'alex@email.com');")?;
        db.exec("INSERT INTO users(id, name, email) VALUES (2, 'Bob', 'bob@email.com');")?;
        db.exec("INSERT INTO users(id, name, email) VALUES (3, 'David', 'david@email.com');")?;
        db.exec("INSERT INTO users(id, name, email) VALUES (4, 'John', 'john@email.com');")?;

        let query = db.exec("SELECT * FROM users WHERE email >= 'carla@email.com';")?;

        assert_eq!(query, QuerySet {
            schema: Schema::new(vec![
                Column::primary_key("id", DataType::Int),
                Column::new("name", DataType::Varchar(255)),
                Column::unique("email", DataType::Varchar(255)),
            ]),
            tuples: vec![
                vec![
                    Value::Number(3),
                    Value::String("David".into()),
                    Value::String("david@email.com".into()),
                ],
                vec![
                    Value::Number(4),
                    Value::String("John".into()),
                    Value::String("john@email.com".into()),
                ],
            ]
        });

        Ok(())
    }

    #[test]
    fn select_where_multiple_key_ranges() -> Result<(), DbError> {
        let mut db = init_database()?;

        db.exec(
            "CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(255), email VARCHAR(255) UNIQUE);",
        )?;
        db.exec("INSERT INTO users(id, name, email) VALUES (1, 'Alex', 'alex@email.com');")?;
        db.exec("INSERT INTO users(id, name, email) VALUES (2, 'Bob', 'bob@email.com');")?;
        db.exec("INSERT INTO users(id, name, email) VALUES (3, 'David', 'david@email.com');")?;
        db.exec("INSERT INTO users(id, name, email) VALUES (4, 'Mark', 'mark@email.com');")?;
        db.exec("INSERT INTO users(id, name, email) VALUES (5, 'Carla', 'carla@email.com');")?;
        db.exec("INSERT INTO users(id, name, email) VALUES (6, 'Mia', 'mia@email.com');")?;
        db.exec("INSERT INTO users(id, name, email) VALUES (7, 'Jay', 'jay@email.com');")?;
        db.exec("INSERT INTO users(id, name, email) VALUES (8, 'Lauren', 'lauren@email.com');")?;

        let query = db.exec(
            "SELECT * FROM users WHERE (id >= 2 AND id <= 3) OR (id > 4 AND id < 7) OR id >= 8;",
        )?;

        assert_eq!(query, QuerySet {
            schema: Schema::new(vec![
                Column::primary_key("id", DataType::Int),
                Column::new("name", DataType::Varchar(255)),
                Column::unique("email", DataType::Varchar(255)),
            ]),
            tuples: vec![
                vec![
                    Value::Number(2),
                    Value::String("Bob".into()),
                    Value::String("bob@email.com".into()),
                ],
                vec![
                    Value::Number(3),
                    Value::String("David".into()),
                    Value::String("david@email.com".into()),
                ],
                vec![
                    Value::Number(5),
                    Value::String("Carla".into()),
                    Value::String("carla@email.com".into()),
                ],
                vec![
                    Value::Number(6),
                    Value::String("Mia".into()),
                    Value::String("mia@email.com".into()),
                ],
                vec![
                    Value::Number(8),
                    Value::String("Lauren".into()),
                    Value::String("lauren@email.com".into()),
                ],
            ]
        });

        Ok(())
    }

    #[test]
    fn select_where_ranges_cancel() -> Result<(), DbError> {
        let mut db = init_database()?;

        db.exec(
            "CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(255), email VARCHAR(255) UNIQUE);",
        )?;
        db.exec("INSERT INTO users(id, name, email) VALUES (1, 'Alex', 'alex@email.com');")?;
        db.exec("INSERT INTO users(id, name, email) VALUES (2, 'Bob', 'bob@email.com');")?;
        db.exec("INSERT INTO users(id, name, email) VALUES (3, 'David', 'david@email.com');")?;
        db.exec("INSERT INTO users(id, name, email) VALUES (4, 'Mark', 'mark@email.com');")?;
        db.exec("INSERT INTO users(id, name, email) VALUES (5, 'Carla', 'carla@email.com');")?;
        db.exec("INSERT INTO users(id, name, email) VALUES (6, 'Mia', 'mia@email.com');")?;
        db.exec("INSERT INTO users(id, name, email) VALUES (7, 'Jay', 'jay@email.com');")?;
        db.exec("INSERT INTO users(id, name, email) VALUES (8, 'Lauren', 'lauren@email.com');")?;

        let query = db.exec(
            "SELECT * FROM users WHERE (id >= 2 AND id <= 3) AND (id > 4 AND id < 7) AND id >= 8;",
        )?;

        assert_eq!(query, QuerySet {
            schema: Schema::new(vec![
                Column::primary_key("id", DataType::Int),
                Column::new("name", DataType::Varchar(255)),
                Column::unique("email", DataType::Varchar(255)),
            ]),
            tuples: vec![]
        });

        Ok(())
    }

    #[test]
    fn select_where_multiple_indexes_exact_match_and_ranges() -> Result<(), DbError> {
        let mut db = init_database()?;

        db.exec(
            "CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(255) UNIQUE, email VARCHAR(255) UNIQUE);",
        )?;
        db.exec("INSERT INTO users(id, name, email) VALUES (1, 'Alex', 'alex@email.com');")?;
        db.exec("INSERT INTO users(id, name, email) VALUES (2, 'Bob', 'bob@email.com');")?;
        db.exec("INSERT INTO users(id, name, email) VALUES (3, 'David', 'david@email.com');")?;
        db.exec("INSERT INTO users(id, name, email) VALUES (4, 'Mark', 'mark@email.com');")?;
        db.exec("INSERT INTO users(id, name, email) VALUES (5, 'Carla', 'carla@email.com');")?;
        db.exec("INSERT INTO users(id, name, email) VALUES (6, 'Mia', 'mia@email.com');")?;
        db.exec("INSERT INTO users(id, name, email) VALUES (7, 'Jay', 'jay@email.com');")?;
        db.exec("INSERT INTO users(id, name, email) VALUES (8, 'Lauren', 'lauren@email.com');")?;

        let query = db.exec(
            "SELECT * FROM users WHERE id < 4 OR email = 'mia@email.com' OR name = 'Lauren';",
        )?;

        assert_eq!(query, QuerySet {
            schema: Schema::new(vec![
                Column::primary_key("id", DataType::Int),
                Column::unique("name", DataType::Varchar(255)),
                Column::unique("email", DataType::Varchar(255)),
            ]),
            tuples: vec![
                vec![
                    Value::Number(1),
                    Value::String("Alex".into()),
                    Value::String("alex@email.com".into()),
                ],
                vec![
                    Value::Number(2),
                    Value::String("Bob".into()),
                    Value::String("bob@email.com".into()),
                ],
                vec![
                    Value::Number(3),
                    Value::String("David".into()),
                    Value::String("david@email.com".into()),
                ],
                vec![
                    Value::Number(6),
                    Value::String("Mia".into()),
                    Value::String("mia@email.com".into()),
                ],
                vec![
                    Value::Number(8),
                    Value::String("Lauren".into()),
                    Value::String("lauren@email.com".into()),
                ],
            ]
        });

        Ok(())
    }

    #[test]
    fn select_where_multiple_indexes_ranges_only() -> Result<(), DbError> {
        let mut db = init_database()?;

        db.exec(
            "CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(255), email VARCHAR(255) UNIQUE);",
        )?;
        db.exec("INSERT INTO users(id, name, email) VALUES (1, 'Alex', '1@email.com');")?;
        db.exec("INSERT INTO users(id, name, email) VALUES (2, 'Bob', '2@email.com');")?;
        db.exec("INSERT INTO users(id, name, email) VALUES (3, 'David', '3@email.com');")?;
        db.exec("INSERT INTO users(id, name, email) VALUES (4, 'Mark', '4@email.com');")?;
        db.exec("INSERT INTO users(id, name, email) VALUES (5, 'Carla', '5@email.com');")?;
        db.exec("INSERT INTO users(id, name, email) VALUES (6, 'Mia', '6@email.com');")?;
        db.exec("INSERT INTO users(id, name, email) VALUES (7, 'Jay', '7@email.com');")?;
        db.exec("INSERT INTO users(id, name, email) VALUES (8, 'Lauren', '8@email.com');")?;

        let query = db.exec("SELECT * FROM users WHERE id >= 7 OR email < '3@email.com';")?;

        assert_eq!(query, QuerySet {
            schema: Schema::new(vec![
                Column::primary_key("id", DataType::Int),
                Column::new("name", DataType::Varchar(255)),
                Column::unique("email", DataType::Varchar(255)),
            ]),
            tuples: vec![
                vec![
                    Value::Number(1),
                    Value::String("Alex".into()),
                    Value::String("1@email.com".into()),
                ],
                vec![
                    Value::Number(2),
                    Value::String("Bob".into()),
                    Value::String("2@email.com".into()),
                ],
                vec![
                    Value::Number(7),
                    Value::String("Jay".into()),
                    Value::String("7@email.com".into()),
                ],
                vec![
                    Value::Number(8),
                    Value::String("Lauren".into()),
                    Value::String("8@email.com".into()),
                ],
            ]
        });

        Ok(())
    }

    #[test]
    fn select_where_multiple_indexes_multiple_ranges() -> Result<(), DbError> {
        let mut db = init_database()?;

        db.exec(
            "CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(255), email VARCHAR(255) UNIQUE);",
        )?;
        db.exec("INSERT INTO users(id, name, email) VALUES (1, 'Alex', '1@email.com');")?;
        db.exec("INSERT INTO users(id, name, email) VALUES (2, 'Bob', '2@email.com');")?;
        db.exec("INSERT INTO users(id, name, email) VALUES (3, 'David', '3@email.com');")?;
        db.exec("INSERT INTO users(id, name, email) VALUES (4, 'Mark', '4@email.com');")?;
        db.exec("INSERT INTO users(id, name, email) VALUES (5, 'Carla', '5@email.com');")?;
        db.exec("INSERT INTO users(id, name, email) VALUES (6, 'Mia', '6@email.com');")?;
        db.exec("INSERT INTO users(id, name, email) VALUES (7, 'Jay', '7@email.com');")?;
        db.exec("INSERT INTO users(id, name, email) VALUES (8, 'Lauren', '8@email.com');")?;
        db.exec("INSERT INTO users(id, name, email) VALUES (9, 'Jack', '9@email.com');")?;

        #[rustfmt::skip]
        let query = db.exec("
            SELECT * FROM users
            WHERE (id >= 4 AND id < 6)
            OR email < '3@email.com'
            OR (id > 6 AND id < 9)
            OR email = '9@email.com';
        ")?;

        assert_eq!(query, QuerySet {
            schema: Schema::new(vec![
                Column::primary_key("id", DataType::Int),
                Column::new("name", DataType::Varchar(255)),
                Column::unique("email", DataType::Varchar(255)),
            ]),
            tuples: vec![
                vec![
                    Value::Number(1),
                    Value::String("Alex".into()),
                    Value::String("1@email.com".into()),
                ],
                vec![
                    Value::Number(2),
                    Value::String("Bob".into()),
                    Value::String("2@email.com".into()),
                ],
                vec![
                    Value::Number(4),
                    Value::String("Mark".into()),
                    Value::String("4@email.com".into()),
                ],
                vec![
                    Value::Number(5),
                    Value::String("Carla".into()),
                    Value::String("5@email.com".into()),
                ],
                vec![
                    Value::Number(7),
                    Value::String("Jay".into()),
                    Value::String("7@email.com".into()),
                ],
                vec![
                    Value::Number(8),
                    Value::String("Lauren".into()),
                    Value::String("8@email.com".into()),
                ],
                vec![
                    Value::Number(9),
                    Value::String("Jack".into()),
                    Value::String("9@email.com".into()),
                ],
            ]
        });

        Ok(())
    }

    #[cfg(not(miri))]
    #[test]
    fn large_auto_index_scan() -> Result<(), DbError> {
        let mut db = init_database_with(DbConf {
            page_size: 96,
            cache_size: 1024,
        })?;

        db.exec("CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(255));")?;

        let mut users = Vec::new();

        for i in 1..=400 {
            let name = format!("User{i:03}");
            users.push(vec![Value::Number(i), Value::String(name)]);
        }

        for user in &users {
            db.exec(&format!(
                "INSERT INTO users(id, name) VALUES ({}, {});",
                user[0], user[1]
            ))?;
        }

        let total_users = users.len();
        users.drain(..users.len() / 2);

        let query = db.exec(&format!(
            "SELECT * FROM users WHERE id > {};",
            total_users / 2
        ))?;

        assert_eq!(query, QuerySet {
            schema: Schema::new(vec![
                Column::primary_key("id", DataType::Int),
                Column::new("name", DataType::Varchar(255)),
            ]),
            tuples: users,
        });

        Ok(())
    }

    #[cfg(not(miri))]
    #[test]
    fn large_auto_index_scan_key_not_found() -> Result<(), DbError> {
        let mut db = init_database_with(DbConf {
            page_size: 96,
            cache_size: 1024,
        })?;

        db.exec("CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(255));")?;

        let mut users = Vec::new();

        let dont_insert_key = 200;

        for i in 1..=400 {
            if i != dont_insert_key {
                let name = format!("User{i:03}");
                db.exec(&format!(
                    "INSERT INTO users(id, name) VALUES ({i}, '{name}');"
                ))?;

                if i > dont_insert_key {
                    users.push(vec![Value::Number(i), Value::String(name)]);
                }
            }
        }

        let query = db.exec(&format!(
            "SELECT * FROM users WHERE id > {dont_insert_key};"
        ))?;

        assert_eq!(query, QuerySet {
            schema: Schema::new(vec![
                Column::primary_key("id", DataType::Int),
                Column::new("name", DataType::Varchar(255)),
            ]),
            tuples: users,
        });

        Ok(())
    }

    // Binary search will yield "should be at index 0".
    #[test]
    fn select_where_index_key_should_be_first_on_page() -> Result<(), DbError> {
        let mut db = init_database()?;

        db.exec(
            "CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(255), email VARCHAR(255) UNIQUE);",
        )?;
        db.exec("INSERT INTO users(id, name, email) VALUES (2, 'Alex', 'alex@email.com');")?;
        db.exec("INSERT INTO users(id, name, email) VALUES (3, 'Bob', 'bob@email.com');")?;
        db.exec("INSERT INTO users(id, name, email) VALUES (4, 'David', 'david@email.com');")?;

        let query = db.exec("SELECT * FROM users WHERE id > 1;")?;

        assert_eq!(query, QuerySet {
            schema: Schema::new(vec![
                Column::primary_key("id", DataType::Int),
                Column::new("name", DataType::Varchar(255)),
                Column::unique("email", DataType::Varchar(255)),
            ]),
            tuples: vec![
                vec![
                    Value::Number(2),
                    Value::String("Alex".into()),
                    Value::String("alex@email.com".into()),
                ],
                vec![
                    Value::Number(3),
                    Value::String("Bob".into()),
                    Value::String("bob@email.com".into()),
                ],
                vec![
                    Value::Number(4),
                    Value::String("David".into()),
                    Value::String("david@email.com".into()),
                ],
            ]
        });

        Ok(())
    }

    // BTree is configured to store at least 4 keys in each page by default. If
    // we store large rows in a small page we'll be guaranteed to only have 4
    // keys in that page because we can't fit anymore. We'll use that to test
    // the edge case where a binary search results in out of bounds index.
    #[test]
    fn select_where_index_key_should_be_last_on_page() -> Result<(), DbError> {
        let mut db = init_database_with(DbConf {
            page_size: 96,
            cache_size: 256,
        })?;

        let num_pages = 3;
        let mut users = Vec::new();

        let mut id = 1;

        for _ in 0..num_pages {
            for _ in 0..4 {
                users.push(vec![
                    Value::Number(id),
                    Value::String(format!("User{id}").repeat(20)),
                    Value::String(format!("user{id}.com")),
                ]);
                id += 1;
            }
            // Skip IDs to query one that doesn't exist later.
            id += 1;
        }

        db.exec(
            "CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(255), email VARCHAR(255) UNIQUE);",
        )?;

        for user in &users {
            db.exec(&format!(
                "INSERT INTO users (id, name, email) VALUES ({}, {}, {});",
                user[0], user[1], user[2]
            ))?;
        }

        users.drain(..4);
        let query = db.exec("SELECT * FROM users WHERE id > 5;")?;

        assert_eq!(query, QuerySet {
            schema: Schema::new(vec![
                Column::primary_key("id", DataType::Int),
                Column::new("name", DataType::Varchar(255)),
                Column::unique("email", DataType::Varchar(255)),
            ]),
            tuples: users
        });

        Ok(())
    }

    #[test]
    fn delete_all() -> Result<(), DbError> {
        let mut db = init_database()?;

        db.exec("CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(255), email VARCHAR(255) UNIQUE, age INT);")?;
        db.exec(
            "INSERT INTO users(id, name, email, age) VALUES (1, 'John Doe', 'john@doe.com', 18);",
        )?;
        db.exec(
            "INSERT INTO users(id, name, email, age) VALUES (2, 'Jane Doe', 'jane@doe.com', 22);",
        )?;
        db.exec(
            "INSERT INTO users(id, name, email, age) VALUES (3, 'Some Dude', 'some@dude.com', 24);",
        )?;

        db.exec("DELETE FROM users;")?;

        let query = db.exec("SELECT * FROM users;")?;

        assert!(query.tuples.is_empty());

        assert_index_contains(&mut db, "users_email_uq_index", &[])
    }

    #[test]
    fn delete_where() -> Result<(), DbError> {
        let mut db = init_database()?;

        db.exec("CREATE TABLE users (id INT PRIMARY KEY, email VARCHAR(255) UNIQUE, age INT);")?;
        db.exec("INSERT INTO users(id, email, age) VALUES (1, 'john@email.com', 18);")?;
        db.exec("INSERT INTO users(id, email, age) VALUES (2, 'jane@email.com', 22);")?;
        db.exec("INSERT INTO users(id, email, age) VALUES (3, 'some_dude@email.com', 24);")?;

        db.exec("DELETE FROM users WHERE age > 18;")?;

        let query = db.exec("SELECT * FROM users;")?;

        assert_eq!(query, QuerySet {
            schema: Schema::new(vec![
                Column::primary_key("id", DataType::Int),
                Column::unique("email", DataType::Varchar(255)),
                Column::new("age", DataType::Int),
            ]),
            tuples: vec![vec![
                Value::Number(1),
                Value::String("john@email.com".into()),
                Value::Number(18)
            ]]
        });

        assert_index_contains(&mut db, "users_email_uq_index", &[vec![
            Value::String("john@email.com".into()),
            Value::Number(1),
        ]])
    }

    // If buffering doesn't work correctly the delete should destroy the cursor
    // of the scan plan.
    #[test]
    fn delete_where_auto_index_exact() -> Result<(), DbError> {
        let mut db = init_database()?;

        db.exec("CREATE TABLE users (id INT PRIMARY KEY, email VARCHAR(255) UNIQUE, age INT);")?;
        db.exec("INSERT INTO users(id, email, age) VALUES (1, 'john@email.com', 18);")?;
        db.exec("INSERT INTO users(id, email, age) VALUES (2, 'jane@email.com', 22);")?;
        db.exec("INSERT INTO users(id, email, age) VALUES (3, 'some_dude@email.com', 24);")?;

        db.exec("DELETE FROM users WHERE id = 2;")?;

        let query = db.exec("SELECT * FROM users;")?;

        assert_eq!(query, QuerySet {
            schema: Schema::new(vec![
                Column::primary_key("id", DataType::Int),
                Column::unique("email", DataType::Varchar(255)),
                Column::new("age", DataType::Int),
            ]),
            tuples: vec![
                vec![
                    Value::Number(1),
                    Value::String("john@email.com".into()),
                    Value::Number(18)
                ],
                vec![
                    Value::Number(3),
                    Value::String("some_dude@email.com".into()),
                    Value::Number(24)
                ]
            ]
        });

        assert_index_contains(&mut db, "users_email_uq_index", &[
            vec![Value::String("john@email.com".into()), Value::Number(1)],
            vec![
                Value::String("some_dude@email.com".into()),
                Value::Number(3),
            ],
        ])
    }

    // Again, if buffering doesn't work this will fail.
    #[test]
    fn delete_where_auto_index_ranged() -> Result<(), DbError> {
        let mut db = init_database()?;

        db.exec("CREATE TABLE users (id INT PRIMARY KEY, email VARCHAR(255) UNIQUE, age INT);")?;
        db.exec("INSERT INTO users(id, email, age) VALUES (1, 'john@email.com', 18);")?;
        db.exec("INSERT INTO users(id, email, age) VALUES (2, 'jane@email.com', 22);")?;
        db.exec("INSERT INTO users(id, email, age) VALUES (3, 'some_dude@email.com', 24);")?;

        db.exec("DELETE FROM users WHERE id >= 2;")?;

        let query = db.exec("SELECT * FROM users;")?;

        assert_eq!(query, QuerySet {
            schema: Schema::new(vec![
                Column::primary_key("id", DataType::Int),
                Column::unique("email", DataType::Varchar(255)),
                Column::new("age", DataType::Int),
            ]),
            tuples: vec![vec![
                Value::Number(1),
                Value::String("john@email.com".into()),
                Value::Number(18)
            ],]
        });

        assert_index_contains(&mut db, "users_email_uq_index", &[vec![
            Value::String("john@email.com".into()),
            Value::Number(1),
        ]])
    }

    #[test]
    fn delete_where_indexed_column_ranged() -> Result<(), DbError> {
        let mut db = init_database()?;

        db.exec("CREATE TABLE users (id INT PRIMARY KEY, email VARCHAR(255) UNIQUE, age INT);")?;
        db.exec("INSERT INTO users(id, email, age) VALUES (1, 'john@email.com', 18);")?;
        db.exec("INSERT INTO users(id, email, age) VALUES (2, 'jane@email.com', 22);")?;
        db.exec("INSERT INTO users(id, email, age) VALUES (3, 'some_dude@email.com', 24);")?;

        db.exec("DELETE FROM users WHERE email >= 'john@email.com';")?;

        let query = db.exec("SELECT * FROM users;")?;

        assert_eq!(query, QuerySet {
            schema: Schema::new(vec![
                Column::primary_key("id", DataType::Int),
                Column::unique("email", DataType::Varchar(255)),
                Column::new("age", DataType::Int),
            ]),
            tuples: vec![vec![
                Value::Number(2),
                Value::String("jane@email.com".into()),
                Value::Number(22)
            ]]
        });

        assert_index_contains(&mut db, "users_email_uq_index", &[vec![
            Value::String("jane@email.com".into()),
            Value::Number(2),
        ]])
    }

    #[test]
    fn delete_where_auto_index_key_no_match() -> Result<(), DbError> {
        let mut db = init_database()?;

        db.exec("CREATE TABLE users (id INT PRIMARY KEY, email VARCHAR(255) UNIQUE, age INT);")?;
        db.exec("INSERT INTO users(id, email, age) VALUES (1, 'john@email.com', 18);")?;
        db.exec("INSERT INTO users(id, email, age) VALUES (2, 'jane@email.com', 22);")?;
        db.exec("INSERT INTO users(id, email, age) VALUES (3, 'some_dude@email.com', 24);")?;

        db.exec("DELETE FROM users WHERE id >= 4;")?;

        let query = db.exec("SELECT * FROM users;")?;

        assert_eq!(query, QuerySet {
            schema: Schema::new(vec![
                Column::primary_key("id", DataType::Int),
                Column::unique("email", DataType::Varchar(255)),
                Column::new("age", DataType::Int),
            ]),
            tuples: vec![
                vec![
                    Value::Number(1),
                    Value::String("john@email.com".into()),
                    Value::Number(18)
                ],
                vec![
                    Value::Number(2),
                    Value::String("jane@email.com".into()),
                    Value::Number(22)
                ],
                vec![
                    Value::Number(3),
                    Value::String("some_dude@email.com".into()),
                    Value::Number(24)
                ]
            ]
        });

        assert_index_contains(&mut db, "users_email_uq_index", &[
            vec![Value::String("jane@email.com".into()), Value::Number(2)],
            vec![Value::String("john@email.com".into()), Value::Number(1)],
            vec![
                Value::String("some_dude@email.com".into()),
                Value::Number(3),
            ],
        ])
    }

    // Top level LogicalOrScan plan.
    #[test]
    fn delete_where_auto_index_multiple_ranges() -> Result<(), DbError> {
        let mut db = init_database()?;

        db.exec("CREATE TABLE users (id INT PRIMARY KEY, email VARCHAR(255) UNIQUE, age INT);")?;
        db.exec("INSERT INTO users(id, email, age) VALUES (1, 'john@email.com', 18);")?;
        db.exec("INSERT INTO users(id, email, age) VALUES (2, 'jane@email.com', 22);")?;
        db.exec("INSERT INTO users(id, email, age) VALUES (3, 'some_dude@email.com', 24);")?;
        db.exec("INSERT INTO users(id, email, age) VALUES (4, 'another_dude@email.com', 24);")?;
        db.exec("INSERT INTO users(id, email, age) VALUES (5, 'dude_five@email.com', 24);")?;
        db.exec("INSERT INTO users(id, email, age) VALUES (6, 'dude_six@email.com', 24);")?;

        db.exec("DELETE FROM users WHERE id >= 2 AND id <= 3 OR id > 4;")?;

        let query = db.exec("SELECT * FROM users;")?;

        assert_eq!(query, QuerySet {
            schema: Schema::new(vec![
                Column::primary_key("id", DataType::Int),
                Column::unique("email", DataType::Varchar(255)),
                Column::new("age", DataType::Int),
            ]),
            tuples: vec![
                vec![
                    Value::Number(1),
                    Value::String("john@email.com".into()),
                    Value::Number(18)
                ],
                vec![
                    Value::Number(4),
                    Value::String("another_dude@email.com".into()),
                    Value::Number(24)
                ]
            ]
        });

        assert_index_contains(&mut db, "users_email_uq_index", &[
            vec![
                Value::String("another_dude@email.com".into()),
                Value::Number(4),
            ],
            vec![Value::String("john@email.com".into()), Value::Number(1)],
        ])
    }

    #[test]
    fn delete_from_multiple_indexes() -> Result<(), DbError> {
        let mut db = init_database()?;

        db.exec(
            "CREATE TABLE users (email VARCHAR(255) UNIQUE, name VARCHAR(64) UNIQUE, id INT PRIMARY KEY);",
        )?;

        db.exec("INSERT INTO users(id, name, email) VALUES (100, 'John Doe', 'john@email.com');")?;
        db.exec("INSERT INTO users(id, name, email) VALUES (200, 'Jane Doe', 'jane@email.com');")?;
        db.exec("INSERT INTO users(id, name, email) VALUES (300, 'Some Dude', 'some@dude.com');")?;

        db.exec("DELETE FROM users WHERE name >= 'John Doe';")?;

        assert_index_contains(&mut db, "users_email_uq_index", &[vec![
            Value::String("jane@email.com".into()),
            Value::Number(2),
        ]])?;

        assert_index_contains(&mut db, "users_pk_index", &[vec![
            Value::Number(200),
            Value::Number(2),
        ]])?;

        assert_index_contains(&mut db, "users_name_uq_index", &[vec![
            Value::String("Jane Doe".into()),
            Value::Number(2),
        ]])
    }

    #[test]
    fn delete_from_empty_table() -> Result<(), DbError> {
        let mut db = init_database()?;

        db.exec(
            "CREATE TABLE users (id INT PRIMARY KEY, email VARCHAR(255) UNIQUE, name VARCHAR(64));",
        )?;
        db.exec("CREATE UNIQUE INDEX name_idx ON users(name);")?;

        db.exec("DELETE FROM users WHERE id >= 500;")?;

        let query = db.exec("SELECT * FROM users;")?;

        assert!(query.is_empty());

        assert_index_contains(&mut db, "users_email_uq_index", &[])?;
        assert_index_contains(&mut db, "name_idx", &[])
    }

    // Test the collect plan.
    #[cfg(not(miri))]
    #[test]
    fn delete_many() -> Result<(), DbError> {
        let mut db = init_database_with(DbConf {
            page_size: 96,
            cache_size: 512,
        })?;

        db.exec("CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(255), age INT);")?;

        for i in 1..=500 {
            let name = format!("User{i}");
            db.exec(&format!(
                "INSERT INTO users(id, name, age) VALUES ({i}, '{name}', 18);"
            ))?;
        }

        db.exec("DELETE FROM users WHERE id != 355;")?;

        let query = db.exec("SELECT * FROM users;")?;

        assert_eq!(query, QuerySet {
            schema: Schema::new(vec![
                Column::primary_key("id", DataType::Int),
                Column::new("name", DataType::Varchar(255)),
                Column::new("age", DataType::Int),
            ]),
            tuples: vec![vec![
                Value::Number(355),
                Value::String("User355".into()),
                Value::Number(18)
            ]]
        });

        Ok(())
    }

    #[test]
    fn update() -> Result<(), DbError> {
        let mut db = init_database()?;

        db.exec("CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(255), age INT);")?;
        db.exec("INSERT INTO users(id, name, age) VALUES (1, 'John Doe', 18);")?;
        db.exec("INSERT INTO users(id, name, age) VALUES (2, 'Jane Doe', 22);")?;
        db.exec("INSERT INTO users(id, name, age) VALUES (3, 'Some Dude', 24);")?;

        db.exec("UPDATE users SET age = 20;")?;

        let query = db.exec("SELECT * FROM users;")?;

        assert_eq!(query, QuerySet {
            schema: Schema::new(vec![
                Column::primary_key("id", DataType::Int),
                Column::new("name", DataType::Varchar(255)),
                Column::new("age", DataType::Int),
            ]),
            tuples: vec![
                vec![
                    Value::Number(1),
                    Value::String("John Doe".into()),
                    Value::Number(20)
                ],
                vec![
                    Value::Number(2),
                    Value::String("Jane Doe".into()),
                    Value::Number(20)
                ],
                vec![
                    Value::Number(3),
                    Value::String("Some Dude".into()),
                    Value::Number(20)
                ]
            ]
        });

        Ok(())
    }

    #[test]
    fn update_where() -> Result<(), DbError> {
        let mut db = init_database()?;

        db.exec("CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(255), age INT);")?;
        db.exec("INSERT INTO users(id, name, age) VALUES (1, 'John Doe', 18);")?;
        db.exec("INSERT INTO users(id, name, age) VALUES (2, 'Jane Doe', 22);")?;
        db.exec("INSERT INTO users(id, name, age) VALUES (3, 'Some Dude', 24);")?;

        db.exec("UPDATE users SET age = 20, name = 'Updated Name' WHERE age > 18;")?;

        let query = db.exec("SELECT * FROM users;")?;

        assert_eq!(query, QuerySet {
            schema: Schema::new(vec![
                Column::primary_key("id", DataType::Int),
                Column::new("name", DataType::Varchar(255)),
                Column::new("age", DataType::Int),
            ]),
            tuples: vec![
                vec![
                    Value::Number(1),
                    Value::String("John Doe".into()),
                    Value::Number(18)
                ],
                vec![
                    Value::Number(2),
                    Value::String("Updated Name".into()),
                    Value::Number(20)
                ],
                vec![
                    Value::Number(3),
                    Value::String("Updated Name".into()),
                    Value::Number(20)
                ]
            ]
        });

        Ok(())
    }

    #[test]
    fn update_no_match() -> Result<(), DbError> {
        let mut db = init_database()?;

        db.exec("CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(255), age INT);")?;
        db.exec("INSERT INTO users(id, name, age) VALUES (1, 'John Doe', 18);")?;
        db.exec("INSERT INTO users(id, name, age) VALUES (2, 'Jane Doe', 22);")?;
        db.exec("INSERT INTO users(id, name, age) VALUES (3, 'Some Dude', 24);")?;

        db.exec("UPDATE users SET age = 20, name = 'Updated Name' WHERE age > 50;")?;

        let query = db.exec("SELECT * FROM users;")?;

        assert_eq!(query, QuerySet {
            schema: Schema::new(vec![
                Column::primary_key("id", DataType::Int),
                Column::new("name", DataType::Varchar(255)),
                Column::new("age", DataType::Int),
            ]),
            tuples: vec![
                vec![
                    Value::Number(1),
                    Value::String("John Doe".into()),
                    Value::Number(18)
                ],
                vec![
                    Value::Number(2),
                    Value::String("Jane Doe".into()),
                    Value::Number(22)
                ],
                vec![
                    Value::Number(3),
                    Value::String("Some Dude".into()),
                    Value::Number(24)
                ]
            ]
        });

        Ok(())
    }

    // Test the collect plan.
    #[cfg(not(miri))]
    #[test]
    fn update_many() -> Result<(), DbError> {
        let mut db = init_database_with(DbConf {
            page_size: 96,
            cache_size: 512,
        })?;

        db.exec("CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(255), age INT);")?;

        let pk_range = 1..=500;

        for i in pk_range.clone() {
            let name = format!("User{i}");
            db.exec(&format!(
                "INSERT INTO users(id, name, age) VALUES ({i}, '{name}', 18);"
            ))?;
        }

        db.exec("UPDATE users SET name = 'Updated Name', age = 20;")?;

        let query = db.exec("SELECT * FROM users;")?;

        assert_eq!(query, QuerySet {
            schema: Schema::new(vec![
                Column::primary_key("id", DataType::Int),
                Column::new("name", DataType::Varchar(255)),
                Column::new("age", DataType::Int),
            ]),
            tuples: pk_range
                .map(|pk| vec![
                    Value::Number(pk),
                    Value::String("Updated Name".into()),
                    Value::Number(20)
                ])
                .collect()
        });

        Ok(())
    }

    #[test]
    fn update_primary_key() -> Result<(), DbError> {
        let mut db = init_database()?;

        db.exec("CREATE TABLE users (name VARCHAR(255), age INT, id INT PRIMARY KEY, email VARCHAR(255) UNIQUE);")?;
        db.exec(
            "INSERT INTO users(id, name, email, age) VALUES (1, 'John Doe', 'john@doe.com', 18);",
        )?;
        db.exec(
            "INSERT INTO users(id, name, email, age) VALUES (2, 'Jane Doe', 'jane@doe.com', 22);",
        )?;
        db.exec(
            "INSERT INTO users(id, name, email, age) VALUES (3, 'Some Dude', 'some@dude.com', 24);",
        )?;

        db.exec("UPDATE users SET id = 10 where id = 1;")?;

        let query = db.exec("SELECT id, name, email, age FROM users;")?;

        assert_eq!(query, QuerySet {
            schema: Schema::new(vec![
                Column::primary_key("id", DataType::Int),
                Column::new("name", DataType::Varchar(255)),
                Column::unique("email", DataType::Varchar(255)),
                Column::new("age", DataType::Int),
            ]),
            tuples: vec![
                vec![
                    Value::Number(10),
                    Value::String("John Doe".into()),
                    Value::String("john@doe.com".into()),
                    Value::Number(18)
                ],
                vec![
                    Value::Number(2),
                    Value::String("Jane Doe".into()),
                    Value::String("jane@doe.com".into()),
                    Value::Number(22)
                ],
                vec![
                    Value::Number(3),
                    Value::String("Some Dude".into()),
                    Value::String("some@dude.com".into()),
                    Value::Number(24)
                ]
            ]
        });

        assert_index_contains(&mut db, "users_pk_index", &[
            vec![Value::Number(2), Value::Number(2)],
            vec![Value::Number(3), Value::Number(3)],
            vec![Value::Number(10), Value::Number(1)],
        ])?;

        assert_index_contains(&mut db, "users_email_uq_index", &[
            vec![Value::String("jane@doe.com".into()), Value::Number(2)],
            vec![Value::String("john@doe.com".into()), Value::Number(1)],
            vec![Value::String("some@dude.com".into()), Value::Number(3)],
        ])
    }

    #[test]
    fn update_pk_index_on_insert() -> Result<(), DbError> {
        let mut db = init_database()?;

        db.exec("CREATE TABLE users (id VARCHAR(3) PRIMARY KEY, name VARCHAR(255), age INT);")?;
        db.exec("INSERT INTO users(id, name, age) VALUES ('abc', 'John Doe', 18);")?;
        db.exec("INSERT INTO users(id, name, age) VALUES ('def', 'Jane Doe', 22);")?;
        db.exec("INSERT INTO users(id, name, age) VALUES ('ghi', 'Some Dude', 24);")?;

        assert_index_contains(&mut db, "users_pk_index", &[
            vec![Value::String("abc".into()), Value::Number(1)],
            vec![Value::String("def".into()), Value::Number(2)],
            vec![Value::String("ghi".into()), Value::Number(3)],
        ])
    }

    #[test]
    fn update_multiple_indexes_on_insert() -> Result<(), DbError> {
        let mut db = init_database()?;

        db.exec(
            "CREATE TABLE users (id VARCHAR(3) PRIMARY KEY, email VARCHAR(255) UNIQUE, name VARCHAR(64));",
        )?;
        db.exec("CREATE UNIQUE INDEX name_idx ON users(name);")?;

        db.exec("INSERT INTO users(id, name, email) VALUES ('ab', 'John Doe', 'john@email.com');")?;
        db.exec("INSERT INTO users(id, name, email) VALUES ('cd', 'Jane Doe', 'jane@email.com');")?;
        db.exec("INSERT INTO users(id, name, email) VALUES ('ef', 'Some Dude', 'some@dude.com');")?;

        assert_index_contains(&mut db, "users_pk_index", &[
            vec![Value::String("ab".into()), Value::Number(1)],
            vec![Value::String("cd".into()), Value::Number(2)],
            vec![Value::String("ef".into()), Value::Number(3)],
        ])?;

        assert_index_contains(&mut db, "users_email_uq_index", &[
            vec![Value::String("jane@email.com".into()), Value::Number(2)],
            vec![Value::String("john@email.com".into()), Value::Number(1)],
            vec![Value::String("some@dude.com".into()), Value::Number(3)],
        ])?;

        assert_index_contains(&mut db, "name_idx", &[
            vec![Value::String("Jane Doe".into()), Value::Number(2)],
            vec![Value::String("John Doe".into()), Value::Number(1)],
            vec![Value::String("Some Dude".into()), Value::Number(3)],
        ])
    }

    #[test]
    fn update_indexed_columns_on_sql_update_statement() -> Result<(), DbError> {
        let mut db = init_database()?;

        db.exec(
            "CREATE TABLE users (id INT PRIMARY KEY, email VARCHAR(255) UNIQUE, name VARCHAR(64));",
        )?;
        db.exec("CREATE UNIQUE INDEX name_idx ON users(name);")?;

        db.exec("INSERT INTO users(id, name, email) VALUES (100, 'John Doe', 'john@email.com');")?;
        db.exec("INSERT INTO users(id, name, email) VALUES (200, 'Jane Doe', 'jane@email.com');")?;
        db.exec(
            "INSERT INTO users(id, name, email) VALUES (300, 'Some Dude', 'some_dude@email.com');",
        )?;

        db.exec("UPDATE users SET id = 3, email = 'some@dude.com' WHERE id = 300;")?;
        db.exec("UPDATE users SET email = 'updated@email.com' WHERE id = 200;")?;

        assert_index_contains(&mut db, "users_email_uq_index", &[
            vec![Value::String("john@email.com".into()), Value::Number(100)],
            vec![Value::String("some@dude.com".into()), Value::Number(3)],
            vec![
                Value::String("updated@email.com".into()),
                Value::Number(200),
            ],
        ])?;

        assert_index_contains(&mut db, "name_idx", &[
            vec![Value::String("Jane Doe".into()), Value::Number(200)],
            vec![Value::String("John Doe".into()), Value::Number(100)],
            vec![Value::String("Some Dude".into()), Value::Number(3)],
        ])
    }

    #[test]
    fn build_up_index_on_create_index_statement() -> Result<(), DbError> {
        let mut db = init_database()?;

        db.exec("CREATE TABLE users (id INT PRIMARY KEY, email VARCHAR(255));")?;

        db.exec("INSERT INTO users(id, email) VALUES (100, 'john@email.com');")?;
        db.exec("INSERT INTO users(id, email) VALUES (200, 'jane@email.com');")?;
        db.exec("INSERT INTO users(id, email) VALUES (300, 'some@dude.com');")?;

        db.exec("CREATE UNIQUE INDEX email_uq ON users(email);")?;

        assert_index_contains(&mut db, "email_uq", &[
            vec![Value::String("jane@email.com".into()), Value::Number(200)],
            vec![Value::String("john@email.com".into()), Value::Number(100)],
            vec![Value::String("some@dude.com".into()), Value::Number(300)],
        ])
    }

    #[test]
    fn select_invalid_column() -> Result<(), DbError> {
        let mut db = init_database()?;

        db.exec("CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(255), age INT);")?;
        db.exec("INSERT INTO users(id, name, age) VALUES (1, 'John Doe', 18);")?;

        assert_eq!(
            db.exec("SELECT incorrect_col, id, name FROM users;"),
            Err(DbError::Sql(SqlError::InvalidColumn(
                "incorrect_col".into()
            )))
        );

        Ok(())
    }

    #[test]
    fn insert_invalid_column() -> Result<(), DbError> {
        let mut db = init_database()?;

        db.exec("CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(255));")?;

        assert_eq!(
            db.exec("INSERT INTO users(id, name, incorrect_col) VALUES (1, 'John Doe', 50);"),
            Err(DbError::Sql(SqlError::InvalidColumn(
                "incorrect_col".into()
            )))
        );

        Ok(())
    }

    #[test]
    fn insert_missing_columns() -> Result<(), DbError> {
        let mut db = init_database()?;

        db.exec("CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(255), age INT);")?;

        assert_eq!(
            db.exec("INSERT INTO users(id, name) VALUES (1, 'John Doe');"),
            Err(DbError::Sql(SqlError::AnalyzerError(
                AnalyzerError::MissingColumns
            )))
        );

        Ok(())
    }

    #[test]
    fn insert_column_count_mismatch() -> Result<(), DbError> {
        let mut db = init_database()?;

        db.exec("CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(255), age INT);")?;

        assert_eq!(
            db.exec("INSERT INTO users(id, name, age) VALUES (1, 'John Doe');"),
            Err(DbError::Sql(SqlError::AnalyzerError(
                AnalyzerError::ColumnValueCountMismatch
            )))
        );

        Ok(())
    }

    #[test]
    fn insert_type_error() -> Result<(), DbError> {
        let mut db = init_database()?;

        db.exec("CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(255));")?;

        assert_eq!(
            db.exec("INSERT INTO users(id, name) VALUES ('String', 10);"),
            Err(DbError::Sql(SqlError::TypeError(TypeError::ExpectedType {
                expected: VmDataType::Number,
                found: Expression::Value(Value::String("String".into()))
            })))
        );

        Ok(())
    }

    #[test]
    fn insert_duplicated_keys_auto_index() -> Result<(), DbError> {
        let mut db = init_database()?;

        db.exec("CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(255), age INT);")?;
        db.exec("INSERT INTO users(id, name, age) VALUES (1, 'John Doe', 18);")?;
        db.exec("INSERT INTO users(id, name, age) VALUES (2, 'Jane Doe', 22);")?;

        let dup = db.exec("INSERT INTO users(id, name, age) VALUES (2, 'Dup Key', 24);");
        let query = db.exec("SELECT * FROM users;")?;

        assert_eq!(
            dup,
            Err(DbError::Sql(SqlError::DuplicatedKey(Value::Number(2))))
        );

        assert_eq!(query, QuerySet {
            schema: Schema::new(vec![
                Column::primary_key("id", DataType::Int),
                Column::new("name", DataType::Varchar(255)),
                Column::new("age", DataType::Int),
            ]),
            tuples: vec![
                vec![
                    Value::Number(1),
                    Value::String("John Doe".into()),
                    Value::Number(18)
                ],
                vec![
                    Value::Number(2),
                    Value::String("Jane Doe".into()),
                    Value::Number(22)
                ],
            ]
        });

        Ok(())
    }

    #[test]
    fn create_unique_index_with_duplicated_keys() -> Result<(), DbError> {
        let mut db = init_database()?;

        let create_table = "CREATE TABLE users (id INT PRIMARY KEY, email VARCHAR(255));";

        db.exec(create_table)?;
        db.exec("INSERT INTO users(id, email) VALUES (1, 'john@doe.com');")?;
        db.exec("INSERT INTO users(id, email) VALUES (2, 'dup@email.com');")?;
        db.exec("INSERT INTO users(id, email) VALUES (3, 'dup@email.com');")?;

        let dup = db.exec("CREATE UNIQUE INDEX email_uq ON users(email);");
        let query = db.exec("SELECT * FROM mkdb_meta;")?;

        assert_eq!(
            dup,
            Err(DbError::Sql(SqlError::DuplicatedKey(Value::String(
                "dup@email.com".into()
            ))))
        );

        assert_eq!(query, QuerySet {
            schema: mkdb_meta_schema(),
            tuples: vec![vec![
                Value::String("table".into()),
                Value::String("users".into()),
                Value::Number(1),
                Value::String("users".into()),
                Value::String(Parser::new(create_table).parse_statement()?.to_string())
            ]]
        });

        Ok(())
    }
}
