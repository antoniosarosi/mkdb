use std::{
    cell::RefCell,
    collections::HashMap,
    fmt::Display,
    fs::File,
    io::{self, Read, Seek, Write},
    path::{Path, PathBuf},
    rc::Rc,
    usize,
};

use crate::{
    os::{Disk, HardwareBlockSize},
    paging::{
        self,
        io::FileOps,
        pager::{PageNumber, Pager},
    },
    query,
    sql::{
        self,
        analyzer::AnalyzerError,
        parser::{Parser, ParserError},
        statement::{Column, Create, DataType, Statement, Value},
    },
    storage::{tuple, BTree, FixedSizeMemCmp},
    vm::{self, plan::Plan, TypeError, VmError},
};

/// Database file default page size.
pub(crate) const DEFAULT_PAGE_SIZE: usize = 4096;

/// Name of the meta-table used to keep track of other tables.
pub(crate) const MKDB_META: &str = "mkdb_meta";

/// Root page of the meta-table.
pub(crate) const MKDB_META_ROOT: PageNumber = 0;

/// Rows are uniquely identified by an 8 byte key stored in big endian at the
/// beginning of each tuple.
pub(crate) type RowId = u64;

/// Main entry point to everything.
///
/// Provides the high level [`Database::exec`] API that receives SQL text and
/// runs it.
pub(crate) struct Database<F> {
    /// The database owns the pager. TODO: [`Rc<Refcell>`] is a temporary
    /// solution until we make the pager multithreaded.
    pub pager: Rc<RefCell<Pager<F>>>,
    /// Database context. See [`DatabaseContext`].
    pub context: Context,
    /// Working directory (the directory of the file).
    pub work_dir: PathBuf,
    /// `true` if we are currently in a transaction.
    transaction_started: bool,
}

impl Database<File> {
    /// Initializes a [`Database`] instance from the given file.
    pub fn init(path: impl AsRef<Path>) -> io::Result<Self> {
        let file = File::options()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&path)?;

        let metadata = file.metadata()?;

        if !metadata.is_file() {
            return Err(io::Error::new(io::ErrorKind::Unsupported, "not a file"));
        }

        let block_size = Disk::from(&path).block_size()?;

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

        Ok(Database::new(Rc::new(RefCell::new(pager)), work_dir))
    }
}

#[derive(Debug, PartialEq)]
pub(crate) enum SqlError {
    InvalidTable(String),
    InvalidColumn(String),
    AnalyzerError(AnalyzerError),
    TypeError(TypeError),
    VmError(VmError),
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
            Self::AnalyzerError(analyzer_error) => write!(f, "{analyzer_error}"),
            Self::VmError(vm_error) => write!(f, "{vm_error}"),
            Self::TypeError(type_error) => write!(f, "{type_error}"),
            Self::Other(message) => f.write_str(message),
        }
    }
}

#[derive(Debug)]
pub(crate) enum DbError {
    Io(io::Error),
    Parser(ParserError),
    Sql(SqlError),
    Corrupted(String),
}

impl Display for DbError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "{e}"),
            Self::Parser(e) => write!(f, "{e}"),
            Self::Sql(e) => write!(f, "{e}"),
            Self::Corrupted(message) => f.write_str(message),
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

/// Table schema.
#[derive(Debug, PartialEq, Clone)]
pub(crate) struct Schema {
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
        debug_assert!(self.columns[0].name != "row_id");

        let col = Column::new("row_id", DataType::UnsignedBigInt);

        self.columns.insert(0, col);
        self.index.values_mut().for_each(|idx| *idx += 1);
        self.index.insert(String::from("row_id"), 0);
    }
}

impl From<Vec<Column>> for Schema {
    fn from(columns: Vec<Column>) -> Self {
        Self::new(columns)
    }
}

#[derive(Debug, PartialEq)]
pub(crate) struct Projection {
    pub schema: Schema,
    pub results: Vec<Vec<Value>>,
}

impl Projection {
    pub fn new(schema: Schema, results: Vec<Vec<Value>>) -> Self {
        Self { schema, results }
    }

    pub fn empty() -> Self {
        Self {
            schema: Schema::empty(),
            results: Vec::new(),
        }
    }

    pub fn get(&self, row: usize, column: &str) -> Option<&Value> {
        self.results.get(row)?.get(self.schema.index_of(column)?)
    }

    pub fn is_empty(&self) -> bool {
        self.results.is_empty()
    }
}

impl<F: Seek + Read + Write + FileOps> TryFrom<Plan<F>> for Projection {
    type Error = DbError;

    fn try_from(plan: Plan<F>) -> Result<Self, Self::Error> {
        let schema = plan.schema().unwrap_or(Schema::empty());
        let results = plan.collect::<Result<Vec<_>, DbError>>()?;

        Ok(Self { schema, results })
    }
}

/// Schema of the table used to keep track of the database information.
pub(crate) fn mkdb_meta_schema() -> Schema {
    Schema::from(vec![
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
        Column::new("sql", DataType::Varchar(255)),
    ])
}

/// Data that we need to know about an index at runtime.
#[derive(Debug, Clone)]
pub(crate) struct IndexMetadata {
    /// Root page of the index-
    pub root: PageNumber,
    /// Index name.
    pub name: String,
    /// Column on which the index was created.
    pub column: Column,
}

impl IndexMetadata {
    pub fn schema(&self) -> Schema {
        Schema::new(vec![
            self.column.clone(),
            Column::new("row_id", DataType::UnsignedBigInt),
        ])
    }
}

/// Data that we need to know about tables at runtime.
#[derive(Debug, Clone)]
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

impl TableMetadata {
    pub fn next_row_id(&mut self) -> RowId {
        let row_id = self.row_id;
        self.row_id += 1;

        row_id
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
    tables: HashMap<String, TableMetadata>,
    max_size: Option<usize>,
}

impl Context {
    pub fn new() -> Self {
        Self {
            tables: HashMap::new(),
            max_size: None,
        }
    }

    pub fn with_max_size(max_size: usize) -> Self {
        Self {
            tables: HashMap::with_capacity(max_size),
            max_size: Some(max_size),
        }
    }

    pub fn contains(&self, table: &str) -> bool {
        self.tables.contains_key(table)
    }

    pub fn insert(&mut self, table: String, metadata: TableMetadata) {
        if self.max_size.is_some_and(|size| self.tables.len() >= size) {
            let evict = self.tables.keys().next().unwrap().clone();
            self.tables.remove(&evict);
        }

        self.tables.insert(table, metadata);
    }

    pub fn invalidate(&mut self, table: &str) {
        self.tables.remove(table);
    }
}

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
                    let mut schema = Schema::from(columns.clone());
                    schema.prepend_row_id();

                    let mut metadata = TableMetadata {
                        root,
                        name: name.clone(),
                        row_id: 1,
                        schema,
                        indexes: vec![],
                    };
                    root += 1;

                    use crate::sql::statement::Constraint;

                    for column in columns {
                        for constraint in &column.constraints {
                            let index_name = match constraint {
                                Constraint::PrimaryKey => format!("{name}_pk_index"),
                                Constraint::Unique => format!("{name}_{}_uq_index", column.name),
                            };

                            metadata.indexes.push(IndexMetadata {
                                column: column.clone(),
                                name: index_name,
                                root,
                            });

                            root += 1;
                        }
                    }

                    context.insert(name.clone(), metadata);
                }

                Statement::Create(Create::Index {
                    name,
                    column,
                    unique,
                    ..
                }) if unique => {
                    let table = context.table_metadata(&name)?;
                    let col_idx = table.schema.index_of(&column).unwrap();

                    table.indexes.push(IndexMetadata {
                        column: table.schema.columns[col_idx].clone(),
                        name,
                        root,
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
    pub fn new(pager: Rc<RefCell<Pager<F>>>, work_dir: PathBuf) -> Self {
        Self {
            pager,
            work_dir,
            context: Context::with_max_size(DEFAULT_RELATION_CACHE_SIZE),
            transaction_started: false,
        }
    }
}

impl<F: Seek + Read + Write + paging::io::FileOps> DatabaseContext for Database<F> {
    fn table_metadata(&mut self, table: &str) -> Result<&mut TableMetadata, DbError> {
        if !self.context.contains(table) {
            let metadata = self.load_table_metadata(table)?;
            self.context.insert(table.into(), metadata);
        }

        self.context.table_metadata(table)
    }
}

impl<F: Seek + Read + Write + paging::io::FileOps> Database<F> {
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

        let query = self.exec(&format!(
            "SELECT root, sql FROM {MKDB_META} where table_name = '{table}';"
        ))?;

        if query.is_empty() {
            return Err(DbError::Sql(SqlError::InvalidTable(table.into())));
        }

        let mut metadata = TableMetadata {
            root: 1,
            name: String::from(table),
            row_id: 1,
            schema: Schema::empty(),
            indexes: Vec::new(),
        };

        let mut found_table_definition = false;

        for i in 0..query.results.len() {
            let root = match query.get(i, "root") {
                Some(Value::Number(root)) => *root as PageNumber,
                _ => unreachable!(),
            };

            match query.get(i, "sql") {
                Some(Value::String(sql)) => match Parser::new(sql).parse_statement()? {
                    Statement::Create(Create::Table { columns, .. }) => {
                        assert!(
                            !found_table_definition,
                            "multiple definitions of table '{table}'"
                        );

                        metadata.root = root;
                        metadata.schema = Schema::from(columns);
                        metadata.row_id = self.load_next_row_id(root)?;

                        metadata.schema.prepend_row_id();

                        found_table_definition = true;
                    }

                    Statement::Create(Create::Index { column, name, .. }) => {
                        // The table schema should be loaded by this time
                        // because it's impossible to define an index unless the
                        // table exists and the results are returned sorted by
                        // row_id.
                        let col_idx = metadata.schema.index_of(&column).ok_or(
                            SqlError::Other(format!(
                                "could not find index column {column} in the definition of table {table}"
                            )),
                        )?;

                        metadata.indexes.push(IndexMetadata {
                            column: metadata.schema.columns[col_idx].clone(),
                            name,
                            root,
                        });
                    }

                    _ => unreachable!(),
                },
                _ => unreachable!(),
            };
        }

        Ok(metadata)
    }

    /// Returns the root page of `index` if it exists.
    fn root_of_index(&mut self, index: &str) -> Result<PageNumber, DbError> {
        let query = self.exec(&format!(
            "SELECT root FROM {MKDB_META} where name = '{index}' AND type = 'index';"
        ))?;

        if query.is_empty() {
            return Err(DbError::Sql(SqlError::Other(format!(
                "index {index} does not exist"
            ))));
        }

        match query.get(0, "root") {
            Some(Value::Number(root)) => Ok(*root as PageNumber),
            _ => unreachable!(),
        }
    }

    /// Highest level API in the entire system.
    ///
    /// Receives a SQL string and executes it.
    pub fn exec(&mut self, input: &str) -> Result<Projection, DbError> {
        let statement = sql::pipeline(input, self)?;

        if statement == Statement::StartTransaction && self.transaction_started {
            return Err(DbError::Sql(SqlError::Other(String::from(
                "There is already a transaction in progress",
            ))));
        }

        let commit = statement == Statement::Commit
            || statement != Statement::StartTransaction && !self.transaction_started;

        let rollback = statement == Statement::Rollback;

        if !self.transaction_started {
            self.transaction_started = true;
        }

        let projection = match &statement {
            Statement::Create(_) => {
                vm::statement::exec(statement, self).map(|_| Projection::empty())
            }

            Statement::StartTransaction | Statement::Commit | Statement::Rollback => {
                Ok(Projection::empty())
            }

            _ => query::planner::generate_plan(statement, self).and_then(vm::plan::exec),
        };

        let mut pager = self.pager.borrow_mut();

        if commit {
            self.transaction_started = false;
            pager.commit()?;
        } else if rollback || projection.is_err() {
            self.transaction_started = false;

            if let Err(rollback_err) = pager.rollback() {
                if let Err(query_err) = projection {
                    panic!("database bruh moment: found error {query_err:?} while processing query so we had to rollback, but then there was an error while rolling back: {rollback_err:?}");
                } else {
                    return Err(rollback_err);
                }
            }
        }

        projection
    }
}

#[cfg(test)]
mod tests {
    use std::{
        cell::RefCell,
        collections::HashMap,
        io::{self, Read, Seek, Write},
        path::PathBuf,
        rc::Rc,
    };

    use super::{Database, DbError, DEFAULT_PAGE_SIZE};
    use crate::{
        db::{mkdb_meta_schema, Projection, Schema, SqlError, TypeError},
        paging::{
            self,
            cache::{Cache, DEFAULT_MAX_CACHE_SIZE},
            io::MemBuf,
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
        let cache = Cache::builder()
            .page_size(conf.page_size)
            .max_size(conf.cache_size)
            .build();

        let mut pager = Pager::<MemBuf>::builder()
            .page_size(conf.page_size)
            .cache(cache)
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

    #[test]
    fn create_table() -> Result<(), DbError> {
        let mut db = init_database()?;

        let sql = "CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(255));";
        db.exec(sql)?;

        let query = db.exec("SELECT * FROM mkdb_meta;")?;

        assert_eq!(
            query,
            Projection::new(mkdb_meta_schema(), vec![
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

    #[test]
    fn insert_data() -> Result<(), DbError> {
        let mut db = init_database()?;

        db.exec("CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(255));")?;
        db.exec("INSERT INTO users(id, name) VALUES (1, 'John Doe');")?;
        db.exec("INSERT INTO users(id, name) VALUES (2, 'Jane Doe');")?;

        let query = db.exec("SELECT * FROM users;")?;

        assert_eq!(query, Projection {
            schema: Schema::from(vec![
                Column::primary_key("id", DataType::Int),
                Column::new("name", DataType::Varchar(255)),
            ]),
            results: vec![
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

        assert_eq!(query, Projection {
            schema: Schema::from(vec![
                Column::primary_key("id", DataType::Int),
                Column::new("name", DataType::Varchar(255)),
                Column::new("age", DataType::Int),
            ]),
            results: vec![
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

        assert_eq!(query, Projection {
            schema: Schema::from(vec![
                Column::primary_key("id", DataType::Int),
                Column::new("price", DataType::Int),
                Column::new("discount", DataType::Int),
            ]),
            results: vec![
                vec![Value::Number(1), Value::Number(110), Value::Number(4),],
                vec![Value::Number(2), Value::Number(40), Value::Number(20),],
            ]
        });

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

        assert_eq!(query, Projection {
            schema: Schema::from(vec![
                Column::primary_key("id", DataType::Int),
                Column::new("name", DataType::Varchar(255)),
                Column::new("age", DataType::Int),
            ]),
            results: vec![
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
    fn select_order_by() -> Result<(), DbError> {
        let mut db = init_database()?;

        db.exec("CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(255), age INT);")?;
        db.exec("INSERT INTO users(id, name, age) VALUES (3, 'Some Dude', 24);")?;
        db.exec("INSERT INTO users(id, name, age) VALUES (2, 'John Doe', 22);")?;
        db.exec("INSERT INTO users(id, name, age) VALUES (1, 'John Doe', 18);")?;

        let query = db.exec("SELECT * FROM users ORDER BY name, age;")?;

        assert_eq!(query, Projection {
            schema: Schema::from(vec![
                Column::primary_key("id", DataType::Int),
                Column::new("name", DataType::Varchar(255)),
                Column::new("age", DataType::Int),
            ]),
            results: vec![
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

        assert_eq!(query, Projection {
            schema: Schema::from(vec![Column::new("name", DataType::Varchar(255)),]),
            results: expected,
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

        assert_eq!(query, Projection {
            schema: Schema::from(vec![
                Column::new("age", DataType::Int),
                Column::new("name", DataType::Varchar(255)),
                Column::primary_key("id", DataType::Int),
                Column::new("is_admin", DataType::Bool),
            ]),
            results: vec![
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

        assert_eq!(query, Projection {
            schema: Schema::from(vec![
                Column::primary_key("id", DataType::Int),
                Column::new("price / 10", DataType::BigInt),
                Column::new("discount * 100", DataType::BigInt),
            ]),
            results: vec![
                vec![Value::Number(1), Value::Number(10), Value::Number(500),],
                vec![Value::Number(2), Value::Number(25), Value::Number(1000),],
            ]
        });

        Ok(())
    }

    #[test]
    fn create_multiple_tables() -> Result<(), DbError> {
        let mut db = init_database()?;

        let t1 = "CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(255));";
        let i1 = "CREATE UNIQUE INDEX users_pk_index ON users(id);";

        let t2 = "CREATE TABLE tasks (id INT PRIMARY KEY, title VARCHAR(255), description VARCHAR(255));";
        let i2 = "CREATE UNIQUE INDEX tasks_pk_index ON tasks(id);";

        let t3 = "CREATE TABLE products (id INT PRIMARY KEY, price INT);";
        let i3 = "CREATE UNIQUE INDEX products_pk_index ON products(id);";

        db.exec(t1)?;
        db.exec(t2)?;
        db.exec(t3)?;

        let query = db.exec("SELECT * FROM mkdb_meta;")?;

        assert_eq!(query, Projection {
            schema: mkdb_meta_schema(),
            results: vec![
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
    fn create_many_tables() -> Result<(), DbError> {
        let mut db = init_database()?;

        let mut expected = Vec::new();

        let schema = "(id INT PRIMARY KEY, title VARCHAR(255), description VARCHAR(255))";

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

            assert_eq!(query.results[i], tuple);
        }

        Ok(())
    }

    #[test]
    fn delete_all() -> Result<(), DbError> {
        let mut db = init_database()?;

        db.exec("CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(255), age INT);")?;
        db.exec("INSERT INTO users(id, name, age) VALUES (1, 'John Doe', 18);")?;
        db.exec("INSERT INTO users(id, name, age) VALUES (2, 'Jane Doe', 22);")?;
        db.exec("INSERT INTO users(id, name, age) VALUES (3, 'Some Dude', 24);")?;

        db.exec("DELETE FROM users;")?;

        let query = db.exec("SELECT * FROM users;")?;

        assert!(query.results.is_empty());

        assert_index_contains(
            &mut db,
            "users_pk_index",
            Column::primary_key("id", DataType::Int),
            &[],
        )
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

        assert_eq!(query, Projection {
            schema: Schema::from(vec![
                Column::primary_key("id", DataType::Int),
                Column::unique("email", DataType::Varchar(255)),
                Column::new("age", DataType::Int),
            ]),
            results: vec![vec![
                Value::Number(1),
                Value::String("john@email.com".into()),
                Value::Number(18)
            ]]
        });

        assert_index_contains(
            &mut db,
            "users_pk_index",
            Column::primary_key("id", DataType::Int),
            &[vec![Value::Number(1), Value::Number(1)]],
        )?;

        assert_index_contains(
            &mut db,
            "users_email_uq_index",
            Column::unique("email", DataType::Varchar(255)),
            &[vec![
                Value::String("john@email.com".into()),
                Value::Number(1),
            ]],
        )
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

        assert_eq!(query, Projection {
            schema: Schema::from(vec![
                Column::primary_key("id", DataType::Int),
                Column::new("name", DataType::Varchar(255)),
                Column::new("age", DataType::Int),
            ]),
            results: vec![
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

        assert_eq!(query, Projection {
            schema: Schema::from(vec![
                Column::primary_key("id", DataType::Int),
                Column::new("name", DataType::Varchar(255)),
                Column::new("age", DataType::Int),
            ]),
            results: vec![
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

    fn assert_index_contains<F: Seek + Read + Write + paging::io::FileOps>(
        db: &mut Database<F>,
        name: &str,
        key: Column,
        expected_entries: &[Vec<Value>],
    ) -> Result<(), DbError> {
        let root = db.root_of_index(name)?;

        let mut pager = db.pager.borrow_mut();
        let mut cursor = Cursor::new(root, 0);

        let mut entries = Vec::new();

        while let Some((page, slot)) = cursor.try_next(&mut pager)? {
            let entry = reassemble_payload(&mut pager, page, slot)?;
            entries.push(tuple::deserialize(
                entry.as_ref(),
                &Schema::from(vec![
                    key.clone(),
                    Column::new("row_id", DataType::UnsignedBigInt),
                ]),
            ));
        }

        assert_eq!(entries, expected_entries);

        Ok(())
    }

    #[test]
    fn update_pk_index_on_insert() -> Result<(), DbError> {
        let mut db = init_database()?;

        db.exec("CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(255), age INT);")?;
        db.exec("INSERT INTO users(id, name, age) VALUES (100, 'John Doe', 18);")?;
        db.exec("INSERT INTO users(id, name, age) VALUES (200, 'Jane Doe', 22);")?;
        db.exec("INSERT INTO users(id, name, age) VALUES (300, 'Some Dude', 24);")?;

        assert_index_contains(
            &mut db,
            "users_pk_index",
            Column::new("id", DataType::Int),
            &[
                vec![Value::Number(100), Value::Number(1)],
                vec![Value::Number(200), Value::Number(2)],
                vec![Value::Number(300), Value::Number(3)],
            ],
        )
    }

    #[test]
    fn update_multiple_indexes_on_insert() -> Result<(), DbError> {
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

        assert_index_contains(
            &mut db,
            "users_pk_index",
            Column::new("id", DataType::Int),
            &[
                vec![Value::Number(100), Value::Number(1)],
                vec![Value::Number(200), Value::Number(2)],
                vec![Value::Number(300), Value::Number(3)],
            ],
        )?;

        assert_index_contains(
            &mut db,
            "users_email_uq_index",
            Column::new("email", DataType::Varchar(255)),
            &[
                vec![Value::String("jane@email.com".into()), Value::Number(2)],
                vec![Value::String("john@email.com".into()), Value::Number(1)],
                vec![
                    Value::String("some_dude@email.com".into()),
                    Value::Number(3),
                ],
            ],
        )?;

        assert_index_contains(
            &mut db,
            "name_idx",
            Column::new("name", DataType::Varchar(255)),
            &[
                vec![Value::String("Jane Doe".into()), Value::Number(2)],
                vec![Value::String("John Doe".into()), Value::Number(1)],
                vec![Value::String("Some Dude".into()), Value::Number(3)],
            ],
        )
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
        let mut expected_pk_index_entries = Vec::new();
        let mut expected_email_uq_index_entries = Vec::new();

        for i in 1..200 {
            let name = format!("User {i}");
            let email = format!("user{i}@test.com");

            expected_table_entries.push(vec![
                Value::Number(i),
                Value::String(name.clone()),
                Value::String(email.clone()),
            ]);

            expected_pk_index_entries.push(vec![Value::Number(i), Value::Number(i)]);

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

        let query = db.exec("SELECT * FROM users ORDER BY id;")?;

        assert_eq!(&query, &Projection {
            schema: Schema::from(vec![
                Column::primary_key("id", DataType::Int),
                Column::new("name", DataType::Varchar(255)),
                Column::unique("email", DataType::Varchar(255)),
            ]),
            results: expected_table_entries.clone()
        });

        assert_index_contains(
            &mut db,
            "users_pk_index",
            Column::new("id", DataType::Int),
            &expected_pk_index_entries,
        )?;

        expected_table_entries.sort_by(|a, b| {
            let (Value::String(email_a), Value::String(email_b)) = (&a[2], &b[2]) else {
                unreachable!()
            };

            email_a.cmp(email_b)
        });

        assert_index_contains(
            &mut db,
            "users_email_uq_index",
            Column::new("email", DataType::Varchar(255)),
            &expected_email_uq_index_entries,
        )?;

        Ok(())
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
    fn select_where_indexed_exact() -> Result<(), DbError> {
        let mut db = init_database()?;

        db.exec("CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(255), age INT);")?;
        db.exec("INSERT INTO users(id, name, age) VALUES (1, 'John Doe', 18);")?;
        db.exec("INSERT INTO users(id, name, age) VALUES (2, 'Jane Doe', 22);")?;
        db.exec("INSERT INTO users(id, name, age) VALUES (3, 'Some Dude', 24);")?;

        let query = db.exec("SELECT * FROM users WHERE id = 2;")?;

        assert_eq!(query, Projection {
            schema: Schema::from(vec![
                Column::primary_key("id", DataType::Int),
                Column::new("name", DataType::Varchar(255)),
                Column::new("age", DataType::Int),
            ]),
            results: vec![vec![
                Value::Number(2),
                Value::String("Jane Doe".into()),
                Value::Number(22)
            ],]
        });

        Ok(())
    }

    #[test]
    fn select_where_indexed_less_than() -> Result<(), DbError> {
        let mut db = init_database()?;

        db.exec("CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(255), age INT);")?;
        db.exec("INSERT INTO users(id, name, age) VALUES (1, 'John Doe', 18);")?;
        db.exec("INSERT INTO users(id, name, age) VALUES (2, 'Jane Doe', 22);")?;
        db.exec("INSERT INTO users(id, name, age) VALUES (3, 'Some Dude', 24);")?;
        db.exec("INSERT INTO users(id, name, age) VALUES (4, 'Another Dude', 30);")?;

        let query = db.exec("SELECT * FROM users WHERE id < 3;")?;

        assert_eq!(query, Projection {
            schema: Schema::from(vec![
                Column::primary_key("id", DataType::Int),
                Column::new("name", DataType::Varchar(255)),
                Column::new("age", DataType::Int),
            ]),
            results: vec![
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
    fn select_where_indexed_greather_than() -> Result<(), DbError> {
        let mut db = init_database()?;

        db.exec("CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(255), age INT);")?;
        db.exec("INSERT INTO users(id, name, age) VALUES (1, 'John Doe', 18);")?;
        db.exec("INSERT INTO users(id, name, age) VALUES (2, 'Jane Doe', 22);")?;
        db.exec("INSERT INTO users(id, name, age) VALUES (3, 'Some Dude', 24);")?;
        db.exec("INSERT INTO users(id, name, age) VALUES (4, 'Another Dude', 30);")?;

        let query = db.exec("SELECT * FROM users WHERE id > 2;")?;

        assert_eq!(query, Projection {
            schema: Schema::from(vec![
                Column::primary_key("id", DataType::Int),
                Column::new("name", DataType::Varchar(255)),
                Column::new("age", DataType::Int),
            ]),
            results: vec![
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
    fn select_where_indexed_less_than_or_equal() -> Result<(), DbError> {
        let mut db = init_database()?;

        db.exec("CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(255), age INT);")?;
        db.exec("INSERT INTO users(id, name, age) VALUES (1, 'John Doe', 18);")?;
        db.exec("INSERT INTO users(id, name, age) VALUES (2, 'Jane Doe', 22);")?;
        db.exec("INSERT INTO users(id, name, age) VALUES (3, 'Some Dude', 24);")?;
        db.exec("INSERT INTO users(id, name, age) VALUES (4, 'Another Dude', 30);")?;

        let query = db.exec("SELECT * FROM users WHERE id <= 3;")?;

        assert_eq!(query, Projection {
            schema: Schema::from(vec![
                Column::primary_key("id", DataType::Int),
                Column::new("name", DataType::Varchar(255)),
                Column::new("age", DataType::Int),
            ]),
            results: vec![
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
    fn select_where_indexed_greather_than_or_equal() -> Result<(), DbError> {
        let mut db = init_database()?;

        db.exec("CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(255), age INT);")?;
        db.exec("INSERT INTO users(id, name, age) VALUES (1, 'John Doe', 18);")?;
        db.exec("INSERT INTO users(id, name, age) VALUES (2, 'Jane Doe', 22);")?;
        db.exec("INSERT INTO users(id, name, age) VALUES (3, 'Some Dude', 24);")?;
        db.exec("INSERT INTO users(id, name, age) VALUES (4, 'Another Dude', 30);")?;

        let query = db.exec("SELECT * FROM users WHERE id >= 3;")?;

        assert_eq!(query, Projection {
            schema: Schema::from(vec![
                Column::primary_key("id", DataType::Int),
                Column::new("name", DataType::Varchar(255)),
                Column::new("age", DataType::Int),
            ]),
            results: vec![
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
}
