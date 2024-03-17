use std::{
    cell::RefCell,
    cmp::Ordering,
    collections::HashMap,
    fmt::Display,
    fs::File,
    io::{self, Read, Seek, Write},
    path::Path,
    rc::Rc,
    usize,
};

use crate::{
    os::{Disk, HardwareBlockSize},
    paging::{
        self,
        pager::{PageNumber, Pager},
    },
    query,
    sql::{
        self,
        parser::{Parser, ParserError},
        statement::{
            BinaryOperator, Column, Constraint, Create, DataType, Expression, Statement,
            UnaryOperator, Value,
        },
    },
    storage::{tuple, BTree, BytesCmp, FixedSizeMemCmp},
    vm,
    vm::plan::Plan,
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

pub(crate) struct Database<I> {
    pub pager: Rc<RefCell<Pager<I>>>,
    row_ids: HashMap<String, u64>,
}

#[derive(Debug, PartialEq)]
pub(crate) struct Projection {
    pub schema: Schema,
    pub results: Vec<Vec<Value>>,
}

/// Generic data types used at runtime by [`crate::vm`] without SQL details
/// such as `UNSIGNED` or `VARCHAR(max)`.
///
/// Basically the variants of [`Value`] but without inner data.
#[derive(Debug, PartialEq, Clone, Copy)]
pub(crate) enum VmDataType {
    Bool,
    String,
    Number,
}

impl Display for VmDataType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Bool => "boolean",
            Self::Number => "number",
            Self::String => "string",
        })
    }
}

impl From<DataType> for VmDataType {
    fn from(data_type: DataType) -> Self {
        match data_type {
            DataType::Varchar(_) => VmDataType::String,
            DataType::Bool => VmDataType::Bool,
            _ => VmDataType::Number,
        }
    }
}

#[derive(Debug, PartialEq)]
pub(crate) enum TypeError {
    CannotApplyUnary {
        operator: UnaryOperator,
        value: Value,
    },
    CannotApplyBinary {
        left: Expression,
        operator: BinaryOperator,
        right: Expression,
    },
    ExpectedType {
        expected: VmDataType,
        found: Expression,
    },
}

impl From<TypeError> for SqlError {
    fn from(type_error: TypeError) -> Self {
        SqlError::TypeError(type_error)
    }
}

#[derive(Debug, PartialEq)]
pub(crate) enum SqlError {
    InvalidTable(String),
    InvalidColumn(String),
    ColumnValueCountMismatch,
    MissingColumns,
    MultiplePrimaryKeys,
    TypeError(TypeError),
    DivisionByZero(i128, i128),
    Other(String),
}

impl Display for SqlError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidTable(name) => write!(f, "invalid table '{name}'"),
            Self::InvalidColumn(name) => write!(f, "invalid column '{name}'"),
            Self::ColumnValueCountMismatch => f.write_str("number of columns doesn't match values"),
            Self::MultiplePrimaryKeys => f.write_str("only one primary key per table is allowed"),
            Self::MissingColumns => {
                f.write_str("default values are not supported, all columns must be specified")
            }
            Self::TypeError(type_error) => match type_error {
                TypeError::CannotApplyUnary { operator, value } => {
                    write!(f, "cannot apply unary operator '{operator}' to {value}")
                }

                TypeError::CannotApplyBinary {
                    left,
                    operator,
                    right,
                } => write!(
                    f,
                    "cannot binary operator '{operator}' to {left} and {right}"
                ),

                TypeError::ExpectedType { expected, found } => {
                    write!(
                        f,
                        "expected type {expected} but expression resolved to {found}"
                    )
                }
            },
            Self::DivisionByZero(left, right) => write!(f, "division by zero: {left} / {right}"),
            Self::Other(message) => f.write_str(message),
        }
    }
}

#[derive(Debug)]
pub(crate) enum DbError {
    Io(io::Error),
    Parser(ParserError),
    Sql(SqlError),
}

impl Display for DbError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "{e}"),
            Self::Parser(e) => write!(f, "{e}"),
            Self::Sql(e) => write!(f, "{e}"),
        }
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

impl From<SqlError> for DbError {
    fn from(e: SqlError) -> Self {
        Self::Sql(e)
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

        let col = Column {
            name: String::from("row_id"),
            data_type: DataType::UnsignedBigInt,
            constraints: vec![],
        };

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

pub(crate) type QueryResult = Result<Projection, DbError>;

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

impl<I: Seek + Read + Write> TryFrom<Plan<I>> for Projection {
    type Error = DbError;

    fn try_from(plan: Plan<I>) -> Result<Self, Self::Error> {
        let schema = plan.schema().unwrap_or(Schema::empty());
        let results = plan.collect::<Result<Vec<_>, DbError>>()?;

        Ok(Self { schema, results })
    }
}

/// Schema of the table used to keep track of the database information.
pub(crate) fn mkdb_meta_schema() -> Schema {
    Schema::from(vec![
        // Either "index" or "table"
        Column {
            name: String::from("type"),
            data_type: DataType::Varchar(255),
            constraints: vec![],
        },
        // Index or table name
        Column {
            name: String::from("name"),
            data_type: DataType::Varchar(255),
            constraints: vec![Constraint::Unique],
        },
        // Root page
        Column {
            name: String::from("root"),
            data_type: DataType::Int,
            constraints: vec![Constraint::Unique],
        },
        // Table name
        Column {
            name: String::from("table_name"),
            data_type: DataType::Varchar(255),
            constraints: vec![Constraint::Unique],
        },
        // SQL used to create the index or table.
        // TODO: Implement and use some TEXT data type with higher length limits.
        Column {
            name: String::from("sql"),
            data_type: DataType::Varchar(1000),
            constraints: vec![],
        },
    ])
}

impl<I> Database<I> {
    pub fn new(pager: Rc<RefCell<Pager<I>>>) -> Self {
        Self {
            pager,
            row_ids: HashMap::new(),
        }
    }
}

impl Database<File> {
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

        let mut pager = Pager::new(file, DEFAULT_PAGE_SIZE, block_size);

        pager.init()?;

        Ok(Database::new(Rc::new(RefCell::new(pager))))
    }
}

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

impl<I: Seek + Read + Write + paging::io::Sync> Database<I> {
    pub fn table_metadata(&mut self, table: &str) -> Result<(Schema, PageNumber), DbError> {
        if table == MKDB_META {
            let mut schema = mkdb_meta_schema();
            schema.prepend_row_id();
            return Ok((schema, MKDB_META_ROOT));
        }

        let query = self.exec(&format!(
            "SELECT root, sql FROM {MKDB_META} where table_name = '{table}' AND type = 'table';"
        ))?;

        if query.is_empty() {
            return Err(DbError::Sql(SqlError::InvalidTable(table.into())));
        }

        // TODO: Find some way to avoid parsing SQL every time. Probably a
        // hash map of table name -> schema, we wouldn't even need to update it
        // as we don't support ALTER table statements.
        let mut schema = match query.get(0, "sql") {
            Some(Value::String(sql)) => match Parser::new(sql).parse_statement()? {
                Statement::Create(Create::Table { columns, .. }) => Schema::from(columns),
                _ => unreachable!(),
            },
            _ => unreachable!(),
        };

        let root = match query.get(0, "root") {
            Some(Value::Number(root)) => *root as PageNumber,
            _ => unreachable!(),
        };

        schema.prepend_row_id();

        Ok((schema, root))
    }

    pub fn indexes_of(&mut self, table: &str) -> Result<Vec<(String, PageNumber)>, DbError> {
        let query = self.exec(&format!(
            "SELECT root, sql FROM {MKDB_META} where table_name = '{table}' AND type = 'index';"
        ))?;

        let mut indexes = Vec::new();

        for i in 0..query.results.len() {
            let root = match query.get(i, "root") {
                Some(Value::Number(root)) => *root as PageNumber,
                _ => unreachable!(),
            };

            let column = match query.get(i, "sql") {
                Some(Value::String(sql)) => match Parser::new(sql).parse_statement()? {
                    Statement::Create(Create::Index { column, .. }) => column,
                    _ => unreachable!(),
                },
                _ => unreachable!(),
            };

            indexes.push((column, root));
        }

        Ok(indexes)
    }

    pub fn root_of_index(&mut self, index: &str) -> Result<PageNumber, DbError> {
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

    pub(crate) fn next_row_id(&mut self, table: &str) -> Result<RowId, DbError> {
        if let Some(row_id) = self.row_ids.get_mut(table) {
            *row_id += 1;
            return Ok(*row_id);
        }

        let (_, root) = self.table_metadata(table)?;

        let mut pager = self.pager.borrow_mut();
        let mut btree = BTree::new(&mut pager, root, FixedSizeMemCmp::for_type::<RowId>());

        let row_id = if let Some(max) = btree.max()? {
            tuple::deserialize_row_id(max.as_ref()) + 1
        } else {
            1
        };

        self.row_ids.insert(String::from(table), row_id);
        Ok(row_id)
    }

    pub fn exec(&mut self, input: &str) -> QueryResult {
        let statement = sql::pipeline(input, self)?;

        // TODO: Rollback if it fails.
        let projection = if query::planner::needs_plan(&statement) {
            let plan = query::planner::generate_plan(statement, self)?;
            vm::plan::exec(plan)?
        } else {
            vm::statement::exec(statement, self)?;
            Projection::empty()
        };

        // TODO: Transactions.
        let mut pager = self.pager.borrow_mut();
        pager.write_dirty_pages()?;
        pager.flush()?;
        pager.sync()?;

        Ok(projection)
    }
}

#[cfg(test)]
mod tests {
    use std::{
        cell::RefCell,
        collections::HashMap,
        io::{self, Read, Seek, Write},
        rc::Rc,
    };

    use super::{Database, DbError, DEFAULT_PAGE_SIZE};
    use crate::{
        ascii_table,
        db::{mkdb_meta_schema, Projection, Schema, SqlError, TypeError, VmDataType},
        paging::{
            self,
            cache::{Cache, DEFAULT_MAX_CACHE_SIZE},
            io::MemBuf,
            pager::Pager,
        },
        sql::{
            parser::Parser,
            statement::{Column, Constraint, DataType, Expression, Value},
        },
        storage::{reassemble_payload, tuple, Cursor},
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
        let mut pager = Pager::with_cache(
            io::Cursor::new(Vec::<u8>::new()),
            conf.page_size,
            conf.page_size,
            cache,
        );
        pager.init()?;

        Ok(Database::new(Rc::new(RefCell::new(pager))))
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
                Column {
                    name: "id".into(),
                    data_type: DataType::Int,
                    constraints: vec![Constraint::PrimaryKey],
                },
                Column {
                    name: "name".into(),
                    data_type: DataType::Varchar(255),
                    constraints: vec![]
                }
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
                Column {
                    name: "id".into(),
                    data_type: DataType::Int,
                    constraints: vec![Constraint::PrimaryKey],
                },
                Column {
                    name: "name".into(),
                    data_type: DataType::Varchar(255),
                    constraints: vec![]
                },
                Column {
                    name: "age".into(),
                    data_type: DataType::Int,
                    constraints: vec![]
                }
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
                Column {
                    name: "id".into(),
                    data_type: DataType::Int,
                    constraints: vec![Constraint::PrimaryKey],
                },
                Column {
                    name: "price".into(),
                    data_type: DataType::Int,
                    constraints: vec![]
                },
                Column {
                    name: "discount".into(),
                    data_type: DataType::Int,
                    constraints: vec![]
                }
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
                Column {
                    name: "id".into(),
                    data_type: DataType::Int,
                    constraints: vec![Constraint::PrimaryKey],
                },
                Column {
                    name: "name".into(),
                    data_type: DataType::Varchar(255),
                    constraints: vec![]
                },
                Column {
                    name: "age".into(),
                    data_type: DataType::Int,
                    constraints: vec![]
                }
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
                Column {
                    name: "id".into(),
                    data_type: DataType::Int,
                    constraints: vec![Constraint::PrimaryKey],
                },
                Column {
                    name: "name".into(),
                    data_type: DataType::Varchar(255),
                    constraints: vec![]
                },
                Column {
                    name: "age".into(),
                    data_type: DataType::Int,
                    constraints: vec![]
                }
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
                Column {
                    name: "age".into(),
                    data_type: DataType::Int,
                    constraints: vec![]
                },
                Column {
                    name: "name".into(),
                    data_type: DataType::Varchar(255),
                    constraints: vec![]
                },
                Column {
                    name: "id".into(),
                    data_type: DataType::Int,
                    constraints: vec![Constraint::PrimaryKey],
                },
                Column {
                    name: "is_admin".into(),
                    data_type: DataType::Bool,
                    constraints: vec![],
                },
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
                Column {
                    name: "id".into(),
                    data_type: DataType::Int,
                    constraints: vec![Constraint::PrimaryKey],
                },
                Column {
                    name: "price / 10".into(),
                    data_type: DataType::BigInt,
                    constraints: vec![]
                },
                Column {
                    name: "discount * 100".into(),
                    data_type: DataType::BigInt,
                    constraints: vec![]
                }
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

        Ok(())
    }

    #[test]
    fn delete_where() -> Result<(), DbError> {
        let mut db = init_database()?;

        db.exec("CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(255), age INT);")?;
        db.exec("INSERT INTO users(id, name, age) VALUES (1, 'John Doe', 18);")?;
        db.exec("INSERT INTO users(id, name, age) VALUES (2, 'Jane Doe', 22);")?;
        db.exec("INSERT INTO users(id, name, age) VALUES (3, 'Some Dude', 24);")?;

        db.exec("DELETE FROM users WHERE age > 18;")?;

        let query = db.exec("SELECT * FROM users;")?;

        assert_eq!(query, Projection {
            schema: Schema::from(vec![
                Column {
                    name: "id".into(),
                    data_type: DataType::Int,
                    constraints: vec![Constraint::PrimaryKey],
                },
                Column {
                    name: "name".into(),
                    data_type: DataType::Varchar(255),
                    constraints: vec![]
                },
                Column {
                    name: "age".into(),
                    data_type: DataType::Int,
                    constraints: vec![]
                }
            ]),
            results: vec![vec![
                Value::Number(1),
                Value::String("John Doe".into()),
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

        assert_eq!(query, Projection {
            schema: Schema::from(vec![
                Column {
                    name: "id".into(),
                    data_type: DataType::Int,
                    constraints: vec![Constraint::PrimaryKey],
                },
                Column {
                    name: "name".into(),
                    data_type: DataType::Varchar(255),
                    constraints: vec![]
                },
                Column {
                    name: "age".into(),
                    data_type: DataType::Int,
                    constraints: vec![]
                }
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
                Column {
                    name: "id".into(),
                    data_type: DataType::Int,
                    constraints: vec![Constraint::PrimaryKey],
                },
                Column {
                    name: "name".into(),
                    data_type: DataType::Varchar(255),
                    constraints: vec![]
                },
                Column {
                    name: "age".into(),
                    data_type: DataType::Int,
                    constraints: vec![]
                }
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

    fn assert_index_contains<I: Seek + Read + Write + paging::io::Sync>(
        db: &mut Database<I>,
        name: &str,
        schema: Schema,
        expected_entries: &[Vec<Value>],
    ) -> Result<(), DbError> {
        let root = db.root_of_index(name)?;

        let mut pager = db.pager.borrow_mut();
        let mut cursor = Cursor::new(root, 0);

        let mut entries = Vec::new();

        while let Some((page, slot)) = cursor.try_next(&mut pager)? {
            let entry = reassemble_payload(&mut pager, page, slot)?;
            entries.push(tuple::deserialize_values(entry.as_ref(), &schema));
        }

        assert_eq!(entries.len(), expected_entries.len());

        for (entry, expected) in entries.iter().zip(expected_entries.iter()) {
            assert_eq!(entry, expected);
        }

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
            Schema::from(vec![
                Column::new("id", DataType::Int),
                Column::new("row_id", DataType::UnsignedBigInt),
            ]),
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
            Schema::from(vec![
                Column::new("id", DataType::Int),
                Column::new("row_id", DataType::UnsignedBigInt),
            ]),
            &[
                vec![Value::Number(100), Value::Number(1)],
                vec![Value::Number(200), Value::Number(2)],
                vec![Value::Number(300), Value::Number(3)],
            ],
        )?;

        assert_index_contains(
            &mut db,
            "users_email_uq_index",
            Schema::from(vec![
                Column::new("email", DataType::Varchar(255)),
                Column::new("row_id", DataType::UnsignedBigInt),
            ]),
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
            Schema::from(vec![
                Column::new("name", DataType::Varchar(255)),
                Column::new("row_id", DataType::UnsignedBigInt),
            ]),
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
                Column {
                    name: "id".into(),
                    data_type: DataType::Int,
                    constraints: vec![Constraint::PrimaryKey],
                },
                Column {
                    name: "name".into(),
                    data_type: DataType::Varchar(255),
                    constraints: vec![],
                },
                Column {
                    name: "email".into(),
                    data_type: DataType::Varchar(255),
                    constraints: vec![Constraint::Unique],
                }
            ]),
            results: expected_table_entries.clone()
        });

        assert_index_contains(
            &mut db,
            "users_pk_index",
            Schema::from(vec![
                Column::new("id", DataType::Int),
                Column::new("row_id", DataType::UnsignedBigInt),
            ]),
            &expected_pk_index_entries,
        )?;

        expected_table_entries.sort_by(|a, b| {
            let (Value::String(email_a), Value::String(email_b)) = (&a[2], &b[2]) else {
                unreachable!()
            };

            email_a.cmp(&email_b)
        });

        assert_index_contains(
            &mut db,
            "users_email_uq_index",
            Schema::from(vec![
                Column::new("email", DataType::Varchar(255)),
                Column::new("row_id", DataType::UnsignedBigInt),
            ]),
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
            Err(DbError::Sql(SqlError::MissingColumns))
        );

        Ok(())
    }

    #[test]
    fn insert_column_count_mismatch() -> Result<(), DbError> {
        let mut db = init_database()?;

        db.exec("CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(255), age INT);")?;

        assert_eq!(
            db.exec("INSERT INTO users(id, name, age) VALUES (1, 'John Doe');"),
            Err(DbError::Sql(SqlError::ColumnValueCountMismatch))
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
}
