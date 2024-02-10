use std::{
    cmp::Ordering,
    collections::HashMap,
    fmt::Display,
    fs::File,
    io::{self, Read, Seek, Write},
    os::unix::fs::MetadataExt,
    path::Path,
};

use crate::{
    os::{Disk, HardwareBlockSize},
    paging::{
        cache::Cache,
        pager::{PageNumber, Pager},
    },
    sql::{
        BinaryOperator, Column, Constraint, Create, DataType, Expression, Parser, ParserError,
        Statement, Value,
    },
    storage::{page::Page, BTree, BytesCmp},
};

/// Name of the meta-table used to keep track of other tables.
pub(crate) const MKDB_META: &str = "mkdb_meta";

/// Root page of the meta-table. Page 0 holds the DB header, page 1 holds the
/// beginning of the meta-table.
pub(crate) const MKDB_META_ROOT: PageNumber = 1;

/// Magic number at the beginning of the database file.
pub(crate) const MAGIC: u32 = 0xB74EE;

pub(crate) struct RowIdComparator;

impl BytesCmp for RowIdComparator {
    fn bytes_cmp(&self, a: &[u8], b: &[u8]) -> std::cmp::Ordering {
        // TODO: Store Row ID in Big Endian format to avoid parsing.
        u64::from_le_bytes(a[..8].try_into().unwrap())
            .cmp(&u64::from_le_bytes(b[..8].try_into().unwrap()))
    }
}

/// Database file header. Located at the beginning of the DB file.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub(crate) struct Header {
    pub magic: u32,
    pub page_size: u32,
    pub total_pages: u32,
    pub first_free_page: PageNumber,
}

pub(crate) struct Database<I> {
    cache: Cache<I>,
    row_ids: HashMap<String, u64>,
}

#[derive(Debug, PartialEq)]
pub(crate) struct QueryExecution {
    schema: Vec<Column>,
    results: Vec<Vec<Value>>,
}

#[derive(Debug)]
pub(crate) enum DbError {
    Io(io::Error),
    Parser(ParserError),
}

impl Display for DbError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "{e}"),
            Self::Parser(e) => write!(f, "{e}"),
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

type QueryResult = Result<QueryExecution, DbError>;

impl QueryExecution {
    fn get(&self, row: usize, column: &str) -> Option<Value> {
        self.schema
            .iter()
            .position(|c| c.name == column)
            .map(|position| self.results.get(row)?.get(position).map(ToOwned::to_owned))
            .flatten()
    }

    fn is_empty(&self) -> bool {
        self.results.is_empty()
    }

    pub fn as_ascii_table(&self) -> String {
        let mut width: Vec<_> = self.schema.iter().map(|column| column.name.len()).collect();
        // TODO: Remove repeated calls to "to_string()".
        for row in &self.results {
            for (i, col) in row.iter().enumerate() {
                if col.to_string().len() > width[i] {
                    width[i] = col.to_string().len();
                }
            }
        }

        width.iter_mut().for_each(|w| *w += 2);

        let mut border = String::from('+');
        border.push_str(
            &width
                .iter()
                .map(|w| "-".repeat(*w))
                .collect::<Vec<_>>()
                .join("+"),
        );
        border.push('+');

        let make_row = |row: Vec<String>| -> String {
            let mut string = String::from('|');

            let mut buf = Vec::new();
            for (i, col) in row.iter().enumerate() {
                let text = format!(" {col}");
                let trailing_spaces = " ".repeat(width[i] - text.len());
                buf.push(format!("{text}{trailing_spaces}"));
            }

            string.push_str(&buf.join("|"));
            string.push('|');

            string
        };

        let mut string = String::from(&border);
        string.push('\n');
        string.push_str(&make_row(
            self.schema.iter().map(|c| c.name.clone()).collect(),
        ));
        string.push('\n');
        string.push_str(&border);
        string.push('\n');
        for row in &self.results {
            string.push_str(&make_row(row.iter().map(ToString::to_string).collect()));
            string.push('\n');
        }
        if !self.results.is_empty() {
            string.push_str(&border);
        }

        string
    }
}

/// Schema of the table used to keep track of the database information.
fn mkdb_meta_schema() -> Vec<Column> {
    vec![
        // Either "index" or "table"
        Column {
            name: String::from("type"),
            data_type: DataType::Varchar(255),
            constraint: None,
        },
        // Index or table name
        Column {
            name: String::from("name"),
            data_type: DataType::Varchar(255),
            constraint: Some(Constraint::Unique),
        },
        // Root page
        Column {
            name: String::from("root"),
            data_type: DataType::Int,
            constraint: Some(Constraint::Unique),
        },
        // Table name
        Column {
            name: String::from("table_name"),
            data_type: DataType::Varchar(255),
            constraint: Some(Constraint::Unique),
        },
        // SQL used to create the index or table. TODO: Max char length
        Column {
            name: String::from("sql"),
            data_type: DataType::Varchar(255),
            constraint: None,
        },
    ]
}

impl<I> Database<I> {
    fn new(cache: Cache<I>) -> Self {
        Self {
            cache,
            row_ids: HashMap::new(),
        }
    }
}

impl Database<File> {
    pub fn init(path: impl AsRef<Path>) -> io::Result<Self> {
        let file = File::options()
            .create(true)
            .read(true)
            .write(true)
            .open(&path)?;

        let metadata = file.metadata()?;

        if !metadata.is_file() {
            return Err(io::Error::new(io::ErrorKind::Unsupported, "Not a file"));
        }

        // TODO: Constant
        let page_size = 4096;
        let block_size = Disk::from(&path).block_size()?;

        let mut pager = Pager::new(file, page_size, block_size);

        if pager.read_header()?.magic != MAGIC {
            pager.write_header(Header {
                magic: MAGIC,
                page_size: page_size as _,
                total_pages: 2,
                first_free_page: 0,
            })?;

            let root = Page::new(MKDB_META_ROOT, page_size as _);
            pager.write(MKDB_META_ROOT, root.buffer())?;
        }

        Ok(Database::new(Cache::new(pager)))
    }
}

impl<I: Seek + Read + Write> Database<I> {
    fn exec_where_clause(values: &Vec<Value>, schema: &Vec<Column>, expr: &Expression) -> Value {
        match expr {
            Expression::BinaryOperation {
                left,
                operator,
                right,
            } => {
                let left = Self::exec_where_clause(values, schema, &left);
                let right = Self::exec_where_clause(values, schema, &right);

                match operator {
                    BinaryOperator::Eq => Value::Bool(left == right),
                    BinaryOperator::Neq => Value::Bool(left != right),
                    BinaryOperator::And => Value::Bool(left.is_truthy() && right.is_truthy()),
                    BinaryOperator::Or => Value::Bool(left.is_truthy() || right.is_truthy()),
                    BinaryOperator::Lt => Value::Bool(left < right),
                    BinaryOperator::LtEq => Value::Bool(left <= right),
                    BinaryOperator::Gt => Value::Bool(left > right),
                    BinaryOperator::GtEq => Value::Bool(left >= right),

                    op => {
                        let Value::Number(left) = left else {
                            todo!("return some error");
                        };

                        let Value::Number(right) = right else {
                            todo!("return some error");
                        };

                        // TODO: Different types of integers
                        let left = left.parse::<u32>().unwrap();
                        let right = right.parse::<u32>().unwrap();

                        let result = match op {
                            BinaryOperator::Plus => left + right,
                            BinaryOperator::Minus => left - right,
                            BinaryOperator::Mul => left * right,
                            BinaryOperator::Div => left / right,
                            _ => unreachable!(),
                        };

                        Value::Number(result.to_string())
                    }
                }
            }

            Expression::Value(v) => v.clone(),

            Expression::Identifier(ident) => {
                // TODO: Store the position somewhere.
                let p = schema
                    .iter()
                    .position(|c| &c.name == ident)
                    .expect(&format!("todo: invalid column {ident}"));
                values[p].clone()
            }

            _ => todo!("return error"),
        }
    }

    fn serialize_values(values: Vec<Value>) -> Vec<u8> {
        let mut buf = Vec::new();

        // TODO: Alignment.
        for v in values {
            match v {
                Value::String(string) => {
                    let length = string.len().to_le_bytes();

                    // TODO: Strings longer than 65536 chars are not handled.
                    let n_bytes = if string.len() <= u8::MAX as usize {
                        1
                    } else {
                        2
                    };

                    buf.extend_from_slice(&length[..n_bytes]);
                    buf.extend_from_slice(string.as_bytes());
                }

                Value::Number(num) => {
                    // TODO: Support for types other than u32
                    buf.extend_from_slice(&num.parse::<u32>().unwrap().to_le_bytes());
                }

                Value::Bool(bool) => buf.push(u8::from(bool)),
            }
        }

        buf
    }

    fn deserialize_values(buf: Box<[u8]>, schema: &Vec<Column>) -> Vec<Value> {
        let mut values = Vec::new();

        // Skip row_id
        // TODO: Integrate RowID into Data type
        let mut idx = 8;

        for column in schema {
            match column.data_type {
                DataType::Varchar(_) => {
                    // TODO: Strings longer than 255 should always use 2 bytes to store the length.
                    let length = buf[idx];
                    idx += 1;
                    // TODO: Used lossy to avoid converting to Vec<u8>, check how to make this better.
                    values.push(Value::String(
                        String::from_utf8_lossy(&buf[idx..(idx + length as usize)]).to_string(),
                    ));
                    idx += length as usize;
                }

                DataType::Int => {
                    values.push(Value::Number(
                        u32::from_le_bytes(buf[idx..idx + 4].try_into().unwrap()).to_string(),
                    ));

                    // TODO: Numbers other than u32
                    idx += 4;
                }

                DataType::Bool => {
                    values.push(Value::Bool(if buf[idx] == 0 { false } else { true }));
                    idx += 1;
                }
            }
        }

        values
    }

    fn next_row_id(&mut self, table: String, root: PageNumber) -> u64 {
        if let Some(row_id) = self.row_ids.get_mut(&table) {
            *row_id += 1;
            return *row_id;
        }

        let mut btree = BTree::new_with_comparator(&mut self.cache, root, 1, RowIdComparator);

        // TODO: Error handling, use aggregate (SELECT MAX(row_id)...)
        let row_id = if let Some(max) = btree.max().unwrap() {
            u64::from_le_bytes(max.as_ref()[..8].try_into().unwrap())
        } else {
            1
        };

        self.row_ids.insert(table, row_id);
        row_id
    }

    pub fn execute(&mut self, input: String) -> QueryResult {
        let mut parser = Parser::new(&input);

        let mut statements = parser.try_parse()?;

        if statements.len() > 1 {
            todo!("handle multiple statements at once");
        }

        let statement = statements.remove(0);

        let stmt_sql = statement.to_string();

        // TODO: Parse and execute statements one by one.
        // TODO: SQL injections.
        match statement {
            Statement::Create(Create::Table { name, columns }) => {
                let table_root = Page::new(
                    self.cache.pager.alloc_page().unwrap(),
                    self.cache.pager.page_size as _,
                );
                self.cache
                    .pager
                    .write(table_root.number, table_root.buffer())?;
                let root_page = table_root.number;

                #[rustfmt::skip]
                self.execute(format!(r#"
                    INSERT INTO {MKDB_META} (type, name, root, table_name, sql)
                    VALUES ("table", "{name}", {root_page}, "{name}", '{stmt_sql}');
                "#))?;

                Ok(QueryExecution {
                    schema: Vec::new(),
                    results: Vec::new(),
                })
            }

            Statement::Insert {
                into,
                columns,
                values,
            } => {
                let (schema, root) = 'meta: {
                    if into == MKDB_META {
                        break 'meta (mkdb_meta_schema(), MKDB_META_ROOT);
                    }

                    let query = self.execute(format!(
                        "SELECT root, sql FROM {MKDB_META} where table_name = '{into}';"
                    ))?;

                    if query.is_empty() {
                        todo!("table doesn't exists");
                    }

                    // TODO: Find some way to avoid parsing SQL every time.
                    let schema = match query.get(0, "sql") {
                        Some(Value::String(sql)) => {
                            match Parser::new(&sql).try_parse()?.remove(0) {
                                Statement::Create(Create::Table { columns, .. }) => columns,
                                _ => unreachable!(),
                            }
                        }
                        _ => unreachable!(),
                    };

                    let root = match query.get(0, "root") {
                        Some(Value::Number(root)) => root.parse().unwrap(),
                        _ => unreachable!(),
                    };

                    (schema, root)
                };

                if columns.len() != values.len() {
                    todo!("number of columns doesn't match values");
                }

                if schema.len() != columns.len() {
                    todo!("missing columns");
                }

                let mut values_only = vec![Value::Bool(false); values.len()];

                for (col, expr) in columns.iter().zip(values) {
                    match expr {
                        Expression::Value(value) => {
                            // TODO: Do something about O(n^2)
                            let idx = schema.iter().position(|c| &c.name == col).unwrap();
                            values_only[idx] = value;
                        }
                        _ => todo!("resolve the value or throw error if not possible"),
                    }
                }

                let row_id = self.next_row_id(into, root);

                let mut btree =
                    BTree::new_with_comparator(&mut self.cache, root, 1, RowIdComparator);

                let mut buf = Vec::from(row_id.to_le_bytes());
                buf.append(&mut Self::serialize_values(values_only));

                btree.insert(buf)?;

                Ok(QueryExecution {
                    schema: Vec::new(),
                    results: Vec::new(),
                })
            }

            Statement::Select {
                columns,
                from,
                r#where,
                order_by,
            } => {
                let (mut schema, root) = 'meta: {
                    if from == MKDB_META {
                        break 'meta (mkdb_meta_schema(), MKDB_META_ROOT);
                    }

                    let query = self.execute(format!(
                        "SELECT root, sql FROM {MKDB_META} where table_name = '{from}';"
                    ))?;

                    if query.is_empty() {
                        todo!("table doesn't exist");
                    }

                    // TODO: Find some way to avoid parsing SQL every time.
                    let schema = match query.get(0, "sql") {
                        Some(Value::String(sql)) => {
                            match Parser::new(&sql).try_parse()?.remove(0) {
                                Statement::Create(Create::Table { columns, .. }) => columns,
                                _ => unreachable!(),
                            }
                        }
                        _ => unreachable!(),
                    };

                    let root = match query.get(0, "root") {
                        Some(Value::Number(root)) => root.parse().unwrap(),
                        _ => unreachable!(),
                    };

                    (schema, root)
                };

                let mut results = Vec::new();

                let mut btree =
                    BTree::new_with_comparator(&mut self.cache, root, 1, RowIdComparator);

                let mut identifiers = Vec::new();

                for c in &columns {
                    match c {
                        Expression::Identifier(ident) => identifiers.push(ident.to_owned()),
                        Expression::Wildcard => {
                            for s in &schema {
                                identifiers.push(s.name.to_owned());
                            }
                        }
                        _ => todo!("resolve expressions"),
                    }
                }

                for row in btree.iter() {
                    // TODO: Deserialize only needed values instead of all and the cloning...
                    let values = Self::deserialize_values(row?, &schema);

                    if let Some(expr) = &r#where {
                        if !Self::exec_where_clause(&values, &schema, &expr).is_truthy() {
                            continue;
                        }
                    }

                    let mut result = Vec::new();

                    for ident in &identifiers {
                        // TODO: O(n^2)
                        let p = schema.iter().position(|s| &s.name == ident).unwrap();
                        result.push(values[p].clone());
                    }

                    results.push(result);
                }

                let mut results_schema = Vec::new();

                for ident in identifiers {
                    let c = schema.remove(schema.iter().position(|s| s.name == ident).unwrap());
                    results_schema.push(c);
                }

                // TODO: Order by can contain column that we didn't select.
                if !order_by.is_empty() {
                    let order_by_cols = order_by
                        .iter()
                        .map(|o| results_schema.iter().position(|r| &r.name == o).unwrap())
                        .collect::<Vec<_>>();

                    results.sort_by(|a, b| {
                        for i in &order_by_cols {
                            let cmp = match (&a[*i], &b[*i]) {
                                (Value::Number(a), Value::Number(b)) => {
                                    a.parse::<u32>().unwrap().cmp(&b.parse().unwrap())
                                }
                                (Value::String(a), Value::String(b)) => a.cmp(b),
                                (Value::Bool(a), Value::Bool(b)) => todo!("order bools"),
                                _ => unreachable!("columns should have the same type"),
                            };

                            if cmp != Ordering::Equal {
                                return cmp;
                            }
                        }

                        Ordering::Equal
                    })
                }

                Ok(QueryExecution {
                    schema: results_schema,
                    results,
                })
            }

            _ => todo!("rest of SQL statements"),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::io;

    use super::{Database, DbError};
    use crate::{
        database::{mkdb_meta_schema, Header, QueryExecution, MAGIC, MKDB_META_ROOT},
        paging::{cache::Cache, pager::Pager},
        sql::{Column, Constraint, DataType, Parser, Value},
        storage::page::Page,
    };

    const PAGE_SIZE: usize = 4096;

    fn init_database() -> io::Result<Database<io::Cursor<Vec<u8>>>> {
        let mut pager = Pager::new(io::Cursor::new(Vec::<u8>::new()), PAGE_SIZE, PAGE_SIZE);

        pager.write_header(Header {
            magic: MAGIC,
            page_size: PAGE_SIZE as _,
            total_pages: 2,
            first_free_page: 0,
        })?;

        let root = Page::new(MKDB_META_ROOT, PAGE_SIZE as _);
        pager.write(MKDB_META_ROOT, root.buffer())?;

        Ok(Database::new(Cache::new(pager)))
    }

    #[test]
    fn create_table() -> Result<(), DbError> {
        let mut db = init_database()?;

        let sql = "CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(255));";
        db.execute(sql.into())?;

        let query = db.execute("SELECT * FROM mkdb_meta;".into())?;

        assert_eq!(
            query,
            QueryExecution {
                schema: mkdb_meta_schema(),
                results: vec![vec![
                    Value::String("table".into()),
                    Value::String("users".into()),
                    Value::Number("2".into()),
                    Value::String("users".into()),
                    Value::String(Parser::new(&sql).try_parse().unwrap()[0].to_string())
                ]]
            }
        );

        Ok(())
    }

    #[test]
    fn insert_data() -> Result<(), DbError> {
        let mut db = init_database()?;

        db.execute("CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(255));".into())?;
        db.execute("INSERT INTO users(id, name) VALUES (1, 'John Doe');".into())?;
        db.execute("INSERT INTO users(id, name) VALUES (2, 'Jane Doe');".into())?;

        let query = db.execute("SELECT * FROM users;".into())?;

        assert_eq!(
            query,
            QueryExecution {
                schema: vec![
                    Column {
                        name: "id".into(),
                        data_type: DataType::Int,
                        constraint: Some(Constraint::PrimaryKey),
                    },
                    Column {
                        name: "name".into(),
                        data_type: DataType::Varchar(255),
                        constraint: None
                    }
                ],
                results: vec![
                    vec![Value::Number("1".into()), Value::String("John Doe".into())],
                    vec![Value::Number("2".into()), Value::String("Jane Doe".into())],
                ]
            }
        );

        Ok(())
    }

    #[test]
    fn insert_many() -> Result<(), DbError> {
        let mut db = init_database()?;

        let create_table = r#"
            CREATE TABLE users (
                id INT PRIMARY KEY,
                name VARCHAR(255),
                email VARCHAR(255) UNIQUE
            );
        "#;

        db.execute(create_table.into())?;

        let mut expected = Vec::new();

        for i in 1..=100 {
            let name = format!("User {i}");
            let email = format!("user{i}@test.com");

            expected.push(vec![
                Value::Number(i.to_string()),
                Value::String(name.clone()),
                Value::String(email.clone()),
            ]);

            let insert =
                format!("INSERT INTO users(id, name, email) VALUES ({i}, '{name}', '{email}');");

            db.execute(insert.into())?;
        }

        let query = db.execute("SELECT * FROM users ORDER BY id;".into())?;

        assert_eq!(
            query,
            QueryExecution {
                schema: vec![
                    Column {
                        name: "id".into(),
                        data_type: DataType::Int,
                        constraint: Some(Constraint::PrimaryKey),
                    },
                    Column {
                        name: "name".into(),
                        data_type: DataType::Varchar(255),
                        constraint: None,
                    },
                    Column {
                        name: "email".into(),
                        data_type: DataType::Varchar(255),
                        constraint: Some(Constraint::Unique),
                    }
                ],
                results: expected
            }
        );

        Ok(())
    }

    #[test]
    fn select_where() -> Result<(), DbError> {
        let mut db = init_database()?;

        db.execute("CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(255), age INT);".into())?;
        db.execute("INSERT INTO users(id, name, age) VALUES (1, 'John Doe', 18);".into())?;
        db.execute("INSERT INTO users(id, name, age) VALUES (2, 'Jane Doe', 22);".into())?;
        db.execute("INSERT INTO users(id, name, age) VALUES (3, 'Some Dude', 24);".into())?;

        let query = db.execute("SELECT * FROM users WHERE age > 18;".into())?;

        assert_eq!(
            query,
            QueryExecution {
                schema: vec![
                    Column {
                        name: "id".into(),
                        data_type: DataType::Int,
                        constraint: Some(Constraint::PrimaryKey),
                    },
                    Column {
                        name: "name".into(),
                        data_type: DataType::Varchar(255),
                        constraint: None
                    },
                    Column {
                        name: "age".into(),
                        data_type: DataType::Int,
                        constraint: None
                    }
                ],
                results: vec![
                    vec![
                        Value::Number("2".into()),
                        Value::String("Jane Doe".into()),
                        Value::Number("22".into())
                    ],
                    vec![
                        Value::Number("3".into()),
                        Value::String("Some Dude".into()),
                        Value::Number("24".into())
                    ],
                ]
            }
        );

        Ok(())
    }

    #[test]
    fn create_multiple_tables() -> Result<(), DbError> {
        let mut db = init_database()?;

        let t1 = "CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(255));";
        let t2 = "CREATE TABLE tasks (id INT PRIMARY KEY, title VARCHAR(255), description VARCHAR(255));";
        let t3 = "CREATE TABLE products (id INT PRIMARY KEY, price INT);";

        db.execute(t1.into())?;
        db.execute(t2.into())?;
        db.execute(t3.into())?;

        let query = db.execute("SELECT * FROM mkdb_meta;".into())?;

        assert_eq!(
            query,
            QueryExecution {
                schema: mkdb_meta_schema(),
                results: vec![
                    vec![
                        Value::String("table".into()),
                        Value::String("users".into()),
                        Value::Number("2".into()),
                        Value::String("users".into()),
                        Value::String(Parser::new(&t1).try_parse().unwrap()[0].to_string())
                    ],
                    vec![
                        Value::String("table".into()),
                        Value::String("tasks".into()),
                        Value::Number("3".into()),
                        Value::String("tasks".into()),
                        Value::String(Parser::new(&t2).try_parse().unwrap()[0].to_string())
                    ],
                    vec![
                        Value::String("table".into()),
                        Value::String("products".into()),
                        Value::Number("4".into()),
                        Value::String("products".into()),
                        Value::String(Parser::new(&t3).try_parse().unwrap()[0].to_string())
                    ]
                ]
            }
        );

        Ok(())
    }
}
