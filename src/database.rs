use std::{
    cmp::Ordering,
    collections::HashMap,
    fmt::Display,
    fs::File,
    io::{self, Read, Seek, Write},
    mem,
    path::Path,
    usize,
};

use crate::{
    os::{Disk, HardwareBlockSize},
    paging::{
        cache::Cache,
        pager::{PageNumber, Pager},
    },
    sql::{
        BinaryOperator, Column, Constraint, Create, DataType, Expression, Parser, ParserError,
        Statement, UnaryOperator, Value,
    },
    storage::{page::Page, BTree, FixedSizeMemCmp, DEFAULT_BALANCE_SIBLINGS_PER_SIDE},
};

/// Database file default page size.
pub(crate) const DEFAULT_PAGE_SIZE: usize = 4096;

/// Name of the meta-table used to keep track of other tables.
pub(crate) const MKDB_META: &str = "mkdb_meta";

/// Root page of the meta-table. Page 0 holds the DB header, page 1 holds the
/// beginning of the meta-table.
pub(crate) const MKDB_META_ROOT: PageNumber = 1;

/// Magic number at the beginning of the database file.
pub(crate) const MAGIC: u32 = 0xB74EE;

/// Rows are uniquely identified by an 8 byte key stored in big endian at the
/// beginning of each tuple.
type RowId = u64;

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
pub(crate) struct QueryResolution {
    schema: Schema,
    results: Vec<Vec<Value>>,
}

#[derive(Debug, PartialEq)]
pub(crate) enum TypeError {
    CannotApplyUnary {
        operator: UnaryOperator,
        value: Value,
    },
    CannotApplyBinary {
        left: Value,
        operator: BinaryOperator,
        right: Value,
    },
    ExpectedType {
        expected: DataType,
        found: Value,
    },
}

#[derive(Debug, PartialEq)]
pub(crate) enum ExpectedExpression {
    Identifier,
    Assignment,
    Value,
}

impl Display for ExpectedExpression {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Identifier => "identifier",
            Self::Value => "raw value",
            Self::Assignment => "assignment",
        })
    }
}

#[derive(Debug, PartialEq)]
pub(crate) enum SqlError {
    InvalidTable(String),
    InvalidColumn(String),
    ColumnValueCountMismatch,
    MissingColumns,
    TypeError(TypeError),
    Expected {
        expected: ExpectedExpression,
        found: Expression,
    },
    Other(String),
}

impl Display for SqlError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidTable(name) => write!(f, "invalid table '{name}'"),
            Self::InvalidColumn(name) => write!(f, "invalid column '{name}'"),
            Self::ColumnValueCountMismatch => f.write_str("number of columns doesn't match values"),
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
            Self::Expected { expected, found } => write!(f, "expected {expected}, found {found}"),
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

#[derive(Debug, PartialEq)]
pub(crate) struct Schema {
    pub columns: Vec<Column>,
    pub index: HashMap<String, usize>,
}

impl Schema {
    pub fn new(columns: Vec<Column>) -> Self {
        let index = columns
            .iter()
            .enumerate()
            .map(|(i, col)| (col.name.clone(), i))
            .collect();

        Self { columns, index }
    }

    pub fn empty() -> Self {
        Self::new(vec![])
    }

    pub fn index_of(&self, col: &str) -> Option<usize> {
        self.index.get(col).copied()
    }

    pub fn len(&self) -> usize {
        self.columns.len()
    }

    pub fn push(&mut self, column: Column) {
        self.index.insert(column.name.to_owned(), self.len());
        self.columns.push(column);
    }
}

impl From<Vec<Column>> for Schema {
    fn from(columns: Vec<Column>) -> Self {
        Self::new(columns)
    }
}

type QueryResult = Result<QueryResolution, DbError>;

impl QueryResolution {
    fn new(schema: Schema, results: Vec<Vec<Value>>) -> Self {
        Self { schema, results }
    }

    fn empty() -> Self {
        Self {
            schema: Schema::empty(),
            results: Vec::new(),
        }
    }

    fn get(&self, row: usize, column: &str) -> Option<&Value> {
        self.results.get(row)?.get(self.schema.index_of(column)?)
    }

    pub fn is_empty(&self) -> bool {
        self.results.is_empty()
    }

    pub fn as_ascii_table(&self) -> String {
        // Initialize width of each column to the length of the table headers.
        let mut widths: Vec<usize> = self
            .schema
            .columns
            .iter()
            .map(|col| col.name.len())
            .collect();

        // We only need strings.
        let rows: Vec<Vec<String>> = self
            .results
            .iter()
            .map(|row| row.iter().map(ToString::to_string).collect())
            .collect();

        // Find the maximum width for each column.
        for row in &rows {
            for (i, col) in row.iter().enumerate() {
                if col.len() > widths[i] {
                    widths[i] = col.len();
                }
            }
        }

        // We'll add both a leading and trailing space to the widest string in
        // each column, so increase width by 2.
        widths.iter_mut().for_each(|w| *w += 2);

        // Create border according to width: +-----+---------+------+-----+
        let mut border = String::from('+');
        for width in &widths {
            for _ in 0..*width {
                border.push('-');
            }
            border.push('+');
        }

        // Builds one row: | for | example | this | one |
        let make_row = |row: &Vec<String>| -> String {
            let mut string = String::from('|');

            for (i, col) in row.iter().enumerate() {
                string.push(' ');
                string.push_str(&col);
                for _ in 0..widths[i] - col.len() - 1 {
                    string.push(' ');
                }
                string.push('|');
            }

            string
        };

        // Header
        let mut table = String::from(&border);
        table.push('\n');

        table.push_str(&make_row(
            &self
                .schema
                .columns
                .iter()
                .map(|col| col.name.clone())
                .collect(),
        ));
        table.push('\n');

        table.push_str(&border);
        table.push('\n');

        // Content
        for row in &rows {
            table.push_str(&make_row(row));
            table.push('\n');
        }

        if !rows.is_empty() {
            table.push_str(&border);
        }

        table
    }
}

/// Schema of the table used to keep track of the database information.
fn mkdb_meta_schema() -> Schema {
    Schema::from(vec![
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
        // SQL used to create the index or table.
        // TODO: Implement and use some TEXT data type with higher length limits.
        Column {
            name: String::from("sql"),
            data_type: DataType::Varchar(1000),
            constraint: None,
        },
    ])
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

        let block_size = Disk::from(&path).block_size()?;

        let mut pager = Pager::new(file, DEFAULT_PAGE_SIZE, block_size);

        // TODO: Recreate the pager if the file exists and the header page size
        // is different than the default.
        if pager.read_header()?.magic != MAGIC {
            pager.write_header(Header {
                magic: MAGIC,
                page_size: DEFAULT_PAGE_SIZE as _,
                total_pages: 2,
                first_free_page: 0,
            })?;

            let root = Page::new(MKDB_META_ROOT, DEFAULT_PAGE_SIZE as _);
            pager.write(MKDB_META_ROOT, root.buffer())?;
        }

        Ok(Database::new(Cache::new(pager)))
    }
}

fn deserialize_row_id(buf: &[u8]) -> RowId {
    RowId::from_be_bytes(buf[..mem::size_of::<RowId>()].try_into().unwrap())
}

fn serialize_row_id(row_id: RowId) -> [u8; mem::size_of::<RowId>()] {
    row_id.to_be_bytes()
}

impl<I: Seek + Read + Write> Database<I> {
    fn btree(&mut self, root: PageNumber) -> BTree<'_, I, FixedSizeMemCmp> {
        BTree::new(
            &mut self.cache,
            root,
            DEFAULT_BALANCE_SIBLINGS_PER_SIDE,
            FixedSizeMemCmp::for_type::<RowId>(),
        )
    }

    fn resolve_expression(
        values: &Vec<Value>,
        schema: &Schema,
        expr: &Expression,
    ) -> Result<Value, SqlError> {
        match expr {
            Expression::Value(value) => Ok(value.clone()),

            Expression::Identifier(ident) => match schema.index_of(&ident) {
                Some(index) => Ok(values[index].clone()),
                None => Err(SqlError::InvalidColumn(ident.clone())),
            },

            Expression::UnaryOperation { operator, expr } => {
                match Self::resolve_expression(values, schema, expr)? {
                    Value::Number(mut num) => {
                        if let UnaryOperator::Minus = operator {
                            num = -num;
                        }

                        Ok(Value::Number(num))
                    }

                    value => Err(SqlError::TypeError(TypeError::CannotApplyUnary {
                        operator: *operator,
                        value,
                    })),
                }
            }

            Expression::BinaryOperation {
                left,
                operator,
                right,
            } => {
                let left = Self::resolve_expression(values, schema, &left)?;
                let right = Self::resolve_expression(values, schema, &right)?;

                let mismatched_types = SqlError::TypeError(TypeError::CannotApplyBinary {
                    left: left.clone(),
                    operator: *operator,
                    right: right.clone(),
                });

                if mem::discriminant(&left) != mem::discriminant(&right) {
                    return Err(mismatched_types);
                }

                Ok(match operator {
                    BinaryOperator::Eq => Value::Bool(left == right),
                    BinaryOperator::Neq => Value::Bool(left != right),
                    BinaryOperator::Lt => Value::Bool(left < right),
                    BinaryOperator::LtEq => Value::Bool(left <= right),
                    BinaryOperator::Gt => Value::Bool(left > right),
                    BinaryOperator::GtEq => Value::Bool(left >= right),

                    logical @ (BinaryOperator::And | BinaryOperator::Or) => {
                        let (Value::Bool(left), Value::Bool(right)) = (left, right) else {
                            return Err(mismatched_types);
                        };

                        match logical {
                            BinaryOperator::And => Value::Bool(left && right),
                            BinaryOperator::Or => Value::Bool(left || right),
                            _ => unreachable!(),
                        }
                    }

                    arithmetic => {
                        let (Value::Number(left), Value::Number(right)) = (left, right) else {
                            return Err(mismatched_types);
                        };

                        Value::Number(match arithmetic {
                            BinaryOperator::Plus => left + right,
                            BinaryOperator::Minus => left - right,
                            BinaryOperator::Mul => left * right,
                            BinaryOperator::Div => left / right,
                            _ => unreachable!(),
                        })
                    }
                })
            }

            Expression::Wildcard => {
                unreachable!("wildcards should be resolved into identifiers at this point")
            }
        }
    }

    fn eval_where(
        schema: &Schema,
        values: &Vec<Value>,
        r#where: &Option<Expression>,
    ) -> Result<bool, SqlError> {
        let Some(expr) = r#where else {
            return Ok(true);
        };

        match Self::resolve_expression(&values, &schema, &expr)? {
            Value::Bool(b) => Ok(b),

            other => Err(SqlError::TypeError(TypeError::ExpectedType {
                expected: DataType::Bool,
                found: other,
            })),
        }
    }

    fn serialize_values(row_id: RowId, schema: &Schema, values: &Vec<Value>) -> Vec<u8> {
        let mut buf = Vec::from(serialize_row_id(row_id));

        macro_rules! serialize_little_endian {
            ($num:expr, $int:ty) => {
                TryInto::<$int>::try_into(*$num).unwrap().to_le_bytes()
            };
        }

        // TODO: Alignment.
        for (col, val) in schema.columns.iter().zip(values) {
            match (&col.data_type, val) {
                (DataType::Varchar(max), Value::String(string)) => {
                    let length = string.len().to_le_bytes();

                    // TODO: Strings longer than 65535 chars are not handled.
                    let n_bytes = if *max <= u8::MAX as usize { 1 } else { 2 };

                    buf.extend_from_slice(&length[..n_bytes]);
                    buf.extend_from_slice(string.as_bytes());
                }

                (DataType::Int, Value::Number(num)) => {
                    buf.extend_from_slice(&serialize_little_endian!(num, i32));
                }

                (DataType::UnsignedInt, Value::Number(num)) => {
                    buf.extend_from_slice(&serialize_little_endian!(num, u32));
                }

                (DataType::BigInt, Value::Number(num)) => {
                    buf.extend_from_slice(&serialize_little_endian!(num, i64));
                }

                (DataType::UnsignedBigInt, Value::Number(num)) => {
                    buf.extend_from_slice(&serialize_little_endian!(num, u64));
                }

                (DataType::Bool, Value::Bool(bool)) => buf.push(u8::from(*bool)),

                _ => unreachable!("attempt to serialize {val} into {}", col.data_type),
            }
        }

        buf
    }

    fn deserialize_values(buf: Box<[u8]>, schema: &Schema) -> (RowId, Vec<Value>) {
        let mut values = Vec::new();
        let row_id = deserialize_row_id(&buf);
        let mut index = mem::size_of::<RowId>();

        macro_rules! deserialize_little_endian {
            ($buf:expr, $index:expr, $int:ty) => {
                <$int>::from_le_bytes(
                    $buf[index..index + mem::size_of::<$int>()]
                        .try_into()
                        .unwrap(),
                )
                .into()
            };
        }

        // TODO: Alignment.
        for column in &schema.columns {
            match column.data_type {
                DataType::Varchar(max) => {
                    let length = if max <= u8::MAX as usize {
                        let len = buf[index];
                        index += 1;
                        len as usize
                    } else {
                        let len: usize = deserialize_little_endian!(buf, index, u16);
                        index += 2;
                        len
                    };

                    // TODO: Check if we should use from_ut8_lossy() or from_utf8()
                    values.push(Value::String(
                        String::from_utf8_lossy(&buf[index..(index + length)]).to_string(),
                    ));
                    index += length;
                }

                DataType::Int => {
                    values.push(Value::Number(deserialize_little_endian!(buf, index, i32)));
                    index += mem::size_of::<i32>();
                }

                DataType::UnsignedInt => {
                    values.push(Value::Number(deserialize_little_endian!(buf, index, u32)));
                    index += mem::size_of::<u32>();
                }

                DataType::BigInt => {
                    values.push(Value::Number(deserialize_little_endian!(buf, index, i64)));
                    index += mem::size_of::<i64>();
                }

                DataType::UnsignedBigInt => {
                    values.push(Value::Number(deserialize_little_endian!(buf, index, u64)));
                    index += mem::size_of::<u64>();
                }

                DataType::Bool => {
                    values.push(Value::Bool(if buf[index] == 0 { false } else { true }));
                    index += mem::size_of::<bool>();
                }
            }
        }

        (row_id, values)
    }

    fn table_metadata(&mut self, table: &String) -> Result<(Schema, PageNumber), DbError> {
        if table == MKDB_META {
            return Ok((mkdb_meta_schema(), MKDB_META_ROOT));
        }

        let query = self.exec(&format!(
            "SELECT root, sql FROM {MKDB_META} where table_name = '{table}';"
        ))?;

        if query.is_empty() {
            return Err(DbError::Sql(SqlError::InvalidTable(table.clone())));
        }

        // TODO: Find some way to avoid parsing SQL every time. Probably a
        // hash map of table name -> schema, we wouldn't even need to update it
        // as we don't support ALTER table statements.
        let schema = match query.get(0, "sql") {
            Some(Value::String(sql)) => match Parser::new(&sql).parse_statement()? {
                Statement::Create(Create::Table { columns, .. }) => Schema::from(columns),
                _ => unreachable!(),
            },
            _ => unreachable!(),
        };

        let root = match query.get(0, "root") {
            Some(Value::Number(root)) => *root as PageNumber,
            _ => unreachable!(),
        };

        Ok((schema, root))
    }

    fn next_row_id(&mut self, table: String, root: PageNumber) -> io::Result<RowId> {
        if let Some(row_id) = self.row_ids.get_mut(&table) {
            *row_id += 1;
            return Ok(*row_id);
        }

        let mut btree = self.btree(root);

        // TODO: Error handling, use aggregate (SELECT MAX(row_id)...)
        let row_id = if let Some(max) = btree.max()? {
            deserialize_row_id(max.as_ref()) + 1
        } else {
            1
        };

        self.row_ids.insert(table, row_id);
        Ok(row_id)
    }

    pub fn exec(&mut self, input: &str) -> QueryResult {
        let mut parser = Parser::new(input);

        let mut statements = parser.try_parse()?;

        if statements.len() > 1 {
            todo!("handle multiple statements at once");
        }

        let statement = statements.remove(0);

        // TODO: Parse and execute statements one by one.
        // TODO: SQL injections through the table name?.
        match statement {
            Statement::Create(Create::Table { name, columns }) => {
                let table_root = Page::new(
                    self.cache.pager.alloc_page()?,
                    self.cache.pager.page_size as _,
                );
                self.cache
                    .pager
                    .write(table_root.number, table_root.buffer())?;
                let root_page = table_root.number;

                self.exec(&format!(
                    r#"
                        INSERT INTO {MKDB_META} (type, name, root, table_name, sql)
                        VALUES ("table", "{name}", {root_page}, "{name}", '{sql}');
                    "#,
                    sql = Statement::Create(Create::Table {
                        name: name.clone(),
                        columns,
                    })
                ))?;

                Ok(QueryResolution::empty())
            }

            Statement::Insert {
                into,
                columns,
                values,
            } => {
                let (schema, root) = self.table_metadata(&into)?;

                if columns.len() != values.len() {
                    return Err(DbError::Sql(SqlError::ColumnValueCountMismatch));
                }

                for col in &columns {
                    if !schema.index.contains_key(col) {
                        return Err(DbError::Sql(SqlError::InvalidColumn(col.to_owned())));
                    }
                }

                if schema.len() != columns.len() {
                    return Err(DbError::Sql(SqlError::MissingColumns));
                }

                let mut resolved_values = vec![Value::Bool(false); values.len()];

                for (col, expr) in columns.iter().zip(values) {
                    let value = match Self::resolve_expression(&Vec::new(), &Schema::empty(), &expr)
                    {
                        Ok(value) => value,

                        Err(SqlError::InvalidColumn(_col)) => {
                            Err(DbError::Sql(SqlError::Expected {
                                expected: ExpectedExpression::Value,
                                found: expr,
                            }))?
                        }

                        Err(e) => Err(e)?,
                    };

                    // There should be only valid columns here, we can unwrap.
                    let index = schema.index_of(col).unwrap();

                    match (schema.columns[index].data_type, &value) {
                        (DataType::Bool, Value::Bool(_))
                        | (DataType::Varchar(_), Value::String(_))
                        | (_, Value::Number(_)) => {
                            resolved_values[schema.index_of(col).unwrap()] = value;
                        }
                        (expected, _) => {
                            return Err(DbError::Sql(SqlError::TypeError(
                                TypeError::ExpectedType {
                                    expected,
                                    found: value,
                                },
                            )))
                        }
                    }
                }

                let row_id = self.next_row_id(into, root)?;

                let mut btree = self.btree(root);

                btree.insert(Self::serialize_values(row_id, &schema, &resolved_values))?;

                Ok(QueryResolution::empty())
            }

            Statement::Select {
                mut columns,
                from,
                r#where,
                order_by,
            } => {
                let (schema, root) = self.table_metadata(&from)?;

                let mut results_schema = Schema::empty();
                let mut unknown_types = Vec::new();

                columns = {
                    let mut resolved_wildcards = Vec::new();
                    for expr in columns {
                        if let &Expression::Wildcard = &expr {
                            for col in &schema.columns {
                                resolved_wildcards
                                    .push(Expression::Identifier(col.name.to_owned()));
                            }
                        } else {
                            resolved_wildcards.push(expr);
                        }
                    }
                    resolved_wildcards
                };

                // Adjust results schema
                for (i, expr) in columns.iter().enumerate() {
                    match expr {
                        Expression::Identifier(ident) => match schema.index_of(ident) {
                            Some(index) => results_schema.push(schema.columns[index].clone()),
                            None => Err(DbError::Sql(SqlError::InvalidColumn(ident.clone())))?,
                        },

                        _ => {
                            results_schema.push(Column {
                                name: expr.to_string(),    // TODO: AS alias
                                data_type: DataType::Bool, // We'll set it later
                                constraint: None,
                            });

                            unknown_types.push(i);
                        }
                    }
                }

                let mut results = Vec::new();

                let mut btree = self.btree(root);

                for row in btree.iter() {
                    let (_, values) = Self::deserialize_values(row?, &schema);

                    if !Self::eval_where(&schema, &values, &r#where)? {
                        continue;
                    }

                    let mut result = Vec::new();

                    for expr in &columns {
                        result.push(Self::resolve_expression(&values, &schema, expr)?);
                    }

                    results.push(result);
                }

                // We already set the default of unknown types as bools, if
                // it's a number then change it to BigInt. We don't support any
                // expressions that produce strings. And we don't use the types
                // of results for anything now anyway.
                if !results.is_empty() {
                    for i in unknown_types {
                        if let Value::Number(_) = &results[0][i] {
                            results_schema.columns[i].data_type = DataType::BigInt;
                        }
                    }
                }

                // TODO: Order by can contain column that we didn't select.
                if !order_by.is_empty() {
                    let mut order_by_cols = Vec::new();

                    for identifier in order_by {
                        match results_schema.index_of(&identifier) {
                            Some(index) => order_by_cols.push(index),
                            None => Err(DbError::Sql(SqlError::Other(format!(
                                "ordering by columns not present in SELECT is not supported"
                            ))))?,
                        }
                    }

                    results.sort_by(|a, b| {
                        for i in &order_by_cols {
                            let cmp = match (&a[*i], &b[*i]) {
                                (Value::Number(a), Value::Number(b)) => a.cmp(b),
                                (Value::String(a), Value::String(b)) => a.cmp(b),
                                (Value::Bool(a), Value::Bool(b)) => a.cmp(b),
                                _ => unreachable!("columns should have the same type"),
                            };

                            if cmp != Ordering::Equal {
                                return cmp;
                            }
                        }

                        Ordering::Equal
                    })
                }

                Ok(QueryResolution::new(results_schema, results))
            }

            Statement::Delete { from, r#where } => {
                let (schema, root) = self.table_metadata(&from)?;

                let mut btree = self.btree(root);

                // TODO: Use some cursor or something to delete as we traverse the tree.
                let mut row_ids = Vec::new();

                for row in btree.iter() {
                    // TODO: Deserialize only needed values instead of all and the cloning...
                    let (row_id, values) = Self::deserialize_values(row?, &schema);

                    if !Self::eval_where(&schema, &values, &r#where)? {
                        continue;
                    }

                    row_ids.push(row_id);
                }

                // TODO: Second mutable borrow occurs here?
                let mut btree = self.btree(root);

                for row_id in row_ids {
                    btree.remove(&serialize_row_id(row_id))?;
                }

                Ok(QueryResolution::empty())
            }

            Statement::Update {
                table,
                columns,
                r#where,
            } => {
                let (schema, root) = self.table_metadata(&table)?;

                let mut assignments = Vec::new();

                for col in columns {
                    let Expression::BinaryOperation {
                        left,
                        operator: BinaryOperator::Eq,
                        right,
                    } = col
                    else {
                        return Err(DbError::Sql(SqlError::Expected {
                            expected: ExpectedExpression::Assignment,
                            found: col,
                        }));
                    };

                    let Expression::Identifier(ident) = *left else {
                        return Err(DbError::Sql(SqlError::Expected {
                            expected: ExpectedExpression::Identifier,
                            found: *left,
                        }));
                    };

                    assignments.push((ident, right));
                }

                let mut btree = self.btree(root);

                let mut updates = Vec::new();

                for row in btree.iter() {
                    // TODO: Deserialize only needed values instead of all and then cloning...
                    let (row_id, mut values) = Self::deserialize_values(row?, &schema);

                    if !Self::eval_where(&schema, &values, &r#where)? {
                        continue;
                    }

                    for (col, expr) in &assignments {
                        let value = match **expr {
                            Expression::Value(ref v) => v.clone(),
                            _ => Self::resolve_expression(&values, &schema, expr)?,
                        };

                        let index = schema
                            .index_of(col)
                            .ok_or(SqlError::InvalidColumn(col.clone()))?;
                        values[index] = value;
                        updates.push(Self::serialize_values(row_id, &schema, &values));
                    }
                }

                // TODO: Second mutable borrow occurs here?
                let mut btree = self.btree(root);

                for update in updates {
                    btree.insert(update)?;
                }

                Ok(QueryResolution::empty())
            }

            _ => todo!("rest of SQL statements"),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::io;

    use super::{Database, DbError, DEFAULT_PAGE_SIZE};
    use crate::{
        database::{
            mkdb_meta_schema, Header, QueryResolution, Schema, SqlError, TypeError, MAGIC,
            MKDB_META_ROOT,
        },
        paging::{cache::Cache, pager::Pager},
        sql::{Column, Constraint, DataType, Parser, Value},
        storage::page::Page,
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

    fn init_database() -> io::Result<Database<io::Cursor<Vec<u8>>>> {
        let mut pager = Pager::new(
            io::Cursor::new(Vec::<u8>::new()),
            DEFAULT_PAGE_SIZE,
            DEFAULT_PAGE_SIZE,
        );

        pager.write_header(Header {
            magic: MAGIC,
            page_size: DEFAULT_PAGE_SIZE as _,
            total_pages: 2,
            first_free_page: 0,
        })?;

        let root = Page::new(MKDB_META_ROOT, DEFAULT_PAGE_SIZE as _);
        pager.write(MKDB_META_ROOT, root.buffer())?;

        Ok(Database::new(Cache::new(pager)))
    }

    #[test]
    fn create_table() -> Result<(), DbError> {
        let mut db = init_database()?;

        let sql = "CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(255));";
        db.exec(sql)?;

        let query = db.exec("SELECT * FROM mkdb_meta;")?;

        assert_eq!(
            query,
            QueryResolution::new(
                mkdb_meta_schema(),
                vec![vec![
                    Value::String("table".into()),
                    Value::String("users".into()),
                    Value::Number(2),
                    Value::String("users".into()),
                    Value::String(Parser::new(sql).parse_statement()?.to_string())
                ]]
            )
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

        assert_eq!(
            query,
            QueryResolution {
                schema: Schema::from(vec![
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
                ]),
                results: vec![
                    vec![Value::Number(1), Value::String("John Doe".into())],
                    vec![Value::Number(2), Value::String("Jane Doe".into())],
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

        db.exec(create_table)?;

        let mut expected = Vec::new();

        for i in 1..=100 {
            let name = format!("User {i}");
            let email = format!("user{i}@test.com");

            expected.push(vec![
                Value::Number(i),
                Value::String(name.clone()),
                Value::String(email.clone()),
            ]);

            db.exec(&format!(
                "INSERT INTO users(id, name, email) VALUES ({i}, '{name}', '{email}');"
            ))?;
        }

        let query = db.exec("SELECT * FROM users ORDER BY id;")?;

        assert_eq!(
            query,
            QueryResolution {
                schema: Schema::from(vec![
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
                ]),
                results: expected
            }
        );

        Ok(())
    }

    #[test]
    fn insert_disordered() -> Result<(), DbError> {
        let mut db = init_database()?;

        db.exec("CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(255), age INT);")?;
        db.exec("INSERT INTO users(name, age, id) VALUES ('John Doe', 18, 1);")?;
        db.exec("INSERT INTO users(name, age, id) VALUES ('Jane Doe', 22, 2);")?;

        let query = db.exec("SELECT * FROM users;")?;

        assert_eq!(
            query,
            QueryResolution {
                schema: Schema::from(vec![
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
            }
        );

        Ok(())
    }

    #[test]
    fn insert_expressions() -> Result<(), DbError> {
        let mut db = init_database()?;

        db.exec("CREATE TABLE products (id INT PRIMARY KEY, price INT, discount INT);")?;
        db.exec("INSERT INTO products(id, price, discount) VALUES (1, 10*10 + 10, 2+2);")?;
        db.exec("INSERT INTO products(id, price, discount) VALUES (2, 50 - 5*2, 100 / (3+2));")?;

        let query = db.exec("SELECT * FROM products;")?;

        assert_eq!(
            query,
            QueryResolution {
                schema: Schema::from(vec![
                    Column {
                        name: "id".into(),
                        data_type: DataType::Int,
                        constraint: Some(Constraint::PrimaryKey),
                    },
                    Column {
                        name: "price".into(),
                        data_type: DataType::Int,
                        constraint: None
                    },
                    Column {
                        name: "discount".into(),
                        data_type: DataType::Int,
                        constraint: None
                    }
                ]),
                results: vec![
                    vec![Value::Number(1), Value::Number(110), Value::Number(4),],
                    vec![Value::Number(2), Value::Number(40), Value::Number(20),],
                ]
            }
        );

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

        assert_eq!(
            query,
            QueryResolution {
                schema: Schema::from(vec![
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
            }
        );

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

        assert_eq!(
            query,
            QueryResolution {
                schema: Schema::from(vec![
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
            }
        );

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

        assert_eq!(
            query,
            QueryResolution {
                schema: Schema::from(vec![
                    Column {
                        name: "age".into(),
                        data_type: DataType::Int,
                        constraint: None
                    },
                    Column {
                        name: "name".into(),
                        data_type: DataType::Varchar(255),
                        constraint: None
                    },
                    Column {
                        name: "id".into(),
                        data_type: DataType::Int,
                        constraint: Some(Constraint::PrimaryKey),
                    },
                    Column {
                        name: "is_admin".into(),
                        data_type: DataType::Bool,
                        constraint: None,
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
            }
        );

        Ok(())
    }

    #[test]
    fn select_expressions() -> Result<(), DbError> {
        let mut db = init_database()?;

        db.exec("CREATE TABLE products (id INT PRIMARY KEY, price INT, discount INT);")?;
        db.exec("INSERT INTO products(id, price, discount) VALUES (1, 100, 5);")?;
        db.exec("INSERT INTO products(id, price, discount) VALUES (2, 250, 10);")?;

        let query = db.exec("SELECT id, price / 10, discount * 100 FROM products;")?;

        assert_eq!(
            query,
            QueryResolution {
                schema: Schema::from(vec![
                    Column {
                        name: "id".into(),
                        data_type: DataType::Int,
                        constraint: Some(Constraint::PrimaryKey),
                    },
                    Column {
                        name: "(price) / (10)".into(),
                        data_type: DataType::BigInt,
                        constraint: None
                    },
                    Column {
                        name: "(discount) * (100)".into(),
                        data_type: DataType::BigInt,
                        constraint: None
                    }
                ]),
                results: vec![
                    vec![Value::Number(1), Value::Number(10), Value::Number(500),],
                    vec![Value::Number(2), Value::Number(25), Value::Number(1000),],
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

        db.exec(t1)?;
        db.exec(t2)?;
        db.exec(t3)?;

        let query = db.exec("SELECT * FROM mkdb_meta;")?;

        assert_eq!(
            query,
            QueryResolution {
                schema: mkdb_meta_schema(),
                results: vec![
                    vec![
                        Value::String("table".into()),
                        Value::String("users".into()),
                        Value::Number(2),
                        Value::String("users".into()),
                        Value::String(Parser::new(&t1).parse_statement()?.to_string())
                    ],
                    vec![
                        Value::String("table".into()),
                        Value::String("tasks".into()),
                        Value::Number(3),
                        Value::String("tasks".into()),
                        Value::String(Parser::new(&t2).parse_statement()?.to_string())
                    ],
                    vec![
                        Value::String("table".into()),
                        Value::String("products".into()),
                        Value::Number(4),
                        Value::String("products".into()),
                        Value::String(Parser::new(&t3).parse_statement()?.to_string())
                    ]
                ]
            }
        );

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

        assert_eq!(
            query,
            QueryResolution {
                schema: Schema::from(vec![
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
                ]),
                results: vec![]
            }
        );

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

        assert_eq!(
            query,
            QueryResolution {
                schema: Schema::from(vec![
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
                ]),
                results: vec![vec![
                    Value::Number(1),
                    Value::String("John Doe".into()),
                    Value::Number(18)
                ]]
            }
        );

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

        assert_eq!(
            query,
            QueryResolution {
                schema: Schema::from(vec![
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
            }
        );

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

        assert_eq!(
            query,
            QueryResolution {
                schema: Schema::from(vec![
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
            }
        );

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
                expected: DataType::Int,
                found: Value::String("String".into())
            })))
        );

        Ok(())
    }
}
