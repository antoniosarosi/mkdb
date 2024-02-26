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
        self,
        pager::{PageNumber, Pager},
    },
    query::{analyzer::analyze, optimizer::optimize},
    sql::{
        BinaryOperator, Column, Constraint, Create, DataType, Expression, Parser, ParserError,
        Statement, UnaryOperator, Value,
    },
    storage::{page::Page, BTree, BytesCmp, FixedSizeMemCmp, DEFAULT_BALANCE_SIBLINGS_PER_SIDE},
    vm,
};

/// Database file default page size.
pub(crate) const DEFAULT_PAGE_SIZE: usize = 4096;

/// Name of the meta-table used to keep track of other tables.
pub(crate) const MKDB_META: &str = "mkdb_meta";

/// Root page of the meta-table.
pub(crate) const MKDB_META_ROOT: PageNumber = 0;

/// Rows are uniquely identified by an 8 byte key stored in big endian at the
/// beginning of each tuple.
type RowId = u64;

pub(crate) struct Database<I> {
    pager: Pager<I>,
    row_ids: HashMap<String, u64>,
}

#[derive(Debug, PartialEq)]
pub(crate) struct QueryResolution {
    schema: Schema,
    results: Vec<Vec<Value>>,
}

/// Generic data types without SQL details such as `UNSIGNED` or `VARCHAR(max)`.
///
/// Basically the variants of [`Value`] but without inner data.
#[derive(Debug, PartialEq, Clone, Copy)]
pub(crate) enum GenericDataType {
    Bool,
    String,
    Number,
}

impl Display for GenericDataType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Bool => "boolean",
            Self::Number => "number",
            Self::String => "string",
        })
    }
}

impl From<DataType> for GenericDataType {
    fn from(data_type: DataType) -> Self {
        match data_type {
            DataType::Varchar(_) => GenericDataType::String,
            DataType::Bool => GenericDataType::Bool,
            _ => GenericDataType::Number,
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
        expected: GenericDataType,
        found: Expression,
    },
}

impl From<TypeError> for SqlError {
    fn from(type_error: TypeError) -> Self {
        SqlError::TypeError(type_error)
    }
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
            Self::Value => "literal value",
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
    MultiplePrimaryKeys,
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

/// Table schema.
#[derive(Debug, PartialEq)]
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
            constraint: None,
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

    pub fn to_ascii_table(&self) -> String {
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
    fn new(pager: Pager<I>) -> Self {
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

        Ok(Database::new(pager))
    }
}

fn deserialize_row_id(buf: &[u8]) -> RowId {
    RowId::from_be_bytes(buf[..mem::size_of::<RowId>()].try_into().unwrap())
}

fn serialize_row_id(row_id: RowId) -> [u8; mem::size_of::<RowId>()] {
    row_id.to_be_bytes()
}

struct StringCmp {
    schema: Schema,
}

impl BytesCmp for StringCmp {
    fn bytes_cmp(&self, a: &[u8], b: &[u8]) -> Ordering {
        let a = deserialize_values(a, &self.schema);
        let b = deserialize_values(b, &self.schema);
        match (&a[0], &b[0]) {
            (Value::String(a), Value::String(b)) => a.cmp(&b),
            _ => unreachable!(),
        }
    }
}

fn serialize_values(schema: &Schema, values: &Vec<Value>) -> Vec<u8> {
    macro_rules! serialize_big_endian {
        ($num:expr, $int:ty) => {
            TryInto::<$int>::try_into(*$num).unwrap().to_be_bytes()
        };
    }

    debug_assert_eq!(
        schema.len(),
        values.len(),
        "length of schema and values must be the same"
    );

    let mut buf = Vec::new();

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
                buf.extend_from_slice(&serialize_big_endian!(num, i32));
            }

            (DataType::UnsignedInt, Value::Number(num)) => {
                buf.extend_from_slice(&serialize_big_endian!(num, u32));
            }

            (DataType::BigInt, Value::Number(num)) => {
                buf.extend_from_slice(&serialize_big_endian!(num, i64));
            }

            (DataType::UnsignedBigInt, Value::Number(num)) => {
                buf.extend_from_slice(&serialize_big_endian!(num, u64));
            }

            (DataType::Bool, Value::Bool(bool)) => buf.push(u8::from(*bool)),

            _ => unreachable!("attempt to serialize {val} into {}", col.data_type),
        }
    }

    buf
}

fn deserialize_values(buf: &[u8], schema: &Schema) -> Vec<Value> {
    let mut values = Vec::new();
    let mut index = 0;

    macro_rules! deserialize_big_endian {
        ($buf:expr, $index:expr, $int:ty) => {
            <$int>::from_be_bytes(
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
                    let len: usize =
                        u16::from_le_bytes(buf[index..index + 2].try_into().unwrap()).into();
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
                values.push(Value::Number(deserialize_big_endian!(buf, index, i32)));
                index += mem::size_of::<i32>();
            }

            DataType::UnsignedInt => {
                values.push(Value::Number(deserialize_big_endian!(buf, index, u32)));
                index += mem::size_of::<u32>();
            }

            DataType::BigInt => {
                values.push(Value::Number(deserialize_big_endian!(buf, index, i64)));
                index += mem::size_of::<i64>();
            }

            DataType::UnsignedBigInt => {
                values.push(Value::Number(deserialize_big_endian!(buf, index, u64)));
                index += mem::size_of::<u64>();
            }

            DataType::Bool => {
                values.push(Value::Bool(if buf[index] == 0 { false } else { true }));
                index += mem::size_of::<bool>();
            }
        }
    }

    values
}

impl<I: Seek + Read + Write + paging::io::Sync> Database<I> {
    fn btree(&mut self, root: PageNumber) -> BTree<'_, I, FixedSizeMemCmp> {
        BTree::new(
            &mut self.pager,
            root,
            DEFAULT_BALANCE_SIBLINGS_PER_SIDE,
            FixedSizeMemCmp::for_type::<RowId>(),
        )
    }

    fn index_btree<C: BytesCmp>(&mut self, root: PageNumber, cmp: C) -> BTree<'_, I, C> {
        BTree::new(
            &mut self.pager,
            root,
            DEFAULT_BALANCE_SIBLINGS_PER_SIDE,
            cmp,
        )
    }

    pub fn table_metadata(&mut self, table: &String) -> Result<(Schema, PageNumber), DbError> {
        if table == MKDB_META {
            let mut schema = mkdb_meta_schema();
            schema.prepend_row_id();
            return Ok((schema, MKDB_META_ROOT));
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
        let mut schema = match query.get(0, "sql") {
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

        schema.prepend_row_id();

        Ok((schema, root))
    }

    fn next_row_id(&mut self, table: &str, root: PageNumber) -> io::Result<RowId> {
        if let Some(row_id) = self.row_ids.get_mut(table) {
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

        self.row_ids.insert(String::from(table), row_id);
        Ok(row_id)
    }

    pub fn exec(&mut self, input: &str) -> QueryResult {
        let mut parser = Parser::new(input);

        let mut statements = parser.try_parse()?;

        if statements.len() > 1 {
            todo!("handle multiple statements at once");
        }

        let mut statement = statements.remove(0);

        analyze(&statement, self)?;
        optimize(&mut statement);

        // TODO: Parse and execute statements one by one.
        // TODO: SQL injections through the table name?.
        let query_resolution = match statement {
            Statement::Create(Create::Table { name, columns }) => {
                let root_page = self.pager.alloc_page()?;
                self.pager.init_disk_page::<Page>(root_page)?;

                let mut maybe_primary_key = None;

                for col in &columns {
                    if let Some(Constraint::PrimaryKey) = col.constraint {
                        if maybe_primary_key.is_some() {
                            return Err(DbError::Sql(SqlError::MultiplePrimaryKeys));
                        } else {
                            maybe_primary_key = Some(col.name.clone());
                        }
                    }
                }

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

                if let Some(primary_key) = maybe_primary_key {
                    self.exec(&format!(
                        "CREATE INDEX {name}_pk_index ON {name}({primary_key});"
                    ))?;
                }

                QueryResolution::empty()
            }

            Statement::Create(Create::Index {
                name,
                table,
                column,
            }) => {
                let root_page = self.pager.alloc_page()?;
                self.pager.init_disk_page::<Page>(root_page)?;

                self.exec(&format!(
                    r#"
                        INSERT INTO {MKDB_META} (type, name, root, table_name, sql)
                        VALUES ("index", "{name}", {root_page}, "{table}", '{sql}');
                    "#,
                    sql = Statement::Create(Create::Index {
                        name: name.clone(),
                        table: table.clone(),
                        column
                    })
                ))?;

                QueryResolution::empty()
            }

            Statement::Insert {
                into,
                columns,
                values,
            } => {
                let (schema, root) = self.table_metadata(&into)?;

                let mut resolved_values = vec![Value::Bool(false); schema.len()];

                for (col, expr) in columns.iter().zip(values) {
                    let value = match vm::resolve_expression(&Vec::new(), &Schema::empty(), &expr) {
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
                        (data_type, _) => {
                            return Err(DbError::Sql(SqlError::TypeError(
                                TypeError::ExpectedType {
                                    expected: GenericDataType::from(data_type),
                                    found: Expression::Value(value),
                                },
                            )))
                        }
                    }
                }

                let row_id = self.next_row_id(&into, root)?;
                resolved_values[0] = Value::Number(row_id.into());

                let mut btree = self.btree(root);

                btree.insert(serialize_values(&schema, &resolved_values))?;

                // Update all indexes
                let query = self.exec(&format!(
                    "SELECT root, sql FROM {MKDB_META} WHERE table_name = '{into}' AND type = 'index';",
                ))?;

                // TODO: Instead of panicking with "unreachable" in situtations
                // like this, return a "Corrupt" error or something similar.
                for i in 0..query.results.len() {
                    let root = match query.get(i, "root") {
                        Some(Value::Number(root)) => *root as u32,
                        _ => unreachable!(),
                    };

                    let sql = match query.get(i, "sql") {
                        Some(Value::String(sql)) => Parser::new(sql).parse_statement()?,
                        _ => unreachable!(),
                    };

                    let Statement::Create(Create::Index {
                        name,
                        table,
                        column,
                    }) = sql
                    else {
                        unreachable!();
                    };

                    let col_idx = schema.index_of(&column).unwrap();

                    let key = resolved_values.get(col_idx).unwrap().clone();

                    let tuple = vec![key, Value::Number(row_id.into())];

                    let index_schema = Schema::new(vec![
                        schema.columns[col_idx].clone(),
                        Column {
                            name: "row_id".into(),
                            data_type: DataType::UnsignedInt,
                            constraint: None,
                        },
                    ]);

                    match &index_schema.columns[0].data_type {
                        DataType::Varchar(_) => {
                            let mut btree = self.index_btree(
                                root,
                                StringCmp {
                                    schema: Schema::new(vec![index_schema.columns[0].clone()]),
                                },
                            );

                            btree.insert(serialize_values(&index_schema, &tuple))?;
                        }
                        DataType::Int | DataType::UnsignedInt => {
                            let mut btree = self.index_btree(root, FixedSizeMemCmp(4));
                            btree.insert(serialize_values(&index_schema, &tuple))?;
                        }

                        DataType::BigInt | DataType::UnsignedBigInt => {
                            let mut btree = self.index_btree(root, FixedSizeMemCmp(8));
                            btree.insert(serialize_values(&index_schema, &tuple))?;
                        }
                        _ => unreachable!(),
                    }
                }

                QueryResolution::empty()
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
                            for col in &schema.columns[1..] {
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
                    let values = deserialize_values(&row?, &schema);

                    if !vm::eval_where(&schema, &values, &r#where)? {
                        continue;
                    }

                    let mut result = Vec::new();

                    for expr in &columns {
                        result.push(vm::resolve_expression(&values, &schema, expr)?);
                    }

                    let mut order_by_vals = Vec::new();

                    for expr in &order_by {
                        order_by_vals.push(vm::resolve_expression(&values, &schema, expr)?);
                    }

                    results.push((result, order_by_vals));
                }

                // We already set the default of unknown types as bools, if
                // it's a number then change it to BigInt. We don't support any
                // expressions that produce strings. And we don't use the types
                // of results for anything now anyway.
                if !results.is_empty() {
                    for i in unknown_types {
                        if let Value::Number(_) = &results[0].0[i] {
                            results_schema.columns[i].data_type = DataType::BigInt;
                        }
                    }
                }

                // TODO: Order by can contain column that we didn't select.
                if !order_by.is_empty() {
                    results.sort_by(|(_, a), (_, b)| {
                        for (a, b) in a.iter().zip(b) {
                            let cmp = match (a, b) {
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

                QueryResolution::new(results_schema, results.into_iter().map(|r| r.0).collect())
            }

            Statement::Delete { from, r#where } => {
                let (schema, root) = self.table_metadata(&from)?;

                let mut btree = self.btree(root);

                // TODO: Use some cursor or something to delete as we traverse the tree.
                let mut row_ids = Vec::new();

                for row in btree.iter() {
                    // TODO: Deserialize only needed values instead of all and the cloning...
                    let values = deserialize_values(&row?, &schema);

                    if !vm::eval_where(&schema, &values, &r#where)? {
                        continue;
                    }

                    match values[0] {
                        Value::Number(row_id) => row_ids.push(row_id as RowId),
                        _ => unreachable!(),
                    };
                }

                // TODO: Second mutable borrow occurs here?
                let mut btree = self.btree(root);

                for row_id in row_ids {
                    btree.remove(&serialize_row_id(row_id))?;
                }

                QueryResolution::empty()
            }

            Statement::Update {
                table,
                columns,
                r#where,
            } => {
                let (schema, root) = self.table_metadata(&table)?;

                let mut btree = self.btree(root);

                let mut updates = Vec::new();

                for row in btree.iter() {
                    // TODO: Deserialize only needed values instead of all and then cloning...
                    let mut values = deserialize_values(&row?, &schema);

                    if !vm::eval_where(&schema, &values, &r#where)? {
                        continue;
                    }

                    for assignment in &columns {
                        let value = vm::resolve_expression(&values, &schema, &assignment.value)?;
                        let index = schema
                            .index_of(&assignment.identifier)
                            .ok_or(SqlError::InvalidColumn(assignment.identifier.clone()))?;

                        values[index] = value;
                        updates.push(serialize_values(&schema, &values));
                    }
                }

                // TODO: Second mutable borrow occurs here?
                let mut btree = self.btree(root);

                for update in updates {
                    btree.insert(update)?;
                }

                QueryResolution::empty()
            }

            _ => todo!("rest of SQL statements"),
        };

        // TODO: Transactions.
        self.pager.write_dirty_pages()?;
        self.pager.flush()?;
        self.pager.sync()?;

        Ok(query_resolution)
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, io};

    use super::{Database, DbError, DEFAULT_PAGE_SIZE};
    use crate::{
        db::{mkdb_meta_schema, GenericDataType, QueryResolution, Schema, SqlError, TypeError},
        paging::{io::MemBuf, pager::Pager},
        sql::{self, Column, Constraint, DataType, Expression, Parser, Value},
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

    fn init_database() -> io::Result<Database<MemBuf>> {
        let mut pager = Pager::new(
            io::Cursor::new(Vec::<u8>::new()),
            DEFAULT_PAGE_SIZE,
            DEFAULT_PAGE_SIZE,
        );

        pager.init()?;

        Ok(Database::new(pager))
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
                vec![
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
                            Parser::new("CREATE INDEX users_pk_index ON users(id);")
                                .parse_statement()?
                                .to_string()
                        )
                    ]
                ]
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
                        name: "price / 10".into(),
                        data_type: DataType::BigInt,
                        constraint: None
                    },
                    Column {
                        name: "discount * 100".into(),
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
        let i1 = "CREATE INDEX users_pk_index ON users(id);";

        let t2 = "CREATE TABLE tasks (id INT PRIMARY KEY, title VARCHAR(255), description VARCHAR(255));";
        let i2 = "CREATE INDEX tasks_pk_index ON tasks(id);";

        let t3 = "CREATE TABLE products (id INT PRIMARY KEY, price INT);";
        let i3 = "CREATE INDEX products_pk_index ON products(id);";

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
                        Value::Number(1),
                        Value::String("users".into()),
                        Value::String(Parser::new(&t1).parse_statement()?.to_string())
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
                        Value::String(Parser::new(&t2).parse_statement()?.to_string())
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
                        Value::String(Parser::new(&t3).parse_statement()?.to_string())
                    ],
                    vec![
                        Value::String("index".into()),
                        Value::String("products_pk_index".into()),
                        Value::Number(6),
                        Value::String("products".into()),
                        Value::String(Parser::new(i3).parse_statement()?.to_string())
                    ],
                ]
            }
        );

        Ok(())
    }

    /// Used mostly to test the possibilities of a BTree rooted at page zero,
    /// since that's a special case.
    #[test]
    fn create_many_tables() -> Result<(), DbError> {
        let mut db = init_database()?;

        let mut expected = Vec::new();

        let schema = "(id INT PRIMARY KEY, title VARCHAR(255), description VARCHAR(255))";

        for i in 1..=100 {
            let table_name = format!("table_{i:03}");
            let index_name = format!("{table_name}_pk_index");

            let table_sql = format!("CREATE TABLE {table_name} {schema};");
            let index_sql = format!("CREATE INDEX {index_name} ON {table_name}(id);");

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
                expected: GenericDataType::Number,
                found: Expression::Value(Value::String("String".into()))
            })))
        );

        Ok(())
    }
}
