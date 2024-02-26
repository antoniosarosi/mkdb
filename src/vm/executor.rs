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
///
/// Page 0 holds the DB header, page 1 holds the beginning of the meta-table.
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
}

pub fn exec<I>(statement: Statement, db: &mut Database<I>) -> QueryResult {
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

            exec(&format!(
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

                    Err(SqlError::InvalidColumn(_col)) => Err(DbError::Sql(SqlError::Expected {
                        expected: ExpectedExpression::Value,
                        found: expr,
                    }))?,

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
                        return Err(DbError::Sql(SqlError::TypeError(TypeError::ExpectedType {
                            expected,
                            found: value,
                        })))
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
                            resolved_wildcards.push(Expression::Identifier(col.name.to_owned()));
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

            QueryResolution::new(results_schema, results)
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
                let mut values = deserialize_values(&row?, &schema);

                if !vm::eval_where(&schema, &values, &r#where)? {
                    continue;
                }

                for (col, expr) in &assignments {
                    let value = match **expr {
                        Expression::Value(ref v) => v.clone(),
                        _ => vm::resolve_expression(&values, &schema, expr)?,
                    };

                    let index = schema
                        .index_of(col)
                        .ok_or(SqlError::InvalidColumn(col.clone()))?;
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
}
