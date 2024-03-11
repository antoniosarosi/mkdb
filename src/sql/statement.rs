use std::fmt::{self, Display, Write};

/// SQL statement.
#[derive(Debug, PartialEq)]
pub(crate) enum Statement {
    Create(Create),

    Select {
        columns: Vec<Expression>,
        from: String,
        r#where: Option<Expression>,
        order_by: Vec<Expression>,
    },

    Delete {
        from: String,
        r#where: Option<Expression>,
    },

    Update {
        table: String,
        columns: Vec<Assignment>,
        r#where: Option<Expression>,
    },

    Insert {
        into: String,
        columns: Vec<String>,
        values: Vec<Expression>,
    },

    Drop(Drop),
}

/// Expressions used in select, update, delete and insert statements.
#[derive(Debug, PartialEq, Clone)]
pub(crate) enum Expression {
    Identifier(String),

    Value(Value),

    Wildcard,

    BinaryOperation {
        left: Box<Self>,
        operator: BinaryOperator,
        right: Box<Self>,
    },

    UnaryOperation {
        operator: UnaryOperator,
        expr: Box<Self>,
    },

    Nested(Box<Self>),
}

/// Binary operators used in expressions.
#[derive(Debug, PartialEq, Clone, Copy)]
pub(crate) enum BinaryOperator {
    Eq,
    Neq,
    Lt,
    LtEq,
    Gt,
    GtEq,
    Plus,
    Minus,
    Mul,
    Div,
    And,
    Or,
}

/// Unary operators used in expressions.
#[derive(Debug, PartialEq, Clone, Copy)]
pub(crate) enum UnaryOperator {
    Plus,
    Minus,
}

/// SQL constraints.
#[derive(Debug, PartialEq, Clone, Copy)]
pub(crate) enum Constraint {
    PrimaryKey,
    Unique,
}

/// SQL Data types.
#[derive(Debug, PartialEq, Clone, Copy)]
pub(crate) enum DataType {
    Int,
    UnsignedInt,
    BigInt,
    UnsignedBigInt,
    Bool,
    Varchar(usize),
}

// TODO: We use i128 to store numbers since we don't know their exact type in
// expressions like "SELECT 12 + 12 FROM table". And even if we knew their exact
// type, we would still have to match over i32, u32, i64, u64... to operate on
// them. So what's faster?
// - A bunch of IF statements plus OP (+,-,*,/)
// - Using i128
// - Using some bigint library like https://docs.rs/num-bigint/
// - Using a custom number type
// It's a toy database anyway, not that anyone is gonna run into integer
// overflow issues in production :)

/// Resolved values from expressions.
#[derive(Debug, PartialEq, Clone)]
pub(crate) enum Value {
    Number(i128),
    String(String),
    Bool(bool),
}

#[derive(Debug, PartialEq, Clone)]
pub(crate) struct Assignment {
    pub identifier: String,
    pub value: Expression,
}

/// Column definitions from `INSERT` statements.
#[derive(Debug, PartialEq, Clone)]
pub(crate) struct Column {
    pub name: String,
    pub data_type: DataType,
    // TODO: Vec of constraints. Not important for now.
    pub constraints: Vec<Constraint>,
}

/// `CREATE` statement.
#[derive(Debug, PartialEq)]
pub(crate) enum Create {
    Database(String),
    Table {
        name: String,
        columns: Vec<Column>,
    },
    Index {
        name: String,
        table: String,
        column: String,
    },
}

/// `DROP` statement.
#[derive(Debug, PartialEq)]
pub(crate) enum Drop {
    Table(String),
    Database(String),
}

/// Optimized version of [`std::slice::Join`] with no intermediary [`Vec`] and
/// strings.
fn join<T: Display>(values: &[T], separator: &str) -> String {
    let mut joined = String::new();

    // TODO: What exactly can fail here? Out of memory?
    write!(joined, "{}", &values[0]).unwrap();

    for value in values[1..].iter() {
        joined.push_str(separator);
        write!(joined, "{value}").unwrap();
    }

    joined
}

impl PartialOrd for Value {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match (self, other) {
            (Value::Number(a), Value::Number(b)) => a.partial_cmp(b),
            (Value::String(a), Value::String(b)) => a.partial_cmp(b),
            (Value::Bool(a), Value::Bool(b)) => a.partial_cmp(b),
            _ => None,
        }
    }
}

impl Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Number(number) => write!(f, "{number}"),
            Value::String(string) => write!(f, "\"{string}\""),
            Value::Bool(bool) => f.write_str(if *bool { "TRUE" } else { "FALSE" }),
        }
    }
}

impl Display for DataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DataType::Int => f.write_str("INT"),
            DataType::UnsignedInt => f.write_str("INT UNSIGNED"),
            DataType::BigInt => f.write_str("BIGINT"),
            DataType::UnsignedBigInt => f.write_str("BIGINT UNSIGNED"),
            DataType::Bool => f.write_str("BOOL"),
            DataType::Varchar(max) => write!(f, "VARCHAR({max})"),
        }
    }
}

impl Display for Column {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}", self.name, self.data_type)?;

        for constraint in &self.constraints {
            f.write_char(' ')?;
            f.write_str(match constraint {
                Constraint::PrimaryKey => "PRIMARY KEY",
                Constraint::Unique => "UNIQUE",
            })?;
        }

        Ok(())
    }
}

impl Display for Assignment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} = {}", self.identifier, self.value)
    }
}

impl Display for BinaryOperator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            BinaryOperator::Eq => "=",
            BinaryOperator::Neq => "!=",
            BinaryOperator::Lt => "<",
            BinaryOperator::LtEq => "<=",
            BinaryOperator::Gt => ">",
            BinaryOperator::GtEq => ">=",
            BinaryOperator::Plus => "+",
            BinaryOperator::Minus => "-",
            BinaryOperator::Mul => "*",
            BinaryOperator::Div => "/",
            BinaryOperator::And => "AND",
            BinaryOperator::Or => "OR",
        })
    }
}

impl Display for UnaryOperator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_char(match self {
            UnaryOperator::Minus => '-',
            UnaryOperator::Plus => '+',
        })
    }
}

impl Display for Expression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Identifier(ident) => f.write_str(ident),
            Self::Value(value) => write!(f, "{value}"),
            Self::Wildcard => f.write_char('*'),
            Self::BinaryOperation {
                left,
                operator,
                right,
            } => {
                write!(f, "{left} {operator} {right}")
            }
            Self::UnaryOperation { operator, expr } => {
                write!(f, "{operator}{expr}")
            }
            Self::Nested(expr) => write!(f, "({expr})"),
        }
    }
}

impl Display for Statement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Statement::Create(create) => match create {
                Create::Table { name, columns } => {
                    write!(f, "CREATE TABLE {name} ({})", join(columns, ", "))?;
                }

                Create::Database(name) => {
                    write!(f, "CREATE DATABASE {name}")?;
                }

                Create::Index {
                    name,
                    table,
                    column,
                } => {
                    write!(f, "CREATE INDEX {name} ON {table}({column})")?;
                }
            },

            Statement::Select {
                columns,
                from,
                r#where,
                order_by,
            } => {
                write!(f, "SELECT {} FROM {from}", join(columns, ", "))?;
                if let Some(expr) = r#where {
                    write!(f, " WHERE {expr}")?;
                }
                if !order_by.is_empty() {
                    write!(f, " ORDER BY {}", join(order_by, ", "))?;
                }
            }

            Statement::Delete { from, r#where } => {
                write!(f, "DELETE FROM {from}")?;
                if let Some(expr) = r#where {
                    write!(f, " WHERE {expr}")?;
                }
            }

            Statement::Update {
                table,
                columns,
                r#where,
            } => {
                write!(f, "UPDATE {table} SET {}", join(columns, ", "))?;
                if let Some(expr) = r#where {
                    write!(f, " WHERE {expr}")?;
                }
            }

            Statement::Insert {
                into,
                columns,
                values,
            } => {
                write!(
                    f,
                    "INSERT INTO {into} ({}) VALUES ({})",
                    join(columns, ", "),
                    join(values, ", ")
                )?;
            }

            Statement::Drop(drop) => {
                match drop {
                    Drop::Table(name) => write!(f, "DROP TABLE {name}")?,
                    Drop::Database(name) => write!(f, "DROP DATABASE {name}")?,
                };
            }
        };

        f.write_char(';')
    }
}
