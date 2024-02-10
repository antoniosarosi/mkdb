use std::fmt::{self, Display};

/// SQL statement.
#[derive(Debug, PartialEq)]
pub(crate) enum Statement {
    Create(Create),

    Select {
        columns: Vec<Expression>,
        from: String,
        r#where: Option<Expression>,
        order_by: Vec<String>,
    },

    Delete {
        from: String,
        r#where: Option<Expression>,
    },

    Update {
        table: String,
        columns: Vec<Expression>,
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
#[derive(Debug, PartialEq)]
pub(crate) enum Expression {
    Identifier(String),

    Value(Value),

    Wildcard,

    BinaryOperation {
        left: Box<Self>,
        operator: BinaryOperator,
        right: Box<Self>,
    },
}

#[derive(Debug, PartialEq)]
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

#[derive(Debug, PartialEq)]
pub(crate) enum Constraint {
    PrimaryKey,
    Unique,
}

#[derive(Debug, PartialEq)]
pub(crate) enum DataType {
    Int,
    Bool,
    Varchar(usize),
}

#[derive(Debug, PartialEq, Clone)]
pub(crate) enum Value {
    Number(String),
    String(String),
    Bool(bool),
}

impl Value {
    pub fn is_truthy(&self) -> bool {
        match self {
            Self::Bool(b) => *b,
            Self::Number(n) => n.parse::<u32>().unwrap() != 0,
            Self::String(s) => !s.is_empty(),
        }
    }
}

impl PartialOrd for Value {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match (self, other) {
            (Value::Number(a), Value::Number(b)) => {
                // TODO: Other integer types
                a.parse::<u32>().unwrap().partial_cmp(&b.parse().unwrap())
            }
            (Value::String(a), Value::String(b)) => a.partial_cmp(b),
            (Value::Bool(a), Value::Bool(b)) => a.partial_cmp(b),
            _ => None, // TODO: Return error instead.
        }
    }
}

#[derive(Debug, PartialEq)]
pub(crate) struct Column {
    pub name: String,
    pub data_type: DataType,
    /// TODO: Vec of constraints. Not important for now.
    pub constraint: Option<Constraint>,
}

#[derive(Debug, PartialEq)]
pub(crate) enum Create {
    Table { name: String, columns: Vec<Column> },
    Database(String),
}

#[derive(Debug, PartialEq)]
pub(crate) enum Drop {
    Table(String),
    Database(String),
}

fn join<T: ToString>(values: &Vec<T>, separator: &str) -> String {
    values
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(separator)
}

impl Display for DataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DataType::Int => f.write_str("INT"),
            DataType::Bool => f.write_str("BOOL"),
            DataType::Varchar(max) => write!(f, "VARCHAR({max})"),
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

impl Display for Column {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}", self.name, self.data_type);

        if let Some(constraint) = &self.constraint {
            f.write_str(" ");
            f.write_str(match constraint {
                Constraint::PrimaryKey => "PRIMARY KEY",
                Constraint::Unique => "UNIQUE",
            });
        }

        Ok(())
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

impl Display for Expression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expression::Identifier(ident) => f.write_str(ident),
            Expression::Value(value) => write!(f, "{value}"),
            Expression::Wildcard => f.write_str("*"),
            Expression::BinaryOperation {
                left,
                operator,
                right,
            } => {
                write!(f, "({left}) {operator} ({right})")
            }
        }
    }
}

impl Display for Statement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Statement::Create(create) => match create {
                Create::Table { name, columns } => {
                    write!(f, "CREATE TABLE {name} ({})", join(columns, ","));
                }

                Create::Database(name) => {
                    write!(f, "CREATE DATABASE {name}");
                }
            },

            Statement::Select {
                columns,
                from,
                r#where,
                order_by,
            } => {
                write!(f, "SELECT {} FROM {from}", join(columns, ","));
                if let Some(expr) = r#where {
                    write!(f, " WHERE {expr}");
                }
                if !order_by.is_empty() {
                    write!(f, " ORDER BY {}", join(order_by, ","));
                }
            }

            Statement::Delete { from, r#where } => {
                write!(f, "DELETE FROM {from}");
                if let Some(expr) = r#where {
                    write!(f, " WHERE {expr}");
                }
            }

            Statement::Update {
                table,
                columns,
                r#where,
            } => {
                write!(f, "UPDATE {table} SET {}", join(columns, ","));
                if let Some(expr) = r#where {
                    write!(f, " WHERE {expr}");
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
                    join(columns, ","),
                    join(values, ",")
                );
            }

            Statement::Drop(drop) => {
                match drop {
                    Drop::Table(name) => write!(f, "DROP TABLE {name}"),
                    Drop::Database(name) => write!(f, "DROP DATABASE {name}"),
                };
            }
        };

        f.write_str(";")
    }
}
