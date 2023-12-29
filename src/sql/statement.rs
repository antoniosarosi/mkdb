/// SQL statement.
#[derive(Debug, PartialEq)]
pub(crate) enum Statement {
    Create(Create),

    Select {
        columns: Vec<Expression>,
        from: String,
        r#where: Option<Expression>,
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

#[derive(Debug, PartialEq)]
pub(crate) enum Value {
    Number(String),
    String(String),
    Bool(bool),
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
