#[derive(Debug, PartialEq)]
pub(super) enum BinaryOperator {
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
pub(super) enum Expression {
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
pub(super) enum Constraint {
    PrimaryKey,
    Unique,
}

#[derive(Debug, PartialEq)]
pub(super) enum DataType {
    Int,
    Bool,
    Varchar(usize),
}

#[derive(Debug, PartialEq)]
pub(super) enum Value {
    Number(String),
    String(String),
    Bool(bool),
}

#[derive(Debug, PartialEq)]
pub(super) struct Column {
    pub name: String,
    pub data_type: DataType,
    pub constraint: Option<Constraint>,
}

#[derive(Debug, PartialEq)]
pub(super) enum Create {
    Table { name: String, columns: Vec<Column> },
    Database(String),
}

#[derive(Debug, PartialEq)]
pub(super) enum Drop {
    Table(String),
    Database(String),
}

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
