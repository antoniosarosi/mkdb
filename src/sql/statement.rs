pub(super) enum BinaryOperator {
    Lt,
    LtEq,
    Gt,
    GtEq,
    Plus,
    Minus,
    Mul,
}

pub(super) enum Expression {
    Identifier(String),

    Value(Value),

    BinaryOperation {
        left: Box<Self>,
        operator: BinaryOperator,
        right: Box<Self>,
    },
}

enum Constraint {
    PrimaryKey,
    Unique,
}

enum DataType {
    Int,
    Bool,
    Varchar(usize),
}

pub(super) enum Value {
    Number(String),
    String(String),
    Bool(bool),
}

pub(super) struct Column {
    name: String,
    data_type: DataType,
    constraints: Vec<Constraint>,
}

pub(super) enum Create {
    Table { name: String, columns: Vec<Column> },
    Database(String),
}

pub(super) enum Drop {
    Table(String),
    Database(String),
}

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
        values: Vec<Value>,
    },

    Drop(Drop),
}
