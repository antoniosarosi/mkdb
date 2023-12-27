enum BinaryOperator {
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

enum Value {
    Number(String),
    String(String),
    Bool(bool),
}

struct Column {
    name: String,
    data_type: DataType,
    constraints: Vec<Constraint>,
}

enum Create {
    Table { name: String, columns: Vec<Column> },
    Database { name: String },
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
        r#where: Option<Expression>,
    },

    Insert {
        into: String,
        columns: Vec<String>,
        values: Vec<Value>,
    },
}
