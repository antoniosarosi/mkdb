//! Simple SQL parser and AST for our toy database.

mod parser;
mod statement;
mod token;
mod tokenizer;

pub(crate) use parser::{Parser, ParserError};
pub(crate) use statement::{
    BinaryOperator, Column, Constraint, Create, DataType, Drop, Expression, Statement, Value,
};
