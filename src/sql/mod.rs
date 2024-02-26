//! Simple SQL parser and AST for our toy database.

mod parser;
mod statement;
mod token;
mod tokenizer;

pub(crate) use parser::{ParseResult, Parser, ParserError};
pub(crate) use statement::{
    Assignment, BinaryOperator, Column, Constraint, Create, DataType, Expression, Statement,
    UnaryOperator, Value,
};
