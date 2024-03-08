//! Simple SQL parser and AST for our toy database.

mod token;
mod tokenizer;

pub(crate) mod analyzer;
pub(crate) mod optimizer;
pub(crate) mod parser;
pub(crate) mod prepare;
pub(crate) mod statement;
