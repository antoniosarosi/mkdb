//! Simple SQL parser and AST for our toy database.

mod token;
mod tokenizer;

pub(crate) mod analyzer;
pub(crate) mod optimizer;
pub(crate) mod parser;
pub(crate) mod prepare;
pub(crate) mod statement;

use self::{
    analyzer::analyze, optimizer::optimize, parser::Parser, prepare::prepare, statement::Statement,
};
use crate::db::{DatabaseContext, DbError};

/// Passes the given text input through all the SQL pipeline stages.
///
/// Then end result is a [`Statement`] instance ready to go through the query
/// plan generation final stage.
pub(crate) fn pipeline(input: &str, db: &mut impl DatabaseContext) -> Result<Statement, DbError> {
    let mut statement = Parser::new(input).parse_statement()?;

    analyze(&statement, db)?;
    optimize(&mut statement)?;
    prepare(&mut statement, db)?;

    Ok(statement)
}
