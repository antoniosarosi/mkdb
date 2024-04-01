// Final step in the SQL pipeline before plan generation.

use super::statement::{Expression, Statement, Value};
use crate::db::{DatabaseContext, DbError, ROW_ID_COL};

/// Takes a statement and prepares it for plan generation.
///
/// For now, this function only does two things:
///
/// 1. Gets rid of wildcards in select statements. Basically, replaces
/// [`Expression::Wildcard`] instances with a list of [`Expression::Identifier`]
/// variants that matches the table schema:
///
/// ```sql
/// -- Table Definition
/// CREATE TABLE users (id INT, name VARCHAR(255), age INT UNSIGNED);
///
/// -- Select Statement
/// SELECT * FROM users;
///
/// -- Prepared Statement
/// SELECT id, name, age FROM users;
/// ```
///
/// 2. Reorders values in insert statements so that they match the table schema.
/// Something like this:
///
/// ```sql
/// -- Table Definition
/// CREATE TABLE users (id INT, name VARCHAR(255), age INT UNSIGNED);
///
/// -- Insert Statement
/// INSERT INTO users (age, id, name) VALUES (20, 1, "John Doe");
///
/// -- Prepared Statement
/// INSERT INTO users (id, name, age) VALUES (1, "John Doe", 20);
/// ```
///
/// Also prepends the "row_id" column and value in the insert statement. Not
/// sure if we should do that now or wait until we execute the plan.
///
/// Additionally, we should deal with default values and auto-increment keys or
/// stuff like that here.
pub(crate) fn prepare(
    statement: &mut Statement,
    ctx: &mut impl DatabaseContext,
) -> Result<(), DbError> {
    match statement {
        Statement::Select { columns, from, .. }
            if columns.iter().any(|expr| *expr == Expression::Wildcard) =>
        {
            let metadata = ctx.table_metadata(from)?;

            let identifiers = metadata
                .schema
                .columns
                .iter()
                .filter(|&col| col.name != ROW_ID_COL)
                .cloned()
                .map(|col| Expression::Identifier(col.name))
                .collect::<Vec<Expression>>();

            let mut resolved_wildcards = Vec::new();

            for expr in columns.drain(..) {
                if expr == Expression::Wildcard {
                    resolved_wildcards.extend(identifiers.iter().cloned());
                } else {
                    resolved_wildcards.push(expr);
                }
            }

            *columns = resolved_wildcards;
        }

        Statement::Insert {
            into,
            columns,
            values,
        } => {
            let metadata = ctx.table_metadata(into)?;

            if metadata.schema.columns[0].name == ROW_ID_COL {
                let row_id = metadata.next_row_id();
                columns.insert(0, ROW_ID_COL.into());
                values.insert(0, Expression::Value(Value::Number(row_id.into())));
            }

            for current_index in 0..metadata.schema.len() {
                let sorted_index = metadata.schema.index_of(&columns[current_index]).unwrap();
                columns.swap(current_index, sorted_index);
                values.swap(current_index, sorted_index);
            }
        }

        Statement::Explain(inner) => {
            prepare(&mut *inner, ctx)?;
        }

        _ => {} // Nothing to do here.
    };

    Ok(())
}

#[cfg(test)]
mod tests {

    use super::prepare;
    use crate::{
        db::{Context, DbError},
        sql::parser::Parser,
    };

    struct Prep<'t> {
        setup: &'t [&'t str],
        raw_stmt: &'t str,
        prepared: &'t str,
    }

    fn assert_prep(prep: Prep) -> Result<(), DbError> {
        let mut ctx = Context::try_from(prep.setup)?;

        let mut statement = Parser::new(prep.raw_stmt).parse_statement()?;
        let expected = Parser::new(prep.prepared).parse_statement()?;

        prepare(&mut statement, &mut ctx)?;

        assert_eq!(statement, expected);

        Ok(())
    }

    #[test]
    fn prepare_select_statement() -> Result<(), DbError> {
        assert_prep(Prep {
            setup: &["CREATE TABLE test (a INT, b INT, c INT);"],
            raw_stmt: "SELECT a+2,   *,   b*2,   *   FROM test;",
            prepared: "SELECT a+2, a,b,c, b*2, a,b,c FROM test;",
        })
    }

    #[test]
    fn prepare_insert_statement() -> Result<(), DbError> {
        assert_prep(Prep {
            setup: &["CREATE TABLE users (id INT, name VARCHAR(255), age INT UNSIGNED, email VARCHAR(255));"],
            raw_stmt: "INSERT INTO users(email, id, age, name) VALUES ('john@mail.com', 1, 20, 'John Doe');",
            prepared: "INSERT INTO users(row_id, id, name, age, email) VALUES (1, 1, 'John Doe', 20, 'john@mail.com');"
        })
    }
}
