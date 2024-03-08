// Final step in the SQL pipeline before plan generation.

use std::io::{Read, Seek, Write};

use super::statement::{Expression, Statement};
use crate::{
    db::{Database, DbError},
    paging,
};

/// Takes a statement and prepares it for plan generation.
///
/// For now, the only thing this function does is getting rid of wildcards in
/// select statements. Basically, replacing `SELECT * FROM table` with
/// `SELECT a,b,c FROM table`, where `a`, `b` and `c` are all the table columns.
///
/// Additionally, we should deal with default values and auto-increment keys or
/// stuff like that here.
pub(crate) fn prepare<I: Seek + Read + Write + paging::io::Sync>(
    statement: &mut Statement,
    db: &mut Database<I>,
) -> Result<(), DbError> {
    let Statement::Select { columns, from, .. } = statement else {
        return Ok(());
    };

    if !columns.iter().any(|expr| *expr == Expression::Wildcard) {
        return Ok(());
    }

    let (schema, _) = db.table_metadata(&from)?;

    let identifiers = schema
        .columns
        .into_iter()
        .filter(|col| col.name != "row_id")
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

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{cell::RefCell, io, rc::Rc};

    use super::prepare;
    use crate::{
        db::{Database, DbError, DEFAULT_PAGE_SIZE},
        paging::{io::MemBuf, pager::Pager},
        sql::parser::Parser,
    };

    fn create_database() -> io::Result<Database<MemBuf>> {
        let mut pager = Pager::new(
            io::Cursor::new(Vec::<u8>::new()),
            DEFAULT_PAGE_SIZE,
            DEFAULT_PAGE_SIZE,
        );

        pager.init()?;

        Ok(Database::new(Rc::new(RefCell::new(pager))))
    }

    #[test]
    fn prepare_statement() -> Result<(), DbError> {
        let mut db = create_database()?;
        db.exec("CREATE TABLE test (a INT, b INT, c INT);")?;

        let sql_stmt = "SELECT a+2,   *,   b*2,   *   FROM test;";
        let prepared = "SELECT a+2, a,b,c, b*2, a,b,c FROM test;";

        let mut statement = Parser::new(sql_stmt).parse_statement()?;
        let expected = Parser::new(prepared).parse_statement()?;

        prepare(&mut statement, &mut db)?;

        assert_eq!(statement, expected);

        Ok(())
    }
}
