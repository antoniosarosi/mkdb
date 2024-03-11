//! Code that deals with simple SQL statements that don't require plans.

use std::io::{self, Read, Seek, Write};

use crate::{
    db::{mkdb_meta_schema, Database, DbError, RowId, MKDB_META, MKDB_META_ROOT},
    paging::{self, pager::PageNumber},
    sql::statement::{Constraint, Create, Statement, Value},
    storage::{page::Page, tuple, BTree, FixedSizeMemCmp},
};

/// Executes a SQL statement that doesn't require a query plan.
///
/// For now, this is limited to [`Create`] statements. We don't need a query
/// plan to create a table or index because we only have to insert some data
/// into a BTree and we're not returning anything. We can do that with only
/// the information provided by the [`Statement`] itself.
pub(crate) fn exec_statement<I: Seek + Read + Write + paging::io::Sync>(
    statement: Statement,
    db: &mut Database<I>,
) -> Result<(), DbError> {
    let sql = statement.to_string();

    match statement {
        Statement::Create(Create::Table { name, columns }) => {
            let root = alloc_root_page(db)?;

            let mut maybe_primary_key = None;

            for col in &columns {
                if let Some(Constraint::PrimaryKey) = col.constraint {
                    maybe_primary_key = Some(col.name.clone());
                    break;
                }
            }

            insert_into_mkdb_meta(db, vec![
                Value::String(String::from("table")),
                Value::String(name.clone()),
                Value::Number(root.into()),
                Value::String(name.clone()),
                Value::String(sql),
            ])?;

            if let Some(primary_key) = maybe_primary_key {
                exec_statement(
                    Statement::Create(Create::Index {
                        name: format!("{name}_pk_index"),
                        table: name,
                        column: primary_key,
                    }),
                    db,
                )?;
            }
        }

        Statement::Create(Create::Index { name, table, .. }) => {
            let root = alloc_root_page(db)?;

            insert_into_mkdb_meta(db, vec![
                Value::String(String::from("index")),
                Value::String(name.clone()),
                Value::Number(root.into()),
                Value::String(table),
                Value::String(sql),
            ])?;
        }

        other => unreachable!("unhandled SQL statement: {other}"),
    };

    Ok(())
}

/// Allocates a page on disk that can be used as a table root.
fn alloc_root_page<I: Seek + Read + Write + paging::io::Sync>(
    db: &mut Database<I>,
) -> io::Result<PageNumber> {
    let mut pager = db.pager.borrow_mut();
    let root = pager.alloc_page()?;
    pager.init_disk_page::<Page>(root)?;

    Ok(root)
}

/// Inserts data into the [`MKDB_META`] table.
///
/// This is the same as running:
///
/// ```no_run
/// #[rustfmt::skip]
/// db.exec(r#"
///     INSERT INTO mkdb_meta (type, name, root, table_name, sql) VALUES (
///         "table",
///         "example",
///         1,
///         "example",
///         'CREATE TABLE example (id INT PRIMARY KEY);'
///     );
/// "#);
/// ```
///
/// But we're doing it manually to avoid mutually recursive calls since
/// [`Database`] calls [`exec`] and then [`exec`] would call [`Database`] again
/// which in turn calls [`exec`] again. That actually works perfectly fine and
/// it's how the [`crate::sql::parser`] executes under the hood. Mutual
/// recursion is nice for recursive descent parsers but for this case it makes
/// it harder to debug tests when they fail because we have to figure out which
/// SQL statement is actually failing, the one we want to run or those that
/// are triggered by the VM?
///
/// Plus, if we already know exactly what we're doing we don't need to go
/// through all the SQL parsing stages and since we're not running queries we
/// don't need plans either. So this should be more efficient anyway. Still,
/// that doesn't take away from the fact that mutual recursion is awesome. See
/// here:
///
/// <https://github.com/antoniosarosi/mkdb/blob/cde9f31a7864549f64375ce4bfe69779bf33ab52/src/vm/executor.rs#L59-L74>
fn insert_into_mkdb_meta<I: Seek + Read + Write + paging::io::Sync>(
    db: &mut Database<I>,
    mut values: Vec<Value>,
) -> Result<(), DbError> {
    let mut schema = mkdb_meta_schema();
    // TODO: Avoid shifting elements.
    schema.prepend_row_id();
    values.insert(0, Value::Number(db.next_row_id(MKDB_META)?.into()));

    let mut pager = db.pager.borrow_mut();
    let mut btree = BTree::new(
        &mut pager,
        MKDB_META_ROOT,
        FixedSizeMemCmp::for_type::<RowId>(),
    );

    btree.insert(tuple::serialize_values(&schema, &values))?;

    Ok(())
}
