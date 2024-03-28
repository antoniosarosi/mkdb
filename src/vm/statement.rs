//! Code that deals with simple SQL statements that don't require [`Plan`]
//! trees.
//!
//! This boils down to `CREATE` statements, which don't need plans because they
//! don't work with "tuples".

use std::{
    io::{self, Read, Seek, Write},
    rc::Rc,
};

use super::plan::{Plan, SeqScan};
use crate::{
    db::{
        mkdb_meta_schema, Database, DatabaseContext, DbError, IndexMetadata, RowId, SqlError,
        MKDB_META, MKDB_META_ROOT,
    },
    paging::{io::FileOps, pager::PageNumber},
    sql::statement::{Constraint, Create, Statement, Value},
    storage::{tuple, BTree, BytesCmp, Cursor, FixedSizeMemCmp},
};

/// Executes a SQL statement that doesn't require a query plan.
///
/// For now, this is limited to [`Create`] statements. We don't need a query
/// plan to create a table or index because we only have to insert some data
/// into a BTree and we're not returning anything. We can do that with only
/// the information provided by the [`Statement`] itself.
pub(crate) fn exec<F: Seek + Read + Write + FileOps>(
    statement: Statement,
    db: &mut Database<F>,
) -> Result<(), DbError> {
    let sql = statement.to_string();

    match statement {
        Statement::Create(Create::Table { name, columns }) => {
            let root = alloc_root_page(db)?;

            insert_into_mkdb_meta(db, vec![
                Value::String(String::from("table")),
                Value::String(name.clone()),
                Value::Number(root.into()),
                Value::String(name.clone()),
                Value::String(sql),
            ])?;

            let indexes = columns
                .into_iter()
                .filter(|col| !col.constraints.is_empty())
                .flat_map(|col| {
                    let table_name = name.clone();
                    col.constraints.into_iter().map(move |constraint| {
                        let index_name = match constraint {
                            Constraint::PrimaryKey => format!("{table_name}_pk_index"),
                            Constraint::Unique => format!("{table_name}_{}_uq_index", &col.name),
                        };

                        Create::Index {
                            name: index_name,
                            table: table_name.clone(),
                            column: col.name.clone(),
                            unique: true,
                        }
                    })
                });

            for create_index in indexes {
                exec(Statement::Create(create_index), db)?;
            }
        }

        Statement::Create(Create::Index {
            name,
            table,
            column,
            unique,
        }) => {
            if !unique {
                return Err(DbError::Sql(SqlError::Other(
                    "only unique indexes are supported".into(),
                )));
            }

            // Allocate the root page and add the new entry to the meta table.
            let root = alloc_root_page(db)?;

            insert_into_mkdb_meta(db, vec![
                Value::String(String::from("index")),
                Value::String(name.clone()),
                Value::Number(root.into()),
                Value::String(table.clone()),
                Value::String(sql),
            ])?;

            // Now build up the index.
            let metadata = db.table_metadata(&table)?;

            let col = metadata
                .schema
                .index_of(&column)
                .ok_or(SqlError::InvalidColumn(column))?;

            let index = IndexMetadata {
                column: metadata.schema.columns[col].clone(),
                name: name.clone(),
                root,
                unique,
            };

            let mut scan = Plan::SeqScan(SeqScan {
                cursor: Cursor::new(metadata.root, 0),
                table: metadata.clone(),
                pager: Rc::clone(&db.pager),
            });

            while let Some(mut tuple) = scan.try_next()? {
                let mut pager = db.pager.borrow_mut();

                // TODO: This allocates BytesCmp on every iteration. Can't put
                // it outside of the loop because of the borrow_mut() call. The
                // scan plan also borrows the pager with a mutable reference.
                let mut btree = BTree::new(
                    &mut pager,
                    index.root,
                    Box::<dyn BytesCmp>::from(&index.column.data_type),
                );

                let key = tuple.swap_remove(col);
                let row_id = tuple.swap_remove(0);

                let entry = tuple::serialize(&index.schema(), &[key.clone(), row_id]);

                btree
                    .try_insert(entry)?
                    .map_err(|_| SqlError::DuplicatedKey(key))?;
            }

            // Invalidate the table so that the next time it is loaded it
            // includes the new index. Alternatively we could manually insert
            // the index metadata we constructed previously here.
            db.context.invalidate(&table);
        }

        other => unreachable!("unhandled SQL statement: {other}"),
    };

    Ok(())
}

/// Allocates a page on disk that can be used as a table root.
fn alloc_root_page<F: Seek + Read + Write + FileOps>(
    db: &mut Database<F>,
) -> io::Result<PageNumber> {
    let mut pager = db.pager.borrow_mut();
    let root = pager.alloc_disk_page()?;

    Ok(root)
}

/// Inserts data into the [`MKDB_META`] table.
///
/// This is the same as running:
///
/// ```ignore
/// #[rustfmt::skip]
/// db.exec(r#"
///     INSERT INTO mkdb_meta (type, name, root, table_name, sql)
///     VALUES ("table", "example", 1, "example", 'CREATE TABLE example (id INT);');
/// "#);
/// ```
///
/// But we're doing it manually to avoid mutually recursive calls since
/// [`Database`] calls [`exec`] and then [`exec`] would call [`Database`] again
/// which in turn calls [`exec`] again. That actually works perfectly fine and
/// it's how the [`crate::sql::parser`] executes under the hood. Mutual
/// recursion is nice for recursive descent parsers but for this case it makes
/// it harder to debug tests when they fail because we have to figure out which
/// SQL statement is actually failing, the one we want to run or those that are
/// triggered by the VM?
///
/// Plus, if we already know exactly what we're doing we don't need to go
/// through all the SQL parsing stages and since we're not running queries we
/// don't need plans either. So this should be more efficient anyway. Still,
/// that doesn't take away from the fact that mutual recursion is awesome. See
/// here:
///
/// <https://github.com/antoniosarosi/mkdb/blob/cde9f31a7864549f64375ce4bfe69779bf33ab52/src/vm/executor.rs#L59-L74>
fn insert_into_mkdb_meta<F: Seek + Read + Write + FileOps>(
    db: &mut Database<F>,
    mut values: Vec<Value>,
) -> Result<(), DbError> {
    let mut schema = mkdb_meta_schema();
    // TODO: Avoid shifting elements.
    schema.prepend_row_id();
    values.insert(
        0,
        Value::Number(db.table_metadata(MKDB_META)?.next_row_id().into()),
    );

    let mut pager = db.pager.borrow_mut();
    let mut btree = BTree::new(
        &mut pager,
        MKDB_META_ROOT,
        FixedSizeMemCmp::for_type::<RowId>(),
    );

    btree.insert(tuple::serialize(&schema, &values))?;

    Ok(())
}
