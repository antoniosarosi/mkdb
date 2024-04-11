//! Code that deals with simple SQL statements that don't require [`Plan`]
//! trees.
//!
//! This boils down to `CREATE` statements, which don't need plans because they
//! don't work with "tuples".

use std::{
    io::{self, Read, Seek, Write},
    rc::Rc,
};

use super::plan::{Collect, CollectConfig, Filter, Plan, SeqScan};
use crate::{
    db::{
        has_btree_key, mkdb_meta_schema, Database, DatabaseContext, DbError, IndexMetadata, RowId,
        Schema, SqlError, MKDB_META, MKDB_META_ROOT,
    },
    paging::{io::FileOps, pager::PageNumber},
    sql::{
        parser::Parser,
        statement::{Constraint, Create, Drop, Statement, Value},
    },
    storage::{free_cell, tuple, BTree, BytesCmp, Cursor, FixedSizeMemCmp},
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

            let skip_primary_key_index = if has_btree_key(&columns) { 1 } else { 0 };

            let indexes = columns
                .into_iter()
                .skip(skip_primary_key_index)
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
                schema: Schema::new(vec![
                    metadata.schema.columns[col].clone(),
                    metadata.schema.columns[0].clone(),
                ]),
                name: name.clone(),
                root,
                unique,
            };

            let mut scan = Plan::SeqScan(SeqScan {
                cursor: Cursor::new(metadata.root, 0),
                table: metadata.clone(),
                pager: Rc::clone(&db.pager),
            });

            let comparator = Box::<dyn BytesCmp>::from(&index.column.data_type);

            while let Some(mut tuple) = scan.try_next()? {
                // TODO: We have to borrow the pager and recreate the BTree on
                // every iteration because the scan plan above already borrows
                // the pager when we call .try_next(), so we can't create the
                // BTree before starting the loop.
                let mut pager = db.pager.borrow_mut();
                let mut btree = BTree::new(&mut pager, index.root, &comparator);

                let index_key = tuple.swap_remove(col);
                let primary_key = tuple.swap_remove(0);

                let entry = tuple::serialize(&index.schema.clone(), [&index_key, &primary_key]);

                btree
                    .try_insert(entry)?
                    .map_err(|_| SqlError::DuplicatedKey(index_key))?;
            }

            // Invalidate the table so that the next time it is loaded it
            // includes the new index. Alternatively we could manually insert
            // the index metadata we constructed previously here.
            db.context.invalidate(&table);
        }

        Statement::Drop(Drop::Table(name)) => {
            let comparator = db.table_metadata(MKDB_META)?.comparator()?;

            let mut plan = collect_from_mkdb_meta_where(db, &format!("table_name = '{name}'"))?;

            let schema = plan.schema().ok_or(DbError::Corrupted(format!(
                "could not obtain schema of {MKDB_META} table"
            )))?;

            while let Some(tuple) = plan.try_next()? {
                let Some(Value::Number(root)) =
                    schema.index_of("root").and_then(|index| tuple.get(index))
                else {
                    return Err(DbError::Corrupted(format!(
                        "could not read root of table {name}"
                    )));
                };

                free_btree(db, *root as PageNumber)?;

                BTree::new(&mut db.pager.borrow_mut(), MKDB_META_ROOT, comparator).remove(
                    &tuple::serialize_key(&schema.columns[0].data_type, &tuple[0]),
                )?;
            }
        }

        other => {
            return Err(DbError::Other(format!(
                "statement is not yet implemented or supported: {other}"
            )))
        }
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

/// Drops an entire BTree from disk.
fn free_btree<F: Seek + Read + Write + FileOps>(
    db: &mut Database<F>,
    root: PageNumber,
) -> io::Result<()> {
    let mut stack = vec![root];
    let mut pager = db.pager.borrow_mut();

    // Depth first search. Once we visited a page we no longer need it for
    // anythig, so we can safely drop it from disk. We reverse the children to
    // imporve sequential IO. Won't be really sequential due to how the BTree
    // is build and especially due to overflow pages but still, better than
    // going straight to the end of the file and then backwards.
    //
    // TODO: Overall IO performance can be improved by writing a little plan
    // type that returns all the pages including overflow pages one by one and
    // using the Sort plan to sort them sequentially. This won't improve the
    // "DROP TABLE" performance but it will allow the pager to build a huge
    // free list of sequential pages. Not difficult to do at all, but laziness
    // has won this time.
    while let Some(page_num) = stack.pop() {
        let page = pager.get_mut(page_num)?;
        stack.extend(page.iter_children().rev());

        // This part here hurts IO performance the most because it could
        // potentially be random IO. Maybe there's a way to free only the first
        // overflow page, since that page already links to the rest of them.
        // However the last overflwo page doesn't link to anything and we have
        // to link it to the free list somehow. Probably a circular list or
        // something like that would do.
        let mut cells = page.drain(..).collect::<Vec<_>>().into_iter();
        cells.try_for_each(|cell| free_cell(&mut pager, cell))?;

        pager.free_page(page_num)?;
    }

    Ok(())
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

/// Manual selection from meta table without parsing overhead and mutual
/// recursion.
///
/// Results are automatically collected to allow update/delete operations.
fn collect_from_mkdb_meta_where<F: Seek + Read + Write + FileOps>(
    db: &mut Database<F>,
    filter: &str,
) -> Result<Plan<F>, DbError> {
    let work_dir = db.work_dir.clone();
    let page_size = db.pager.borrow_mut().page_size;

    let table = db.table_metadata(MKDB_META)?;

    Ok(Plan::Collect(Collect::from(CollectConfig {
        work_dir,
        mem_buf_size: page_size,
        schema: table.schema.clone(),
        source: Box::new(Plan::Filter(Filter {
            filter: Parser::new(filter).parse_expression()?,
            schema: table.schema.clone(),
            source: Box::new(Plan::SeqScan(SeqScan {
                table: table.to_owned(),
                pager: Rc::clone(&db.pager),
                cursor: Cursor::new(MKDB_META_ROOT, 0),
            })),
        })),
    })))
}
