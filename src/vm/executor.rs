use std::{
    cmp::Ordering,
    io::{Read, Seek, Write},
};

use crate::{
    db::{
        Database, DbError, GenericDataType, Projection, QueryResult, RowId, Schema, SqlError,
        StringCmp, TypeError, MKDB_META,
    },
    paging::{
        self,
        pager::{PageNumber, Pager},
    },
    query::{self, planner::Plan},
    sql::{
        parser::Parser,
        statement::{Column, Constraint, Create, DataType, Expression, Statement, Value},
    },
    storage::{
        page::Page, tuple, BTree, BytesCmp, FixedSizeMemCmp, DEFAULT_BALANCE_SIBLINGS_PER_SIDE,
    },
    vm,
};

pub(crate) fn btree_new<I>(
    pager: &mut Pager<I>,
    root: PageNumber,
) -> BTree<'_, I, FixedSizeMemCmp> {
    BTree::new(pager, root, FixedSizeMemCmp::for_type::<RowId>())
}

fn index_btree<I, C: BytesCmp>(pager: &mut Pager<I>, root: PageNumber, cmp: C) -> BTree<'_, I, C> {
    BTree::new(pager, root, cmp)
}

pub(crate) fn exec<I: Seek + Read + Write + paging::io::Sync>(
    statement: Statement,
    db: &mut Database<I>,
) -> QueryResult {
    match statement {
        Statement::Create(Create::Table { name, columns }) => {
            let root_page = {
                let mut pager = db.pager.borrow_mut();
                let root_page = pager.alloc_page()?;
                pager.init_disk_page::<Page>(root_page)?;
                root_page
            };

            let mut maybe_primary_key = None;

            for col in &columns {
                if let Some(Constraint::PrimaryKey) = col.constraint {
                    maybe_primary_key = Some(col.name.clone());
                    break;
                }
            }

            db.exec(&format!(
                r#"
                    INSERT INTO {MKDB_META} (type, name, root, table_name, sql)
                    VALUES ("table", "{name}", {root_page}, "{name}", '{sql}');
                "#,
                sql = Statement::Create(Create::Table {
                    name: name.clone(),
                    columns,
                })
            ))?;

            if let Some(primary_key) = maybe_primary_key {
                db.exec(&format!(
                    "CREATE INDEX {name}_pk_index ON {name}({primary_key});"
                ))?;
            }

            Ok(Projection::empty())
        }

        Statement::Create(Create::Index {
            name,
            table,
            column,
        }) => {
            let root_page = {
                let mut pager = db.pager.borrow_mut();
                let root_page = pager.alloc_page()?;
                pager.init_disk_page::<Page>(root_page)?;
                root_page
            };

            db.exec(&format!(
                r#"
                    INSERT INTO {MKDB_META} (type, name, root, table_name, sql)
                    VALUES ("index", "{name}", {root_page}, "{table}", '{sql}');
                "#,
                sql = Statement::Create(Create::Index {
                    name: name.clone(),
                    table: table.clone(),
                    column
                })
            ))?;

            Ok(Projection::empty())
        }

        query => {
            let mut plan = query::generate_plan(query, db)?;

            let Some(first) = plan.next() else {
                return Ok(Projection::empty());
            };

            let mut projection = first?;

            while let Some(result) = plan.next() {
                match result {
                    Ok(mut p) => projection.results.append(&mut p.results),
                    Err(e) => Err(e)?,
                }
            }

            Ok(projection)
        }
    }
}
