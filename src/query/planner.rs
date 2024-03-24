//! Generates [`Plan`] trees.

use std::{
    io::{Read, Seek, Write},
    path::PathBuf,
    rc::Rc,
};

use super::optimizer::generate_scan_plan;
use crate::{
    db::{Database, DatabaseContext, DbError, Schema},
    paging,
    sql::{
        analyzer,
        statement::{Column, DataType, Expression, Statement},
    },
    vm::{
        plan::{BufferedIter, Delete, Insert, Plan, Project, Sort, TupleBuffer, Update, Values},
        VmDataType,
    },
};

/// Generates a query plan that's ready to execute by the VM.
pub(crate) fn generate_plan<F: Seek + Read + Write + paging::io::FileOps>(
    statement: Statement,
    db: &mut Database<F>,
) -> Result<Plan<F>, DbError> {
    Ok(match statement {
        Statement::Insert {
            into,
            columns,
            values,
        } => {
            let metadata = db.table_metadata(&into)?;

            let source = Box::new(Plan::Values(Values {
                values,
                done: false,
            }));

            Plan::Insert(Insert {
                root: metadata.root,
                schema: metadata.schema.clone(),
                source,
                indexes: metadata.indexes.clone(),
                pager: Rc::clone(&db.pager),
            })
        }

        Statement::Select {
            columns,
            from,
            r#where,
            order_by,
        } => {
            let mut source = generate_scan_plan(&from, r#where, db)?;

            let page_size = db.pager.borrow().page_size;

            let work_dir = db.work_dir.clone();
            let metadata = db.table_metadata(&from)?;

            if !order_by.is_empty() {
                let mut sort_schema = metadata.schema.clone();

                for (i, expr) in order_by.iter().enumerate() {
                    let mut col = Column::new(&format!("sort_key_{i}"), DataType::BigInt);
                    match analyzer::analyze_expression(&metadata.schema, expr)? {
                        VmDataType::Bool => col.data_type = DataType::Bool,
                        // TODO: What should we do with strings longer than u16::MAX?
                        VmDataType::String => col.data_type = DataType::Varchar(65535),
                        // TODO: Numbers are BigInt by default. We can figure out if the
                        // expression that generated the number contains some column
                        // identifier and use that type instead. But this does the job
                        // for now.
                        _ => {}
                    }
                    sort_schema.push(col);
                }

                source = Box::new(Plan::Sort(Sort {
                    page_size,
                    work_dir: work_dir.clone(),
                    schema: metadata.schema.clone(),
                    source: BufferedIter::new(source, work_dir, sort_schema.clone(), order_by),
                    sort_schema,
                    sorted: false,
                    input_file: None,
                    output_file: None,
                    output_page: TupleBuffer::empty(),
                    input_file_path: PathBuf::new(),
                    output_file_path: PathBuf::new(),
                }));
            }

            let mut output_schema = Schema::empty();
            let mut unknown_types = Vec::new();

            for (i, expr) in columns.iter().enumerate() {
                match expr {
                    Expression::Identifier(ident) => output_schema.push(
                        metadata.schema.columns[metadata.schema.index_of(ident).unwrap()].clone(),
                    ),

                    _ => {
                        output_schema.push(Column {
                            name: expr.to_string(),    // TODO: AS alias
                            data_type: DataType::Bool, // We'll set it later
                            constraints: vec![],
                        });

                        unknown_types.push(i);
                    }
                }
            }

            // TODO: There are no expressions that can evaluate to strings as of
            // right now and we set the default to be bool. So if there's an
            // expression that evaluates to a number we'll change its type. The
            // problem is that we don't know the exact kind of number, an expression
            // with a raw value like 4294967296 should evaluate to UnsignedBigInt
            // but -65536 should probably evaluate to Int. Expressions that have
            // identifiers in them should probably evaluate to the type of the
            // identifier. Not gonna worry about this for now, this is a toy
            // database after all :)
            for i in unknown_types {
                if let VmDataType::Number =
                    analyzer::analyze_expression(&metadata.schema, &columns[i]).unwrap()
                {
                    output_schema.columns[i].data_type = DataType::BigInt;
                }
            }

            Plan::Project(Project {
                input_schema: metadata.schema.clone(),
                output_schema,
                projection: columns,
                source,
            })
        }

        Statement::Update {
            table,
            columns,
            r#where,
        } => {
            let source = generate_scan_plan(&table, r#where, db)?;
            let work_dir = db.work_dir.clone();

            let metadata = db.table_metadata(&table)?;

            Plan::Update(Update {
                assignments: columns,
                root: metadata.root,
                schema: metadata.schema.clone(),
                source: BufferedIter::new(source, work_dir, metadata.schema.clone(), vec![]),
                pager: Rc::clone(&db.pager),
            })
        }

        Statement::Delete { from, r#where } => {
            let source = generate_scan_plan(&from, r#where, db)?;
            let work_dir = db.work_dir.clone();

            let metadata = db.table_metadata(&from)?;

            Plan::Delete(Delete {
                root: metadata.root,
                schema: metadata.schema.clone(),
                indexes: metadata.indexes.clone(),
                source: BufferedIter::new(source, work_dir, metadata.schema.clone(), vec![]),
                pager: Rc::clone(&db.pager),
            })
        }

        other => todo!("unhandled statement {other}"),
    })
}
