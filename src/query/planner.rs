//! Generates [`Plan`] trees.

use std::{
    io::{Read, Seek, Write},
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
        plan::{BufferedIter, Delete, Insert, Plan, Project, Sort, Update, Values},
        VmDataType,
    },
};

/// Returns `true` if the given `statement` needs a query [`Plan`] to be
/// executed.
pub(crate) fn needs_plan(statement: &Statement) -> bool {
    !matches!(statement, Statement::Create(_))
}

/// Generates a query plan that's ready to execute by the VM.
pub(crate) fn generate_plan<I: Seek + Read + Write + paging::io::FileOps>(
    statement: Statement,
    db: &mut Database<I>,
) -> Result<Plan<I>, DbError> {
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

            let metadata = db.table_metadata(&from)?;

            if !order_by.is_empty() {
                source = Box::new(Plan::Sort(Sort {
                    by: order_by,
                    schema: metadata.schema.clone(),
                    sorted: false,
                    source: BufferedIter::new(source),
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

            let metadata = db.table_metadata(&table)?;

            Plan::Update(Update {
                assignments: columns,
                root: metadata.root,
                schema: metadata.schema.clone(),
                source: BufferedIter::new(source),
                pager: Rc::clone(&db.pager),
            })
        }

        Statement::Delete { from, r#where } => {
            let source = generate_scan_plan(&from, r#where, db)?;
            let metadata = db.table_metadata(&from)?;

            Plan::Delete(Delete {
                root: metadata.root,
                schema: metadata.schema.clone(),
                indexes: metadata.indexes.clone(),
                source: BufferedIter::new(source),
                pager: Rc::clone(&db.pager),
            })
        }

        other => todo!("unhandled statement {other}"),
    })
}
