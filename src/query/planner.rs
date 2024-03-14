//! Generates a query plan.

use std::{
    io::{Read, Seek, Write},
    rc::Rc,
};

use crate::{
    db::{Database, DbError, Schema, VmDataType},
    paging::{self, pager::PageNumber},
    sql::{
        analyzer,
        statement::{Column, DataType, Expression, Statement},
    },
    storage::Cursor,
    vm::plan::{
        BufferedIter, Delete, Filter, Insert, Plan, Project, SeqScan, Sort, Update, Values,
    },
};

pub(crate) fn needs_plan(statement: &Statement) -> bool {
    !matches!(statement, Statement::Create(_))
}

pub(crate) fn generate_plan<I: Seek + Read + Write + paging::io::Sync>(
    statement: Statement,
    db: &mut Database<I>,
) -> Result<Plan<I>, DbError> {
    Ok(match statement {
        Statement::Insert {
            into,
            columns,
            values,
        } => {
            let (schema, root) = db.table_metadata(&into)?;
            let source = Box::new(Plan::Values(Values {
                values,
                done: false,
            }));

            Plan::Insert(Insert {
                pager: Rc::clone(&db.pager),
                root,
                schema,
                source,
            })
        }

        Statement::Select {
            columns,
            from,
            r#where,
            order_by,
        } => {
            let (input_schema, root) = db.table_metadata(&from)?;
            let mut source = generate_seq_scan_plan((input_schema.clone(), root), r#where, db)?;

            if !order_by.is_empty() {
                source = Box::new(Plan::Sort(Sort {
                    by: order_by,
                    schema: input_schema.clone(),
                    sorted: false,
                    source: BufferedIter::new(source),
                }));
            }

            let mut output_schema = Schema::empty();
            let mut unknown_types = Vec::new();

            for (i, expr) in columns.iter().enumerate() {
                match expr {
                    Expression::Identifier(ident) => output_schema
                        .push(input_schema.columns[input_schema.index_of(ident).unwrap()].clone()),

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
                    analyzer::analyze_expression(&input_schema, &columns[i]).unwrap()
                {
                    output_schema.columns[i].data_type = DataType::BigInt;
                }
            }

            Plan::Project(Project {
                input_schema,
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
            let (schema, root) = db.table_metadata(&table)?;
            let source = generate_seq_scan_plan((schema.clone(), root), r#where, db)?;

            Plan::Update(Update {
                assignments: columns,
                pager: Rc::clone(&db.pager),
                root,
                schema: schema.clone(),
                source: BufferedIter::new(source),
            })
        }

        Statement::Delete { from, r#where } => {
            let (schema, root) = db.table_metadata(&from)?;
            let source = generate_seq_scan_plan((schema.clone(), root), r#where, db)?;

            Plan::Delete(Delete {
                pager: Rc::clone(&db.pager),
                root,
                schema,
                source: BufferedIter::new(source),
            })
        }

        other => todo!("unhandled statement {other}"),
    })
}

pub(crate) fn generate_seq_scan_plan<I: Seek + Read + Write + paging::io::Sync>(
    metadata: (Schema, PageNumber),
    filter: Option<Expression>,
    db: &mut Database<I>,
) -> Result<Box<Plan<I>>, DbError> {
    let (schema, root) = metadata;

    let mut plan = Box::new(Plan::SeqScan(SeqScan {
        cursor: Cursor::new(root, 0),
        schema: schema.clone(),
        pager: Rc::clone(&db.pager),
    }));

    if let Some(filter) = filter {
        plan = Box::new(Plan::Filter(Filter {
            filter,
            schema,
            source: plan,
        }));
    }

    Ok(plan)
}
