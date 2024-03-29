//! Generates [`Plan`] trees.
//!
//! See the module level documentation of [`crate::vm::plan`] to understand
//! what exactly we're "generating" here.

use std::{
    collections::VecDeque,
    io::{Read, Seek, Write},
    path::PathBuf,
    rc::Rc,
};

use super::optimizer;
use crate::{
    db::{Database, DatabaseContext, DbError, Schema, SqlError},
    paging,
    sql::{
        analyzer,
        statement::{Column, DataType, Expression, Statement},
    },
    vm::{
        plan::{
            BufferedIter, Delete, Insert, Plan, Project, Sort, SortKeysGen, TupleBuffer,
            TuplesComparator, Update, Values,
        },
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
            let source = Box::new(Plan::Values(Values {
                values: VecDeque::from([values]),
            }));

            Plan::Insert(Insert {
                source,
                table: db.table_metadata(&into)?.clone(),
                pager: Rc::clone(&db.pager),
            })
        }

        Statement::Select {
            columns,
            from,
            r#where,
            order_by,
        } => {
            let mut source = optimizer::generate_scan_plan(&from, r#where, db)?;

            let page_size = db.pager.borrow().page_size;

            let work_dir = db.work_dir.clone();
            let table = db.table_metadata(&from)?;

            if !order_by.is_empty() {
                let mut sort_schema = table.schema.clone();
                let mut sort_keys_indexes = Vec::with_capacity(order_by.len());

                // Precompute all the sort keys indexes so that the sorter
                // doesn't waste time figuring out where the columns are.
                for (i, expr) in order_by.iter().enumerate() {
                    let index = match expr {
                        Expression::Identifier(col) => table.schema.index_of(col).unwrap(),

                        _ => {
                            let index = sort_schema.len();
                            let data_type = resolve_unknown_type(&table.schema, expr)?;
                            let col = Column::new(&format!("sort_key_{i}"), data_type);
                            sort_schema.push(col);

                            index
                        }
                    };

                    sort_keys_indexes.push(index);
                }

                // If there are no expressions that need to be evaluated for
                // sorting then just skip the sort key generation completely,
                // we already have all the sort keys we need.
                let buffered_iter_source = if sort_schema.len() > table.schema.len() {
                    Box::new(Plan::SortKeysGen(SortKeysGen {
                        source,
                        schema: table.schema.clone(),
                        gen_exprs: order_by
                            .into_iter()
                            .filter(|expr| !matches!(expr, Expression::Identifier(_)))
                            .collect(),
                    }))
                } else {
                    source
                };

                source = Box::new(Plan::Sort(Sort {
                    page_size,
                    work_dir: work_dir.clone(),
                    source: BufferedIter::new(
                        buffered_iter_source,
                        work_dir,
                        sort_schema.clone(),
                        page_size,
                    ),
                    comparator: TuplesComparator {
                        schema: table.schema.clone(),
                        sort_schema,
                        sort_keys_indexes,
                    },
                    sorted: false,
                    input_file: None,
                    output_file: None,
                    input_buffers: 4,
                    output_buffer: TupleBuffer::empty(),
                    input_file_path: PathBuf::new(),
                    output_file_path: PathBuf::new(),
                }));
            }

            let mut output_schema = Schema::empty();

            for expr in &columns {
                match expr {
                    Expression::Identifier(ident) => output_schema
                        .push(table.schema.columns[table.schema.index_of(ident).unwrap()].clone()),

                    _ => {
                        output_schema.push(Column {
                            name: expr.to_string(), // TODO: AS alias
                            data_type: resolve_unknown_type(&table.schema, expr)?,
                            constraints: vec![],
                        });
                    }
                }
            }

            Plan::Project(Project {
                input_schema: table.schema.clone(),
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
            let source = optimizer::generate_scan_plan(&table, r#where, db)?;
            let work_dir = db.work_dir.clone();

            let metadata = db.table_metadata(&table)?;

            Plan::Update(Update {
                table: metadata.clone(),
                assignments: columns,
                source: BufferedIter::new(
                    source,
                    work_dir,
                    metadata.schema.clone(),
                    db.pager.borrow().page_size,
                ),
                pager: Rc::clone(&db.pager),
            })
        }

        Statement::Delete { from, r#where } => {
            let source = optimizer::generate_scan_plan(&from, r#where, db)?;
            let work_dir = db.work_dir.clone();

            let metadata = db.table_metadata(&from)?;

            Plan::Delete(Delete {
                table: metadata.clone(),
                source: BufferedIter::new(
                    source,
                    work_dir,
                    metadata.schema.clone(),
                    db.pager.borrow().page_size,
                ),
                pager: Rc::clone(&db.pager),
            })
        }

        other => todo!("unhandled statement {other}"),
    })
}

/// Returns a concrete [`DataType`] for an expression that hasn't been executed
/// yet.
///
/// TODO: There are no expressions that can evaluate to strings as of right now
/// since we didn't implement `CONCAT()` or any other similar function, so
/// strings can only come from identifiers. The [`analyzer`] should never return
/// [`VmDataType::String`], so it doesn't matter what type we return in that
/// case.
///
/// The real problem is when expressions evaluate to numbers becase we don't
/// know the exact kind of number. An expression with a raw value like
/// 4294967296 should evaluate to [`DataType::UnsignedBigInt`] but -65536 should
/// probably evaluate to [`DataType::Int`]. Expressions that have identifiers in
/// them should probably evaluate to the type of the identifier, but what if
/// there are multiple identifiers of different integer types? Not gonna worry
/// about this for now, this is a toy database after all :)
fn resolve_unknown_type(schema: &Schema, expr: &Expression) -> Result<DataType, SqlError> {
    Ok(match expr {
        Expression::Identifier(col) => {
            let index = schema.index_of(col).unwrap();
            schema.columns[index].data_type
        }

        _ => match analyzer::analyze_expression(schema, expr)? {
            VmDataType::Bool => DataType::Bool,
            VmDataType::Number => DataType::BigInt,
            VmDataType::String => DataType::Varchar(65535),
        },
    })
}
