//! Generates a query plan.

use std::{
    io::{Read, Seek, Write},
    rc::Rc,
};

use super::plan::{Delete, Filter, Insert, Plan, Project, SeqScan, Sort, Update, Values};
use crate::{
    db::{Database, DbError, Schema},
    paging::{self, pager::PageNumber},
    sql::statement::{Expression, Statement},
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
            let source = Box::new(Plan::Values(Values::new(values)));

            Plan::Insert(Insert::new(root, Rc::clone(&db.pager), source, schema))
        }

        Statement::Select {
            columns,
            from,
            r#where,
            order_by,
        } => {
            let (schema, root) = db.table_metadata(&from)?;
            let mut source = generate_seq_scan_plan(&(schema, root), r#where, db)?;

            if !order_by.is_empty() {
                source = Box::new(Plan::Sort(Sort::new(source, order_by)));
            }

            Plan::Project(Project::new(source, columns))
        }

        Statement::Update {
            table,
            columns,
            r#where,
        } => {
            let (schema, root) = db.table_metadata(&table)?;
            let source = generate_seq_scan_plan(&(schema, root), r#where, db)?;

            Plan::Update(Update::new(root, Rc::clone(&db.pager), source, columns))
        }

        Statement::Delete { from, r#where } => {
            let (schema, root) = db.table_metadata(&from)?;
            let source = generate_seq_scan_plan(&(schema, root), r#where, db)?;

            Plan::Delete(Delete::new(root, Rc::clone(&db.pager), source))
        }

        other => todo!("unhandled statement {other}"),
    })
}

pub(crate) fn generate_seq_scan_plan<I: Seek + Read + Write + paging::io::Sync>(
    metadata: &(Schema, PageNumber),
    filter: Option<Expression>,
    db: &mut Database<I>,
) -> Result<Box<Plan<I>>, DbError> {
    let (schema, root) = metadata;

    let mut plan = Box::new(Plan::SeqScan(SeqScan::new(
        *root,
        schema.clone(),
        Rc::clone(&db.pager),
    )));

    if let Some(filter) = filter {
        plan = Box::new(Plan::Filter(Filter::new(plan, filter)));
    }

    Ok(plan)
}

#[cfg(test)]
mod tests {}
