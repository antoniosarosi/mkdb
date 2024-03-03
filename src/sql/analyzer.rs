//! This module performs semantic or context dependant analysys on parsed SQL
//! statements.
//!
//! After a statement has been successfuly analyzed by [`analyze`], it should
//! execute with almost no runtime errors, except for stuff like integer
//! overflow above [`i128::MAX`] or similar edge cases.

use std::io::{Read, Seek, Write};

use crate::{
    db::{Database, DbError, GenericDataType, Schema, SqlError, TypeError},
    paging,
    sql::statement::{BinaryOperator, Constraint, Create, DataType, Expression, Statement, Value},
};

/// Analyzes the given statement and returns an error if any.
///
/// If there's no error this function does nothing else.
pub(crate) fn analyze<I: Seek + Read + Write + paging::io::Sync>(
    statement: &Statement,
    db: &mut Database<I>,
) -> Result<(), DbError> {
    match statement {
        Statement::Create(Create::Table { columns, .. }) => {
            let mut found_primary_key = false;

            for col in columns {
                if let Some(Constraint::PrimaryKey) = col.constraint {
                    if found_primary_key {
                        return Err(DbError::Sql(SqlError::MultiplePrimaryKeys));
                    }
                    found_primary_key = true;
                }
            }
        }

        Statement::Create(Create::Index { table, .. }) => {
            db.table_metadata(table)?;
        }

        Statement::Insert {
            into,
            columns,
            values,
        } => {
            let (schema, _) = db.table_metadata(into)?;

            if columns.len() != values.len() {
                return Err(DbError::Sql(SqlError::ColumnValueCountMismatch));
            }

            for col in columns {
                if schema.index_of(col).is_none() {
                    return Err(DbError::Sql(SqlError::InvalidColumn(col.clone())));
                }
            }

            // -1 because row_id
            if schema.len() - 1 != columns.len() {
                return Err(DbError::Sql(SqlError::MissingColumns));
            }

            for (expr, col) in values.iter().zip(columns) {
                analyze_assignment(&schema, col, expr, false)?;
            }
        }

        Statement::Select {
            from,
            columns,
            r#where,
            order_by,
        } => {
            let (schema, _) = db.table_metadata(from)?;

            for expr in columns {
                if expr != &Expression::Wildcard {
                    analyze_expression(&schema, expr)?;
                }
            }

            analyze_where(&schema, r#where)?;

            for expr in order_by {
                analyze_expression(&schema, expr)?;
            }
        }

        Statement::Delete { from, r#where } => {
            let (schema, _) = db.table_metadata(from)?;
            analyze_where(&schema, r#where)?;
        }

        Statement::Update {
            table,
            columns,
            r#where,
        } => {
            let (schema, _) = db.table_metadata(table)?;

            for col in columns {
                analyze_assignment(&schema, &col.identifier, &col.value, true)?;
            }

            analyze_where(&schema, r#where)?;
        }

        _ => todo!("rest of SQL statements"),
    };

    Ok(())
}

/// Makes sure that the given expression is valid and evaluates to a boolean.
fn analyze_where(schema: &Schema, r#where: &Option<Expression>) -> Result<(), DbError> {
    let Some(expr) = r#where else {
        return Ok(());
    };

    if let GenericDataType::Bool = analyze_expression(schema, expr)? {
        return Ok(());
    };

    Err(DbError::Sql(SqlError::TypeError(TypeError::ExpectedType {
        expected: GenericDataType::Bool,
        found: expr.clone(),
    })))
}

/// Makes sure that the expression will evaluate to a data type that can be
/// assigned to the given column.
fn analyze_assignment(
    schema: &Schema,
    column: &str,
    value: &Expression,
    allow_identifiers: bool,
) -> Result<(), SqlError> {
    let index = schema
        .index_of(column)
        .ok_or(SqlError::InvalidColumn(column.into()))?;

    let expected_data_type = GenericDataType::from(schema.columns[index].data_type);
    let pre_eval_data_type = if allow_identifiers {
        analyze_expression(schema, value)?
    } else {
        analyze_expression(&Schema::empty(), value)?
    };

    if expected_data_type != pre_eval_data_type {
        return Err(SqlError::TypeError(TypeError::ExpectedType {
            expected: expected_data_type,
            found: value.clone(),
        }));
    }

    Ok(())
}

/// Predetermines the type that an expression will evaluate to.
///
/// The expression resolver can also do that because it actually evaluates the
/// expression, but in the cases of statements with `WHERE` clauses it won't be
/// called until we loaded tuples into memory, which requires IO, so it's better
/// to do this check now.
///
/// If there are type errors or unknown columns not present in the given
/// schema then an error is returned.
fn analyze_expression(schema: &Schema, expr: &Expression) -> Result<GenericDataType, SqlError> {
    Ok(match expr {
        Expression::Value(value) => match value {
            Value::Bool(_) => GenericDataType::Bool,
            Value::Number(_) => GenericDataType::Number,
            Value::String(_) => GenericDataType::String,
        },

        Expression::Identifier(ident) => {
            let index = schema
                .index_of(ident)
                .ok_or(SqlError::InvalidColumn(ident.clone()))?;

            match schema.columns[index].data_type {
                DataType::Bool => GenericDataType::Bool,
                DataType::Varchar(_) => GenericDataType::String,
                _ => GenericDataType::Number,
            }
        }

        Expression::UnaryOperation { operator, expr } => match analyze_expression(schema, expr)? {
            GenericDataType::Number => GenericDataType::Number,

            _ => Err(TypeError::ExpectedType {
                expected: GenericDataType::Number,
                found: *expr.clone(),
            })?,
        },

        Expression::BinaryOperation {
            left,
            operator,
            right,
        } => {
            let left_data_type = analyze_expression(schema, left)?;
            let right_data_type = analyze_expression(schema, right)?;

            // TODO: We're lazily evaluating this because we have to clone.
            // Figure out if we can refactor this module to avoid cloning
            // for errors.
            let mismatched_types = || {
                SqlError::TypeError(TypeError::CannotApplyBinary {
                    left: *left.clone(),
                    operator: *operator,
                    right: *right.clone(),
                })
            };

            if left_data_type != right_data_type {
                return Err(mismatched_types());
            }

            match operator {
                BinaryOperator::Eq
                | BinaryOperator::Neq
                | BinaryOperator::Lt
                | BinaryOperator::LtEq
                | BinaryOperator::Gt
                | BinaryOperator::GtEq => GenericDataType::Bool,

                BinaryOperator::And | BinaryOperator::Or
                    if left_data_type == GenericDataType::Bool =>
                {
                    GenericDataType::Bool
                }

                BinaryOperator::Plus
                | BinaryOperator::Minus
                | BinaryOperator::Div
                | BinaryOperator::Mul
                    if left_data_type == GenericDataType::Number =>
                {
                    GenericDataType::Number
                }

                _ => Err(mismatched_types())?,
            }
        }

        Expression::Nested(expr) => analyze_expression(schema, expr)?,

        Expression::Wildcard => {
            unreachable!("analyze_expression() shouldn't come across wildcards")
        }
    })
}
