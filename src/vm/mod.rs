//! This module contains code that interprets SQL [`Statement`] instances.
//!
//! TODO: we should make a real "virtual machine" like the one in
//! [SQLite 2](https://www.sqlite.org/vdbe.html) or an executor with JIT and
//! stuff like Postgres or something similar instead of interpreting the raw
//! [`Statement`] trees. But this is good enough for now.

// mod analyzer;
// mod executor;
// mod expression;
// mod optimizer;

use std::mem;

use crate::{
    db::{GenericDataType, Schema, SqlError, TypeError},
    sql::{BinaryOperator, DataType, Expression, UnaryOperator, Value},
};

/// Reduces an [`Expression`] instance to a concrete [`Value`] if possible.
///
/// If the expression cannot be resolved then this function returns a
/// [`SqlError`] variant.
pub(crate) fn resolve_expression(
    tuple: &Vec<Value>,
    schema: &Schema,
    expr: &Expression,
) -> Result<Value, SqlError> {
    match expr {
        Expression::Value(value) => Ok(value.clone()),

        Expression::Identifier(ident) => match schema.index_of(&ident) {
            Some(index) => Ok(tuple[index].clone()),
            None => Err(SqlError::InvalidColumn(ident.clone())),
        },

        Expression::UnaryOperation { operator, expr } => {
            match resolve_expression(tuple, schema, expr)? {
                Value::Number(mut num) => {
                    if let UnaryOperator::Minus = operator {
                        num = -num;
                    }

                    Ok(Value::Number(num))
                }

                value => Err(SqlError::TypeError(TypeError::CannotApplyUnary {
                    operator: *operator,
                    value,
                })),
            }
        }

        Expression::BinaryOperation {
            left,
            operator,
            right,
        } => {
            let left = resolve_expression(tuple, schema, &left)?;
            let right = resolve_expression(tuple, schema, &right)?;

            let mismatched_types = || {
                SqlError::TypeError(TypeError::CannotApplyBinary {
                    left: Expression::Value(left.clone()),
                    operator: *operator,
                    right: Expression::Value(right.clone()),
                })
            };

            if mem::discriminant(&left) != mem::discriminant(&right) {
                return Err(mismatched_types());
            }

            Ok(match operator {
                BinaryOperator::Eq => Value::Bool(left == right),
                BinaryOperator::Neq => Value::Bool(left != right),
                BinaryOperator::Lt => Value::Bool(left < right),
                BinaryOperator::LtEq => Value::Bool(left <= right),
                BinaryOperator::Gt => Value::Bool(left > right),
                BinaryOperator::GtEq => Value::Bool(left >= right),

                logical @ (BinaryOperator::And | BinaryOperator::Or) => {
                    let (Value::Bool(left), Value::Bool(right)) = (&left, &right) else {
                        return Err(mismatched_types());
                    };

                    match logical {
                        BinaryOperator::And => Value::Bool(*left && *right),
                        BinaryOperator::Or => Value::Bool(*left || *right),
                        _ => unreachable!(),
                    }
                }

                arithmetic => {
                    let (Value::Number(left), Value::Number(right)) = (&left, &right) else {
                        return Err(mismatched_types());
                    };

                    Value::Number(match arithmetic {
                        BinaryOperator::Plus => left + right,
                        BinaryOperator::Minus => left - right,
                        BinaryOperator::Mul => left * right,
                        BinaryOperator::Div => left / right,
                        _ => unreachable!(),
                    })
                }
            })
        }

        Expression::Nested(expr) => resolve_expression(tuple, schema, expr),

        Expression::Wildcard => {
            unreachable!("wildcards should be resolved into identifiers at this point")
        }
    }
}

/// Returns `true` if the where [`Expression`] applied to the given tuple
/// evaluates to true.
pub(crate) fn eval_where(
    schema: &Schema,
    tuple: &Vec<Value>,
    r#where: &Option<Expression>,
) -> Result<bool, SqlError> {
    let Some(expr) = r#where else {
        return Ok(true);
    };

    match resolve_expression(&tuple, &schema, &expr)? {
        Value::Bool(b) => Ok(b),

        other => Err(SqlError::TypeError(TypeError::ExpectedType {
            expected: GenericDataType::Bool,
            found: Expression::Value(other),
        })),
    }
}
