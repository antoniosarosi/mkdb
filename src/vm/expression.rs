//! Code that executes [`Expression`] trees and resolves them into [`Value`].

use std::{fmt::Display, mem};

use crate::{
    db::{Schema, SqlError},
    sql::statement::{BinaryOperator, DataType, Expression, UnaryOperator, Value},
};

/// Generic data types used at runtime by [`crate::vm`] without SQL details
/// such as `UNSIGNED` or `VARCHAR(max)`.
///
/// Basically the variants of [`Value`] but without inner data.
#[derive(Debug, PartialEq, Clone, Copy)]
pub(crate) enum VmDataType {
    Bool,
    String,
    Number,
}

impl Display for VmDataType {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str(match self {
            Self::Bool => "boolean",
            Self::Number => "number",
            Self::String => "string",
        })
    }
}

impl From<DataType> for VmDataType {
    fn from(data_type: DataType) -> Self {
        match data_type {
            DataType::Varchar(_) => VmDataType::String,
            DataType::Bool => VmDataType::Bool,
            _ => VmDataType::Number,
        }
    }
}

/// Errors that can only be thrown by the VM itself.
#[derive(Debug, PartialEq)]
pub(crate) enum VmError {
    DivisionByZero(i128, i128),
}

impl Display for VmError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::DivisionByZero(left, right) => write!(f, "division by zero: {left} / {right}"),
        }
    }
}

/// SQL type errors.
///
/// This is not part of [`VmError`] because the [`crate::sql::analyzer`] can
/// also catch type errors without actually executing the expressions like we
/// do here.
#[derive(Debug, PartialEq)]
pub(crate) enum TypeError {
    CannotApplyUnary {
        operator: UnaryOperator,
        value: Value,
    },
    CannotApplyBinary {
        left: Expression,
        operator: BinaryOperator,
        right: Expression,
    },
    ExpectedType {
        expected: VmDataType,
        found: Expression,
    },
}

impl Display for TypeError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            TypeError::CannotApplyUnary { operator, value } => {
                write!(f, "cannot apply unary operator '{operator}' to {value}")
            }

            TypeError::CannotApplyBinary {
                left,
                operator,
                right,
            } => write!(
                f,
                "cannot binary operator '{operator}' to {left} and {right}"
            ),

            TypeError::ExpectedType { expected, found } => {
                write!(
                    f,
                    "expected type {expected} but expression resolved to {found} which is not {expected}"
                )
            }
        }
    }
}

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
        Expression::Value(value) => Ok(value.clone()), // TODO: Avoid cloning

        Expression::Identifier(ident) => match schema.index_of(ident) {
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
            let left = resolve_expression(tuple, schema, left)?;
            let right = resolve_expression(tuple, schema, right)?;

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

                    if arithmetic == &BinaryOperator::Div && *right == 0 {
                        return Err(VmError::DivisionByZero(*left, *right).into());
                    }

                    Value::Number(match arithmetic {
                        BinaryOperator::Plus => left + right,
                        BinaryOperator::Minus => left - right,
                        BinaryOperator::Mul => left * right,
                        BinaryOperator::Div => left / right,
                        _ => unreachable!("unhandled arithmetic operator: {arithmetic}"),
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

/// Same as [`resolve_expression`] but without variables.
///
/// If the given expression actually contains variables
/// (AKA [`Expression::Identifier`]) then an error is returned.
pub(crate) fn resolve_literal_expression(expr: &Expression) -> Result<Value, SqlError> {
    resolve_expression(&vec![], &Schema::empty(), expr)
}

/// Returns `true` if the where [`Expression`] applied to the given tuple
/// evaluates to true.
pub(crate) fn eval_where(
    schema: &Schema,
    tuple: &Vec<Value>,
    expr: &Expression,
) -> Result<bool, SqlError> {
    match resolve_expression(tuple, schema, expr)? {
        Value::Bool(bool) => Ok(bool),

        other => Err(SqlError::TypeError(TypeError::ExpectedType {
            expected: VmDataType::Bool,
            found: Expression::Value(other),
        })),
    }
}

#[cfg(test)]
mod tests {
    use super::VmError;
    use crate::{
        db::{DbError, Schema, SqlError},
        sql::{
            parser::Parser,
            statement::{Column, DataType, Value},
        },
        vm::resolve_expression,
    };

    struct VmCtx {
        tuple: Vec<Value>,
        schema: Schema,
    }

    impl VmCtx {
        fn none() -> Self {
            Self {
                tuple: vec![],
                schema: Schema::empty(),
            }
        }
    }

    struct Resolve<'e> {
        expression: &'e str,
        vm_context: VmCtx,
        expected: Result<Value, SqlError>,
    }

    fn assert_resolve(
        Resolve {
            expression,
            vm_context,
            expected,
        }: Resolve,
    ) -> Result<(), DbError> {
        let expr = Parser::new(expression).parse_expression()?;

        assert_eq!(
            resolve_expression(&vm_context.tuple, &vm_context.schema, &expr),
            expected
        );

        Ok(())
    }

    #[test]
    fn simple_literal_expression() -> Result<(), DbError> {
        assert_resolve(Resolve {
            expression: "2 + 2",
            vm_context: VmCtx::none(),
            expected: Ok(Value::Number(4)),
        })
    }

    #[test]
    fn complex_literal_expression() -> Result<(), DbError> {
        assert_resolve(Resolve {
            expression: "2 * ((10*10) - ((5+5) * -(-3))) / 14",
            vm_context: VmCtx::none(),
            expected: Ok(Value::Number(10)),
        })
    }

    #[test]
    fn complex_expression_with_variables() -> Result<(), DbError> {
        assert_resolve(Resolve {
            expression: "x / y + 10 * z - 5",
            vm_context: VmCtx {
                schema: Schema::new(vec![
                    Column::new("x", DataType::Int),
                    Column::new("y", DataType::Int),
                    Column::new("z", DataType::Int),
                ]),
                tuple: vec![Value::Number(100), Value::Number(10), Value::Number(3)],
            },
            expected: Ok(Value::Number(35)),
        })
    }

    #[test]
    fn resolve_simple_boolean() -> Result<(), DbError> {
        assert_resolve(Resolve {
            expression: "x + y != 10",
            vm_context: VmCtx {
                schema: Schema::new(vec![
                    Column::new("x", DataType::Int),
                    Column::new("y", DataType::Int),
                ]),
                tuple: vec![Value::Number(6), Value::Number(5)],
            },
            expected: Ok(Value::Bool(true)),
        })
    }

    #[test]
    fn resolve_complex_boolean() -> Result<(), DbError> {
        assert_resolve(Resolve {
            expression: "x + 10 < 20 AND y + 1 = 10 OR z != 0",
            vm_context: VmCtx {
                schema: Schema::new(vec![
                    Column::new("x", DataType::Int),
                    Column::new("y", DataType::Int),
                    Column::new("z", DataType::Int),
                ]),
                tuple: vec![Value::Number(10), Value::Number(5), Value::Number(0)],
            },
            expected: Ok(Value::Bool(false)),
        })
    }

    #[test]
    fn division_by_zero() -> Result<(), DbError> {
        assert_resolve(Resolve {
            expression: "x + 10 / (y - 5)",
            vm_context: VmCtx {
                schema: Schema::new(vec![
                    Column::new("x", DataType::Int),
                    Column::new("y", DataType::Int),
                ]),
                tuple: vec![Value::Number(15), Value::Number(5)],
            },
            expected: Err(VmError::DivisionByZero(10, 0).into()),
        })
    }

    #[test]
    fn invalid_column() -> Result<(), DbError> {
        assert_resolve(Resolve {
            expression: "x + 10 / (y - 5)",
            vm_context: VmCtx {
                schema: Schema::new(vec![Column::new("x", DataType::Int)]),
                tuple: vec![Value::Number(15)],
            },
            expected: Err(SqlError::InvalidColumn("y".into())),
        })
    }
}
