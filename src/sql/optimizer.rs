//! This module is the next step in the pipeline after [`super::analyzer`].
//!
//! Here we take an analyzed statement and try to reduce the number of
//! operations that it requires.

use std::mem;

use super::statement::{BinaryOperator, Expression, Statement, UnaryOperator, Value};
use crate::{db::SqlError, vm};

/// Takes a statement and transforms it into an equivalent, optimized one.
pub(crate) fn optimize(statement: &mut Statement) -> Result<(), SqlError> {
    match statement {
        Statement::Insert { values, .. } => {
            simplify_all(values.iter_mut())?;
        }

        Statement::Select {
            columns,
            r#where,
            order_by,
            ..
        } => {
            simplify_all(columns.iter_mut())?;
            simplfy_where(r#where)?;
            simplify_all(order_by.iter_mut())?;
        }

        Statement::Delete { r#where, .. } => simplfy_where(r#where)?,

        Statement::Update {
            columns, r#where, ..
        } => {
            simplfy_where(r#where)?;
            simplify_all(columns.iter_mut().map(|col| &mut col.value))?;
        }

        Statement::Explain(inner) => {
            optimize(&mut *inner)?;
        }

        _ => {}
    };

    Ok(())
}

/// Simplifies an optional where clause.
fn simplfy_where(r#where: &mut Option<Expression>) -> Result<(), SqlError> {
    r#where.as_mut().map(simplify).unwrap_or(Ok(()))
}

/// Simplifies all the expressions in a list.
fn simplify_all<'e>(
    mut expressions: impl Iterator<Item = &'e mut Expression>,
) -> Result<(), SqlError> {
    expressions.try_for_each(simplify)
}

/// Takes an expression and reduces its number of operations.
///
/// For now, the only thing this function does is resolve literal expressions
/// like `x + 2 + 4 + 6` which is equivalent to `x + 12` or apply basic rules
/// such as `x * 0 = 0`. Ideally we should be able to simplify expressions like
/// `2*x + 2*y` into `2*(x+y)` using common factors, but doing so seems to
/// require a "computer algebra system" which doesn't sound precisely easy to
/// implement.
///
/// # Implementation Notes
///
/// Expressions like `x + 2 + 4` are parsed into a tree that looks like this:
///
/// ```text
///      +
///    /   \
/// x + 2   4
/// ```
///
/// In a nutshell, the algorithm is as follows:
///
/// 1. Reorder the tree.
///
/// ```text
///      +
///    /   \
///   x   2 + 4
/// ```
///
/// 2. Solve the right part.
///
/// ```text
///      +
///    /   \
///   x     6
/// ```
///
/// 3. Return the new expression: `x + 6`.
///
/// That's not exactly how we do it in the code below because we have to follow
/// the Rust borrow rules, but we do more or less the same thing.
///
/// Another important note is that we use [`Expression::Wildcard`] as if it was
/// [`Option::take`], which basically moves to value out and leaves [`None`] in
/// its place. We do the same thing but leave [`Expression::Wildcard`] in a box
/// that's about to get dropped. Rust borrow checker and stuff ¯\_(ツ)_/¯
fn simplify(expression: &mut Expression) -> Result<(), SqlError> {
    match expression {
        Expression::UnaryOperation { expr, .. } => {
            simplify(expr)?;
            if let Expression::Value(_) = expr.as_ref() {
                *expression = resolve_literal_expression(expression)?
            }
        }

        Expression::BinaryOperation {
            left,
            operator,
            right,
        } => {
            simplify(left.as_mut())?;
            simplify(right.as_mut())?;

            match (left.as_mut(), operator, right.as_mut()) {
                // Resolve expression with literal values to a single value.
                (Expression::Value(_), _op, Expression::Value(_)) => {
                    *expression = resolve_literal_expression(expression)?;
                }

                // Resolve these expressions to "x":
                // 1 * x
                // x * 1
                // x / 1
                // x + 0
                // x - 0
                // 0 + x
                (
                    Expression::Value(Value::Number(1)),
                    BinaryOperator::Mul,
                    variable @ Expression::Identifier(_),
                )
                | (
                    variable @ Expression::Identifier(_),
                    BinaryOperator::Mul | BinaryOperator::Div,
                    Expression::Value(Value::Number(1)),
                )
                | (
                    variable @ Expression::Identifier(_),
                    BinaryOperator::Plus | BinaryOperator::Minus,
                    Expression::Value(Value::Number(0)),
                )
                | (
                    Expression::Value(Value::Number(0)),
                    BinaryOperator::Plus,
                    variable @ Expression::Identifier(_),
                ) => {
                    *expression = mem::replace(variable, Expression::Wildcard);
                }

                // Resolve these expressions to 0:
                // 0 * x
                // 0 / x
                // x * 0
                (
                    zero @ Expression::Value(Value::Number(0)),
                    BinaryOperator::Mul | BinaryOperator::Div,
                    Expression::Identifier(_),
                )
                | (
                    Expression::Identifier(_),
                    BinaryOperator::Mul,
                    zero @ Expression::Value(Value::Number(0)),
                ) => {
                    *expression = mem::replace(zero, Expression::Wildcard);
                }

                // Resolve binary operation `0 - x` to unary `-x`.
                (
                    Expression::Value(Value::Number(0)),
                    BinaryOperator::Minus,
                    Expression::Identifier(_),
                ) => match mem::replace(expression, Expression::Wildcard) {
                    Expression::BinaryOperation { right, .. } => {
                        *expression = Expression::UnaryOperation {
                            operator: UnaryOperator::Minus,
                            expr: right,
                        }
                    }
                    _ => unreachable!(),
                },

                // Attempt to simplify expressions like `x + 2 + 4` into `x + 6`.
                (
                    Expression::BinaryOperation {
                        left: variable,
                        operator: BinaryOperator::Plus,
                        right: center_value,
                    },
                    BinaryOperator::Plus,
                    right_value @ Expression::Value(_),
                ) if matches!(center_value.as_ref(), Expression::Value(_)) => {
                    // Swap "x" with 4.
                    mem::swap(variable.as_mut(), right_value);
                    // Compute 4 + 2.
                    *left.as_mut() = resolve_literal_expression(left)?;
                    // Swap 6 + x to make it x + 6
                    mem::swap(left, right);
                }

                // Turn expressions like `6 + x` into `x + 6` to make them work
                // with the case above.
                (
                    literal @ Expression::Value(_),
                    BinaryOperator::Plus,
                    variable @ Expression::Identifier(_),
                ) => {
                    mem::swap(variable, literal);
                }

                _other => {}
            }
        }

        // TODO: We remove the nesting here which causes the [`Display`]
        // implementation for [`Expression`] to be wrong, because it will never
        // print parenthesis. The expression tree is the same with nesting or
        // without nesting, but printing isn't. We could maintain the nesting
        // and pattern match over
        // Expression::Value(_) | Expression::Nested(Expression::Value(_))
        // every single time, but that will be complicated since
        // [`Expression::Nested`] contains a box and it's not easy to pattern
        // match over boxes. See these tracking issues:
        //
        // https://github.com/rust-lang/rust/issues/29641
        //
        // https://github.com/rust-lang/rust/issues/87121
        Expression::Nested(nested) => {
            simplify(nested.as_mut())?;
            *expression = mem::replace(nested.as_mut(), Expression::Wildcard);
        }

        _other => {}
    };

    Ok(())
}

/// Resolves an expression that doesn't contain variables into [`Expression::Value`].
///
/// This function is the only reason we need to return [`Result`] in this
/// module. The only possible error for now is division by zero. We could catch
/// arithmetic errors in [`super::analyzer`] but that would require resolving
/// some of the expressions both in the analyzer and here, which doesn't make
/// much sense.
fn resolve_literal_expression(expression: &Expression) -> Result<Expression, SqlError> {
    vm::resolve_literal_expression(expression).map(Expression::Value)
}

#[cfg(test)]
mod tests {
    use super::{optimize, simplify};
    use crate::{
        db::DbError,
        sql::{
            parser::Parser,
            statement::{BinaryOperator, Expression, Statement, Value},
        },
    };

    struct Opt<'e> {
        raw_input: &'e str,
        optimized: &'e str,
    }

    fn assert_optimize_expr(opt: Opt) -> Result<(), DbError> {
        assert_eq!(
            simplify_expr(opt.raw_input)?,
            Parser::new(opt.optimized).parse_expression()?
        );

        Ok(())
    }

    fn assert_optimize_sql(opt: Opt) -> Result<(), DbError> {
        assert_eq!(
            optimize_sql(opt.raw_input)?,
            Parser::new(opt.optimized).parse_statement()?
        );

        Ok(())
    }

    fn simplify_expr(expr: &str) -> Result<Expression, DbError> {
        let mut expr = Parser::new(expr).parse_expression()?;
        simplify(&mut expr)?;

        Ok(expr)
    }

    fn optimize_sql(sql: &str) -> Result<Statement, DbError> {
        let mut statement = Parser::new(sql).parse_statement()?;
        optimize(&mut statement)?;

        Ok(statement)
    }

    #[test]
    fn simplify_addition() -> Result<(), DbError> {
        assert_optimize_expr(Opt {
            raw_input: "x + 2 + 4 + 6",
            optimized: "x + 12",
        })
    }

    #[test]
    fn simplify_addition_reverse_order() -> Result<(), DbError> {
        assert_optimize_expr(Opt {
            raw_input: "1 + 2 + 3 + x",
            optimized: "x + 6",
        })
    }

    #[test]
    fn simplify_addition_on_both_sides() -> Result<(), DbError> {
        assert_optimize_expr(Opt {
            raw_input: "1 + 2 + 3 + x + 3 + 2 + 1",
            optimized: "x + 12",
        })
    }

    #[test]
    fn simplify_multiply_by_one() -> Result<(), DbError> {
        assert_optimize_expr(Opt {
            raw_input: "x * (3 - 2)",
            optimized: "x",
        })
    }

    #[test]
    fn simplify_multiply_by_one_in_other_direction() -> Result<(), DbError> {
        assert_optimize_expr(Opt {
            raw_input: "(2 - 1) * x",
            optimized: "x",
        })
    }

    #[test]
    fn simplify_divide_by_one() -> Result<(), DbError> {
        assert_optimize_expr(Opt {
            raw_input: "x / (10 - 9)",
            optimized: "x",
        })
    }

    #[test]
    fn simplify_add_zero() -> Result<(), DbError> {
        assert_optimize_expr(Opt {
            raw_input: "x + (10 - 10)",
            optimized: "x",
        })
    }

    #[test]
    fn simplify_substract_zero() -> Result<(), DbError> {
        assert_optimize_expr(Opt {
            raw_input: "x - ((6 - 4) - 2)",
            optimized: "x",
        })
    }

    #[test]
    fn simplify_multiply_by_zero() -> Result<(), DbError> {
        assert_optimize_expr(Opt {
            raw_input: "x + x * (2-2)",
            optimized: "x",
        })
    }

    #[test]
    fn dont_alter_expression_if_cant_simplify() -> Result<(), DbError> {
        assert_optimize_expr(Opt {
            raw_input: "x * 2 + 6",
            optimized: "x * 2 + 6",
        })
    }

    /// This one's done manually because if we run the parser on `x + -6` or
    /// `x - 6` we won't get the same result that the optimizer produces.
    #[test]
    fn simplify_unary() -> Result<(), DbError> {
        assert_eq!(
            simplify_expr("x + -(2+2+2)")?,
            Expression::BinaryOperation {
                left: Box::new(Expression::Identifier("x".into())),
                operator: BinaryOperator::Plus,
                right: Box::new(Expression::Value(Value::Number(-6))),
            }
        );

        Ok(())
    }

    #[test]
    fn simplify_zero_minus_var() -> Result<(), DbError> {
        assert_optimize_expr(Opt {
            raw_input: "(2-2) - x",
            optimized: "-x",
        })
    }

    #[test]
    fn optimize_update() -> Result<(), DbError> {
        assert_optimize_sql(Opt {
            raw_input: "UPDATE products SET price = price + 2+2 WHERE discount < 2*10;",
            optimized: "UPDATE products SET price = price + 4 WHERE discount < 20;",
        })
    }

    #[test]
    fn optimize_select() -> Result<(), DbError> {
        assert_optimize_sql(Opt {
            raw_input: "SELECT x * 1, 2 + (2 + 2), y FROM some_table WHERE x < 5 -(-5) ORDER BY x + (y * (9-8));",
            optimized: "SELECT x, 6, y FROM some_table WHERE x < 10 ORDER BY x + y;",
        })
    }

    #[test]
    fn optimize_insert() -> Result<(), DbError> {
        assert_optimize_sql(Opt {
            raw_input: "INSERT INTO some_table (a,b,c) VALUES (2+2, 2*(2*10), -(-5)-5);",
            optimized: "INSERT INTO some_table (a,b,c) VALUES (4, 40, 0);",
        })
    }

    #[test]
    fn optimize_delete() -> Result<(), DbError> {
        assert_optimize_sql(Opt {
            raw_input: "DELETE FROM t WHERE x >= y * (2 - 2) AND x != (10+10);",
            optimized: "DELETE FROM t WHERE x >= 0 AND x != 20;",
        })
    }
}
