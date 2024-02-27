//! This module is the next step in the pipeline after [`super::analyzer`].
//!
//! Here we take an analyzed statement and try to reduce the number of
//! operations that it requires.

use std::mem;

use crate::{
    db::Schema,
    sql::{BinaryOperator, Expression, Statement, UnaryOperator, Value},
    vm,
};

/// Takes a statement and transforms it into an equivalent, optimized one.
pub(crate) fn optimize(statement: &mut Statement) {
    match statement {
        Statement::Insert { values, .. } => {
            simplify_all(values.iter_mut());
        }

        Statement::Select {
            columns,
            r#where,
            order_by,
            ..
        } => {
            simplify_all(columns.iter_mut());
            simplfy_where(r#where);
            simplify_all(order_by.iter_mut());
        }

        Statement::Delete { r#where, .. } => simplfy_where(r#where),

        Statement::Update {
            columns, r#where, ..
        } => {
            simplfy_where(r#where);
            simplify_all(columns.iter_mut().map(|col| &mut col.value));
        }

        _ => {}
    }
}

/// Simplifies an optional where clause.
fn simplfy_where(r#where: &mut Option<Expression>) {
    r#where.as_mut().map(simplfy);
}

/// Simplifies all the expressions in a list.
fn simplify_all<'e>(expressions: impl Iterator<Item = &'e mut Expression>) {
    expressions.for_each(simplfy)
}

/// Takes an expression and reduces its number of operations.
///
/// For now, the only thing this function does is resolve literal expressions
/// like `x + 2 + 4 + 6` which is equivalent to `x + 12`. Ideally we should be
/// able to simplify expressions like `2*x + 2*y` into `2*(x+y)` using common
/// factors, but doing so seems to require a "computer algebra system" which
/// doesn't sound precisely easy to implement.
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
fn simplfy(expression: &mut Expression) {
    match expression {
        Expression::UnaryOperation { expr, .. } => {
            simplfy(expr);
            if let Expression::Value(_) = expr.as_ref() {
                *expression = resolve_literal_expression(&expression)
            }
        }

        Expression::BinaryOperation {
            left,
            operator,
            right,
        } => {
            simplfy(left.as_mut());
            simplfy(right.as_mut());

            match (left.as_mut(), operator, right.as_mut()) {
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
                    *left.as_mut() = resolve_literal_expression(&left);
                    // Swap 6 + x to make it x + 6
                    mem::swap(left, right);
                }

                // Swap expressions like 6 + x to make them work with the case above.
                (
                    literal @ Expression::Value(_),
                    BinaryOperator::Plus,
                    variable @ Expression::Identifier(_),
                ) => {
                    mem::swap(variable, literal);
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

                // Expression with literal values.
                (Expression::Value(_), _op, Expression::Value(_)) => {
                    *expression = resolve_literal_expression(&expression);
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
        // match over boxes. See this tracking issues:
        //
        // https://github.com/rust-lang/rust/issues/29641
        //
        // https://github.com/rust-lang/rust/issues/87121
        Expression::Nested(nested) => {
            simplfy(nested.as_mut());
            *expression = mem::replace(nested.as_mut(), Expression::Wildcard);
        }

        _other => {}
    }
}

/// Resolves an expression that doesn't contain any variables into a value.
fn resolve_literal_expression(expression: &Expression) -> Expression {
    let value = vm::resolve_expression(&vec![], &Schema::empty(), expression);
    // We can unwrap here because there are only literal values and the analyzer
    // has already checked the types. So this should not fail.
    Expression::Value(value.unwrap())
}

#[cfg(test)]
mod tests {
    use crate::{
        query::optimizer::{optimize, simplfy},
        sql::{BinaryOperator, Expression, ParseResult, Parser, Statement, Value},
    };

    struct Opt<'e> {
        raw_input: &'e str,
        optimized: &'e str,
    }

    fn assert_optimize_expr(opt: Opt) -> ParseResult<()> {
        assert_eq!(
            simplify_expr(opt.raw_input)?,
            Parser::new(opt.optimized).parse_expression()?
        );

        Ok(())
    }

    fn assert_optimize_sql(opt: Opt) -> ParseResult<()> {
        assert_eq!(
            optimize_sql(opt.raw_input)?,
            Parser::new(opt.optimized).parse_statement()?
        );

        Ok(())
    }

    fn simplify_expr(expr: &str) -> ParseResult<Expression> {
        let mut expr = Parser::new(expr).parse_expression()?;
        simplfy(&mut expr);

        Ok(expr)
    }

    fn optimize_sql(sql: &str) -> ParseResult<Statement> {
        let mut statement = Parser::new(sql).parse_statement()?;
        optimize(&mut statement);

        Ok(statement)
    }

    #[test]
    fn simplify_addition() -> ParseResult<()> {
        assert_optimize_expr(Opt {
            raw_input: "x + 2 + 4 + 6",
            optimized: "x + 12",
        })
    }

    #[test]
    fn simplify_addition_reverse_order() -> ParseResult<()> {
        assert_optimize_expr(Opt {
            raw_input: "1 + 2 + 3 + x",
            optimized: "x + 6",
        })
    }

    #[test]
    fn simplify_addition_on_both_sides() -> ParseResult<()> {
        assert_optimize_expr(Opt {
            raw_input: "1 + 2 + 3 + x + 3 + 2 + 1",
            optimized: "x + 12",
        })
    }

    #[test]
    fn simplify_multiply_by_one() -> ParseResult<()> {
        assert_optimize_expr(Opt {
            raw_input: "x * (3 - 2)",
            optimized: "x",
        })
    }

    #[test]
    fn simplify_multiply_by_one_in_other_direction() -> ParseResult<()> {
        assert_optimize_expr(Opt {
            raw_input: "(2 - 1) * x",
            optimized: "x",
        })
    }

    #[test]
    fn simplify_divide_by_one() -> ParseResult<()> {
        assert_optimize_expr(Opt {
            raw_input: "x / (10 - 9)",
            optimized: "x",
        })
    }

    #[test]
    fn simplify_add_zero() -> ParseResult<()> {
        assert_optimize_expr(Opt {
            raw_input: "x + (10 - 10)",
            optimized: "x",
        })
    }

    #[test]
    fn simplify_substract_zero() -> ParseResult<()> {
        assert_optimize_expr(Opt {
            raw_input: "x - ((6 - 4) - 2)",
            optimized: "x",
        })
    }

    #[test]
    fn simplify_multiply_by_zero() -> ParseResult<()> {
        assert_optimize_expr(Opt {
            raw_input: "x + x * (2-2)",
            optimized: "x",
        })
    }

    #[test]
    fn dont_alter_expression_if_cant_simplify() -> ParseResult<()> {
        assert_optimize_expr(Opt {
            raw_input: "x * 2 + 6",
            optimized: "x * 2 + 6",
        })
    }

    /// This one's done manually because if we run the parser on `x + -6` or
    /// `x - 6` we won't get the same result that the optimizer produces.
    #[test]
    fn simplify_unary() -> ParseResult<()> {
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
    fn simplify_zero_minus_var() -> ParseResult<()> {
        assert_optimize_expr(Opt {
            raw_input: "(2-2) - x",
            optimized: "-x",
        })
    }

    #[test]
    fn optimize_update() -> ParseResult<()> {
        assert_optimize_sql(Opt {
            raw_input: "UPDATE products SET price = price + 2+2 WHERE discount < 2*10;",
            optimized: "UPDATE products SET price = price + 4 WHERE discount < 20;",
        })
    }

    #[test]
    fn optimize_select() -> ParseResult<()> {
        assert_optimize_sql(Opt {
            raw_input: "SELECT x * 1, 2 + (2 + 2), y FROM some_table WHERE x < 5 -(-5) ORDER BY x + (y * (9-8));",
            optimized: "SELECT x, 6, y FROM some_table WHERE x < 10 ORDER BY x + y;",
        })
    }

    #[test]
    fn optimize_insert() -> ParseResult<()> {
        assert_optimize_sql(Opt {
            raw_input: "INSERT INTO some_table (a,b,c) VALUES (2+2, 2*(2*10), -(-5)-5);",
            optimized: "INSERT INTO some_table (a,b,c) VALUES (4, 40, 0);",
        })
    }

    #[test]
    fn optimize_delete() -> ParseResult<()> {
        assert_optimize_sql(Opt {
            raw_input: "DELETE FROM t WHERE x >= y * (2 - 2) AND x != (10+10);",
            optimized: "DELETE FROM t WHERE x >= 0 AND x != 20;",
        })
    }
}
