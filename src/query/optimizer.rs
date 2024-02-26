//! This module is the next step in the pipeline after [`super::analyzer`].
//!
//! Here we take an analyzed statement and try to reduce the number of
//! operations that it requires.

use std::mem;

use crate::{
    db::Schema,
    sql::{BinaryOperator, Expression, Statement, Value},
    vm,
};

/// Takes a statement and transforms it into an equivalent, optimized one.
pub(crate) fn optimize(statement: &mut Statement) {
    match statement {
        Statement::Insert { values, .. } => {
            simplify_all(values.iter_mut());
        }

        Statement::Select {
            r#where, order_by, ..
        } => {
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
                // Attempt to simplify expressions like `x + 2 + 4`.
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
                    let resolved_value = resolve_literal_expression(&left);

                    // Set the expression to x + 6.
                    mem::swap(left.as_mut(), right_value);
                    *right.as_mut() = resolved_value;
                }

                // Resolve `x * 1` or `1 * x` or `x / 1` to `x`.
                (
                    Expression::Value(Value::Number(1)),
                    BinaryOperator::Mul,
                    ident @ Expression::Identifier(_),
                )
                | (
                    ident @ Expression::Identifier(_),
                    BinaryOperator::Mul | BinaryOperator::Div,
                    Expression::Value(Value::Number(1)),
                ) => {
                    *expression = mem::replace(ident, Expression::Wildcard);
                }

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
        sql::{Assignment, BinaryOperator, Expression, ParseResult, Parser, Statement, Value},
    };

    #[test]
    fn simplify_unary() -> ParseResult<()> {
        let mut expr = Parser::new("x + -(2+2+2)").parse_expression()?;
        simplfy(&mut expr);

        assert_eq!(
            expr,
            Expression::BinaryOperation {
                left: Box::new(Expression::Identifier("x".into())),
                operator: BinaryOperator::Plus,
                right: Box::new(Expression::Value(Value::Number(-6))),
            }
        );

        Ok(())
    }

    #[test]
    fn simplify_binary() -> ParseResult<()> {
        let mut expr = Parser::new("x + 2 + 4 + 6").parse_expression()?;
        simplfy(&mut expr);

        assert_eq!(
            expr,
            Expression::BinaryOperation {
                left: Box::new(Expression::Identifier("x".into())),
                operator: BinaryOperator::Plus,
                right: Box::new(Expression::Value(Value::Number(12))),
            }
        );

        Ok(())
    }

    #[test]
    fn simplify_multiply_by_one() -> ParseResult<()> {
        let mut expr = Parser::new("x * (3 - 2)").parse_expression()?;
        simplfy(&mut expr);

        assert_eq!(expr, Expression::Identifier("x".into()));

        Ok(())
    }

    #[test]
    fn dont_alter_expression_if_cant_simplify() -> ParseResult<()> {
        let mut expr = Parser::new("x * 2 + 6").parse_expression()?;
        let expected = expr.clone();

        simplfy(&mut expr);

        assert_eq!(expr, expected);

        Ok(())
    }

    #[test]
    fn optimize_sql_statement() -> ParseResult<()> {
        let sql = "UPDATE products SET price = price + 2 + 2 WHERE discount < 2 * 10;";
        let mut statement = Parser::new(sql).parse_statement()?;
        optimize(&mut statement);

        assert_eq!(
            statement,
            Statement::Update {
                table: "products".into(),
                columns: vec![Assignment {
                    identifier: "price".into(),
                    value: Expression::BinaryOperation {
                        left: Box::new(Expression::Identifier("price".into())),
                        operator: BinaryOperator::Plus,
                        right: Box::new(Expression::Value(Value::Number(4))),
                    }
                }],
                r#where: Some(Expression::BinaryOperation {
                    left: Box::new(Expression::Identifier("discount".into())),
                    operator: BinaryOperator::Lt,
                    right: Box::new(Expression::Value(Value::Number(20)))
                })
            }
        );

        Ok(())
    }
}
