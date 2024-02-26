//! This module is the next step in the pipeline after [`super::analyzer`].
//!
//! Here we take an analyzed statement and try to reduce the number of
//! operations that it requires.

use std::mem;

use crate::{
    db::Schema,
    sql::{BinaryOperator, Expression, Statement},
    vm,
};

/// Takes a statement and outputs an equivalent, optimized one.
fn optimize(statement: &mut Statement) {
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
    if let Some(expr) = r#where.take() {
        *r#where = Some(simplfy(expr));
    }
}

/// Simplifies all the expressions in a list.
fn simplify_all<'e>(expressions: impl Iterator<Item = &'e mut Expression>) {
    for expr in expressions {
        let old_expr = mem::replace(expr, Expression::Wildcard);
        *expr = simplfy(old_expr);
    }
}

/// Takes an expression and reduces its number of operations.
///
/// For now, the only thing this function does is resolve raw values in
/// expressions like `x + 2 + 4 + 6` which is equivalent to `x + 12`. Ideally
/// we should be able to simplify expressions like `2*x + 2*y` into `2*(x+y)`
/// using common factors, but doing so seems to require a "computer algebra
/// system" which doesn't sound precisely easy to implement.
///
/// # Implementation
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
fn simplfy(expression: Expression) -> Expression {
    match expression {
        Expression::UnaryOperation { operator, mut expr } => {
            *expr = simplfy(*expr);

            let is_raw = match expr.as_ref() {
                Expression::Value(_) => true,
                _ => false,
            };

            let new_expr = Expression::UnaryOperation { operator, expr };

            if is_raw {
                resolve_raw_expression(&new_expr)
            } else {
                new_expr
            }
        }

        Expression::BinaryOperation {
            mut left,
            operator,
            mut right,
        } => {
            match (simplfy(*left), simplfy(*right)) {
                // Attempt to simplify expressions like `x + 2 + 4`.
                (
                    Expression::BinaryOperation {
                        left: left_ident,
                        operator: BinaryOperator::Plus,
                        right: center_val,
                    },
                    right_val @ Expression::Value(_),
                ) if matches!(
                    (center_val.as_ref(), operator),
                    (Expression::Value(_), BinaryOperator::Plus)
                ) =>
                {
                    let value = resolve_raw_expression(&Expression::BinaryOperation {
                        left: center_val,
                        operator,
                        right: Box::new(right_val),
                    });
                    Expression::BinaryOperation {
                        left: left_ident,
                        operator: BinaryOperator::Plus,
                        right: Box::new(value),
                    }
                }

                // No luck, reuse the old boxes and try to resolve raw values.
                (simplified_left, simplified_right) => {
                    *left = simplified_left;
                    *right = simplified_right;

                    let is_raw = match (left.as_ref(), right.as_ref()) {
                        (Expression::Value(_), Expression::Value(_)) => true,
                        _ => false,
                    };

                    let new_expr = Expression::BinaryOperation {
                        left,
                        operator,
                        right,
                    };

                    if is_raw {
                        resolve_raw_expression(&new_expr)
                    } else {
                        new_expr
                    }
                }
            }
        }

        Expression::Nested(expr) => simplfy(*expr),

        other => other,
    }
}

/// Resolves an expression that doesn't contain any variables into a value.
fn resolve_raw_expression(expression: &Expression) -> Expression {
    let value = vm::resolve_expression(&vec![], &Schema::empty(), expression);
    // We can unwrap here because there are only raw values and the analyzer
    // has already checked the types. So this should not fail.
    Expression::Value(value.unwrap())
}

#[cfg(test)]
mod tests {
    use crate::{
        query::optimizer::simplfy,
        sql::{BinaryOperator, Expression, ParseResult, Parser, Value},
    };

    #[test]
    fn simplify_unary() -> ParseResult<()> {
        let expr = Parser::new("x + -(2+2+2)").parse_expression()?;

        assert_eq!(
            simplfy(expr),
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
        let expr = Parser::new("x + 2 + 4 + 6").parse_expression()?;

        assert_eq!(
            simplfy(expr),
            Expression::BinaryOperation {
                left: Box::new(Expression::Identifier("x".into())),
                operator: BinaryOperator::Plus,
                right: Box::new(Expression::Value(Value::Number(12))),
            }
        );

        Ok(())
    }

    #[test]
    fn dont_alter_expression_if_cant_simplify() -> ParseResult<()> {
        let expr = Parser::new("x * 2 + 6").parse_expression()?;
        let expected = expr.clone();

        assert_eq!(simplfy(expr), expected);

        Ok(())
    }
}
