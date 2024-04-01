//! Generates optimized plans.
//!
//! See the module level documentation of [`crate::vm::plan`].

use std::{
    collections::{HashMap, HashSet},
    io::{Read, Seek, Write},
    mem,
    ops::Bound,
    ptr,
    rc::Rc,
};

use crate::{
    db::{Database, DatabaseContext, DbError, IndexMetadata, Relation, SqlError, TableMetadata},
    paging::io::FileOps,
    sql::statement::{BinaryOperator, Expression, Value},
    storage::{tuple, Cursor},
    vm::plan::{
        Collect, CollectConfig, ExactMatch, Filter, KeyScan, Plan, RangeScan, RangeScanConfig,
        Scan, SeqScan, Sort, SortConfig, TuplesComparator,
    },
};

/// Generates an optimized scan plan.
///
/// The generated plan will be either [`SeqScan`], [`RangeScan`] or
/// [`IndexScan`] with an optional [`Filter`] on top of it, depending on how the
/// query looks like.
pub(crate) fn generate_scan_plan<F: Seek + Read + Write + FileOps>(
    table: &str,
    filter: Option<Expression>,
    db: &mut Database<F>,
) -> Result<Plan<F>, DbError> {
    // let source = if let Some(optimized_scan) = generate_optimized_scan_plan(table, db, &mut filter)?
    // {
    //     optimized_scan
    // } else {
    //     generate_sequential_scan_plan(table, db)?
    // };

    let source = generate_sequential_scan_plan(table, db)?;

    let Some(expr) = filter else {
        return Ok(source);
    };

    Ok(Plan::Filter(Filter {
        source: Box::new(source),
        schema: db.table_metadata(table)?.schema.clone(),
        filter: expr,
    }))
}

/// Constructs a [`Plan::SeqScan`] instance.
fn generate_sequential_scan_plan<F: Seek + Read + Write + FileOps>(
    table: &str,
    db: &mut Database<F>,
) -> Result<Plan<F>, DbError> {
    let metadata = db.table_metadata(table)?;

    Ok(Plan::SeqScan(SeqScan {
        cursor: Cursor::new(metadata.root, 0),
        table: metadata.clone(),
        pager: Rc::clone(&db.pager),
    }))
}

/// Attempts to generate a [`RangeScan`] or an [`ExactMatch`] plan.
///
/// It's only possible to do so if we find an expression that contains an
/// indexed column or the table primary key column and must always be executed.
/// Otherwise we'll fallback to sequential scans.
// fn generate_optimized_scan_plan<F: Seek + Read + Write + FileOps>(
//     table_name: &str,
//     db: &mut Database<F>,
//     filter: &mut Option<Expression>,
// ) -> Result<Option<Plan<F>>, DbError> {
//     let Some(filter) = filter else {
//         return Ok(None);
//     };

//     let table = db.table_metadata(table_name)?.clone();

//     // Build index map (column name -> index metadata)
//     let mut indexes = HashMap::from_iter(
//         table
//             .indexes
//             .iter()
//             .map(|index| (index.column.name.clone(), index.clone())),
//     );

//     let exprs = find_indexed_exprs(&table, &indexes, filter) else {
//         return Ok(None);
//     };

//     // This shouldn't fail at this point.
//     let col = table.schema.index_of(expr.c).ok_or(DbError::Other(format!(
//         "found invalid column {indexed_col} while generating query plan"
//     )))?;

//     let relation = if let Some(index) = indexes.get(indexed_col) {
//         Relation::Index(index.clone())
//     } else {
//         Relation::Table(table.clone())
//     };

//     // Preserialize keys.
//     let (start, end) = match (&**left, &**right) {
//         (Expression::Identifier(_col), Expression::Value(value))
//         | (Expression::Value(value), Expression::Identifier(_col)) => (
//             tuple::serialize_key(&table.schema.columns[col].data_type, value),
//             vec![],
//         ),

//         _ => unreachable!(),
//     };

//     let scan = match (left.as_ref(), operator, right.as_ref()) {
//         // Case 1:
//         // SELECT * FROM t WHERE x = 5;
//         // SELECT * FROM t WHERE 5 = x;
//         //
//         // Position the cursor at key 5 and stop after returning that key.
//         (Expression::Identifier(_col), BinaryOperator::Eq, Expression::Value(value))
//         | (Expression::Value(value), BinaryOperator::Eq, Expression::Identifier(_col)) => {
//             Scan::Match(key)
//         }

//         // Case 2:
//         // SELECT * FROM t WHERE x > 5;
//         // SELECT * FROM t WHERE 5 < x;
//         //
//         // Position the cursor at key 5, consume key 5 which will cause the
//         // cursor to move to the successor and return keys until finished.
//         (Expression::Identifier(_col), BinaryOperator::Gt, Expression::Value(_value))
//         | (Expression::Value(_value), BinaryOperator::Lt, Expression::Identifier(_col)) => {
//             Scan::Range((Bound::Excluded(key), Bound::Unbounded))
//         }

//         // Case 3:
//         // SELECT * FROM t WHERE x < 5;
//         // SELECT * FROM t WHERE 5 > x;
//         //
//         // Allow the cursor to initialize normally (at the smallest key) and
//         // tell it to stop once it finds a key >= 5.
//         (Expression::Identifier(_col), BinaryOperator::Lt, Expression::Value(_value))
//         | (Expression::Value(_value), BinaryOperator::Gt, Expression::Identifier(_col)) => {
//             Scan::Range((Bound::Unbounded, Bound::Excluded(key)))
//         }

//         // Case 4:
//         // SELECT * FROM t WHERE x >= 5;
//         // SELECT * FROM t WHERE 5 <= x;
//         //
//         // Position the cursor at key 5 and assume it's already initialized. The
//         // cursor will then return key 5 and everything after.
//         (Expression::Identifier(_col), BinaryOperator::GtEq, Expression::Value(_value))
//         | (Expression::Value(_value), BinaryOperator::LtEq, Expression::Identifier(_col)) => {
//             Scan::Range((Bound::Included(key), Bound::Unbounded))
//         }

//         // Case 5:
//         // SELECT * FROM t WHERE x <= 5;
//         // SELECT * FROM t WHERE 5 >= x;
//         //
//         // Allow the cursor to initialize normally (at the smallest key) and
//         // tell it to stop once it finds a key > 5
//         (Expression::Identifier(_col), BinaryOperator::LtEq, Expression::Value(_value))
//         | (Expression::Value(_value), BinaryOperator::GtEq, Expression::Identifier(_col)) => {
//             Scan::Range((Bound::Unbounded, Bound::Included(key)))
//         }

//         // Fallback to sequential scan in case we mess up here trying to
//         // optimize the expressions.
//         _ => return Ok(None),
//     };

//     let work_dir = db.work_dir.clone();
//     let page_size = db.pager.borrow().page_size;

//     let maybe_index = indexes.remove(&table.schema.columns[col].name);

//     let mut source = match scan {
//         Scan::Match(key) => Plan::ExactMatch(ExactMatch {
//             key,
//             relation,
//             expr: expr.clone(),
//             done: false,
//             pager: Rc::clone(&db.pager),
//         }),

//         Scan::Range(range) => Plan::RangeScan(RangeScan::from(RangeScanConfig {
//             relation,
//             range,
//             expr: expr.clone(),
//             pager: Rc::clone(&db.pager),
//         })),
//     };

//     // Table BTree scan, no additional index needed.
//     let Some(index) = maybe_index else {
//         return Ok(Some(source));
//     };

//     // Sort the keys in ascending order to favor sequential IO.
//     if let Plan::RangeScan(_) = source {
//         source = Plan::Sort(Sort::from(SortConfig {
//             page_size,
//             work_dir: work_dir.clone(),
//             collection: Collect::from(CollectConfig {
//                 source: Box::new(source),
//                 work_dir: work_dir.clone(),
//                 schema: index.schema.clone(),
//                 mem_buf_size: page_size,
//             }),
//             comparator: TuplesComparator {
//                 schema: index.schema.clone(),
//                 sort_schema: index.schema.clone(),
//                 sort_keys_indexes: vec![1],
//             },
//             input_buffers: 4,
//         }));
//     }

//     Ok(Some(Plan::KeyScan(KeyScan {
//         comparator: table.comparator()?,
//         table,
//         index,
//         pager: Rc::clone(&db.pager),
//         source: Box::new(source),
//     })))
// }

/// Finds expressions that are applied to an indexed column.
///
/// The expressions can't be "any" expression that contains an indexed column.
/// Consider this case:
///
/// ```sql
/// CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(255));
///
/// SELECT * FROM users WHERE id < 5 OR name < 'c';
/// ```
///
/// We can't use the "id" primary key index because the OR expression can
/// evaluate to true even if `id < 5` evaluates to false. However, we can use
/// the index if the logical operator is AND instead of OR:
///
/// ```sql
/// SELECT * FROM users WHERE id < 5 AND name < 'c';
/// ```
///
/// In the case of AND the WHERE clause will never evaluate to true unless
/// `id < 5`, so we can safely use the index.
///
/// Now consider this:
///
/// ```sql
/// SELECT * FROM users WHERE (id < 5 AND name < 'c') OR name > 'd';
/// ```
///
/// The OR expression makes the indexed column irrelevant again because
/// `name > 'd'` can evaluate to true anywhere in the table. But if the OR
/// expression contains the same indexed column we can still use the index:
///
/// ```sql
/// SELECT * FROM users WHERE (id < 5 AND name < 'c') OR id > 1000;
/// ```
///
/// So basically we have to build a list of AND expressions separated by OR
/// expressions. In the example above we'd build something like this:
///
/// ```text
/// [[id < 5], [id > 1000]]
/// ```
fn find_indexed_exprs<'e>(
    indexes: &HashSet<String>,
    output: &mut HashMap<String, Vec<Vec<&'e Expression>>>,
    expr: &'e Expression,
) {
    println!("{expr}");

    match expr {
        Expression::BinaryOperation {
            left,
            operator,
            right,
        } => match (&**left, &**right) {
            // Case 1. Simple expressions:
            //
            // SELECT * FROM t WHERE x >= 5
            //
            // These are the leaves of the expression tree.
            (Expression::Identifier(col), Expression::Value(_))
            | (Expression::Value(_), Expression::Identifier(col))
                if indexes.contains(col)
                    && matches!(
                        operator,
                        BinaryOperator::Eq
                            | BinaryOperator::Lt
                            | BinaryOperator::LtEq
                            | BinaryOperator::Gt
                            | BinaryOperator::GtEq
                    ) =>
            {
                output.insert(col.into(), vec![vec![expr]]);
            }

            // Case 2. AND | OR expressions.
            //
            // Subcase A: AND expression:
            // SELECT * FROM t WHERE (x < 10 OR x > 20) AND (x < 50 OR x > 70)
            //
            // Join the expressions in a single vec: [[x > 10, x < 20, x < 50, x > 70]]
            //
            // Subcase B: OR expressions:
            // SELECT * FROM t WHERE x > 10 AND x < 20 OR x > 50 AND x < 70
            //
            // Separate the expressions in two vecs: [[x > 10, x < 20], [x > 50, x < 70]]
            //
            // This is the complicated case.
            (left, right) if matches!(operator, BinaryOperator::And | BinaryOperator::Or) => {
                let mut left_output = HashMap::new();
                find_indexed_exprs(indexes, &mut left_output, left);

                let mut right_output = HashMap::new();
                find_indexed_exprs(indexes, &mut right_output, right);

                for (col, mut left_exprs) in left_output.into_iter() {
                    let mut both_branches_contain_column = false;

                    if let Some(mut right_exprs) = right_output.remove(&col) {
                        both_branches_contain_column = true;
                        left_exprs.append(&mut right_exprs);
                    }

                    match operator {
                        BinaryOperator::And => {
                            output.insert(col, vec![left_exprs.into_iter().flatten().collect()]);
                        }
                        BinaryOperator::Or => {
                            if both_branches_contain_column {
                                output.insert(col, left_exprs);
                            }
                        }
                        _ => unreachable!(),
                    }
                }
            }

            _ => {}
        },

        Expression::Nested(inner) => find_indexed_exprs(indexes, output, inner),

        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};

    use super::find_indexed_exprs;
    use crate::{
        db::{IndexMetadata, Schema},
        sql::{
            parser::Parser,
            statement::{Column, DataType},
        },
    };

    struct IndexedExprs<'i> {
        indexes: &'i [&'i str],
        expr: &'i str,
        expected: &'i [(&'i str, &'i [&'i [&'i str]])],
    }

    fn assert_indexed_exprs(
        IndexedExprs {
            indexes,
            expr,
            expected,
        }: IndexedExprs,
    ) {
        let tree = Parser::new(expr).parse_expression().unwrap();
        let indexes = HashSet::from_iter(indexes.iter().map(|name| String::from(*name)));
        let mut output = HashMap::new();

        find_indexed_exprs(&indexes, &mut output, &tree);

        let output: HashMap<String, Vec<Vec<String>>> =
            HashMap::from_iter(output.iter().map(|(name, exprs)| {
                let strings = exprs
                    .iter()
                    .map(|segment| segment.iter().map(|expr| format!("{expr}")).collect())
                    .collect();

                (name.to_owned(), strings)
            }));

        let expected: HashMap<String, Vec<Vec<String>>> =
            HashMap::from_iter(expected.into_iter().copied().map(|(name, str_exprs)| {
                let strings = str_exprs
                    .iter()
                    .map(|strs| {
                        strs.iter()
                            .copied()
                            .map(|s| Parser::new(s).parse_expression().unwrap().to_string())
                            .collect()
                    })
                    .collect();

                (name.to_owned(), strings)
            }));

        assert_eq!(output, expected);
    }

    #[test]
    fn find_simple_indexed_expr() {
        assert_indexed_exprs(IndexedExprs {
            indexes: &["id"],
            expr: "id < 5",
            expected: &[("id", &[&["id < 5"]])],
        })
    }

    #[test]
    fn find_leaf_indexed_expr() {
        assert_indexed_exprs(IndexedExprs {
            indexes: &["id"],
            expr: "id < 5 AND name > 'test' AND age < 30",
            expected: &[("id", &[&["id < 5"]])],
        })
    }

    #[test]
    fn find_multiple_indexed_exprs_with_ands() {
        assert_indexed_exprs(IndexedExprs {
            indexes: &["id"],
            expr: "id < 5 AND name > 'test' AND id < 3",
            expected: &[("id", &[&[("id < 5"), ("id < 3")]])],
        })
    }

    #[test]
    fn find_simple_or_indexed_expr() {
        assert_indexed_exprs(IndexedExprs {
            indexes: &["id"],
            expr: "id < 5 OR id > 10",
            expected: &[("id", &[&["id < 5"], &["id > 10"]])],
        })
    }

    #[test]
    fn find_indexed_expr_and_or() {
        assert_indexed_exprs(IndexedExprs {
            indexes: &["id"],
            expr: "id > 10 AND id < 20 OR id > 50 AND id < 70",
            expected: &[("id", &[&[("id > 10"), ("id < 20")], &[
                ("id > 50"),
                ("id < 70"),
            ]])],
        })
    }

    #[test]
    fn find_indexed_expr_or_and() {
        assert_indexed_exprs(IndexedExprs {
            indexes: &["id"],
            expr: "(id < 20 OR id > 50) AND (id < 100 OR id > 200)",
            expected: &[("id", &[&["id < 20", "id > 50", "id < 100", "id > 200"]])],
        })
    }

    #[test]
    fn find_complicated_and_or_indexed_expr() {
        assert_indexed_exprs(IndexedExprs {
            indexes: &["id"],
            expr: "(id > 10 AND id < 20) OR (id > 50 AND id < 70) OR (id > 100 AND id < 200) OR (id > 500 AND id < 600)",
            expected: &[
                ("id", &[
                    &[("id > 10"), ("id < 20")],
                    &[("id > 50"), ("id < 70")],
                    &[("id > 100"), ("id < 200")],
                    &[("id > 500"), ("id < 600")]
                ])
            ],
        })
    }

    #[test]
    fn dont_use_any_column_if_not_possible() {
        assert_indexed_exprs(IndexedExprs {
            indexes: &["id", "email"],
            expr: "id < 5 OR id > 10 OR email = 'test@test.com'",
            expected: &[],
        })
    }
}
