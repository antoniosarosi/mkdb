//! Generates optimized plans.
//!
//! See the module level documentation of [`crate::vm::plan`].

use std::{
    cmp::{self, Ordering},
    collections::{BinaryHeap, HashMap, HashSet, VecDeque},
    io::{Read, Seek, Write},
    iter, mem,
    ops::{Bound, RangeBounds},
    ptr,
    rc::Rc,
};

use libc::group;

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

/// Finds all the combinations of expressions applied to indexed columns.
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
/// If an `OR` expression contains another indexed column we can use two
/// indexes:
///
/// ```sql
/// SELECT * FROM users WHERE id < 5 OR email = "test@test.com";
/// ```
///
/// So this function ends up producing "groups" of expressions that we can use.
/// Each "group" is separated by an `OR` expression and the groups themselves
/// can contain multiple expressions for multiple indexes. See the tests below
/// in the [`tests`] module for examples.
fn find_indexed_exprs<'e>(
    indexes: &HashSet<String>,
    expr: &'e Expression,
) -> Vec<HashMap<&'e str, Vec<&'e Expression>>> {
    println!("{expr}");

    match expr {
        Expression::BinaryOperation {
            left,
            operator,
            right,
        } => match (&**left, &**right) {
            // Case 1: Base case.
            //
            // SELECT * FROM t WHERE x >= 5
            //
            // These are the leaves of the expression tree, we start from here
            // and build upwards.
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
                vec![HashMap::from([(col.as_str(), vec![expr])])]
            }

            // Case 2: AND | OR expressions.
            (left, right) if matches!(operator, BinaryOperator::And | BinaryOperator::Or) => {
                let mut left_groups = find_indexed_exprs(indexes, left);
                let mut right_groups = find_indexed_exprs(indexes, right);

                match operator {
                    // Subcase A: AND expression.
                    //
                    // We have to join all the groups into one single group to
                    // produce the intersection.
                    //
                    // This algorithm is O(n^2) or worse (didn't do the math,
                    // probably something like depth of expr tree * number of
                    // groups * number of indexes), but query trees in general
                    // shouldn't be that deep and tables in general won't have
                    // hundreds of indexes to consider. Should be safe for
                    // general use cases.
                    BinaryOperator::And => {
                        let mut intersection: HashMap<&str, Vec<&Expression>> = HashMap::new();

                        for group in left_groups.into_iter().chain(right_groups.into_iter()) {
                            for (column, mut bounds) in group.into_iter() {
                                if let Some(existing) = intersection.get_mut(&column) {
                                    existing.append(&mut bounds);
                                } else {
                                    intersection.insert(column, bounds);
                                }
                            }
                        }

                        vec![intersection]
                    }

                    // Subcase B: OR expression.
                    //
                    // In this case we append the groups of the right side to
                    // the groups of the left side to produce the union. If
                    // either of the sides contains no groups then we return
                    // nothing because one side of the OR statement invalidates
                    // the indexes found on the other.
                    BinaryOperator::Or => {
                        if left_groups.is_empty() || right_groups.is_empty() {
                            return vec![];
                        }

                        left_groups.append(&mut right_groups);

                        left_groups
                    }
                    _ => unreachable!(),
                }
            }

            // Anything else produces 0 groups, which means there is no indexed
            // expression that can be used.
            _ => vec![],
        },

        Expression::Nested(inner) => find_indexed_exprs(indexes, inner),

        // Expressions that are not binary will produce 0 groups as well.
        _ => vec![],
    }
}

fn cmp_start_bounds(bound1: &Bound<&Value>, bound2: &Bound<&Value>) -> Ordering {
    match (bound1, bound2) {
        (Bound::Unbounded, Bound::Unbounded) => Ordering::Equal,
        (Bound::Unbounded, _) => Ordering::Less,
        (_, Bound::Unbounded) => Ordering::Greater,
        (
            Bound::Excluded(value1) | Bound::Included(value1),
            Bound::Excluded(value2) | Bound::Included(value2),
        ) => {
            let ordering = value1.partial_cmp(value2).unwrap_or_else(|| {
                panic!(
                    "Type errors at this point should be impossible: cmp {value1} against {value2}"
                )
            });

            if ordering != Ordering::Equal {
                return ordering;
            }

            match (bound1, bound2) {
                (Bound::Included(_), Bound::Excluded(_)) => Ordering::Less,
                (Bound::Excluded(_), Bound::Included(_)) => Ordering::Greater,
                _ => Ordering::Equal,
            }
        }
    }
}

fn cmp_end_bounds(bound1: &Bound<&Value>, bound2: &Bound<&Value>) -> Ordering {
    let ordering = cmp_start_bounds(bound1, bound2);

    if let (Bound::Unbounded, _) | (_, Bound::Unbounded) = (bound1, bound2) {
        return ordering.reverse();
    }

    ordering
}

fn cmp_start_end_bounds(start: &Bound<&Value>, end: &Bound<&Value>) -> Ordering {
    if *start == Bound::Unbounded {
        return Ordering::Less;
    }

    cmp_end_bounds(start, end)
}

fn range_intersection<'v>(
    (start1, end1): (Bound<&'v Value>, Bound<&'v Value>),
    (start2, end2): (Bound<&'v Value>, Bound<&'v Value>),
) -> Option<(Bound<&'v Value>, Bound<&'v Value>)> {
    let intersection_start = cmp::max_by(start1, start2, cmp_start_bounds);
    let intersection_end = cmp::min_by(end1, end2, cmp_end_bounds);

    if cmp_start_end_bounds(&intersection_start, &intersection_end) == Ordering::Greater {
        return None;
    }

    Some((intersection_start, intersection_end))
}

fn range_union<'v>(
    (start1, end1): (Bound<&'v Value>, Bound<&'v Value>),
    (start2, end2): (Bound<&'v Value>, Bound<&'v Value>),
) -> Option<(Bound<&'v Value>, Bound<&'v Value>)> {
    debug_assert!(
        cmp_start_bounds(&start1, &start2) != Ordering::Greater,
        "ranges should be sorted at this point to reduce comparisons: {:?} > {:?}",
        (start1, end1),
        (start2, end2),
    );

    if cmp_start_end_bounds(&start2, &end1) == Ordering::Greater {
        return None;
    }

    let union_start = cmp::min_by(start1, start2, cmp_start_bounds);
    let union_end = cmp::max_by(end1, end2, cmp_end_bounds);

    Some((union_start, union_end))
}

fn cmp_ranges(
    (start1, end1): &(Bound<&Value>, Bound<&Value>),
    (start2, end2): &(Bound<&Value>, Bound<&Value>),
) -> Ordering {
    let start_bound_ordering = cmp_start_bounds(start1, start2);

    if cmp_start_bounds(&start1, &start2) != Ordering::Equal {
        start_bound_ordering
    } else {
        cmp_end_bounds(end1, end2)
    }
}

/// Computes the minimum set of bounds that have to be checked for each index.
///
/// Again, this algorithm is pretty expensive but we shouldn't be working many
/// indexes at once. If we implement JOINs then large queries that join many
/// tables together using different indexes might run into some performance
/// issues, but then we can probably implement some heuristics or figure out
/// if this computation is worth it.
fn fold_range_bounds<'g>(
    groups: &'g [HashMap<&str, Vec<&Expression>>],
) -> Vec<HashMap<&'g str, Vec<(Bound<&'g Value>, Bound<&'g Value>)>>> {
    // Compute the intersections of all ranges first.
    let mut fold = groups
        .iter()
        .map(|group| {
            group.iter().filter_map(|(col, exprs)| {
                let mut bounds = exprs.iter().copied().map(determine_bounds);
                let mut intersection = bounds.next()?;

                // Short circuit if AND expression produces no intersection.
                // That means the AND expression will never evaluate to true.
                // Something like this for example: id < 5 AND id > 10.
                for range in bounds {
                    intersection = range_intersection(intersection, range)?;
                }

                Some((*col, vec![intersection]))
            })
        })
        .map(HashMap::from_iter)
        .filter(|map| !map.is_empty())
        .collect::<Vec<_>>();

    // Now compute the union of all individual columns.
    for i in 0..fold.len().saturating_sub(1) {
        let mut merge: HashMap<&str, VecDeque<(Bound<&Value>, Bound<&Value>)>> = HashMap::new();

        for j in (i + 1)..fold.len() {
            let [group, next] = fold.get_many_mut([i, j]).unwrap();
            for (col, ranges) in group.iter_mut() {
                if let Some(intersection) = next.remove(*col) {
                    if let Some(existing) = merge.get_mut(col) {
                        existing.extend(intersection);
                    } else {
                        merge.insert(
                            *col,
                            VecDeque::from_iter(ranges.drain(..).chain(intersection.into_iter())),
                        );
                    }
                }
            }
        }

        let group = &mut fold[i];

        for (col, mut ranges) in merge.into_iter() {
            let unions = group.get_mut(col).unwrap();

            ranges.make_contiguous().sort_by(cmp_ranges);

            while let Some(range) = ranges.pop_front() {
                if let Some(last) = unions.last_mut() {
                    if let Some(union) = range_union(*last, range) {
                        *last = union;
                    } else {
                        unions.push(range);
                    }
                } else {
                    unions.push(range);
                }
                println!("{unions:?}");
            }
        }
    }

    fold.retain(|range| !range.is_empty());

    fold
}

/// Returns the indexed columns that should be used in each group.
///
/// For now, this only attempts to use the primary key as much as possible
/// because it doesn't require an external index. But if we had more metadata
/// about tables such as the current number of rows, maximum/minimum key value
/// in each index, etc, we could determine if it's actually worth it to use an
/// index or fallback to sequential scan like Postgres does for example.
///
/// If the primary key can't be used then we'll resort to the index that appears
/// the most amount of times throughout the groups. That's not the best index
/// to use, it's just the index that will probably produce the simplest query
/// plan. Considering every single combination of indexes requires an
/// exponential algorithm.
fn find_best_path<'r>(
    key_col: &str,
    indexes_range_bounds: &'r [HashMap<&str, Vec<(Bound<&Value>, Bound<&Value>)>>],
) -> Vec<&'r str> {
    debug_assert!(
        !indexes_range_bounds.is_empty(),
        "find_best_path() called with empty vec"
    );

    let mut index_frequency = HashMap::new();

    indexes_range_bounds
        .iter()
        .flat_map(|group| group.iter())
        .for_each(|(col, _)| {
            if let Some(frequency) = index_frequency.get_mut(col) {
                *frequency += 1;
            } else {
                index_frequency.insert(col, 1);
            };
        });

    println!("\n{index_frequency:?}");

    indexes_range_bounds
        .iter()
        .map(|group| {
            if let Some((col, _)) = group.get_key_value(key_col) {
                return *col;
            }

            group
                .iter()
                .max_by(|(col1, _), (col2, _)| index_frequency[*col1].cmp(&index_frequency[*col2]))
                .map(|(col, _)| col)
                .unwrap_or_else(|| panic!("find_best_path() called with empty group: {group:?}"))
        })
        .collect()
}

fn determine_bounds(expr: &Expression) -> (Bound<&Value>, Bound<&Value>) {
    let Expression::BinaryOperation {
        left,
        operator,
        right,
    } = expr
    else {
        unreachable!();
    };

    match (&**left, operator, &**right) {
        // Case 1:
        // SELECT * FROM t WHERE x = 5;
        // SELECT * FROM t WHERE 5 = x;
        //
        // Position the cursor at key 5 and stop after returning that key.
        (Expression::Identifier(_col), BinaryOperator::Eq, Expression::Value(value))
        | (Expression::Value(value), BinaryOperator::Eq, Expression::Identifier(_col)) => {
            (Bound::Included(value), Bound::Included(value))
        }

        // Case 2:
        // SELECT * FROM t WHERE x > 5;
        // SELECT * FROM t WHERE 5 < x;
        //
        // Position the cursor at key 5, consume key 5 which will cause the
        // cursor to move to the successor and return keys until finished.
        (Expression::Identifier(_col), BinaryOperator::Gt, Expression::Value(value))
        | (Expression::Value(value), BinaryOperator::Lt, Expression::Identifier(_col)) => {
            (Bound::Excluded(value), Bound::Unbounded)
        }

        // Case 3:
        // SELECT * FROM t WHERE x < 5;
        // SELECT * FROM t WHERE 5 > x;
        //
        // Allow the cursor to initialize normally (at the smallest key) and
        // tell it to stop once it finds a key >= 5.
        (Expression::Identifier(_col), BinaryOperator::Lt, Expression::Value(value))
        | (Expression::Value(value), BinaryOperator::Gt, Expression::Identifier(_col)) => {
            (Bound::Unbounded, Bound::Excluded(value))
        }

        // Case 4:
        // SELECT * FROM t WHERE x >= 5;
        // SELECT * FROM t WHERE 5 <= x;
        //
        // Position the cursor at key 5 and assume it's already initialized. The
        // cursor will then return key 5 and everything after.
        (Expression::Identifier(_col), BinaryOperator::GtEq, Expression::Value(value))
        | (Expression::Value(value), BinaryOperator::LtEq, Expression::Identifier(_col)) => {
            (Bound::Included(value), Bound::Unbounded)
        }

        // Case 5:
        // SELECT * FROM t WHERE x <= 5;
        // SELECT * FROM t WHERE 5 >= x;
        //
        // Allow the cursor to initialize normally (at the smallest key) and
        // tell it to stop once it finds a key > 5
        (Expression::Identifier(_col), BinaryOperator::LtEq, Expression::Value(value))
        | (Expression::Value(value), BinaryOperator::GtEq, Expression::Identifier(_col)) => {
            (Bound::Unbounded, Bound::Included(value))
        }

        // Fallback to sequential scan in case we mess up here trying to
        // optimize the expressions.
        _ => unreachable!(),
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::{HashMap, HashSet},
        ops::Bound,
    };

    use super::find_indexed_exprs;
    use crate::{
        db::{IndexMetadata, Schema},
        query::optimizer::{find_best_path, fold_range_bounds},
        sql::{
            parser::Parser,
            statement::{Column, DataType, Expression, Value},
        },
    };

    struct IndexedExprs<'i> {
        indexes: &'i [&'i str],
        expr: &'i str,
        expected: Vec<HashMap<&'i str, Vec<&'i str>>>,
    }

    fn parse_indexed_expr(expr: &str, indexes: &[&str]) -> (Expression, HashSet<String>) {
        let tree = Parser::new(expr).parse_expression().unwrap();
        let indexes = HashSet::from_iter(indexes.iter().map(|name| String::from(*name)));

        (tree, indexes)
    }

    fn assert_indexed_exprs(
        IndexedExprs {
            indexes,
            expr,
            expected,
        }: IndexedExprs,
    ) {
        let (tree, indexes) = parse_indexed_expr(expr, indexes);
        let expressions = find_indexed_exprs(&indexes, &tree);

        let output: Vec<HashMap<String, Vec<String>>> = expressions
            .iter()
            .map(|group| {
                group
                    .into_iter()
                    .map(|(col, exprs)| {
                        let expr_strings = exprs.iter().map(|expr| expr.to_string()).collect();
                        (col.to_string(), expr_strings)
                    })
                    .collect()
            })
            .collect();

        let expected: Vec<HashMap<String, Vec<String>>> = expected
            .iter()
            .map(|group| {
                group
                    .into_iter()
                    .map(|(col, exprs)| {
                        let expr_strings = exprs.iter().map(|expr| expr.to_string()).collect();
                        (col.to_string(), expr_strings)
                    })
                    .collect()
            })
            .collect();

        assert_eq!(output, expected);
    }

    struct BestPath<'i> {
        pk: &'i str,
        indexes: &'i [&'i str],
        expr: &'i str,
        expected: &'i [&'i str],
    }

    fn assert_best_path(
        BestPath {
            pk,
            indexes,
            expr,
            expected,
        }: BestPath,
    ) {
        let (tree, indexes) = parse_indexed_expr(expr, indexes);

        assert_eq!(
            find_best_path(pk, &fold_range_bounds(&find_indexed_exprs(&indexes, &tree))),
            expected
        )
    }

    struct BuildBounds<'i> {
        indexes: &'i [&'i str],
        expr: &'i str,
        expected: Vec<HashMap<&'i str, Vec<(Bound<&'i Value>, Bound<&'i Value>)>>>,
    }

    fn assert_build_range_bounds(
        BuildBounds {
            indexes,
            expr,
            expected,
        }: BuildBounds,
    ) {
        let (tree, indexes) = parse_indexed_expr(expr, indexes);

        assert_eq!(
            &fold_range_bounds(&find_indexed_exprs(&indexes, &tree)),
            &expected
        );
    }

    #[test]
    fn find_simple_indexed_expr() {
        assert_indexed_exprs(IndexedExprs {
            indexes: &["id"],
            expr: "id < 5",
            expected: vec![HashMap::from([("id", vec!["id < 5"])])],
        })
    }

    #[test]
    fn find_leaf_indexed_expr() {
        assert_indexed_exprs(IndexedExprs {
            indexes: &["id"],
            expr: "id < 5 AND name > 'test' AND age < 30",
            expected: vec![HashMap::from([("id", vec!["id < 5"])])],
        })
    }

    #[test]
    fn find_simple_and_indexed_expr() {
        assert_indexed_exprs(IndexedExprs {
            indexes: &["id"],
            expr: "id > 5 AND id < 10",
            expected: vec![HashMap::from([("id", vec![("id > 5"), ("id < 10")])])],
        })
    }

    #[test]
    fn find_simple_or_indexed_expr() {
        assert_indexed_exprs(IndexedExprs {
            indexes: &["id"],
            expr: "id < 5 OR id > 10",
            expected: vec![
                HashMap::from([("id", vec!["id < 5"])]),
                HashMap::from([("id", vec!["id > 10"])]),
            ],
        })
    }

    #[test]
    fn find_distributed_and_indexed_exprs() {
        assert_indexed_exprs(IndexedExprs {
            indexes: &["id"],
            expr: "id < 5 AND name > 'test' AND id < 3",
            expected: vec![HashMap::from([("id", vec![("id < 5"), ("id < 3")])])],
        })
    }

    #[test]
    fn find_indexed_expr_and_or() {
        assert_indexed_exprs(IndexedExprs {
            indexes: &["id"],
            expr: "(id > 10 AND id < 20) OR (id > 50 AND id < 70)",
            expected: vec![
                HashMap::from([("id", vec![("id > 10"), ("id < 20")])]),
                HashMap::from([("id", vec![("id > 50"), ("id < 70")])]),
            ],
        })
    }

    #[test]
    fn find_indexed_expr_or_and() {
        assert_indexed_exprs(IndexedExprs {
            indexes: &["id"],
            expr: "(id < 20 OR id > 50) AND (id < 100 OR id > 200)",
            expected: vec![HashMap::from([("id", vec![
                ("id < 20"),
                ("id > 50"),
                ("id < 100"),
                ("id > 200"),
            ])])],
        })
    }

    #[test]
    fn find_multiple_and_or_indexed_expr() {
        assert_indexed_exprs(IndexedExprs {
            indexes: &["id"],
            expr: "(id > 10 AND id < 20) OR (id > 50 AND id < 70) OR (id > 100 AND id < 200) OR (id > 500 AND id < 600)",
            expected: vec![
                HashMap::from([("id", vec![("id > 10"), ("id < 20")])]),
                HashMap::from([("id", vec![("id > 50"), ("id < 70")])]),
                HashMap::from([("id", vec![("id > 100"), ("id < 200")])]),
                HashMap::from([("id", vec![("id > 500"), ("id < 600")])]),
            ],
        })
    }

    #[test]
    fn find_multiple_indexed_columns_and() {
        assert_indexed_exprs(IndexedExprs {
            indexes: &["id", "email", "uuid"],
            expr: "id < 5 AND uuid < 'ffff' AND email = 'test@test.com'",
            expected: vec![HashMap::from([
                ("id", vec!["id < 5"]),
                ("uuid", vec![r#"uuid < "ffff""#]),
                ("email", vec![r#"email = "test@test.com""#]),
            ])],
        })
    }

    #[test]
    fn find_multiple_indexed_columns_or() {
        assert_indexed_exprs(IndexedExprs {
            indexes: &["id", "email"],
            expr: "id < 5 OR id > 10 OR email = 'test@test.com'",
            expected: vec![
                HashMap::from([("id", vec!["id < 5"])]),
                HashMap::from([("id", vec!["id > 10"])]),
                HashMap::from([("email", vec![r#"email = "test@test.com""#])]),
            ],
        })
    }

    #[test]
    fn short_circuit_on_or_expr() {
        assert_indexed_exprs(IndexedExprs {
            indexes: &["id", "email"],
            expr: "id > 5 AND id < 10 OR name = 'Some Name'",
            expected: vec![],
        })
    }

    #[test]
    fn build_simple_range_bound() {
        assert_build_range_bounds(BuildBounds {
            indexes: &["id"],
            expr: "id > 5",
            expected: vec![HashMap::from([("id", vec![(
                Bound::Excluded(&Value::Number(5)),
                Bound::Unbounded,
            )])])],
        })
    }

    #[test]
    fn build_exact_match_range_bound() {
        assert_build_range_bounds(BuildBounds {
            indexes: &["id"],
            expr: "id = 5",
            expected: vec![HashMap::from([("id", vec![(
                Bound::Included(&Value::Number(5)),
                Bound::Included(&Value::Number(5)),
            )])])],
        })
    }

    #[test]
    fn build_simple_and_range_bound() {
        assert_build_range_bounds(BuildBounds {
            indexes: &["id"],
            expr: "id > 5 AND id <= 10",
            expected: vec![HashMap::from([("id", vec![(
                Bound::Excluded(&Value::Number(5)),
                Bound::Included(&Value::Number(10)),
            )])])],
        })
    }

    #[test]
    fn build_simple_or_range_bound() {
        assert_build_range_bounds(BuildBounds {
            indexes: &["id"],
            expr: "id <= 5 OR id > 10",
            expected: vec![HashMap::from([("id", vec![
                (Bound::Unbounded, Bound::Included(&Value::Number(5))),
                (Bound::Excluded(&Value::Number(10)), Bound::Unbounded),
            ])])],
        })
    }

    #[test]
    fn short_circuit_on_no_intersection() {
        assert_build_range_bounds(BuildBounds {
            indexes: &["id"],
            expr: "id < 5 AND id > 10",
            expected: vec![],
        })
    }

    #[test]
    fn intersect_and_range_bounds() {
        assert_build_range_bounds(BuildBounds {
            indexes: &["id"],
            expr: "(id > 5 AND id < 20) AND (id > 15 AND id < 25)",
            expected: vec![HashMap::from([("id", vec![(
                Bound::Excluded(&Value::Number(15)),
                Bound::Excluded(&Value::Number(20)),
            )])])],
        })
    }

    #[test]
    fn join_or_range_bounds() {
        assert_build_range_bounds(BuildBounds {
            indexes: &["id"],
            expr: "(id < 10 OR id > 20) OR (id < 15 OR id > 25)",
            expected: vec![HashMap::from([("id", vec![
                (Bound::Unbounded, Bound::Excluded(&Value::Number(15))),
                (Bound::Excluded(&Value::Number(20)), Bound::Unbounded),
            ])])],
        })
    }

    #[test]
    fn build_range_bound_multiple_columns_and() {
        assert_build_range_bounds(BuildBounds {
            indexes: &["id", "email"],
            expr: "id <= 5 AND email = 'test@test.com'",
            expected: vec![HashMap::from([
                ("id", vec![(
                    Bound::Unbounded,
                    Bound::Included(&Value::Number(5)),
                )]),
                ("email", vec![(
                    Bound::Included(&Value::String("test@test.com".into())),
                    Bound::Included(&Value::String("test@test.com".into())),
                )]),
            ])],
        })
    }

    #[test]
    fn build_range_bound_multiple_columns_or() {
        assert_build_range_bounds(BuildBounds {
            indexes: &["id", "email"],
            expr: "id <= 5 OR email = 'test@test.com'",
            expected: vec![
                HashMap::from([("id", vec![(
                    Bound::Unbounded,
                    Bound::Included(&Value::Number(5)),
                )])]),
                HashMap::from([("email", vec![(
                    Bound::Included(&Value::String("test@test.com".into())),
                    Bound::Included(&Value::String("test@test.com".into())),
                )])]),
            ],
        })
    }

    #[test]
    fn intersect_and_bounds_with_multiple_columns() {
        assert_build_range_bounds(BuildBounds {
            indexes: &["id", "email"],
            expr: "(id > 5 AND id < 20) AND (id > 15 AND id < 25) AND email < 'test@test.com'",
            expected: vec![HashMap::from([
                ("id", vec![(
                    Bound::Excluded(&Value::Number(15)),
                    Bound::Excluded(&Value::Number(20)),
                )]),
                ("email", vec![(
                    Bound::Unbounded,
                    Bound::Excluded(&Value::String("test@test.com".into())),
                )]),
            ])],
        })
    }

    #[test]
    fn join_or_bounds_with_multiple_columns() {
        assert_build_range_bounds(BuildBounds {
            indexes: &["id", "email"],
            expr: "(id > 10 AND id < 30) OR (id > 5 AND id < 20) AND (email < 'test@test.com' OR email > 'a')",
            expected: vec![
                HashMap::from([("id", vec![(
                    Bound::Excluded(&Value::Number(5)),
                    Bound::Excluded(&Value::Number(30)),
                )])]),
                HashMap::from([("email", vec![(
                    Bound::Excluded(&Value::String("a".into())),
                    Bound::Excluded(&Value::String("test@test.com".into())),
                )])]),
            ],
        })
    }

    #[test]
    fn or_bounds_union_with_multiple_columns() {
        assert_build_range_bounds(BuildBounds {
            expr: "id > 5 OR (email > 'a' AND email < 'c') OR (email = 'test' AND uuid = 'aaaa' AND name = 'what') OR (name = 'test')",
            indexes: &["id", "email", "uuid", "name"],
            expected: vec![
                HashMap::from([("id", vec![(
                    Bound::Excluded(&Value::Number(5)),
                    Bound::Unbounded,
                )])]),
                HashMap::from([("email", vec![
                    (
                        Bound::Excluded(&Value::String("a".into())),
                        Bound::Excluded(&Value::String("c".into())),
                    ),
                    (
                        Bound::Included(&Value::String("test".into())),
                        Bound::Included(&Value::String("test".into())),
                    ),
                ])]),
                HashMap::from([
                    ("uuid", vec![(
                        Bound::Included(&Value::String("aaaa".into())),
                        Bound::Included(&Value::String("aaaa".into())),
                    )]),
                    ("name", vec![
                        (
                            Bound::Included(&Value::String("test".into())),
                            Bound::Included(&Value::String("test".into())),
                        ),
                        (
                            Bound::Included(&Value::String("what".into())),
                            Bound::Included(&Value::String("what".into())),
                        ),
                    ]),
                ]),
            ],
        })
    }

    #[test]
    fn simple_best_path() {
        assert_best_path(BestPath {
            pk: "id",
            indexes: &["id", "email"],
            expr: "id > 5",
            expected: &["id"],
        })
    }

    #[test]
    fn best_path_or() {
        assert_best_path(BestPath {
            pk: "id",
            indexes: &["id", "email"],
            expr: "id < 5 OR id > 10",
            expected: &["id"],
        })
    }

    #[test]
    fn best_path_and() {
        assert_best_path(BestPath {
            pk: "id",
            indexes: &["id", "email"],
            expr: "id > 5 AND id < 10",
            expected: &["id"],
        })
    }

    #[test]
    fn best_path_and_two_columns() {
        assert_best_path(BestPath {
            pk: "id",
            indexes: &["id", "email"],
            expr: "id > 5 AND email = 'test@test.com'",
            expected: &["id"],
        })
    }

    #[test]
    fn best_path_or_two_columns() {
        assert_best_path(BestPath {
            pk: "id",
            indexes: &["id", "email"],
            expr: "id > 5 OR email = 'test@test.com'",
            expected: &["id", "email"],
        })
    }

    #[test]
    fn best_path_or_multiple_columns() {
        assert_best_path(BestPath {
            pk: "id",
            indexes: &["id", "email", "uuid", "name"],
            expr: "id > 5 OR (email > 'a' AND email < 'c') OR (email = 'test' AND uuid = 'aaaa' AND name = 'what') OR (name = 'test')",
            expected: &["id", "email", "name"],
        })
    }
}
