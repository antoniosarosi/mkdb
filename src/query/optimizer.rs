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

use crate::{
    db::{
        Database, DatabaseContext, DbError, IndexMetadata, Relation, Schema, SqlError,
        TableMetadata,
    },
    paging::io::FileOps,
    sql::{
        parser::Parser,
        statement::{BinaryOperator, Expression, Value},
    },
    storage::{tuple, Cursor},
    vm::plan::{
        Collect, CollectConfig, ExactMatch, Filter, KeyScan, LogicalOrScan, Plan, RangeScan,
        RangeScanConfig, SeqScan, Sort, SortConfig, TuplesComparator,
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
    let source = if let Some(optimized_scan) = generate_optimized_scan_plan(table, db, &filter)? {
        optimized_scan
    } else {
        generate_sequential_scan_plan(table, db)?
    };

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

fn generate_optimized_scan_plan<F: Seek + Read + Write + FileOps>(
    table_name: &str,
    db: &mut Database<F>,
    filter: &Option<Expression>,
) -> Result<Option<Plan<F>>, DbError> {
    let Some(filter) = filter else {
        return Ok(None);
    };

    let table = db.table_metadata(table_name)?.clone();

    // Build index map (column name -> index metadata)
    let indexes = HashSet::from_iter(table.indexes.iter().map(|index| index.name.to_owned()));

    let index_paths = find_index_paths(&table.schema.columns[0].name, &indexes, filter);

    if index_paths.is_empty() {
        return Ok(None);
    }

    let mut indexes_metadata = HashMap::new();

    for index in &table.indexes {
        if index_paths.contains_key(index.name.as_str()) {
            indexes_metadata.insert(index.name.as_str(), index);
        }
    }

    let mut index_ranges = Vec::new();

    for (col, ranges) in index_paths {
        let relation = if let Some(index) = indexes_metadata.get(col).copied() {
            Relation::Index(index.clone())
        } else {
            Relation::Table(table.clone())
        };

        let data_type = table.schema.columns[table.schema.index_of(col).unwrap()].data_type;

        let bounds = ranges.iter().map(|range| {
            let expr = range_to_expr(col, *range);
            let pager = Rc::clone(&db.pager.clone());

            let start = range.0.map(|v| tuple::serialize_key(&data_type, v));
            let end = range.1.map(|v| tuple::serialize_key(&data_type, v));

            if is_exact_match(*range) {
                let Bound::Included(key) = start else {
                    unreachable!();
                };
                Plan::ExactMatch(ExactMatch {
                    key,
                    relation: relation.clone(),
                    expr,
                    done: false,
                    pager,
                    emit_key_only: true,
                })
            } else {
                Plan::RangeScan(RangeScan::from(RangeScanConfig {
                    range: (start, end),
                    relation: relation.clone(),
                    expr,
                    pager,
                    emit_key_only: true,
                }))
            }
        });

        index_ranges.push((col, bounds.collect::<VecDeque<_>>()));
    }

    index_ranges.sort_by_key(|(col, _)| {
        if let Some(index) = indexes_metadata.get(col) {
            index.root
        } else {
            0
        }
    });

    let work_dir = db.work_dir.clone();
    let page_size = db.pager.borrow().page_size;

    let mut source = if index_ranges.len() == 1 {
        let (col, mut ranges) = index_ranges.remove(0);

        if col == table.schema.columns[0].name {
            for r in &mut ranges {
                match r {
                    Plan::RangeScan(s) => s.emit_key_only = false,
                    Plan::ExactMatch(e) => e.emit_key_only = false,
                    _ => unreachable!(),
                }
            }
        }

        let plan = if ranges.len() == 1 {
            ranges.remove(0).unwrap()
        } else {
            Plan::LogicalOrScan(LogicalOrScan { scans: ranges })
        };

        if col == table.schema.columns[0].name {
            return Ok(Some(plan));
        }

        plan
    } else {
        Plan::LogicalOrScan(LogicalOrScan {
            scans: index_ranges
                .into_iter()
                .map(|(_, scan)| scan)
                .flatten()
                .collect(),
        })
    };

    if let Plan::RangeScan(_) | Plan::LogicalOrScan(_) = source {
        source = Plan::Sort(Sort::from(SortConfig {
            page_size,
            work_dir: work_dir.clone(),
            collection: Collect::from(CollectConfig {
                source: Box::new(source),
                work_dir: work_dir.clone(),
                schema: Schema::new(vec![table.schema.columns[0].clone()]),
                mem_buf_size: page_size,
            }),
            comparator: TuplesComparator {
                schema: Schema::new(vec![table.schema.columns[0].clone()]),
                sort_schema: Schema::new(vec![table.schema.columns[0].clone()]),
                sort_keys_indexes: vec![0],
            },
            input_buffers: 4,
        }));
    };

    Ok(Some(Plan::KeyScan(KeyScan {
        comparator: table.comparator()?,
        table,
        pager: Rc::clone(&db.pager),
        source: Box::new(source),
    })))
}

type IndexRangeBounds<'v> = (Bound<&'v Value>, Bound<&'v Value>);

/// Index path finding recursive algorithm inspired by Postgres.
///
/// See the [indxpath.c] file in the Postgres source.
///
/// [indxpath.c]: https://github.com/postgres/postgres/blob/REL_14_STABLE/src/backend/optimizer/path/indxpath.c#L1255
///
/// This algorithm attempts to find the "best" index path that we can follow to
/// retrieve the tuples specified by a query.
///
/// # Base Cases
///
/// The simplest possible case is a single expression with a single index:
///
/// ```sql
/// CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(255));
///
/// SELECT * FROM users WHERE id < 5;
/// ```
///
/// In this case we only need to scan the BTree table using [`RangeScan`] from
/// the beginning of the BTree until key 5 excluded.
///
/// # AND Expressions
///
/// AND clauses allow us to choose either side of the expression tree. Consider
/// this case:
///
/// ```sql
/// CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(255), email VARCHAR(255));
///
/// SELECT * FROM users WHERE id = 5 AND email = 'test@test.com';
/// ```
///
/// We can either use the primary key direct table index or the external email
/// unique index. In this case the primary key would be better because it
/// doesn't require scanning a separate BTree. However, if we tweak the
/// expression a bit:
///
/// ```sql
/// SELECT * FROM users WHERE id < 100 AND email = 'test@test.com';
/// ```
///
/// Now the external email index is better because it will only produce one
/// tuple whereas the key index will produce a range of tuples.
///
/// Regardless of which option we pick, the other one will act as a filter.
/// Same holds true when only one of the branches contains an index. For
/// example:
///
/// ```sql
/// SELECT * FROM users WHERE id < 100 AND name = 'Some Name';
/// ```
///
/// The name column is not indexed, so we'll use the id key index to scan the
/// table BTree and apply the `name = 'Some Name'` filter to returned tuples.
///
/// ## Intersection
///
/// When both sides of the AND clause use the same index they can be intersected
/// to reduce the number of visited tuples:
///
/// ```sql
/// SELECT * FROM users WHERE id > 10 AND id < 20;
/// ```
///
/// In this case a [`RangeScan`] from 10 to 20 will do. Everything else will
/// never be true on both sides of the AND clause. We can also get rid of the
/// [`Filter`] plan completely because the [`RangeScan`] is already filtering
/// out evrything we need.
///
/// Last but not least, we can short circuit if there is no intersection. For
/// instance:
///
/// ```sql
/// SELECT * FROM users WHERE id < 10 AND id > 20;
/// ```
///
/// A number can never be less than 10 while being greater than 20 at the same
/// time, so there's no point in using indexes. We'll fallback to sequential
/// scan in such case and apply the filter even though no tuple will ever
/// evaluate to true.
///
/// # OR Expressions
///
/// OR clauses come with two important characteristics:
///
/// 1. When both branches of the clause contain index columns we must visit them
/// all, we can't pick one or the other. For example:
///
/// ```sql
/// SELECT * FROM users WHERE id < 10 OR email = 'test@test.com';
/// ```
///
/// We can't visit only the id index because `email = 'test@test.com'` might
/// evaluate to true outside of the id range. Both indexes must be visited and
/// the entire expression will act as a [`Filter`].
///
/// 2. If either of the branches contains no indexes the OR clause becomes a
/// sequential scan. Example:
///
/// ```sql
/// SELECT * FROM users WHERE id < 5 OR name < 'c';
/// ```
///
/// The name column is not an index and it might evaluate to true anywhere in
/// the users table, so this becomes a sequential scan.
///
/// ## Union
///
/// Just like AND clauses, when we see the same column on both sides of an OR
/// expression we can compute the union instead of the intersection. Example:
///
/// ```sql
/// SELECT * FROM users WHERE id < 100 OR id < 1000;
/// ```
///
/// This expression doesn't make sense from a logical perspective because it's
/// redundant, but precisly because of that we have to compute the union of the
/// ranges `[..100, ..1000]` and conclude that we only need to scan `..1000`
/// once instead of scanning the same index twice.
///
/// # Combinations Of AND | OR sub-expressions.
///
/// This is where it gets interesting. Consider an expression like this one:
///
/// ```sql
/// SELECT * FROM users WHERE (id < 10 AND email > 't@t.com') OR (id > 100 AND email < 'b@b.com');
/// ```
///
/// What should we do in this case? Well, we can't skip any of the branches
/// because this is an OR statement. Each branch is an AND statement that
/// offers two possible options for scanning, so there are 4 total combinations:
///
/// - Scan id in the range `..10` and email in the range `..'b@b.com'`
/// - Scan id in the range `100..` and email in the range `'t@t.com'..`
/// - Scan id in the ranges `[..10, 100..]`
/// - Scan email in the ranges `[..'b@b.com', 't@t.com'..]`
///
/// Choosing the best combination is very complicated. We discuss
/// "choosing rules" at the end.
///
/// What about this case?
///
/// ```sql
/// SELECT * FROM users WHERE (id < 10 OR email > 't@t.com') AND (id > 1000 OR email < 'b@b.com');
/// ```
///
/// In this case we must pick one of the sides and compute that one while using
/// the other one as a filter, there's no intersection that can help us because
/// one column invalidates the other. The expression could evaluate to true
/// on something like `id = 5, email = 'a@a.com'`, so instead of trying to
/// figure out exactly the best path with the minimum ranges on every single
/// column we'll just prefer simplicity.
///
/// Intersections are only computed on simple cases such as:
///
/// ```sql
/// SELECT * FROM users WHERE id > 5 AND id < 10;
/// SELECT * FROM users WHERE id > 5 AND (id < 20 OR id > 30);
/// ```
///
/// Both sides of the AND must use the same column and one of them must use it
/// only once (no OR), because if it's used multiple times the intersection is
/// harder to compute:
///
/// ```sql
/// SELECT * FROM users WHERE (id < 20 OR id > 50) AND (id < 100 OR id > 200);
/// ```
///
/// In this case we'd have to "multiply everything". You can imagine something
/// like this: `(x + y) * (a + b)`. The result is `xa + xb + ya + yb`. We could
/// definitely do that, but not even production level databases do that. Or at
/// least Postgres doesn't, it just picks one side of the AND branch (from
/// my non-scientific testing, I don't know what I'm doing anyway).
///
/// # AND Rules
///
/// As mentioned earlier, when we find AND expressions that are not simple we
/// have to choose one of the sides and use the other one as a filter. These are
/// the rules that we will follow for choosing:
///
/// 1. If either of the sides contains less columns, return that side because
/// it will produce a simpler query plan that visits less indexes overall.
///
/// 2. If the AND expression is simple enough and we can compute the
/// intersection of the ranges, return a new path that doesn't appear in the
/// original query but is described by the computed intersection.
///
/// 3. When both sides have the same number of columns, prefer exact matches
/// over ranges. Example:
///
/// ```sql
/// SELECT * FROM users WHERE id < 100 AND email = 'test@test.com';
/// ```
///
/// We would choose the right side with the email expression.
///
/// 4. If we can't rely on exact matches, then prefer the table BTree key over
/// external indexes. Example:
///
/// ```sql
/// SELECT * FROM users WHERE id < 100 AND email > 'test@test.com';
/// ```
///
/// We choose the left side because it doesn't require visiting external BTrees.
///
/// These rules will not always choose the "best" path at all, they will simply
/// choose some reliable path without producing an overcomplicated query plan.
/// Choosing the real "best" path would require storing additional metadata
/// about tables such as the number of rows, maximum & minimum key value in each
/// index, etc. With such metadata we could compute cost estimates of visiting
/// indexes and try to pick based on the costs, which is what Postgres does. But
/// even then, considering every single combination of indexes that can be used
/// requires an exponential algorithm, so it's almost impossible to know for
/// sure which one is the "best" path.
fn find_index_paths<'e>(
    key_col: &str,
    indexes: &HashSet<String>,
    expr: &'e Expression,
) -> HashMap<&'e str, VecDeque<IndexRangeBounds<'e>>> {
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
                if (indexes.contains(col) || col == key_col)
                    && matches!(
                        operator,
                        BinaryOperator::Eq
                            | BinaryOperator::Lt
                            | BinaryOperator::LtEq
                            | BinaryOperator::Gt
                            | BinaryOperator::GtEq
                    ) =>
            {
                HashMap::from([(col.as_str(), VecDeque::from([determine_bounds(expr)]))])
            }

            // Case 2: AND | OR expressions.
            (left, right) if matches!(operator, BinaryOperator::And | BinaryOperator::Or) => {
                let mut left_paths = find_index_paths(key_col, indexes, left);
                let mut right_paths = find_index_paths(key_col, indexes, right);

                match operator {
                    // Subcase A: AND expression.
                    //
                    // Either simple AND expressions such as:
                    // SELECT * FROM t WHERE x > 5 AND x < 10
                    //
                    // Or AND clauses involving deeper expression branches:
                    // SELECT * FROM t WHERE (x > 5 AND x < 10) AND (y < 10 OR y > 20)
                    //
                    // The Rules applied to AND clauses are described below and
                    // also in the function documentation.
                    BinaryOperator::And => {
                        // Rule 1: Return the branch with less columns.
                        if right_paths.len() == 0
                            || left_paths.len() > 0 && left_paths.len() < right_paths.len()
                        {
                            return left_paths;
                        }

                        if left_paths.len() == 0
                            || right_paths.len() > 0 && right_paths.len() < left_paths.len()
                        {
                            return right_paths;
                        }

                        // Rule 2: If the expression is simple enough and we
                        // can compute the intersection of the ranges, then
                        // return the intersection.
                        'intersection: {
                            if left_paths.len() > 1 && right_paths.len() > 1 {
                                break 'intersection;
                            };

                            let (col, left_bounds) = left_paths.iter_mut().next().unwrap();
                            let Some(right_bounds) = right_paths.get_mut(col) else {
                                break 'intersection;
                            };

                            if left_bounds.len() > 1 && right_bounds.len() > 1 {
                                break 'intersection;
                            }

                            let factor = if left_bounds.len() == 1 {
                                left_bounds.pop_front().unwrap()
                            } else {
                                right_bounds.pop_front().unwrap()
                            };

                            let mut intersections = VecDeque::new();

                            for range in left_bounds.drain(..).chain(right_bounds.drain(..)) {
                                if let Some(intersection) = range_intersection(factor, range) {
                                    intersections.push_back(intersection);
                                }
                            }

                            // No intersection at all. Something like
                            // id < 5 AND id > 10. Fallback to sequential scan.
                            if intersections.is_empty() {
                                return HashMap::new();
                            }

                            *left_bounds = intersections;

                            return left_paths;
                        };

                        // Rule 3: If a branch is composed of only exact
                        // matches, prefer that branch over ranges. If both
                        // contain exact matches, prefer the one that contains
                        // the primary key.
                        //
                        // Rule 4: Otherwise prefer anything that contains the
                        // primary key or default to the left branch.
                        let left_is_exact_match = left_paths
                            .iter()
                            .all(|(_, bounds)| bounds.iter().copied().all(is_exact_match));

                        let right_is_exact_match = right_paths
                            .iter()
                            .all(|(_, bounds)| bounds.iter().copied().all(is_exact_match));

                        let right_contains_key_col = right_paths.contains_key(key_col);

                        // Check all the cases in which we would prefer the
                        // right branch first. This will allow us to to write
                        // only 2 return statements.
                        if right_is_exact_match && right_contains_key_col
                            || right_is_exact_match && !left_is_exact_match
                            || right_contains_key_col
                                && !left_is_exact_match
                                && !right_contains_key_col
                        {
                            return right_paths;
                        }

                        return left_paths;
                    }

                    // Subcase B: OR expression.
                    //
                    // We only need to compute the union of the ranges in each
                    // column and return the result. For that we need to sort
                    // the ranges first. Picture this convoluted case:
                    //
                    // (id > 10 AND id < 20) OR (id < 40 AND id > 30) OR (id < 25 AND id > 15) OR (id > 35 AND id < 45)
                    //
                    // We would obtain these ranges: [10..20, 30..40, 15..25, 35..45]
                    //
                    // It looks like we can't merge them at first. However, once
                    // we sort them by start bound and then end bound:
                    //
                    // [10..20, 15..25, 30..40, 35..45]
                    //
                    // Now it's pretty clear that [10..20, 15..25] becomes
                    // [10..25] and [30..40, 35..45] becomes [30..45], so we end
                    // up with this range list:
                    //
                    // [10..25, 30..45]
                    //
                    // which makes everything simpler. It also makes this
                    // algorithm quite expensive because we have to visit every
                    // single range of every single column, so it's probably
                    // O(n^2) or a little worse than that. Not 100% sure, didn't
                    // do the math, but probably something like
                    // O(expression_tree_nodes * index_columns * ranges_per_col).
                    //
                    // But expression trees shouldn't be that deep in general
                    // and we shouldn't be working with hundreds of indexes in
                    // one single expression, so this should be fine for general
                    // use cases.
                    BinaryOperator::Or => {
                        if left_paths.is_empty() || right_paths.is_empty() {
                            return HashMap::new();
                        }

                        let mut merged: HashMap<&str, VecDeque<IndexRangeBounds>> = HashMap::new();

                        for (col, mut left_bounds) in left_paths.into_iter() {
                            let Some(mut right_bounds) = right_paths.remove(col) else {
                                merged.insert(col, left_bounds);
                                continue;
                            };

                            let mut sorted_bounds = VecDeque::new();

                            // This is some sort of implicit merge sort. Since
                            // the expression tree is already a "tree" we don't
                            // have to divide anything, we straight up merge.
                            while !left_bounds.is_empty() && !right_bounds.is_empty() {
                                sorted_bounds.push_back(
                                    if cmp_ranges(&left_bounds[0], &right_bounds[0])
                                        != Ordering::Greater
                                    {
                                        left_bounds.pop_front().unwrap()
                                    } else {
                                        right_bounds.pop_front().unwrap()
                                    },
                                );
                            }

                            sorted_bounds.append(&mut left_bounds);
                            sorted_bounds.append(&mut right_bounds);

                            let mut bounds_union = VecDeque::new();

                            for range in sorted_bounds {
                                if let Some(previous) = bounds_union.back_mut() {
                                    if let Some(union) = range_union(*previous, range) {
                                        *previous = union;
                                    } else {
                                        bounds_union.push_back(range);
                                    }
                                } else {
                                    bounds_union.push_back(range);
                                }
                            }

                            // This is basically a sequential scan, doesn't make
                            // sense to use an index.
                            if bounds_union[0] != (Bound::Unbounded, Bound::Unbounded) {
                                merged.insert(col, bounds_union);
                            }
                        }

                        // Push columns that were not present on both sides.
                        merged.extend(right_paths);

                        merged
                    }

                    _ => unreachable!(),
                }
            }

            // Anything else will just fallback to sequential scan.
            _ => HashMap::new(),
        },

        Expression::Nested(inner) => find_index_paths(key_col, indexes, inner),

        // Expressions that are not binary will produce nothing.
        _ => HashMap::new(),
    }
}

/// Transforms a simple binary expression into range bounds.
///
/// The caller must guarantee that `expr` is a simple binary expression with
/// comparison operators. This only exists to avoid further nesting in
/// [`find_index_paths`] and it's only called there.
fn determine_bounds(expr: &Expression) -> (Bound<&Value>, Bound<&Value>) {
    let Expression::BinaryOperation {
        left,
        operator,
        right,
    } = expr
    else {
        unreachable!("determine_bounds() called with non-binary expression: {expr}");
    };

    match (&**left, operator, &**right) {
        // Case 1:
        // SELECT * FROM t WHERE x = 5;
        // SELECT * FROM t WHERE 5 = x;
        //
        // Exact match on a key.
        (Expression::Identifier(_col), BinaryOperator::Eq, Expression::Value(value))
        | (Expression::Value(value), BinaryOperator::Eq, Expression::Identifier(_col)) => {
            (Bound::Included(value), Bound::Included(value))
        }

        // Case 2:
        // SELECT * FROM t WHERE x > 5;
        // SELECT * FROM t WHERE 5 < x;
        //
        // Excluded start bound and unknown end bound.
        (Expression::Identifier(_col), BinaryOperator::Gt, Expression::Value(value))
        | (Expression::Value(value), BinaryOperator::Lt, Expression::Identifier(_col)) => {
            (Bound::Excluded(value), Bound::Unbounded)
        }

        // Case 3:
        // SELECT * FROM t WHERE x < 5;
        // SELECT * FROM t WHERE 5 > x;
        //
        // Unkown start bound and excluded end bound.
        (Expression::Identifier(_col), BinaryOperator::Lt, Expression::Value(value))
        | (Expression::Value(value), BinaryOperator::Gt, Expression::Identifier(_col)) => {
            (Bound::Unbounded, Bound::Excluded(value))
        }

        // Case 4:
        // SELECT * FROM t WHERE x >= 5;
        // SELECT * FROM t WHERE 5 <= x;
        //
        // Included start bound and unknown end bound.
        (Expression::Identifier(_col), BinaryOperator::GtEq, Expression::Value(value))
        | (Expression::Value(value), BinaryOperator::LtEq, Expression::Identifier(_col)) => {
            (Bound::Included(value), Bound::Unbounded)
        }

        // Case 5:
        // SELECT * FROM t WHERE x <= 5;
        // SELECT * FROM t WHERE 5 >= x;
        //
        // Unknown start bound and included end bound.
        (Expression::Identifier(_col), BinaryOperator::LtEq, Expression::Value(value))
        | (Expression::Value(value), BinaryOperator::GtEq, Expression::Identifier(_col)) => {
            (Bound::Unbounded, Bound::Included(value))
        }

        _ => unreachable!("determine_bounds() called with wrong operator: {expr}"),
    }
}

/// Inverse of [`determine_bounds`].
fn range_to_expr(col: &str, (start, end): (Bound<&Value>, Bound<&Value>)) -> Expression {
    let expr = match (start, end) {
        (Bound::Unbounded, Bound::Excluded(v)) => format!("{col} < {v}"),
        (Bound::Unbounded, Bound::Included(v)) => format!("{col} <= {v}"),
        (Bound::Excluded(v), Bound::Unbounded) => format!("{col} > {v}"),
        (Bound::Included(v), Bound::Unbounded) => format!("{col} >= {v}"),
        (Bound::Excluded(v1), Bound::Excluded(v2)) => format!("{col} > {v1} AND {col} < {v2}"),
        (Bound::Included(v1), Bound::Included(v2)) => format!("{col} >= {v1} AND {col} <= {v2}"),
        (Bound::Excluded(v1), Bound::Included(v2)) => format!("{col} > {v1} AND {col} <= {v2}"),
        (Bound::Included(v1), Bound::Excluded(v2)) => format!("{col} >= {v1} AND {col} < {v2}"),
        _ => unreachable!("can't build expr from range {:?}", (start, end)),
    };

    Parser::new(&expr).parse_expression().unwrap()
}

/// Returns true if a range is an exact match like `id = 5`.
fn is_exact_match(range: IndexRangeBounds) -> bool {
    let (Bound::Included(v1), Bound::Included(v2)) = range else {
        return false;
    };

    // See determine_bounds(). We point to the same [`Value`] when building
    // exact match bounds.
    if ptr::eq(v1, v2) {
        return true;
    }

    // Fallback to comparisons if we can't immediately determine that this is
    // an exact match. Cases like (id <= 30) AND (id >= 30) produce an exact
    // match but we point to different values in memory.
    v1 == v2
}

/// Compares two "start bounds".
///
/// [`Bound::Unbounded`] is treated as "0" or initial value, which means it's
/// always [`Ordering::Less`] than anything.
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

/// Compares two "end bounds".
///
/// [`Bound::Unbounded`] is treated as "infinite" or "last value", which means
/// it's always [`Ordering::Greater`] than anything else.
fn cmp_end_bounds(bound1: &Bound<&Value>, bound2: &Bound<&Value>) -> Ordering {
    let ordering = cmp_start_bounds(bound1, bound2);

    if let (Bound::Unbounded, _) | (_, Bound::Unbounded) = (bound1, bound2) {
        return ordering.reverse();
    }

    ordering
}

/// Compares a "start bound" with an "end bound".
fn cmp_start_end_bounds(start: &Bound<&Value>, end: &Bound<&Value>) -> Ordering {
    if *start == Bound::Unbounded {
        return Ordering::Less;
    }

    cmp_end_bounds(start, end)
}

/// Compares two complete ranges and returns an [`Ordering`] variant.
///
/// The rules for comparing are as follows:
///
/// If comparing the start bounds yields [`Ordering::Less`] or [`Ordering::Greater`]
/// then that's the [`Ordering`] of the ranges. On the other hand, if the start
/// bounds are [`Ordering::Equal`] then the [`Ordering`] of the ranges is that
/// of the end bounds. That way ranges can be sorted based on which one "starts"
/// earlier.
fn cmp_ranges(
    (start1, end1): &(Bound<&Value>, Bound<&Value>),
    (start2, end2): &(Bound<&Value>, Bound<&Value>),
) -> Ordering {
    let start_bound_ordering = cmp_start_bounds(start1, start2);

    if cmp_start_bounds(start1, start2) != Ordering::Equal {
        start_bound_ordering
    } else {
        cmp_end_bounds(end1, end2)
    }
}

/// Intersects two ranges together and returns a new range if possible.
///
/// Range don't need to be sorted.
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

/// Computes the union of two ranges and returns a new range only if they can be
/// joined.
///
/// If they can't be joined together then the union is simply both of them. For
/// example, the ranges `[5..15, 10..25]` can be merged to produce `[5..25]`,
/// but the ranges `[5..15, 20..30]` cannot be merged to produce a new one.
/// Ranges must already be sorted.
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

#[cfg(test)]
mod tests {
    use std::{
        collections::{HashMap, HashSet, VecDeque},
        ops::Bound,
    };

    use super::{find_index_paths, IndexRangeBounds};
    use crate::{
        db::{IndexMetadata, Schema},
        sql::{
            parser::Parser,
            statement::{Column, DataType, Expression, Value},
        },
    };

    struct IndexPath<'i> {
        pk: &'i str,
        indexes: &'i [&'i str],
        expr: &'i str,
        expected: HashMap<&'i str, VecDeque<IndexRangeBounds<'i>>>,
    }

    fn assert_find_index_path(
        IndexPath {
            pk,
            indexes,
            expr,
            expected,
        }: IndexPath,
    ) {
        let tree = Parser::new(expr).parse_expression().unwrap();
        let indexes = HashSet::from_iter(indexes.iter().map(|name| String::from(*name)));

        assert_eq!(find_index_paths(pk, &indexes, &tree), expected);
    }

    #[test]
    fn find_simple_key_path() {
        assert_find_index_path(IndexPath {
            pk: "id",
            indexes: &[],
            expr: "id < 5",
            expected: HashMap::from([(
                "id",
                VecDeque::from([(Bound::Unbounded, Bound::Excluded(&Value::Number(5)))]),
            )]),
        })
    }

    #[test]
    fn find_simple_index_path() {
        assert_find_index_path(IndexPath {
            pk: "id",
            indexes: &["email"],
            expr: "email > 'test@test.com'",
            expected: HashMap::from([(
                "email",
                VecDeque::from([(
                    Bound::Excluded(&Value::String("test@test.com".into())),
                    Bound::Unbounded,
                )]),
            )]),
        })
    }

    #[test]
    fn intersect_simple_and_key_path() {
        assert_find_index_path(IndexPath {
            pk: "id",
            indexes: &[],
            expr: "id > 10 AND id < 20",
            expected: HashMap::from([(
                "id",
                VecDeque::from([(
                    Bound::Excluded(&Value::Number(10)),
                    Bound::Excluded(&Value::Number(20)),
                )]),
            )]),
        })
    }

    #[test]
    fn merge_simple_or_key_path() {
        assert_find_index_path(IndexPath {
            pk: "id",
            indexes: &[],
            expr: "id < 10 OR id > 20",
            expected: HashMap::from([(
                "id",
                VecDeque::from([
                    (Bound::Unbounded, Bound::Excluded(&Value::Number(10))),
                    (Bound::Excluded(&Value::Number(20)), Bound::Unbounded),
                ]),
            )]),
        })
    }

    #[test]
    fn short_circuit_when_and_never_evaluates_to_true() {
        assert_find_index_path(IndexPath {
            pk: "id",
            indexes: &[],
            expr: "id < 10 AND id > 20",
            expected: HashMap::new(),
        })
    }

    #[test]
    fn short_circuit_when_or_always_evaluates_to_true() {
        assert_find_index_path(IndexPath {
            pk: "id",
            indexes: &[],
            expr: "id > 10 OR id < 20",
            expected: HashMap::new(),
        })
    }

    #[test]
    fn intersect_multiple_and_key_paths() {
        assert_find_index_path(IndexPath {
            pk: "id",
            indexes: &[],
            expr: "(id > 10 AND id < 30) AND (id > 15 AND id < 45)",
            expected: HashMap::from([(
                "id",
                VecDeque::from([(
                    Bound::Excluded(&Value::Number(15)),
                    Bound::Excluded(&Value::Number(30)),
                )]),
            )]),
        })
    }

    #[test]
    fn merge_multiple_or_key_paths() {
        assert_find_index_path(IndexPath {
            pk: "id",
            indexes: &[],
            expr: "(id < 10 OR id > 30) OR (id > 20 OR id < 5)",
            expected: HashMap::from([(
                "id",
                VecDeque::from([
                    (Bound::Unbounded, Bound::Excluded(&Value::Number(10))),
                    (Bound::Excluded(&Value::Number(20)), Bound::Unbounded),
                ]),
            )]),
        })
    }

    #[test]
    fn intersect_one_and_range_with_multiple_or_ranges() {
        assert_find_index_path(IndexPath {
            pk: "id",
            indexes: &[],
            expr: "id < 20 AND (id < 5 OR id > 10)",
            expected: HashMap::from([(
                "id",
                VecDeque::from([
                    (Bound::Unbounded, Bound::Excluded(&Value::Number(5))),
                    (
                        Bound::Excluded(&Value::Number(10)),
                        Bound::Excluded(&Value::Number(20)),
                    ),
                ]),
            )]),
        })
    }

    #[test]
    fn merge_multiple_indexes_on_or_clause() {
        assert_find_index_path(IndexPath {
            pk: "id",
            indexes: &["email", "uuid"],
            expr: "id = 5 OR email = 'test@test.com' OR uuid <= 'ffaa'",
            expected: HashMap::from([
                (
                    "id",
                    VecDeque::from([(
                        Bound::Included(&Value::Number(5)),
                        Bound::Included(&Value::Number(5)),
                    )]),
                ),
                (
                    "email",
                    VecDeque::from([(
                        Bound::Included(&Value::String("test@test.com".into())),
                        Bound::Included(&Value::String("test@test.com".into())),
                    )]),
                ),
                (
                    "uuid",
                    VecDeque::from([(
                        Bound::Unbounded,
                        Bound::Included(&Value::String("ffaa".into())),
                    )]),
                ),
            ]),
        })
    }

    #[test]
    fn choose_key_range_over_index_range() {
        assert_find_index_path(IndexPath {
            pk: "id",
            indexes: &["email"],
            expr: "id < 10 AND email < 'test@test.com'",
            expected: HashMap::from([(
                "id",
                VecDeque::from([(Bound::Unbounded, Bound::Excluded(&Value::Number(10)))]),
            )]),
        })
    }

    #[test]
    fn choose_index_exact_match_over_key_range() {
        assert_find_index_path(IndexPath {
            pk: "id",
            indexes: &["email"],
            expr: "id < 10 AND email = 'test@test.com'",
            expected: HashMap::from([(
                "email",
                VecDeque::from([(
                    Bound::Included(&Value::String("test@test.com".into())),
                    Bound::Included(&Value::String("test@test.com".into())),
                )]),
            )]),
        })
    }

    #[test]
    fn choose_key_exact_match_over_index_exact_match() {
        assert_find_index_path(IndexPath {
            pk: "id",
            indexes: &["email"],
            expr: "email = 'test@test.com' AND id = 10",
            expected: HashMap::from([(
                "id",
                VecDeque::from([(
                    Bound::Included(&Value::Number(10)),
                    Bound::Included(&Value::Number(10)),
                )]),
            )]),
        })
    }

    #[test]
    fn choose_branch_with_less_columns() {
        assert_find_index_path(IndexPath {
            pk: "id",
            indexes: &["email", "uuid", "test", "col", "idk"],
            expr: "(idk < 5 OR col > 10 OR test = 'test') AND (email = 'test@test.com' OR uuid >= 'ffaa')",
            expected: HashMap::from([
                (
                    "email",
                    VecDeque::from([(
                        Bound::Included(&Value::String("test@test.com".into())),
                        Bound::Included(&Value::String("test@test.com".into())),
                    )]),
                ),
                (
                    "uuid",
                    VecDeque::from([(
                        Bound::Included(&Value::String("ffaa".into())),
                        Bound::Unbounded,
                    )]),
                ),
            ]),
        })
    }
}
