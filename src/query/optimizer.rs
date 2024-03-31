//! Generates optimized plans.
//!
//! See the module level documentation of [`crate::vm::plan`].

use std::{
    collections::HashMap,
    io::{Read, Seek, Write},
    ops::Bound,
    rc::Rc,
};

use crate::{
    db::{Database, DatabaseContext, DbError, IndexMetadata, Relation, SqlError, TableMetadata},
    paging::io::FileOps,
    sql::statement::{BinaryOperator, Expression},
    storage::{tuple, Cursor},
    vm::plan::{
        BufferedIter, BufferedIterConfig, ExactMatch, Filter, KeyScan, Plan, RangeScan,
        RangeScanConfig, Scan, SeqScan, Sort, SortConfig, TuplesComparator,
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
    let Some(expr) = filter else {
        return generate_sequential_scan_plan(table, db);
    };

    let source = if let Some(optimized_scan) = generate_optimized_scan_plan(table, db, &expr)? {
        optimized_scan
    } else {
        generate_sequential_scan_plan(table, db)?
    };

    if !needs_filter(&source, &expr) {
        return Ok(source);
    }

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

/// Attempts to generate a [`RangeScan`] or an [`IndexScan`] plan.
///
/// It's only possible to do so if we find an expression that contains an
/// indexed column or the table primary key column and must always be executed.
/// Otherwise we'll fallback to sequential scans.
fn generate_optimized_scan_plan<F: Seek + Read + Write + FileOps>(
    table_name: &str,
    db: &mut Database<F>,
    filter: &Expression,
) -> Result<Option<Plan<F>>, DbError> {
    let table = db.table_metadata(table_name)?.clone();

    // Build index map (column name -> index metadata)
    let mut indexes = HashMap::from_iter(
        table
            .indexes
            .iter()
            .map(|index| (index.column.name.clone(), index.clone())),
    );

    // Find expression that's using an indexed column or the primary key.
    let Some(Expression::BinaryOperation {
        left,
        operator,
        right,
    }) = find_indexed_expr(&table, &indexes, filter)
    else {
        return Ok(None);
    };

    let (key, col, relation) = match (left.as_ref(), right.as_ref()) {
        (Expression::Identifier(col), Expression::Value(value))
        | (Expression::Value(value), Expression::Identifier(col)) => {
            let index_col = table
                .schema
                .index_of(col)
                .ok_or(SqlError::InvalidColumn(col.into()))?;

            let key = tuple::serialize_key(&table.schema.columns[index_col].data_type, value);

            let relation = if let Some(index) = indexes.get(col) {
                Relation::Index(index.clone())
            } else {
                Relation::Table(table.clone())
            };

            (key, index_col, relation)
        }

        _ => unreachable!(),
    };

    let scan = match (left.as_ref(), operator, right.as_ref()) {
        // Case 1:
        // SELECT * FROM t WHERE x = 5;
        // SELECT * FROM t WHERE 5 = x;
        //
        // Position the cursor at key 5 and stop after returning that key.
        (Expression::Identifier(_col), BinaryOperator::Eq, Expression::Value(_value))
        | (Expression::Value(_value), BinaryOperator::Eq, Expression::Identifier(_col)) => {
            Scan::Match(key)
        }

        // Case 2:
        // SELECT * FROM t WHERE x > 5;
        // SELECT * FROM t WHERE 5 < x;
        //
        // Position the cursor at key 5, consume key 5 which will cause the
        // cursor to move to the successor and return keys until finished.
        (Expression::Identifier(_col), BinaryOperator::Gt, Expression::Value(_value))
        | (Expression::Value(_value), BinaryOperator::Lt, Expression::Identifier(_col)) => {
            Scan::Range((Bound::Excluded(key), Bound::Unbounded))
        }

        // Case 3:
        // SELECT * FROM t WHERE x < 5;
        // SELECT * FROM t WHERE 5 > x;
        //
        // Allow the cursor to initialize normally (at the smallest key) and
        // tell it to stop once it finds a key >= 5.
        (Expression::Identifier(_col), BinaryOperator::Lt, Expression::Value(_value))
        | (Expression::Value(_value), BinaryOperator::Gt, Expression::Identifier(_col)) => {
            Scan::Range((Bound::Unbounded, Bound::Excluded(key)))
        }

        // Case 4:
        // SELECT * FROM t WHERE x >= 5;
        // SELECT * FROM t WHERE 5 <= x;
        //
        // Position the cursor at key 5 and assume it's already initialized. The
        // cursor will then return key 5 and everything after.
        (Expression::Identifier(_col), BinaryOperator::GtEq, Expression::Value(_value))
        | (Expression::Value(_value), BinaryOperator::LtEq, Expression::Identifier(_col)) => {
            Scan::Range((Bound::Included(key), Bound::Unbounded))
        }

        // Case 5:
        // SELECT * FROM t WHERE x <= 5;
        // SELECT * FROM t WHERE 5 >= x;
        //
        // Allow the cursor to initialize normally (at the smallest key) and
        // tell it to stop once it finds a key > 5
        (Expression::Identifier(_col), BinaryOperator::LtEq, Expression::Value(_value))
        | (Expression::Value(_value), BinaryOperator::GtEq, Expression::Identifier(_col)) => {
            Scan::Range((Bound::Unbounded, Bound::Included(key)))
        }

        _ => unreachable!(),
    };

    let work_dir = db.work_dir.clone();
    let page_size = db.pager.borrow().page_size;

    let maybe_index = indexes.remove(&table.schema.columns[col].name);

    let mut source = match scan {
        Scan::Match(key) => Plan::ExactMatch(ExactMatch {
            key,
            relation,
            done: false,
            pager: Rc::clone(&db.pager),
        }),

        Scan::Range(range) => Plan::RangeScan(RangeScan::from(RangeScanConfig {
            relation,
            range,
            pager: Rc::clone(&db.pager),
        })),
    };

    // Table BTree scan, no additional index needed.
    let Some(index) = maybe_index else {
        return Ok(Some(source));
    };

    // Sort the keys in ascending order to favor sequential IO.
    if let Plan::RangeScan(_) = source {
        source = Plan::Sort(Sort::from(SortConfig {
            page_size,
            work_dir: work_dir.clone(),
            source: BufferedIter::from(BufferedIterConfig {
                source: Box::new(source),
                work_dir: work_dir.clone(),
                schema: index.schema.clone(),
                mem_buf_size: page_size,
            }),
            comparator: TuplesComparator {
                schema: index.schema.clone(),
                sort_schema: index.schema.clone(),
                sort_keys_indexes: vec![1],
            },
            input_buffers: 4,
        }));
    }

    Ok(Some(Plan::KeyScan(KeyScan {
        comparator: table.comparator()?,
        table,
        index,
        pager: Rc::clone(&db.pager),
        source: Box::new(source),
    })))
}

/// Finds an expression that is applied to an indexed column.
///
/// The expression can't be "any" expression that contains an indexed column.
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
/// The OR expression makes the indexed column irrelevant again. So basically
/// every time we find an OR expression we can return [`None`]. There are
/// cases in which an index would be useful even when OR is used, but figuring
/// out exactly how to traverse the index BTree would be complicated. Another
/// important detail is that this function returns the first indexed column
/// that it finds, but if a table contains multiple indexes some of them might
/// be better than others for a certain query. This is a toy database so ain't
/// nobody gonna worry about that :)
fn find_indexed_expr<'e>(
    table: &TableMetadata,
    indexes: &HashMap<String, IndexMetadata>,
    expr: &'e Expression,
) -> Option<&'e Expression> {
    match expr {
        Expression::BinaryOperation {
            left,
            operator,
            right,
        } => match (left.as_ref(), right.as_ref()) {
            (Expression::Identifier(col), Expression::Value(_))
            | (Expression::Value(_), Expression::Identifier(col))
                if (indexes.contains_key(col) || &table.schema.columns[0].name == col)
                    && matches!(
                        operator,
                        BinaryOperator::Eq
                            | BinaryOperator::Lt
                            | BinaryOperator::LtEq
                            | BinaryOperator::Gt
                            | BinaryOperator::GtEq
                    ) =>
            {
                Some(expr)
            }

            _ if *operator == BinaryOperator::And => find_indexed_expr(table, indexes, left)
                .or_else(|| find_indexed_expr(table, indexes, right)),

            _ => None,
        },

        Expression::Nested(expr) => find_indexed_expr(table, indexes, expr),

        _ => None,
    }
}

/// Returns `true` if the `plan` needs the `expr` filter applied.
///
/// Simple expressions with indexed columns like the following one don't need
/// filters because the [`IndexScan`] already filters the output:
///
/// ```sql
/// SELECT * FROM users WHERE id < 5;
/// ```
///
/// There are more cases where the filter wouldn't be needed but this is good
/// enough for now.
fn needs_filter<F>(plan: &Plan<F>, expr: &Expression) -> bool {
    if let Expression::BinaryOperation {
        left,
        operator,
        right,
    } = expr
    {
        if matches!(
            (left.as_ref(), operator, right.as_ref()),
            (Expression::Identifier(_col), _, Expression::Value(_value))
                | (Expression::Value(_value), _, Expression::Identifier(_col))
        ) {
            return !matches!(
                plan,
                Plan::KeyScan(_) | Plan::ExactMatch(_) | Plan::RangeScan(_)
            );
        }
    }

    true
}
