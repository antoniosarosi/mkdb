// Generates optimized plans.

use std::{
    collections::HashMap,
    io::{self, Read, Seek, Write},
    rc::Rc,
};

use crate::{
    db::{Database, DatabaseContext, DbError, IndexMetadata, Schema},
    paging::{
        io::FileOps,
        pager::{PageNumber, Pager},
    },
    sql::statement::{BinaryOperator, DataType, Expression},
    storage::{tuple, BTree, BytesCmp, Cursor},
    vm::plan::{Filter, IndexScan, Plan, SeqScan},
};

/// Generates an optimized scan plan.
///
/// The generated plan will be either [`SeqScan`] or [`IndexScan`] with an
/// optional [`Filter`] on top of it, depending on how the query looks like.
pub(crate) fn generate_scan_plan<F: Seek + Read + Write + FileOps>(
    table: &str,
    filter: Option<Expression>,
    db: &mut Database<F>,
) -> Result<Box<Plan<F>>, DbError> {
    let Some(expr) = filter else {
        return generate_sequential_scan_plan(table, db);
    };

    let source = if let Some(index_scan) = generate_index_scan_plan(table, db, &expr)? {
        index_scan
    } else {
        generate_sequential_scan_plan(table, db)?
    };

    Ok(Box::new(Plan::Filter(Filter {
        source,
        schema: db.table_metadata(table)?.schema.clone(),
        filter: expr,
    })))
}

/// Creates a new cursor and positions it at the given key.
///
/// The cursor will return the given key when calling [`Cursor::try_next`].
fn position_cursor_at_key<F: Seek + Read + Write + FileOps>(
    key: &[u8],
    data_type: &DataType,
    root: PageNumber,
    pager: &mut Pager<F>,
) -> io::Result<Cursor> {
    let mut descent = Vec::new();

    let mut btree = BTree::new(pager, root, Box::<dyn BytesCmp>::from(data_type));
    let search = btree.search(root, key, &mut descent)?;

    match search.index {
        Ok(slot) => Ok(Cursor::initialized(search.page, slot, descent)),
        Err(_) => Ok(Cursor::done()),
    }
}

/// Basically a constructor. We'll use this until we figure out exactly how to
/// build stuff here.
fn generate_sequential_scan_plan<F: Seek + Read + Write + FileOps>(
    table: &str,
    db: &mut Database<F>,
) -> Result<Box<Plan<F>>, DbError> {
    let metadata = db.table_metadata(table)?;

    Ok(Box::new(Plan::SeqScan(SeqScan {
        cursor: Cursor::new(metadata.root, 0),
        table: metadata.clone(),
        pager: Rc::clone(&db.pager),
    })))
}

/// Attempts to generate an [`IndexScan`] plan.
///
/// It's only possible to do so if we find an expression that contains an
/// indexed column and must always be executed. Otherwise we'll fallback to
/// sequential scans.
fn generate_index_scan_plan<F: Seek + Read + Write + FileOps>(
    table: &str,
    db: &mut Database<F>,
    filter: &Expression,
) -> Result<Option<Box<Plan<F>>>, DbError> {
    let metadata = db.table_metadata(table)?;

    if metadata.indexes.is_empty() {
        return Ok(None);
    }

    // Build index map (column name -> index metadata)
    let indexes = HashMap::from_iter(
        metadata
            .indexes
            .iter()
            .map(|index| (index.column.name.clone(), index.clone())),
    );

    // Find expression that's using an indexed column.
    let Some(Expression::BinaryOperation {
        left,
        operator,
        right,
    }) = find_indexed_expr(&indexes, filter)
    else {
        return Ok(None);
    };

    // Grab some metadata
    let table = metadata.clone();

    let (key, index) = match (left.as_ref(), right.as_ref()) {
        (Expression::Identifier(col), Expression::Value(value))
        | (Expression::Value(value), Expression::Identifier(col)) => {
            let index = indexes[col].clone();
            let key = tuple::serialize(&Schema::from(vec![index.column.clone()]), &[value.clone()]);

            (key, index)
        }

        _ => unreachable!(),
    };

    // Now do the magic.
    let pager = &mut db.pager.borrow_mut();

    let (cursor, stop_when) = match (left.as_ref(), operator, right.as_ref()) {
        // Case 1:
        // SELECT * FROM t WHERE x = 5;
        // SELECT * FROM t WHERE 5 = x;
        //
        // Position the cursor at key 5 and stop at the next key.
        (Expression::Identifier(_col), BinaryOperator::Eq, Expression::Value(_value))
        | (Expression::Value(_value), BinaryOperator::Eq, Expression::Identifier(_col)) => {
            let cursor = position_cursor_at_key(&key, &index.column.data_type, index.root, pager)?;
            (cursor, Some(BinaryOperator::Gt))
        }

        // Case 2:
        // SELECT * FROM t WHERE x > 5;
        // SELECT * FROM t WHERE 5 < x;
        //
        // Position the cursor at key 5, assume it's initialized and consume key
        // 5 which will cause the cursor to move to the successor.
        (Expression::Identifier(_col), BinaryOperator::Gt, Expression::Value(_value))
        | (Expression::Value(_value), BinaryOperator::Lt, Expression::Identifier(_col)) => {
            let mut cursor =
                position_cursor_at_key(&key, &index.column.data_type, index.root, pager)?;
            cursor.try_next(pager)?;
            (cursor, None)
        }

        // Case 3:
        // SELECT * FROM t WHERE x < 5;
        // SELECT * FROM t WHERE 5 > x;
        //
        // Allow the cursor to initialize normally (at the smallest key) and
        // tell it to stop once it finds a key >= 5.
        (Expression::Identifier(_col), BinaryOperator::Lt, Expression::Value(_value))
        | (Expression::Value(_value), BinaryOperator::Gt, Expression::Identifier(_col)) => {
            let cursor = Cursor::new(index.root, 0);
            (cursor, Some(BinaryOperator::GtEq))
        }

        // Case 4:
        // SELECT * FROM t WHERE x >= 5;
        // SELECT * FROM t WHERE 5 <= x;
        //
        // Position the cursor at key 5 and assume it's already initialized. The
        // cursor will then return key 5 and everything after.
        (Expression::Identifier(_col), BinaryOperator::GtEq, Expression::Value(_value))
        | (Expression::Value(_value), BinaryOperator::LtEq, Expression::Identifier(_col)) => {
            let cursor = position_cursor_at_key(&key, &index.column.data_type, index.root, pager)?;
            (cursor, None)
        }

        // Case 5:
        // SELECT * FROM t WHERE x <= 5;
        // SELECT * FROM t WHERE 5 >= x;
        //
        // Allow the cursor to initialize normally (at the smallest key) and
        // tell it to stop once it finds a key > 5
        (Expression::Identifier(_col), BinaryOperator::LtEq, Expression::Value(_value))
        | (Expression::Value(_value), BinaryOperator::GtEq, Expression::Identifier(_col)) => {
            let cursor = Cursor::new(index.root, 0);
            (cursor, Some(BinaryOperator::Gt))
        }

        _ => unreachable!(),
    };

    Ok(Some(Box::new(Plan::IndexScan(IndexScan {
        cursor,
        stop_when: stop_when.map(|operator| {
            (
                key,
                operator,
                Box::<dyn BytesCmp>::from(&index.column.data_type),
            )
        }),
        table,
        index,
        done: false,
        pager: Rc::clone(&db.pager),
    }))))
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
                if indexes.contains_key(col)
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

            _ if *operator == BinaryOperator::And => {
                find_indexed_expr(indexes, left).or_else(|| find_indexed_expr(indexes, right))
            }

            _ => None,
        },

        Expression::Nested(expr) => find_indexed_expr(indexes, expr),

        _ => None,
    }
}
