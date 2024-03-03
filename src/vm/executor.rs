use std::{
    cmp::Ordering,
    io::{Read, Seek, Write},
};

use crate::{
    db::{
        Database, DbError, GenericDataType, Projection, QueryResult, RowId, Schema, SqlError,
        StringCmp, TypeError, MKDB_META,
    },
    paging::{
        self,
        pager::{PageNumber, Pager},
    },
    sql::{
        parser::Parser,
        statement::{Column, Constraint, Create, DataType, Expression, Statement, Value},
    },
    storage::{
        page::Page, tuple, BTree, BytesCmp, FixedSizeMemCmp, DEFAULT_BALANCE_SIBLINGS_PER_SIDE,
    },
    vm,
};

pub(crate) fn btree_new<I>(
    pager: &mut Pager<I>,
    root: PageNumber,
) -> BTree<'_, I, FixedSizeMemCmp> {
    BTree::new(pager, root, FixedSizeMemCmp::for_type::<RowId>())
}

fn index_btree<I, C: BytesCmp>(pager: &mut Pager<I>, root: PageNumber, cmp: C) -> BTree<'_, I, C> {
    BTree::new(pager, root, cmp)
}

pub(crate) fn exec<I: Seek + Read + Write + paging::io::Sync>(
    statement: Statement,
    db: &mut Database<I>,
) -> QueryResult {
    match statement {
        Statement::Create(Create::Table { name, columns }) => {
            let root_page = db.pager.alloc_page()?;
            db.pager.init_disk_page::<Page>(root_page)?;

            let mut maybe_primary_key = None;

            for col in &columns {
                if let Some(Constraint::PrimaryKey) = col.constraint {
                    maybe_primary_key = Some(col.name.clone());
                    break;
                }
            }

            db.exec(&format!(
                r#"
                    INSERT INTO {MKDB_META} (type, name, root, table_name, sql)
                    VALUES ("table", "{name}", {root_page}, "{name}", '{sql}');
                "#,
                sql = Statement::Create(Create::Table {
                    name: name.clone(),
                    columns,
                })
            ))?;

            if let Some(primary_key) = maybe_primary_key {
                db.exec(&format!(
                    "CREATE INDEX {name}_pk_index ON {name}({primary_key});"
                ))?;
            }

            Ok(Projection::empty())
        }

        Statement::Create(Create::Index {
            name,
            table,
            column,
        }) => {
            let root_page = db.pager.alloc_page()?;
            db.pager.init_disk_page::<Page>(root_page)?;

            db.exec(&format!(
                r#"
                    INSERT INTO {MKDB_META} (type, name, root, table_name, sql)
                    VALUES ("index", "{name}", {root_page}, "{table}", '{sql}');
                "#,
                sql = Statement::Create(Create::Index {
                    name: name.clone(),
                    table: table.clone(),
                    column
                })
            ))?;

            Ok(Projection::empty())
        }

        Statement::Insert {
            into,
            columns,
            values,
        } => {
            let (schema, root) = db.table_metadata(&into)?;

            let mut resolved_values = vec![Value::Bool(false); schema.len()];

            for (col, expr) in columns.iter().zip(values) {
                let value = match vm::resolve_expression(&Vec::new(), &Schema::empty(), &expr) {
                    Ok(value) => value,

                    Err(e) => Err(e)?,
                };

                // There should be only valid columns here, we can unwrap.
                let index = schema.index_of(col).unwrap();

                match (schema.columns[index].data_type, &value) {
                    (DataType::Bool, Value::Bool(_))
                    | (DataType::Varchar(_), Value::String(_))
                    | (_, Value::Number(_)) => {
                        resolved_values[schema.index_of(col).unwrap()] = value;
                    }
                    (data_type, _) => {
                        return Err(DbError::Sql(SqlError::TypeError(TypeError::ExpectedType {
                            expected: GenericDataType::from(data_type),
                            found: Expression::Value(value),
                        })))
                    }
                }
            }

            let row_id = db.next_row_id(&into, root)?;
            resolved_values[0] = Value::Number(row_id.into());

            let mut btree = btree_new(&mut db.pager, root);

            btree.insert(tuple::serialize_values(&schema, &resolved_values))?;

            // Update all indexes
            let query = db.exec(&format!(
                "SELECT root, sql FROM {MKDB_META} WHERE table_name = '{into}' AND type = 'index';",
            ))?;

            // TODO: Instead of panicking with "unreachable" in situtations
            // like this, return a "Corrupt" error or something similar.
            for i in 0..query.results.len() {
                let root = match query.get(i, "root") {
                    Some(Value::Number(root)) => *root as u32,
                    _ => unreachable!(),
                };

                let sql = match query.get(i, "sql") {
                    Some(Value::String(sql)) => Parser::new(sql).parse_statement()?,
                    _ => unreachable!(),
                };

                let Statement::Create(Create::Index {
                    name,
                    table,
                    column,
                }) = sql
                else {
                    unreachable!();
                };

                let col_idx = schema.index_of(&column).unwrap();

                let key = resolved_values.get(col_idx).unwrap().clone();

                let tuple = vec![key, Value::Number(row_id.into())];

                let index_schema = Schema::new(vec![
                    schema.columns[col_idx].clone(),
                    Column {
                        name: "row_id".into(),
                        data_type: DataType::UnsignedInt,
                        constraint: None,
                    },
                ]);

                match &index_schema.columns[0].data_type {
                    DataType::Varchar(_) => {
                        let mut btree = index_btree(
                            &mut db.pager,
                            root,
                            StringCmp {
                                schema: Schema::new(vec![index_schema.columns[0].clone()]),
                            },
                        );

                        btree.insert(tuple::serialize_values(&index_schema, &tuple))?;
                    }
                    DataType::Int | DataType::UnsignedInt => {
                        let mut btree = index_btree(&mut db.pager, root, FixedSizeMemCmp(4));
                        btree.insert(tuple::serialize_values(&index_schema, &tuple))?;
                    }

                    DataType::BigInt | DataType::UnsignedBigInt => {
                        let mut btree = index_btree(&mut db.pager, root, FixedSizeMemCmp(8));
                        btree.insert(tuple::serialize_values(&index_schema, &tuple))?;
                    }
                    _ => unreachable!(),
                }
            }

            Ok(Projection::empty())
        }

        Statement::Select {
            mut columns,
            from,
            r#where,
            order_by,
        } => {
            let (schema, root) = db.table_metadata(&from)?;

            let mut results_schema = Schema::empty();
            let mut unknown_types = Vec::new();

            columns = {
                let mut resolved_wildcards = Vec::new();
                for expr in columns {
                    if let &Expression::Wildcard = &expr {
                        for col in &schema.columns[1..] {
                            resolved_wildcards.push(Expression::Identifier(col.name.to_owned()));
                        }
                    } else {
                        resolved_wildcards.push(expr);
                    }
                }
                resolved_wildcards
            };

            // Adjust results schema
            for (i, expr) in columns.iter().enumerate() {
                match expr {
                    Expression::Identifier(ident) => match schema.index_of(ident) {
                        Some(index) => results_schema.push(schema.columns[index].clone()),
                        None => Err(DbError::Sql(SqlError::InvalidColumn(ident.clone())))?,
                    },

                    _ => {
                        results_schema.push(Column {
                            name: expr.to_string(),    // TODO: AS alias
                            data_type: DataType::Bool, // We'll set it later
                            constraint: None,
                        });

                        unknown_types.push(i);
                    }
                }
            }

            let mut results = Vec::new();

            let mut btree = btree_new(&mut db.pager, root);

            for row in btree.iter() {
                let values = tuple::deserialize_values(&row?, &schema);

                if !vm::eval_where(&schema, &values, &r#where)? {
                    continue;
                }

                let mut result = Vec::new();

                for expr in &columns {
                    result.push(vm::resolve_expression(&values, &schema, expr)?);
                }

                let mut order_by_vals = Vec::new();

                for expr in &order_by {
                    order_by_vals.push(vm::resolve_expression(&values, &schema, expr)?);
                }

                results.push((result, order_by_vals));
            }

            // We already set the default of unknown types as bools, if
            // it's a number then change it to BigInt. We don't support any
            // expressions that produce strings. And we don't use the types
            // of results for anything now anyway.
            if !results.is_empty() {
                for i in unknown_types {
                    if let Value::Number(_) = &results[0].0[i] {
                        results_schema.columns[i].data_type = DataType::BigInt;
                    }
                }
            }

            // TODO: Order by can contain column that we didn't select.
            if !order_by.is_empty() {
                results.sort_by(|(_, a), (_, b)| {
                    for (a, b) in a.iter().zip(b) {
                        let cmp = match (a, b) {
                            (Value::Number(a), Value::Number(b)) => a.cmp(b),
                            (Value::String(a), Value::String(b)) => a.cmp(b),
                            (Value::Bool(a), Value::Bool(b)) => a.cmp(b),
                            _ => unreachable!("columns should have the same type"),
                        };

                        if cmp != Ordering::Equal {
                            return cmp;
                        }
                    }

                    Ordering::Equal
                })
            }

            Ok(Projection::new(
                results_schema,
                results.into_iter().map(|r| r.0).collect(),
            ))
        }

        Statement::Delete { from, r#where } => {
            let (schema, root) = db.table_metadata(&from)?;

            let mut btree = btree_new(&mut db.pager, root);

            // TODO: Use some cursor or something to delete as we traverse the tree.
            let mut row_ids = Vec::new();

            for row in btree.iter() {
                // TODO: Deserialize only needed values instead of all and the cloning...
                let values = tuple::deserialize_values(&row?, &schema);

                if !vm::eval_where(&schema, &values, &r#where)? {
                    continue;
                }

                match values[0] {
                    Value::Number(row_id) => row_ids.push(row_id as RowId),
                    _ => unreachable!(),
                };
            }

            // TODO: Second mutable borrow occurs here?
            let mut btree = btree_new(&mut db.pager, root);

            for row_id in row_ids {
                btree.remove(&tuple::serialize_row_id(row_id))?;
            }

            Ok(Projection::empty())
        }

        Statement::Update {
            table,
            columns,
            r#where,
        } => {
            let (schema, root) = db.table_metadata(&table)?;

            let mut btree = btree_new(&mut db.pager, root);

            let mut updates = Vec::new();

            for row in btree.iter() {
                // TODO: Deserialize only needed values instead of all and then cloning...
                let mut values = tuple::deserialize_values(&row?, &schema);

                if !vm::eval_where(&schema, &values, &r#where)? {
                    continue;
                }

                for assignment in &columns {
                    let value = vm::resolve_expression(&values, &schema, &assignment.value)?;
                    let index = schema
                        .index_of(&assignment.identifier)
                        .ok_or(SqlError::InvalidColumn(assignment.identifier.clone()))?;

                    values[index] = value;
                    updates.push(tuple::serialize_values(&schema, &values));
                }
            }

            // TODO: Second mutable borrow occurs here?
            let mut btree = btree_new(&mut db.pager, root);

            for update in updates {
                btree.insert(update)?;
            }

            Ok(Projection::empty())
        }

        _ => todo!("rest of SQL statements"),
    }
}
