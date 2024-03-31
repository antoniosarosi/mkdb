//! Generates [`Plan`] trees.
//!
//! See the module level documentation of [`crate::vm::plan`] to understand
//! what exactly we're "generating" here.

use std::{
    collections::VecDeque,
    io::{Read, Seek, Write},
    rc::Rc,
};

use super::optimizer;
use crate::{
    db::{Database, DatabaseContext, DbError, Schema, SqlError},
    paging,
    sql::{
        analyzer,
        statement::{Column, DataType, Expression, Statement},
    },
    vm::{
        plan::{
            BufferedIter, BufferedIterConfig, Delete, Insert, Plan, Project, Sort, SortConfig,
            SortKeysGen, TuplesComparator, Update, Values,
        },
        VmDataType,
    },
};

/// Generates a query plan that's ready to execute by the VM.
pub(crate) fn generate_plan<F: Seek + Read + Write + paging::io::FileOps>(
    statement: Statement,
    db: &mut Database<F>,
) -> Result<Plan<F>, DbError> {
    Ok(match statement {
        Statement::Insert {
            into,
            columns,
            values,
        } => {
            let source = Box::new(Plan::Values(Values {
                values: VecDeque::from([values]),
            }));

            let table = db.table_metadata(&into)?.clone();

            Plan::Insert(Insert {
                source,
                comparator: table.comparator()?,
                table: db.table_metadata(&into)?.clone(),
                pager: Rc::clone(&db.pager),
            })
        }

        Statement::Select {
            columns,
            from,
            r#where,
            order_by,
        } => {
            let mut source = optimizer::generate_scan_plan(&from, r#where, db)?;

            let page_size = db.pager.borrow().page_size;

            let work_dir = db.work_dir.clone();
            let table = db.table_metadata(&from)?;

            if !order_by.is_empty() {
                let mut sort_schema = table.schema.clone();
                let mut sort_keys_indexes = Vec::with_capacity(order_by.len());

                // Precompute all the sort keys indexes so that the sorter
                // doesn't waste time figuring out where the columns are.
                for (i, expr) in order_by.iter().enumerate() {
                    let index = match expr {
                        Expression::Identifier(col) => table.schema.index_of(col).unwrap(),

                        _ => {
                            let index = sort_schema.len();
                            let data_type = resolve_unknown_type(&table.schema, expr)?;
                            let col = Column::new(&format!("sort_key_{i}"), data_type);
                            sort_schema.push(col);

                            index
                        }
                    };

                    sort_keys_indexes.push(index);
                }

                // If there are no expressions that need to be evaluated for
                // sorting then just skip the sort key generation completely,
                // we already have all the sort keys we need.
                let buffered_iter_source = if sort_schema.len() > table.schema.len() {
                    Plan::SortKeysGen(SortKeysGen {
                        source: Box::new(source),
                        schema: table.schema.clone(),
                        gen_exprs: order_by
                            .into_iter()
                            .filter(|expr| !matches!(expr, Expression::Identifier(_)))
                            .collect(),
                    })
                } else {
                    source
                };

                source = Plan::Sort(Sort::from(SortConfig {
                    page_size,
                    work_dir: work_dir.clone(),
                    source: BufferedIter::from(BufferedIterConfig {
                        source: Box::new(buffered_iter_source),
                        work_dir,
                        schema: sort_schema.clone(),
                        mem_buf_size: page_size,
                    }),
                    comparator: TuplesComparator {
                        schema: table.schema.clone(),
                        sort_schema,
                        sort_keys_indexes,
                    },
                    input_buffers: 4,
                }));
            }

            let mut output_schema = Schema::empty();

            for expr in &columns {
                match expr {
                    Expression::Identifier(ident) => output_schema
                        .push(table.schema.columns[table.schema.index_of(ident).unwrap()].clone()),

                    _ => {
                        output_schema.push(Column {
                            name: expr.to_string(), // TODO: AS alias
                            data_type: resolve_unknown_type(&table.schema, expr)?,
                            constraints: vec![],
                        });
                    }
                }
            }

            // No need to project if the output schema is the exact same as the
            // table schema.
            if table.schema == output_schema {
                return Ok(source);
            }

            Plan::Project(Project {
                input_schema: table.schema.clone(),
                output_schema,
                projection: columns,
                source: Box::new(source),
            })
        }

        Statement::Update {
            table,
            columns,
            r#where,
        } => {
            let mut source = optimizer::generate_scan_plan(&table, r#where, db)?;
            let work_dir = db.work_dir.clone();
            let page_size = db.pager.borrow().page_size;
            let metadata = db.table_metadata(&table)?;

            // Index scans have their own internal buffering for sorting.
            // Sequential scans plans don't, which is useful for SELECT
            // statements that don't need any buffering. Updates and deletes do
            // need buffering because BTree operations can destroy the scan
            // cursor. Maybe we can keep track of every cursor when updating the
            // BTree but as of right now it seems pretty complicated because the
            // BTree is not a self contained unit that can be passed around like
            // the pager.
            if !is_scan_plan_buffered(&source) {
                source = Plan::BufferedIter(BufferedIter::from(BufferedIterConfig {
                    source: Box::new(source),
                    work_dir,
                    schema: metadata.schema.clone(),
                    mem_buf_size: page_size,
                }));
            }

            Plan::Update(Update {
                comparator: metadata.comparator()?,
                table: metadata.clone(),
                assignments: columns,
                pager: Rc::clone(&db.pager),
                source: Box::new(source),
            })
        }

        Statement::Delete { from, r#where } => {
            let mut source = optimizer::generate_scan_plan(&from, r#where, db)?;
            let work_dir = db.work_dir.clone();
            let page_size = db.pager.borrow().page_size;
            let metadata = db.table_metadata(&from)?;

            if !is_scan_plan_buffered(&source) {
                source = Plan::BufferedIter(BufferedIter::from(BufferedIterConfig {
                    source: Box::new(source),
                    work_dir,
                    mem_buf_size: page_size,
                    schema: metadata.schema.clone(),
                }));
            }

            Plan::Delete(Delete {
                comparator: metadata.comparator()?,
                table: metadata.clone(),
                pager: Rc::clone(&db.pager),
                source: Box::new(source),
            })
        }

        other => todo!("unhandled statement {other}"),
    })
}

/// Returns a concrete [`DataType`] for an expression that hasn't been executed
/// yet.
///
/// TODO: There are no expressions that can evaluate to strings as of right now
/// since we didn't implement `CONCAT()` or any other similar function, so
/// strings can only come from identifiers. The [`analyzer`] should never return
/// [`VmDataType::String`], so it doesn't matter what type we return in that
/// case.
///
/// The real problem is when expressions evaluate to numbers becase we don't
/// know the exact kind of number. An expression with a raw value like
/// 4294967296 should evaluate to [`DataType::UnsignedBigInt`] but -65536 should
/// probably evaluate to [`DataType::Int`]. Expressions that have identifiers in
/// them should probably evaluate to the type of the identifier, but what if
/// there are multiple identifiers of different integer types? Not gonna worry
/// about this for now, this is a toy database after all :)
fn resolve_unknown_type(schema: &Schema, expr: &Expression) -> Result<DataType, SqlError> {
    Ok(match expr {
        Expression::Identifier(col) => {
            let index = schema.index_of(col).unwrap();
            schema.columns[index].data_type
        }

        _ => match analyzer::analyze_expression(schema, expr)? {
            VmDataType::Bool => DataType::Bool,
            VmDataType::Number => DataType::BigInt,
            VmDataType::String => DataType::Varchar(65535),
        },
    })
}

/// Returns `true` if the given scan buffers the tuples before returning
/// them.
fn is_scan_plan_buffered<F>(plan: &Plan<F>) -> bool {
    match plan {
        Plan::Filter(filter) => is_scan_plan_buffered(&filter.source),
        Plan::KeyScan(_) | Plan::ExactMatch(_) => true,
        Plan::SeqScan(_) | Plan::RangeScan(_) => false,
        _ => unreachable!("is_scan_plan_buffered() called with plan that is not a 'scan' plan"),
    }
}

#[cfg(test)]
mod tests {
    use std::{cell::RefCell, collections::HashMap, io, ops::Bound, path::PathBuf, rc::Rc};

    use crate::{
        db::{Database, DatabaseContext, IndexMetadata, Relation, Schema, TableMetadata},
        paging::{io::MemBuf, pager::Pager},
        sql::{
            self,
            parser::Parser,
            statement::{Column, Create, DataType, Expression, Statement, Value},
        },
        storage::{
            tuple::{self, byte_length_of_integer_type},
            Cursor, FixedSizeMemCmp,
        },
        vm::plan::{
            ExactMatch, Filter, KeyScan, Plan, Project, RangeScan, RangeScanConfig, SeqScan,
        },
        DbError,
    };

    /// Test database context.
    struct DbCtx {
        inner: Database<MemBuf>,
        tables: HashMap<String, TableMetadata>,
        indexes: HashMap<String, IndexMetadata>,
    }

    impl DbCtx {
        fn pager(&self) -> Rc<RefCell<Pager<MemBuf>>> {
            Rc::clone(&self.inner.pager)
        }
    }

    impl AsMut<Database<MemBuf>> for DbCtx {
        fn as_mut(&mut self) -> &mut Database<MemBuf> {
            &mut self.inner
        }
    }

    fn init_db(ctx: &[&str]) -> Result<DbCtx, DbError> {
        let mut pager = Pager::<MemBuf>::builder().wrap(io::Cursor::new(Vec::<u8>::new()));
        pager.init()?;

        let mut db = Database::new(Rc::new(RefCell::new(pager)), PathBuf::new());

        let mut tables = HashMap::new();
        let mut indexes = HashMap::new();

        let mut fetch_tables = Vec::new();

        for sql in ctx {
            if let Statement::Create(Create::Table { name, .. }) =
                Parser::new(sql).parse_statement()?
            {
                fetch_tables.push(name);
            }

            db.exec(sql)?;
        }

        for table_name in fetch_tables {
            let table = db.table_metadata(&table_name)?;

            for index in &table.indexes {
                indexes.insert(index.name.to_owned(), index.to_owned());
            }

            tables.insert(table_name, table.to_owned());
        }

        Ok(DbCtx {
            inner: db,
            tables,
            indexes,
        })
    }

    fn gen_plan(
        db: &mut impl AsMut<Database<MemBuf>>,
        query: &str,
    ) -> Result<Plan<MemBuf>, DbError> {
        let statement = sql::pipeline(query, db.as_mut())?;
        super::generate_plan(statement, db.as_mut())
    }

    fn parse_expr(expr: &str) -> Expression {
        Parser::new(expr).parse_expression().unwrap()
    }

    #[test]
    fn generate_basic_sequential_scan() -> Result<(), DbError> {
        let mut db = init_db(&["CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(255));"])?;

        assert_eq!(
            gen_plan(&mut db, "SELECT * FROM users;")?,
            Plan::SeqScan(SeqScan {
                pager: db.pager(),
                cursor: Cursor::new(db.tables["users"].root, 0),
                table: db.tables["users"].to_owned(),
            })
        );

        Ok(())
    }

    #[test]
    fn generate_sequential_scan_with_filter() -> Result<(), DbError> {
        let mut db =
            init_db(&["CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(255), age INT);"])?;

        assert_eq!(
            gen_plan(&mut db, "SELECT * FROM users WHERE age >= 20;")?,
            Plan::Filter(Filter {
                filter: parse_expr("age >= 20"),
                schema: db.tables["users"].schema.to_owned(),
                source: Box::new(Plan::SeqScan(SeqScan {
                    pager: db.pager(),
                    cursor: Cursor::new(db.tables["users"].root, 0),
                    table: db.tables["users"].to_owned(),
                }))
            })
        );

        Ok(())
    }

    // Tables with no primary key have a special "row_id" column.
    #[test]
    fn generate_sequential_scan_with_projection_when_using_row_id() -> Result<(), DbError> {
        let mut db = init_db(&["CREATE TABLE users (id INT, name VARCHAR(255));"])?;

        assert_eq!(
            gen_plan(&mut db, "SELECT * FROM users;")?,
            Plan::Project(Project {
                input_schema: db.tables["users"].schema.to_owned(),
                output_schema: Schema::new(vec![
                    Column::new("id", DataType::Int),
                    Column::new("name", DataType::Varchar(255))
                ]),
                projection: vec![
                    Expression::Identifier("id".into()),
                    Expression::Identifier("name".into())
                ],
                source: Box::new(Plan::SeqScan(SeqScan {
                    pager: db.pager(),
                    cursor: Cursor::new(db.tables["users"].root, 0),
                    table: db.tables["users"].to_owned(),
                }))
            })
        );

        Ok(())
    }

    // Tables with no primary key have a special "row_id" column.
    #[test]
    fn generate_sequential_scan_with_projection_when_selecting_columns() -> Result<(), DbError> {
        let mut db = init_db(&[
            "CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(255), email VARCHAR(255));",
        ])?;

        assert_eq!(
            gen_plan(&mut db, "SELECT email, id FROM users;")?,
            Plan::Project(Project {
                input_schema: db.tables["users"].schema.to_owned(),
                output_schema: Schema::new(vec![
                    Column::new("email", DataType::Varchar(255)),
                    Column::primary_key("id", DataType::Int),
                ]),
                projection: vec![
                    Expression::Identifier("email".into()),
                    Expression::Identifier("id".into()),
                ],
                source: Box::new(Plan::SeqScan(SeqScan {
                    cursor: Cursor::new(db.tables["users"].root, 0),
                    table: db.tables["users"].to_owned(),
                    pager: db.pager()
                }))
            })
        );

        Ok(())
    }

    #[test]
    fn generate_basic_sequential_scan_with_filter_and_projection() -> Result<(), DbError> {
        let mut db =
            init_db(&["CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(255), age INT);"])?;

        assert_eq!(
            gen_plan(&mut db, "SELECT name FROM users WHERE age >= 20;")?,
            Plan::Project(Project {
                input_schema: db.tables["users"].schema.to_owned(),
                output_schema: Schema::new(vec![Column::new("name", DataType::Varchar(255))]),
                projection: vec![Expression::Identifier("name".into())],
                source: Box::new(Plan::Filter(Filter {
                    filter: parse_expr("age >= 20"),
                    schema: db.tables["users"].schema.to_owned(),
                    source: Box::new(Plan::SeqScan(SeqScan {
                        cursor: Cursor::new(db.tables["users"].root, 0),
                        table: db.tables["users"].to_owned(),
                        pager: db.pager()
                    }))
                }))
            })
        );

        Ok(())
    }

    #[test]
    fn generate_exact_match_on_auto_index() -> Result<(), DbError> {
        let mut db = init_db(&["CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(255));"])?;

        assert_eq!(
            gen_plan(&mut db, "SELECT * FROM users WHERE id = 5;")?,
            Plan::ExactMatch(ExactMatch {
                key: tuple::serialize_key(&DataType::Int, &Value::Number(5)),
                pager: db.pager(),
                relation: Relation::Table(db.tables["users"].to_owned()),
                done: false,
            })
        );

        Ok(())
    }

    #[test]
    fn generate_exact_match_on_external_index() -> Result<(), DbError> {
        let mut db =
            init_db(&["CREATE TABLE users (id INT PRIMARY KEY, email VARCHAR(255) UNIQUE);"])?;

        assert_eq!(
            gen_plan(
                &mut db,
                "SELECT * FROM users WHERE email = 'bob@email.com';"
            )?,
            Plan::KeyScan(KeyScan {
                pager: db.pager(),
                comparator: FixedSizeMemCmp(byte_length_of_integer_type(&DataType::Int)),
                index: db.indexes["users_email_uq_index"].to_owned(),
                table: db.tables["users"].to_owned(),
                source: Box::new(Plan::ExactMatch(ExactMatch {
                    pager: db.pager(),
                    relation: Relation::Index(db.indexes["users_email_uq_index"].to_owned()),
                    key: tuple::serialize_key(
                        &DataType::Varchar(255),
                        &Value::String("bob@email.com".into())
                    ),
                    done: false,
                }))
            })
        );

        Ok(())
    }

    #[test]
    fn skip_filter_on_simple_range_scan() -> Result<(), DbError> {
        let mut db = init_db(&["CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(255));"])?;

        assert_eq!(
            gen_plan(&mut db, "SELECT * FROM users WHERE id < 5;")?,
            Plan::RangeScan(RangeScan::from(RangeScanConfig {
                pager: db.pager(),
                relation: Relation::Table(db.tables["users"].to_owned()),
                range: (
                    Bound::Unbounded,
                    Bound::Excluded(tuple::serialize_key(&DataType::Int, &Value::Number(5)))
                ),
            }))
        );

        Ok(())
    }

    #[test]
    fn apply_filter_if_cant_be_skipped() -> Result<(), DbError> {
        let mut db = init_db(&["CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(255));"])?;

        assert_eq!(
            gen_plan(
                &mut db,
                "SELECT * FROM users WHERE id < 5 AND name = 'Bob';"
            )?,
            Plan::Filter(Filter {
                filter: parse_expr("id < 5 AND name = 'Bob'"),
                schema: db.tables["users"].schema.to_owned(),
                source: Box::new(Plan::RangeScan(RangeScan::from(RangeScanConfig {
                    pager: db.pager(),
                    relation: Relation::Table(db.tables["users"].to_owned()),
                    range: (
                        Bound::Unbounded,
                        Bound::Excluded(tuple::serialize_key(&DataType::Int, &Value::Number(5)))
                    )
                })))
            })
        );

        Ok(())
    }
}
