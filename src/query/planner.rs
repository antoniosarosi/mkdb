//! Generates a query plan.

use std::{
    cell::RefCell,
    cmp::Ordering,
    collections::VecDeque,
    io::{Read, Seek, Write},
    mem,
    rc::Rc,
};

use crate::{
    db::{Database, DbError, Projection, RowId, Schema},
    paging::pager::{PageNumber, Pager},
    sql::statement::{Assignment, Column, DataType, Expression, Statement, Value},
    storage::{page::SlotId, tuple, BTree, FixedSizeMemCmp},
    vm,
};

pub(crate) fn generate_plan<I: Seek + Read + Write + crate::paging::io::Sync>(
    statement: Statement,
    db: &mut Database<I>,
) -> Result<Plan<I>, DbError> {
    match statement {
        Statement::Select {
            columns,
            from,
            r#where,
            order_by,
        } => {
            let (schema, root) = db.table_metadata(&from)?;

            let mut plan = Box::new(Plan::Scan(Scan::new(root, schema, Rc::clone(&db.pager))));

            if let Some(filter) = r#where {
                plan = Box::new(Plan::Filter(Filter::new(plan, filter)));
            }

            if !order_by.is_empty() {
                plan = Box::new(Plan::Sort(Sort::new(plan, order_by)));
            }

            Ok(Plan::Project(Project::new(plan, columns)))
        }

        Statement::Update {
            table,
            columns,
            r#where,
        } => {
            let (schema, root) = db.table_metadata(&table)?;

            let mut plan = Box::new(Plan::Scan(Scan::new(root, schema, Rc::clone(&db.pager))));

            if let Some(filter) = r#where {
                plan = Box::new(Plan::Filter(Filter::new(plan, filter)));
            }

            Ok(Plan::Update(Update::new(
                root,
                Rc::clone(&db.pager),
                plan,
                columns,
            )))
        }

        Statement::Insert {
            into,
            mut columns,
            mut values,
        } => {
            let (schema, root) = db.table_metadata(&into)?;

            // TODO: Should we do this here or in prepare()?
            if schema.columns[0].name == "row_id" {
                let row_id = db.next_row_id(&into, root)?;
                columns.insert(0, "row_id".into());
                values.insert(0, Expression::Value(Value::Number(row_id.into())));
            }

            let mut sorted = vec![Expression::Wildcard; values.len()];

            for (val, col) in values.into_iter().zip(columns) {
                sorted[schema.index_of(&col).unwrap()] = val;
            }

            let plan = Box::new(Plan::Values(Values::new(schema, sorted)));

            Ok(Plan::Insert(Insert::new(root, Rc::clone(&db.pager), plan)))
        }

        Statement::Delete { from, r#where } => {
            let (schema, root) = db.table_metadata(&from)?;

            let mut plan = Box::new(Plan::Scan(Scan::new(root, schema, Rc::clone(&db.pager))));

            if let Some(filter) = r#where {
                plan = Box::new(Plan::Filter(Filter::new(plan, filter)));
            }

            Ok(Plan::Delete(Delete::new(root, Rc::clone(&db.pager), plan)))
        }

        other => todo!("unhandled statement {other}"),
    }
}

pub(crate) enum Plan<I> {
    Values(Values),
    Scan(Scan<I>),
    Filter(Filter<I>),
    Project(Project<I>),
    Sort(Sort<I>),
    Update(Update<I>),
    Insert(Insert<I>),
    Delete(Delete<I>),
}

// TODO: Same as [`crate::paging::pager::get_as`], we could use enum_dispatch
impl<I: Seek + Read + Write> Plan<I> {
    pub fn next(&mut self) -> Option<Result<Projection, DbError>> {
        match self {
            Self::Scan(scan) => scan.next(),
            Self::Filter(filter) => filter.next(),
            Self::Project(project) => project.next(),
            Self::Values(values) => values.next(),
            Self::Sort(sort) => sort.next(),
            Self::Update(update) => update.next(),
            Self::Insert(insert) => insert.next(),
            Self::Delete(delete) => delete.next(),
        }
    }
}

struct Scan<I> {
    schema: Schema,
    queue: VecDeque<PageNumber>,
    slot: SlotId,
    pager: Rc<RefCell<Pager<I>>>,
}

impl<I: Seek + Read + Write> Scan<I> {
    fn new(root: PageNumber, schema: Schema, pager: Rc<RefCell<Pager<I>>>) -> Self {
        Self {
            pager,
            schema,
            slot: 0,
            queue: VecDeque::from([root]),
        }
    }

    fn next(&mut self) -> Option<Result<Projection, DbError>> {
        let mut pager = self.pager.borrow_mut();

        // Finding the page from which we should return content while maintaing
        // the iterator state is not trivial. In theory, a page could be empty
        // but still have one child. That is not a bug, although the BTree will
        // not allow it except for some edge cases of page zero, since page zero
        // has less space and the balancing algorithm sometimes needs to leave
        // the page empty. See the documentation of [`crate::storage::btree`]
        // and [`crate::storage::page`] to fully understand what's going on
        // here. This guard should work for all cases anyway.
        //
        // TODO: This is not exactly sequential IO since BTree pages are not
        // guaranteed to be linked sequentially. Pretty sure there must be a way
        // to make this 100% sequential IO, but it would require some other disk
        // data structure to maintain free pages sorted sequentially so that
        // when the BTree allocates a page the page is guaranteed to be located
        // after the BTree root. But even so the balancing algorithm cannot
        // maintain all pages in sequential order when the BTree root overflows,
        // so this optimization is definitely not easy to implement.
        let page = 'find_current_page: loop {
            match pager.get(*self.queue.front()?) {
                // The page where we left last time has cells, keep returning content.
                Ok(page) if page.len() > 0 => break 'find_current_page page,

                // Page is empty, find the next page that has some content.
                Ok(page) => {
                    self.queue.pop_front();
                    self.queue.extend(page.iter_children());
                }

                // Not our problem, yeet it :)
                Err(e) => return Some(Err(DbError::Io(e))),
            }
        };

        let row = tuple::deserialize_values(&page.cell(self.slot).content, &self.schema);

        self.slot += 1;

        // We're done with this page, move to the next one.
        if self.slot >= page.len() {
            self.queue.pop_front();
            self.queue.extend(page.iter_children());
            self.slot = 0;
        }

        Some(Ok(Projection {
            results: vec![row],
            schema: self.schema.clone(), // TODO: Shouldn't need to clone this.
        }))
    }
}

struct Filter<I> {
    plan: Box<Plan<I>>,
    filter: Expression,
}

impl<I: Seek + Read + Write> Filter<I> {
    fn new(plan: Box<Plan<I>>, filter: Expression) -> Self {
        Self { plan, filter }
    }

    fn next(&mut self) -> Option<Result<Projection, DbError>> {
        loop {
            match self.plan.next()? {
                Ok(projection) => {
                    match vm::eval_where(&projection.schema, &projection.results[0], &self.filter) {
                        Ok(eval) => {
                            if eval {
                                return Some(Ok(projection));
                            }
                        }
                        Err(e) => return Some(Err(DbError::Sql(e))),
                    }
                }
                Err(e) => return Some(Err(e)),
            }
        }
    }
}

struct Project<I> {
    plan: Box<Plan<I>>,
    output: Vec<Expression>,
}

impl<I: Seek + Read + Write> Project<I> {
    fn new(plan: Box<Plan<I>>, output: Vec<Expression>) -> Self {
        Self { plan, output }
    }

    fn next(&mut self) -> Option<Result<Projection, DbError>> {
        match self.plan.next()? {
            Ok(projection) => {
                let mut results_schema = Schema::empty();
                let mut resolved_values = Vec::new();
                let mut unknown_types = Vec::new();

                for (i, expr) in self.output.iter().enumerate() {
                    match expr {
                        Expression::Identifier(ident) => results_schema.push(
                            projection.schema.columns[projection.schema.index_of(&ident).unwrap()]
                                .clone(),
                        ),

                        _ => {
                            results_schema.push(Column {
                                name: expr.to_string(),    // TODO: AS alias
                                data_type: DataType::Bool, // We'll set it later
                                constraint: None,
                            });

                            unknown_types.push(i);
                        }
                    }

                    match vm::resolve_expression(&projection.results[0], &projection.schema, expr) {
                        Ok(v) => resolved_values.push(v),
                        Err(e) => return Some(Err(e.into())),
                    }
                }

                for i in unknown_types {
                    if let Value::Number(_) = &projection.results[0][i] {
                        results_schema.columns[i].data_type = DataType::BigInt;
                    }
                }

                Some(Ok(Projection::new(results_schema, vec![resolved_values])))
            }
            Err(e) => return Some(Err(e)),
        }
    }
}

struct Values {
    schema: Option<Schema>, // TODO: need to move out of schema
    values: Vec<Expression>,
    done: bool,
}

impl Values {
    fn new(schema: Schema, values: Vec<Expression>) -> Self {
        Self {
            schema: Some(schema),
            values,
            done: false,
        }
    }

    fn next(&mut self) -> Option<Result<Projection, DbError>> {
        if self.done {
            return None;
        }

        let schema = self.schema.take().unwrap();
        self.done = true;

        let values = self
            .values
            .iter()
            .map(|expr| vm::resolve_expression(&vec![], &Schema::empty(), &expr))
            .collect::<Result<_, _>>();

        match values {
            Ok(values) => Some(Ok(Projection::new(schema, vec![values]))),

            Err(e) => Some(Err(e.into())),
        }
    }
}

struct Sort<I> {
    plan: Box<Plan<I>>,
    by: Vec<Expression>,
    collection: Projection,
    collected: bool,
}

impl<I: Seek + Read + Write> Sort<I> {
    fn new(plan: Box<Plan<I>>, by: Vec<Expression>) -> Self {
        Self {
            plan,
            by,
            collection: Projection::empty(),
            collected: false,
        }
    }

    fn next(&mut self) -> Option<Result<Projection, DbError>> {
        if self.collected {
            if self.collection.is_empty() {
                return None;
            }

            // TODO: Most unoptimized shit ever written at 3 AM fix this
            return Some(Ok(Projection::new(
                self.collection.schema.clone(),
                vec![self.collection.results.remove(0)],
            )));
        }

        self.collection = match self.plan.next()? {
            Ok(p) => p,
            Err(e) => return Some(Err(e)),
        };

        while let Some(result) = self.plan.next() {
            match result {
                Ok(mut p) => self.collection.results.append(&mut p.results),
                Err(e) => return Some(Err(e)),
            }
        }

        self.collected = true;

        self.collection.results.sort_by(|a, b| {
            for expr in &self.by {
                // TODO: Don't unwrap as there are errors that we can't
                // control such as division by 0. Implement merge sort
                // or other algorithm manually just like we did with
                // binary search for the BTree.
                let a = vm::resolve_expression(a, &self.collection.schema, expr).unwrap();
                let b = vm::resolve_expression(b, &self.collection.schema, expr).unwrap();

                let cmp = match (a, b) {
                    (Value::Number(a), Value::Number(b)) => a.cmp(&b),
                    (Value::String(a), Value::String(b)) => a.cmp(&b),
                    (Value::Bool(a), Value::Bool(b)) => a.cmp(&b),
                    _ => todo!("implement merge sort manually and return db errors"),
                };

                if cmp != Ordering::Equal {
                    return cmp;
                }
            }

            Ordering::Equal
        });

        // TODO: See the early return above
        Some(Ok(Projection::new(
            self.collection.schema.clone(),
            vec![self.collection.results.remove(0)],
        )))
    }
}

struct Update<I> {
    root: PageNumber,
    assignments: Vec<Assignment>,
    pager: Rc<RefCell<Pager<I>>>,
    plan: Box<Plan<I>>,
}

impl<I: Seek + Read + Write> Update<I> {
    fn new(
        root: PageNumber,
        pager: Rc<RefCell<Pager<I>>>,
        plan: Box<Plan<I>>,
        assignments: Vec<Assignment>,
    ) -> Self {
        Self {
            assignments,
            root,
            pager,
            plan,
        }
    }

    fn next(&mut self) -> Option<Result<Projection, DbError>> {
        let mut row = match self.plan.next()? {
            Ok(projection) => projection,
            Err(e) => return Some(Err(e)),
        };

        for assignment in &self.assignments {
            let value = vm::resolve_expression(&row.results[0], &row.schema, &assignment.value);

            match value {
                Err(e) => return Some(Err(e.into())),

                Ok(v) => {
                    let idx = row.schema.index_of(&assignment.identifier).unwrap();
                    row.results[0][idx] = v;
                }
            }
        }

        let mut pager = self.pager.borrow_mut();
        let mut btree = BTree::new(&mut pager, self.root, FixedSizeMemCmp::for_type::<RowId>());

        // TODO: ModifyKind
        match btree.insert(tuple::serialize_values(&row.schema, &row.results[0])) {
            Ok(_) => Some(Ok(Projection::empty())),
            Err(e) => Some(Err(e.into())),
        }
    }
}

struct Insert<I> {
    root: PageNumber,
    pager: Rc<RefCell<Pager<I>>>,
    plan: Box<Plan<I>>,
}

impl<I: Seek + Read + Write> Insert<I> {
    fn new(root: PageNumber, pager: Rc<RefCell<Pager<I>>>, plan: Box<Plan<I>>) -> Self {
        Self { root, pager, plan }
    }

    fn next(&mut self) -> Option<Result<Projection, DbError>> {
        let mut pager = self.pager.borrow_mut();

        let values = match self.plan.next()? {
            Ok(projection) => projection,
            Err(e) => return Some(Err(e)),
        };

        let mut btree = BTree::new(&mut pager, self.root, FixedSizeMemCmp::for_type::<RowId>());

        match btree.insert(tuple::serialize_values(&values.schema, &values.results[0])) {
            Ok(_) => Some(Ok(Projection::empty())),
            Err(e) => Some(Err(e.into())),
        }
    }
}

struct Delete<I> {
    root: PageNumber,
    pager: Rc<RefCell<Pager<I>>>,
    plan: Box<Plan<I>>,
    collected: bool,
    collection: Projection,
}

impl<I: Seek + Read + Write> Delete<I> {
    fn new(root: PageNumber, pager: Rc<RefCell<Pager<I>>>, plan: Box<Plan<I>>) -> Self {
        Self {
            root,
            pager,
            plan,
            collected: false,
            collection: Projection::empty(),
        }
    }

    fn next(&mut self) -> Option<Result<Projection, DbError>> {
        // TODO: Huge problem here, if we delete from the BTree we invalidate
        // the underlying Scan iterator, but at the same time we shouldn't need
        // to collect all the values.
        if !self.collected {
            while let Some(result) = self.plan.next() {
                match result {
                    Ok(mut p) => {
                        self.collection.schema = p.schema;
                        self.collection.results.append(&mut p.results)
                    }
                    Err(e) => return Some(Err(e)),
                }
            }

            self.collected = true;
        }

        if self.collection.is_empty() {
            return None;
        }

        let mut pager = self.pager.borrow_mut();
        let mut btree = BTree::new(&mut pager, self.root, FixedSizeMemCmp::for_type::<RowId>());

        let values = self.collection.results.remove(0);

        match btree.remove(&tuple::serialize_values(&self.collection.schema, &values)) {
            Ok(_) => Some(Ok(Projection::empty())),
            Err(e) => Some(Err(e.into())),
        }
    }
}
