use std::{
    cell::{RefCell, RefMut},
    cmp::Ordering,
    collections::VecDeque,
    io::{self, Read, Seek, Write},
    mem,
    rc::Rc,
};

use crate::{
    db::{DbError, Projection, RowId, Schema},
    paging::pager::{PageNumber, Pager},
    sql::statement::{Assignment, Column, DataType, Expression, Value},
    storage::{page::SlotId, tuple, BTree, Cursor, FixedSizeMemCmp},
    vm,
};

pub(crate) enum Plan<I> {
    Values(Values),
    SeqScan(SeqScan<I>),
    Filter(Filter<I>),
    Project(Project<I>),
    Sort(Sort<I>),
    Update(Update<I>),
    Insert(Insert<I>),
    Delete(Delete<I>),
}

// TODO: As mentioned at [`crate::paging::pager::get_as`], we could also use
// [`enum_dispatch`](https://docs.rs/enum_dispatch/) here to automate the match
// statement or switch to Box<dyn Iterator<Item = Result<Projection, DbError>>>
// but that's even more verbose than this and requires I: 'static everywhere. So
// match it for now :)
impl<I: Seek + Read + Write> Plan<I> {
    pub fn next(&mut self) -> Option<Result<Projection, DbError>> {
        match self {
            Self::SeqScan(scan) => scan.next(),
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

pub(crate) struct BufferedIter<I> {
    source: Box<Plan<I>>,
    pub collection: Projection,
    collected: bool,
}

impl<I: Seek + Read + Write> BufferedIter<I> {
    pub fn new(source: Box<Plan<I>>) -> Self {
        Self {
            source,
            collection: Projection::empty(),
            collected: false,
        }
    }

    pub fn collect(&mut self) -> Result<(), DbError> {
        let Some(first) = self.source.next() else {
            return Ok(());
        };

        self.collection = first?;

        while let Some(projection) = self.source.next() {
            self.collection.results.append(&mut projection?.results);
        }

        Ok(())
    }

    pub fn next(&mut self) -> Option<Result<Projection, DbError>> {
        if !self.collected {
            if let Err(e) = self.collect() {
                return Some(Err(e));
            }
            self.collected = true;
        }

        if self.collection.is_empty() {
            return None;
        }

        // TODO: Is it even possible to write this more inefficiently?
        Some(Ok(Projection::new(self.collection.schema.clone(), vec![
            self.collection.results.remove(0),
        ])))
    }
}

pub(crate) struct SeqScan<I> {
    schema: Schema,
    pager: Rc<RefCell<Pager<I>>>,
    cursor: Cursor,
}

impl<I: Seek + Read + Write> SeqScan<I> {
    pub fn new(root: PageNumber, schema: Schema, pager: Rc<RefCell<Pager<I>>>) -> Self {
        Self {
            pager,
            schema,
            cursor: Cursor::new(root, 0),
        }
    }

    fn next(&mut self) -> Option<Result<Projection, DbError>> {
        let mut pager = self.pager.borrow_mut();

        let (page_number, slot) = match self.cursor.next(&mut pager)? {
            Ok(position) => position,
            Err(e) => return Some(Err(e.into())),
        };

        let page = match pager.get(page_number) {
            Ok(page) => page,
            Err(e) => return Some(Err(e.into())),
        };

        let row = tuple::deserialize_values(&page.cell(slot).content, &self.schema);

        Some(Ok(Projection {
            results: vec![row],
            schema: self.schema.clone(), // TODO: Shouldn't need to clone this.
        }))
    }
}

pub(crate) struct Filter<I> {
    source: Box<Plan<I>>,
    filter: Expression,
}

impl<I: Seek + Read + Write> Filter<I> {
    pub fn new(source: Box<Plan<I>>, filter: Expression) -> Self {
        Self { source, filter }
    }

    fn next(&mut self) -> Option<Result<Projection, DbError>> {
        loop {
            let projection = match self.source.next()? {
                Ok(projection) => projection,
                Err(e) => return Some(Err(e)),
            };

            let matches_predicate =
                match vm::eval_where(&projection.schema, &projection.results[0], &self.filter) {
                    Ok(eval) => eval,
                    Err(e) => return Some(Err(DbError::Sql(e))),
                };

            if matches_predicate {
                return Some(Ok(projection));
            }
        }
    }
}

pub(crate) struct Project<I> {
    source: Box<Plan<I>>,
    output: Vec<Expression>,
}

impl<I: Seek + Read + Write> Project<I> {
    pub fn new(source: Box<Plan<I>>, output: Vec<Expression>) -> Self {
        let mut schema = Schema::empty();

        Self { source, output }
    }

    fn next(&mut self) -> Option<Result<Projection, DbError>> {
        let projection = match self.source.next()? {
            Ok(projection) => projection,
            Err(e) => return Some(Err(e)),
        };

        let mut results_schema = Schema::empty();
        let mut resolved_values = Vec::new();
        let mut unknown_types = Vec::new();

        for (i, expr) in self.output.iter().enumerate() {
            match expr {
                Expression::Identifier(ident) => results_schema.push(
                    projection.schema.columns[projection.schema.index_of(ident).unwrap()].clone(),
                ),

                _ => {
                    results_schema.push(Column {
                        name: expr.to_string(),    // TODO: AS alias
                        data_type: DataType::Bool, // We'll set it later
                        constraints: vec![],
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
}

pub(crate) struct Values {
    schema: Option<Schema>, // TODO: need to move out of schema
    values: Vec<Expression>,
    done: bool,
}

impl Values {
    pub fn new(schema: Schema, values: Vec<Expression>) -> Self {
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
            .map(vm::resolve_literal_expression)
            .collect::<Result<_, _>>();

        match values {
            Ok(values) => Some(Ok(Projection::new(schema, vec![values]))),
            Err(e) => Some(Err(e.into())),
        }
    }
}

pub(crate) struct Sort<I> {
    source: BufferedIter<I>,
    by: Vec<Expression>,
    sorted: bool,
}

impl<I: Seek + Read + Write> Sort<I> {
    pub fn new(source: Box<Plan<I>>, by: Vec<Expression>) -> Self {
        Self {
            source: BufferedIter::new(source),
            by,
            sorted: false,
        }
    }

    fn sort(&mut self) -> Result<(), DbError> {
        for row in &mut self.source.collection.results {
            for expr in &self.by {
                let sort_key = vm::resolve_expression(row, &self.source.collection.schema, expr)?;
                row.push(sort_key);
            }
        }

        let sort_keys_idx = self.source.collection.schema.len();

        self.source.collection.results.sort_by(|a, b| {
            for (sort_key_a, sort_key_b) in a[sort_keys_idx..].iter().zip(&b[sort_keys_idx..]) {
                match sort_key_a.partial_cmp(sort_key_b) {
                    Some(ordering) if ordering != Ordering::Equal => return ordering,
                    _ => {
                        if mem::discriminant(sort_key_a) != mem::discriminant(sort_key_b)  {
                            unreachable!(
                                "it should be impossible to run into type errors at this point: cmp() {sort_key_a} against {sort_key_b}"
                            );
                        }
                    }
                }
            }

            Ordering::Equal
        });

        // TODO: This can probably be done in one single pass when returning the results
        for row in &mut self.source.collection.results {
            row.drain(sort_keys_idx..);
        }

        Ok(())
    }

    fn next(&mut self) -> Option<Result<Projection, DbError>> {
        if !self.sorted {
            if let Err(e) = self.source.collect().and_then(|_| self.sort()) {
                return Some(Err(e));
            }
            self.sorted = true;
        }

        self.source.next()
    }
}

pub(crate) struct Insert<I> {
    root: PageNumber,
    pager: Rc<RefCell<Pager<I>>>,
    source: Box<Plan<I>>,
}

impl<I: Seek + Read + Write> Insert<I> {
    pub fn new(root: PageNumber, pager: Rc<RefCell<Pager<I>>>, source: Box<Plan<I>>) -> Self {
        Self {
            root,
            pager,
            source,
        }
    }

    fn next(&mut self) -> Option<Result<Projection, DbError>> {
        let mut pager = self.pager.borrow_mut();

        let values = match self.source.next()? {
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

pub(crate) struct Update<I> {
    root: PageNumber,
    assignments: Vec<Assignment>,
    pager: Rc<RefCell<Pager<I>>>,
    source: BufferedIter<I>,
}

impl<I: Seek + Read + Write> Update<I> {
    pub fn new(
        root: PageNumber,
        pager: Rc<RefCell<Pager<I>>>,
        source: Box<Plan<I>>,
        assignments: Vec<Assignment>,
    ) -> Self {
        Self {
            assignments,
            root,
            pager,
            source: BufferedIter::new(source),
        }
    }

    fn next(&mut self) -> Option<Result<Projection, DbError>> {
        let mut row = match self.source.next()? {
            Ok(projection) => projection,
            Err(e) => return Some(Err(e)),
        };

        for assignment in &self.assignments {
            let value = vm::resolve_expression(&row.results[0], &row.schema, &assignment.value);

            match value {
                Ok(v) => {
                    let idx = row.schema.index_of(&assignment.identifier).unwrap();
                    row.results[0][idx] = v;
                }

                Err(e) => return Some(Err(e.into())),
            }
        }

        let mut pager = self.pager.borrow_mut();
        let mut btree = BTree::new(&mut pager, self.root, FixedSizeMemCmp::for_type::<RowId>());

        match btree.insert(tuple::serialize_values(&row.schema, &row.results[0])) {
            Ok(_) => Some(Ok(Projection::empty())),
            Err(e) => Some(Err(e.into())),
        }
    }
}

pub(crate) struct Delete<I> {
    root: PageNumber,
    pager: Rc<RefCell<Pager<I>>>,
    source: BufferedIter<I>,
}

impl<I: Seek + Read + Write> Delete<I> {
    pub fn new(root: PageNumber, pager: Rc<RefCell<Pager<I>>>, source: Box<Plan<I>>) -> Self {
        Self {
            root,
            pager,
            source: BufferedIter::new(source),
        }
    }

    fn next(&mut self) -> Option<Result<Projection, DbError>> {
        let row = match self.source.next()? {
            Ok(projection) => projection,
            Err(e) => return Some(Err(e)),
        };

        let mut pager = self.pager.borrow_mut();
        let mut btree = BTree::new(&mut pager, self.root, FixedSizeMemCmp::for_type::<RowId>());

        match btree.remove(&tuple::serialize_values(&row.schema, &row.results[0])) {
            Ok(_) => Some(Ok(Projection::empty())),
            Err(e) => Some(Err(e.into())),
        }
    }
}
