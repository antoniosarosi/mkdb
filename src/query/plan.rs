use std::{
    cell::{RefCell, RefMut},
    cmp::Ordering,
    collections::VecDeque,
    io::{self, Read, Seek, Write},
    mem,
    rc::Rc,
};

use crate::{
    db::{DbError, Projection, RowId, Schema, SqlError, VmDataType},
    paging::pager::{PageNumber, Pager},
    sql::{
        analyzer,
        statement::{Assignment, Column, DataType, Expression, Value},
    },
    storage::{page::SlotId, tuple, BTree, Cursor, FixedSizeMemCmp},
    vm,
};

pub(crate) type Tuple = Vec<Value>;

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
    pub fn next(&mut self) -> Option<Result<Tuple, DbError>> {
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

    pub fn schema(&self) -> &Schema {
        match self {
            Self::SeqScan(scan) => &scan.schema,
            Self::Project(project) => &project.schema,
            Self::Insert(insert) => &insert.schema,

            Self::Filter(filter) => filter.source.schema(),
            Self::Sort(sort) => sort.source.schema(),
            Self::Update(update) => update.source.schema(),
            Self::Delete(delete) => delete.source.schema(),

            _ => unreachable!(),
        }
    }
}

pub(crate) struct BufferedIter<I> {
    source: Box<Plan<I>>,
    pub collection: Vec<Tuple>,
    collected: bool,
}

impl<I: Seek + Read + Write> BufferedIter<I> {
    pub fn new(source: Box<Plan<I>>) -> Self {
        Self {
            source,
            collection: Vec::new(),
            collected: false,
        }
    }

    pub fn schema(&self) -> &Schema {
        self.source.schema()
    }

    pub fn collect(&mut self) -> Result<(), DbError> {
        let Some(first) = self.source.next() else {
            return Ok(());
        };

        self.collection = vec![first?];

        while let Some(tuple) = self.source.next() {
            self.collection.push(tuple?);
        }

        Ok(())
    }

    pub fn next(&mut self) -> Option<Result<Tuple, DbError>> {
        if !self.collected {
            if let Err(e) = self.collect() {
                return Some(Err(e));
            }
            self.collected = true;
        }

        if self.collection.is_empty() {
            return None;
        }

        Some(Ok(self.collection.remove(0)))
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

    fn next(&mut self) -> Option<Result<Tuple, DbError>> {
        let mut pager = self.pager.borrow_mut();

        let (page_number, slot) = match self.cursor.next(&mut pager)? {
            Ok(position) => position,
            Err(e) => return Some(Err(e.into())),
        };

        let page = match pager.get(page_number) {
            Ok(page) => page,
            Err(e) => return Some(Err(e.into())),
        };

        Some(Ok(tuple::deserialize_values(
            &page.cell(slot).content,
            &self.schema,
        )))
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

    fn next(&mut self) -> Option<Result<Tuple, DbError>> {
        loop {
            let tuple = match self.source.next()? {
                Ok(tuple) => tuple,
                Err(e) => return Some(Err(e)),
            };

            let matches_predicate = match vm::eval_where(self.source.schema(), &tuple, &self.filter)
            {
                Ok(eval) => eval,
                Err(e) => return Some(Err(DbError::Sql(e))),
            };

            if matches_predicate {
                return Some(Ok(tuple));
            }
        }
    }
}

pub(crate) struct Project<I> {
    source: Box<Plan<I>>,
    schema: Schema,
    projection: Vec<Expression>,
}

impl<I: Seek + Read + Write> Project<I> {
    pub fn new(source: Box<Plan<I>>, projection: Vec<Expression>) -> Self {
        let mut schema = Schema::empty();
        let mut unknown_types = Vec::new();

        for (i, expr) in projection.iter().enumerate() {
            match expr {
                Expression::Identifier(ident) => schema.push(
                    source.schema().columns[source.schema().index_of(ident).unwrap()].clone(),
                ),

                _ => {
                    schema.push(Column {
                        name: expr.to_string(),    // TODO: AS alias
                        data_type: DataType::Bool, // We'll set it later
                        constraints: vec![],
                    });

                    unknown_types.push(i);
                }
            }
        }

        // TODO: There are no expressions that can evaluate to strings as of
        // right now and we set the default to be bool. So if there's an
        // expression that evaluates to a number we'll change its type. The
        // problem is that we don't know the exact kind of number, an expression
        // with a raw value like 4294967296 should evaluate to UnsignedBigInt
        // but -65536 should probably evaluate to Int. Expressions that have
        // identifiers in them should probably evaluate to the type of the
        // identifier. Not gonna worry about this for now, this is a toy
        // database after all :)
        for i in unknown_types {
            if let VmDataType::Number =
                analyzer::analyze_expression(source.schema(), &projection[i]).unwrap()
            {
                schema.columns[i].data_type = DataType::BigInt;
            }
        }

        Self {
            source,
            projection,
            schema,
        }
    }

    fn next(&mut self) -> Option<Result<Tuple, DbError>> {
        let tuple = match self.source.next()? {
            Ok(tuple) => tuple,
            Err(e) => return Some(Err(e)),
        };

        let mut resolved_values = Vec::new();

        for expr in &self.projection {
            match vm::resolve_expression(&tuple, self.source.schema(), expr) {
                Ok(v) => resolved_values.push(v),
                Err(e) => return Some(Err(e.into())),
            }
        }

        Some(Ok(resolved_values))
    }
}

pub(crate) struct Values {
    values: Vec<Expression>,
    done: bool,
}

impl Values {
    pub fn new(values: Vec<Expression>) -> Self {
        Self {
            values,
            done: false,
        }
    }

    fn next(&mut self) -> Option<Result<Tuple, DbError>> {
        if self.done {
            return None;
        }

        self.done = true;

        let values = self
            .values
            .iter()
            .map(vm::resolve_literal_expression)
            .collect::<Result<Vec<Value>, SqlError>>()
            .map_err(|e| e.into());

        Some(values)
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
        for i in 0..self.source.collection.len() {
            for expr in &self.by {
                let sort_key =
                    vm::resolve_expression(&self.source.collection[i], self.source.schema(), expr)?;
                self.source.collection[i].push(sort_key);
            }
        }

        let sort_keys_idx = self.source.schema().len();

        self.source.collection.sort_by(|a, b| {
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
        for row in &mut self.source.collection {
            row.drain(sort_keys_idx..);
        }

        Ok(())
    }

    fn next(&mut self) -> Option<Result<Tuple, DbError>> {
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
    schema: Schema,
}

impl<I: Seek + Read + Write> Insert<I> {
    pub fn new(
        root: PageNumber,
        pager: Rc<RefCell<Pager<I>>>,
        source: Box<Plan<I>>,
        schema: Schema,
    ) -> Self {
        Self {
            root,
            pager,
            source,
            schema,
        }
    }

    fn next(&mut self) -> Option<Result<Tuple, DbError>> {
        let mut pager = self.pager.borrow_mut();

        let tuple = match self.source.next()? {
            Ok(tuple) => tuple,
            Err(e) => return Some(Err(e)),
        };

        let mut btree = BTree::new(&mut pager, self.root, FixedSizeMemCmp::for_type::<RowId>());

        match btree.insert(tuple::serialize_values(&self.schema, &tuple)) {
            Ok(_) => Some(Ok(Vec::new())),
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

    fn next(&mut self) -> Option<Result<Tuple, DbError>> {
        let mut tuple = match self.source.next()? {
            Ok(tuple) => tuple,
            Err(e) => return Some(Err(e)),
        };

        for assignment in &self.assignments {
            let value = vm::resolve_expression(&tuple, self.source.schema(), &assignment.value);

            match value {
                Ok(v) => {
                    let idx = self
                        .source
                        .schema()
                        .index_of(&assignment.identifier)
                        .unwrap();
                    tuple[idx] = v;
                }

                Err(e) => return Some(Err(e.into())),
            }
        }

        let mut pager = self.pager.borrow_mut();
        let mut btree = BTree::new(&mut pager, self.root, FixedSizeMemCmp::for_type::<RowId>());

        match btree.insert(tuple::serialize_values(self.source.schema(), &tuple)) {
            Ok(_) => Some(Ok(Vec::new())),
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

    fn next(&mut self) -> Option<Result<Tuple, DbError>> {
        let row = match self.source.next()? {
            Ok(projection) => projection,
            Err(e) => return Some(Err(e)),
        };

        let mut pager = self.pager.borrow_mut();
        let mut btree = BTree::new(&mut pager, self.root, FixedSizeMemCmp::for_type::<RowId>());

        match btree.remove(&tuple::serialize_values(self.source.schema(), &row)) {
            Ok(_) => Some(Ok(Tuple::new())),
            Err(e) => Some(Err(e.into())),
        }
    }
}
