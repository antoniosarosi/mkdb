//! Code that executes [`Plan`] trees.

use std::{
    cell::RefCell,
    cmp::Ordering,
    io::{Read, Seek, Write},
    mem,
    rc::Rc,
};

use crate::{
    db::{DbError, IndexMetadata, Projection, RowId, Schema, SqlError},
    paging::{
        self,
        io::FileOps,
        pager::{PageNumber, Pager},
    },
    sql::statement::{Assignment, BinaryOperator, Expression, Value},
    storage::{reassemble_payload, tuple, BTree, BytesCmp, Cursor, FixedSizeMemCmp},
    vm,
};

pub(crate) fn exec<I: Seek + Read + Write + paging::io::FileOps>(
    plan: Plan<I>,
) -> Result<Projection, DbError> {
    Projection::try_from(plan)
}

pub(crate) type Tuple = Vec<Value>;

pub(crate) enum Plan<I> {
    Values(Values),
    SeqScan(SeqScan<I>),
    IndexScan(IndexScan<I>),
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
// but that's even more verbose than this and requires I: 'static everywhere. We
// also woudn't know the type of a plan because dyn Trait doesn't have a tag. So
// match it for now :)
impl<I: Seek + Read + Write + FileOps> Plan<I> {
    pub fn try_next(&mut self) -> Result<Option<Tuple>, DbError> {
        match self {
            Self::SeqScan(seq_scan) => seq_scan.try_next(),
            Self::IndexScan(index_scan) => index_scan.try_next(),
            Self::Filter(filter) => filter.try_next(),
            Self::Project(project) => project.try_next(),
            Self::Values(values) => values.try_next(),
            Self::Sort(sort) => sort.try_next(),
            Self::Update(update) => update.try_next(),
            Self::Insert(insert) => insert.try_next(),
            Self::Delete(delete) => delete.try_next(),
        }
    }
}

impl<I: Seek + Read + Write + FileOps> Iterator for Plan<I> {
    type Item = Result<Tuple, DbError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.try_next().transpose()
    }
}

impl<I> Plan<I> {
    pub fn schema(&self) -> Option<Schema> {
        match self {
            Self::Project(project) => Some(project.output_schema.clone()),
            _ => None,
        }
    }
}

pub(crate) struct BufferedIter<I> {
    pub source: Box<Plan<I>>,
    pub collection: Vec<Tuple>,
    pub collected: bool,
}

impl<I: Seek + Read + Write + FileOps> BufferedIter<I> {
    pub fn new(source: Box<Plan<I>>) -> Self {
        Self {
            source,
            collected: false,
            collection: vec![],
        }
    }

    pub fn collect(&mut self) -> Result<(), DbError> {
        while let Some(tuple) = self.source.try_next()? {
            self.collection.push(tuple);
        }

        Ok(())
    }

    pub fn try_next(&mut self) -> Result<Option<Tuple>, DbError> {
        if !self.collected {
            self.collect()?;
            self.collected = true;
        }

        if self.collection.is_empty() {
            return Ok(None);
        }

        Ok(Some(self.collection.remove(0)))
    }
}

pub(crate) struct SeqScan<I> {
    pub schema: Schema,
    pub pager: Rc<RefCell<Pager<I>>>,
    pub cursor: Cursor,
}

impl<I: Seek + Read + Write + FileOps> SeqScan<I> {
    fn try_next(&mut self) -> Result<Option<Tuple>, DbError> {
        let mut pager = self.pager.borrow_mut();

        let Some((page, slot)) = self.cursor.try_next(&mut pager)? else {
            return Ok(None);
        };

        Ok(Some(tuple::deserialize_values(
            reassemble_payload(&mut pager, page, slot)?.as_ref(),
            &self.schema,
        )))
    }
}

pub(crate) struct IndexScan<I> {
    pub index_schema: Schema,
    pub table_schema: Schema,
    pub table_root: PageNumber,
    pub index_root: PageNumber,
    pub stop_when: Option<(Vec<u8>, BinaryOperator, Box<dyn BytesCmp>)>,
    pub done: bool,
    pub pager: Rc<RefCell<Pager<I>>>,
    pub cursor: Cursor,
}

impl<I: Seek + Read + Write + FileOps> IndexScan<I> {
    fn try_next(&mut self) -> Result<Option<Tuple>, DbError> {
        if self.done {
            return Ok(None);
        };

        let mut pager = self.pager.borrow_mut();

        let Some((page, slot)) = self.cursor.try_next(&mut pager)? else {
            self.done = true;
            return Ok(None);
        };

        if let Some((key, operator, cmp)) = &self.stop_when {
            let entry = &pager.get(page)?.cell(slot).content;

            let ordering = cmp.bytes_cmp(entry, key);

            let stop = match operator {
                BinaryOperator::Gt => ordering == Ordering::Greater,
                BinaryOperator::GtEq => matches!(ordering, Ordering::Greater | Ordering::Equal),
                BinaryOperator::Lt => ordering == Ordering::Less,
                BinaryOperator::LtEq => matches!(ordering, Ordering::Less | Ordering::Equal),
                _ => unreachable!(),
            };

            if stop {
                self.done = true;
                return Ok(None);
            }
        }

        let index_entry =
            tuple::deserialize_values(&pager.get(page)?.cell(slot).content, &self.index_schema);

        let Value::Number(row_id) = index_entry[1] else {
            panic!("indexes should always map to row IDs but this one doesn't: {index_entry:?}");
        };

        let mut btree = BTree::new(
            &mut pager,
            self.table_root,
            FixedSizeMemCmp::for_type::<RowId>(),
        );

        let table_entry = btree
            .get(&tuple::serialize_row_id(row_id as RowId))?
            .unwrap_or_else(|| {
                panic!(
                    "index at root {} maps to row ID {} that doesn't exist in table at root {}",
                    self.index_root, row_id, self.table_root
                )
            });

        Ok(Some(tuple::deserialize_values(
            table_entry.as_ref(),
            &self.table_schema,
        )))
    }
}

pub(crate) struct Filter<I> {
    pub source: Box<Plan<I>>,
    pub schema: Schema,
    pub filter: Expression,
}

impl<I: Seek + Read + Write + FileOps> Filter<I> {
    fn try_next(&mut self) -> Result<Option<Tuple>, DbError> {
        while let Some(tuple) = self.source.try_next()? {
            if vm::eval_where(&self.schema, &tuple, &self.filter)? {
                return Ok(Some(tuple));
            }
        }

        Ok(None)
    }
}

pub(crate) struct Project<I> {
    pub source: Box<Plan<I>>,
    pub input_schema: Schema,
    pub output_schema: Schema,
    pub projection: Vec<Expression>,
}

impl<I: Seek + Read + Write + FileOps> Project<I> {
    fn try_next(&mut self) -> Result<Option<Tuple>, DbError> {
        let Some(tuple) = self.source.try_next()? else {
            return Ok(None);
        };

        Ok(Some(
            self.projection
                .iter()
                .map(|expr| vm::resolve_expression(&tuple, &self.input_schema, expr))
                .collect::<Result<Tuple, _>>()?,
        ))
    }
}

pub(crate) struct Values {
    pub values: Vec<Expression>,
    pub done: bool,
}

impl Values {
    fn try_next(&mut self) -> Result<Option<Tuple>, DbError> {
        if self.done {
            return Ok(None);
        }

        self.done = true;

        Ok(Some(
            self.values
                .iter()
                .map(vm::resolve_literal_expression)
                .collect::<Result<Vec<Value>, SqlError>>()?,
        ))
    }
}

pub(crate) struct Sort<I> {
    pub source: BufferedIter<I>,
    pub schema: Schema,
    pub by: Vec<Expression>,
    pub sorted: bool,
}

impl<I: Seek + Read + Write + FileOps> Sort<I> {
    fn sort(&mut self) -> Result<(), DbError> {
        for tuple in self.source.collection.iter_mut() {
            for expr in &self.by {
                let sort_key = vm::resolve_expression(tuple, &self.schema, expr)?;
                tuple.push(sort_key);
            }
        }

        let sort_keys = self.schema.len();

        self.source.collection.sort_by(|a, b| {
            for (a, b) in a[sort_keys..].iter().zip(&b[sort_keys..]) {
                match a.partial_cmp(b) {
                    Some(ordering) => {
                        if ordering != Ordering::Equal {
                            return ordering;
                        }
                    }
                    None => {
                        if mem::discriminant(a) != mem::discriminant(b) {
                            unreachable!("it should be impossible to run into type errors at this point: cmp() {a} against {b}");
                        }
                    }
                }
            }

            Ordering::Equal
        });

        // TODO: This can probably be done in one single pass when returning the results
        for tuple in &mut self.source.collection {
            tuple.drain(sort_keys..);
        }

        Ok(())
    }

    fn try_next(&mut self) -> Result<Option<Tuple>, DbError> {
        if !self.sorted {
            self.source.collect()?;
            self.sort()?;
            self.sorted = true;
        }

        self.source.try_next()
    }
}

pub(crate) struct Insert<I> {
    pub root: PageNumber,
    pub pager: Rc<RefCell<Pager<I>>>,
    pub source: Box<Plan<I>>,
    pub schema: Schema,
    pub indexes: Vec<IndexMetadata>,
}

impl<I: Seek + Read + Write + FileOps> Insert<I> {
    fn try_next(&mut self) -> Result<Option<Tuple>, DbError> {
        let Some(tuple) = self.source.try_next()? else {
            return Ok(None);
        };

        let mut pager = self.pager.borrow_mut();

        let mut btree = BTree::new(&mut pager, self.root, FixedSizeMemCmp::for_type::<RowId>());
        btree.insert(tuple::serialize_values(&self.schema, &tuple))?;

        for IndexMetadata { root, column, .. } in &self.indexes {
            let idx = self.schema.index_of(column).unwrap();

            assert_eq!(self.schema.columns[0].name, "row_id");
            let key_col = self.schema.columns[idx].clone();
            let row_id_col = self.schema.columns[0].clone();

            let key = tuple[idx].clone();
            let row_id = tuple[0].clone();

            let entry =
                tuple::serialize_values(&Schema::from(vec![key_col, row_id_col]), &[key, row_id]);

            let mut btree = BTree::new(
                &mut pager,
                *root,
                Box::<dyn BytesCmp>::from(&self.schema.columns[idx].data_type),
            );

            btree.insert(entry)?;
        }

        Ok(Some(vec![]))
    }
}

pub(crate) struct Update<I> {
    pub root: PageNumber,
    pub assignments: Vec<Assignment>,
    pub pager: Rc<RefCell<Pager<I>>>,
    pub source: BufferedIter<I>,
    pub schema: Schema,
}

impl<I: Seek + Read + Write + FileOps> Update<I> {
    fn try_next(&mut self) -> Result<Option<Tuple>, DbError> {
        let Some(mut tuple) = self.source.try_next()? else {
            return Ok(None);
        };

        for assignment in &self.assignments {
            let idx = self.schema.index_of(&assignment.identifier).unwrap();
            tuple[idx] = vm::resolve_expression(&tuple, &self.schema, &assignment.value)?;
        }

        let mut pager = self.pager.borrow_mut();
        let mut btree = BTree::new(&mut pager, self.root, FixedSizeMemCmp::for_type::<RowId>());

        btree.insert(tuple::serialize_values(&self.schema, &tuple))?;

        Ok(Some(vec![]))
    }
}

pub(crate) struct Delete<I> {
    pub root: PageNumber,
    pub pager: Rc<RefCell<Pager<I>>>,
    pub source: BufferedIter<I>,
    pub schema: Schema,
    pub indexes: Vec<IndexMetadata>,
}

impl<I: Seek + Read + Write + FileOps> Delete<I> {
    fn try_next(&mut self) -> Result<Option<Tuple>, DbError> {
        let Some(tuple) = self.source.try_next()? else {
            return Ok(None);
        };

        let mut pager = self.pager.borrow_mut();
        let mut btree = BTree::new(&mut pager, self.root, FixedSizeMemCmp::for_type::<RowId>());

        btree.remove(&tuple::serialize_values(&self.schema, &tuple))?;

        for IndexMetadata { root, column, .. } in &self.indexes {
            let idx = self.schema.index_of(column).unwrap();
            let key_col = self.schema.columns[idx].clone();
            let key = tuple[idx].clone();

            let entry = tuple::serialize_values(&Schema::from(vec![key_col]), &[key]);

            let mut btree = BTree::new(
                &mut pager,
                *root,
                Box::<dyn BytesCmp>::from(&self.schema.columns[idx].data_type),
            );

            btree.remove(&entry)?;
        }

        Ok(Some(vec![]))
    }
}
