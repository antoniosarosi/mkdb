//! Code that executes [`Plan`] trees.

use std::{
    cell::RefCell,
    cmp::Ordering,
    io::{self, BufRead, BufReader, Read, Seek, Write},
    mem,
    path::PathBuf,
    rc::Rc,
};

use crate::{
    db::{DbError, IndexMetadata, Projection, RowId, Schema, SqlError, DEFAULT_PAGE_SIZE},
    paging::{
        self,
        io::FileOps,
        pager::{PageNumber, Pager},
    },
    sql::statement::{Assignment, BinaryOperator, Column, DataType, Expression, Value},
    storage::{
        page::MAX_PAGE_SIZE, reassemble_payload, tuple, BTree, BytesCmp, Cursor, FixedSizeMemCmp,
    },
    vm,
};

pub(crate) fn exec<F: Seek + Read + Write + paging::io::FileOps>(
    plan: Plan<F>,
) -> Result<Projection, DbError> {
    Projection::try_from(plan)
}

pub(crate) type Tuple = Vec<Value>;

pub(crate) enum Plan<F> {
    Values(Values),
    SeqScan(SeqScan<F>),
    IndexScan(IndexScan<F>),
    Filter(Filter<F>),
    Project(Project<F>),
    Sort(Sort<F>),
    Update(Update<F>),
    Insert(Insert<F>),
    Delete(Delete<F>),
}

// TODO: As mentioned at [`crate::paging::pager::get_as`], we could also use
// [`enum_dispatch`](https://docs.rs/enum_dispatch/) here to automate the match
// statement or switch to Box<dyn Iterator<Item = Result<Projection, DbError>>>
// but that's even more verbose than this and requires I: 'static everywhere. We
// also woudn't know the type of a plan because dyn Trait doesn't have a tag. So
// match it for now :)
impl<F: Seek + Read + Write + FileOps> Plan<F> {
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

impl<F: Seek + Read + Write + FileOps> Iterator for Plan<F> {
    type Item = Result<Tuple, DbError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.try_next().transpose()
    }
}

impl<F> Plan<F> {
    pub fn schema(&self) -> Option<Schema> {
        match self {
            Self::Project(project) => Some(project.output_schema.clone()),
            _ => None,
        }
    }
}

pub(crate) struct BufferedIter<F> {
    pub source: Box<Plan<F>>,
    pub schema: Schema,
    pub mem_buf: Vec<u8>,
    pub max_size: usize,
    pub file: Option<F>,
    pub reader: Option<BufReader<F>>,
    pub work_dir: PathBuf,
    pub collected: bool,
    pub file_path: PathBuf,
    pub max_buffered_tuple_size: usize,
}

impl<F: Seek + Read + Write + FileOps> BufferedIter<F> {
    pub fn new(source: Box<Plan<F>>, work_dir: PathBuf, schema: Schema) -> Self {
        // TODO: Use uuid or tempfile or something.
        let file_path = work_dir.join(format!("{}.mkdb.query", &*source as *const _ as usize));

        Self {
            source,
            schema,
            work_dir,
            collected: false,
            mem_buf: vec![],
            max_size: 256,
            file_path,
            file: None,
            reader: None,
            max_buffered_tuple_size: 0,
        }
    }

    pub fn write_buffer_to_file(&mut self) -> io::Result<()> {
        if self.mem_buf.is_empty() {
            return Ok(());
        }

        if self.file.is_none() {
            self.file = Some(F::create(&self.file_path)?);
        }

        self.file.as_mut().unwrap().write_all(&self.mem_buf)?;

        self.mem_buf.clear();

        Ok(())
    }

    // TODO: This is extremely inefficient right now due to constantly
    // serializing and deserializing. We don't need to do that, tuples are
    // stored in a binary format that can be interpreted using raw pointers.
    // We need to write some unsafe to optimize this.
    pub fn collect(&mut self) -> Result<(), DbError> {
        while let Some(tuple) = self.source.try_next()? {
            let mut serialized = tuple::serialize_values(&self.schema, &tuple);

            if serialized.len() > self.max_buffered_tuple_size {
                self.max_buffered_tuple_size = serialized.len();
            }

            if self.mem_buf.len() + mem::size_of::<u32>() + serialized.len() > self.max_size {
                self.write_buffer_to_file()?
            }

            self.mem_buf
                .extend_from_slice(&(serialized.len() as u32).to_le_bytes());

            self.mem_buf.append(&mut serialized);
        }

        if let Some(mut file) = self.file.take() {
            file.rewind()?;
            self.reader = Some(BufReader::new(file));
        }

        Ok(())
    }

    pub fn try_next(&mut self) -> Result<Option<Tuple>, DbError> {
        if !self.collected {
            self.collect()?;
            self.collected = true;
        }

        // While there's stuff written to the file, return from there.
        if let Some(reader) = self.reader.as_mut() {
            if reader.has_data_left()? {
                let mut size = [0; mem::size_of::<u32>()];
                reader.read_exact(&mut size)?;

                let mut serialized = vec![0; u32::from_le_bytes(size) as usize];
                reader.read_exact(&mut serialized)?;

                return Ok(Some(tuple::deserialize_values(&serialized, &self.schema)));
            } else {
                drop(self.reader.take());
                F::destroy(&self.file_path)?;
            }
        }

        if self.mem_buf.is_empty() {
            return Ok(None);
        }

        // If there's no file or the file has been consumed, return from memory.
        let size = u32::from_le_bytes(self.mem_buf[..4].try_into().unwrap());

        let tuple = tuple::deserialize_values(&self.mem_buf[4..4 + size as usize], &self.schema);

        self.mem_buf.drain(..4 + size as usize);

        Ok(Some(tuple))
    }
}

pub(crate) struct SeqScan<F> {
    pub schema: Schema,
    pub pager: Rc<RefCell<Pager<F>>>,
    pub cursor: Cursor,
}

impl<F: Seek + Read + Write + FileOps> SeqScan<F> {
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

pub(crate) struct IndexScan<F> {
    pub index_schema: Schema,
    pub table_schema: Schema,
    pub table_root: PageNumber,
    pub index_root: PageNumber,
    pub stop_when: Option<(Vec<u8>, BinaryOperator, Box<dyn BytesCmp>)>,
    pub done: bool,
    pub pager: Rc<RefCell<Pager<F>>>,
    pub cursor: Cursor,
}

impl<F: Seek + Read + Write + FileOps> IndexScan<F> {
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

pub(crate) struct Filter<F> {
    pub source: Box<Plan<F>>,
    pub schema: Schema,
    pub filter: Expression,
}

impl<F: Seek + Read + Write + FileOps> Filter<F> {
    fn try_next(&mut self) -> Result<Option<Tuple>, DbError> {
        while let Some(tuple) = self.source.try_next()? {
            if vm::eval_where(&self.schema, &tuple, &self.filter)? {
                return Ok(Some(tuple));
            }
        }

        Ok(None)
    }
}

pub(crate) struct Project<F> {
    pub source: Box<Plan<F>>,
    pub input_schema: Schema,
    pub output_schema: Schema,
    pub projection: Vec<Expression>,
}

impl<F: Seek + Read + Write + FileOps> Project<F> {
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

pub(crate) struct Sort<F> {
    pub source: BufferedIter<F>,
    pub schema: Schema,
    pub schema_with_sort_keys: Schema,
    pub by: Vec<Expression>,
    pub sorted: bool,
    pub page_size: usize,
    pub sort_page_size: usize,
    pub input_file: Option<F>,
    pub output_file: Option<F>,
    pub mem_buf: Vec<Tuple>,
    pub total_sorted_pages: usize,
    pub next_page: usize,
    pub input_file_path: PathBuf,
    pub output_file_path: PathBuf,
}

impl<F: Seek + Read + Write + FileOps> Sort<F> {
    fn sort(&mut self) -> Result<(), DbError> {
        // mem only sorting, didn't need files
        if self.source.reader.is_none() {
            let mut tuples = Vec::new();
            while let Some(mut tuple) = self.source.try_next()? {
                for expr in &self.by {
                    let sort_key = vm::resolve_expression(&tuple, &self.schema, expr)?;
                    tuple.push(sort_key);
                }

                tuples.push(tuple);
            }

            let sort_keys = self.schema.len();
            sort_tuples(&mut tuples, sort_keys);

            self.mem_buf = tuples;
            return Ok(());
        }

        // Needs files.
        let page_size = {
            let mut t = mem::size_of::<u32>() * 2 + self.source.max_buffered_tuple_size;
            // WTF? XD?
            t -= 1;
            t |= t >> 1;
            t |= t >> 2;
            t |= t >> 4;
            t |= t >> 8;
            t |= t >> 16;
            t += 1;

            if t <= self.page_size {
                self.page_size
            } else {
                t
            }
        };

        self.sort_page_size = page_size;

        // One page runs. Sort pages and spill to disk.
        let mut tuples: Vec<Tuple> = Vec::new();

        let sort_keys_start_idx = self.schema.len();
        let mut schema_with_sort_keys = self.schema.clone();

        let mut input_pages = 0;
        let mut total_mem_page_size = 4; // 4 bytes to store num tuples
        while let Some(mut tuple) = self.source.try_next()? {
            for expr in &self.by {
                let sort_key = vm::resolve_expression(&tuple, &self.schema, expr)?;
                tuple.push(sort_key);
            }

            // Update schema to deseriliaze later.
            if schema_with_sort_keys.len() == self.schema.len() {
                let mut i = 0;
                for v in &tuple[self.schema.len()..] {
                    let mut col = Column::new(&format!("sort_key_{i}"), DataType::BigInt);
                    match v {
                        Value::Bool(_) => col.data_type = DataType::Bool,
                        Value::String(_) => col.data_type = DataType::Varchar(65535),
                        _ => {}
                    }
                    i += 1;
                    schema_with_sort_keys.push(col);
                }
                self.schema_with_sort_keys = schema_with_sort_keys.clone();
            }

            let tuple_size = tuple::size_of(&schema_with_sort_keys, &tuple);
            let total_tuple_size = 4 + tuple_size;

            // Write page if page is full
            if total_mem_page_size + total_tuple_size > page_size {
                // Create files
                if self.input_file.is_none() {
                    let input_file = self
                        .source
                        .work_dir
                        .join(format!("{}.mkdb.sort", &self.source as *const _ as usize));
                    let output_file = self
                        .source
                        .work_dir
                        .join(format!("{}.mkdb.sort", &self.mem_buf as *const _ as usize));
                    self.input_file = Some(F::create(&input_file)?);
                    self.output_file = Some(F::create(&output_file)?);
                    self.input_file_path = input_file;
                    self.output_file_path = output_file;
                }

                // sort em
                sort_tuples(&mut tuples, sort_keys_start_idx);

                // write em
                Self::write_page(
                    self.input_file.as_mut().unwrap(),
                    &schema_with_sort_keys,
                    &tuples,
                    page_size,
                )?;

                // Reset everything.
                tuples.clear();
                total_mem_page_size = 4;
                input_pages += 1;
            }

            total_mem_page_size += total_tuple_size;
            tuples.push(tuple);
        }

        // Last page.
        if !tuples.is_empty() {
            // sort em
            let sort_keys = self.schema.len();
            sort_tuples(&mut tuples, sort_keys);
            // write em!
            Self::write_page(
                self.input_file.as_mut().unwrap(),
                &schema_with_sort_keys,
                &tuples,
                page_size,
            )?;
            input_pages += 1;
            tuples.clear();
        }

        // One page run sort algorithm done. Now do the "merge" runs till fully
        // sorted.
        let mut page_runs = 2;

        let mut input_page1 = Vec::new();
        let mut input_page2 = Vec::new();
        let mut output_page = Vec::new();

        while page_runs <= (input_pages + (input_pages % 2)) {
            println!("PAGE RUNS {page_runs}\n");
            let mut output_pages = 0;

            let mut chunk = 0;

            while chunk < input_pages {
                let mut cursor1 = chunk;
                let mut cursor2 = cursor1 + page_runs / 2;

                if cursor1 < input_pages {
                    Self::read_page(
                        self.input_file.as_mut().unwrap(),
                        &mut input_page1,
                        &schema_with_sort_keys,
                        cursor1,
                        page_size,
                    )?;
                }
                cursor1 += 1;
                if cursor2 < input_pages {
                    Self::read_page(
                        self.input_file.as_mut().unwrap(),
                        &mut input_page2,
                        &schema_with_sort_keys,
                        cursor2,
                        page_size,
                    )?;
                }
                cursor2 += 1;

                // Merge
                total_mem_page_size = 4;
                while !input_page1.is_empty() || !input_page2.is_empty() {
                    let next_tuple = if !input_page1.is_empty() && !input_page2.is_empty() {
                        let cmp = cmp_tuples(&input_page1[0], &input_page2[0], sort_keys_start_idx);
                        if matches!(cmp, Ordering::Less | Ordering::Equal) {
                            let t = input_page1.remove(0);

                            if input_page1.is_empty()
                                && cursor1 < input_pages
                                && cursor1 < (chunk + page_runs / 2)
                            {
                                Self::read_page(
                                    self.input_file.as_mut().unwrap(),
                                    &mut input_page1,
                                    &schema_with_sort_keys,
                                    cursor1,
                                    page_size,
                                )?;
                                cursor1 += 1;
                            }

                            t
                        } else {
                            let t = input_page2.remove(0);

                            if input_page2.is_empty() && cursor2 < chunk + page_runs {
                                Self::read_page(
                                    self.input_file.as_mut().unwrap(),
                                    &mut input_page2,
                                    &schema_with_sort_keys,
                                    cursor2,
                                    page_size,
                                )?;
                                cursor2 += 1;
                            }

                            t
                        }
                    } else if input_page2.is_empty() {
                        let t = input_page1.remove(0);

                        if input_page1.is_empty()
                            && cursor1 < input_pages
                            && cursor1 < (chunk + page_runs / 2)
                        {
                            Self::read_page(
                                self.input_file.as_mut().unwrap(),
                                &mut input_page1,
                                &schema_with_sort_keys,
                                cursor1,
                                page_size,
                            )?;
                            cursor1 += 1;
                        }

                        t
                    } else {
                        let t = input_page2.remove(0);

                        if input_page2.is_empty()
                            && cursor2 < chunk + page_runs
                            && cursor2 < input_pages
                        {
                            Self::read_page(
                                self.input_file.as_mut().unwrap(),
                                &mut input_page2,
                                &schema_with_sort_keys,
                                cursor2,
                                page_size,
                            )?;
                            cursor2 += 1;
                        }

                        t
                    };

                    let tuple_size = tuple::size_of(&schema_with_sort_keys, &next_tuple);
                    let total_tuple_size = 4 + tuple_size;

                    if total_mem_page_size + total_tuple_size > page_size {
                        println!("{output_page:?}");
                        Self::write_page(
                            self.output_file.as_mut().unwrap(),
                            &schema_with_sort_keys,
                            &output_page,
                            page_size,
                        )?;
                        output_pages += 1;
                        total_mem_page_size = 4;
                        output_page.clear();
                    }

                    output_page.push(next_tuple);
                    total_mem_page_size += total_tuple_size;
                }

                if !output_page.is_empty() {
                    println!("{output_page:?}");

                    Self::write_page(
                        self.output_file.as_mut().unwrap(),
                        &schema_with_sort_keys,
                        &output_page,
                        page_size,
                    )?;
                    output_pages += 1;
                    output_page.clear();
                }

                chunk += page_runs;
                println!("\n\n")
            }

            // Now swap
            let mut input_file = self.input_file.take().unwrap();
            let output_file = self.output_file.take().unwrap();

            input_file.truncate()?;

            self.input_file = Some(output_file);
            self.output_file = Some(input_file);

            page_runs *= 2;
            input_pages = output_pages;
        }

        self.total_sorted_pages = input_pages;

        Ok(())
    }

    fn try_next(&mut self) -> Result<Option<Tuple>, DbError> {
        if !self.sorted {
            self.source.collect()?;
            self.sort()?;
            self.sorted = true;
        }

        if self.mem_buf.is_empty() {
            if let Some(input_file) = self.input_file.as_mut() {
                if self.next_page < self.total_sorted_pages {
                    Self::read_page(
                        input_file,
                        &mut self.mem_buf,
                        &self.schema_with_sort_keys,
                        self.next_page,
                        self.sort_page_size,
                    )?;
                    self.next_page += 1;
                } else {
                    // TODO: Drop if iter not consumed.
                    drop(self.input_file.take());
                    drop(self.output_file.take());
                    F::destroy(&self.input_file_path)?;
                    F::destroy(&self.output_file_path)?;
                }
            }
        }

        if self.mem_buf.is_empty() {
            return Ok(None);
        }

        let mut tuple = self.mem_buf.remove(0);
        tuple.drain(self.schema.len()..);

        Ok(Some(tuple))
    }

    fn write_page(
        file: &mut F,
        schema: &Schema,
        tuples: &Vec<Tuple>,
        page_size: usize,
    ) -> io::Result<()> {
        // num tuples
        file.write_all(&(tuples.len() as u32).to_le_bytes())?;
        let mut bytes_written = 4;

        // tuple size + tuple content
        for i in 0..tuples.len() {
            let serialized = tuple::serialize_values(schema, &tuples[i]);
            file.write_all(&(serialized.len() as u32).to_le_bytes())?;
            file.write_all(&serialized)?;

            bytes_written += 4 + serialized.len();
        }
        // Rest of page if not 100% full
        file.write_all(&vec![0; page_size - bytes_written])?;

        Ok(())
    }

    fn read_page(
        file: &mut F,
        into: &mut Vec<Tuple>,
        schema: &Schema,
        page: usize,
        page_size: usize,
    ) -> io::Result<()> {
        file.seek(io::SeekFrom::Start((page_size * page) as u64))?;

        let mut n_tuples = [0; 4];
        file.read_exact(&mut n_tuples)?;

        let n_tuples = u32::from_le_bytes(n_tuples);

        for _ in 0..n_tuples {
            let mut size = [0; 4];
            file.read_exact(&mut size)?;
            let mut tup = vec![0; u32::from_le_bytes(size) as usize];
            file.read_exact(&mut tup)?;
            into.push(tuple::deserialize_values(&tup, schema));
        }

        Ok(())
    }
}

fn sort_tuples(tuples: &mut Vec<Vec<Value>>, sort_keys_start_idx: usize) {
    tuples.sort_by(|t1, t2| cmp_tuples(t1, t2, sort_keys_start_idx));
}

fn cmp_tuples(t1: &Tuple, t2: &Tuple, sort_keys_start_idx: usize) -> Ordering {
    for (a, b) in t1[sort_keys_start_idx..]
        .iter()
        .zip(&t2[sort_keys_start_idx..])
    {
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
}

pub(crate) struct Insert<F> {
    pub root: PageNumber,
    pub pager: Rc<RefCell<Pager<F>>>,
    pub source: Box<Plan<F>>,
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

pub(crate) struct Update<F> {
    pub root: PageNumber,
    pub assignments: Vec<Assignment>,
    pub pager: Rc<RefCell<Pager<F>>>,
    pub source: BufferedIter<F>,
    pub schema: Schema,
}

impl<F: Seek + Read + Write + FileOps> Update<F> {
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

pub(crate) struct Delete<F> {
    pub root: PageNumber,
    pub pager: Rc<RefCell<Pager<F>>>,
    pub source: BufferedIter<F>,
    pub schema: Schema,
    pub indexes: Vec<IndexMetadata>,
}

impl<F: Seek + Read + Write + FileOps> Delete<F> {
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
