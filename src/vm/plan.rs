//! Code that executes [`Plan`] trees.

use std::{
    cell::RefCell,
    cmp::Ordering,
    collections::VecDeque,
    io::{self, BufRead, BufReader, Read, Seek, Write},
    mem,
    ops::Index,
    path::PathBuf,
    rc::Rc,
};

use crate::{
    db::{DbError, IndexMetadata, Projection, RowId, Schema, SqlError},
    paging::{
        io::FileOps,
        pager::{PageNumber, Pager},
    },
    sql::statement::{Assignment, BinaryOperator, Column, DataType, Expression, Value},
    storage::{reassemble_payload, tuple, BTree, BytesCmp, Cursor, FixedSizeMemCmp},
    vm,
};

pub(crate) fn exec<F: Seek + Read + Write + FileOps>(plan: Plan<F>) -> Result<Projection, DbError> {
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

#[derive(Debug)]
pub(crate) struct TupleBuffer {
    page_size: usize,
    current_size: usize,
    largest_tuple_size: usize,
    packed: bool,
    schema: Schema,
    tuples: VecDeque<Tuple>,
}

impl Index<usize> for TupleBuffer {
    type Output = Tuple;

    fn index(&self, index: usize) -> &Self::Output {
        &self.tuples[index]
    }
}

impl TupleBuffer {
    fn new(page_size: usize, schema: Schema, packed: bool) -> Self {
        Self {
            page_size,
            schema,
            packed,
            current_size: if packed { 0 } else { mem::size_of::<u32>() },
            largest_tuple_size: 0,
            tuples: VecDeque::new(),
        }
    }

    pub fn can_fit(&self, tuple: &Tuple) -> bool {
        let tuple_size = tuple::size_of(&tuple, &self.schema);
        let total_tuple_size = mem::size_of::<u32>() + tuple_size;

        self.current_size + total_tuple_size <= self.page_size
    }

    pub fn push(&mut self, tuple: Tuple) {
        let tuple_size = tuple::size_of(&tuple, &self.schema);
        let total_tuple_size = mem::size_of::<u32>() + tuple_size;

        if tuple_size > self.largest_tuple_size {
            self.largest_tuple_size = tuple_size;
        }

        self.current_size += total_tuple_size;
        self.tuples.push_back(tuple);
    }

    pub fn pop_front(&mut self) -> Option<Tuple> {
        self.tuples.pop_front().inspect(|tuple| {
            self.current_size -= mem::size_of::<u32>() + tuple::size_of(tuple, &self.schema);
        })
    }

    pub fn is_empty(&self) -> bool {
        self.tuples.is_empty()
    }

    pub fn clear(&mut self) {
        self.tuples.clear();
        self.current_size = mem::size_of::<u32>();
    }

    pub fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(self.page_size);

        if !self.packed {
            buf.extend_from_slice(&(self.tuples.len() as u32).to_le_bytes());
        }

        for tuple in &self.tuples {
            let serialized = tuple::serialize(&self.schema, &tuple);
            buf.extend_from_slice(&(serialized.len() as u32).to_le_bytes());
            buf.extend_from_slice(&serialized);
        }

        if !self.packed {
            buf.resize(self.page_size, 0);
        }

        buf
    }

    pub fn write_to(&self, file: &mut impl Write) -> io::Result<()> {
        file.write_all(&self.serialize())
    }

    pub fn read_from(&mut self, file: &mut impl Read) -> io::Result<()> {
        assert!(
            self.is_empty() && !self.packed,
            "read_from() only works with fixed size empty buffers"
        );

        let mut buf = vec![0; self.page_size];
        file.read_exact(&mut buf)?;

        let number_of_tuples = u32::from_le_bytes(buf[..mem::size_of::<u32>()].try_into().unwrap());
        let mut index = mem::size_of::<u32>();

        for _ in 0..number_of_tuples {
            let size = u32::from_le_bytes(
                buf[index..index + mem::size_of::<u32>()]
                    .try_into()
                    .unwrap(),
            ) as usize;

            index += mem::size_of::<u32>();

            self.push(tuple::deserialize(&buf[index..index + size], &self.schema));
            index += size;
        }

        Ok(())
    }

    pub fn read_page(&mut self, file: &mut (impl Seek + Read), page: usize) -> io::Result<()> {
        file.seek(io::SeekFrom::Start((self.page_size * page) as u64))?;
        self.read_from(file)
    }

    pub fn page_size_needed_for(tuple_size: usize) -> usize {
        let mut page_size = mem::size_of::<u32>() * 2 + tuple_size;
        page_size -= 1;
        page_size |= page_size >> 1;
        page_size |= page_size >> 2;
        page_size |= page_size >> 4;
        page_size |= page_size >> 8;
        page_size |= page_size >> 16;
        page_size += 1;

        page_size
    }
}

/// Similar to [`io::BufReader`] and [`io::BufWriter`].
///
/// This structure consumes all the tuples from its source through the
/// [`Self::collect`] operation and writes them to a "collection" file if they
/// don't fit in memory. Once tuples have been successfully collected,
/// [`BufferedIter`] acts like a normal iterator that returns tuples one at a
/// time. If the collected tuples fit in memory there is no IO, so that's the
/// best case scenario.
pub(crate) struct BufferedIter<F> {
    /// Tuple source. This is where we collect from.
    pub source: Box<Plan<F>>,
    /// Tuple schema.
    pub schema: Schema,
    /// `true` if [`Self::collect`] completed successfully.
    pub collected: bool,
    /// In-memory buffer that stores tuples from the source.
    pub mem_buf: TupleBuffer,
    /// File handle/descriptor in case we had to create the collection file.
    pub file: Option<F>,
    /// Buffered reader in case we created the file and have to read from it.
    pub reader: Option<BufReader<F>>,
    /// Path of the collection file.
    pub file_path: PathBuf,
    /// Executes the expressions and appends the results to each tuple.
    pub append_exprs: Vec<Expression>,
}

impl<F: FileOps> BufferedIter<F> {
    /// Drops the IO resource and deletes it from the file system.
    fn drop_file(&mut self) -> io::Result<()> {
        drop(self.file.take());
        drop(self.reader.take());
        F::remove(&self.file_path)
    }
}

// TODO: Requires defining the struct as BufferdIter<F: FileOps>
// impl<F: FileOps> Drop for BufferedIter<F> {
//     fn drop(&mut self) {
//         if self.file.is_some() {
//             self.drop_file();
//         }
//     }
// }

impl<F: Seek + Read + Write + FileOps> BufferedIter<F> {
    pub fn new(
        source: Box<Plan<F>>,
        work_dir: PathBuf,
        schema: Schema,
        append_exprs: Vec<Expression>,
    ) -> Self {
        // TODO: Use uuid or tempfile or something. This is poor man's random
        // file name.
        let file_path = work_dir.join(format!("{}.mkdb.query", &*source as *const _ as usize));

        Self {
            source,
            mem_buf: TupleBuffer::new(256, schema.clone(), true),
            schema,
            collected: false,
            file_path,
            file: None,
            reader: None,
            append_exprs,
        }
    }

    /// Collects all the tuples from [`Self::source`].
    fn collect(&mut self) -> Result<(), DbError> {
        // Buffer tuples in-memory until we have no space left. At that point
        // create the file if it doesn't exist, write the buffer to disk and
        // repeat until there are no more tuples.
        while let Some(mut tuple) = self.source.try_next()? {
            for expr in &self.append_exprs {
                tuple.push(vm::resolve_expression(&tuple, &self.schema, expr)?);
            }

            if !self.mem_buf.can_fit(&tuple) {
                if self.file.is_none() {
                    self.file = Some(F::create(&self.file_path)?);
                }
                self.mem_buf.write_to(self.file.as_mut().unwrap())?;
                self.mem_buf.clear();
            }

            self.mem_buf.push(tuple);
        }

        // If we ended up creating a file and writing to it we must set the
        // cursor position back to the first byte in order to read from it
        // later.
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

        // While there's stuff written to the file return from there.
        if let Some(reader) = self.reader.as_mut() {
            if reader.has_data_left()? {
                let mut size = [0; mem::size_of::<u32>()];
                reader.read_exact(&mut size)?;

                let mut serialized = vec![0; u32::from_le_bytes(size) as usize];
                reader.read_exact(&mut serialized)?;

                return Ok(Some(tuple::deserialize(&serialized, &self.schema)));
            }

            // Reader is done, drop the file.
            self.drop_file()?;
        }

        // If there's no file or the file has been consumed return from memory.
        // Tuples that were not written to the file because it wasn't necessary
        // are also returned here.
        Ok(self.mem_buf.pop_front())
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

        Ok(Some(tuple::deserialize(
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
            tuple::deserialize(&pager.get(page)?.cell(slot).content, &self.index_schema);

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

        Ok(Some(tuple::deserialize(
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
    pub sort_schema: Schema,
    pub sorted: bool,
    pub page_size: usize,
    pub input_file: Option<F>,
    pub output_file: Option<F>,
    pub mem_buf: Vec<Tuple>,
    pub total_sorted_pages: usize,
    pub next_page: usize,
    pub work_dir: PathBuf,
    pub input_file_path: PathBuf,
    pub output_file_path: PathBuf,
}

impl<F: Seek + Read + Write + FileOps> Sort<F> {
    fn cmp_tuples(&self, t1: &[Value], t2: &[Value]) -> Ordering {
        let sort_keys_start_index = self.schema.len();

        for (a, b) in t1[sort_keys_start_index..]
            .iter()
            .zip(&t2[sort_keys_start_index..])
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

    fn sort_tuples(&self, tuples: &mut [Tuple]) {
        tuples.sort_by(|t1, t2| self.cmp_tuples(t1, t2));
    }

    fn sort(&mut self) -> Result<(), DbError> {
        // Mem only sorting, didn't need files.
        if self.source.reader.is_none() {
            let mut tuples = Vec::new();
            while let Some(tuple) = self.source.try_next()? {
                tuples.push(tuple);
            }

            self.sort_tuples(&mut tuples);

            self.mem_buf = tuples;
            return Ok(());
        }

        // Needs files.
        self.page_size = std::cmp::max(
            TupleBuffer::page_size_needed_for(self.source.mem_buf.largest_tuple_size),
            self.page_size,
        );

        // One page runs. Sort pages and spill to disk.
        let mut tuples = TupleBuffer::new(self.page_size, self.sort_schema.clone(), false);

        let mut input_pages = 0;
        while let Some(tuple) = self.source.try_next()? {
            // Write page if page is full
            if !tuples.can_fit(&tuple) {
                // Create files
                if self.input_file.is_none() {
                    self.input_file_path = self
                        .work_dir
                        .join(format!("{}.mkdb.sort", &self.source as *const _ as usize));
                    self.output_file_path = self
                        .work_dir
                        .join(format!("{}.mkdb.sort", &self.mem_buf as *const _ as usize));
                    self.input_file = Some(F::create(&self.input_file_path)?);
                    self.output_file = Some(F::create(&self.output_file_path)?);
                }

                self.sort_tuples(tuples.tuples.make_contiguous());
                tuples.write_to(self.input_file.as_mut().unwrap())?;

                tuples.clear();
                input_pages += 1;
            }

            tuples.push(tuple);
        }

        // Last page.
        if !tuples.is_empty() {
            self.sort_tuples(&mut tuples.tuples.make_contiguous());
            tuples.write_to(self.input_file.as_mut().unwrap())?;

            tuples.clear();
            input_pages += 1;
        }

        // One page run sort algorithm done. Now do the "merge" runs till fully
        // sorted.
        let mut page_runs = 2;

        let mut input_page1 = TupleBuffer::new(self.page_size, self.sort_schema.clone(), false);
        let mut input_page2 = TupleBuffer::new(self.page_size, self.sort_schema.clone(), false);
        let mut output_page = TupleBuffer::new(self.page_size, self.sort_schema.clone(), false);

        while page_runs <= (input_pages + (input_pages % 2)) {
            let mut output_pages = 0;

            let mut chunk = 0;

            while chunk < input_pages {
                let mut cursor1 = chunk;
                let mut cursor2 = cursor1 + page_runs / 2;

                if cursor1 < input_pages {
                    input_page1.read_page(self.input_file.as_mut().unwrap(), cursor1)?;
                }
                cursor1 += 1;
                if cursor2 < input_pages {
                    input_page2.read_page(self.input_file.as_mut().unwrap(), cursor2)?;
                }
                cursor2 += 1;

                // Merge
                while !input_page1.is_empty() || !input_page2.is_empty() {
                    let tuple = if !input_page1.is_empty() && !input_page2.is_empty() {
                        let cmp = self.cmp_tuples(&input_page1[0], &input_page2[0]);
                        if matches!(cmp, Ordering::Less | Ordering::Equal) {
                            input_page1.pop_front().unwrap()
                        } else {
                            input_page2.pop_front().unwrap()
                        }
                    } else if input_page2.is_empty() {
                        input_page1.pop_front().unwrap()
                    } else {
                        input_page2.pop_front().unwrap()
                    };

                    if input_page1.is_empty()
                        && cursor1 < input_pages
                        && cursor1 < (chunk + page_runs / 2)
                    {
                        input_page1.read_page(self.input_file.as_mut().unwrap(), cursor1)?;
                        cursor1 += 1;
                    }

                    if input_page2.is_empty()
                        && cursor2 < chunk + page_runs
                        && cursor2 < input_pages
                    {
                        input_page2.read_page(self.input_file.as_mut().unwrap(), cursor2)?;
                        cursor2 += 1;
                    }

                    if !output_page.can_fit(&tuple) {
                        output_page.write_to(self.output_file.as_mut().unwrap())?;
                        output_page.clear();
                        output_pages += 1;
                    }

                    output_page.push(tuple);
                }

                if !output_page.is_empty() {
                    output_page.write_to(self.output_file.as_mut().unwrap())?;
                    output_page.clear();
                    output_pages += 1;
                }

                chunk += page_runs;
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
                        &self.sort_schema,
                        self.next_page,
                        self.page_size,
                    )?;
                    self.next_page += 1;
                } else {
                    // TODO: Drop if iter not consumed.
                    drop(self.input_file.take());
                    drop(self.output_file.take());
                    F::remove(&self.input_file_path)?;
                    F::remove(&self.output_file_path)?;
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
            let serialized = tuple::serialize(schema, &tuples[i]);
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
            into.push(tuple::deserialize(&tup, schema));
        }

        Ok(())
    }
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
        btree.insert(tuple::serialize(&self.schema, &tuple))?;

        for IndexMetadata { root, column, .. } in &self.indexes {
            let idx = self.schema.index_of(column).unwrap();

            assert_eq!(self.schema.columns[0].name, "row_id");
            let key_col = self.schema.columns[idx].clone();
            let row_id_col = self.schema.columns[0].clone();

            let key = tuple[idx].clone();
            let row_id = tuple[0].clone();

            let entry = tuple::serialize(&Schema::from(vec![key_col, row_id_col]), &[key, row_id]);

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

        btree.insert(tuple::serialize(&self.schema, &tuple))?;

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

        btree.remove(&tuple::serialize(&self.schema, &tuple))?;

        for IndexMetadata { root, column, .. } in &self.indexes {
            let idx = self.schema.index_of(column).unwrap();
            let key_col = self.schema.columns[idx].clone();
            let key = tuple[idx].clone();

            let entry = tuple::serialize(&Schema::from(vec![key_col]), &[key]);

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
