//! Code that executes [`crate::query::plan::Plan`] trees.

use std::io::{Read, Seek, Write};

use crate::{
    db::QueryResult,
    paging,
    query::plan::{BufferedIter, Plan},
};

pub(crate) fn exec_plan<I: Seek + Read + Write + paging::io::Sync>(plan: Plan<I>) -> QueryResult {
    let mut iter = BufferedIter::new(Box::new(plan));
    iter.collect()?;

    Ok(iter.collection)
}

// TODO: Move the plan iterators here or something.