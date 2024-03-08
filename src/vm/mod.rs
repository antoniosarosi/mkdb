//! This module contains code that interprets SQL [`crate::sql::Statement`]
//! instances.
//!
//! TODO: we should make a real "virtual machine" like the one in
//! [SQLite 2](https://www.sqlite.org/vdbe.html) or an executor with JIT and
//! stuff like Postgres or something similar instead of interpreting the raw
//! [`crate::sql::Statement`] trees. But this is good enough for now.

mod executor;
mod expression;

pub(crate) use executor::{btree_new, exec};
pub(crate) use expression::{eval_optional_where, eval_where, resolve_expression};
