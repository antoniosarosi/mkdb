//! Code that runs on parsed SQL statements.
//!
//! This is where we generate query plans that the virtual machine will execute.

mod optimizer;

pub(crate) mod planner;
