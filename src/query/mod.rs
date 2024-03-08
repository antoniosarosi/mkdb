//! Code that runs on parsed SQL statements.

mod optimizer;

pub(crate) mod planner;

pub(crate) use planner::generate_plan;
