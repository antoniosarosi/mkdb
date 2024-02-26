//! Code that runs on parsed SQL statements.
//!
//! The submodules in here could be placed in [`crate::sql`] but as of right now
//! [`crate::sql`] is only concerned about parsing, it doesn't care about
//! "table does not exist" or similar errors, so having this code separate seems
//! more logical.

pub(crate) mod analyzer;
pub(crate) mod optimizer;
