//! Generates a query plan.

use crate::{db::Projection, paging::pager::PageNumber, sql::statement::Expression};

enum Node {
    Scan(PageNumber),
    Filter(Expression),
    Project,
    Modify,
}
