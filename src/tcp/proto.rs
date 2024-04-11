//! Simple protocol inspired by [RESP] (Redis Serialization Protocol).
//!
//! [RESP]: https://redis.io/docs/reference/protocol-spec/
//!
//! # Client Side
//!
//! Clients can only do 3 actions:
//!
//! - Open connection.
//! - Send one raw SQL statement (UTF-8 string) at a time.
//! - Close connection.
//!
//! # Server Side
//!
//! The server will execute the SQL statement that the client sends and respond
//! with one of the [`Response`] variants. The serialization format depends on
//! the variant, but there are a couple of rules applied to all of them:
//!
//! 1. All responses must include a header that contains a 4 byte little endian
//! integer which indicates the length of the payload in bytes.
//!
//! 2. All responses must start with a one byte ASCII prefix.
//!
//! ## Empty Set
//!
//! [`Response::EmptySet`] indicates that the SQL statement was executed
//! successfuly but did not produce any results. It might have affected a number
//! of rows (for example by deleting them) but there is no row/column data to
//! return. Instead, this response only contains the number of affected rows.
//!
//! The empty set payload is encoded with the prefix `!` followed by a 4 byte
//! little endian integer that stores the number of affected rows. For an empty
//! set response that affected 9 rows the complete packet would be as follows:
//!
//! ```text
//!   Payload   ASCII   Affected
//!     Len     Prefix    Rows
//! +---------+-------+---------+
//! | 5 0 0 0 |  '!'  | 9 0 0 0 |
//! +---------+-------+---------+
//!   4 bytes  1 byte   4 bytes
//! ```
//!
//! ## Error
//!
//! All error responses are just simple strings. The prefix for
//! [`Response::Err`] is `-`. After the prefix there's a 2 byte little endian
//! integer that encodes the length of the error message in bytes followed by
//! the UTF-8 string itself. For a response error with the message
//! `table "test" does not exist` the complete packet would be:
//!
//! ```text
//!   Payload   ASCII   Message
//!     Len     Prefix  Length        Error Message
//! +----------+-------+------+-----------------------------+
//! | 30 0 0 0 |  '-'  | 27 0 | table "test" does not exist |
//! +----------+-------+------+-----------------------------+
//!   4 bytes   1 byte  2 bytes         27 bytes
//! ```
//!
//! ## Query Set
//!
//! [`Response::QuerySet`] includes the schema of the returned values and
//! the returned values themselves. The prefix for this variant is `+`.
//! Following the prefix there's a 2 byte little endian integer that encodes
//! the number of columns in the schema. After that, columns are serialized in
//! the following format:
//!
//! ```text
//!    Name     Column     Data
//!    Len       Name      Type
//! +-------+-------------+-----+
//! |  5 0  | column name |  1  |
//! +-------+-------------+-----+
//! 2 bytes    N bytes     1 byte
//! ```
//!
//! The name of the column is a UTF-8 string prefixed by a 2 byte little endian
//! integer corresponding to its byte length (ideally columns shouldn't have
//! names longer than 65535 bytes). The final component of a column is one byte
//! that encodes the column [`DataType`] variant. The mapping is as follows:
//!
//! ```ignore
//! match col.data_type {
//!     DataType::Bool => 0,
//!     DataType::Int => 1,
//!     DataType::UnsignedInt => 2,
//!     DataType::BigInt => 3,
//!     DataType::UnsignedBigInt => 4,
//!     DataType::Varchar(_) => 5,
//! }
//! ```
//!
//! The maximum number of characters in `VARCHAR` types is not encoded anywhere.
//! It's only used to check the length in characters of a string before
//! inserting it into the database.
//!
//! Finally, after all the columns, the response packet encodes the tuple
//! results prefixed by a 4 byte little endian integer that indicates the total
//! number of tuples. Tuples are encoded using the exact same format that we
//! use to store them in the database. Refer to [`crate::storage::tuple`] for
//! details, but in a nutshell the tuple `(1, "hello", 3)` encoded with the
//! data types `[BigInt, Varchar, Int]` looks like this:
//!
//! ```text
//! +-----------------+---------+---------------------+---------+
//! | 0 0 0 0 0 0 0 1 | 5 0 0 0 | 'h' 'e' 'l' 'l' 'o' | 0 0 0 2 |
//! +-----------------+---------+---------------------+---------+
//!  8 byte big endian  4 byte       String bytes       4 byte
//!       BigInt        little                        big endian
//!                     endian                            Int
//!                     String
//!                     length
//! ```
//!
//! So, putting it all together and assuming that the names of the columns
//! for the data types mentioned above are `("id", "msg", "num")`, a complete
//! packet that encodes the 2 tuples `[(1, "hello", 2), (2, "world", 4)]` would
//! look like this:
//!
//! ```text
//!   Payload   ASCII   Num of    Name            Name            Name              Num
//!     Len     Prefix  Columns   Len     Name    Len     Name    Len     Name     Tuples
//! +----------+-------+-------+-------+--------+-------+-------+-------+-------+---------+
//! | 65 0 0 0 |  '+'  |  3 0  |  2 0  |  "id"  |  3 0  | "msg" |  3 0  | "num" | 2 0 0 0 |
//! +----------+-------+-------+-------+--------+-------+-------+-------+-------+---------+
//!   4 bytes   1 byte  4 bytes 2 bytes 2 bytes  2 bytes 3 bytes 2 bytes 3 bytes  4 bytes
//!
//!     id column                msg column            num column
//! +-----------------+---------+---------------------+---------+
//! | 0 0 0 0 0 0 0 1 | 5 0 0 0 | 'h' 'e' 'l' 'l' 'o' | 0 0 0 2 |
//! +-----------------+---------+---------------------+---------+
//!  8 byte big endian  4 bytes         5 bytes         4 byte
//!
//!     id column                msg column            num column
//! +-----------------+---------+---------------------+---------+
//! | 0 0 0 0 0 0 0 2 | 5 0 0 0 | 'w' 'o' 'r' 'l' 'd' | 0 0 0 4 |
//! +-----------------+---------+---------------------+---------+
//!  8 byte big endian  4 bytes         5 bytes         4 byte
//! ```
use std::{array::TryFromSliceError, fmt, num::TryFromIntError, string::FromUtf8Error};

use crate::{
    db::{DbError, QuerySet},
    sql::statement::{Column, DataType},
    storage::tuple,
    Value,
};

/// Errors that we can find while serializing or deserializing packets.
#[derive(Debug, PartialEq)]
pub enum EncodingError {
    IntConversion(TryFromIntError),
    SliceConversion(String),
    UtfDecode(FromUtf8Error),
    InvalidPrefix(u8),
    InvalidDataType(u8),
}

impl From<TryFromIntError> for EncodingError {
    fn from(e: TryFromIntError) -> Self {
        Self::IntConversion(e)
    }
}

impl From<TryFromSliceError> for EncodingError {
    fn from(e: TryFromSliceError) -> Self {
        Self::SliceConversion(e.to_string())
    }
}

impl From<FromUtf8Error> for EncodingError {
    fn from(e: FromUtf8Error) -> Self {
        Self::UtfDecode(e)
    }
}

impl fmt::Display for EncodingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IntConversion(e) => write!(f, "{e}"),
            Self::SliceConversion(message) => f.write_str(message),
            Self::UtfDecode(e) => write!(f, "{e}"),
            Self::InvalidPrefix(prefix) => write!(f, "invalid ASCII prefix: {prefix}"),
            Self::InvalidDataType(byte) => write!(f, "invalid data type: {byte}"),
        }
    }
}

/// Response type. See the [`self`] module documentation.
#[derive(Debug, PartialEq)]
pub enum Response {
    QuerySet(QuerySet),
    EmptySet(usize),
    Err(String),
}

impl From<Result<QuerySet, DbError>> for Response {
    fn from(result: Result<QuerySet, DbError>) -> Self {
        match result {
            Ok(empty_set) if empty_set.schema.columns.is_empty() => {
                // See [`crate::db::PreparedStatement::try_next`].
                let affected_rows = match empty_set.tuples.first().map(|t| t.as_slice()) {
                    Some([Value::Number(number)]) => *number as usize,
                    _ => empty_set.tuples.len(),
                };

                Response::EmptySet(affected_rows)
            }

            Ok(query_set) => Response::QuerySet(query_set),

            Err(e) => Response::Err(e.to_string()),
        }
    }
}

/// Returns a complete serialized packet (including the header).
///
/// See the module level documentation for details.
pub fn serialize(payload: &Response) -> Result<Vec<u8>, EncodingError> {
    let mut packet = Vec::from(0u32.to_le_bytes());

    match payload {
        Response::Err(e) => {
            packet.push(b'-');
            packet.extend(e.as_bytes());
        }

        Response::EmptySet(affected_rows) => {
            packet.push(b'!');
            packet.extend_from_slice(&(u32::try_from(*affected_rows)?).to_le_bytes());
        }

        Response::QuerySet(query_set) => {
            packet.push(b'+');
            packet.extend_from_slice(&(u16::try_from(query_set.schema.len())?).to_le_bytes());
            for col in &query_set.schema.columns {
                packet.extend_from_slice(&(u16::try_from(col.name.len())?).to_le_bytes());
                packet.extend_from_slice(col.name.as_bytes());
                packet.push(match col.data_type {
                    DataType::Bool => 0,
                    DataType::Int => 1,
                    DataType::UnsignedInt => 2,
                    DataType::BigInt => 3,
                    DataType::UnsignedBigInt => 4,
                    DataType::Varchar(_) => 5,
                });
            }
            packet.extend_from_slice(&(u32::try_from(query_set.tuples.len())?).to_le_bytes());
            for tuple in &query_set.tuples {
                packet.extend_from_slice(&tuple::serialize(&query_set.schema, tuple));
            }
        }
    }

    let payload_len = u32::try_from(packet[4..].len())?;

    packet[..4].copy_from_slice(&payload_len.to_le_bytes());

    Ok(packet)
}

/// Deserializes the payload of a packet (the header must be skipped).
pub fn deserialize(payload: &[u8]) -> Result<Response, EncodingError> {
    Ok(match payload[0] {
        b'-' => Response::Err(String::from_utf8(Vec::from(&payload[1..]))?),

        b'!' => {
            let affected_rows = u32::from_le_bytes(payload[1..].try_into()?) as usize;
            Response::EmptySet(affected_rows)
        }

        b'+' => {
            let mut query_set = QuerySet::empty();
            let mut cursor = 1;

            let schema_len = u16::from_le_bytes(payload[cursor..cursor + 2].try_into()?);
            cursor += 2;

            for _ in 0..schema_len {
                let name_len = u16::from_le_bytes(payload[cursor..cursor + 2].try_into()?) as usize;
                cursor += 2;

                let name = String::from_utf8(Vec::from(&payload[cursor..cursor + name_len]))?;
                cursor += name_len;

                let data_type = match payload[cursor] {
                    0 => DataType::Bool,
                    1 => DataType::Int,
                    2 => DataType::UnsignedInt,
                    3 => DataType::BigInt,
                    4 => DataType::UnsignedBigInt,
                    5 => DataType::Varchar(65535),
                    invalid => Err(EncodingError::InvalidDataType(invalid))?,
                };
                cursor += 1;

                query_set.schema.push(Column::new(&name, data_type));
            }

            let num_tuples = u32::from_le_bytes(payload[cursor..cursor + 4].try_into()?);
            cursor += 4;

            for _ in 0..num_tuples {
                let tuple = tuple::deserialize(&payload[cursor..], &query_set.schema);
                cursor += tuple::size_of(&tuple, &query_set.schema);
                query_set.tuples.push(tuple);
            }

            Response::QuerySet(query_set)
        }

        prefix => Err(EncodingError::InvalidPrefix(prefix))?,
    })
}

#[cfg(test)]
mod test {
    use super::EncodingError;
    use crate::{
        db::{QuerySet, Schema},
        sql::statement::{Column, DataType, Value},
        tcp::proto::{deserialize, serialize, Response},
        DbError,
    };

    #[test]
    fn serialize_deserialize_query_set() -> Result<(), EncodingError> {
        let payload = Response::QuerySet(QuerySet::new(
            Schema::new(vec![
                Column::new("id", DataType::UnsignedBigInt),
                Column::new("name", DataType::Varchar(65535)),
                Column::new("email", DataType::Varchar(65535)),
                Column::new("age", DataType::UnsignedInt),
            ]),
            vec![
                vec![
                    Value::Number(1),
                    Value::String("John Doe".into()),
                    Value::String("john@doe.com".into()),
                    Value::Number(20),
                ],
                vec![
                    Value::Number(2),
                    Value::String("Some Dude".into()),
                    Value::String("some@dude.com".into()),
                    Value::Number(22),
                ],
                vec![
                    Value::Number(3),
                    Value::String("Jane Doe".into()),
                    Value::String("jane@doe.com".into()),
                    Value::Number(24),
                ],
            ],
        ));

        let packet = serialize(&payload)?;

        assert_eq!(deserialize(&packet[4..])?, payload);

        Ok(())
    }

    #[test]
    fn serialize_deserialize_empty_set() -> Result<(), EncodingError> {
        let empty_set = QuerySet::new(Schema::new(vec![]), vec![vec![], vec![], vec![]]);
        let response = Response::from(Ok(empty_set));
        let packet = serialize(&response)?;

        assert_eq!(deserialize(&packet[4..])?, Response::EmptySet(3));

        Ok(())
    }

    #[test]
    fn serialize_deserialize_err() -> Result<(), EncodingError> {
        let response = Response::from(Err(DbError::Other("custom error".into())));
        let packet = serialize(&response)?;

        assert_eq!(
            deserialize(&packet[4..])?,
            Response::Err("custom error".into())
        );

        Ok(())
    }
}
