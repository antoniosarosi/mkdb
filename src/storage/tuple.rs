//! A tuple is a single database row or any collection of [`Value`] instances.
//!
//! TODO: Right now we're "serializing" and "deserializing" rows which is not
//! really needed. We already store the rows in such a format that we can
//! interpret the bytes in them as numbers or strings without having to copy
//! them into [`Value`] structures. Serializing and deserializing made it easy
//! to develop in the beginning because it doesn't require any unsafe code, but
//! it's probably the biggest performance hit not counting unoptimized IO.
//!
//! # Serialization Format
//!
//! All numbers are serialized in big endian format because that allows the
//! BTree to compare them using a simple memcmp(). Normally only the first
//! column of a tuple needs to be compared as that's where we store the
//! [`RowId`], but for simplicity we just encode every number in big endian.
//!
//! Strings on the other hand are UTF-8 encoded with a 4 byte little endian
//! prefix where we store the byte length of the string (number of bytes, not
//! number of characters). So, putting it all together, a tuple like this one:
//!
//! ```ignore
//! [
//!     Value::Number(1),
//!     Value::String("hello".into()),
//!     Value::Number(2),
//! ]
//! ```
//!
//! with a schema like this one:
//!
//! ```ignore
//! [DataType::BigInt, DataType::Varchar(255), DataType::Int]
//! ```
//!
//! would serialize into the following bytes (not bits, bytes):
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
//! The only thing we're missing here is alignment. The page module already
//! supports 64 bit alignment, so if we align columns and write some unsafe
//! code to obtain references to values from a binary buffer we would get rid
//! of serialization / deserialization. It would require some changes throughout
//! the codebase, but definitely doable.
use std::mem;

use crate::{
    db::{RowId, Schema},
    sql::statement::{DataType, Value},
};

/// Almost all tuples (except BTree index tuples) have a [`RowId`] as the first
/// element.
///
/// This function returns the [`RowId`].
pub(crate) fn deserialize_row_id(buf: &[u8]) -> RowId {
    RowId::from_be_bytes(buf[..mem::size_of::<RowId>()].try_into().unwrap())
}

/// Serializes the `row_id` into a big endian buffer.
pub(crate) fn serialize_row_id(row_id: RowId) -> [u8; mem::size_of::<RowId>()] {
    row_id.to_be_bytes()
}

/// Returns the byte length of the given data type. Only works with integers.
pub(crate) fn byte_length_of_integer_type(data_type: &DataType) -> usize {
    match data_type {
        DataType::Int | DataType::UnsignedInt => 4,
        DataType::BigInt | DataType::UnsignedBigInt => 8,
        _ => unreachable!("byte_length_of_integer_type() called with incorrect {data_type:?}"),
    }
}

/// Calculates the size that the given tuple would take on disk once serialized.
pub(crate) fn size_of(tuple: &[Value], schema: &Schema) -> usize {
    schema
        .columns
        .iter()
        .enumerate()
        .map(|(i, col)| match col.data_type {
            DataType::Bool => 1,

            DataType::Varchar(max) => {
                let Value::String(string) = &tuple[i] else {
                    panic!(
                        "expected data type {}, found value {}",
                        DataType::Varchar(max),
                        tuple[i]
                    );
                };

                mem::size_of::<u32>() + string.as_bytes().len()
            }

            integer_type => byte_length_of_integer_type(&integer_type),
        })
        .sum()
}

/// See the module level documentation for the serialization format.
pub(crate) fn serialize(schema: &Schema, values: &[Value]) -> Vec<u8> {
    debug_assert_eq!(
        schema.len(),
        values.len(),
        "length of schema and values must be the same"
    );

    let mut buf = Vec::new();

    // TODO: Alignment.
    for (col, val) in schema.columns.iter().zip(values) {
        match (&col.data_type, val) {
            (DataType::Varchar(_), Value::String(string)) => {
                if string.as_bytes().len() > u32::MAX as usize {
                    todo!("strings longer than {} bytes are not handled", u32::MAX);
                }

                let byte_length = string.as_bytes().len() as u32;

                buf.extend_from_slice(&byte_length.to_le_bytes());
                buf.extend_from_slice(string.as_bytes());
            }

            (DataType::Bool, Value::Bool(bool)) => buf.push(u8::from(*bool)),

            (integer_type, Value::Number(num)) => {
                let bounds = match integer_type {
                    DataType::Int => i32::MIN as i128..=i32::MAX as i128,
                    DataType::UnsignedInt => 0..=u32::MAX as i128,
                    DataType::BigInt => i64::MIN as i128..=i64::MAX as i128,
                    DataType::UnsignedBigInt => 0..=u64::MAX as i128,
                    _ => unreachable!(),
                };

                assert!(
                    bounds.contains(num),
                    "integer overflow while serializing number {num} into data type {integer_type:?}"
                );

                let byte_length = byte_length_of_integer_type(integer_type);
                let big_endian_bytes = num.to_be_bytes();
                buf.extend_from_slice(&big_endian_bytes[big_endian_bytes.len() - byte_length..]);
            }

            _ => unreachable!("attempt to serialize {val} into {}", col.data_type),
        }
    }

    buf
}

/// See the module level documentation for the serialization format.
pub fn deserialize(buf: &[u8], schema: &Schema) -> Vec<Value> {
    let mut values = Vec::new();
    let mut cursor = 0;

    // TODO: Alignment.
    for column in &schema.columns {
        match column.data_type {
            DataType::Varchar(_) => {
                let length = u32::from_le_bytes(
                    buf[cursor..cursor + mem::size_of::<u32>()]
                        .try_into()
                        .unwrap(),
                ) as usize;

                cursor += mem::size_of::<u32>();

                // TODO: We need to validate somewhere that this is actually
                // valid UTF-8 (not here with unwrap(), before inserting into the DB).
                values.push(Value::String(
                    std::str::from_utf8(&buf[cursor..cursor + length])
                        .unwrap()
                        .into(),
                ));
                cursor += length;
            }

            DataType::Bool => {
                values.push(Value::Bool(buf[cursor] != 0));
                cursor += 1;
            }

            integer_type => {
                let byte_length = byte_length_of_integer_type(&integer_type);
                let mut big_endian_buf = [0; mem::size_of::<i128>()];

                big_endian_buf[mem::size_of::<i128>() - byte_length..]
                    .copy_from_slice(&buf[cursor..cursor + byte_length]);

                values.push(Value::Number(i128::from_be_bytes(big_endian_buf)));
                cursor += byte_length;
            }
        }
    }

    values
}
