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
use std::{
    io::{self, Read},
    mem,
};

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

/// Serialize a single value.
///
/// It's called serialize key because otherwise we just use [`serialize`].
/// This is only used to serialize the first part of a tuple in order to search
/// BTrees.
pub(crate) fn serialize_key(data_type: &DataType, value: &Value) -> Vec<u8> {
    let mut buf = Vec::new();
    serialize_value_into(&mut buf, data_type, value);
    buf
}

/// Serialize a complete tuple.
///
/// See the module level documentation for the serialization format.
pub(crate) fn serialize<'v>(
    schema: &Schema,
    values: (impl IntoIterator<Item = &'v Value> + Copy),
) -> Vec<u8> {
    let mut buf = Vec::new();

    debug_assert_eq!(
        schema.len(),
        values.into_iter().count(),
        "length of schema and values must the same",
    );

    // TODO: Alignment.
    for (col, val) in schema.columns.iter().zip(values.into_iter()) {
        serialize_value_into(&mut buf, &col.data_type, val);
    }

    buf
}

/// Low level serialization.
///
/// This one takes a reference instead of producing a new [`Vec<u8>`] because
/// that would be expensive for serializing complete tuples as we'd have to
/// allocate multiple vectors and join them together.
fn serialize_value_into(buf: &mut Vec<u8>, data_type: &DataType, value: &Value) {
    match (data_type, value) {
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

        _ => unreachable!("attempt to serialize {value} into {data_type}"),
    }
}

/// See the module level documentation for the serialization format.
pub fn deserialize(buf: &[u8], schema: &Schema) -> Vec<Value> {
    read_from(&mut io::Cursor::new(buf), schema).unwrap()
}

/// Reads one single tuple from the given reader.
///
/// This will call [`Read::read_exact`] many times so make sure the reader is
/// buffered or is an in-memory array such as [`io::Cursor<Vec<u8>>`].
pub fn read_from(reader: &mut impl Read, schema: &Schema) -> io::Result<Vec<Value>> {
    let values = schema.columns.iter().map(|column| {
        Ok(match column.data_type {
            DataType::Varchar(_) => {
                let mut length_buffer = [0; mem::size_of::<u32>()];
                reader.read_exact(&mut length_buffer)?;
                let length = u32::from_le_bytes(length_buffer) as usize;

                let mut string = vec![0; length];
                reader.read_exact(&mut string)?;

                // TODO: We can probably call from_utf8_unchecked() here.
                Value::String(String::from_utf8(string).unwrap())
            }

            DataType::Bool => {
                let mut byte = [0];
                reader.read_exact(&mut byte)?;
                Value::Bool(byte[0] != 0)
            }

            integer_type => {
                let byte_length = byte_length_of_integer_type(&integer_type);
                let mut big_endian_buf = [0; mem::size_of::<i128>()];
                reader.read_exact(&mut big_endian_buf[mem::size_of::<i128>() - byte_length..])?;

                Value::Number(i128::from_be_bytes(big_endian_buf))
            }
        })
    });

    values.collect()
}
