/// A tuple is a single database row.
///
/// TODO: Right now we're "serializing" and "deserializing" rows which is not
/// really needed. We already store the rows in such a format that we can
/// interpret the bytes in them as numbers or strings without having to copy
/// them into [`Value`] structures. Serializing and deserializing made it easy
/// to develop in the beginning because it doesn't require any unsafe code, but
/// it's probably the biggest performance hit because we do it many times.
use std::mem;

use crate::{
    db::{RowId, Schema},
    sql::statement::{DataType, Value},
};

pub(crate) fn deserialize_row_id(buf: &[u8]) -> RowId {
    RowId::from_be_bytes(buf[..mem::size_of::<RowId>()].try_into().unwrap())
}

pub(crate) fn serialize_row_id(row_id: RowId) -> [u8; mem::size_of::<RowId>()] {
    row_id.to_be_bytes()
}

fn byte_length_of_integer_type(data_type: &DataType) -> usize {
    match data_type {
        DataType::Int | DataType::UnsignedInt => 4,
        DataType::BigInt | DataType::UnsignedBigInt => 8,
        _ => unreachable!("byte_length_of_integer_type() called with incorrect {data_type:?}"),
    }
}

pub(crate) fn size_of(schema: &Schema, tuple: &[Value]) -> usize {
    schema
        .columns
        .iter()
        .enumerate()
        .map(|(i, col)| match col.data_type {
            DataType::Bool => 1,

            DataType::Varchar(max) => {
                let length_bytes = if max <= u8::MAX as usize { 1 } else { 2 };

                let Value::String(string) = &tuple[i] else {
                    panic!(
                        "expected data type {}, found value {}",
                        DataType::Varchar(max),
                        tuple[i]
                    );
                };

                length_bytes + string.as_bytes().len()
            }

            integer_type => byte_length_of_integer_type(&integer_type),
        })
        .sum()
}

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
            (DataType::Varchar(max), Value::String(string)) => {
                if string.as_bytes().len() > u16::MAX as usize {
                    todo!("strings longer than 65535 bytes are not handled");
                }

                let length = string.len().to_le_bytes();
                let length_bytes = if *max <= u8::MAX as usize { 1 } else { 2 };

                buf.extend_from_slice(&length[..length_bytes]);
                buf.extend_from_slice(string.as_bytes());
            }

            (DataType::Bool, Value::Bool(bool)) => buf.push(u8::from(*bool)),

            (integer_type, Value::Number(num)) => {
                let byte_length = byte_length_of_integer_type(integer_type);
                let big_endian_bytes = num.to_be_bytes();
                buf.extend_from_slice(&big_endian_bytes[big_endian_bytes.len() - byte_length..]);
            }

            _ => unreachable!("attempt to serialize {val} into {}", col.data_type),
        }
    }

    buf
}

pub(crate) fn deserialize(buf: &[u8], schema: &Schema) -> Vec<Value> {
    let mut values = Vec::new();
    let mut index = 0;

    // TODO: Alignment.
    for column in &schema.columns {
        match column.data_type {
            DataType::Varchar(max) => {
                // TODO: We're only supporting strings of maximum length 65535
                // in bytes (not characters).
                let mut length_buf = [0; mem::size_of::<u16>()];
                let length_bytes = if max <= u8::MAX as usize { 1 } else { 2 };

                length_buf[..length_bytes].copy_from_slice(&buf[index..index + length_bytes]);
                index += length_bytes;

                let length = u16::from_le_bytes(length_buf) as usize;

                // TODO: We need to validate somewhere that this is actually valid UTF-8
                values.push(Value::String(
                    std::str::from_utf8(&buf[index..index + length])
                        .unwrap()
                        .into(),
                ));
                index += length;
            }

            DataType::Bool => {
                values.push(Value::Bool(buf[index] != 0));
                index += 1;
            }

            integer_type => {
                let byte_length = byte_length_of_integer_type(&integer_type);
                let mut big_endian_buf = [0; mem::size_of::<i128>()];

                big_endian_buf[mem::size_of::<i128>() - byte_length..]
                    .copy_from_slice(&buf[index..index + byte_length]);

                values.push(Value::Number(i128::from_be_bytes(big_endian_buf)));
                index += byte_length;
            }
        }
    }

    values
}
