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

macro_rules! serialize_big_endian {
    ($num:expr, $int:ty) => {
        TryInto::<$int>::try_into(*$num).unwrap().to_be_bytes()
    };
}

// TODO: Fix this nonsense. Or maybe find a way to make it even more
// inefficient :)
pub(crate) fn size_of(schema: &Schema, values: &[Value]) -> usize {
    serialize_values(schema, values).len()
}

pub(crate) fn serialize_values(schema: &Schema, values: &[Value]) -> Vec<u8> {
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
                let length = string.len().to_le_bytes();

                // TODO: Strings longer than 65535 chars are not handled.
                let n_bytes = if *max <= u8::MAX as usize { 1 } else { 2 };

                buf.extend_from_slice(&length[..n_bytes]);
                buf.extend_from_slice(string.as_bytes());
            }

            (DataType::Int, Value::Number(num)) => {
                buf.extend_from_slice(&serialize_big_endian!(num, i32));
            }

            (DataType::UnsignedInt, Value::Number(num)) => {
                buf.extend_from_slice(&serialize_big_endian!(num, u32));
            }

            (DataType::BigInt, Value::Number(num)) => {
                buf.extend_from_slice(&serialize_big_endian!(num, i64));
            }

            (DataType::UnsignedBigInt, Value::Number(num)) => {
                buf.extend_from_slice(&serialize_big_endian!(num, u64));
            }

            (DataType::Bool, Value::Bool(bool)) => buf.push(u8::from(*bool)),

            _ => unreachable!("attempt to serialize {val} into {}", col.data_type),
        }
    }

    buf
}

macro_rules! deserialize_big_endian {
    ($buf:expr, $index:expr, $int:ty) => {
        <$int>::from_be_bytes(
            $buf[$index..$index + mem::size_of::<$int>()]
                .try_into()
                .unwrap(),
        )
        .into()
    };
}

pub(crate) fn deserialize_values(buf: &[u8], schema: &Schema) -> Vec<Value> {
    let mut values = Vec::new();
    let mut index = 0;

    // TODO: Alignment.
    for column in &schema.columns {
        match column.data_type {
            DataType::Varchar(max) => {
                let length = if max <= u8::MAX as usize {
                    let len = buf[index];
                    index += 1;
                    len as usize
                } else {
                    let len: usize =
                        u16::from_le_bytes(buf[index..index + 2].try_into().unwrap()).into();
                    index += 2;
                    len
                };

                // TODO: We need to validate somewhere that this is actually valid UTF-8
                values.push(Value::String(
                    std::str::from_utf8(&buf[index..(index + length)])
                        .unwrap()
                        .into(),
                ));
                index += length;
            }

            DataType::Int => {
                values.push(Value::Number(deserialize_big_endian!(buf, index, i32)));
                index += mem::size_of::<i32>();
            }

            DataType::UnsignedInt => {
                values.push(Value::Number(deserialize_big_endian!(buf, index, u32)));
                index += mem::size_of::<u32>();
            }

            DataType::BigInt => {
                values.push(Value::Number(deserialize_big_endian!(buf, index, i64)));
                index += mem::size_of::<i64>();
            }

            DataType::UnsignedBigInt => {
                values.push(Value::Number(deserialize_big_endian!(buf, index, u64)));
                index += mem::size_of::<u64>();
            }

            DataType::Bool => {
                values.push(Value::Bool(buf[index] != 0));
                index += mem::size_of::<bool>();
            }
        }
    }

    values
}
