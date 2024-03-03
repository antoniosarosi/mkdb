use std::mem;

use crate::{
    db::{RowId, Schema},
    sql::{DataType, Value},
};

pub(crate) fn deserialize_row_id(buf: &[u8]) -> RowId {
    RowId::from_be_bytes(buf[..mem::size_of::<RowId>()].try_into().unwrap())
}

pub(crate) fn serialize_row_id(row_id: RowId) -> [u8; mem::size_of::<RowId>()] {
    row_id.to_be_bytes()
}

pub(crate) fn serialize_values(schema: &Schema, values: &Vec<Value>) -> Vec<u8> {
    macro_rules! serialize_big_endian {
        ($num:expr, $int:ty) => {
            TryInto::<$int>::try_into(*$num).unwrap().to_be_bytes()
        };
    }

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

pub(crate) fn deserialize_values(buf: &[u8], schema: &Schema) -> Vec<Value> {
    let mut values = Vec::new();
    let mut index = 0;

    macro_rules! deserialize_big_endian {
        ($buf:expr, $index:expr, $int:ty) => {
            <$int>::from_be_bytes(
                $buf[index..index + mem::size_of::<$int>()]
                    .try_into()
                    .unwrap(),
            )
            .into()
        };
    }

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

                // TODO: Check if we should use from_ut8_lossy() or from_utf8()
                values.push(Value::String(
                    String::from_utf8_lossy(&buf[index..(index + length)]).to_string(),
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
