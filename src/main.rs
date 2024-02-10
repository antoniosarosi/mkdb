#![feature(non_null_convenience)]
#![feature(debug_closure_helpers)]
#![feature(allocator_api)]

mod database;
mod os;
mod paging;
mod sql;
mod storage;

use std::{
    env,
    io::{self, Read, Write},
    net::TcpListener,
};

use crate::database::Database;

fn main() -> io::Result<()> {
    let file = env::args().nth(1).expect("database file not provided");

    let listener = TcpListener::bind("127.0.0.1:8000")?;
    println!("Listening on 8000");

    let mut db = Database::init(file)?;

    for stream in listener.incoming() {
        let stream = &mut stream.unwrap();
        let conn = stream.peer_addr().unwrap().to_string();
        println!("Connection from {}", conn);

        let mut statement = String::new();

        while let Some(byte) = stream.bytes().next() {
            let byte = byte.unwrap();
            statement.push(byte.into());
            if byte == b';' {
                match db.execute(statement) {
                    Ok(result) => stream.write(result.as_ascii_table().as_bytes()),
                    Err(e) => stream.write(e.to_string().as_bytes()),
                };
                stream.write("\n".as_bytes());
                statement = String::new();
            }
        }
    }

    Ok(())
}
