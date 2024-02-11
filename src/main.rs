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

        let exit_command = "quit";

        while let Some(byte) = stream.bytes().next() {
            let byte = byte.unwrap();
            statement.push(byte.into());

            if statement.len() >= exit_command.len()
                && &statement[statement.len() - exit_command.len()..] == exit_command
            {
                println!("Close {conn} connection");
                break;
            }

            if byte == b';' {
                let _ = match db.execute(statement) {
                    Ok(result) => stream.write(result.as_ascii_table().as_bytes()),
                    Err(e) => stream.write(e.to_string().as_bytes()),
                };
                stream.write("\n".as_bytes()).unwrap();
                statement = String::new();
            }
        }
    }

    Ok(())
}
