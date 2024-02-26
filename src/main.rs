#![feature(non_null_convenience)]
#![feature(debug_closure_helpers)]
#![feature(allocator_api)]
#![feature(map_try_insert)]

mod db;
mod os;
mod paging;
mod query;
mod sql;
mod storage;
mod vm;

use std::{
    env,
    io::{self, Read, Write},
    net::TcpListener,
};

use crate::db::Database;

fn main() -> io::Result<()> {
    let file = env::args().nth(1).expect("database file not provided");

    let listener = TcpListener::bind("127.0.0.1:8000")?;
    println!("Listening on 8000");

    let mut db = Database::init(file)?;

    for stream in listener.incoming() {
        let stream = &mut stream.unwrap();
        let conn = stream.peer_addr().unwrap().to_string();
        println!("Connection from {}", conn);

        let prompt = "mkdb> ";

        stream.write(
            format!("Welcome to the MKDB 'shell' (not really a shell). Type SQL statements below or 'quit' to exit the program.\n\n{prompt}")
                .as_bytes(),
        )?;

        let mut statement = String::new();

        let exit_command = "quit";

        while let Some(byte) = stream.bytes().next() {
            let byte = byte.unwrap();
            statement.push(byte.into());

            if statement.len() >= exit_command.len()
                && &statement[statement.len() - exit_command.len()..] == exit_command
            {
                println!("Close {conn} connection");
                stream.write("Closing connection\n".as_bytes())?;
                break;
            }

            if byte == b';' {
                let _ = match db.exec(&statement) {
                    Ok(result) => {
                        if !result.is_empty() {
                            stream.write(result.to_ascii_table().as_bytes())
                        } else {
                            stream.write("OK".as_bytes())
                        }
                    }

                    Err(e) => stream.write(e.to_string().as_bytes()),
                };
                stream.write(format!("\n{prompt}").as_bytes()).unwrap();
                statement = String::new();
            }
        }
    }

    Ok(())
}

// pub fn to_ascii_table(query: QueryResolution) -> String {
//     // Initialize width of each column to the length of the table headers.
//     let mut widths: Vec<usize> = query
//         .schema
//         .columns
//         .iter()
//         .map(|col| col.name.len())
//         .collect();

//     // We only need strings.
//     let rows: Vec<Vec<String>> = query
//         .results
//         .iter()
//         .map(|row| row.iter().map(ToString::to_string).collect())
//         .collect();

//     // Find the maximum width for each column.
//     for row in &rows {
//         for (i, col) in row.iter().enumerate() {
//             if col.len() > widths[i] {
//                 widths[i] = col.len();
//             }
//         }
//     }

//     // We'll add both a leading and trailing space to the widest string in
//     // each column, so increase width by 2.
//     widths.iter_mut().for_each(|w| *w += 2);

//     // Create border according to width: +-----+---------+------+-----+
//     let mut border = String::from('+');
//     for width in &widths {
//         for _ in 0..*width {
//             border.push('-');
//         }
//         border.push('+');
//     }

//     // Builds one row: | for | example | this | one |
//     let make_row = |row: &Vec<String>| -> String {
//         let mut string = String::from('|');

//         for (i, col) in row.iter().enumerate() {
//             string.push(' ');
//             string.push_str(&col);
//             for _ in 0..widths[i] - col.len() - 1 {
//                 string.push(' ');
//             }
//             string.push('|');
//         }

//         string
//     };

//     // Header
//     let mut table = String::from(&border);
//     table.push('\n');

//     table.push_str(&make_row(
//         &self
//             .schema
//             .columns
//             .iter()
//             .map(|col| col.name.clone())
//             .collect(),
//     ));
//     table.push('\n');

//     table.push_str(&border);
//     table.push('\n');

//     // Content
//     for row in &rows {
//         table.push_str(&make_row(row));
//         table.push('\n');
//     }

//     if !rows.is_empty() {
//         table.push_str(&border);
//     }

//     table
// }
