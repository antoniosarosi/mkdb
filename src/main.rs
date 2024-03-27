#![feature(non_null_convenience)]
#![feature(debug_closure_helpers)]
#![feature(allocator_api)]
#![feature(map_try_insert)]
#![feature(vec_into_raw_parts)]
#![feature(set_ptr_value)]
#![feature(slice_ptr_get)]
#![feature(get_many_mut)]
#![feature(map_many_mut)]
#![feature(trait_alias)]
#![feature(pointer_is_aligned)]
#![feature(buf_read_has_data_left)]

mod db;
mod os;
mod paging;
mod pool;
mod query;
mod sql;
mod storage;
mod vm;

use std::{
    env,
    fs::File,
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    sync::Mutex,
    thread,
};

use db::{DbError, Projection};

use crate::{db::Database, pool::ThreadPool};

fn main() -> Result<(), DbError> {
    let file = env::args().nth(1).expect("database file not provided");

    let listener = TcpListener::bind("127.0.0.1:8000")?;
    println!("Listening on 8000");

    let db = &*Box::leak(Box::new(Mutex::new(Database::init(file)?)));
    let pool = ThreadPool::new(8);

    for stream in listener.incoming() {
        pool.execute(|| {
            if let Err(e) = handle_client(&mut stream.unwrap(), db) {
                eprintln!(
                    "error on thread {:?} while processing connection: {e:?}",
                    thread::current().id()
                )
            }
        });
    }

    Ok(())
}

fn handle_client(
    stream: &mut TcpStream,
    db: &'static Mutex<Database<File>>,
) -> Result<(), DbError> {
    let conn = stream.peer_addr().unwrap().to_string();
    println!("Connection from {}", conn);

    let prompt = "mkdb> ";

    stream.write_all(
        format!("Welcome to the MKDB 'shell' (not really a shell). Type SQL statements below or 'quit' to exit the program.\n\n{prompt}")
            .as_bytes(),
    )?;

    let mut statement = String::new();

    let exit_command = "quit";

    // Db mutext guard. We'll set it to Some once we acquire it and then set it
    // back to None when the transaction ends.
    let mut guard = None;

    while let Some(byte) = stream.bytes().next() {
        let Ok(byte) = byte else {
            println!("Close {conn} connection");
            break;
        };

        statement.push(byte.into());

        if statement.len() >= exit_command.len()
            && &statement[statement.len() - exit_command.len()..] == exit_command
        {
            println!("Close {conn} connection");
            stream.write_all("Closing connection\n".as_bytes())?;
            break;
        }

        // Keep reading bytes until we find the SQL statement terminator and we
        // can actually do something.
        if byte != b';' {
            continue;
        }

        // We don't have a guard, try to acquire one.
        if guard.is_none() {
            guard = match db.try_lock() {
                Ok(guard) => Some(guard),

                Err(_) => {
                    stream.write_all(
                        "database is locked by another connection, blocking until we can acquire the lock...\n"
                            .as_bytes(),
                    )?;
                    Some(db.lock().unwrap())
                }
            };
        }

        let lock = guard.as_mut().unwrap();

        match lock.exec(&statement) {
            Ok(projection) => {
                if projection.schema.columns.is_empty() {
                    let rows_affected = projection.results.len();
                    stream.write_all(format!("OK, {rows_affected} rows affected").as_bytes())?;
                } else {
                    stream.write_all(ascii_table(projection).as_bytes())?;
                }
            }

            Err(e) => stream.write_all(e.to_string().as_bytes())?,
        };

        // We only drop the Mutex guard when the transaction ends. Otherwise we
        // keep the database locked so no other threads can access.
        if !lock.transaction_started {
            drop(guard.take());
        }

        stream.write_all(format!("\n{prompt}").as_bytes()).unwrap();
        statement = String::new();
    }

    Ok(())
}

fn ascii_table(projection: Projection) -> String {
    // Initialize width of each column to the length of the table headers.
    let mut widths: Vec<usize> = projection
        .schema
        .columns
        .iter()
        .map(|col| col.name.len())
        .collect();

    // We only need strings.
    let rows: Vec<Vec<String>> = projection
        .results
        .iter()
        .map(|row| row.iter().map(ToString::to_string).collect())
        .collect();

    // Find the maximum width for each column.
    for row in &rows {
        for (i, col) in row.iter().enumerate() {
            if col.len() > widths[i] {
                widths[i] = col.len();
            }
        }
    }

    // We'll add both a leading and trailing space to the widest string in
    // each column, so increase width by 2.
    widths.iter_mut().for_each(|w| *w += 2);

    // Create border according to width: +-----+---------+------+-----+
    let mut border = String::from('+');
    for width in &widths {
        for _ in 0..*width {
            border.push('-');
        }
        border.push('+');
    }

    // Builds one row: | for | example | this | one |
    let make_row = |row: &Vec<String>| -> String {
        let mut string = String::from('|');

        for (i, col) in row.iter().enumerate() {
            string.push(' ');
            string.push_str(col);
            for _ in 0..widths[i] - col.len() - 1 {
                string.push(' ');
            }
            string.push('|');
        }

        string
    };

    // Header
    let mut table = String::from(&border);
    table.push('\n');

    table.push_str(&make_row(
        &projection
            .schema
            .columns
            .iter()
            .map(|col| col.name.clone())
            .collect(),
    ));
    table.push('\n');

    table.push_str(&border);
    table.push('\n');

    // Content
    for row in &rows {
        table.push_str(&make_row(row));
        table.push('\n');
    }

    if !rows.is_empty() {
        table.push_str(&border);
    }

    table
}
