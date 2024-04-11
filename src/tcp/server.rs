//! TCP server.

use std::{
    fs::File,
    io::{Read, Write},
    mem,
    net::{SocketAddr, TcpListener, TcpStream},
    path::Path,
    sync::{Mutex, MutexGuard},
    thread,
};

use crate::{
    db::{Database, DbError},
    pool::ThreadPool,
    tcp::proto::{self, Response},
};

/// Initializes the database on the given `file` and listens on `addr`.
pub fn start(addr: SocketAddr, file: impl AsRef<Path>) -> Result<(), DbError> {
    // We're leaking this because the database will never get dropped, so we
    // won't bother with Arc<T> and stuff. The program will either crash or
    // run forever.
    let db = &*Box::leak(Box::new(Mutex::new(Database::init(&file)?)));
    println!("Database file initialized: {}", file.as_ref().display());

    let pool = ThreadPool::new(8);

    let listener = TcpListener::bind(addr)?;
    println!("Listening on {addr}");

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
    db_mutex: &'static Mutex<Database<File>>,
) -> Result<(), DbError> {
    let conn = stream.peer_addr().unwrap().to_string();
    println!("Connection from {}", conn);

    let mut payload_len_buf = [0; mem::size_of::<u32>()];

    // Db mutext guard. We'll set it to Some once we acquire it and then set it
    // back to None when the transaction ends.
    let mut guard: Option<MutexGuard<'_, Database<File>>> = None;

    // TODO: Gracefull shutdown. We have to use the ctrlc crate and drop the
    // thread pool instance.
    loop {
        let mut payload_buf = Vec::new();

        let result = stream.read_exact(&mut payload_len_buf).and_then(|_| {
            let payload_len = u32::from_le_bytes(payload_len_buf);
            payload_buf.resize(payload_len as usize, 0);
            stream.read_exact(&mut payload_buf)
        });

        if result.is_err() {
            break;
        }

        let statement = match String::from_utf8(payload_buf) {
            Ok(string) => string,

            Err(e) => {
                let packet = proto::serialize(&Response::Err(format!("UTF-8 decode error: {e}")));
                stream.write_all(&packet.unwrap())?;
                continue;
            }
        };

        // We don't have a guard, try to acquire one.
        if guard.is_none() {
            guard = match db_mutex.try_lock() {
                Ok(guard) => Some(guard),

                Err(_) => {
                    println!("Connection {} locked on mutex", conn);
                    Some(db_mutex.lock().unwrap())
                }
            };
        }

        let db = guard.as_mut().unwrap();
        let result = db.exec(&statement);

        match proto::serialize(&Response::from(result)) {
            Ok(packet) => stream.write_all(&packet)?,

            Err(e) => {
                let packet =
                    proto::serialize(&Response::Err(format!("could not encode response {e}")));
                stream.write_all(&packet.unwrap())?;
                if db.active_transaction() {
                    db.rollback()?;
                }
            }
        };

        // We only drop the Mutex guard when the transaction ends. Otherwise we
        // keep the database locked so no other threads can access.
        if !db.active_transaction() {
            drop(guard.take());
        }
    }

    println!("Close {conn} connection");

    // If the connection is closed while a transaction is still in
    // progress we must rollback because the client didn't commit.
    if let Some(mut db) = guard {
        if db.active_transaction() {
            db.rollback()?;
        }
    }

    Ok(())
}
