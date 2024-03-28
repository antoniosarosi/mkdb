use std::{
    env,
    io::{Read, Write},
    net::TcpStream,
};

use mkdb::tcp::proto::Response;
use rustyline::{error::ReadlineError, DefaultEditor};

const EXIT_CMD: &str = "quit";

fn main() -> rustyline::Result<()> {
    let port = env::args()
        .nth(1)
        .expect("port not provided")
        .parse::<u16>()
        .expect("port parse error");

    let mut stream = TcpStream::connect(("127.0.0.1", port))?;
    println!("Welcome to the MKDB shell. Type SQL statements below or '{EXIT_CMD}' to exit the program.\n");

    // `()` can be used when no completer is required
    let mut rl = DefaultEditor::new()?;
    #[cfg(feature = "with-file-history")]
    if rl.load_history("history.txt").is_err() {
        println!("No previous history.");
    }

    let mut sql = String::new();
    let mut payload = Vec::new();

    loop {
        let readline = rl.readline("mkdb> ");
        match readline {
            Ok(line) => {
                sql.push_str(&line.trim_end());

                if !line.contains(";") {
                    continue;
                }

                rl.add_history_entry(&sql)?;

                if !sql.ends_with(";") {
                    sql.clear();
                    println!("Only one SQL statement at a time terminated with ';' can be sent");
                    continue;
                }

                stream.write_all(sql.as_bytes())?;
                sql.clear();

                let mut payload_len_buf = [0; 4];
                stream.read_exact(&mut payload_len_buf)?;
                let payload_len = u32::from_le_bytes(payload_len_buf) as usize;

                payload.resize(payload_len, 0);
                stream.read_exact(&mut payload)?;

                match mkdb::tcp::proto::deserialize(&payload) {
                    Ok(response) => match response {
                        Response::EmptySet(affected_rows) => {
                            println!("Query OK, {affected_rows} rows affected")
                        }
                        Response::Err(e) => println!("{e}"),
                        Response::Projection(projection) => println!("{}", ascii_table(projection)),
                    },

                    Err(e) => println!("decode error: {e}"),
                };
            }
            Err(ReadlineError::Interrupted) => {
                println!("CTRL-C");
                break;
            }
            Err(ReadlineError::Eof) => {
                println!("CTRL-D");
                break;
            }
            Err(err) => {
                println!("Error: {:?}", err);
                break;
            }
        }
    }

    #[cfg(feature = "with-file-history")]
    rl.save_history("history.txt");
    Ok(())
}

fn ascii_table(projection: mkdb::Projection) -> String {
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
