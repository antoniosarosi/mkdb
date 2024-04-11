use std::{
    env,
    io::{Read, Write},
    net::TcpStream,
    time::Instant,
};

use mkdb::{tcp::proto::Response, Value};
use rustyline::{error::ReadlineError, DefaultEditor};

const EXIT_CMD: &str = "quit";
const PROMPT: &str = "mkdb> ";
const CONTINUATION_PROMPT: &str = "sql> ";
const SINGLE_QUOTE_STR_PROMPT: &str = "string(')> ";
const DOUBLE_QUOTE_STR_PROMPT: &str = "string(\")> ";

fn main() -> rustyline::Result<()> {
    let port = env::args()
        .nth(1)
        .expect("port not provided")
        .parse::<u16>()
        .expect("port parse error");

    let mut rl = DefaultEditor::new()?;
    if rl.load_history("history.mkdb").is_err() {
        println!("No previous history.");
    }

    let mut stream = TcpStream::connect(("127.0.0.1", port))?;
    println!("Connected to {}.", stream.peer_addr()?);
    println!("Welcome to the MKDB shell. Type SQL statements below or '{EXIT_CMD}' to exit the program.\n");

    let mut string_quote = None;
    let mut sql = String::new();
    let mut payload = Vec::new();
    let mut prompt = PROMPT;

    loop {
        let line = match rl.readline(prompt) {
            Ok(line) => line,

            Err(e) => {
                match e {
                    ReadlineError::Interrupted => println!("CTRL-C"),
                    ReadlineError::Eof => println!("CTRL-D"),
                    other => println!("Error: {other:?}"),
                }

                break;
            }
        };

        let mut terminators_found = 0;

        for chr in line.chars() {
            match chr {
                '"' | '\'' => match string_quote {
                    None => string_quote = Some(chr),
                    Some(opening_quote) => {
                        if opening_quote == chr {
                            string_quote.take();
                        }
                    }
                },

                ';' => {
                    if string_quote.is_none() {
                        terminators_found += 1;
                    }
                }

                _ => {}
            }
        }

        // quit
        if sql.is_empty() && line.trim() == EXIT_CMD {
            break;
        }

        // Empty line, nothing to do.
        if line.trim().is_empty() {
            continue;
        }

        // Keep line breaks to avoid syntax errors.
        if !sql.is_empty() {
            sql.push('\n');
        }
        sql.push_str(&line);

        // Statement is not complete.
        if terminators_found == 0 {
            prompt = if let Some(quote) = string_quote {
                if quote == '"' {
                    DOUBLE_QUOTE_STR_PROMPT
                } else {
                    SINGLE_QUOTE_STR_PROMPT
                }
            } else {
                CONTINUATION_PROMPT
            };
            continue;
        }

        // Now we have a full statement, add it to the history and reset prompt.
        prompt = PROMPT;
        rl.add_history_entry(&sql)?;

        if terminators_found > 1 || terminators_found == 1 && !line.trim_end().ends_with(';') {
            sql.clear();
            string_quote.take();
            println!("Only one SQL statement at a time terminated with ';' can be sent. Multiple statements are not supported.");
            continue;
        }

        let packet_transmission = Instant::now();

        // Send the statement to the server.
        stream.write_all(&(sql.len() as u32).to_le_bytes())?;
        stream.write_all(sql.as_bytes())?;
        sql.clear();

        // Read header.
        let mut payload_len_buf = [0; 4];
        stream.read_exact(&mut payload_len_buf)?;
        let payload_len = u32::from_le_bytes(payload_len_buf) as usize;

        // Read payload.
        payload.resize(payload_len, 0);
        stream.read_exact(&mut payload)?;

        match mkdb::tcp::proto::deserialize(&payload) {
            Ok(response) => match response {
                Response::Err(e) => println!("{e}"),

                Response::EmptySet(affected_rows) => {
                    println!(
                        "Query OK, {affected_rows} {} affected ({:.2?})",
                        plural("row", affected_rows),
                        packet_transmission.elapsed(),
                    )
                }

                Response::QuerySet(collection) => {
                    println!(
                        "{}\n{} {} ({:.2?})",
                        ascii_table(&collection),
                        collection.tuples.len(),
                        plural("row", collection.tuples.len()),
                        packet_transmission.elapsed(),
                    );
                }
            },

            Err(e) => println!("decode error: {e}"),
        };
    }

    rl.save_history("history.mkdb")?;
    Ok(())
}

fn plural(word: &str, length: usize) -> String {
    if length == 1 {
        String::from(word)
    } else {
        format!("{word}s")
    }
}

fn ascii_table(query: &mkdb::QuerySet) -> String {
    // Initialize width of each column to the length of the table headers.
    let mut widths: Vec<usize> = query
        .schema
        .columns
        .iter()
        .map(|col| col.name.chars().count())
        .collect();

    // We only need strings.
    let rows: Vec<Vec<String>> = query
        .tuples
        .iter()
        .map(|row| {
            row.iter()
                .map(|col| match col {
                    Value::String(string) => string.replace('\n', "\\n"),
                    other => other.to_string(),
                })
                .collect()
        })
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
        &query
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
