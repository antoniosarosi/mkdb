use core::iter::Peekable;
use std::{
    fmt::{self, Display},
    iter,
    str::Chars,
};

use super::token::{Keyword, Token, Whitespace};

/// Token stream. Wraps a [`Peekable<Chars>`] instance and allows reading the
/// next character in the stream without consuming it.
struct Stream<'c> {
    /// Current location in the stream.
    location: Location,
    /// Character input.
    chars: Peekable<Chars<'c>>,
}

#[derive(Clone, Copy, Debug)]
struct Location {
    /// Line number.
    line: usize,
    /// Column number.
    col: usize,
}

impl<'i> Stream<'i> {
    /// Creates a new stream over `input`.
    fn new(input: &'i str) -> Self {
        Self {
            location: Location { line: 1, col: 1 },
            chars: input.chars().peekable(),
        }
    }

    /// Consumes the next value updating [`Self::location`] in the process.
    fn next(&mut self) -> Option<char> {
        self.chars.next().inspect(|chr| {
            if *chr == '\n' {
                self.location.line += 1;
                self.location.col = 1;
            } else {
                self.location.col += 1;
            }
        })
    }

    /// Returns a reference to the next character in the stream without
    /// consuming it.
    fn peek(&mut self) -> Option<&char> {
        self.chars.peek()
    }

    /// Consumes the next value only if `predicate` returns `true`.
    fn next_if(&mut self, predicate: impl FnOnce(&char) -> bool) -> Option<char> {
        if self.peek().is_some_and(predicate) {
            self.next()
        } else {
            None
        }
    }

    /// Current location in the stream.
    fn location(&self) -> Location {
        self.location
    }
}

/// Syntax error.
#[derive(Debug)]
pub(super) struct TokenizerError {
    message: String,
    location: Location,
}

impl Display for TokenizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} at line {} column {}",
            self.message, self.location.line, self.location.col
        )
    }
}

impl TokenizerError {
    fn new(message: String, location: Location) -> Self {
        Self {
            message: String::from(message),
            location,
        }
    }
}

/// Main parsing structure. See [`Tokenizer::tokenize`].
pub(super) struct Tokenizer<'i> {
    /// Token stream.
    stream: Stream<'i>,
}

impl<'i> Tokenizer<'i> {
    pub fn new(input: &'i str) -> Self {
        Self {
            stream: Stream::new(input),
        }
    }

    pub fn iter_mut<'t>(&'t mut self) -> IterMut<'t, 'i> {
        self.into_iter()
    }

    /// Reads the characters in [`Self::stream`] one by one parsing the results
    /// into [`Token`] variants. If an error is encountered in the process, this
    /// function returns immediately.
    pub fn tokenize(&mut self) -> Result<Vec<Token>, TokenizerError> {
        self.iter_mut().collect()
    }

    /// Discards [`Token::Eof`] as [`Option::None`]. Useful for iterators.
    fn optional_next_token(&mut self) -> Option<Result<Token, TokenizerError>> {
        match self.next_token() {
            Ok(Token::Eof) => None,
            result => Some(result),
        }
    }

    /// Consumes and returns the next [`Token`] variant in [`Self::stream`].
    fn next_token(&mut self) -> Result<Token, TokenizerError> {
        // Done, no more chars.
        let Some(chr) = self.stream.next() else {
            return Ok(Token::Eof);
        };

        match chr {
            ' ' => Ok(Token::Whitespace(Whitespace::Space)),

            '\t' => Ok(Token::Whitespace(Whitespace::Tab)),

            '\n' => Ok(Token::Whitespace(Whitespace::Newline)),

            '\r' => match self.stream.peek() {
                Some('\n') => self.consume(Token::Whitespace(Whitespace::Newline)),
                _ => Ok(Token::Whitespace(Whitespace::Newline)),
            },

            '<' => match self.stream.peek() {
                Some('=') => self.consume(Token::LtEq),
                _ => Ok(Token::Lt),
            },

            '>' => match self.stream.peek() {
                Some('=') => self.consume(Token::GtEq),
                _ => Ok(Token::Gt),
            },

            '*' => Ok(Token::Mul),

            '+' => Ok(Token::Plus),

            '-' => Ok(Token::Minus),

            '=' => Ok(Token::Eq),

            '!' => match self.stream.peek().copied() {
                Some('=') => self.consume(Token::Neq),
                Some(unexpected) => Err(self.error(format!("unexpected token '{unexpected}'"))),
                None => Err(self.error(format!("'{}' operator not closed", Token::Neq))),
            },

            '(' => Ok(Token::LeftParen),

            ')' => Ok(Token::RightParen),

            ',' => Ok(Token::Comma),

            ';' => Ok(Token::SemiColon),

            '"' => self.tokenize_string(),

            '0'..='9' => self.tokenize_number(chr),

            _ => self.tokenize_keyword_or_identifier(chr),
        }
    }

    /// Consumes one character in the stream and returns a [`Result`] containing
    /// the given [`Token`] variant. This is used for parsing operators like
    /// `<=` where we have to peek the second character and consume it
    /// afterwards if it matches what we expect.
    fn consume(&mut self, token: Token) -> Result<Token, TokenizerError> {
        self.stream.next();
        Ok(token)
    }

    /// Parses a double quoted string like `"this one"` into [`Token::String`].
    fn tokenize_string(&mut self) -> Result<Token, TokenizerError> {
        let string = iter::from_fn(|| self.stream.next_if(|chr| *chr != '"')).collect();

        match self.stream.next() {
            Some('"') => Ok(Token::String(string)),
            _ => Err(self.error("double quoted string not closed".into())),
        }
    }

    /// Tokenizes numbers like `1234`. Floats and negatives not supported.
    fn tokenize_number(&mut self, first_digit: char) -> Result<Token, TokenizerError> {
        let mut number = String::from(first_digit);
        number.extend(iter::from_fn(|| self.stream.next_if(char::is_ascii_digit)));

        Ok(Token::Number(number))
    }

    /// Parses the next [`Word`] in the stream.
    fn tokenize_keyword_or_identifier(
        &mut self,
        first_char: char,
    ) -> Result<Token, TokenizerError> {
        if !Token::is_part_of_ident_or_keyword(&first_char) {
            return Err(self.error("unexpected keyword or identifier part".into()));
        }

        let mut value = String::from(first_char);

        value.extend(iter::from_fn(|| {
            self.stream.next_if(Token::is_part_of_ident_or_keyword)
        }));

        let keyword = match value.to_uppercase().as_str() {
            "SELECT" => Keyword::Select,
            "CREATE" => Keyword::Create,
            "UPDATE" => Keyword::Update,
            "DELETE" => Keyword::Delete,
            "INSERT" => Keyword::Insert,
            "DROP" => Keyword::Drop,
            "FROM" => Keyword::From,
            "WHERE" => Keyword::Where,
            "PRIMARY" => Keyword::Primary,
            "KEY" => Keyword::Key,
            "UNIQUE" => Keyword::Unique,
            "TABLE" => Keyword::Table,
            "DATABASE" => Keyword::Database,
            "INT" => Keyword::Int,
            "VARCHAR" => Keyword::Varchar,
            _ => Keyword::None,
        };

        Ok(match keyword {
            Keyword::None => Token::Identifier(value),
            _ => Token::Keyword(keyword),
        })
    }

    /// Builds a [`TokenizerError`] giving it the current location.
    fn error(&self, message: String) -> TokenizerError {
        TokenizerError::new(message, self.stream.location())
    }
}

/// Struct returned by [`Tokenizer::iter_mut`].
pub(super) struct IterMut<'t, 'i> {
    tokenizer: &'t mut Tokenizer<'i>,
}

/// Used to implement [`IntoIterator`] for [`Tokenizer`].
pub(super) struct IntoIter<'i> {
    tokenizer: Tokenizer<'i>,
}

impl<'t, 'i> Iterator for IterMut<'t, 'i> {
    type Item = Result<Token, TokenizerError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.tokenizer.optional_next_token()
    }
}

impl<'i> Iterator for IntoIter<'i> {
    type Item = Result<Token, TokenizerError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.tokenizer.optional_next_token()
    }
}

impl<'t, 'i> IntoIterator for &'t mut Tokenizer<'i> {
    type IntoIter = IterMut<'t, 'i>;
    type Item = Result<Token, TokenizerError>;

    fn into_iter(self) -> Self::IntoIter {
        IterMut { tokenizer: self }
    }
}

impl<'i> IntoIterator for Tokenizer<'i> {
    type IntoIter = IntoIter<'i>;
    type Item = Result<Token, TokenizerError>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter { tokenizer: self }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenize_simple_select() {
        let sql = "SELECT id, name FROM users;";

        assert_eq!(
            Tokenizer::new(sql).tokenize().unwrap(),
            vec![
                Token::Keyword(Keyword::Select),
                Token::Whitespace(Whitespace::Space),
                Token::Identifier("id".into()),
                Token::Comma,
                Token::Whitespace(Whitespace::Space),
                Token::Identifier("name".into()),
                Token::Whitespace(Whitespace::Space),
                Token::Keyword(Keyword::From),
                Token::Whitespace(Whitespace::Space),
                Token::Identifier("users".into()),
                Token::SemiColon,
            ]
        );
    }

    #[test]
    fn tokenize_select_where() {
        let sql = "SELECT id, price, discount FROM products WHERE price >= 100;";

        assert_eq!(
            Tokenizer::new(sql).tokenize().unwrap(),
            vec![
                Token::Keyword(Keyword::Select),
                Token::Whitespace(Whitespace::Space),
                Token::Identifier("id".into()),
                Token::Comma,
                Token::Whitespace(Whitespace::Space),
                Token::Identifier("price".into()),
                Token::Comma,
                Token::Whitespace(Whitespace::Space),
                Token::Identifier("discount".into()),
                Token::Whitespace(Whitespace::Space),
                Token::Keyword(Keyword::From),
                Token::Whitespace(Whitespace::Space),
                Token::Identifier("products".into()),
                Token::Whitespace(Whitespace::Space),
                Token::Keyword(Keyword::Where),
                Token::Whitespace(Whitespace::Space),
                Token::Identifier("price".into()),
                Token::Whitespace(Whitespace::Space),
                Token::GtEq,
                Token::Whitespace(Whitespace::Space),
                Token::Number("100".into()),
                Token::SemiColon,
            ]
        );
    }

    #[test]
    fn tokenize_create_table() {
        let sql = "CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(255));";

        assert_eq!(
            Tokenizer::new(sql).tokenize().unwrap(),
            vec![
                Token::Keyword(Keyword::Create),
                Token::Whitespace(Whitespace::Space),
                Token::Keyword(Keyword::Table),
                Token::Whitespace(Whitespace::Space),
                Token::Identifier("users".into()),
                Token::Whitespace(Whitespace::Space),
                Token::LeftParen,
                Token::Identifier("id".into()),
                Token::Whitespace(Whitespace::Space),
                Token::Keyword(Keyword::Int),
                Token::Whitespace(Whitespace::Space),
                Token::Keyword(Keyword::Primary),
                Token::Whitespace(Whitespace::Space),
                Token::Keyword(Keyword::Key),
                Token::Comma,
                Token::Whitespace(Whitespace::Space),
                Token::Identifier("name".into()),
                Token::Whitespace(Whitespace::Space),
                Token::Keyword(Keyword::Varchar),
                Token::LeftParen,
                Token::Number("255".into()),
                Token::RightParen,
                Token::RightParen,
                Token::SemiColon,
            ]
        );
    }

    #[test]
    fn tokenizer_error() {
        let sql = "SELECT * FROM table WHERE column ! other";
        assert!(Tokenizer::new(sql).tokenize().is_err());
    }
}
