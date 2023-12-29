use std::{
    fmt::{self, Display},
    iter::Peekable,
    str::Chars,
};

use super::token::{Keyword, Token, Whitespace};

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Location {
    /// Line number.
    line: usize,
    /// Column number.
    col: usize,
}

impl Default for Location {
    fn default() -> Self {
        Self { line: 1, col: 1 }
    }
}

/// Stores both the [`Token`] and its starting location in the input string.
#[derive(Debug, PartialEq)]
pub(super) struct TokenWithLocation {
    pub token: Token,
    pub location: Location,
}

impl TokenWithLocation {
    /// Discards the location. Used mostly for mapping:
    /// `.map(TokenWithLocation::token_only)`.
    pub fn token_only(self) -> Token {
        self.token
    }

    /// Reference to [`Token`].
    pub fn token(&self) -> &Token {
        &self.token
    }
}

/// Token stream. Wraps a [`Peekable<Chars>`] instance and allows reading the
/// next character in the stream without consuming it.
struct Stream<'i> {
    /// Current location in the stream.
    location: Location,
    /// Character input.
    chars: Peekable<Chars<'i>>,
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

    /// Consumes one character in the stream and returns a reference to the next
    /// one without consuming it.
    fn peek_next(&mut self) -> Option<&char> {
        self.next();
        self.peek()
    }

    /// Safe version of [`std::iter::TakeWhile`] that does not discard elements
    /// when `predicate` returns `false`.
    fn take_while<P: FnMut(&char) -> bool>(&mut self, predicate: P) -> TakeWhile<'_, 'i, P> {
        TakeWhile {
            stream: self,
            predicate,
        }
    }

    /// Current location in the stream. [`Location`] is [`Copy`], no need for
    /// references.
    fn location(&self) -> Location {
        self.location
    }
}

/// See [`Stream::take_while`] for more details.
struct TakeWhile<'s, 'i, P> {
    stream: &'s mut Stream<'i>,
    predicate: P,
}

impl<'s, 'c, P: FnMut(&char) -> bool> Iterator for TakeWhile<'s, 'c, P> {
    type Item = char;

    fn next(&mut self) -> Option<Self::Item> {
        if (self.predicate)(self.stream.peek()?) {
            self.stream.next()
        } else {
            None
        }
    }
}

/// If the tokenizer finds an error it means to syntax is not correct. Some
/// examples are unclosed strings, unclosed operators, etc.
#[derive(Debug, PartialEq)]
pub(super) struct TokenizerError {
    pub message: String,
    pub location: Location,
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

/// Main parsing structure. See [`Tokenizer::next_token`].
pub(super) struct Tokenizer<'i> {
    /// Character stream.
    stream: Stream<'i>,
}

type TokenResult = Result<Token, TokenizerError>;

impl<'i> Tokenizer<'i> {
    /// Creates a new tokenizer for the given `input`. The tokenizer won't parse
    /// anything until [`Tokenizer::next_token`] through helper functions or
    /// iterators. See [`Tokenizer::iter_mut`] and [`Tokenizer::tokenize`].
    pub fn new(input: &'i str) -> Self {
        Self {
            stream: Stream::new(input),
        }
    }

    /// Creates an iterator over [`Self`]. Used mainly to parse tokens as they
    /// are found instead of waiting for the tokenizer to consume the entire
    /// input string.
    pub fn iter_mut<'t>(&'t mut self) -> IterMut<'t, 'i> {
        self.into_iter()
    }

    /// Reads the characters in [`Self::stream`] one by one parsing the results
    /// into [`Token`] variants. If an error is encountered in the process, this
    /// function returns immediately.
    pub fn tokenize(&mut self) -> Result<Vec<Token>, TokenizerError> {
        self.iter_mut()
            .map(|result| result.map(TokenWithLocation::token_only))
            .collect()
    }

    /// Same as [`Self::optional_next_token`] but stores the starting location
    /// of the token as well.
    fn optional_next_token_with_location(
        &mut self,
    ) -> Option<Result<TokenWithLocation, TokenizerError>> {
        let location = self.stream.location();

        self.optional_next_token()
            .map(|result| result.map(|token| TokenWithLocation { token, location }))
    }

    /// Discards [`Token::Eof`] as [`Option::None`]. Useful for iterators.
    fn optional_next_token(&mut self) -> Option<TokenResult> {
        match self.next_token() {
            Ok(Token::Eof) => None,
            result => Some(result),
        }
    }

    /// Consumes and returns the next [`Token`] variant in [`Self::stream`].
    fn next_token(&mut self) -> TokenResult {
        // Done, no more chars.
        let Some(chr) = self.stream.peek() else {
            return Ok(Token::Eof);
        };

        match chr {
            ' ' => self.consume(Token::Whitespace(Whitespace::Space)),

            '\t' => self.consume(Token::Whitespace(Whitespace::Tab)),

            '\n' => self.consume(Token::Whitespace(Whitespace::Newline)),

            '\r' => match self.stream.peek_next() {
                Some('\n') => self.consume(Token::Whitespace(Whitespace::Newline)),
                _ => Ok(Token::Whitespace(Whitespace::Newline)),
            },

            '<' => match self.stream.peek_next() {
                Some('=') => self.consume(Token::LtEq),
                _ => Ok(Token::Lt),
            },

            '>' => match self.stream.peek_next() {
                Some('=') => self.consume(Token::GtEq),
                _ => Ok(Token::Gt),
            },

            '*' => self.consume(Token::Mul),

            '/' => self.consume(Token::Div),

            '+' => self.consume(Token::Plus),

            '-' => self.consume(Token::Minus),

            '=' => self.consume(Token::Eq),

            '!' => match self.stream.peek_next() {
                Some('=') => self.consume(Token::Neq),

                Some(unexpected) => {
                    let message =
                        format!("unexpected token '{unexpected}' while parsing '!=' operator");
                    self.error(message)
                }

                None => self.error(format!("'!=' operator not closed")),
            },

            '(' => self.consume(Token::LeftParen),

            ')' => self.consume(Token::RightParen),

            ',' => self.consume(Token::Comma),

            ';' => self.consume(Token::SemiColon),

            '"' => self.tokenize_string(),

            '0'..='9' => self.tokenize_number(),

            _ if Token::is_part_of_ident_or_keyword(&chr) => self.tokenize_keyword_or_identifier(),

            _ => {
                let message = format!("unexpected or unsupported token '{chr}'");
                self.error(message)
            }
        }
    }

    /// Consumes one character in the stream and returns an [`Ok(Token)`] result
    /// containing the given [`Token`] variant.
    fn consume(&mut self, token: Token) -> TokenResult {
        self.stream.next();
        Ok(token)
    }

    /// Builds an instance of [`Err(TokenizerError)`] giving it the current
    /// location of the stream.
    fn error(&self, message: impl Into<String>) -> TokenResult {
        Err(TokenizerError::new(message.into(), self.stream.location()))
    }

    /// Parses a double quoted string like `"this one"` into [`Token::String`].
    fn tokenize_string(&mut self) -> TokenResult {
        let Some('"') = self.stream.next() else {
            return self.error("expected double quoted string opening");
        };

        let string = self.stream.take_while(|chr| *chr != '"').collect();

        match self.stream.next() {
            Some('"') => Ok(Token::String(string)),
            _ => self.error("double quoted string not closed"),
        }
    }

    /// Tokenizes numbers like `1234`. Floats and negatives not supported.
    fn tokenize_number(&mut self) -> TokenResult {
        Ok(Token::Number(
            self.stream.take_while(char::is_ascii_digit).collect(),
        ))
    }

    /// Attempts to parse an instance of [`Token::Keyword`] or
    /// [`Token::Identifier`].
    fn tokenize_keyword_or_identifier(&mut self) -> TokenResult {
        let value: String = self
            .stream
            .take_while(Token::is_part_of_ident_or_keyword)
            .collect();

        let keyword = match value.to_uppercase().as_str() {
            "SELECT" => Keyword::Select,
            "CREATE" => Keyword::Create,
            "UPDATE" => Keyword::Update,
            "DELETE" => Keyword::Delete,
            "INSERT" => Keyword::Insert,
            "VALUES" => Keyword::Values,
            "INTO" => Keyword::Into,
            "SET" => Keyword::Set,
            "DROP" => Keyword::Drop,
            "FROM" => Keyword::From,
            "WHERE" => Keyword::Where,
            "AND" => Keyword::And,
            "OR" => Keyword::Or,
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
}

/// Struct returned by [`Tokenizer::iter_mut`].
pub(super) struct IterMut<'t, 'i> {
    tokenizer: &'t mut Tokenizer<'i>,
}

impl<'t, 'i> Iterator for IterMut<'t, 'i> {
    type Item = Result<TokenWithLocation, TokenizerError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.tokenizer.optional_next_token_with_location()
    }
}

impl<'t, 'i> IntoIterator for &'t mut Tokenizer<'i> {
    type IntoIter = IterMut<'t, 'i>;
    type Item = Result<TokenWithLocation, TokenizerError>;

    fn into_iter(self) -> Self::IntoIter {
        IterMut { tokenizer: self }
    }
}

/// Used to implement [`IntoIterator`] for [`Tokenizer`].
pub(super) struct IntoIter<'i> {
    tokenizer: Tokenizer<'i>,
}

impl<'i> Iterator for IntoIter<'i> {
    type Item = Result<TokenWithLocation, TokenizerError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.tokenizer.optional_next_token_with_location()
    }
}

impl<'i> IntoIterator for Tokenizer<'i> {
    type IntoIter = IntoIter<'i>;
    type Item = Result<TokenWithLocation, TokenizerError>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter { tokenizer: self }
    }
}

#[cfg(test)]
mod tests {
    use super::{Keyword, Token, Tokenizer, Whitespace};
    use crate::sql::tokenizer::{Location, TokenizerError};

    #[test]
    fn tokenize_simple_select() {
        let sql = "SELECT id, name FROM users;";

        assert_eq!(
            Tokenizer::new(sql).tokenize(),
            Ok(vec![
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
            ])
        );
    }

    #[test]
    fn tokenize_select_where() {
        let sql = "SELECT id, price, discount FROM products WHERE price >= 100;";

        assert_eq!(
            Tokenizer::new(sql).tokenize(),
            Ok(vec![
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
            ])
        );
    }

    #[test]
    fn tokenize_select_where_with_and_or() {
        let sql = "SELECT id, name FROM users WHERE age >= 20 AND age <= 30 OR is_admin = 1;";

        assert_eq!(
            Tokenizer::new(sql).tokenize(),
            Ok(vec![
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
                Token::Whitespace(Whitespace::Space),
                Token::Keyword(Keyword::Where),
                Token::Whitespace(Whitespace::Space),
                Token::Identifier("age".into()),
                Token::Whitespace(Whitespace::Space),
                Token::GtEq,
                Token::Whitespace(Whitespace::Space),
                Token::Number("20".into()),
                Token::Whitespace(Whitespace::Space),
                Token::Keyword(Keyword::And),
                Token::Whitespace(Whitespace::Space),
                Token::Identifier("age".into()),
                Token::Whitespace(Whitespace::Space),
                Token::LtEq,
                Token::Whitespace(Whitespace::Space),
                Token::Number("30".into()),
                Token::Whitespace(Whitespace::Space),
                Token::Keyword(Keyword::Or),
                Token::Whitespace(Whitespace::Space),
                Token::Identifier("is_admin".into()),
                Token::Whitespace(Whitespace::Space),
                Token::Eq,
                Token::Whitespace(Whitespace::Space),
                Token::Number("1".into()),
                Token::SemiColon
            ])
        );
    }

    #[test]
    fn tokenize_create_table() {
        let sql = "CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(255));";

        assert_eq!(
            Tokenizer::new(sql).tokenize(),
            Ok(vec![
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
            ])
        );
    }

    #[test]
    fn tokenize_update_table() {
        let sql = r#"UPDATE products SET code = "promo", discount = 10 WHERE price < 100;"#;

        assert_eq!(
            Tokenizer::new(sql).tokenize(),
            Ok(vec![
                Token::Keyword(Keyword::Update),
                Token::Whitespace(Whitespace::Space),
                Token::Identifier("products".into()),
                Token::Whitespace(Whitespace::Space),
                Token::Keyword(Keyword::Set),
                Token::Whitespace(Whitespace::Space),
                Token::Identifier("code".into()),
                Token::Whitespace(Whitespace::Space),
                Token::Eq,
                Token::Whitespace(Whitespace::Space),
                Token::String("promo".into()),
                Token::Comma,
                Token::Whitespace(Whitespace::Space),
                Token::Identifier("discount".into()),
                Token::Whitespace(Whitespace::Space),
                Token::Eq,
                Token::Whitespace(Whitespace::Space),
                Token::Number("10".into()),
                Token::Whitespace(Whitespace::Space),
                Token::Keyword(Keyword::Where),
                Token::Whitespace(Whitespace::Space),
                Token::Identifier("price".into()),
                Token::Whitespace(Whitespace::Space),
                Token::Lt,
                Token::Whitespace(Whitespace::Space),
                Token::Number("100".into()),
                Token::SemiColon,
            ])
        );
    }

    #[test]
    fn tokenize_insert_into() {
        let sql = r#"INSERT INTO users (name, email, age) VALUES ("Test", "test@test.com", 20);"#;

        assert_eq!(
            Tokenizer::new(sql).tokenize(),
            Ok(vec![
                Token::Keyword(Keyword::Insert),
                Token::Whitespace(Whitespace::Space),
                Token::Keyword(Keyword::Into),
                Token::Whitespace(Whitespace::Space),
                Token::Identifier("users".into()),
                Token::Whitespace(Whitespace::Space),
                Token::LeftParen,
                Token::Identifier("name".into()),
                Token::Comma,
                Token::Whitespace(Whitespace::Space),
                Token::Identifier("email".into()),
                Token::Comma,
                Token::Whitespace(Whitespace::Space),
                Token::Identifier("age".into()),
                Token::RightParen,
                Token::Whitespace(Whitespace::Space),
                Token::Keyword(Keyword::Values),
                Token::Whitespace(Whitespace::Space),
                Token::LeftParen,
                Token::String("Test".into()),
                Token::Comma,
                Token::Whitespace(Whitespace::Space),
                Token::String("test@test.com".into()),
                Token::Comma,
                Token::Whitespace(Whitespace::Space),
                Token::Number("20".into()),
                Token::RightParen,
                Token::SemiColon,
            ])
        );
    }

    #[test]
    fn tokenize_incorrect_neq_operator() {
        let sql = "SELECT * FROM table WHERE column ! other";
        assert_eq!(
            Tokenizer::new(sql).tokenize(),
            Err(TokenizerError {
                message: "unexpected token ' ' while parsing '!=' operator".into(),
                location: Location { line: 1, col: 35 }
            })
        );
    }

    #[test]
    fn tokenize_unclosed_neq_operator() {
        let sql = "SELECT * FROM table WHERE column !";
        assert_eq!(
            Tokenizer::new(sql).tokenize(),
            Err(TokenizerError {
                message: "'!=' operator not closed".into(),
                location: Location { line: 1, col: 35 }
            })
        );
    }

    #[test]
    fn tokenize_double_quoted_string_not_closed() {
        let sql = "SELECT * FROM table WHERE string = \"not closed";
        assert_eq!(
            Tokenizer::new(sql).tokenize(),
            Err(TokenizerError {
                message: "double quoted string not closed".into(),
                location: Location { line: 1, col: 47 }
            })
        );
    }

    #[test]
    fn tokenize_unsupported_token() {
        let sql = "SELECT * FROM ^ WHERE unsupported = 1;";
        assert_eq!(
            Tokenizer::new(sql).tokenize(),
            Err(TokenizerError {
                message: "unexpected or unsupported token '^'".into(),
                location: Location { line: 1, col: 15 }
            })
        );
    }
}
