use core::iter::Peekable;
use std::{
    fmt::{self, Display},
    iter,
    str::Chars,
};

/// SQL tokens.
#[derive(PartialEq, Debug)]
pub(crate) enum Token {
    Eof,
    Whitespace(Whitespace),
    Word(Word),
    String(String),
    Number(String),
    Eq,
    Neq,
    Lt,
    Gt,
    LtEq,
    GtEq,
    Mul,
    Plus,
    Minus,
    LeftParen,
    RightParen,
    Comma,
    SemiColon,
}

impl Display for Token {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Eof => f.write_str("Eof"),
            Self::Whitespace(whitespace) => write!(f, "{whitespace}"),
            Self::Word(word) => write!(f, "{word}"),
            Self::String(string) => write!(f, "\"{string}\""),
            Self::Number(number) => write!(f, "{number}"),
            Self::Eq => f.write_str("="),
            Self::Neq => f.write_str("!="),
            Self::Lt => f.write_str("<"),
            Self::Gt => f.write_str(">"),
            Self::LtEq => f.write_str("<="),
            Self::GtEq => f.write_str(">="),
            Self::Mul => f.write_str("*"),
            Self::Plus => f.write_str("+"),
            Self::Minus => f.write_str("-"),
            Self::LeftParen => f.write_str("("),
            Self::RightParen => f.write_str(")"),
            Self::Comma => f.write_str(","),
            Self::SemiColon => f.write_str(";"),
        }
    }
}

/// SQL keywords.
#[derive(PartialEq, Debug)]
pub(crate) enum Keyword {
    Select,
    Create,
    Update,
    Delete,
    Insert,
    Drop,
    From,
    Where,
    Primary,
    Key,
    Unique,
    Table,
    Database,
    Int,
    Varchar,
    // Not a keyword.
    None,
}

impl Display for Keyword {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Select => f.write_str("SELECT"),
            Self::Create => f.write_str("CREATE"),
            Self::Update => f.write_str("UPDATE"),
            Self::Delete => f.write_str("DELETE"),
            Self::Insert => f.write_str("INSERT"),
            Self::Drop => f.write_str("DROP"),
            Self::From => f.write_str("FROM"),
            Self::Where => f.write_str("WHERE"),
            Self::Primary => f.write_str("PRIMARY"),
            Self::Key => f.write_str("KEY"),
            Self::Unique => f.write_str("UNIQUE"),
            Self::Table => f.write_str("TABLE"),
            Self::Database => f.write_str("DATABASE"),
            Self::Int => f.write_str("INT"),
            Self::Varchar => f.write_str("VARCHAR"),
            Self::None => f.write_str("_"),
        }
    }
}

/// Either a keyword or identifier.
#[derive(PartialEq, Debug)]
pub(crate) struct Word {
    /// Keyword or identifier value without surrounding quotes.
    value: String,
    /// Contains a variant other than [`Keyword::None`] if this is an SQL keyword.
    keyword: Keyword,
}

impl Display for Word {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.value)
    }
}

impl Word {
    fn identifier(identifier: &str) -> Self {
        Self {
            value: identifier.into(),
            keyword: Keyword::None,
        }
    }

    fn keyword(keyword: Keyword) -> Self {
        Self {
            value: format!("{keyword}"),
            keyword,
        }
    }

    /// Returns `true` if the given char can be used for identifiers or
    /// keywords.
    fn is_identifier_part(chr: &char) -> bool {
        chr.is_ascii_lowercase() || chr.is_ascii_uppercase() || chr.is_ascii_digit() || *chr == '_'
    }
}

/// Separators between keywords, identifiers, operators, etc.
#[derive(PartialEq, Debug)]
pub(crate) enum Whitespace {
    Space,
    Tab,
    Newline,
}

impl Display for Whitespace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Space => f.write_str(" "),
            Self::Tab => f.write_str("\t"),
            Self::Newline => f.write_str("\n"),
        }
    }
}

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

    /// Returns the next character in the stream updating [`Self::location`] in
    /// the process.
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

    /// Consumes characters from the stream until the given `predicate` returns
    /// `false`.
    fn consume_while(&mut self, predicate: impl FnMut(&char) -> bool) -> Option<char> {
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
struct TokenizerError {
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
struct Tokenizer<'i> {
    /// Token stream.
    stream: Stream<'i>,
}

impl<'i> Tokenizer<'i> {
    pub fn new(input: &'i str) -> Self {
        Self {
            stream: Stream::new(input),
        }
    }

    /// Reads the characters in [`Self::stream`] one by one parsing the results
    /// into [`Token`] variants. If an error is encountered in the process, this
    /// function returns immediately.
    pub fn tokenize(&mut self) -> Result<Vec<Token>, TokenizerError> {
        let tokens = iter::from_fn(|| match self.next_token() {
            Ok(Token::Eof) => None,
            result => Some(result),
        });

        tokens.collect()
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
        let string = iter::from_fn(|| self.stream.consume_while(|chr| *chr != '"')).collect();

        match self.stream.next() {
            Some('"') => Ok(Token::String(string)),
            _ => Err(self.error("double quoted string not closed".into())),
        }
    }

    /// Tokenizes numbers like `1234`. Floats and negatives not supported.
    fn tokenize_number(&mut self, first_digit: char) -> Result<Token, TokenizerError> {
        let mut number = String::from(first_digit);
        number.extend(iter::from_fn(|| {
            self.stream.consume_while(|chr| chr.is_ascii_digit())
        }));

        Ok(Token::Number(number))
    }

    /// Parses the next [`Word`] in the stream.
    fn tokenize_keyword_or_identifier(
        &mut self,
        first_char: char,
    ) -> Result<Token, TokenizerError> {
        if !Word::is_identifier_part(&first_char) {
            return Err(self.error("unexpected keyword or identifier part".into()));
        }

        let mut value = String::from(first_char);
        value.extend(iter::from_fn(|| {
            self.stream
                .consume_while(|chr| Word::is_identifier_part(chr))
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

        Ok(Token::Word(Word { value, keyword }))
    }

    /// Builds a [`TokenizerError`] giving it the current location.
    fn error(&self, message: String) -> TokenizerError {
        TokenizerError::new(message, self.stream.location())
    }
}

pub enum BinaryOperator {
    Lt,
    LtEq,
    Gt,
    GtEq,
    Plus,
    Minus,
    Mul,
}

pub enum Expression {
    Identifier(String),

    BinaryOperation {
        left: Box<Self>,
        operator: BinaryOperator,
        right: Box<Self>,
    },
}

enum Constraint {
    PrimaryKey,
    Unique,
}

enum DataType {
    Int,
    Bool,
    Varchar(usize),
}

enum Value {
    Number(String),
    String(String),
    Bool(bool),
}

struct Column {
    name: String,
    data_type: DataType,
    constraints: Vec<Constraint>,
}

enum Create {
    Table { name: String, columns: Vec<Column> },
    Database { name: String },
}

enum Statement {
    Create(Create),

    Select {
        columns: Vec<Expression>,
        from: String,
        r#where: Option<Expression>,
    },

    Delete {
        from: String,
        r#where: Option<Expression>,
    },

    Update {
        table: String,
        r#where: Option<Expression>,
    },

    Insert {
        into: String,
        columns: Vec<String>,
        values: Vec<Value>,
    },
}

struct ParserError {
    message: String,
}

impl ParserError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl From<TokenizerError> for ParserError {
    fn from(err: TokenizerError) -> Self {
        Self {
            message: format!("syntax error: {err}"),
        }
    }
}

struct Parser<'i> {
    input: &'i str,
    tokens: Vec<Token>,
}

impl<'i> Parser<'i> {
    fn new(input: &'i str) -> Self {
        Self {
            input,
            tokens: Vec::new(),
        }
    }

    fn try_parse(&mut self) -> Result<Vec<Statement>, ParserError> {
        self.tokens = Tokenizer::new(self.input)
            .tokenize()?
            .into_iter()
            .rev()
            .collect();

        let mut statements = Vec::new();

        while !self.tokens.is_empty() {
            statements.push(self.parse_statement()?);
        }

        Ok(statements)
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.last()
    }

    fn skip_white_spaces(&mut self) {
        while let Some(Token::Whitespace(_)) = self.tokens.last() {
            self.tokens.pop();
        }
    }

    fn next_after_whitespaces(&mut self) -> Result<Token, ParserError> {
        let Some(Token::Whitespace(_)) = self.tokens.last() else {
            return Err(ParserError::new("expected whitespace separator"));
        };

        self.skip_white_spaces();

        self.tokens.pop().ok_or(ParserError::new("unexpected eof"))
    }

    fn expect_keyword_after_whitespaces(&mut self, expected: Keyword) -> Result<(), ParserError> {
        self.next_after_whitespaces().and_then(|token| match token {
            Token::Word(Word { keyword, value }) => {
                if keyword == expected {
                    Ok(())
                } else {
                    Err(ParserError::new(format!(
                        "unexpected identifier or keyword {value}, expected {expected} keyword",
                    )))
                }
            }
            _ => Err(ParserError::new(format!(
                "unexpected token {token}, expected keyword {expected}"
            ))),
        })
    }

    fn expect_semicolon(&mut self) -> Result<(), ParserError> {
        self.next_after_whitespaces().and_then(|token| match token {
            Token::SemiColon => Ok(()),

            _ => Err(ParserError::new(format!(
                "missing {} statement terminator",
                Token::SemiColon,
            ))),
        })
    }

    fn consume_optional_keyword(&mut self, keyword: Keyword) -> bool {
        self.skip_white_spaces();

        self.peek().is_some_and(|token| match token {
            Token::Word(word) => word.keyword == keyword,
            _ => false,
        })
    }

    fn parse_statement(&mut self) -> Result<Statement, ParserError> {
        match self.next_after_whitespaces()? {
            Token::Word(Word { value, keyword }) => match keyword {
                Keyword::None => Err(ParserError::new("expected SQL statement")),

                Keyword::Select => {
                    let columns = self.parse_comma_separated_expressions()?;
                    self.expect_keyword_after_whitespaces(Keyword::From)?;

                    let from = self.parse_identifier()?;
                    let r#where = self.parse_optional_where()?;
                    self.expect_semicolon()?;

                    Ok(Statement::Select {
                        columns,
                        from,
                        r#where,
                    })
                }

                Keyword::Create => todo!(),

                Keyword::Update => todo!(),

                Keyword::Insert => todo!(),

                Keyword::Delete => {
                    self.expect_keyword_after_whitespaces(Keyword::From)?;
                    let from = self.parse_identifier()?;
                    let r#where = self.parse_optional_where()?;
                    self.expect_semicolon()?;

                    Ok(Statement::Delete { from, r#where })
                }

                Keyword::Drop => todo!(),

                _ => Err(ParserError::new(format!("unexpected keyword {keyword}"))),
            },

            unexpected => Err(ParserError::new(format!(
                "unexpected first token {unexpected}"
            ))),
        }
    }

    fn parse_comma_separated_expressions(&mut self) -> Result<Vec<Expression>, ParserError> {
        Ok(vec![])
    }

    fn parse_identifier(&mut self) -> Result<String, ParserError> {
        Ok(String::from(""))
    }

    fn parse_optional_where(&mut self) -> Result<Option<Expression>, ParserError> {
        Ok(None)
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
                Token::Word(Word::keyword(Keyword::Select)),
                Token::Whitespace(Whitespace::Space),
                Token::Word(Word::identifier("id")),
                Token::Comma,
                Token::Whitespace(Whitespace::Space),
                Token::Word(Word::identifier("name")),
                Token::Whitespace(Whitespace::Space),
                Token::Word(Word::keyword(Keyword::From)),
                Token::Whitespace(Whitespace::Space),
                Token::Word(Word::identifier("users")),
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
                Token::Word(Word::keyword(Keyword::Select)),
                Token::Whitespace(Whitespace::Space),
                Token::Word(Word::identifier("id")),
                Token::Comma,
                Token::Whitespace(Whitespace::Space),
                Token::Word(Word::identifier("price")),
                Token::Comma,
                Token::Whitespace(Whitespace::Space),
                Token::Word(Word::identifier("discount")),
                Token::Whitespace(Whitespace::Space),
                Token::Word(Word::keyword(Keyword::From)),
                Token::Whitespace(Whitespace::Space),
                Token::Word(Word::identifier("products")),
                Token::Whitespace(Whitespace::Space),
                Token::Word(Word::keyword(Keyword::Where)),
                Token::Whitespace(Whitespace::Space),
                Token::Word(Word::identifier("price")),
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
                Token::Word(Word::keyword(Keyword::Create)),
                Token::Whitespace(Whitespace::Space),
                Token::Word(Word::keyword(Keyword::Table)),
                Token::Whitespace(Whitespace::Space),
                Token::Word(Word::identifier("users")),
                Token::Whitespace(Whitespace::Space),
                Token::LeftParen,
                Token::Word(Word::identifier("id")),
                Token::Whitespace(Whitespace::Space),
                Token::Word(Word::keyword(Keyword::Int)),
                Token::Whitespace(Whitespace::Space),
                Token::Word(Word::keyword(Keyword::Primary)),
                Token::Whitespace(Whitespace::Space),
                Token::Word(Word::keyword(Keyword::Key)),
                Token::Comma,
                Token::Whitespace(Whitespace::Space),
                Token::Word(Word::identifier("name")),
                Token::Whitespace(Whitespace::Space),
                Token::Word(Word::keyword(Keyword::Varchar)),
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
