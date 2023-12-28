use core::iter::Peekable;

use super::{
    statement::{Column, Create, Drop, Expression, Statement, Value},
    token::{Keyword, Token},
    tokenizer::{self, Location, TokenWithLocation, Tokenizer, TokenizerError},
};

struct ParserError {
    message: String,
    location: Location,
}

impl ParserError {
    fn new(message: impl Into<String>, location: Location) -> Self {
        Self {
            message: message.into(),
            location,
        }
    }
}

impl From<TokenizerError> for ParserError {
    fn from(err: TokenizerError) -> Self {
        Self {
            message: format!("syntax error: {}", err.message),
            location: err.location,
        }
    }
}

type ParseResult<T> = Result<T, ParserError>;

/// Main parsing structure. Takes a string input and outputs [`Statement`]
/// instances.
pub(super) struct Parser<'i> {
    /// [`Token`] peekable iterator.
    tokenizer: Peekable<tokenizer::IntoIter<'i>>,
    /// Location of the last token we've consumed from the iterator.
    last_token_location: Location,
}

impl<'i> Parser<'i> {
    /// Creates a new parser for the given `input` string.
    pub fn new(input: &'i str) -> Self {
        Self {
            tokenizer: Tokenizer::new(input).into_iter().peekable(),
            last_token_location: Location::default(),
        }
    }

    /// Attempts to parse the `input` string into a list of [`Statement`]
    /// instances.
    pub fn try_parse(&mut self) -> ParseResult<Vec<Statement>> {
        let mut statements = Vec::new();

        while let Some(_) = self.tokenizer.peek() {
            statements.push(self.parse_statement()?);
        }

        Ok(statements)
    }

    /// Parses a single SQL statement in the input string. If the statement
    /// terminator is not found then it returns [`Err`].
    fn parse_statement(&mut self) -> ParseResult<Statement> {
        let statement = match self.next_after_whitespaces()? {
            Token::Keyword(keyword) => match keyword {
                Keyword::None => Err(self.error("expected SQL statement")),

                Keyword::Select => {
                    let columns = self.parse_comma_separated_expressions()?;
                    self.expect_keyword(Keyword::From)?;

                    let (from, r#where) = self.parse_from_and_optional_where()?;

                    Ok(Statement::Select {
                        columns,
                        from,
                        r#where,
                    })
                }

                Keyword::Create => {
                    let keyword = self.expect_either(Keyword::Database, Keyword::Table)?;
                    let identifier = self.parse_identifier()?;

                    Ok(Statement::Create(match keyword {
                        Keyword::Database => Create::Database(identifier),

                        Keyword::Table => Create::Table {
                            name: identifier,
                            columns: self.parse_schema()?,
                        },

                        _ => unreachable!(),
                    }))
                }

                Keyword::Update => {
                    let table = self.parse_identifier()?;
                    self.expect_keyword(Keyword::Set)?;

                    let columns = self.parse_comma_separated_expressions()?;
                    let r#where = self.parse_optional_where()?;

                    Ok(Statement::Update {
                        table,
                        columns,
                        r#where,
                    })
                }

                Keyword::Insert => {
                    self.expect_keyword(Keyword::Into)?;
                    let into = self.parse_identifier()?;
                    let columns = self.parse_identifier_list()?;

                    self.expect_keyword(Keyword::Values)?;
                    let values = self.parse_values()?;

                    Ok(Statement::Insert {
                        into,
                        columns,
                        values,
                    })
                }

                Keyword::Delete => {
                    self.expect_keyword(Keyword::From)?;
                    let (from, r#where) = self.parse_from_and_optional_where()?;

                    Ok(Statement::Delete { from, r#where })
                }

                Keyword::Drop => {
                    let keyword = self.expect_either(Keyword::Database, Keyword::Table)?;
                    let identifier = self.parse_identifier()?;

                    Ok(Statement::Drop(match keyword {
                        Keyword::Database => Drop::Database(identifier),
                        Keyword::Table => Drop::Table(identifier),
                        _ => unreachable!(),
                    }))
                }

                _ => Err(self.error(format!("unexpected keyword {keyword}"))),
            },

            unexpected => Err(self.error(format!(
                "unexpected initial token {unexpected}. Statements must start with a keyword."
            ))),
        };

        if statement.is_ok() {
            self.expect_semicolon()?;
        }

        statement
    }

    fn parse_expression(&mut self) -> ParseResult<Expression> {
        todo!()
    }

    fn parse_comma_separated_expressions(&mut self) -> ParseResult<Vec<Expression>> {
        todo!()
    }

    fn parse_values(&mut self) -> ParseResult<Vec<Value>> {
        todo!()
    }

    fn parse_schema(&mut self) -> ParseResult<Vec<Column>> {
        todo!()
    }

    fn parse_identifier_list(&mut self) -> ParseResult<Vec<String>> {
        todo!()
    }

    fn parse_identifier(&mut self) -> ParseResult<String> {
        self.next_after_whitespaces().and_then(|token| match token {
            Token::Identifier(ident) => Ok(ident),
            _ => Err(self.error(format!("expected identifier. Got {token} instead"))),
        })
    }

    fn parse_optional_where(&mut self) -> ParseResult<Option<Expression>> {
        if self.consume_optional_keyword(Keyword::Where) {
            Ok(Some(self.parse_expression()?))
        } else {
            Ok(None)
        }
    }

    fn parse_from_and_optional_where(&mut self) -> ParseResult<(String, Option<Expression>)> {
        let from = self.parse_identifier()?;
        let r#where = self.parse_optional_where()?;

        Ok((from, r#where))
    }

    /// Expects `one` keyword or the `other` and returns whichever matches.
    fn expect_either(&mut self, one: Keyword, other: Keyword) -> ParseResult<Keyword> {
        self.expect_keyword(one)
            .or_else(|_| self.expect_keyword(other))
    }

    /// Skips all instances of [`Token::Whitespace`] in the stream.
    fn skip_white_spaces(&mut self) {
        while let Some(Ok(Token::Whitespace(_))) = self.peek_token() {
            self.next_token();
        }
    }

    /// Skips all instances of [`Token::Whitespace`] and returns the next
    /// [`Token`].
    fn next_after_whitespaces(&mut self) -> ParseResult<Token> {
        self.skip_white_spaces();

        match self.next_token() {
            None => Err(self.error("unexpected EOF")),
            Some(result) => Ok(result?),
        }
    }

    /// Ensures that there is at least one instance of [`Token::Whitespace`]
    /// followed by the given `expected` keyword. If so, every token before the
    /// keyword and the keyword itself are consumed.
    fn expect_keyword(&mut self, expected: Keyword) -> ParseResult<Keyword> {
        self.next_after_whitespaces().and_then(|token| match token {
            Token::Keyword(keyword) if keyword == expected => Ok(expected),

            _ => Err(self.error(format!(
                "unexpected token {token}, expected keyword {expected} instead"
            ))),
        })
    }

    /// SQL statements must end with `;`, or [`Token::SemiColon`] in this
    /// context.
    fn expect_semicolon(&mut self) -> ParseResult<()> {
        self.next_after_whitespaces().and_then(|token| match token {
            Token::SemiColon => Ok(()),

            _ => Err(self.error(format!("missing {} statement terminator", Token::SemiColon))),
        })
    }

    /// Consumes all the tokens before and including the given `optional`
    /// keyword. If the keyword is not found, only whitespaces are consumed.
    fn consume_optional_keyword(&mut self, optional: Keyword) -> bool {
        self.skip_white_spaces();

        self.peek_token().is_some_and(|token| match token {
            Ok(Token::Keyword(keyword)) => keyword == &optional,
            _ => false,
        })
    }

    fn error(&self, message: impl Into<String>) -> ParserError {
        ParserError {
            message: message.into(),
            location: self.last_token_location,
        }
    }

    fn peek_token(&mut self) -> Option<Result<&Token, &TokenizerError>> {
        self.tokenizer
            .peek()
            .map(|result| result.as_ref().map(TokenWithLocation::token))
    }

    fn next_token(&mut self) -> Option<Result<Token, TokenizerError>> {
        self.tokenizer.next().map(|result| {
            result.map(|TokenWithLocation { token, location }| {
                self.last_token_location = location;
                token
            })
        })
    }
}
