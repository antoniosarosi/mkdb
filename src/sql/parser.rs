use core::iter::Peekable;

use super::{
    statement::{
        BinaryOperator, Column, Constraint, Create, DataType, Drop, Expression, Statement, Value,
    },
    token::{Keyword, Token},
    tokenizer::{self, Location, TokenWithLocation, Tokenizer, TokenizerError},
};

#[derive(Debug, PartialEq)]
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

/// TODP (Top-Down Operator Precedence) recursive descent parser. See this
/// [tutorial] for an introduction to the algorithms used here and see also the
/// [sqlparser] Github repo for a more complete and robust SQL parser written in
/// Rust. This one is simply a toy parser implemented for the sake of it.
///
/// [tutorial]: https://eli.thegreenplace.net/2010/01/02/top-down-operator-precedence-parsing
/// [sqlparser]: https://github.com/sqlparser-rs/sqlparser-rs
pub(super) struct Parser<'i> {
    /// [`Token`] peekable iterator.
    tokenizer: Peekable<tokenizer::IntoIter<'i>>,
    /// Location of the last token we've consumed from the iterator.
    location: Location,
}

impl<'i> Parser<'i> {
    /// Creates a new parser for the given `input` string.
    pub fn new(input: &'i str) -> Self {
        Self {
            tokenizer: Tokenizer::new(input).into_iter().peekable(),
            location: Location::default(),
        }
    }

    /// Attempts to parse the `input` string into a list of [`Statement`]
    /// instances.
    pub fn try_parse(&mut self) -> ParseResult<Vec<Statement>> {
        let mut statements = Vec::new();

        loop {
            match self.peek_after_whitespaces() {
                Some(Ok(Token::Eof)) | None => return Ok(statements),
                _ => statements.push(self.parse_statement()?),
            }
        }
    }

    /// Parses a single SQL statement in the input string. If the statement
    /// terminator is not found then it returns [`Err`].
    fn parse_statement(&mut self) -> ParseResult<Statement> {
        let Token::Keyword(keyword) = self.next_after_whitespaces()? else {
            return Err(self.error(format!(
                "unexpected initial token. Statements must start with one of the supported keywords."
            )));
        };

        let statement = match keyword {
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
                let keyword = self.expect_one_of(&[Keyword::Database, Keyword::Table])?;
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
                let values = self.parse_comma_separated_expressions()?;

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
                let keyword = self.expect_one_of(&[Keyword::Database, Keyword::Table])?;
                let identifier = self.parse_identifier()?;

                Ok(Statement::Drop(match keyword {
                    Keyword::Database => Drop::Database(identifier),
                    Keyword::Table => Drop::Table(identifier),
                    _ => unreachable!(),
                }))
            }

            Keyword::None => Err(self.error("expected SQL statement")),

            _ => Err(self.error(format!("unexpected initial statement keyword: {keyword}"))),
        };

        if statement.is_ok() {
            self.expect_semicolon()?;
        }

        statement
    }

    fn parse_expression(&mut self, precedence: u8) -> ParseResult<Expression> {
        let mut expr = self.parse_prefix()?;
        let mut next_precedence = self.get_next_precedence();

        while precedence < next_precedence {
            expr = self.parse_infix(expr, next_precedence)?;
            next_precedence = self.get_next_precedence();
        }

        Ok(expr)
    }

    fn get_next_precedence(&mut self) -> u8 {
        let Some(Ok(token)) = self.peek_after_whitespaces() else {
            return 0;
        };

        match token {
            Token::Keyword(Keyword::Or) => 5,
            Token::Keyword(Keyword::And) => 10,
            Token::Eq | Token::Neq | Token::Gt | Token::GtEq | Token::Lt | Token::LtEq => 20,
            Token::Plus | Token::Minus => 30,
            Token::Mul | Token::Div => 40,
            _ => 0,
        }
    }

    fn parse_prefix(&mut self) -> ParseResult<Expression> {
        match self.next_after_whitespaces()? {
            Token::Identifier(ident) => Ok(Expression::Identifier(ident)),
            Token::Mul => Ok(Expression::Wildcard),
            Token::Number(num) => Ok(Expression::Value(Value::Number(num))),
            Token::String(string) => Ok(Expression::Value(Value::String(string))),

            Token::LeftParen => {
                let expr = self.parse_expression(0)?;
                self.expect_token(Token::RightParen)?;
                Ok(expr)
            }

            unexpected => Err(self.error(format!(
                "expected an identifier, raw value or opening parenthesis. Got '{unexpected}' instead",
            ))),
        }
    }

    fn parse_infix(&mut self, left: Expression, precedence: u8) -> ParseResult<Expression> {
        let token = self.next_after_whitespaces()?;

        let operator = match token {
            Token::Plus => BinaryOperator::Plus,
            Token::Minus => BinaryOperator::Minus,
            Token::Div => BinaryOperator::Div,
            Token::Mul => BinaryOperator::Mul,
            Token::Eq => BinaryOperator::Eq,
            Token::Neq => BinaryOperator::Neq,
            Token::Gt => BinaryOperator::Gt,
            Token::GtEq => BinaryOperator::GtEq,
            Token::Lt => BinaryOperator::Lt,
            Token::LtEq => BinaryOperator::LtEq,
            Token::Keyword(Keyword::And) => BinaryOperator::And,
            Token::Keyword(Keyword::Or) => BinaryOperator::Or,

            _ => Err(self.error(format!(
                "expected an operator: [+, -, *, /, =, !=, <, >, <=, >=, AND, OR]. Got {token} instead"
            )))?,
        };

        Ok(Expression::BinaryOperation {
            left: Box::new(left),
            operator,
            right: Box::new(self.parse_expression(precedence)?),
        })
    }

    fn parse_comma_separated_expressions(&mut self) -> ParseResult<Vec<Expression>> {
        let expect_right_paren = self.consume_optional_token(Token::LeftParen);

        let mut vec = vec![self.parse_expression(0)?];
        while self.consume_optional_token(Token::Comma) {
            vec.push(self.parse_expression(0)?);
        }

        if expect_right_paren {
            self.expect_token(Token::RightParen)?;
        }

        Ok(vec)
    }

    fn parse_column(&mut self) -> ParseResult<Column> {
        let name = self.parse_identifier()?;

        let Token::Keyword(keyword) = self.next_after_whitespaces()? else {
            return Err(self.error("expected data type"));
        };

        let data_type = match keyword {
            Keyword::Int => DataType::Int,

            Keyword::Varchar => {
                self.expect_token(Token::LeftParen)?;

                let length = match self.next_after_whitespaces()? {
                    Token::Number(num) => num
                        .parse()
                        .map_err(|_| self.error("incorrect varchar length"))?,
                    _ => Err(self.error("expected varchar length"))?,
                };

                self.expect_token(Token::RightParen)?;
                DataType::Varchar(length)
            }

            unexpected => Err(self.error(format!("unexpected keyword {unexpected}")))?,
        };

        let constraint = match self.consume_one_of_keywords(&[Keyword::Primary, Keyword::Unique]) {
            Keyword::Primary => {
                self.expect_keyword(Keyword::Key)?;
                Some(Constraint::PrimaryKey)
            }

            Keyword::Unique => Some(Constraint::Unique),

            Keyword::None => None,

            _ => unreachable!(),
        };

        Ok(Column {
            name,
            data_type,
            constraint,
        })
    }

    fn consume_one_of_keywords(&mut self, keywords: &[Keyword]) -> Keyword {
        for keyword in keywords {
            if self.consume_optional_keyword(*keyword) {
                return *keyword;
            }
        }

        Keyword::None
    }

    fn parse_schema(&mut self) -> ParseResult<Vec<Column>> {
        self.expect_token(Token::LeftParen)?;
        let mut columns = vec![self.parse_column()?];
        while self.consume_optional_token(Token::Comma) {
            columns.push(self.parse_column()?);
        }
        self.expect_token(Token::RightParen)?;

        Ok(columns)
    }

    fn parse_identifier_list(&mut self) -> ParseResult<Vec<String>> {
        self.expect_token(Token::LeftParen)?;
        let mut identifiers = vec![self.parse_identifier()?];
        while self.consume_optional_token(Token::Comma) {
            identifiers.push(self.parse_identifier()?);
        }

        self.expect_token(Token::RightParen)?;

        Ok(identifiers)
    }

    fn parse_identifier(&mut self) -> ParseResult<String> {
        self.next_after_whitespaces().and_then(|token| match token {
            Token::Identifier(ident) => Ok(ident),
            _ => Err(self.error(format!("expected identifier. Got {token} instead"))),
        })
    }

    fn parse_optional_where(&mut self) -> ParseResult<Option<Expression>> {
        if self.consume_optional_keyword(Keyword::Where) {
            Ok(Some(self.parse_expression(0)?))
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
    fn expect_one_of(&mut self, keywords: &[Keyword]) -> ParseResult<Keyword> {
        match self.consume_one_of_keywords(keywords) {
            Keyword::None => {
                let token = self.next_after_whitespaces()?;
                Err(self.error(format!("expected one of {keywords:?}. Got {token} instead")))
            }
            keyword => Ok(keyword),
        }
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

        let t = self.next_token();

        match t {
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
    fn expect_semicolon(&mut self) -> ParseResult<Token> {
        self.expect_token(Token::SemiColon)
            .map_err(|_| self.error(format!("missing {} statement terminator", Token::SemiColon)))
    }

    fn expect_token(&mut self, expect: Token) -> ParseResult<Token> {
        self.next_after_whitespaces().and_then(|token| {
            if token == expect {
                Ok(token)
            } else {
                Err(self.error(format!("expected token {expect}. Got {token} instead")))
            }
        })
    }

    /// Consumes all the tokens before and including the given `optional`
    /// keyword. If the keyword is not found, only whitespaces are consumed.
    fn consume_optional_keyword(&mut self, optional: Keyword) -> bool {
        self.consume_optional_token(Token::Keyword(optional))
    }

    fn consume_optional_token(&mut self, optional: Token) -> bool {
        match self.peek_after_whitespaces() {
            Some(Ok(token)) if token == &optional => {
                self.next_token();
                true
            }
            _ => false,
        }
    }

    fn error(&self, message: impl Into<String>) -> ParserError {
        ParserError {
            message: message.into(),
            location: self.location,
        }
    }

    fn peek_token(&mut self) -> Option<Result<&Token, &TokenizerError>> {
        self.tokenizer
            .peek()
            .map(|result| result.as_ref().map(TokenWithLocation::token))
    }

    fn peek_after_whitespaces(&mut self) -> Option<Result<&Token, &TokenizerError>> {
        self.skip_white_spaces();
        self.peek_token()
    }

    fn next_token(&mut self) -> Option<Result<Token, TokenizerError>> {
        self.tokenizer.next().map(|result| {
            result.map(|TokenWithLocation { token, location }| {
                self.location = location;
                token
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_simple_select() {
        let sql = "SELECT id, name FROM users;";

        assert_eq!(
            Parser::new(sql).parse_statement(),
            Ok(Statement::Select {
                columns: vec![
                    Expression::Identifier("id".into()),
                    Expression::Identifier("name".into())
                ],
                from: "users".into(),
                r#where: None
            })
        )
    }

    #[test]
    fn parse_select_wildcard() {
        let sql = "SELECT * FROM users;";

        assert_eq!(
            Parser::new(sql).parse_statement(),
            Ok(Statement::Select {
                columns: vec![Expression::Wildcard],
                from: "users".into(),
                r#where: None
            })
        )
    }

    #[test]
    fn parse_select_where() {
        let sql = "SELECT id, price, discount FROM products WHERE price >= 100;";

        assert_eq!(
            Parser::new(sql).parse_statement(),
            Ok(Statement::Select {
                columns: vec![
                    Expression::Identifier("id".into()),
                    Expression::Identifier("price".into()),
                    Expression::Identifier("discount".into())
                ],
                from: "products".into(),
                r#where: Some(Expression::BinaryOperation {
                    left: Box::new(Expression::Identifier("price".into())),
                    operator: BinaryOperator::GtEq,
                    right: Box::new(Expression::Value(Value::Number("100".into())))
                })
            })
        )
    }

    #[test]
    fn parse_select_with_expressions() {
        let sql = r#"
            SELECT id, price, discount, price * discount / 100
            FROM products
            WHERE 100 <= price AND price < 1000 OR discount < 10 + (2 * 20);
        "#;

        assert_eq!(
            Parser::new(sql).parse_statement(),
            Ok(Statement::Select {
                columns: vec![
                    Expression::Identifier("id".into()),
                    Expression::Identifier("price".into()),
                    Expression::Identifier("discount".into()),
                    Expression::BinaryOperation {
                        left: Box::new(Expression::BinaryOperation {
                            left: Box::new(Expression::Identifier("price".into())),
                            operator: BinaryOperator::Mul,
                            right: Box::new(Expression::Identifier("discount".into())),
                        }),
                        operator: BinaryOperator::Div,
                        right: Box::new(Expression::Value(Value::Number("100".into()))),
                    }
                ],
                from: "products".into(),
                r#where: Some(Expression::BinaryOperation {
                    left: Box::new(Expression::BinaryOperation {
                        left: Box::new(Expression::BinaryOperation {
                            left: Box::new(Expression::Value(Value::Number("100".into()))),
                            operator: BinaryOperator::LtEq,
                            right: Box::new(Expression::Identifier("price".into())),
                        }),
                        operator: BinaryOperator::And,
                        right: Box::new(Expression::BinaryOperation {
                            left: Box::new(Expression::Identifier("price".into())),
                            operator: BinaryOperator::Lt,
                            right: Box::new(Expression::Value(Value::Number("1000".into()))),
                        }),
                    }),
                    operator: BinaryOperator::Or,
                    right: Box::new(Expression::BinaryOperation {
                        left: Box::new(Expression::Identifier("discount".into())),
                        operator: BinaryOperator::Lt,
                        right: Box::new(Expression::BinaryOperation {
                            left: Box::new(Expression::Value(Value::Number("10".into()))),
                            operator: BinaryOperator::Plus,
                            right: Box::new(Expression::BinaryOperation {
                                left: Box::new(Expression::Value(Value::Number("2".into()))),
                                operator: BinaryOperator::Mul,
                                right: Box::new(Expression::Value(Value::Number("20".into()))),
                            })
                        }),
                    })
                })
            })
        )
    }

    #[test]
    fn parse_create_database() {
        let sql = "CREATE DATABASE test;";

        assert_eq!(
            Parser::new(sql).parse_statement(),
            Ok(Statement::Create(Create::Database("test".into())))
        )
    }

    #[test]
    fn parse_create_table() {
        let sql = r#"
            CREATE TABLE users (
                id INT PRIMARY KEY,
                name VARCHAR(255),
                email VARCHAR(255) UNIQUE
            );
        "#;

        assert_eq!(
            Parser::new(sql).parse_statement(),
            Ok(Statement::Create(Create::Table {
                name: "users".into(),
                columns: vec![
                    Column {
                        name: "id".into(),
                        data_type: DataType::Int,
                        constraint: Some(Constraint::PrimaryKey),
                    },
                    Column {
                        name: "name".into(),
                        data_type: DataType::Varchar(255),
                        constraint: None,
                    },
                    Column {
                        name: "email".into(),
                        data_type: DataType::Varchar(255),
                        constraint: Some(Constraint::Unique),
                    },
                ]
            }))
        )
    }

    #[test]
    fn parse_simple_update() {
        let sql = "UPDATE users SET is_admin = 1;";

        assert_eq!(
            Parser::new(sql).parse_statement(),
            Ok(Statement::Update {
                table: "users".into(),
                columns: vec![Expression::BinaryOperation {
                    left: Box::new(Expression::Identifier("is_admin".into())),
                    operator: BinaryOperator::Eq,
                    right: Box::new(Expression::Value(Value::Number("1".into()))),
                }],
                r#where: None,
            })
        )
    }

    #[test]
    fn parse_update_where() {
        let sql = r#"
            UPDATE products
            SET price = price - 10, discount = 15, stock = 10
            WHERE price > 100;
        "#;

        assert_eq!(
            Parser::new(sql).parse_statement(),
            Ok(Statement::Update {
                table: "products".into(),
                columns: vec![
                    Expression::BinaryOperation {
                        left: Box::new(Expression::Identifier("price".into())),
                        operator: BinaryOperator::Eq,
                        right: Box::new(Expression::BinaryOperation {
                            left: Box::new(Expression::Identifier("price".into())),
                            operator: BinaryOperator::Minus,
                            right: Box::new(Expression::Value(Value::Number("10".into()))),
                        }),
                    },
                    Expression::BinaryOperation {
                        left: Box::new(Expression::Identifier("discount".into())),
                        operator: BinaryOperator::Eq,
                        right: Box::new(Expression::Value(Value::Number("15".into()))),
                    },
                    Expression::BinaryOperation {
                        left: Box::new(Expression::Identifier("stock".into())),
                        operator: BinaryOperator::Eq,
                        right: Box::new(Expression::Value(Value::Number("10".into()))),
                    }
                ],
                r#where: Some(Expression::BinaryOperation {
                    left: Box::new(Expression::Identifier("price".into())),
                    operator: BinaryOperator::Gt,
                    right: Box::new(Expression::Value(Value::Number("100".into()))),
                })
            })
        )
    }

    #[test]
    fn parse_delete_from() {
        let sql = "DELETE FROM products;";

        assert_eq!(
            Parser::new(sql).parse_statement(),
            Ok(Statement::Delete {
                from: "products".into(),
                r#where: None
            })
        )
    }

    #[test]
    fn parse_delete_from_where() {
        let sql = "DELETE FROM products WHERE price > 5000;";

        assert_eq!(
            Parser::new(sql).parse_statement(),
            Ok(Statement::Delete {
                from: "products".into(),
                r#where: Some(Expression::BinaryOperation {
                    left: Box::new(Expression::Identifier("price".into())),
                    operator: BinaryOperator::Gt,
                    right: Box::new(Expression::Value(Value::Number("5000".into()))),
                })
            })
        )
    }

    #[test]
    fn parse_insert_into() {
        let sql = r#"INSERT INTO users (id, name, email) VALUES (1, "Test", "test@test.com");"#;

        assert_eq!(
            Parser::new(sql).parse_statement(),
            Ok(Statement::Insert {
                into: "users".into(),
                columns: ["id", "name", "email"].map(String::from).into(),
                values: vec![
                    Expression::Value(Value::Number("1".into())),
                    Expression::Value(Value::String("Test".into())),
                    Expression::Value(Value::String("test@test.com".into())),
                ]
            })
        );
    }

    #[test]
    fn parse_drop_database() {
        let sql = "DROP DATABASE test;";

        assert_eq!(
            Parser::new(sql).parse_statement(),
            Ok(Statement::Drop(Drop::Database("test".into())))
        )
    }

    #[test]
    fn parse_drop_table() {
        let sql = "DROP TABLE test;";

        assert_eq!(
            Parser::new(sql).parse_statement(),
            Ok(Statement::Drop(Drop::Table("test".into())))
        )
    }

    #[test]
    fn parse_multiple_statements() {
        let sql = r#"
            DROP TABLE test;
            UPDATE users SET is_admin = 1;
            SELECT * FROM products;
        "#;

        assert_eq!(
            Parser::new(sql).try_parse(),
            Ok(vec![
                Statement::Drop(Drop::Table("test".into())),
                Statement::Update {
                    table: "users".into(),
                    columns: vec![Expression::BinaryOperation {
                        left: Box::new(Expression::Identifier("is_admin".into())),
                        operator: BinaryOperator::Eq,
                        right: Box::new(Expression::Value(Value::Number("1".into()))),
                    }],
                    r#where: None,
                },
                Statement::Select {
                    columns: vec![Expression::Wildcard],
                    from: "products".into(),
                    r#where: None,
                }
            ])
        )
    }

    #[test]
    fn arithmetic_operator_precedence() {
        let expr = "price * discount / 100 < 10 + 20 * 30";

        assert_eq!(
            Parser::new(expr).parse_expression(0),
            Ok(Expression::BinaryOperation {
                left: Box::new(Expression::BinaryOperation {
                    left: Box::new(Expression::BinaryOperation {
                        left: Box::new(Expression::Identifier("price".into())),
                        operator: BinaryOperator::Mul,
                        right: Box::new(Expression::Identifier("discount".into())),
                    }),
                    operator: BinaryOperator::Div,
                    right: Box::new(Expression::Value(Value::Number("100".into()))),
                }),
                operator: BinaryOperator::Lt,
                right: Box::new(Expression::BinaryOperation {
                    left: Box::new(Expression::Value(Value::Number("10".into()))),
                    operator: BinaryOperator::Plus,
                    right: Box::new(Expression::BinaryOperation {
                        left: Box::new(Expression::Value(Value::Number("20".into()))),
                        operator: BinaryOperator::Mul,
                        right: Box::new(Expression::Value(Value::Number("30".into()))),
                    })
                })
            })
        )
    }

    #[test]
    fn nested_arithmetic_precedence() {
        let expr = "price * discount >= 10 - (20 + 50) / (2 * (4 + (1 - 1)))";

        assert_eq!(
            Parser::new(expr).parse_expression(0),
            Ok(Expression::BinaryOperation {
                left: Box::new(Expression::BinaryOperation {
                    left: Box::new(Expression::Identifier("price".into())),
                    operator: BinaryOperator::Mul,
                    right: Box::new(Expression::Identifier("discount".into())),
                }),
                operator: BinaryOperator::GtEq,
                right: Box::new(Expression::BinaryOperation {
                    left: Box::new(Expression::Value(Value::Number("10".into()))),
                    operator: BinaryOperator::Minus,
                    right: Box::new(Expression::BinaryOperation {
                        left: Box::new(Expression::BinaryOperation {
                            left: Box::new(Expression::Value(Value::Number("20".into()))),
                            operator: BinaryOperator::Plus,
                            right: Box::new(Expression::Value(Value::Number("50".into()))),
                        }),
                        operator: BinaryOperator::Div,
                        right: Box::new(Expression::BinaryOperation {
                            left: Box::new(Expression::Value(Value::Number("2".into()))),
                            operator: BinaryOperator::Mul,
                            right: Box::new(Expression::BinaryOperation {
                                left: Box::new(Expression::Value(Value::Number("4".into()))),
                                operator: BinaryOperator::Plus,
                                right: Box::new(Expression::BinaryOperation {
                                    left: Box::new(Expression::Value(Value::Number("1".into()))),
                                    operator: BinaryOperator::Minus,
                                    right: Box::new(Expression::Value(Value::Number("1".into()))),
                                })
                            })
                        })
                    })
                })
            })
        )
    }

    #[test]
    fn and_or_operators_precedence() {
        let expr = "100 <= price AND price <= 200 OR price > 1000";

        assert_eq!(
            Parser::new(expr).parse_expression(0),
            Ok(Expression::BinaryOperation {
                left: Box::new(Expression::BinaryOperation {
                    left: Box::new(Expression::BinaryOperation {
                        left: Box::new(Expression::Value(Value::Number("100".into()))),
                        operator: BinaryOperator::LtEq,
                        right: Box::new(Expression::Identifier("price".into())),
                    }),
                    operator: BinaryOperator::And,
                    right: Box::new(Expression::BinaryOperation {
                        left: Box::new(Expression::Identifier("price".into())),
                        operator: BinaryOperator::LtEq,
                        right: Box::new(Expression::Value(Value::Number("200".into()))),
                    }),
                }),
                operator: BinaryOperator::Or,
                right: Box::new(Expression::BinaryOperation {
                    left: Box::new(Expression::Identifier("price".into())),
                    operator: BinaryOperator::Gt,
                    right: Box::new(Expression::Value(Value::Number("1000".into()))),
                })
            })
        )
    }
}
