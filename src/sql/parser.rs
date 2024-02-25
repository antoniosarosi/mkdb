use core::iter::Peekable;
use std::fmt::Display;

use super::{
    statement::{
        BinaryOperator, Column, Constraint, Create, DataType, Drop, Expression, Statement,
        UnaryOperator, Value,
    },
    token::{Keyword, Token},
    tokenizer::{self, Location, TokenWithLocation, Tokenizer, TokenizerError},
};

/// See [`Parser::get_next_precedence`] for details.
const UNARY_ARITHMETIC_OPERATOR_PRECEDENCE: u8 = 50;

/// Parser error kind.
#[derive(Debug, PartialEq)]
pub(crate) enum ErrorKind {
    TokenizerError(tokenizer::ErrorKind),

    Expected { expected: Token, found: Token },

    ExpectedOneOf { expected: Vec<Token>, found: Token },

    UnexpectedOrUnsupported(Token),

    UnexpectedEof,

    Other(String),
}

impl ErrorKind {
    /// We need a display value for [`Token`] variants that hold inner data.
    ///
    /// We could also use something like [strum], but it's unnecessary for now.
    ///
    /// [strum]: https://docs.rs/strum/latest/strum/derive.EnumDiscriminants.html
    fn expected_token_string(token: &Token) -> String {
        match token {
            Token::Identifier(_) => "identifier".into(),
            Token::Number(_) => "number".into(),
            Token::String(_) => "string".into(),
            _ => format!("'{token}'"),
        }
    }
}

impl Display for ErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ErrorKind::TokenizerError(err) => write!(f, "{err}"),

            ErrorKind::Expected { expected, found } => write!(
                f,
                "expected {}, found '{found}' instead",
                ErrorKind::expected_token_string(&expected)
            ),

            ErrorKind::ExpectedOneOf { expected, found } => {
                let mut one_of = String::new(); // Token1, Token2 or Token3

                one_of.push_str(&ErrorKind::expected_token_string(&expected[0]));

                for token in &expected[1..expected.len() - 1] {
                    one_of.push_str(", ");
                    one_of.push_str(&ErrorKind::expected_token_string(token));
                }

                if expected.len() > 1 {
                    one_of.push_str(" or ");
                    one_of.push_str(&ErrorKind::expected_token_string(
                        &expected[expected.len() - 1],
                    ));
                }

                write!(f, "expected {one_of}. Found '{found}' instead")
            }

            ErrorKind::UnexpectedOrUnsupported(token) => {
                write!(f, "unexpected or unsupported token {token}")
            }

            ErrorKind::UnexpectedEof => f.write_str("unexpected EOF"),

            ErrorKind::Other(message) => f.write_str(&message),
        }
    }
}

/// Holds the error kind and location of the token where the error was
/// originated.
#[derive(Debug, PartialEq)]
pub(crate) struct ParserError {
    pub kind: ErrorKind,
    pub location: Location,
}

impl Display for ParserError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.kind.to_string())
    }
}

impl ParserError {
    fn new(kind: ErrorKind, location: Location) -> Self {
        Self { kind, location }
    }
}

impl From<TokenizerError> for ParserError {
    fn from(TokenizerError { kind, location }: TokenizerError) -> Self {
        Self {
            kind: ErrorKind::TokenizerError(kind),
            location,
        }
    }
}

pub(crate) type ParseResult<T> = Result<T, ParserError>;

/// TDOP (Top-Down Operator Precedence) recursive descent parser.
///
/// See this [tutorial] for an introduction to the algorithms used here and see
/// also the [sqlparser] Github repo for a more complete and robust SQL parser
/// written in Rust. This one is simply a toy parser implemented for the sake of
/// it.
///
/// [tutorial]: https://eli.thegreenplace.net/2010/01/02/top-down-operator-precedence-parsing
/// [sqlparser]: https://github.com/sqlparser-rs/sqlparser-rs
pub(crate) struct Parser<'i> {
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
            match self.peek_token() {
                Some(Ok(Token::Eof)) | None => return Ok(statements),
                _ => statements.push(self.parse_statement()?),
            }
        }
    }

    /// Parses a single SQL statement in the input string.
    ///
    /// If the statement terminator is not found then it returns [`Err`].
    pub fn parse_statement(&mut self) -> ParseResult<Statement> {
        let statement = match self.expect_one_of(&Self::supported_statements())? {
            Keyword::Select => {
                let columns = self.parse_comma_separated_expressions()?;
                self.expect_keyword(Keyword::From)?;

                let (from, r#where) = self.parse_from_and_optional_where()?;

                let order_by = self.parse_optional_order_by()?;

                Statement::Select {
                    columns,
                    from,
                    r#where,
                    order_by,
                }
            }

            Keyword::Create => {
                let keyword =
                    self.expect_one_of(&[Keyword::Database, Keyword::Table, Keyword::Index])?;

                let identifier = self.parse_identifier()?;

                Statement::Create(match keyword {
                    Keyword::Database => Create::Database(identifier),

                    Keyword::Table => Create::Table {
                        name: identifier,
                        columns: self.parse_column_definitions()?,
                    },

                    Keyword::Index => {
                        self.expect_keyword(Keyword::On)?;
                        let table = self.parse_identifier()?;

                        self.expect_token(Token::LeftParen)?;
                        let column = self.parse_identifier()?;
                        self.expect_token(Token::RightParen)?;

                        Create::Index {
                            name: identifier,
                            table,
                            column,
                        }
                    }

                    _ => unreachable!(),
                })
            }

            Keyword::Update => {
                let table = self.parse_identifier()?;
                self.expect_keyword(Keyword::Set)?;

                let columns = self.parse_comma_separated_expressions()?;
                let r#where = self.parse_optional_where()?;

                Statement::Update {
                    table,
                    columns,
                    r#where,
                }
            }

            Keyword::Insert => {
                self.expect_keyword(Keyword::Into)?;
                let into = self.parse_identifier()?;
                let columns = self.parse_identifier_list(true)?;

                self.expect_keyword(Keyword::Values)?;
                let values = self.parse_comma_separated_expressions()?;

                Statement::Insert {
                    into,
                    columns,
                    values,
                }
            }

            Keyword::Delete => {
                self.expect_keyword(Keyword::From)?;
                let (from, r#where) = self.parse_from_and_optional_where()?;

                Statement::Delete { from, r#where }
            }

            Keyword::Drop => {
                let keyword = self.expect_one_of(&[Keyword::Database, Keyword::Table])?;
                let identifier = self.parse_identifier()?;

                Statement::Drop(match keyword {
                    Keyword::Database => Drop::Database(identifier),
                    Keyword::Table => Drop::Table(identifier),
                    _ => unreachable!(),
                })
            }

            _ => unreachable!(),
        };

        self.expect_token(Token::SemiColon)?;
        Ok(statement)
    }

    /// Starts the TDOP recursive descent.
    ///
    /// TDOP consists of 3 functions that call each other recursively:
    ///
    /// - [`Self::parse_expr`]
    /// - [`Self::parse_prefix`]
    /// - [`Self::parse_infix`]
    ///
    /// This one simply initiates the process, see the others for details and
    /// see the [tutorial] mentioned above to understand how the algorithm
    /// works.
    ///
    /// [tutorial]: https://eli.thegreenplace.net/2010/01/02/top-down-operator-precedence-parsing
    fn parse_expression(&mut self) -> ParseResult<Expression> {
        self.parse_expr(0)
    }

    /// Main TDOP loop.
    fn parse_expr(&mut self, precedence: u8) -> ParseResult<Expression> {
        let mut expr = self.parse_prefix()?;
        let mut next_precedence = self.get_next_precedence();

        while precedence < next_precedence {
            expr = self.parse_infix(expr, next_precedence)?;
            next_precedence = self.get_next_precedence();
        }

        Ok(expr)
    }

    /// Parses the beginning of an expression.
    fn parse_prefix(&mut self) -> ParseResult<Expression> {
        match self.next_token()? {
            Token::Identifier(ident) => Ok(Expression::Identifier(ident)),
            Token::Mul => Ok(Expression::Wildcard),
            Token::Number(num) => Ok(Expression::Value(Value::Number(num.parse().unwrap()))),
            Token::String(string) => Ok(Expression::Value(Value::String(string))),
            Token::Keyword(Keyword::True) => Ok(Expression::Value(Value::Bool(true))),
            Token::Keyword(Keyword::False) => Ok(Expression::Value(Value::Bool(false))),

            token @ (Token::Minus | Token::Plus) => {
                let operator = match token {
                    Token::Plus => UnaryOperator::Plus,
                    Token::Minus => UnaryOperator::Minus,
                    _ => unreachable!(),
                };

                let expr = Box::new(self.parse_expr(UNARY_ARITHMETIC_OPERATOR_PRECEDENCE)?);

                Ok(Expression::UnaryOperation { operator, expr })
            }

            Token::LeftParen => {
                let expr = self.parse_expression()?;
                self.expect_token(Token::RightParen)?;
                Ok(expr)
            }

            unexpected => Err(self.error(ErrorKind::ExpectedOneOf {
                expected: vec![
                    Token::Identifier(Default::default()),
                    Token::Number(Default::default()),
                    Token::String(Default::default()),
                    Token::Mul,
                    Token::Minus,
                    Token::Plus,
                    Token::LeftParen,
                ],
                found: unexpected,
            })),
        }
    }

    /// Parses an infix expression in the form of
    /// (left expr | operator | right expr).
    fn parse_infix(&mut self, left: Expression, precedence: u8) -> ParseResult<Expression> {
        let operator = match self.next_token()? {
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

            unexpected => Err(self.error(ErrorKind::ExpectedOneOf {
                expected: Self::supported_operators(),
                found: unexpected,
            }))?,
        };

        Ok(Expression::BinaryOperation {
            left: Box::new(left),
            operator,
            right: Box::new(self.parse_expr(precedence)?),
        })
    }

    /// Returns the precedence value of the next operator in the stream.
    fn get_next_precedence(&mut self) -> u8 {
        let Some(Ok(token)) = self.peek_token() else {
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

    /// Parses a column definition for `CREATE TABLE` statements.
    fn parse_column(&mut self) -> ParseResult<Column> {
        let name = self.parse_identifier()?;

        let data_type = match self.expect_one_of(&Self::supported_data_types())? {
            int @ (Keyword::Int | Keyword::BigInt) => {
                let unsigned = self.consume_optional_keyword(Keyword::Unsigned);
                match (int, unsigned) {
                    (Keyword::Int, true) => DataType::UnsignedInt,
                    (Keyword::Int, false) => DataType::Int,
                    (Keyword::BigInt, true) => DataType::UnsignedBigInt,
                    (Keyword::BigInt, false) => DataType::BigInt,
                    _ => unreachable!(),
                }
            }

            Keyword::Varchar => {
                self.expect_token(Token::LeftParen)?;

                let length = match self.next_token()? {
                    Token::Number(num) => num.parse().map_err(|_| {
                        self.error(ErrorKind::Other(
                            "incorrect VARCHAR length definition".into(),
                        ))
                    })?,
                    unexpected => Err(self.error(ErrorKind::Expected {
                        expected: Token::Number(Default::default()),
                        found: unexpected,
                    }))?,
                };

                self.expect_token(Token::RightParen)?;
                DataType::Varchar(length)
            }

            Keyword::Bool => DataType::Bool,

            _ => unreachable!(),
        };

        let constraint = match self.consume_one_of(&[Keyword::Primary, Keyword::Unique]) {
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

    /// Parses the next identifier in the stream or fails if it's not an
    /// identifier.
    fn parse_identifier(&mut self) -> ParseResult<String> {
        self.next_token().and_then(|token| match token {
            Token::Identifier(ident) => Ok(ident),

            _ => Err(self.error(ErrorKind::Expected {
                expected: Token::Identifier(Default::default()),
                found: token,
            })),
        })
    }

    /// Takes a `subparser` as input and calls it after every instance of
    /// [`Token::Comma`].
    fn parse_comma_separated<T>(
        &mut self,
        mut subparser: impl FnMut(&mut Self) -> ParseResult<T>,
        required_parenthesis: bool,
    ) -> ParseResult<Vec<T>> {
        let found_left_paren = self.consume_optional_token(Token::LeftParen);

        if required_parenthesis && !found_left_paren {
            let found = self.next_token()?;
            return Err(self.error(ErrorKind::Expected {
                expected: Token::LeftParen,
                found,
            }));
        }

        let mut results = vec![subparser(self)?];
        while self.consume_optional_token(Token::Comma) {
            results.push(subparser(self)?);
        }

        if found_left_paren {
            self.expect_token(Token::RightParen)?;
        }

        Ok(results)
    }

    /// Used to parse the expressions after `SELECT`, `WHERE`, `SET` or `VALUES`.
    fn parse_comma_separated_expressions(&mut self) -> ParseResult<Vec<Expression>> {
        self.parse_comma_separated(Self::parse_expression, false)
    }

    /// Used to parse `CREATE TABLE` column definitions.
    fn parse_column_definitions(&mut self) -> ParseResult<Vec<Column>> {
        self.parse_comma_separated(Self::parse_column, true)
    }

    /// Expects a list of identifiers, not complete expressions.
    fn parse_identifier_list(&mut self, required_parenthesis: bool) -> ParseResult<Vec<String>> {
        self.parse_comma_separated(Self::parse_identifier, required_parenthesis)
    }

    /// Parses the entire `WHERE` clause if the next token is [`Keyword::Where`].
    fn parse_optional_where(&mut self) -> ParseResult<Option<Expression>> {
        if self.consume_optional_keyword(Keyword::Where) {
            Ok(Some(self.parse_expression()?))
        } else {
            Ok(None)
        }
    }

    /// Parses an optional `FROM ... WHERE ...` construct.
    ///
    /// These statements all have a `FROM` clause and an optional `WHERE`
    /// clause:
    ///
    /// ```sql
    /// SELECT * FROM table WHERE condition;
    /// UPDATE table SET column = "value" WHERE condition;
    /// DELETE FROM table WHERE condition;
    /// ```
    fn parse_from_and_optional_where(&mut self) -> ParseResult<(String, Option<Expression>)> {
        let from = self.parse_identifier()?;
        let r#where = self.parse_optional_where()?;

        Ok((from, r#where))
    }

    /// Parses the `ORDER BY` clause at the end of `SELECT` statements.
    ///
    /// It only works with identifiers (not expressions) for now.
    fn parse_optional_order_by(&mut self) -> ParseResult<Vec<String>> {
        if self.consume_optional_keyword(Keyword::Order) {
            self.expect_keyword(Keyword::By)?;
            self.parse_identifier_list(false)
        } else {
            Ok(Vec::new())
        }
    }

    /// Same as [`Self::expect_token`] but takes [`Keyword`] variants instead.
    fn expect_keyword(&mut self, expected: Keyword) -> ParseResult<Keyword> {
        self.expect_token(Token::Keyword(expected))
            .map(|_| expected)
    }

    /// Automatically fails if the `expected` token is not the next one in the
    /// stream (after whitespaces).
    ///
    /// If it is, it will be returned back.
    fn expect_token(&mut self, expected: Token) -> ParseResult<Token> {
        self.next_token().and_then(|token| {
            if token == expected {
                Ok(token)
            } else {
                Err(self.error(ErrorKind::Expected {
                    expected,
                    found: token,
                }))
            }
        })
    }

    /// Automatically fails if the next token does not match one of the given
    /// `keywords`.
    ///
    /// If it does, then the keyword that matched is returned back.
    fn expect_one_of<'k, K>(&mut self, keywords: &'k K) -> ParseResult<Keyword>
    where
        &'k K: IntoIterator<Item = &'k Keyword>,
    {
        match self.consume_one_of(keywords) {
            Keyword::None => {
                let token = self.next_token()?;
                Err(self.error(ErrorKind::ExpectedOneOf {
                    expected: Self::tokens_from_keywords(keywords),
                    found: token,
                }))
            }
            keyword => Ok(keyword),
        }
    }

    /// Consumes all the tokens before and including the given `optional`
    /// keyword.
    ///
    /// If the keyword is not found, only whitespaces are consumed.
    fn consume_optional_keyword(&mut self, optional: Keyword) -> bool {
        self.consume_optional_token(Token::Keyword(optional))
    }

    /// If the next token in the stream matches the given `optional` token, then
    /// this function consumes the token and returns `true`.
    ///
    /// Otherwise the token will not be consumed and the tokenizer will still be
    /// pointing at it.
    fn consume_optional_token(&mut self, optional: Token) -> bool {
        match self.peek_token() {
            Some(Ok(token)) if token == &optional => {
                let _ = self.next_token();
                true
            }
            _ => false,
        }
    }

    /// Consumes the next token in the stream only if it matches one of the
    /// given `keywords`.
    ///
    /// If so, the matched [`Keyword`] variant is returned. Otherwise returns
    /// [`Keyword::None`].
    fn consume_one_of<'k, K>(&mut self, keywords: &'k K) -> Keyword
    where
        &'k K: IntoIterator<Item = &'k Keyword>,
    {
        *keywords
            .into_iter()
            .find(|keyword| self.consume_optional_keyword(**keyword))
            .unwrap_or(&Keyword::None)
    }

    /// Builds an instance of [`ParserError`] giving it the current
    /// [`Self::location`].
    fn error(&self, kind: ErrorKind) -> ParserError {
        ParserError {
            kind,
            location: self.location,
        }
    }

    /// Skips all instances of [`Token::Whitespace`] in the stream.
    fn skip_white_spaces(&mut self) {
        while let Some(Ok(Token::Whitespace(_))) = self.peek_token_in_stream() {
            let _ = self.next_token_in_stream();
        }
    }

    /// Skips all instances of [`Token::Whitespace`] and returns the next
    /// relevant [`Token`].
    ///
    /// This function doesn't return [`Option`] because it's used in all cases
    /// to expect some token. If we dont' expect any more tokens (for example,
    /// after we've found [`Token::SemiColon`] or [`Token::Eof`]) then we just
    /// won't call this function at all.
    fn next_token(&mut self) -> ParseResult<Token> {
        self.skip_white_spaces();
        self.next_token_in_stream()
    }

    /// Returns a reference to the next relevant [`Token`] after whitespaces
    /// without consuming it.
    fn peek_token(&mut self) -> Option<Result<&Token, &TokenizerError>> {
        self.skip_white_spaces();
        self.peek_token_in_stream()
    }

    /// Consumes and returns the next [`Token`] in the stream updating
    /// [`Self::location`] in the process.
    ///
    /// This removes the need to use [`TokenWithLocation`] instances. This
    /// method should not be called after [`Token::Eof`] has been returned once
    /// since it will error with [`ErrorKind::UnexpectedEof`].
    fn next_token_in_stream(&mut self) -> ParseResult<Token> {
        match self.tokenizer.peek() {
            None => Err(self.error(ErrorKind::UnexpectedEof)),

            _ => {
                let token = self.tokenizer.next().unwrap()?;
                self.location = token.location;
                Ok(token.variant)
            }
        }
    }

    /// Same as [`Self::next_token_in_stream`] but does not consume the next
    /// token.
    fn peek_token_in_stream(&mut self) -> Option<Result<&Token, &TokenizerError>> {
        self.tokenizer
            .peek()
            .map(|result| result.as_ref().map(TokenWithLocation::token))
    }

    /// Maps [`Keyword`] variants to [`Token`] variants.
    fn tokens_from_keywords<'k, K>(keywords: &'k K) -> Vec<Token>
    where
        &'k K: IntoIterator<Item = &'k Keyword>,
    {
        keywords.into_iter().map(From::from).collect()
    }
}

// Supported statements and keywords.
impl<'i> Parser<'i> {
    /// Initial SQL statements that we support. For now, CRUD stuff only.
    fn supported_statements() -> Vec<Keyword> {
        vec![
            Keyword::Select,
            Keyword::Create,
            Keyword::Update,
            Keyword::Insert,
            Keyword::Delete,
            Keyword::Drop,
        ]
    }

    /// Data type that can be used for column definitions.
    fn supported_data_types() -> Vec<Keyword> {
        // For integers types the unsigned version doesn't need to be here.
        // Specifying the initial keyword (INT, BIGINT) takes care of the
        // optional UNSIGNED that follows.
        vec![
            Keyword::Int,
            Keyword::BigInt,
            Keyword::Bool,
            Keyword::Varchar,
        ]
    }

    /// Supported binary operators.
    fn supported_operators() -> Vec<Token> {
        vec![
            Token::Plus,
            Token::Minus,
            Token::Div,
            Token::Mul,
            Token::Eq,
            Token::Neq,
            Token::Gt,
            Token::GtEq,
            Token::Lt,
            Token::LtEq,
            Token::Keyword(Keyword::And),
            Token::Keyword(Keyword::Or),
        ]
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
                r#where: None,
                order_by: vec![]
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
                r#where: None,
                order_by: vec![]
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
                    right: Box::new(Expression::Value(Value::Number(100)))
                }),
                order_by: vec![]
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
                        right: Box::new(Expression::Value(Value::Number(100))),
                    }
                ],
                from: "products".into(),
                r#where: Some(Expression::BinaryOperation {
                    left: Box::new(Expression::BinaryOperation {
                        left: Box::new(Expression::BinaryOperation {
                            left: Box::new(Expression::Value(Value::Number(100))),
                            operator: BinaryOperator::LtEq,
                            right: Box::new(Expression::Identifier("price".into())),
                        }),
                        operator: BinaryOperator::And,
                        right: Box::new(Expression::BinaryOperation {
                            left: Box::new(Expression::Identifier("price".into())),
                            operator: BinaryOperator::Lt,
                            right: Box::new(Expression::Value(Value::Number(1000))),
                        }),
                    }),
                    operator: BinaryOperator::Or,
                    right: Box::new(Expression::BinaryOperation {
                        left: Box::new(Expression::Identifier("discount".into())),
                        operator: BinaryOperator::Lt,
                        right: Box::new(Expression::BinaryOperation {
                            left: Box::new(Expression::Value(Value::Number(10))),
                            operator: BinaryOperator::Plus,
                            right: Box::new(Expression::BinaryOperation {
                                left: Box::new(Expression::Value(Value::Number(2))),
                                operator: BinaryOperator::Mul,
                                right: Box::new(Expression::Value(Value::Number(20))),
                            })
                        }),
                    })
                }),
                order_by: vec![],
            })
        )
    }

    #[test]
    fn parse_select_order_by() {
        let sql = "SELECT name, email FROM users ORDER BY email;";

        assert_eq!(
            Parser::new(sql).parse_statement(),
            Ok(Statement::Select {
                columns: vec![
                    Expression::Identifier("name".into()),
                    Expression::Identifier("email".into())
                ],
                from: "users".into(),
                r#where: None,
                order_by: vec!["email".into()]
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
                    right: Box::new(Expression::Value(Value::Number(1))),
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
                            right: Box::new(Expression::Value(Value::Number(10))),
                        }),
                    },
                    Expression::BinaryOperation {
                        left: Box::new(Expression::Identifier("discount".into())),
                        operator: BinaryOperator::Eq,
                        right: Box::new(Expression::Value(Value::Number(15))),
                    },
                    Expression::BinaryOperation {
                        left: Box::new(Expression::Identifier("stock".into())),
                        operator: BinaryOperator::Eq,
                        right: Box::new(Expression::Value(Value::Number(10))),
                    }
                ],
                r#where: Some(Expression::BinaryOperation {
                    left: Box::new(Expression::Identifier("price".into())),
                    operator: BinaryOperator::Gt,
                    right: Box::new(Expression::Value(Value::Number(100))),
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
                    right: Box::new(Expression::Value(Value::Number(5000))),
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
                    Expression::Value(Value::Number(1)),
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
                        right: Box::new(Expression::Value(Value::Number(1))),
                    }],
                    r#where: None,
                },
                Statement::Select {
                    columns: vec![Expression::Wildcard],
                    from: "products".into(),
                    r#where: None,
                    order_by: vec![],
                }
            ])
        )
    }

    #[test]
    fn arithmetic_operator_precedence() {
        let expr = "price * discount / 100 < 10 + 20 * 30";

        assert_eq!(
            Parser::new(expr).parse_expression(),
            Ok(Expression::BinaryOperation {
                left: Box::new(Expression::BinaryOperation {
                    left: Box::new(Expression::BinaryOperation {
                        left: Box::new(Expression::Identifier("price".into())),
                        operator: BinaryOperator::Mul,
                        right: Box::new(Expression::Identifier("discount".into())),
                    }),
                    operator: BinaryOperator::Div,
                    right: Box::new(Expression::Value(Value::Number(100))),
                }),
                operator: BinaryOperator::Lt,
                right: Box::new(Expression::BinaryOperation {
                    left: Box::new(Expression::Value(Value::Number(10))),
                    operator: BinaryOperator::Plus,
                    right: Box::new(Expression::BinaryOperation {
                        left: Box::new(Expression::Value(Value::Number(20))),
                        operator: BinaryOperator::Mul,
                        right: Box::new(Expression::Value(Value::Number(30))),
                    })
                })
            })
        )
    }

    #[test]
    fn nested_arithmetic_precedence() {
        let expr = "price * discount >= 10 - (20 + 50) / (2 * (4 + (1 - 1)))";

        assert_eq!(
            Parser::new(expr).parse_expression(),
            Ok(Expression::BinaryOperation {
                left: Box::new(Expression::BinaryOperation {
                    left: Box::new(Expression::Identifier("price".into())),
                    operator: BinaryOperator::Mul,
                    right: Box::new(Expression::Identifier("discount".into())),
                }),
                operator: BinaryOperator::GtEq,
                right: Box::new(Expression::BinaryOperation {
                    left: Box::new(Expression::Value(Value::Number(10))),
                    operator: BinaryOperator::Minus,
                    right: Box::new(Expression::BinaryOperation {
                        left: Box::new(Expression::BinaryOperation {
                            left: Box::new(Expression::Value(Value::Number(20))),
                            operator: BinaryOperator::Plus,
                            right: Box::new(Expression::Value(Value::Number(50))),
                        }),
                        operator: BinaryOperator::Div,
                        right: Box::new(Expression::BinaryOperation {
                            left: Box::new(Expression::Value(Value::Number(2))),
                            operator: BinaryOperator::Mul,
                            right: Box::new(Expression::BinaryOperation {
                                left: Box::new(Expression::Value(Value::Number(4))),
                                operator: BinaryOperator::Plus,
                                right: Box::new(Expression::BinaryOperation {
                                    left: Box::new(Expression::Value(Value::Number(1))),
                                    operator: BinaryOperator::Minus,
                                    right: Box::new(Expression::Value(Value::Number(1))),
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
            Parser::new(expr).parse_expression(),
            Ok(Expression::BinaryOperation {
                left: Box::new(Expression::BinaryOperation {
                    left: Box::new(Expression::BinaryOperation {
                        left: Box::new(Expression::Value(Value::Number(100))),
                        operator: BinaryOperator::LtEq,
                        right: Box::new(Expression::Identifier("price".into())),
                    }),
                    operator: BinaryOperator::And,
                    right: Box::new(Expression::BinaryOperation {
                        left: Box::new(Expression::Identifier("price".into())),
                        operator: BinaryOperator::LtEq,
                        right: Box::new(Expression::Value(Value::Number(200))),
                    }),
                }),
                operator: BinaryOperator::Or,
                right: Box::new(Expression::BinaryOperation {
                    left: Box::new(Expression::Identifier("price".into())),
                    operator: BinaryOperator::Gt,
                    right: Box::new(Expression::Value(Value::Number(1000))),
                })
            })
        )
    }

    #[test]
    fn unary_arithmetic_operator_precedence() {
        let expr = "-2 * -(2 + 2 * 2)";

        assert_eq!(
            Parser::new(expr).parse_expression(),
            Ok(Expression::BinaryOperation {
                left: Box::new(Expression::UnaryOperation {
                    operator: UnaryOperator::Minus,
                    expr: Box::new(Expression::Value(Value::Number(2)))
                }),
                operator: BinaryOperator::Mul,
                right: Box::new(Expression::UnaryOperation {
                    operator: UnaryOperator::Minus,
                    expr: Box::new(Expression::BinaryOperation {
                        left: Box::new(Expression::Value(Value::Number(2))),
                        operator: BinaryOperator::Plus,
                        right: Box::new(Expression::BinaryOperation {
                            left: Box::new(Expression::Value(Value::Number(2))),
                            operator: BinaryOperator::Mul,
                            right: Box::new(Expression::Value(Value::Number(2))),
                        })
                    })
                })
            })
        )
    }

    #[test]
    fn parse_unterminated_statement() {
        let sql = "SELECT * FROM users";

        assert_eq!(
            Parser::new(sql).parse_statement(),
            Err(ParserError {
                kind: ErrorKind::Expected {
                    expected: Token::SemiColon,
                    found: Token::Eof
                },
                location: Location { line: 1, col: 20 }
            })
        )
    }

    #[test]
    fn parse_partial_select() {
        let sql = "SELECT";

        assert_eq!(
            Parser::new(sql).parse_statement(),
            Err(ParserError {
                kind: ErrorKind::ExpectedOneOf {
                    expected: vec![
                        Token::Identifier(Default::default()),
                        Token::Number(Default::default()),
                        Token::String(Default::default()),
                        Token::Mul,
                        Token::Minus,
                        Token::Plus,
                        Token::LeftParen
                    ],
                    found: Token::Eof
                },
                location: Location { line: 1, col: 7 }
            })
        )
    }

    #[test]
    fn parse_partial_insert() {
        let sql = "INSERT";

        assert_eq!(
            Parser::new(sql).parse_statement(),
            Err(ParserError {
                kind: ErrorKind::Expected {
                    expected: Token::Keyword(Keyword::Into),
                    found: Token::Eof
                },
                location: Location { line: 1, col: 7 }
            })
        )
    }

    #[test]
    fn parse_unexpected_initial_token() {
        let sql = "/ SELECT * FROM users;";

        assert_eq!(
            Parser::new(sql).parse_statement(),
            Err(ParserError {
                kind: ErrorKind::ExpectedOneOf {
                    expected: Parser::tokens_from_keywords(&Parser::supported_statements()),
                    found: Token::Div,
                },
                location: Location { line: 1, col: 1 }
            })
        )
    }

    #[test]
    fn parse_unexpected_initial_keyword() {
        let sql = "VARCHAR * FROM users;";

        assert_eq!(
            Parser::new(sql).parse_statement(),
            Err(ParserError {
                kind: ErrorKind::ExpectedOneOf {
                    expected: Parser::tokens_from_keywords(&Parser::supported_statements()),
                    found: Token::Keyword(Keyword::Varchar),
                },
                location: Location { line: 1, col: 1 }
            }),
        )
    }

    #[test]
    fn parse_unexpected_expression_token() {
        let sql = "SELECT ) FROM table;";

        assert_eq!(
            Parser::new(sql).parse_statement(),
            Err(ParserError {
                kind: ErrorKind::ExpectedOneOf {
                    expected: vec![
                        Token::Identifier(Default::default()),
                        Token::Number(Default::default()),
                        Token::String(Default::default()),
                        Token::Mul,
                        Token::Minus,
                        Token::Plus,
                        Token::LeftParen
                    ],
                    found: Token::RightParen,
                },
                location: Location { line: 1, col: 8 }
            })
        )
    }

    #[test]
    fn expect_keyword() {
        let sql = "SELECT * VALUES users";

        assert_eq!(
            Parser::new(sql).parse_statement(),
            Err(ParserError {
                kind: ErrorKind::Expected {
                    expected: Token::Keyword(Keyword::From),
                    found: Token::Keyword(Keyword::Values)
                },
                location: Location { line: 1, col: 10 }
            })
        )
    }

    #[test]
    fn expect_one_of_keywords() {
        let sql = "DROP VALUES test";

        assert_eq!(
            Parser::new(sql).parse_statement(),
            Err(ParserError {
                kind: ErrorKind::ExpectedOneOf {
                    expected: vec![
                        Token::Keyword(Keyword::Database),
                        Token::Keyword(Keyword::Table),
                    ],
                    found: Token::Keyword(Keyword::Values)
                },
                location: Location { line: 1, col: 6 }
            })
        )
    }

    #[test]
    fn expect_data_type() {
        let sql = "CREATE TABLE test (id INCORRECT);";

        assert_eq!(
            Parser::new(sql).parse_statement(),
            Err(ParserError {
                kind: ErrorKind::ExpectedOneOf {
                    expected: Parser::tokens_from_keywords(&Parser::supported_data_types()),
                    found: Token::Identifier("INCORRECT".into())
                },
                location: Location { line: 1, col: 23 }
            })
        )
    }

    #[test]
    fn expect_identifier() {
        let sql = "INSERT INTO 1 VALUES (2);";

        assert_eq!(
            Parser::new(sql).parse_statement(),
            Err(ParserError {
                kind: ErrorKind::Expected {
                    expected: Token::Identifier(Default::default()),
                    found: Token::Number("1".into())
                },
                location: Location { line: 1, col: 13 }
            })
        )
    }

    #[test]
    fn expect_varchar_length() {
        let sql = "CREATE TABLE test (name VARCHAR(test));";

        assert_eq!(
            Parser::new(sql).parse_statement(),
            Err(ParserError {
                kind: ErrorKind::Expected {
                    expected: Token::Number(Default::default()),
                    found: Token::Identifier("test".into())
                },
                location: Location { line: 1, col: 33 }
            })
        )
    }

    #[test]
    fn required_parenthesis() {
        let sql = "INSERT INTO test column VALUES 2;";

        assert_eq!(
            Parser::new(sql).parse_statement(),
            Err(ParserError {
                kind: ErrorKind::Expected {
                    expected: Token::LeftParen,
                    found: Token::Identifier("column".into())
                },
                location: Location { line: 1, col: 18 }
            })
        )
    }
}
