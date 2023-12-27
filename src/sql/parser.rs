use core::iter::Peekable;

use super::{
    statement::{Expression, Statement},
    token::{Keyword, Token},
    tokenizer::{self, Tokenizer, TokenizerError},
};

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

pub(super) struct Parser<'i> {
    tokenizer: Peekable<tokenizer::IntoIter<'i>>,
}

impl<'i> Parser<'i> {
    fn new(input: &'i str) -> Self {
        Self {
            tokenizer: Tokenizer::new(input).into_iter().peekable(),
        }
    }

    fn try_parse(&mut self) -> Result<Vec<Statement>, ParserError> {
        let mut statements = Vec::new();

        while let Some(_) = self.tokenizer.peek() {
            statements.push(self.parse_statement()?);
        }

        Ok(statements)
    }

    fn skip_white_spaces(&mut self) {
        while let Some(Ok(Token::Whitespace(_))) = self.tokenizer.peek() {
            self.tokenizer.next();
        }
    }

    fn next_after_whitespaces(&mut self) -> Result<Token, ParserError> {
        let Some(Ok(Token::Whitespace(_))) = self.tokenizer.peek() else {
            return Err(ParserError::new("expected whitespace separator"));
        };

        self.skip_white_spaces();

        match self.tokenizer.next() {
            None => Err(ParserError::new("unepexted EOF")),
            Some(result) => Ok(result?),
        }
    }

    fn expect_keyword_after_whitespaces(&mut self, expected: Keyword) -> Result<(), ParserError> {
        self.next_after_whitespaces().and_then(|token| match token {
            Token::Keyword(keyword) if keyword == expected => Ok(()),

            _ => Err(ParserError::new(format!(
                "unexpected token {token}, expected keyword {expected} instead"
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

    fn consume_optional_keyword(&mut self, optional: Keyword) -> bool {
        self.skip_white_spaces();

        self.tokenizer.peek().is_some_and(|token| match token {
            Ok(Token::Keyword(keyword)) => keyword == &optional,
            _ => false,
        })
    }

    fn parse_statement(&mut self) -> Result<Statement, ParserError> {
        match self.next_after_whitespaces()? {
            Token::Keyword(keyword) => match keyword {
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
                "unexpected first token {unexpected}. Should be a keyword."
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
