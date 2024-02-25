use std::fmt::{self, Display, Write};

/// SQL tokens.
#[derive(PartialEq, Debug)]
pub(crate) enum Token {
    Keyword(Keyword),
    Identifier(String),
    Whitespace(Whitespace),
    String(String),
    Number(String),
    Eq,
    Neq,
    Lt,
    Gt,
    LtEq,
    GtEq,
    Mul,
    Div,
    Plus,
    Minus,
    LeftParen,
    RightParen,
    Comma,
    SemiColon,
    /// Not a real token, used to mark the end of a token stream.
    Eof,
}

/// SQL keywords.
#[derive(PartialEq, Debug, Clone, Copy)]
pub(crate) enum Keyword {
    Select,
    Create,
    Update,
    Delete,
    Insert,
    Into,
    Values,
    Set,
    Drop,
    From,
    Where,
    And,
    Or,
    Primary,
    Key,
    Unique,
    Table,
    Database,
    Int,
    BigInt,
    Unsigned,
    Varchar,
    Bool,
    True,
    False,
    Order,
    By,
    Index,
    On,
    /// Not a keyword, used for convenience. See [`super::tokenizer::Tokenizer`].
    None,
}

/// Separators between keywords, identifiers, operators, etc.
#[derive(PartialEq, Debug)]
pub(crate) enum Whitespace {
    Space,
    Tab,
    Newline,
}

impl Token {
    /// Returns `true` if the given char can be used for identifiers or
    /// keywords.
    pub(super) fn is_part_of_ident_or_keyword(chr: &char) -> bool {
        chr.is_ascii_lowercase() || chr.is_ascii_uppercase() || chr.is_ascii_digit() || *chr == '_'
    }
}

impl From<&Keyword> for Token {
    fn from(keyword: &Keyword) -> Self {
        Token::Keyword(*keyword)
    }
}

impl Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Eof => f.write_str("EOF"),
            Self::Whitespace(whitespace) => write!(f, "{whitespace}"),
            Self::Keyword(keyword) => write!(f, "{keyword}"),
            Self::Identifier(identifier) => f.write_str(identifier),
            Self::String(string) => write!(f, "\"{string}\""),
            Self::Number(number) => write!(f, "{number}"),
            Self::Eq => f.write_str("="),
            Self::Neq => f.write_str("!="),
            Self::Lt => f.write_str("<"),
            Self::Gt => f.write_str(">"),
            Self::LtEq => f.write_str("<="),
            Self::GtEq => f.write_str(">="),
            Self::Mul => f.write_str("*"),
            Self::Div => f.write_str("/"),
            Self::Plus => f.write_str("+"),
            Self::Minus => f.write_str("-"),
            Self::LeftParen => f.write_str("("),
            Self::RightParen => f.write_str(")"),
            Self::Comma => f.write_str(","),
            Self::SemiColon => f.write_str(";"),
        }
    }
}

impl Display for Keyword {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Select => "SELECT",
            Self::Create => "CREATE",
            Self::Update => "UPDATE",
            Self::Delete => "DELETE",
            Self::Insert => "INSERT",
            Self::Into => "INTO",
            Self::Values => "VALUES",
            Self::Set => "SET",
            Self::Drop => "DROP",
            Self::From => "FROM",
            Self::Where => "WHERE",
            Self::And => "AND",
            Self::Or => "OR",
            Self::Primary => "PRIMARY",
            Self::Key => "KEY",
            Self::Unique => "UNIQUE",
            Self::Table => "TABLE",
            Self::Database => "DATABASE",
            Self::Int => "INT",
            Self::BigInt => "BIGINT",
            Self::Unsigned => "UNSIGNED",
            Self::Varchar => "VARCHAR",
            Self::Bool => "BOOL",
            Self::True => "TRUE",
            Self::False => "FALSE",
            Self::Order => "ORDER",
            Self::By => "BY",
            Self::Index => "INDEX",
            Self::On => "ON",
            Self::None => "_",
        })
    }
}

impl Display for Whitespace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_char(match self {
            Self::Space => ' ',
            Self::Tab => '\t',
            Self::Newline => '\n',
        })
    }
}
