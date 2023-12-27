use std::fmt::{self, Display};

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

impl Display for Token {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Eof => f.write_str("Eof"),
            Self::Whitespace(whitespace) => write!(f, "{whitespace}"),
            Self::Keyword(keyword) => write!(f, "{keyword}"),
            Self::Identifier(identifier) => f.write_str(&identifier),
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

impl Display for Whitespace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Space => f.write_str(" "),
            Self::Tab => f.write_str("\t"),
            Self::Newline => f.write_str("\n"),
        }
    }
}
