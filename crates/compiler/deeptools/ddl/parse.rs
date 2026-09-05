//! Hand-written tokenizer + recursive-descent parser for the `deeptools` DDL
//! text dialect (see `ast.rs` for the produced typed tree). No external
//! parsing crates are used.

use crate::ast::{AttrValue, Module, Operand, Operation};

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Percent(String),
    Ident(String),
    Str(String),
    Int(i64),
    Float(f64),
    True,
    False,
    Else,
    Hash,
    Colon,
    Comma,
    Equals,
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    Eof,
}

pub type ParseResult<T> = Result<T, String>;

/// Operands parsed from a `(...)`-delimited operand list, plus any
/// `keyword=value` style entries (e.g. `ddl.padded_dimension(primary=%x, ...)`)
/// found interspersed with plain operand refs, folded into attrs.
type OperandListResult = (Vec<Operand>, Vec<(String, AttrValue)>);

fn tokenize(src: &str) -> ParseResult<Vec<Tok>> {
    let bytes: Vec<char> = src.chars().collect();
    let mut i = 0usize;
    let n = bytes.len();
    let mut toks = Vec::new();

    while i < n {
        let c = bytes[i];
        if c.is_whitespace() {
            i += 1;
            continue;
        }
        if c == '/' && i + 1 < n && bytes[i + 1] == '/' {
            while i < n && bytes[i] != '\n' {
                i += 1;
            }
            continue;
        }
        match c {
            '(' => {
                toks.push(Tok::LParen);
                i += 1;
            }
            ')' => {
                toks.push(Tok::RParen);
                i += 1;
            }
            '{' => {
                toks.push(Tok::LBrace);
                i += 1;
            }
            '}' => {
                toks.push(Tok::RBrace);
                i += 1;
            }
            '[' => {
                toks.push(Tok::LBracket);
                i += 1;
            }
            ']' => {
                toks.push(Tok::RBracket);
                i += 1;
            }
            ',' => {
                toks.push(Tok::Comma);
                i += 1;
            }
            '=' => {
                toks.push(Tok::Equals);
                i += 1;
            }
            '#' => {
                toks.push(Tok::Hash);
                i += 1;
            }
            ':' => {
                toks.push(Tok::Colon);
                i += 1;
            }
            '%' => {
                let start = i + 1;
                let mut j = start;
                while j < n && (bytes[j].is_alphanumeric() || bytes[j] == '_') {
                    j += 1;
                }
                if j == start {
                    return Err(format!("empty %-ref at byte offset {i}"));
                }
                toks.push(Tok::Percent(bytes[start..j].iter().collect()));
                i = j;
            }
            '"' => {
                let mut j = i + 1;
                let mut s = String::new();
                while j < n && bytes[j] != '"' {
                    s.push(bytes[j]);
                    j += 1;
                }
                if j >= n {
                    return Err("unterminated string literal".to_string());
                }
                toks.push(Tok::Str(s));
                i = j + 1;
            }
            _ if c.is_ascii_digit() || (c == '-' && i + 1 < n && bytes[i + 1].is_ascii_digit()) => {
                let start = i;
                let mut j = i;
                if bytes[j] == '-' {
                    j += 1;
                }
                if j + 1 < n && bytes[j] == '0' && (bytes[j + 1] == 'x' || bytes[j + 1] == 'X') {
                    j += 2;
                    let hex_start = j;
                    while j < n && bytes[j].is_ascii_hexdigit() {
                        j += 1;
                    }
                    let hex: String = bytes[hex_start..j].iter().collect();
                    let val = u64::from_str_radix(&hex, 16)
                        .map_err(|e| format!("bad hex literal: {e}"))?
                        as i64;
                    toks.push(Tok::Int(val));
                    i = j;
                } else {
                    while j < n && bytes[j].is_ascii_digit() {
                        j += 1;
                    }
                    let mut is_float = false;
                    if j < n && bytes[j] == '.' && j + 1 < n && bytes[j + 1].is_ascii_digit() {
                        is_float = true;
                        j += 1;
                        while j < n && bytes[j].is_ascii_digit() {
                            j += 1;
                        }
                    }
                    let text: String = bytes[start..j].iter().collect();
                    if is_float {
                        let v: f64 = text
                            .parse()
                            .map_err(|e| format!("bad float literal {text}: {e}"))?;
                        toks.push(Tok::Float(v));
                    } else {
                        let v: i64 = text
                            .parse()
                            .map_err(|e| format!("bad int literal {text}: {e}"))?;
                        toks.push(Tok::Int(v));
                    }
                    i = j;
                }
            }
            _ if c.is_alphabetic() || c == '_' => {
                let start = i;
                let mut j = i;
                while j < n && (bytes[j].is_alphanumeric() || bytes[j] == '_' || bytes[j] == '.') {
                    j += 1;
                }
                let text: String = bytes[start..j].iter().collect();
                match text.as_str() {
                    "true" => toks.push(Tok::True),
                    "false" => toks.push(Tok::False),
                    "else" => toks.push(Tok::Else),
                    _ => toks.push(Tok::Ident(text)),
                }
                i = j;
            }
            other => return Err(format!("unexpected character {other:?} at byte offset {i}")),
        }
    }
    toks.push(Tok::Eof);
    Ok(toks)
}

struct Parser {
    toks: Vec<Tok>,
    pos: usize,
}

impl Parser {
    fn peek(&self) -> &Tok {
        &self.toks[self.pos]
    }

    fn bump(&mut self) -> Tok {
        let t = self.toks[self.pos].clone();
        if self.pos + 1 < self.toks.len() {
            self.pos += 1;
        }
        t
    }

    fn expect(&mut self, want: &Tok) -> ParseResult<()> {
        if self.peek() == want {
            self.bump();
            Ok(())
        } else {
            Err(format!(
                "expected {want:?}, found {:?} at token {}",
                self.peek(),
                self.pos
            ))
        }
    }

    fn expect_ident(&mut self) -> ParseResult<String> {
        match self.bump() {
            Tok::Ident(s) => Ok(s),
            other => Err(format!("expected identifier, found {other:?}")),
        }
    }

    /// Parses an attribute-dict key, which is normally a bareword identifier
    /// but may also be a quoted string (e.g. `params={"in0_unroll"="lxlu"}`).
    fn parse_attr_key(&mut self) -> ParseResult<String> {
        match self.bump() {
            Tok::Ident(s) => Ok(s),
            Tok::Str(s) => Ok(s),
            other => Err(format!("expected attribute key, found {other:?}")),
        }
    }

    /// Parses a `%name` or `%name#N` reference, returning its literal text
    /// (e.g. `"%outer_dim#0"`).
    fn parse_ref(&mut self) -> ParseResult<String> {
        let name = match self.bump() {
            Tok::Percent(s) => s,
            other => return Err(format!("expected %-ref, found {other:?}")),
        };
        let mut text = format!("%{name}");
        if *self.peek() == Tok::Hash {
            self.bump();
            match self.bump() {
                Tok::Int(n) => text.push_str(&format!("#{n}")),
                other => return Err(format!("expected index after '#', found {other:?}")),
            }
        }
        Ok(text)
    }

    /// Parses the LHS result-name list of an operation, e.g. `%a, %b:5`,
    /// stopping right before `=`.
    fn parse_result_names(&mut self) -> ParseResult<Vec<String>> {
        let mut names = Vec::new();
        loop {
            let name = match self.bump() {
                Tok::Percent(s) => s,
                other => return Err(format!("expected %-ref in result list, found {other:?}")),
            };
            let mut text = format!("%{name}");
            if *self.peek() == Tok::Colon {
                self.bump();
                match self.bump() {
                    Tok::Int(n) => text.push_str(&format!(":{n}")),
                    other => return Err(format!("expected count after ':', found {other:?}")),
                }
            }
            names.push(text);
            if *self.peek() == Tok::Comma {
                self.bump();
                continue;
            }
            break;
        }
        Ok(names)
    }

    /// Parses a single attribute value (string/bool/number/typed-int/list/ref).
    fn parse_attr_value(&mut self) -> ParseResult<AttrValue> {
        match self.peek().clone() {
            Tok::Str(s) => {
                self.bump();
                Ok(AttrValue::String(s))
            }
            Tok::True => {
                self.bump();
                Ok(AttrValue::Bool(true))
            }
            Tok::False => {
                self.bump();
                Ok(AttrValue::Bool(false))
            }
            Tok::Float(f) => {
                self.bump();
                Ok(AttrValue::Float(f))
            }
            Tok::Int(n) => {
                self.bump();
                if *self.peek() == Tok::Colon {
                    self.bump();
                    let ty = self.expect_ident()?;
                    Ok(AttrValue::TypedInt { value: n, ty })
                } else {
                    Ok(AttrValue::Int(n))
                }
            }
            Tok::LBracket => {
                self.bump();
                let mut items = Vec::new();
                if *self.peek() != Tok::RBracket {
                    loop {
                        items.push(self.parse_attr_value()?);
                        if *self.peek() == Tok::Comma {
                            self.bump();
                            continue;
                        }
                        break;
                    }
                }
                self.expect(&Tok::RBracket)?;
                Ok(AttrValue::List(items))
            }
            Tok::Percent(_) => {
                let r = self.parse_ref()?;
                Ok(AttrValue::Ref(r))
            }
            Tok::LBrace => {
                let dict = self.parse_attr_dict()?;
                Ok(AttrValue::Dict(dict))
            }
            other => Err(format!("unexpected token in attribute value: {other:?}")),
        }
    }

    /// Parses a `{key=value, ...}` dict; assumes the opening `{` has NOT yet
    /// been consumed. Keys are normally barewords but may also be quoted
    /// strings (e.g. `params={"in0_unroll"="lxlu"}`).
    fn parse_attr_dict(&mut self) -> ParseResult<Vec<(String, AttrValue)>> {
        self.expect(&Tok::LBrace)?;
        let mut attrs = Vec::new();
        if *self.peek() != Tok::RBrace {
            loop {
                let key = self.parse_attr_key()?;
                self.expect(&Tok::Equals)?;
                let value = self.parse_attr_value()?;
                attrs.push((key, value));
                if *self.peek() == Tok::Comma {
                    self.bump();
                    continue;
                }
                break;
            }
        }
        self.expect(&Tok::RBrace)?;
        Ok(attrs)
    }

    /// Parses the parenthesized operand/keyword-arg list of an operation.
    /// Returns (positional operands, keyword attrs found inline). Assumes
    /// the opening `(` has already been consumed.
    fn parse_operand_list(&mut self) -> ParseResult<OperandListResult> {
        let mut operands = Vec::new();
        let mut kw_attrs = Vec::new();
        if *self.peek() != Tok::RParen {
            loop {
                // Keyword form: `ident = value` (only `ddl.padded_dimension`
                // uses this in the vendored templates).
                if let Tok::Ident(_) = self.peek() {
                    let save = self.pos;
                    let key = self.expect_ident()?;
                    if *self.peek() == Tok::Equals {
                        self.bump();
                        let value = self.parse_attr_value()?;
                        kw_attrs.push((key, value));
                        if *self.peek() == Tok::Comma {
                            self.bump();
                            continue;
                        }
                        break;
                    } else {
                        self.pos = save;
                    }
                }
                match self.peek() {
                    Tok::LBracket => {
                        self.bump();
                        let mut refs = Vec::new();
                        if *self.peek() != Tok::RBracket {
                            loop {
                                refs.push(self.parse_ref()?);
                                if *self.peek() == Tok::Comma {
                                    self.bump();
                                    continue;
                                }
                                break;
                            }
                        }
                        self.expect(&Tok::RBracket)?;
                        operands.push(Operand::RefList(refs));
                    }
                    Tok::Percent(_) => {
                        operands.push(Operand::Ref(self.parse_ref()?));
                    }
                    other => return Err(format!("unexpected token in operand list: {other:?}")),
                }
                if *self.peek() == Tok::Comma {
                    self.bump();
                    continue;
                }
                break;
            }
        }
        self.expect(&Tok::RParen)?;
        Ok((operands, kw_attrs))
    }

    fn parse_result_types(&mut self) -> ParseResult<Vec<String>> {
        let mut types = Vec::new();
        loop {
            types.push(self.expect_ident()?);
            if *self.peek() == Tok::Comma {
                self.bump();
                continue;
            }
            break;
        }
        Ok(types)
    }

    /// Parses a `{ ... }` body block of nested operations. Assumes the
    /// opening `{` has NOT yet been consumed.
    fn parse_body(&mut self) -> ParseResult<Vec<Operation>> {
        self.expect(&Tok::LBrace)?;
        let mut ops = Vec::new();
        while *self.peek() != Tok::RBrace {
            ops.push(self.parse_operation()?);
        }
        self.expect(&Tok::RBrace)?;
        Ok(ops)
    }

    fn parse_operation(&mut self) -> ParseResult<Operation> {
        let results = if let Tok::Percent(_) = self.peek() {
            let save = self.pos;
            let names = self.parse_result_names()?;
            if *self.peek() == Tok::Equals {
                self.bump();
                names
            } else {
                self.pos = save;
                Vec::new()
            }
        } else {
            Vec::new()
        };

        let name = self.expect_ident()?;
        let mut op = Operation::new(name.clone());
        op.results = results;

        match name.as_str() {
            "ddl.dataflow" | "ddl.transformations" => {
                op.body = self.parse_body()?;
            }
            "ddl.if" => {
                self.expect(&Tok::LParen)?;
                let (operands, kw_attrs) = self.parse_operand_list()?;
                op.operands = operands;
                op.attrs = kw_attrs;
                op.body = self.parse_body()?;
                if *self.peek() == Tok::Else {
                    self.bump();
                    op.else_body = Some(self.parse_body()?);
                }
            }
            "ddl.loop" | "ddl.parametric_loop" => {
                self.expect(&Tok::LParen)?;
                let (operands, kw_attrs) = self.parse_operand_list()?;
                op.operands = operands;
                let mut attrs = kw_attrs;
                attrs.extend(self.parse_attr_dict()?);
                op.attrs = attrs;
                op.body = self.parse_body()?;
            }
            _ => {
                let mut attrs = Vec::new();
                if *self.peek() == Tok::LParen {
                    self.bump();
                    let (operands, kw_attrs) = self.parse_operand_list()?;
                    op.operands = operands;
                    attrs.extend(kw_attrs);
                }
                if *self.peek() == Tok::LBrace {
                    attrs.extend(self.parse_attr_dict()?);
                }
                op.attrs = attrs;
                if *self.peek() == Tok::Colon {
                    self.bump();
                    op.result_types = self.parse_result_types()?;
                }
            }
        }
        Ok(op)
    }

    fn parse_module(&mut self) -> ParseResult<Module> {
        let kw = self.expect_ident()?;
        if kw != "module" {
            return Err(format!("expected top-level `module`, found `{kw}`"));
        }
        let body = self.parse_body()?;
        if *self.peek() != Tok::Eof {
            return Err(format!(
                "unexpected trailing tokens after module body, starting at {:?}",
                self.peek()
            ));
        }
        Ok(Module { body })
    }
}

/// Parses one `.ddl` file's full text into a [`Module`].
pub fn parse_module(src: &str) -> ParseResult<Module> {
    let toks = tokenize(src)?;
    let mut p = Parser { toks, pos: 0 };
    p.parse_module()
}

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! ddl_test {
        ($fn_name:ident, $file:literal) => {
            #[test]
            fn $fn_name() {
                let src = include_str!(concat!("../ddl_templates/", $file));
                let result = parse_module(src);
                assert!(
                    result.is_ok(),
                    "failed to parse {}: {:?}",
                    $file,
                    result.err()
                );
                let module = result.unwrap();
                assert!(
                    !module.body.is_empty(),
                    "{} parsed to an empty module",
                    $file
                );
            }
        };
    }

    ddl_test!(parses_argmax, "argmax.ddl");
    ddl_test!(parses_bmm_dd1, "bmm_dd1.ddl");
    ddl_test!(parses_bmm_sen1p5, "bmm_sen1p5.ddl");
    ddl_test!(parses_bmm, "bmm.ddl");
    ddl_test!(parses_broadcast_ops, "broadcast_ops.ddl");
    ddl_test!(parses_convolution2d_dd1, "convolution2d_dd1.ddl");
    ddl_test!(parses_convolution2d_os1, "convolution2d_os1.ddl");
    ddl_test!(parses_convolution2d, "convolution2d.ddl");
    ddl_test!(parses_depthwise_conv_fwd, "depthwise_conv_fwd.ddl");
    ddl_test!(parses_exx2_32, "exx2_32.ddl");
    ddl_test!(parses_gelu_bwd, "gelu_bwd.ddl");
    ddl_test!(
        parses_inter_slice_transpose_with_bottomdatastage,
        "inter_slice_transpose_with_bottomdatastage.ddl"
    );
    ddl_test!(parses_inter_slice_transpose, "inter_slice_transpose.ddl");
    ddl_test!(parses_layernormbackwardnorm, "layernormbackwardnorm.ddl");
    ddl_test!(parses_layernormnorm_fp32, "layernormnorm_fp32.ddl");
    ddl_test!(parses_layernormnorm, "layernormnorm.ddl");
    ddl_test!(parses_layernormscale_32, "layernormscale_32.ddl");
    ddl_test!(parses_lstmactp2, "lstmactp2.ddl");
    ddl_test!(parses_pooling, "pooling.ddl");
    ddl_test!(parses_quant_scale_per_token, "quant_scale_per_token.ddl");
    ddl_test!(
        parses_quantization_double_pad,
        "quantization_double_pad.ddl"
    );
    ddl_test!(parses_quantization_no_pad, "quantization_no_pad.ddl");
    ddl_test!(
        parses_quantization_single_pad_v2,
        "quantization_single_pad_v2.ddl"
    );
    ddl_test!(
        parses_quantization_single_pad,
        "quantization_single_pad.ddl"
    );
    ddl_test!(parses_restickify_sen1p5, "restickify_sen1p5.ddl");
    ddl_test!(parses_restickify, "restickify.ddl");
    ddl_test!(parses_rope, "rope.ddl");
    ddl_test!(parses_summeanmaxexx2_fp32, "summeanmaxexx2_fp32.ddl");
    ddl_test!(parses_summeanmaxexx2, "summeanmaxexx2.ddl");
    ddl_test!(parses_topk, "topk.ddl");
    ddl_test!(parses_unary_parallel, "unary_parallel.ddl");
    ddl_test!(parses_unary_pipeline, "unary_pipeline.ddl");
}
