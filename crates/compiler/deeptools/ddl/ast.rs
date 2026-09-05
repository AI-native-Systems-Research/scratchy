//! Typed AST for the `deeptools` DDL text dialect (`ddl_templates/*.ddl`).
//!
//! The DDL dialect is a small, custom MLIR-flavored text format used by the
//! IBM `deeptools` C++ compiler to describe how a hardware op template binds
//! dimensions, layouts, tensors, and a per-op dataflow loop nest. This module
//! only models constructs actually observed in the vendored `.ddl` text.

/// A single attribute value in a `{key=value, ...}` dict.
#[derive(Debug, Clone, PartialEq)]
pub enum AttrValue {
    String(String),
    Int(i64),
    Bool(bool),
    Float(f64),
    /// A signed/typed integer literal, e.g. `-1:si64`.
    TypedInt {
        value: i64,
        ty: String,
    },
    List(Vec<AttrValue>),
    /// A nested `{key=value, ...}` dict attribute value, e.g.
    /// `params={"in0_unroll"="lxlu"}` (both keys and values quoted strings).
    Dict(Vec<(String, AttrValue)>),
    /// A bare `%name` SSA-value reference used as an attribute value (e.g.
    /// `primary=%wrdd#0`).
    Ref(String),
}

/// A single operand to an operation.
#[derive(Debug, Clone, PartialEq)]
pub enum Operand {
    /// A plain `%name` or `%name#N` SSA-value reference.
    Ref(String),
    /// A bracketed `[%a, %b, ...]` list of refs (may be empty: `[]`).
    RefList(Vec<String>),
}

/// The result-type annotation trailing an operation (`: index, index, ...`).
pub type ResultTypes = Vec<String>;

/// A single DDL operation, possibly carrying nested regions (bodies of
/// `ddl.if`/`ddl.loop`/`ddl.parametric_loop`/`ddl.dataflow`/
/// `ddl.transformations`).
#[derive(Debug, Clone, PartialEq)]
pub struct Operation {
    /// SSA result names bound by this op (empty for statement-only ops like
    /// `ddl.constraint`).
    pub results: Vec<String>,
    /// Fully-qualified op name, e.g. `"ddl.dimension"`.
    pub name: String,
    /// Positional operands, in source order. Some ops mix bare `Ref`
    /// operands with bracketed `RefList` operands (e.g. `ddl.tensor`).
    pub operands: Vec<Operand>,
    /// The trailing `{key=value, ...}` attribute dictionary, if present.
    pub attrs: Vec<(String, AttrValue)>,
    /// Trailing `: index, index, ...` result-type annotation, if present.
    pub result_types: ResultTypes,
    /// The primary nested body, for `ddl.if`/`ddl.loop`/`ddl.parametric_loop`
    /// /`ddl.dataflow`/`ddl.transformations`. Empty for non-region ops.
    pub body: Vec<Operation>,
    /// The `else` branch body of a `ddl.if`, if present.
    pub else_body: Option<Vec<Operation>>,
}

impl Operation {
    pub fn new(name: String) -> Self {
        Operation {
            results: Vec::new(),
            name,
            operands: Vec::new(),
            attrs: Vec::new(),
            result_types: Vec::new(),
            body: Vec::new(),
            else_body: None,
        }
    }
}

/// A fully parsed `.ddl` file: `module { ... }`.
#[derive(Debug, Clone, PartialEq)]
pub struct Module {
    pub body: Vec<Operation>,
}
