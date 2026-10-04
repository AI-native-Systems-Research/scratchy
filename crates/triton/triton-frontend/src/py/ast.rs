//! This crate's own Python AST: exactly the node kinds the measured census reaches.
//!
//! THE CENSUS, re-runnable, is the definition of this type:
//!
//! ```text
//! python3 ../triton-superdsc/tools/ast_census.py \
//!     ../../test/fixtures/{vector_add,mul,bias_add_f32,swiglu_mlp,attention_flash}.py
//! -> 33 distinct AST node types, 33 distinct call targets
//! ```
//!
//! Of those 33 node types, several are not data: `Load`/`Store` are expression contexts
//! (folded into the place a name appears), `arguments`/`arg`/`keyword` are parameter and
//! call syntax (folded into [`FunctionDef`] and [`Call`]), and the operator tokens
//! (`Add`, `Mult`, `Sub`, `Div`, `FloorDiv`, `Mod`, `BitAnd`, `USub`, `Eq`, `LtE`) are the
//! [`BinOpKind`] / [`UnaryOpKind`] / [`CmpKind`] enums. What remains is the node set
//! below, and it is closed: adding a fixture that needs more is a change to this file plus
//! the census, in that order.

/// A source position, 1-based line and 0-based column, matching CPython's `ast` and
/// therefore matching the `loc(...)` in the goldens.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Pos {
    pub line: u32,
    pub col: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinOpKind {
    Add,
    Sub,
    Mult,
    Div,
    FloorDiv,
    Mod,
    BitAnd,
    BitOr,
    BitXor,
    LShift,
    RShift,
    Pow,
    MatMult,
}

impl BinOpKind {
    /// The Python dunder Triton dispatches to, from
    /// `code_generator.py:_method_name_for_bin_op`.
    pub fn dunder(self) -> &'static str {
        match self {
            BinOpKind::Add => "__add__",
            BinOpKind::Sub => "__sub__",
            BinOpKind::Mult => "__mul__",
            BinOpKind::Div => "__truediv__",
            BinOpKind::FloorDiv => "__floordiv__",
            BinOpKind::Mod => "__mod__",
            BinOpKind::BitAnd => "__and__",
            BinOpKind::BitOr => "__or__",
            BinOpKind::BitXor => "__xor__",
            BinOpKind::LShift => "__lshift__",
            BinOpKind::RShift => "__rshift__",
            BinOpKind::Pow => "__pow__",
            BinOpKind::MatMult => "__matmul__",
        }
    }

    pub fn py_symbol(self) -> &'static str {
        match self {
            BinOpKind::Add => "+",
            BinOpKind::Sub => "-",
            BinOpKind::Mult => "*",
            BinOpKind::Div => "/",
            BinOpKind::FloorDiv => "//",
            BinOpKind::Mod => "%",
            BinOpKind::BitAnd => "&",
            BinOpKind::BitOr => "|",
            BinOpKind::BitXor => "^",
            BinOpKind::LShift => "<<",
            BinOpKind::RShift => ">>",
            BinOpKind::Pow => "**",
            BinOpKind::MatMult => "@",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnaryOpKind {
    USub,
    UAdd,
    Not,
    Invert,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CmpKind {
    Eq,
    NotEq,
    Lt,
    LtE,
    Gt,
    GtE,
}

impl CmpKind {
    pub fn dunder(self) -> &'static str {
        match self {
            CmpKind::Eq => "__eq__",
            CmpKind::NotEq => "__ne__",
            CmpKind::Lt => "__lt__",
            CmpKind::LtE => "__le__",
            CmpKind::Gt => "__gt__",
            CmpKind::GtE => "__ge__",
        }
    }
}

/// A literal. Python's `ast.Constant`.
#[derive(Clone, Debug, PartialEq)]
pub enum Literal {
    Int(i128),
    Float(f64),
    Bool(bool),
    Str(String),
    None,
}

#[derive(Clone, Debug)]
pub enum Expr {
    Name {
        id: String,
        pos: Pos,
    },
    Constant {
        value: Literal,
        pos: Pos,
    },
    /// `a.b`, including the `tl.x` / `tl.math.x` chains and `desc.load`.
    Attribute {
        value: Box<Expr>,
        attr: String,
        pos: Pos,
    },
    Call {
        func: Box<Expr>,
        args: Vec<Expr>,
        /// `keyword` nodes: `(name, value)`. `**kwargs` is outside the census.
        keywords: Vec<(String, Expr)>,
        pos: Pos,
    },
    BinOp {
        left: Box<Expr>,
        op: BinOpKind,
        right: Box<Expr>,
        pos: Pos,
    },
    UnaryOp {
        op: UnaryOpKind,
        operand: Box<Expr>,
        pos: Pos,
    },
    /// Only single comparisons; `a < b < c` is refused by name.
    Compare {
        left: Box<Expr>,
        op: CmpKind,
        right: Box<Expr>,
        pos: Pos,
    },
    Subscript {
        value: Box<Expr>,
        index: Box<Expr>,
        pos: Pos,
    },
    /// `x[:, None]` -- a slice appearing inside a subscript.
    Slice {
        lower: Option<Box<Expr>>,
        upper: Option<Box<Expr>>,
        step: Option<Box<Expr>>,
        pos: Pos,
    },
    List {
        elts: Vec<Expr>,
        pos: Pos,
    },
    Tuple {
        elts: Vec<Expr>,
        pos: Pos,
    },
}

impl Expr {
    pub fn pos(&self) -> Pos {
        match self {
            Expr::Name { pos, .. }
            | Expr::Constant { pos, .. }
            | Expr::Attribute { pos, .. }
            | Expr::Call { pos, .. }
            | Expr::BinOp { pos, .. }
            | Expr::UnaryOp { pos, .. }
            | Expr::Compare { pos, .. }
            | Expr::Subscript { pos, .. }
            | Expr::Slice { pos, .. }
            | Expr::List { pos, .. }
            | Expr::Tuple { pos, .. } => *pos,
        }
    }

    /// The dotted name of an attribute/name chain: `tl.math.exp2` -> `"tl.math.exp2"`.
    /// `None` if the chain is not purely names and attributes (e.g. `f(x).y`).
    pub fn dotted(&self) -> Option<String> {
        match self {
            Expr::Name { id, .. } => Some(id.clone()),
            Expr::Attribute { value, attr, .. } => Some(format!("{}.{}", value.dotted()?, attr)),
            _ => None,
        }
    }

    /// The node-type name, for error messages that must NAME the construct.
    pub fn kind_name(&self) -> &'static str {
        match self {
            Expr::Name { .. } => "Name",
            Expr::Constant { .. } => "Constant",
            Expr::Attribute { .. } => "Attribute",
            Expr::Call { .. } => "Call",
            Expr::BinOp { .. } => "BinOp",
            Expr::UnaryOp { .. } => "UnaryOp",
            Expr::Compare { .. } => "Compare",
            Expr::Subscript { .. } => "Subscript",
            Expr::Slice { .. } => "Slice",
            Expr::List { .. } => "List",
            Expr::Tuple { .. } => "Tuple",
        }
    }
}

/// An assignment target. Only the two forms the census reaches.
///
/// Named `AssignTarget` rather than `Target` so it cannot be confused with
/// [`crate::target::Target`], which is the compilation TARGET's policy switches.
#[derive(Clone, Debug)]
pub enum AssignTarget {
    Name { id: String, pos: Pos },
    Tuple { elts: Vec<AssignTarget>, pos: Pos },
}

impl AssignTarget {
    pub fn pos(&self) -> Pos {
        match self {
            AssignTarget::Name { pos, .. } | AssignTarget::Tuple { pos, .. } => *pos,
        }
    }
}

#[derive(Clone, Debug)]
pub enum Stmt {
    Assign {
        target: AssignTarget,
        value: Expr,
        pos: Pos,
    },
    /// `x: tl.constexpr = e`. The annotation matters: `constexpr` changes how the value is
    /// bound (`code_generator.py:visit_AnnAssign`).
    AnnAssign {
        target: AssignTarget,
        annotation: Expr,
        value: Option<Expr>,
        pos: Pos,
    },
    AugAssign {
        target: AssignTarget,
        op: BinOpKind,
        value: Expr,
        pos: Pos,
    },
    /// `for <target> in <iter>: body`. `iter` must be a call to `range`, `tl.range` or
    /// `tl.static_range` -- anything else is refused by name, matching
    /// `visit_For`'s own `RuntimeError`.
    For {
        target: String,
        iter: Expr,
        body: Vec<Stmt>,
        orelse: Vec<Stmt>,
        pos: Pos,
    },
    If {
        test: Expr,
        body: Vec<Stmt>,
        orelse: Vec<Stmt>,
        pos: Pos,
    },
    /// A bare expression statement, e.g. `desc.store(...)` or `tl.static_assert(...)`.
    Expr {
        value: Expr,
        pos: Pos,
    },
    Return {
        value: Option<Expr>,
        pos: Pos,
    },
    Pass {
        pos: Pos,
    },
}

impl Stmt {
    pub fn pos(&self) -> Pos {
        match self {
            Stmt::Assign { pos, .. }
            | Stmt::AnnAssign { pos, .. }
            | Stmt::AugAssign { pos, .. }
            | Stmt::For { pos, .. }
            | Stmt::If { pos, .. }
            | Stmt::Expr { pos, .. }
            | Stmt::Return { pos, .. }
            | Stmt::Pass { pos, .. } => *pos,
        }
    }

    pub fn kind_name(&self) -> &'static str {
        match self {
            Stmt::Assign { .. } => "Assign",
            Stmt::AnnAssign { .. } => "AnnAssign",
            Stmt::AugAssign { .. } => "AugAssign",
            Stmt::For { .. } => "For",
            Stmt::If { .. } => "If",
            Stmt::Expr { .. } => "Expr",
            Stmt::Return { .. } => "Return",
            Stmt::Pass { .. } => "Pass",
        }
    }
}

/// A formal parameter.
#[derive(Clone, Debug)]
pub struct Param {
    pub name: String,
    /// The dotted annotation, if any: `tl.constexpr` for a constexpr parameter.
    pub annotation: Option<String>,
    pub pos: Pos,
}

impl Param {
    pub fn is_constexpr(&self) -> bool {
        matches!(self.annotation.as_deref(), Some(a) if a.ends_with("constexpr"))
    }
}

/// A `@triton.jit` function.
#[derive(Clone, Debug)]
pub struct FunctionDef {
    pub name: String,
    pub params: Vec<Param>,
    pub body: Vec<Stmt>,
    /// Whether a `@triton.jit` (or `@jit`) decorator was present. Only jit functions are
    /// lowered; a plain `def` in the module is host code the front end never sees.
    pub is_jit: bool,
    pub pos: Pos,
}

/// A parsed module: just the functions, since nothing else is in scope.
#[derive(Clone, Debug, Default)]
pub struct PyModule {
    pub functions: Vec<FunctionDef>,
    /// Module-level `NAME = <literal>` bindings.
    ///
    /// WHY THESE ARE CAPTURED BUT NOT USABLE AS CONSTANTS: Triton REFUSES a plain module
    /// global read inside a jitted kernel. Measured on `bias_add_f32.py`, whose
    /// `tl.full([BLOCK], BIAS, tl.float32)` reads the module-level `BIAS = 1.0`:
    ///
    /// ```text
    /// NameError: Cannot access global variable BIAS from within @jit'ed function.
    /// Triton kernels can only access global variables that are instanstiated as
    /// constexpr (`x = triton.language.constexpr(42)`).
    /// ```
    ///
    /// They are captured so [`crate::codegen`] can reproduce that refusal WITH THE NAME,
    /// instead of failing with an unhelpful "undefined name".
    pub globals: Vec<(String, Literal)>,
}

impl PyModule {
    pub fn jit_functions(&self) -> impl Iterator<Item = &FunctionDef> {
        self.functions.iter().filter(|f| f.is_jit)
    }

    pub fn function(&self, name: &str) -> Option<&FunctionDef> {
        self.functions.iter().find(|f| f.name == name)
    }
}
