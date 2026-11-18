// SPDX-License-Identifier: Apache-2.0
//! A decoder state's own DSL vocabulary: the op names and the layered state inputs a decoder state
//! contributes from its own crate, so the shared [`OpKind`] and [`ExternKind`] each carry ONE
//! opaque variant ([`OpKind::State`], [`ExternKind::State`]) and never a variant per decoder state.
//! Deleting a decoder state deletes its declaration; nothing shared names it.
//!
//! [`StateTable::assemble`] is the one place the decoder states a build passes become a vocabulary: it
//! rejects a name claimed twice (by the core vocabulary and a decoder state, or by two decoder states), so a
//! lookup never picks between two claims. The decoder states' derived dims assemble there too, into
//! one [`DimTable`] with the core ones.

use std::collections::BTreeMap;
use std::fmt;

use crate::classified::{ExternKind, OpKind};
use crate::classify::ClassifyError;
use crate::dim::{DERIVED_DIMS, DerivedDim};

/// A decoder state's vocabulary, declared in the decoder state's own crate.
pub trait StateVocabulary: Sync {
    fn name(&self) -> StateName;
    fn ops(&self) -> &'static [StateOpDecl];
    fn inputs(&self) -> &'static [StateInputDecl];
    /// The absolute Rust path (`::crate::module`) of the module that holds the signatures and input
    /// constructors: the typed carriers call it from another crate.
    fn signatures(&self) -> SignaturePath;
    /// The derived dims the decoder state adds after the core ones ([`DERIVED_DIMS`]), in the order
    /// they derive: a row reads the core rows and the rows above it here.
    fn derived_dims(&self) -> &'static [DerivedDim] {
        &[]
    }
}

pub use scratchy_tensors::compiler_ids::{
    Arity, InputName, OpName, OutputCount, StateInputId, StateName, StateOpId,
};
use scratchy_tensors::static_name;

static_name!(
    /// A Rust module path (`crate::module`).
    SignaturePath
);

/// A count read off a carrier: the arguments a call passes, or the names an assignment binds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SourceCount(usize);

impl SourceCount {
    /// What a single-name assignment, or a call nested as an argument, binds.
    pub const ONE: Self = Self(1);

    pub(crate) fn of(count: usize) -> Self {
        Self(count)
    }

    pub fn get(self) -> usize {
        self.0
    }

    /// Whether a call passing this many arguments matches `arity`.
    pub fn admitted_by(self, arity: Arity) -> bool {
        usize::from(arity.get()) == self.0
    }

    /// Whether an assignment binding this many names binds every one of `outputs`.
    pub fn binds(self, outputs: OutputCount) -> bool {
        usize::from(outputs.get().get()) == self.0
    }
}

/// One op a decoder state declares.
#[derive(Debug)]
pub struct StateOpDecl {
    pub name: OpName,
    pub arity: Arity,
    pub outputs: OutputCount,
}

/// One layered state input a decoder state declares: a body reads it as `name[layer]`.
#[derive(Debug)]
pub struct StateInputDecl {
    pub name: InputName,
}

/// A state op, as a classified program refers to it: its declaration and the decoder state that made it.
#[derive(Clone, Copy, Debug)]
pub struct StateOp(&'static StateOpDecl, StateName, SignaturePath);

impl StateOp {
    pub fn name(self) -> OpName {
        self.0.name
    }

    pub fn state(self) -> StateName {
        self.1
    }

    /// The module of its decoder state that holds its signature.
    pub fn signatures(self) -> SignaturePath {
        self.2
    }

    pub fn arity(self) -> Arity {
        self.0.arity
    }

    pub fn outputs(self) -> OutputCount {
        self.0.outputs
    }

    /// The identity the shared passes carry in place of the handle.
    pub fn id(self) -> StateOpId {
        StateOpId::new(self.1, self.0.name)
    }
}

impl PartialEq for StateOp {
    fn eq(&self, other: &Self) -> bool {
        (self.1, self.0.name) == (other.1, other.0.name)
    }
}

impl Eq for StateOp {}

impl std::hash::Hash for StateOp {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        (self.1, self.0.name).hash(state);
    }
}

impl fmt::Display for StateOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "`{}` (decoder state `{}`)", self.0.name, self.1)
    }
}

/// A state input, as a classified program refers to it.
#[derive(Clone, Copy, Debug)]
pub struct StateInput(&'static StateInputDecl, StateName, SignaturePath);

impl StateInput {
    pub fn name(self) -> InputName {
        self.0.name
    }

    pub fn state(self) -> StateName {
        self.1
    }

    /// The module of its decoder state that holds its constructor.
    pub fn signatures(self) -> SignaturePath {
        self.2
    }

    /// The identity the shared passes carry in place of the handle.
    pub fn id(self) -> StateInputId {
        StateInputId::new(self.1, self.0.name)
    }
}

impl PartialEq for StateInput {
    fn eq(&self, other: &Self) -> bool {
        (self.1, self.0.name) == (other.1, other.0.name)
    }
}

impl Eq for StateInput {}

impl std::hash::Hash for StateInput {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        (self.1, self.0.name).hash(state);
    }
}

impl fmt::Display for StateInput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "`{}` (decoder state `{}`)", self.0.name, self.1)
    }
}

/// Who claims a name: the compiler's core vocabulary, or a decoder state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Vocabulary {
    Core,
    State(StateName),
}

impl fmt::Display for Vocabulary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Core => f.write_str("the core vocabulary"),
            Self::State(name) => write!(f, "decoder state `{name}`"),
        }
    }
}

/// The decoder states a build passes, assembled into one vocabulary in which every name has one claim,
/// and the derived dims they add to the core ones.
#[derive(Clone, Debug, Default)]
pub struct StateTable {
    ops: BTreeMap<OpName, StateOp>,
    inputs: BTreeMap<InputName, StateInput>,
    dims: DimTable,
}

impl StateTable {
    /// The vocabulary of `decoder_states`: every op, input and derived dim they declare, each claimed
    /// once. A name the core vocabulary already has, or one two decoder states declare, is a
    /// `Duplicate*Name` error naming both claimants; the dims assemble as [`DimTable::assemble`].
    pub fn assemble(decoder_states: &[&dyn StateVocabulary]) -> Result<Self, ClassifyError> {
        let mut table = Self {
            dims: DimTable::assemble(decoder_states)?,
            ..Self::default()
        };
        for state in decoder_states {
            let second = state.name();
            let signatures = state.signatures();
            for decl in state.ops() {
                let name = decl.name;
                let first = match (OpKind::is_core_name(name.get()), table.ops.get(&name)) {
                    (true, _) => Some(Vocabulary::Core),
                    (false, Some(op)) => Some(Vocabulary::State(op.state())),
                    (false, None) => None,
                };
                if let Some(first) = first {
                    return Err(ClassifyError::DuplicateOpName {
                        name,
                        first,
                        second,
                    });
                }
                table.ops.insert(name, StateOp(decl, second, signatures));
            }
            for decl in state.inputs() {
                let name = decl.name;
                let first = match (
                    ExternKind::is_core_name(name.get()),
                    table.inputs.get(&name),
                ) {
                    (true, _) => Some(Vocabulary::Core),
                    (false, Some(input)) => Some(Vocabulary::State(input.state())),
                    (false, None) => None,
                };
                if let Some(first) = first {
                    return Err(ClassifyError::DuplicateInputName {
                        name,
                        first,
                        second,
                    });
                }
                table
                    .inputs
                    .insert(name, StateInput(decl, second, signatures));
            }
        }
        Ok(table)
    }

    /// The state op a body calls by `name`, if a decoder state declares one.
    pub fn op(&self, name: &str) -> Option<StateOp> {
        self.ops.get(name).copied()
    }

    /// The state input a body reads by `name`, if a decoder state declares one.
    pub fn input(&self, name: &str) -> Option<StateInput> {
        self.inputs.get(name).copied()
    }

    /// The derived dims: the core rows, then each decoder state's.
    pub fn dims(&self) -> &DimTable {
        &self.dims
    }
}

static_name!(
    /// The name a derived dim is read by: a config bound no config states.
    DimName
);

/// Every derived dim a build knows: the core rows ([`DERIVED_DIMS`]), then each decoder state's
/// rows in the order the build lists the decoder states and each declares its rows. Assembled once,
/// so every name has one claim and every row reads only rows it may.
#[derive(Clone, Debug)]
pub struct DimTable {
    rows: Vec<(Vocabulary, &'static DerivedDim)>,
}

impl DimTable {
    /// The table of the core rows and `decoder_states`' rows. A name the core rows or an earlier
    /// row already claim is a `DuplicateDimName` error naming both claimants. A row reading a name
    /// the table derives may read only a core row or a row of its own decoder state, above it and of
    /// a stage no later than its own; any other read is a `DimReadsUnavailable` error, since it
    /// would read a value not yet derived, or one deleting another decoder state takes away.
    pub fn assemble(decoder_states: &[&dyn StateVocabulary]) -> Result<Self, ClassifyError> {
        let Self { mut rows } = Self::default();
        for state in decoder_states {
            let second = state.name();
            for d in state.derived_dims() {
                if let Some(&(first, _)) = rows.iter().find(|(_, e)| e.name == d.name) {
                    return Err(ClassifyError::DuplicateDimName {
                        name: DimName::new(d.name),
                        first,
                        second,
                    });
                }
                rows.push((Vocabulary::State(second), d));
            }
        }
        for (i, &(owner, d)) in rows.iter().enumerate() {
            let mut reads = Vec::new();
            d.rule.reads(&mut reads);
            for name in reads {
                let Some(j) = rows.iter().position(|(_, e)| e.name == name) else {
                    continue;
                };
                let (reads_owner, e) = rows[j];
                let readable = j < i
                    && e.stage <= d.stage
                    && (reads_owner == Vocabulary::Core || reads_owner == owner);
                if !readable {
                    return Err(ClassifyError::DimReadsUnavailable {
                        name: DimName::new(d.name),
                        owner,
                        reads: DimName::new(name),
                        reads_owner,
                    });
                }
            }
        }
        Ok(Self { rows })
    }

    /// Every row, the core rows first, then each decoder state's in declaration order.
    pub fn rows(&self) -> impl Iterator<Item = &DerivedDim> {
        self.rows.iter().map(|&(_, d)| d)
    }

    /// The row deriving `name`, if the table has one.
    pub fn find(&self, name: &str) -> Option<&DerivedDim> {
        self.rows().find(|d| d.name == name)
    }
}

/// The core rows alone: the table of a build listing no decoder state.
impl Default for DimTable {
    fn default() -> Self {
        const CORE: &[DerivedDim] = &DERIVED_DIMS;
        Self {
            rows: CORE.iter().map(|d| (Vocabulary::Core, d)).collect(),
        }
    }
}

/// The test decoder state, shared with the passes that read a decoder state's handles.
#[cfg(test)]
pub(crate) mod probe {
    use std::num::NonZeroU8;

    use super::*;
    use crate::dim::{DeriveStage, DerivedDim, DerivedRule};

    pub(crate) const TWO: NonZeroU8 = NonZeroU8::MIN.saturating_add(1);

    static PROBE_OPS: [StateOpDecl; 2] = [
        StateOpDecl {
            name: OpName::new("probe_split"),
            arity: Arity::new(1),
            outputs: OutputCount::new(TWO),
        },
        StateOpDecl {
            name: OpName::new("probe_mix"),
            arity: Arity::new(3),
            outputs: OutputCount::ONE,
        },
    ];
    static PROBE_INPUTS: [StateInputDecl; 1] = [StateInputDecl {
        name: InputName::new("probe_state"),
    }];
    static PROBE_DIMS: [DerivedDim; 1] = [DerivedDim {
        name: "probe_width",
        stage: DeriveStage::Implicit,
        rule: DerivedRule::Add(&[
            DerivedRule::Bound("hidden_size"),
            DerivedRule::Bound("num_attention_heads"),
        ]),
    }];

    /// The test decoder state: a two-output split, a three-argument mix, one layered state input
    /// and one derived dim.
    pub(crate) struct Probe;

    impl StateVocabulary for Probe {
        fn name(&self) -> StateName {
            StateName::new("probe")
        }
        fn ops(&self) -> &'static [StateOpDecl] {
            &PROBE_OPS
        }
        fn inputs(&self) -> &'static [StateInputDecl] {
            &PROBE_INPUTS
        }
        fn signatures(&self) -> SignaturePath {
            SignaturePath::new("::probe::ops")
        }
        fn derived_dims(&self) -> &'static [DerivedDim] {
            &PROBE_DIMS
        }
    }
}

#[cfg(test)]
mod tests {
    use super::probe::{Probe, TWO};
    use super::*;
    use crate::classified::{Expr, ExternKind, OpKind, Prelude, Program, Stmt};
    use crate::classify::{ClassifyError, classify_with};
    use crate::dim::{DeriveStage, DerivedDim, DerivedRule};
    use crate::parse_python::parse_body;

    static SQUATTER_OPS: [StateOpDecl; 1] = [StateOpDecl {
        name: OpName::new("attention"),
        arity: Arity::new(5),
        outputs: OutputCount::ONE,
    }];

    /// A decoder state that claims a core op's name.
    struct Squatter;

    impl StateVocabulary for Squatter {
        fn name(&self) -> StateName {
            StateName::new("squatter")
        }
        fn ops(&self) -> &'static [StateOpDecl] {
            &SQUATTER_OPS
        }
        fn inputs(&self) -> &'static [StateInputDecl] {
            &[]
        }
        fn signatures(&self) -> SignaturePath {
            SignaturePath::new("squatter::signatures")
        }
    }

    static SQRT_OPS: [StateOpDecl; 1] = [StateOpDecl {
        name: OpName::new("sqrt"),
        arity: Arity::new(1),
        outputs: OutputCount::ONE,
    }];
    static MUL_OPS: [StateOpDecl; 1] = [StateOpDecl {
        name: OpName::new("mul"),
        arity: Arity::new(2),
        outputs: OutputCount::ONE,
    }];

    /// A decoder state whose one op is `ops`, named `name`.
    struct Claims(&'static str, &'static [StateOpDecl]);

    impl StateVocabulary for Claims {
        fn name(&self) -> StateName {
            StateName::new(self.0)
        }
        fn ops(&self) -> &'static [StateOpDecl] {
            self.1
        }
        fn inputs(&self) -> &'static [StateInputDecl] {
            &[]
        }
        fn signatures(&self) -> SignaturePath {
            SignaturePath::new("claims::signatures")
        }
    }

    static TWIN_OPS: [StateOpDecl; 1] = [StateOpDecl {
        name: OpName::new("probe_split"),
        arity: Arity::new(1),
        outputs: OutputCount::new(TWO),
    }];

    /// A second decoder state that claims one of [`Probe`]'s ops.
    struct Twin;

    impl StateVocabulary for Twin {
        fn name(&self) -> StateName {
            StateName::new("twin")
        }
        fn ops(&self) -> &'static [StateOpDecl] {
            &TWIN_OPS
        }
        fn inputs(&self) -> &'static [StateInputDecl] {
            &[]
        }
        fn signatures(&self) -> SignaturePath {
            SignaturePath::new("twin::signatures")
        }
    }

    static KV_CACHE_INPUTS: [StateInputDecl; 1] = [StateInputDecl {
        name: InputName::new("kv_cache"),
    }];
    static PROBE_STATE_INPUTS: [StateInputDecl; 1] = [StateInputDecl {
        name: InputName::new("probe_state"),
    }];

    /// A decoder state whose one input is `inputs`, named `name`.
    struct Reads(&'static str, &'static [StateInputDecl]);

    impl StateVocabulary for Reads {
        fn name(&self) -> StateName {
            StateName::new(self.0)
        }
        fn ops(&self) -> &'static [StateOpDecl] {
            &[]
        }
        fn inputs(&self) -> &'static [StateInputDecl] {
            self.1
        }
        fn signatures(&self) -> SignaturePath {
            SignaturePath::new("::reads::signatures")
        }
    }

    fn classify_under(
        src: &str,
        decoder_states: &[&dyn StateVocabulary],
    ) -> Result<Program, ClassifyError> {
        let ast = parse_body(src).expect("DSL parse");
        classify_with(
            &ast,
            Prelude::Decoder,
            &StateTable::assemble(decoder_states)?,
        )
    }

    #[test]
    fn a_state_ops_id_names_it() {
        let table = StateTable::assemble(&[&Probe]).expect("probe assembles");
        let op = table.op("probe_mix").expect("probe declares probe_mix");
        assert_eq!(
            op.id(),
            StateOpId::new(StateName::new("probe"), OpName::new("probe_mix"))
        );
        let input = table
            .input("probe_state")
            .expect("probe declares its input");
        assert_eq!(
            input.id(),
            StateInputId::new(StateName::new("probe"), InputName::new("probe_state"))
        );
    }

    #[test]
    fn a_state_op_classifies_as_state() {
        let p = classify_under("(a, b) = probe_split(x)", &[&Probe]).expect("classification");
        let Stmt::AssignTuple {
            targets,
            value:
                Expr::Call {
                    op: OpKind::State(op),
                    args,
                },
        } = &p.statements[0]
        else {
            panic!(
                "expected a state op's tuple assignment, got {:?}",
                p.statements[0]
            );
        };
        assert_eq!(targets.len(), 2);
        assert_eq!(args.len(), 1);
        assert_eq!(op.name(), OpName::new("probe_split"));
        assert_eq!(op.state(), StateName::new("probe"));
        assert_eq!(op.outputs(), OutputCount::new(TWO));
    }

    #[test]
    fn a_state_input_is_layer_indexed() {
        let p = classify_under(
            "
            for layer in range(num_hidden_layers):
                y = probe_mix(x, probe_state[layer], x)
            ",
            &[&Probe],
        )
        .expect("classification");
        let Stmt::For { ivar, body, .. } = &p.statements[0] else {
            panic!("expected a for-loop");
        };
        let Stmt::Assign {
            value: Expr::Call { args, .. },
            ..
        } = &body[0]
        else {
            panic!("expected a call assignment");
        };
        let Expr::Extern {
            kind: ExternKind::State(input),
            index: Some(index),
        } = &args[1]
        else {
            panic!("expected a layer-indexed state input, got {:?}", args[1]);
        };
        assert_eq!(index, ivar);
        assert_eq!(input.name(), InputName::new("probe_state"));
        assert_eq!(input.state(), StateName::new("probe"));
    }

    #[test]
    fn a_state_input_needs_its_layer() {
        let err = classify_under("y = probe_mix(x, probe_state, x)", &[&Probe])
            .expect_err("an unindexed state input");
        let ClassifyError::StateInputUnindexed { input, at } = &err else {
            panic!("expected StateInputUnindexed, got {err}");
        };
        assert_eq!(*input, InputName::new("probe_state"));
        assert!(at.is_some(), "the error names where: {err}");
        assert!(err.to_string().contains("line 3:9"), "{err}");
    }

    #[test]
    fn an_unregistered_state_op_is_unknown() {
        let err = classify_under("(a, b) = probe_split(x)", &[]).expect_err("an unknown op");
        let msg = err.to_string();
        assert!(msg.contains("unknown op `probe_split`"), "{msg}");
        assert!(msg.contains("line 3:14"), "{msg}");
    }

    #[test]
    fn a_state_cannot_reuse_a_core_op_name() {
        let Err(err) = StateTable::assemble(&[&Squatter]) else {
            panic!("a decoder state claimed a core op's name");
        };
        let ClassifyError::DuplicateOpName {
            name,
            first,
            second,
        } = &err
        else {
            panic!("expected DuplicateOpName, got {err}");
        };
        assert_eq!(*name, OpName::new("attention"));
        assert_eq!(*first, Vocabulary::Core);
        assert_eq!(*second, StateName::new("squatter"));
        let msg = err.to_string();
        assert!(msg.contains("core") && msg.contains("squatter"), "{msg}");
    }

    #[test]
    fn two_decoder_states_cannot_share_an_op() {
        let Err(err) = StateTable::assemble(&[&Probe, &Twin]) else {
            panic!("two decoder states claimed one op");
        };
        let ClassifyError::DuplicateOpName {
            name,
            first,
            second,
        } = &err
        else {
            panic!("expected DuplicateOpName, got {err}");
        };
        assert_eq!(*name, OpName::new("probe_split"));
        assert_eq!(*first, Vocabulary::State(StateName::new("probe")));
        assert_eq!(*second, StateName::new("twin"));
        let msg = err.to_string();
        assert!(msg.contains("probe") && msg.contains("twin"), "{msg}");
    }

    #[test]
    fn a_state_cannot_reuse_a_core_input_name() {
        let Err(err) = StateTable::assemble(&[&Reads("kv_reader", &KV_CACHE_INPUTS)]) else {
            panic!("a decoder state claimed a core extern's name");
        };
        let ClassifyError::DuplicateInputName {
            name,
            first,
            second,
        } = &err
        else {
            panic!("expected DuplicateInputName, got {err}");
        };
        assert_eq!(*name, InputName::new("kv_cache"));
        assert_eq!(*first, Vocabulary::Core);
        assert_eq!(*second, StateName::new("kv_reader"));
        let msg = err.to_string();
        assert!(msg.contains("core") && msg.contains("kv_reader"), "{msg}");
    }

    #[test]
    fn two_decoder_states_cannot_share_an_input() {
        let Err(err) = StateTable::assemble(&[&Probe, &Reads("twin", &PROBE_STATE_INPUTS)]) else {
            panic!("two decoder states claimed one input");
        };
        let ClassifyError::DuplicateInputName {
            name,
            first,
            second,
        } = &err
        else {
            panic!("expected DuplicateInputName, got {err}");
        };
        assert_eq!(*name, InputName::new("probe_state"));
        assert_eq!(*first, Vocabulary::State(StateName::new("probe")));
        assert_eq!(*second, StateName::new("twin"));
        let msg = err.to_string();
        assert!(msg.contains("probe") && msg.contains("twin"), "{msg}");
    }

    /// The decoder states' dims assemble where their ops do: a carrier classified under them reads
    /// one table, and a dim claimed twice fails the same assembly.
    #[test]
    fn a_states_dims_assemble_with_its_vocabulary() {
        let table = StateTable::assemble(&[&Probe]).expect("probe assembles");
        assert!(table.dims().find("probe_width").is_some());
        assert!(table.dims().find("kv_dim").is_some());
        let Err(err) = StateTable::assemble(&[&Dims("dim_squatter", &GDN_CONV_CLAIM)]) else {
            panic!("a decoder state claimed a core dim's name");
        };
        let ClassifyError::DuplicateDimName { first, second, .. } = &err else {
            panic!("expected DuplicateDimName, got {err}");
        };
        assert_eq!(*first, Vocabulary::Core);
        assert_eq!(*second, StateName::new("dim_squatter"));
    }

    #[test]
    fn a_state_op_checks_its_arity() {
        let err = classify_under("y = probe_mix(x)", &[&Probe]).expect_err("a wrong arity");
        let ClassifyError::StateOpArity {
            op,
            expected,
            got,
            at,
        } = &err
        else {
            panic!("expected StateOpArity, got {err}");
        };
        assert_eq!(*op, OpName::new("probe_mix"));
        assert_eq!(*expected, Arity::new(3));
        assert_eq!(got.get(), 1);
        assert!(at.is_some(), "{err}");
        assert!(err.to_string().contains("line 3:9"), "{err}");
    }

    #[test]
    fn a_state_op_binds_its_declared_outputs() {
        let err = classify_under("y = probe_split(x)", &[&Probe]).expect_err("a wrong binding");
        let ClassifyError::StateOpOutputs {
            op, expected, got, ..
        } = &err
        else {
            panic!("expected StateOpOutputs, got {err}");
        };
        assert_eq!(*op, OpName::new("probe_split"));
        assert_eq!(*expected, OutputCount::new(TWO));
        assert_eq!(got.get(), 1);
    }

    /// The claim `state` makes on a core name: `DuplicateOpName` naming the core vocabulary.
    fn core_claim(state: &Claims, op: &str) {
        let Err(err) = StateTable::assemble(&[state]) else {
            panic!("a decoder state claimed the core name `{op}`");
        };
        let ClassifyError::DuplicateOpName { name, first, .. } = &err else {
            panic!("expected DuplicateOpName, got {err}");
        };
        assert_eq!(name.get(), op);
        assert_eq!(*first, Vocabulary::Core);
    }

    #[test]
    fn a_state_cannot_claim_a_special_form() {
        core_claim(&Claims("sqrt_claimer", &SQRT_OPS), "sqrt");
    }

    #[test]
    fn a_state_cannot_claim_a_synthesized_op() {
        core_claim(&Claims("mul_claimer", &MUL_OPS), "mul");
    }

    /// Every [`SpecialForm`] is a form the parser or classify takes before an op lookup: a body
    /// calling one classifies, with no decoder state, to no op call. So the reserved set is the forms the
    /// compiler really intercepts.
    #[test]
    fn every_special_form_is_intercepted() {
        use crate::ast::SpecialForm;
        for form in SpecialForm::ALL {
            let call = match form {
                SpecialForm::Sqrt => "sqrt(hidden_size)",
                SpecialForm::Scalar => "scalar(rms_norm_eps)",
                SpecialForm::RecipScalar => "recip_scalar(logits_scaling)",
                SpecialForm::Reshape => "reshape(x, [num_tokens, hidden_size])",
                SpecialForm::AsDim => "as_dim(x, hidden_size)",
            };
            let src = format!("x = embed(input_ids, embed_tokens)\ny = {call}");
            let p = classify_under(&src, &[])
                .unwrap_or_else(|e| panic!("`{}` is not intercepted: {e}", form.name()));
            let Stmt::Assign { value, .. } = &p.statements[1] else {
                panic!("expected an assignment");
            };
            let intercepted = match value {
                Expr::Call { op, .. } => *op == OpKind::Reshape,
                _ => true,
            };
            assert!(intercepted, "`{}` reached the op table", form.name());
        }
    }

    /// No op a pass inserts is one a body can call: its name is reserved only through
    /// [`OpKind::SYNTHESIZED`].
    #[test]
    fn synthesized_ops_have_no_dsl_surface() {
        for op in OpKind::SYNTHESIZED {
            assert_eq!(OpKind::from_name(op.as_str()), None, "{op:?}");
            assert!(OpKind::is_core_name(op.as_str()), "{op:?}");
        }
    }

    /// A decoder state that declares only the derived dims `.1`, named `.0`.
    struct Dims(&'static str, &'static [DerivedDim]);

    impl StateVocabulary for Dims {
        fn name(&self) -> StateName {
            StateName::new(self.0)
        }
        fn ops(&self) -> &'static [StateOpDecl] {
            &[]
        }
        fn inputs(&self) -> &'static [StateInputDecl] {
            &[]
        }
        fn signatures(&self) -> SignaturePath {
            SignaturePath::new("::dims::signatures")
        }
        fn derived_dims(&self) -> &'static [DerivedDim] {
            self.1
        }
    }

    /// A row named `name` of `stage` reading the bound `reads`.
    const fn row(name: &'static str, stage: DeriveStage, reads: &'static str) -> DerivedDim {
        DerivedDim {
            name,
            stage,
            rule: DerivedRule::Bound(reads),
        }
    }

    /// Every row's name, in the table's order.
    fn names(table: &DimTable) -> Vec<&'static str> {
        table.rows().map(|d| d.name).collect()
    }

    #[test]
    fn a_state_dim_joins_the_table_after_the_core_rows() {
        let table = DimTable::assemble(&[&Probe]).expect("probe assembles");
        let mut expected: Vec<_> = DERIVED_DIMS.iter().map(|d| d.name).collect();
        expected.push("probe_width");
        assert_eq!(names(&table), expected);
        let probe = table
            .find("probe_width")
            .expect("probe declares probe_width");
        let bounds = [("hidden_size", 8), ("num_attention_heads", 2)]
            .map(|(k, v)| (k.to_string(), v))
            .into();
        assert_eq!(probe.rule.eval(&bounds), Some(10));
        assert!(table.find("gdn_conv_dim").is_some());
        assert!(table.find("hidden_size").is_none());
    }

    static GDN_CONV_CLAIM: [DerivedDim; 1] =
        [row("gdn_conv_dim", DeriveStage::Implicit, "hidden_size")];

    #[test]
    fn a_state_cannot_claim_a_core_dim_name() {
        let Err(err) = DimTable::assemble(&[&Dims("squatter", &GDN_CONV_CLAIM)]) else {
            panic!("a decoder state claimed a core dim's name");
        };
        let ClassifyError::DuplicateDimName {
            name,
            first,
            second,
        } = &err
        else {
            panic!("expected DuplicateDimName, got {err}");
        };
        assert_eq!(*name, DimName::new("gdn_conv_dim"));
        assert_eq!(*first, Vocabulary::Core);
        assert_eq!(*second, StateName::new("squatter"));
        let msg = err.to_string();
        assert!(msg.contains("core") && msg.contains("squatter"), "{msg}");
    }

    static PROBE_WIDTH_CLAIM: [DerivedDim; 1] =
        [row("probe_width", DeriveStage::Implicit, "hidden_size")];
    static SELF_TWICE: [DerivedDim; 2] = [
        row("twice", DeriveStage::Implicit, "hidden_size"),
        row("twice", DeriveStage::Implicit, "hidden_size"),
    ];

    #[test]
    fn two_states_cannot_share_a_dim() {
        let Err(err) = DimTable::assemble(&[&Probe, &Dims("twin", &PROBE_WIDTH_CLAIM)]) else {
            panic!("two decoder states claimed one dim");
        };
        let ClassifyError::DuplicateDimName {
            name,
            first,
            second,
        } = &err
        else {
            panic!("expected DuplicateDimName, got {err}");
        };
        assert_eq!(*name, DimName::new("probe_width"));
        assert_eq!(*first, Vocabulary::State(StateName::new("probe")));
        assert_eq!(*second, StateName::new("twin"));
        let msg = err.to_string();
        assert!(msg.contains("probe") && msg.contains("twin"), "{msg}");

        let Err(err) = DimTable::assemble(&[&Dims("stutter", &SELF_TWICE)]) else {
            panic!("one decoder state claimed one dim twice");
        };
        let ClassifyError::DuplicateDimName { first, second, .. } = &err else {
            panic!("expected DuplicateDimName, got {err}");
        };
        assert_eq!(*first, Vocabulary::State(StateName::new("stutter")));
        assert_eq!(*second, StateName::new("stutter"));
    }

    /// The order law `name`'s row breaks by reading `reads`, as the error names them.
    fn breaks_order(states: &[&dyn StateVocabulary], name: &str, reads: &str) {
        let Err(err) = DimTable::assemble(states) else {
            panic!("`{name}` reads `{reads}`, which it may not, yet the table assembled");
        };
        let ClassifyError::DimReadsUnavailable {
            name: got_name,
            reads: got_reads,
            ..
        } = &err
        else {
            panic!("expected DimReadsUnavailable, got {err}");
        };
        assert_eq!((got_name.get(), got_reads.get()), (name, reads), "{err}");
        let msg = err.to_string();
        assert!(msg.contains(name) && msg.contains(reads), "{msg}");
    }

    static READS_OWN_EARLIER: [DerivedDim; 2] = [
        row("own_first", DeriveStage::Implicit, "hidden_size"),
        row("own_second", DeriveStage::Implicit, "own_first"),
    ];
    static READS_OWN_LATER: [DerivedDim; 2] = [
        row("own_first", DeriveStage::Implicit, "own_second"),
        row("own_second", DeriveStage::Implicit, "hidden_size"),
    ];
    static READS_LATER_STAGE: [DerivedDim; 1] =
        [row("too_early", DeriveStage::Implicit, "q_global_dim")];
    static READS_BEFORE_FALLBACKS_CORE: [DerivedDim; 1] =
        [row("before_core", DeriveStage::BeforeFallbacks, "kv_dim")];
    static READS_PROBE: [DerivedDim; 1] = [row("borrows", DeriveStage::Implicit, "probe_width")];

    /// The table's order is load-bearing: a rule reading another derived dim finds it derived only
    /// if that one comes EARLIER, at the same or an earlier stage. So a reordering that would
    /// silently change a value fails here.
    #[test]
    fn a_derived_dim_reads_only_derived_dims_above_it() {
        DimTable::assemble(&[]).expect("the core rows keep the order law");
    }

    /// A decoder state's row reads the core rows and its own rows above it, at the same or an earlier
    /// stage, and nothing else the table derives: not its own later row, not a later stage's row,
    /// and not another decoder state's, which deleting that decoder state would take away.
    #[test]
    fn a_state_row_reads_only_earlier_rows() {
        let table = DimTable::assemble(&[&Probe, &Dims("own", &READS_OWN_EARLIER)])
            .expect("a row reading the core inputs and its own earlier row assembles");
        assert!(table.find("own_second").is_some());
        breaks_order(&[&Dims("own", &READS_OWN_LATER)], "own_first", "own_second");
        breaks_order(
            &[&Dims("early", &READS_LATER_STAGE)],
            "too_early",
            "q_global_dim",
        );
        breaks_order(
            &[&Dims("first", &READS_BEFORE_FALLBACKS_CORE)],
            "before_core",
            "kv_dim",
        );
        breaks_order(
            &[&Probe, &Dims("borrower", &READS_PROBE)],
            "borrows",
            "probe_width",
        );
        breaks_order(
            &[&Dims("borrower", &READS_PROBE), &Probe],
            "borrows",
            "probe_width",
        );
    }
}
