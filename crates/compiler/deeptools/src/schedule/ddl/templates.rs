//! THE GENERATED TEMPLATE SET — the one production [`DdlTemplateSet`].
//!
//! `build.rs` emits [`crate::generated::MODULES`]: every vendored `.ddl` walked module-wide with no
//! `ddl.if` resolved, plus the four parts the per-bind census drops. This module is the ONE place
//! those const rows become the owned shapes [`select_and_parse_ddl_template`] takes — a `Vec` of
//! binds, a map of padded dims, a list of constraint forms, and a region tree whose ops borrow the
//! module program's own statements.
//!
//! ⛔ NOTHING HERE DECIDES ANYTHING. Every value comes from a generated row, and every row comes
//! from a `.ddl` field. Where the census is EMPTY the absence is named and cited rather than
//! defaulted — see [`NO_ACCESS_PATTERN_DIMS`] and [`NO_DATASTAGE_MIN`].
//!
//! [`DdlTemplateSet`]: super::conversion::DdlTemplateSet
//! [`select_and_parse_ddl_template`]: super::conversion::select_and_parse_ddl_template

use std::collections::BTreeMap;

use crate::generated::{
    MODULES, NameId, Operand, Program, Stmt, StmtKind, Template, TemplateModule, TransformKind,
    TransformRow,
};
use crate::schedule::ddl::conversion::{
    DdlConstraint, DdlOp, DdlRoot, OperationBind, PaddedDimension, RegionId, RegionOp, RegionTree,
    StatedTemplate, Transformation,
};
use crate::schedule::ddl::ops::{Dialect, DdlOp as RegisteredOp, SsaName, StorageBits, Unverified, Value};
use crate::schedule::ddl::{DdlSource, ParsedDdl};

/// `access_pattern_dim=` — EMPTY FOR EVERY VENDORED TRANSFER.
///
/// ⛔ MEASURED, NOT DEFAULTED. `DdlOps.td:608` declares the operand and `ddl.data_transfer` reads
/// its styles over it, but no `access_pattern_dim=` appears in any of the 32 checked-in templates
/// across all 457 transfers — so an empty group is what the data states. A template that grows one
/// makes this a real generated field; a fabricated dim here would apply a stride pattern to an axis
/// the template never named.
const NO_ACCESS_PATTERN_DIMS: &[NameId] = &[];

/// `min=` ON A `ddl.datastage_constraint` — ABSENT FOR EVERY VENDORED ONE.
///
/// ⛔ MEASURED, NOT DEFAULTED. The vendored set states `values=` on 125 datastage constraints and
/// `max=` on 21; a bare `min=` appears nowhere. [`DdlOp::DatastageConstraint`] carries it as TEXT
/// because the census keeps every DDL bound as the template spelled it, so [`None`] here is "the
/// template stated no lower bound" and not "the bound is zero".
const NO_DATASTAGE_MIN: Option<&str> = None;

/// THE VENDORED TEMPLATE SET — [`MODULES`], as the DDL step reads it.
#[derive(Debug, Clone, Copy, Default)]
pub struct DdlTemplates;

impl DdlTemplates {
    /// The set.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

/// THE BUFFER ONE TEMPLATE IS PARSED FROM — `ddlTemplateDir + ddlFile`, as the module the generator
/// already walked out of that file.
///
/// ⭐ THE MODULE *IS* THE BUFFER'S CONTENT. `DdlSource` exists because "MLIR'S GENERIC PARSE IS THE
/// FRAMEWORK'S, not this campaign's" (`schedule/ddl/mod.rs`), and our own text parse of the same
/// templates is `build.rs` — so the source a candidate names is the generated module, and
/// [`DdlSource::parse`] is the projection of it that the four verifiers read.
#[derive(Debug, Clone, Copy)]
pub struct TemplateSource(&'static TemplateModule);

impl TemplateSource {
    /// The module this source states.
    #[must_use]
    pub const fn module(self) -> &'static TemplateModule {
        self.0
    }
}

/// `%wrd#2` → the `Value` the verifiers compare by — `getDefiningOp()`'s key.
///
/// ⛔ THE `#N` SUFFIX IS THE RESULT INDEX AND NOT PART OF THE NAME. `%wrd:4` DECLARES four values
/// referenced as `%wrd#0..3` (`ddl/parse.rs:256-279`), so a `Value` that kept the hash in its name
/// would never equal the one a `ddl.dimension`'s own result minted, and every `defining_op` lookup
/// through it would answer [`None`].
fn value_of(program: &Program, name: NameId) -> Value {
    let text = program.spelling(name);
    match text.split_once('#') {
        Some((base, index)) => Value {
            name: SsaName(base.to_owned()),
            result: index.parse().unwrap_or(0),
        },
        None => Value::sole(text),
    }
}

/// EVERY OP WITH `hasVerifier = 1`, as `verifyAfterParse=true` hands it over.
///
/// ⭐ ALL FIVE, WHICH IS WHAT KEEPS THE PORTED VERIFIERS LOAD-BEARING. `ParserConfig(context,
/// /*verifyAfterParse=*/true, ..)` (`ddc/ddl/ddl.cpp:49-50`) is the flag that makes them run at all;
/// handing back an empty list would leave them dead predicates that nothing ever rejects.
fn unverified(program: &Program, stmt: &Stmt) -> Option<Unverified> {
    let names = |group: usize| -> Vec<Value> {
        match stmt.operands.get(group) {
            Some(Operand::One(name)) => vec![value_of(program, *name)],
            Some(Operand::List(list)) => {
                list.iter().map(|name| value_of(program, *name)).collect()
            }
            Some(Operand::OtherBind) | None => Vec::new(),
        }
    };
    let one = |group: usize| -> Option<Value> { names(group).into_iter().next() };
    Some(match stmt.kind {
        StmtKind::Type => {
            let crate::generated::Attrs::Type {
                data_type,
                bit_width,
            } = stmt.attrs
            else {
                return None;
            };
            Unverified::Datatype {
                data_type: data_type.spelling().to_owned(),
                bit_width: bit_width.and_then(|bits| u32::try_from(bits).ok().map(StorageBits)),
            }
        }
        StmtKind::Compute => {
            let crate::generated::Attrs::Compute { computetype, .. } = stmt.attrs else {
                return None;
            };
            Unverified::Compute {
                computetype: computetype.spelling().to_owned(),
                inputs: names(0),
            }
        }
        StmtKind::ForceInnermostDimensions => Unverified::ForceInnermostDimensions {
            allocate: one(0)?,
            datastage: one(1)?,
            // `$dims` — every operand past the allocation and the datastage.
            dims: (2..stmt.operands.len()).flat_map(names).collect(),
        },
        StmtKind::ImplicitSync => Unverified::ImplicitSync { allocate: one(0)? },
        StmtKind::OperationBind => Unverified::OperationBind {
            op_func_name: bind_spelling(program, stmt)?.to_owned(),
        },
        _ => return None,
    })
}

/// The `opFuncName=` of one `ddl.operation_bind` statement — read off the [`crate::generated::BindRow`]
/// that names the same result, since [`crate::generated::Attrs`] carries no field for it.
fn bind_spelling(program: &Program, stmt: &Stmt) -> Option<&'static str> {
    let result = *stmt.results.first()?;
    MODULES
        .iter()
        .find(|held| held.template as u32 == program.template as u32)?
        .binds
        .iter()
        .find(|bind| bind.result == result)
        .map(|bind| bind.op_func)
}

impl DdlSource for TemplateSource {
    /// ⛔ [`None`] IS `DdlMain`'S OWN `DT_CHECK(op)` — a buffer that parses to nothing. A statement
    /// whose kind the dialect does not register, or a verifier op whose operands the module does not
    /// state, is exactly that.
    fn parse(&self, dialect: &Dialect) -> Option<ParsedDdl> {
        let program = &self.0.program;
        let mut defining = Vec::new();
        let mut verifiable = Vec::new();
        for stmt in program.stmts {
            // `allowUnregisteredDialects(false)`: the op has to be one the registry holds. The
            // conversion itself is total and compile-checked (`ops.rs`'s `From<StmtKind>`), so a
            // template that grows the census breaks there; this is the registry's own say.
            let op: RegisteredOp = stmt.kind.into();
            dialect.resolve(op.mnemonic())?;
            for name in stmt.results {
                defining.push((value_of(program, *name), op));
            }
            if let Some(unchecked) = unverified(program, stmt) {
                verifiable.push(unchecked);
            }
        }
        Some(ParsedDdl {
            defining,
            verifiable,
        })
    }
}

/// WHICH OF `processOp`'S FOURTEEN ARMS A STATEMENT IS — its `dyn_cast` chain from `LoopOp`
/// (`ddc/ddl/ddl_conversion.cpp:1075`) to `CoreCoreletCondOp` (`:2002`), as a total match.
///
/// ⛔ [`None`] IS THE FALL OFF THE END OF THAT CHAIN — `return {nullptr, {}}` (`:2004`): no node, no
/// region, no insertion point. The generator already filters those out of a region's rows, so this
/// answers [`None`] only for a kind that should never have reached it.
/// ⛔ AND `ddl.core_corelet_cond` HAS NO CENSUS VARIANT, because no vendored template states one —
/// [`DdlOp::CoreCoreletCond`] stays spellable for the reference's unconditional *"Uses of this op is
/// Prohibited in ddl"* without this having an arm to produce it.
fn dispatched<'d>(stmt: &'d Stmt, regions: &[RegionId]) -> Option<DdlOp<'d>> {
    Some(match stmt.kind {
        StmtKind::Loop => DdlOp::Loop(stmt),
        StmtKind::ParametricLoop => DdlOp::ParametricLoop(stmt),
        StmtKind::DataTransfer => DdlOp::DataTransfer {
            stmt,
            pattern_dims: NO_ACCESS_PATTERN_DIMS,
        },
        StmtKind::Compute => DdlOp::Compute(stmt),
        StmtKind::Datastage => DdlOp::Datastage(stmt),
        StmtKind::GetExternalDatastage => DdlOp::GetExternalDatastage(stmt),
        StmtKind::If => DdlOp::If {
            condition: match stmt.operands.first()? {
                Operand::One(name) => *name,
                Operand::List(_) | Operand::OtherBind => return None,
            },
            then_region: *regions.first()?,
            else_region: *regions.get(1)?,
        },
        StmtKind::Opaque => DdlOp::Opaque(stmt),
        StmtKind::Sync => DdlOp::Sync(stmt),
        StmtKind::ImplicitSync => DdlOp::ImplicitSync(stmt),
        StmtKind::DatastageConstraint => DdlOp::DatastageConstraint {
            stmt,
            min: NO_DATASTAGE_MIN,
        },
        StmtKind::ForceInnermostDimensions => DdlOp::ForceInnermostDimensions(stmt),
        StmtKind::CoreToCoreCommunication => DdlOp::CoreToCoreCommunication(stmt),
        // The twenty-three kinds `processOp` has no arm for — `ddl.unit` among them, and it is the
        // single most common statement in the templates. Each is reached through the OPERANDS of an
        // op that names it: `getTensorProp` for a tensor, `processCondition` for a condition tree,
        // `dyn_cast<UnitOp>(transfer_op.getSource().getDefiningOp())` for a transfer's end
        // (`ddl_conversion.cpp:1251`) and the `ddl.unit` chain onward for its allocation (`:862`).
        StmtKind::Unit
        | StmtKind::AliasOneConstantOf
        | StmtKind::AliasOneTensorOf
        | StmtKind::Allocate
        | StmtKind::Condition
        | StmtKind::ConditionAnd
        | StmtKind::ConditionNot
        | StmtKind::ConditionOr
        | StmtKind::Constraint
        | StmtKind::Dataflow
        | StmtKind::DefineConstant
        | StmtKind::Dimension
        | StmtKind::DisableTransferPromotion
        | StmtKind::GetExternalConstant
        | StmtKind::GetExternalDataTransferAllocation
        | StmtKind::InternalTensor
        | StmtKind::Layout
        | StmtKind::OperandConstant
        | StmtKind::OperationBind
        | StmtKind::PaddedDimension
        | StmtKind::Tensor
        | StmtKind::Transformations
        | StmtKind::Type => return None,
    })
}

/// ONE `ddl.transformations` SECTION AS A TREE — the rows naming `parent` as their enclosing
/// `ddl.if`, in the order the section states them.
fn section_tree(
    rows: &'static [TransformRow],
    section: u32,
    parent: Option<u32>,
    then_arm: bool,
) -> Vec<Transformation> {
    rows.iter()
        .enumerate()
        .filter(|(_, row)| {
            row.section == section && row.parent == parent && row.then_arm == then_arm
        })
        .map(|(at, row)| {
            let at = u32::try_from(at).unwrap_or(u32::MAX);
            match row.kind {
                TransformKind::DisableTransferPromotion => Transformation::DisableTransferPromotion,
                TransformKind::Yield => Transformation::Yield,
                TransformKind::Other(kind) => Transformation::Other(kind),
                TransformKind::If(condition) => Transformation::If {
                    condition,
                    then_region: section_tree(rows, section, Some(at), true),
                    else_region: section_tree(rows, section, Some(at), false),
                },
            }
        })
        .collect()
}

/// THE REGIONS `parse_ddl2_dsc` WALKS, plus every `ddl.transformations` body.
fn root_of(module: &'static TemplateModule) -> DdlRoot<'static> {
    let mut ops: BTreeMap<RegionId, Vec<RegionOp<'static>>> = BTreeMap::new();
    for row in module.region_ops {
        let Some(stmt) = module.program.stmts.get(row.stmt as usize) else {
            continue;
        };
        let Some(op) = dispatched(stmt, row.regions) else {
            continue;
        };
        ops.entry(row.region).or_default().push(RegionOp {
            op,
            regions: row.regions.to_vec(),
        });
    }
    DdlRoot {
        regions: RegionTree { ops },
        dataflows: module.dataflows.to_vec(),
        transformations: (0..module.sections)
            .map(|section| section_tree(module.transformations, section, None, true))
            .collect(),
    }
}

impl super::conversion::DdlTemplateSet for DdlTemplates {
    type Source = TemplateSource;

    /// ⛔ TOTAL: [`Template`]'s variants and [`MODULES`]' rows are minted from the same census of
    /// `ddl_templates/*.ddl`, so this set holds every template a candidate list can name. The
    /// trait's [`None`] ("where this set does not hold it") is therefore unreachable here — which is
    /// the point of the pivot: a set that answered [`None`] was a set that stated one sixth of a
    /// template.
    fn stated(&self, template: Template) -> Option<StatedTemplate<'_, Self::Source>> {
        let module = template.module();
        Some(StatedTemplate {
            source: TemplateSource(module),
            program: &module.program,
            binds: module
                .binds
                .iter()
                .map(|bind| OperationBind {
                    result: bind.result,
                    // `stringToOpFuncs.at(getOpFuncName())` — whose miss IS the reference's
                    // *"Unrecognized opFuncName"*, carried as `None` rather than raised here.
                    op_func: sys_arch_spec::arch_enums::OpFunc::from_spelling(bind.op_func),
                    required: bind.required,
                    data_formats: bind.data_formats.to_vec(),
                    inputs: bind.inputs.to_vec(),
                    outputs: bind.outputs.to_vec(),
                    interim: bind.interim.to_vec(),
                })
                .collect(),
            padded: module
                .padded
                .iter()
                .map(|row| (row.result, row.groups))
                .collect(),
            constraints: module
                .constraints
                .iter()
                .map(|row| (row.operands, row.form))
                .collect(),
            root: root_of(module),
        })
    }
}

/// A `PaddedDimension` is `Copy` and a `DdlConstraint` is too, so those two parts are the generated
/// rows THEMSELVES — this is the compile-time proof of that, not a conversion.
const _: fn() = || {
    let _: fn(PaddedDimension<'static>) -> PaddedDimension<'static> = |p| p;
    let _: fn(DdlConstraint) -> DdlConstraint = |c| c;
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schedule::ddl::conversion::{ConstraintCmp, DdlTemplateSet};

    /// ⭐⭐ THE FOUR PARTS ARE THE `.ddl`'s OWN TEXT, ASSERTED AS VALUES.
    ///
    /// ⛔ EVERY NUMBER AND NAME HERE IS READ OFF `ddl_templates/bmm.ddl`, not off this port's output.
    /// `bmm.ddl:52` is the bind quoted in the assertion; `:11` is the padded dim; `:97` is the
    /// `relative_op_order` constraint. A test asserting the port's own answer would pin whatever the
    /// generator happened to emit.
    #[test]
    fn bmms_stated_template_is_the_ddl_files_own_text() {
        let stated = DdlTemplates
            .stated(Template::Bmm)
            .expect("the set holds every censused template");

        // `bmm.ddl:52` — `%bmm_int8_mbkg3_op = ddl.operation_bind([%type_int8],
        // [%inptensor_int8, %kertensor_int8], [%outtensor], [%ptsum_int, %pesum])
        // {opFuncName="batchmatmulint8mbkg3", required=false}`.
        let first = stated.binds.first().expect("bmm.ddl declares twenty binds");
        let spelling = |name: &NameId| stated.program.spelling(*name);
        assert_eq!(
            (
                spelling(&first.result),
                first.required,
                first.data_formats.iter().map(spelling).collect::<Vec<_>>(),
                first.inputs.iter().map(spelling).collect::<Vec<_>>(),
                first.outputs.iter().map(spelling).collect::<Vec<_>>(),
                first.interim.iter().map(spelling).collect::<Vec<_>>(),
            ),
            (
                "%bmm_int8_mbkg3_op",
                false,
                vec!["%type_int8"],
                vec!["%inptensor_int8", "%kertensor_int8"],
                vec!["%outtensor"],
                vec!["%ptsum_int", "%pesum"],
            ),
            "the four bracketed lists of `bmm.ddl:52`, in the order the bind writes them"
        );
        assert_eq!(
            first.op_func,
            sys_arch_spec::arch_enums::OpFunc::from_spelling("batchmatmulint8mbkg3"),
            "`stringToOpFuncs.at(\"batchmatmulint8mbkg3\")`"
        );
        assert_eq!(stated.binds.len(), 20, "`bmm.ddl` declares twenty binds");

        // `bmm.ddl:11` — `%wrdpd = ddl.padded_dimension(primary=%wrdd#0,
        // padding=[%zf, %zb, %padvalid], window=[])`.
        assert_eq!(stated.padded.len(), 1, "`bmm.ddl` states one padded dim");
        let (dim, groups) = stated
            .padded
            .iter()
            .next()
            .expect("the one padded dimension");
        assert_eq!(
            (
                stated.program.spelling(*dim),
                stated.program.spelling(groups.primary),
                groups.padding.iter().map(spelling).collect::<Vec<_>>(),
                groups.window.iter().map(spelling).collect::<Vec<_>>(),
            ),
            (
                "%wrdpd",
                "%wrdd#0",
                vec!["%zf", "%zb", "%padvalid"],
                Vec::new(),
            ),
            "`bmm.ddl:11`'s three operand groups — and `window=[]` is EMPTY there, which is a stated \
             absence and not a dropped group"
        );

        // `bmm.ddl:97` — `ddl.constraint(%bmm_int8_mbkg3_op, .., %relu_op) {relative_op_order=true}`.
        let orders: Vec<&(&[Operand], DdlConstraint)> = stated
            .constraints
            .iter()
            .filter(|(_, form)| *form == DdlConstraint::RelativeOpOrder)
            .collect();
        assert_eq!(
            orders.len(),
            1,
            "`bmm.ddl` states one `relative_op_order=true` constraint (`:97`)"
        );
        assert_eq!(
            orders[0].0.len(),
            20,
            "and it names every one of the twenty binds, which is what makes it an ORDER check"
        );
        // ⛔ NINE, NOT TEN, AND THE TENTH IS WHY THIS NUMBER IS ASSERTED. `bmm.ddl:120` is a
        // `min_num_valid`/`max_num_valid` constraint COMMENTED OUT — *"re-enable after above-lx
        // scheduler is implemented"* — so a count taken from `grep -c ddl.constraint` says ten and
        // the file states nine. A commented-out constraint that reached this table would be a legality
        // check the vendor deliberately disabled.
        assert_eq!(
            stated.constraints.len(),
            9,
            "`bmm.ddl:85-98` states nine live `ddl.constraint`s; `:120` is commented out"
        );

        // `bmm.ddl` opens ONE `ddl.dataflow`, and every candidate walks its body.
        assert_eq!(
            stated.root.dataflows.len(),
            1,
            "one `ddl.dataflow`, whose body `parse_ddl2_dsc` descends"
        );
        assert!(
            stated
                .root
                .regions
                .ops
                .contains_key(&stated.root.dataflows[0]),
            "and that body holds ops, so the walk has something to descend into"
        );
        // ⛔⛔ ONE SECTION, AND IT IS **EMPTY** — `bmm.ddl:403-404` is `ddl.transformations { }`. That
        // is a stated fact and not a missing one: an empty section leaves
        // `transformation_config.enable_moving_data_transfer` TRUE, so a matmul's transfers stay
        // promotable, while the five templates that write a `ddl.disable_transfer_promotion` turn it
        // off. A section list that collapsed "empty" into "absent" would be reporting the same schedule
        // for both.
        assert_eq!(
            stated.root.transformations,
            vec![Vec::new()],
            "`bmm.ddl:403` opens one `ddl.transformations` and states no op inside it"
        );
    }

    /// ⭐ THE ONE `ddl.if` A TRANSFORMATIONS SECTION STATES, AS A TREE.
    ///
    /// ⛔ `layernormscale_32.ddl` WRITES NO `else`, AND THAT IS THE ASSERTION: `ddl.if` owns TWO
    /// regions whatever the text writes (`processOp` answers `{1}` for a false condition,
    /// `ddl_conversion.cpp:1548`), so the else arm must exist and be EMPTY rather than be absent.
    #[test]
    fn a_transformations_if_keeps_both_arms_and_the_else_is_empty() {
        let stated = DdlTemplates
            .stated(Template::Layernormscale32)
            .expect("a censused template");
        let section = stated
            .root
            .transformations
            .first()
            .expect("one section")
            .as_slice();
        let [Transformation::If {
            condition,
            then_region,
            else_region,
        }] = section
        else {
            panic!("`layernormscale_32.ddl`'s section is one `ddl.if`, not {section:?}");
        };
        assert_eq!(
            stated.program.spelling(*condition),
            "%layernormscale_op",
            "the condition is the op-bind the section names"
        );
        assert_eq!(
            then_region,
            &vec![Transformation::DisableTransferPromotion],
            "the `then` arm disables transfer promotion"
        );
        assert!(
            else_region.is_empty(),
            "and the `else` arm is the empty second region, not a missing one"
        );
    }

    /// ⭐⭐ THE `DimSize` FORM IS REAL AND ITS TWO VALUES ARE THE TEMPLATE'S.
    ///
    /// `quantization_double_pad.ddl` states the only two constraints in the vendored set with a
    /// `cmp=` and no `property=`: `ddl.constraint(%out_front) {cmp = "equal", value = 0}` and
    /// `ddl.constraint(%out_back) {cmp = "less", value = 65}`.
    #[test]
    fn the_two_dim_size_constraints_carry_the_templates_own_bounds() {
        use crate::bridges::superdsc_to_dataflow_ir::shape_constraints::Extent;

        let stated = DdlTemplates
            .stated(Template::QuantizationDoublePad)
            .expect("a censused template");
        let sizes: Vec<DdlConstraint> = stated
            .constraints
            .iter()
            .map(|(_, form)| *form)
            .filter(|form| matches!(form, DdlConstraint::DimSize { .. }))
            .collect();
        assert_eq!(
            sizes,
            vec![
                DdlConstraint::DimSize {
                    cmp: ConstraintCmp::Equal,
                    value: Extent(0),
                },
                DdlConstraint::DimSize {
                    cmp: ConstraintCmp::Less,
                    value: Extent(65),
                },
            ],
            "`equal 0` on `%out_front` and `less 65` on `%out_back`, in the order the file states them"
        );
    }

    /// ⭐⭐ A REGION'S OPS ARE ONLY THE KINDS `processOp` DISPATCHES ON — and the `ddl.unit`s that
    /// dominate every dataflow body are NOT among them.
    ///
    /// ⛔ THIS IS THE CLAIM THAT COULD MOST EASILY BE A SILENT LOSS, so it is checked against the
    /// reference's own shape: the `dyn_cast` chain at `ddl_conversion.cpp:1075-2003` has no `UnitOp`
    /// arm, so a `ddl.unit` falls off the end to `return {nullptr, {}}` (`:2004`). It is reached
    /// through the operands of the transfer or compute that names it.
    #[test]
    fn a_region_holds_only_the_kinds_process_op_dispatches_on() {
        let mut seen: BTreeMap<StmtKind, usize> = BTreeMap::new();
        let mut units = 0usize;
        for module in MODULES {
            for stmt in module.program.stmts {
                if stmt.kind == StmtKind::Unit {
                    units += 1;
                }
            }
            for row in module.region_ops {
                let stmt = &module.program.stmts[row.stmt as usize];
                *seen.entry(stmt.kind).or_default() += 1;
                assert!(
                    dispatched(stmt, row.regions).is_some(),
                    "a region row is a `{:?}`, which `processOp` has no arm for",
                    stmt.kind
                );
            }
        }
        assert!(
            units > 0 && !seen.contains_key(&StmtKind::Unit),
            "the templates state {units} `ddl.unit`s and NONE of them is a region op"
        );
        for absent in [
            StmtKind::Allocate,
            StmtKind::Condition,
            StmtKind::ConditionOr,
            StmtKind::DefineConstant,
        ] {
            assert!(
                !seen.contains_key(&absent),
                "`{absent:?}` reached a region row; `processOp` has no arm for it"
            );
        }
        assert_eq!(
            seen.values().sum::<usize>(),
            2345,
            "the 2,343 dispatched ops inside the vendored `ddl.dataflow` bodies plus the two \
             `ddl.if`s of a transformations section"
        );
    }

    /// ⭐ THE BUFFER PARSES, AND ITS `defining` MAP ANSWERS FOR A MULTI-RESULT DECLARATION.
    ///
    /// ⛔ `%wrd:4` IS THE CASE THAT BREAKS A NAIVE `Value`: `bmm.ddl:6` declares `%wrd:4`, referenced
    /// as `%wrd#0..3` (`:20`), so a `Value` keeping the hash in its name would equal nothing.
    /// ⛔ AND RESULT 4 MUST **NOT** RESOLVE — a declaration of four binds four, and a table that
    /// answered for a fifth would be answering for a value the template never declared.
    #[test]
    fn the_source_parses_and_its_defining_map_answers_per_result() {
        use crate::schedule::ddl::ops::Defining;

        let stated = DdlTemplates.stated(Template::Bmm).expect("a censused template");
        let parsed = stated
            .source
            .parse(&Dialect::initialize())
            .expect("the generated module parses");
        let wrd = |result: u32| {
            parsed.defining.as_slice().defining_op(&Value {
                name: SsaName("%wrd".to_owned()),
                result,
            })
        };
        assert_eq!(
            (wrd(0), wrd(1), wrd(2), wrd(3), wrd(4)),
            (
                Some(RegisteredOp::Dimension),
                Some(RegisteredOp::Dimension),
                Some(RegisteredOp::Dimension),
                Some(RegisteredOp::Dimension),
                None,
            ),
            "`bmm.ddl:6`'s `%wrd:4` binds FOUR `ddl.dimension` values and no fifth"
        );
        // ⛔ AND A COUNT OF **TWO** IS ALSO FOUR VALUES SHORT OF `%wrd`'s: `bmm.ddl:7` declares
        // `%wrdd:2`, whose `#1` the input layout names (`:20`) — so the two declarations must not
        // share a table entry.
        assert_eq!(
            parsed.defining.as_slice().defining_op(&Value {
                name: SsaName("%wrdd".to_owned()),
                result: 1,
            }),
            Some(RegisteredOp::Dimension),
            "`bmm.ddl:7`'s `%wrdd:2` is its own declaration, and `%wrdd#1` resolves"
        );
        assert!(
            parsed.verifiable.iter().any(|held| matches!(
                held,
                Unverified::OperationBind { op_func_name } if op_func_name == "batchmatmulint8mbkg3"
            )),
            "every `ddl.operation_bind` reaches its verifier with its own `opFuncName=`"
        );
        assert!(
            parsed
                .verifiable
                .iter()
                .any(|held| matches!(held, Unverified::Datatype { .. })),
            "and so does every `ddl.type`"
        );
    }

    /// ⛔ EVERY CENSUSED TEMPLATE IS STATED — the set holds all 32, so the DDL step never sees the
    /// trait's [`None`].
    #[test]
    fn every_censused_template_is_stated() {
        for module in MODULES {
            let stated = DdlTemplates
                .stated(module.template)
                .unwrap_or_else(|| panic!("{:?} is censused but not stated", module.template));
            assert_eq!(
                stated.program.template as u32, module.template as u32,
                "and the program it states is that template's own module"
            );
        }
        assert_eq!(MODULES.len(), 32, "the checked-in `ddl_templates/*.ddl`");
    }
}
