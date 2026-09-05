//! Turns the vendored `.ddl` text into const tables.
//!
//! This is the ONE place text exists. Parsing text is the only thing in this crate that cannot be a
//! constant, so it happens here, once, when the crate compiles — and what reaches `src/` is `const`s.
//! The parser lives in `ddl/`, outside `src/`, so its `String`s cannot reach the emitter.
//!
//! Everything emitted is derived from the templates, never listed by hand: the statement-kind set IS
//! the census of `ddl.*` mnemonics across `ddl_templates/*.ddl`, and the program set IS the census of
//! `opFuncName=` — so adding a template grows the tables and a stale hand-written list cannot drift
//! from the data.

#[path = "ddl/ast.rs"]
mod ast;
#[path = "ddl/dataflow.rs"]
mod dataflow;
#[path = "ddl/parse.rs"]
mod parse;
#[path = "ddl/selection.rs"]
mod selection;
#[path = "ddl/smc.rs"]
#[allow(dead_code)]
mod smc;

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::Path;

/// `ddl.data_transfer` → `DataTransfer`. The mnemonic's own spelling, so a variant cannot be named
/// something the template does not say.
fn variant_of(mnemonic: &str) -> String {
    ident_of(mnemonic.trim_start_matches("ddl."))
}

/// A TEMPLATE'S OWN SPELLING AS A RUST VARIANT NAME.
///
/// ⛔ `-` BECOMES `To`, NOT `_`. The PT row spans are spelled `ptrow1-7` and `ptrow1-3`
/// (`bmm.ddl:260`), and a dash is not an identifier character. Mapping it to an underscore would give
/// `Ptrow1_7`, which is not upper-camel and so is a naming-lint failure that could only be silenced
/// with an allow — and the span reads correctly as `Ptrow1To7` anyway.
///
/// ⛔ AND THE CASE OF THE SOURCE IS DISCARDED, which is why [`check_no_ident_collisions`] exists:
/// `FMA16` and `fma16` would both become `Fma16`, and a table that silently merged two computetypes
/// is exactly the kind of wrong table that reads as correct.
fn ident_of(spelling: &str) -> String {
    // ⛔ A DOT IS NOT A WORD BREAK HERE. Constant names include `"0.0"` and `"1.0"`
    // (`unary_parallel.ddl`), and splitting on `.` the way `_` is split would spell both as digits
    // run together — `"1.0"` and a hypothetical `"10"` would become one variant. Keeping the point
    // as a word makes them distinct, and `check_no_ident_collisions` still holds the line.
    let spelling = spelling.replace('.', "_point_");
    let ident: String = spelling
        .replace('-', "_to_")
        .split('_')
        .filter(|word| !word.is_empty())
        .map(|word| {
            let mut chars = word.chars();
            match chars.next() {
                Some(first) => {
                    first.to_uppercase().collect::<String>() + &chars.as_str().to_lowercase()
                }
                None => String::new(),
            }
        })
        .collect();
    // A variant may not start with a digit, and `"0.0"` does. The prefix is uniform so two
    // digit-leading spellings stay as distinct as they were.
    if ident.starts_with(|c: char| c.is_ascii_digit()) {
        format!("N{ident}")
    } else {
        ident
    }
}

/// 🛑 TWO SPELLINGS THAT PRODUCE ONE VARIANT ARE A MERGED TABLE, so say so rather than emitting it.
fn check_no_ident_collisions(what: &str, spellings: &BTreeSet<String>) {
    let mut seen: BTreeMap<String, &str> = BTreeMap::new();
    for spelling in spellings {
        let ident = ident_of(spelling);
        if let Some(first) = seen.insert(ident.clone(), spelling) {
            panic!(
                "{what}: `{first}` and `{spelling}` both spell the variant `{ident}`; \
                 emitting this table would merge two distinct values into one"
            );
        }
    }
}

/// EMIT A CLOSED SET FROM A CENSUS: one variant per distinct spelling, plus the spelling back.
///
/// ⭐ THE SET IS THE DATA'S, NOT ANYONE'S. A template that names a value this crate has no variant
/// for grows the enum, so the match arms that read it stop compiling — which is the point of having
/// no `_` arm.
fn codegen_closed_set(out: &mut String, name: &str, doc: &str, spellings: &BTreeSet<String>) {
    check_no_ident_collisions(name, spellings);
    let _ = writeln!(out, "/// {doc}");
    let _ = writeln!(
        out,
        "///\n/// Generated from the census of the vendored templates — {} distinct spellings.",
        spellings.len()
    );
    out.push_str("#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]\n");
    let _ = writeln!(out, "pub enum {name} {{");
    for spelling in spellings {
        let _ = writeln!(out, "    /// `{spelling}`.\n    {},", ident_of(spelling));
    }
    out.push_str("}\n\n");
    let _ = writeln!(
        out,
        "impl {name} {{\n    /// The template's own spelling, which is what reaches the IR text.\n    \
         pub const fn spelling(self) -> &'static str {{\n        match self {{"
    );
    for spelling in spellings {
        let _ = writeln!(
            out,
            "            Self::{} => \"{spelling}\",",
            ident_of(spelling)
        );
    }
    out.push_str("        }\n    }\n}\n\n");
}

/// `RegName::at_slice` — WHICH REGISTER THE i-TH SLICE OF A `_unroll` NAME IS.
///
/// ⭐⭐ THE TAPE AND THE DICTIONARY SPEAK DIFFERENT HALVES OF ONE FACT. A template writes
/// `internal_registers=["p0_unroll", ..]` and that is what the tape carries; the spliced `.smc` body
/// asks the backend for `p0_0`, and `insertReg` is what turns one into the other
/// (`ddc/ddcv1.cpp:3349-3357`). Generating the join from the SAME census that built both sets is why
/// they cannot drift — a hand-written match would be a third place to forget.
///
/// ⛔ `None` FOR A NAME THAT DOES NOT EXPAND, and for a slice past what the template's
/// `max_unroll_factor=` declared. Both are questions with no answer rather than a register 0.
fn codegen_reg_slices(out: &mut String, slices: &BTreeMap<String, Vec<String>>) {
    out.push_str(
        "impl RegName {\n    /// The register this name's `i`-th slice is, or `None` where the \
         name does not\n    /// expand or the template declared fewer slices than `i`.\n    \
         #[must_use]\n    pub const fn at_slice(self, i: u32) -> Option<RegName> {\n        \
         match (self, i) {\n",
    );
    for (name, expansions) in slices {
        for (i, expansion) in expansions.iter().enumerate() {
            let _ = writeln!(
                out,
                "            (Self::{}, {i}) => Some(Self::{}),",
                ident_of(name),
                ident_of(expansion)
            );
        }
    }
    out.push_str("            _ => None,\n        }\n    }\n}\n\n");
}

fn main() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let templates = dir.join("ddl_templates");
    // ⛔ `build.rs` HAS TO NAME ITSELF HERE. Emitting any `rerun-if-changed` REPLACES cargo's default
    // of watching the whole package, so without this line an edit to the generator itself does not
    // rerun it — the build "succeeds" in 0.06s against the tables the previous version emitted, which
    // reads exactly like a change that had no effect.
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=ddl_templates");
    println!("cargo:rerun-if-changed=ddl");

    // Every template, parsed, in a stable order: the emitted tables are keyed by stem and a directory
    // read order would make the output depend on the filesystem.
    let mut stems: Vec<String> = std::fs::read_dir(&templates)
        .unwrap_or_else(|e| panic!("read {}: {e}", templates.display()))
        .map(|entry| entry.expect("a readable directory entry").file_name())
        .filter_map(|name| {
            let name = name.to_string_lossy().into_owned();
            name.strip_suffix(".ddl").map(str::to_owned)
        })
        .collect();
    stems.sort();
    assert!(!stems.is_empty(), "ddl_templates/ holds no .ddl files");

    let parsed: Vec<(String, ast::Module)> = stems
        .iter()
        .map(|stem| {
            let path = templates.join(format!("{stem}.ddl"));
            let text = std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
            let module = parse::parse_module(&text)
                .unwrap_or_else(|e| panic!("parse {}: {e}", path.display()));
            (stem.clone(), module)
        })
        .collect();

    // ⭐ THE STATEMENT-KIND SET IS A CENSUS, NOT A LIST. Every `ddl.*` mnemonic that appears anywhere
    // in any template, so a template naming one this crate has no arm for is a BUILD failure rather
    // than a statement silently walked past.
    let mut mnemonics: BTreeSet<String> = BTreeSet::new();
    for (_, module) in &parsed {
        collect_mnemonics(&module.body, &mut mnemonics);
    }

    // The value sets, censused over the SAME walk the tapes come from — so a spelling that only
    // appears in a program nothing walks cannot enter a table, and one that appears in a walked
    // program cannot be missing from it.
    let programs = walk_every_program(&parsed);
    let census = Census::over(&programs);

    let mut out = String::new();
    codegen_stmt_kind(&mut out, &mnemonics);
    codegen_closed_set(
        &mut out,
        "Unit",
        "WHICH UNIT — the `unit=` a `ddl.unit`, `ddl.compute` or `ddl.opaque` names.",
        &census.units,
    );
    codegen_closed_set(
        &mut out,
        "Memory",
        "WHICH STORAGE a `ddl.allocate` reserves from. `ptxrf`, `ptarf`, `sfplrf`, `pelrf` and \
         `sfpstate` are register FILES; `lx`, `l0` and `l0scale` are MEMORY. A compute may target \
         the first and a transfer moves through the second, so the two must not be one thing.",
        &census.memories,
    );
    codegen_closed_set(
        &mut out,
        "ComputeType",
        "WHICH COMPUTE — the `computetype=` of a `ddl.compute`.",
        &census.compute_types,
    );
    codegen_closed_set(
        &mut out,
        "OpaqueFunc",
        "WHICH OPAQUE BODY — the `op=` of a `ddl.opaque`. It becomes `dataflow.opaque`'s \
         `func_name`, lowercased, and dcc splices the matching `.smc` itself \
         (`SNComputeLowering.cpp:1536-1537`).",
        &census.opaque_funcs,
    );
    codegen_closed_set(
        &mut out,
        "DataConnect",
        "WHICH LINK a unit reads or writes on — the `data_connect=`. Membership is the only \
         question ever asked of it: does this statement take a stick off this connect.",
        &census.data_connects,
    );
    codegen_closed_set(
        &mut out,
        "SyncSignal",
        "WHICH SIGNAL a `ddl.sync` sends or waits on — the `signal_name=`.",
        &census.sync_signals,
    );
    codegen_closed_set(
        &mut out,
        "LoopLabel",
        "A `ddl.loop`'s `label=`, which is what a `ddl.condition` names to say WHICH loop's \
         position it tests.",
        &census.loop_labels,
    );
    codegen_closed_set(
        &mut out,
        "DataType",
        "AN ELEMENT FORMAT — a `ddl.type`'s `data_type=`. This is what the memref and vector types \
         in the emitted DataflowIR are written in.",
        &census.data_types,
    );
    codegen_closed_set(
        &mut out,
        "DimProperty",
        "WHAT A DIMENSION IS FOR — a `ddl.dimension`'s `dim_property=`. A plain dimension states \
         none; these are the padding, window, stride and dilation axes.",
        &census.dim_properties,
    );
    codegen_closed_set(
        &mut out,
        "DatastageProperty",
        "WHICH AXIS A `ddl.get_external_datastage` NAMES — its `property=`.",
        &census.ds_properties,
    );
    codegen_closed_set(
        &mut out,
        "PaddingType",
        "HOW AN ALLOCATION IS PADDED — a `ddl.allocate`'s `padding_type=`.",
        &census.padding_types,
    );
    codegen_closed_set(
        &mut out,
        "AccessPattern",
        "HOW A TRANSFER READS AND WRITES — a `ddl.data_transfer`'s `access_pattern_style=`.",
        &census.access_patterns,
    );
    codegen_closed_set(
        &mut out,
        "ConstName",
        "WHICH CONSTANT — the `name=` of a `ddl.define_constant`, `ddl.get_external_constant` or \
         `ddl.operand_constant`.\n///\n/// ⭐⭐ THIS IS THE JOIN TO THE FRONTEND AND THE ONLY ONE. \
         An external constant states no value; dxp resolves it by scanning `dsc.constantInfo_` for \
         an entry whose `name_` matches and aborts otherwise \
         (`ddl_conversion.cpp:706-719`). So the frontend has to name the same constant this does, \
         and the two agreeing is a build-time fact rather than a bake-time one.",
        &census.const_names,
    );
    codegen_closed_set(
        &mut out,
        "RegName",
        "A REGISTER AN OPAQUE BODY USES — one entry of `internal_registers=` or \
         `input_output_registers=`.\n///\n/// ⛔ THE VALUE BOUND TO IT IS AN ADDRESS, NOT A NAME AND \
         NOT BLANK. `insertReg(regName, startAddress + n, ...)` (`ddc/ddcv1.cpp:3369-3391`) binds \
         each one to its allocation's start address, and `ConstructProgIRHelper.cpp:3999-4014` \
         substitutes that value straight into an instruction's operand field. An empty string \
         passes the caller's `StringAttr` check and then substitutes an EMPTY OPERAND.",
        &census.reg_names,
    );
    codegen_reg_slices(&mut out, &census.reg_slices);
    codegen_closed_set(
        &mut out,
        "ParamKey",
        "WHICH VARIABLE OF AN OPAQUE BODY A `params=` ENTRY FILLS.\n///\n/// ⭐ THESE ARE THE SAME \
         VARIABLES THE `.smc` BODIES LEAVE OPEN, which `smc::Hole` already models as an enum \
         (`prec`, `unroll`, `l0`, `in0`/`in1`/`in2`, `out0`, the `_unroll` slice forms). Registers \
         like `c0` and `t0_0` are STATED in the bodies rather than held open, which is why they are \
         a separate set from this one.",
        &census.param_keys,
    );
    codegen_closed_set(
        &mut out,
        "ParamValue",
        "WHAT A `params=` ENTRY BINDS ITS VARIABLE TO.\n///\n/// ⭐ `lxlu` and `pe` are UNITS — the \
         same vocabulary `Unit` carries — while `no` and `result` say where the body left its \
         output. Closed, four spellings across every vendored template.",
        &census.param_values,
    );
    codegen_closed_set(
        &mut out,
        "Strategy",
        "WHICH DIRECTION A `ddl.datastage`'s EXTENT IS OPTIMISED — its `strategy=`.\n///\n/// \u{2b50} IT          RESOLVES WHAT THE CONSTRAINTS LEAVE OPEN. Most datastages are pinned to one value by a          `ddl.datastage_constraint`; where a choice remains (`values=[\"1\",\"2\",\"4\"]`,          `bmm.ddl:143`) this says which end of it to take.",
        &census.strategies,
    );
    codegen_closed_set(
        &mut out,
        "Template",
        "WHICH `.ddl` FILE A PROGRAM WAS WALKED FROM.\n///\n/// ⭐⭐ IT IS PART OF A DIM'S IDENTITY, \
         which is why this is an enum and not the file name as text. `%outer_dim#3` is `MB` in \
         `summeanmaxexx2.ddl` (`%outer_dim:5 // X, Y, I, J, MB`) and `J` in \
         `quantization_double_pad.ddl` (`%outer_dim:4 // X, Y, I, MB` with `%j_dim` separate) — the \
         SAME spelling naming two different axes. A dim mapping keyed on the spelling alone is \
         therefore wrong for one of them, and keyed on `(Template, spelling)` it is a lookup.",
        &stems.iter().cloned().collect(),
    );
    codegen_support_types(&mut out);
    check_every_attribute_is_modelled(&programs);
    let binds_per_template: BTreeMap<String, BTreeSet<String>> = parsed
        .iter()
        .map(|(stem, module)| {
            (
                stem.clone(),
                dataflow::op_binds(&module.body)
                    .into_iter()
                    .map(|(bound, _)| bound.to_owned())
                    .collect(),
            )
        })
        .collect();
    check_every_operand_resolves(&programs, &binds_per_template);
    codegen_programs(&mut out, &programs);
    check_every_opaque_hole_is_filled(&programs);
    check_every_op_func_in_scope_has_a_program(&programs);
    codegen_op_func(&mut out, &programs);

    // 🛑 THE LEDGER OF WHAT THE WALK FLATTENED. A `ddl.if` whose condition is a LOOP POSITION keeps
    // BOTH arms, because which one runs depends on an extent the walk does not have. Those arms are
    // MUTUALLY EXCLUSIVE, so emitting them as siblings puts two versions of one statement in the
    // program — and a value produced in one arm is then read from the other, which is not in scope.
    println!(
        "cargo:warning=walk flattened {} exclusive branches, {} of them with both arms populated",
        dataflow::flattened_exclusive_branches(),
        dataflow::flattened_with_both_arms()
    );

    // 📏 HOW OFTEN A LOOP SITS INSIDE AN ARM — the case depth-plus-guard cannot express, and which
    // emitting loops-before-ifs would invert.
    let mut loop_inside_arm = 0usize;
    let mut arm_inside_loop = 0usize;
    for program in &programs {
        for path in &program.paths {
            let mut seen_arm = false;
            let mut seen_loop = false;
            for step in path {
                match step {
                    dataflow::Enclosing::Arm(_) => {
                        seen_arm = true;
                        if seen_loop {
                            arm_inside_loop += 1;
                        }
                    }
                    dataflow::Enclosing::Loop { .. } => {
                        seen_loop = true;
                        if seen_arm {
                            loop_inside_arm += 1;
                        }
                    }
                }
            }
        }
    }
    println!(
        "cargo:warning=nesting: {loop_inside_arm} loops sit inside an undecided arm, \
         {arm_inside_loop} arms sit inside a loop"
    );

    let dest =
        Path::new(&std::env::var("OUT_DIR").expect("cargo sets OUT_DIR")).join("generated.rs");
    std::fs::write(&dest, out).unwrap_or_else(|e| panic!("write {}: {e}", dest.display()));
}

fn collect_mnemonics(ops: &[ast::Operation], into: &mut BTreeSet<String>) {
    for op in ops {
        into.insert(op.name.clone());
        collect_mnemonics(&op.body, into);
        if let Some(other) = &op.else_body {
            collect_mnemonics(other, into);
        }
    }
}

/// One (template, op-bind) pair's walk: the statements that op-func runs, in execution order.
struct Program {
    /// The `.ddl` stem it came from.
    stem: String,
    /// `opFuncName=` — what a scratchy op names.
    op_func: String,
    /// The SSA name of the `ddl.operation_bind`, which is what distinguishes two binds sharing an
    /// op-func (`%stradd_op` and `%stradd2_op` are both `stridedadd`, and they read different
    /// operands — `bmm.ddl:79-80`).
    bind: String,
    /// The statements, in the order the unit executes them.
    stmts: Vec<Stmt>,
    /// Every SSA name this program's walk binds, in id order.
    names: Vec<String>,
    /// Per statement, the name ids it binds.
    results: Vec<Vec<u16>>,
    /// Per statement, its operands in source order.
    operands: Vec<Vec<OperandRef>>,
    /// Per statement, every construct enclosing it, outermost first and in order.
    paths: Vec<Vec<dataflow::Enclosing>>,
    /// WHICH OF THE OP'S TENSORS EACH TEMPLATE NAME IS, by position — the join between the template
    /// and the op scratchy holds. See [`Program::role_of`].
    roles: Vec<(u16, String)>,
    /// ⭐⭐ THE DATA FORMATS THIS BIND ADMITS, as `DataType` idents — resolved from the bind's
    /// `[%type_*]` list through this template's own `ddl.type` declarations.
    ///
    /// ⛔ EMPTY MEANS "ANY", which is dxp's own reading: `if (!types.empty())` guards the whole
    /// format check (`ddl_conversion.cpp:2137`), so a bind that states no types matches every one.
    formats: Vec<String>,
}

/// ONE OPERAND OF A STATEMENT, resolved to name ids.
enum OperandRef {
    /// `%name` or `%name#N`.
    One(u16),
    /// `[%a, %b, ..]`, possibly empty.
    List(Vec<u16>),
    /// An `ddl.operation_bind` of this template that THIS walk did not activate.
    OtherBind,
}

/// One walked statement, with the attributes DataflowIR reads off it.
struct Stmt {
    /// The `ddl.*` mnemonic.
    mnemonic: String,
    /// How many `ddl.loop`s enclose it.
    depth: u16,
    /// The attributes, as the template wrote them. Rendering them is [`render_attrs`]'s job.
    op: ast::Operation,
}

/// ⭐⭐ THE SSA NAMES OF ONE PROGRAM, INTERNED — so the operand graph is const indices, not strings.
///
/// A template's substance is its SSA graph: `ddl.data_transfer(%src_inp_lxl0, [%dst_inp_lxl0])` is a
/// transfer between the two units those names were bound to, and without the names there is no way
/// to know which. Carrying them as `&'static str` would put text back in the islands and make every
/// lookup a comparison.
///
/// ⛔ SO THEY ARE INTERNED PER PROGRAM AT BUILD TIME. A [`NameId`] is an index into the program's own
/// name table, minted here where the text still exists; `src/` sees indices and never a string.
///
/// ⛔ AND `%wrd:4` BINDS FOUR NAMES, NOT ONE. A multi-result declaration is written with a colon and
/// referenced with a hash — `%wrd:4` declares `%wrd#0 .. %wrd#3` (`ddl/parse.rs:256-279`) — so a
/// table that interned the declaration's own text would leave every reference to it dangling.
#[derive(Default)]
struct Names {
    /// The interned texts, in id order.
    order: Vec<String>,
    /// Text to id.
    index: BTreeMap<String, u16>,
}

impl Names {
    fn intern(&mut self, text: &str) -> u16 {
        if let Some(id) = self.index.get(text) {
            return *id;
        }
        let id = u16::try_from(self.order.len())
            .unwrap_or_else(|_| panic!("a template binds more than 65535 SSA names"));
        self.order.push(text.to_owned());
        self.index.insert(text.to_owned(), id);
        id
    }

    /// The ids a RESULT declaration binds.
    fn declare(&mut self, text: &str) -> Vec<u16> {
        match text.split_once(':') {
            // `%wrd:4` — four values, referenced as `%wrd#0` .. `%wrd#3`.
            Some((base, count)) => {
                let count: u32 = count
                    .parse()
                    .unwrap_or_else(|e| panic!("a result count `{count}` is not a number: {e}"));
                let ids: Vec<u16> = (0..count)
                    .map(|i| self.intern(&format!("{base}#{i}")))
                    .collect();
                // ⛔ `%x:1` IS ALSO `%x`, AND MISSING THAT LEFT A REAL REFERENCE DANGLING.
                // `inter_slice_transpose.ddl:7` declares `%asdin:1 = ddl.dimension{}` and `:11`
                // references it bare as `%asdin`. A count of one is one value, so both spellings name
                // it; for a count above one the bare form is ambiguous and MLIR does not admit it, so
                // the alias is registered only here.
                if let (1, Some(first)) = (count, ids.first()) {
                    self.index.insert(base.to_owned(), *first);
                }
                ids
            }
            // ⭐ A SINGLE RESULT IS REACHABLE BOTH WAYS. `%psum_end` is referenced bare, but a
            // one-result op's value is also legitimately `%psum_end#0`, and both spellings occur.
            // Interning the alias to the SAME id is what makes the two resolve to one value.
            None => {
                let id = self.intern(text);
                let alias = format!("{text}#0");
                self.index.insert(alias, id);
                vec![id]
            }
        }
    }

    /// The id a REFERENCE names, or `None` where nothing in this program's walk bound it.
    fn reference(&self, text: &str) -> Option<u16> {
        self.index.get(text).copied()
    }
}

// ───────────────────────────────────────────────────────────────────────────────────────────────
// Reading attributes off a statement.
// ───────────────────────────────────────────────────────────────────────────────────────────────

fn attr<'a>(op: &'a ast::Operation, key: &str) -> Option<&'a ast::AttrValue> {
    op.attrs.iter().find(|(k, _)| k == key).map(|(_, v)| v)
}

fn attr_str<'a>(op: &'a ast::Operation, key: &str) -> Option<&'a str> {
    match attr(op, key)? {
        ast::AttrValue::String(s) => Some(s.as_str()),
        _ => None,
    }
}

fn attr_int(op: &ast::Operation, key: &str) -> Option<i64> {
    match attr(op, key)? {
        ast::AttrValue::Int(v) | ast::AttrValue::TypedInt { value: v, .. } => Some(*v),
        _ => None,
    }
}

fn attr_bool(op: &ast::Operation, key: &str) -> Option<bool> {
    match attr(op, key)? {
        ast::AttrValue::Bool(b) => Some(*b),
        _ => None,
    }
}

fn attr_strs<'a>(op: &'a ast::Operation, key: &str) -> Vec<&'a str> {
    match attr(op, key) {
        Some(ast::AttrValue::List(items)) => items
            .iter()
            .filter_map(|item| match item {
                ast::AttrValue::String(s) => Some(s.as_str()),
                _ => None,
            })
            .collect(),
        _ => Vec::new(),
    }
}

fn attr_dict<'a>(op: &'a ast::Operation, key: &str) -> Vec<(&'a str, &'a str)> {
    match attr(op, key) {
        Some(ast::AttrValue::Dict(entries)) => entries
            .iter()
            .filter_map(|(k, v)| match v {
                ast::AttrValue::String(s) => Some((k.as_str(), s.as_str())),
                _ => None,
            })
            .collect(),
        _ => Vec::new(),
    }
}

// ───────────────────────────────────────────────────────────────────────────────────────────────
// The census.
// ───────────────────────────────────────────────────────────────────────────────────────────────

/// EVERY DISTINCT SPELLING THE WALKED PROGRAMS NAME, per attribute that has a closed set.
#[derive(Default)]
struct Census {
    units: BTreeSet<String>,
    memories: BTreeSet<String>,
    compute_types: BTreeSet<String>,
    opaque_funcs: BTreeSet<String>,
    data_connects: BTreeSet<String>,
    sync_signals: BTreeSet<String>,
    loop_labels: BTreeSet<String>,
    data_types: BTreeSet<String>,
    dim_properties: BTreeSet<String>,
    ds_properties: BTreeSet<String>,
    padding_types: BTreeSet<String>,
    access_patterns: BTreeSet<String>,
    const_names: BTreeSet<String>,
    strategies: BTreeSet<String>,
    reg_names: BTreeSet<String>,
    /// WHICH REGISTERS A `_unroll` NAME EXPANDS TO, in slice order — `p0_unroll` -> `[p0_0, p0_1]`.
    /// The template writes the first, the `.smc` body asks for the rest.
    reg_slices: BTreeMap<String, Vec<String>>,
    param_keys: BTreeSet<String>,
    param_values: BTreeSet<String>,
}

impl Census {
    /// NAMES THE FRONTEND STATES THAT NO `.ddl` RESOLVES.
    ///
    /// ⭐ THE TABLE AND THE TEMPLATES ARE NOT THE SAME SET. `constantInfo_` is what the frontend
    /// hands dxp; `ddl.get_external_constant` is what a template asks for BY NAME. Every asked-for
    /// name must be in the table, but the table may hold entries nothing asks for — dxp reads those
    /// elsewhere, and they still have to be emitted for the JSON to match IBM's own goldens.
    ///
    /// ⛔ SO THIS IS NOT A LOOPHOLE FOR MISSING SPELLINGS. One entry, cited: `dontSplatOutput`
    /// appears as a `name_` in `ddc/ddl_templates/test/sdsc_{sqrt,tanh,min,max,sigmoid}.json` and in
    /// no `.ddl`. It sits at index 0 of the SFP constant table, so omitting it would renumber every
    /// entry after it.
    const FRONTEND_ONLY: &'static [&'static str] = &["dontSplatOutput"];

    fn over(programs: &[Program]) -> Census {
        let mut census = Census::default();
        for name in Self::FRONTEND_ONLY {
            census.const_names.insert((*name).to_owned());
        }
        for program in programs {
            for stmt in &program.stmts {
                let op = &stmt.op;
                if let Some(unit) = attr_str(op, "unit") {
                    census.units.insert(unit.to_owned());
                }
                // `ddl.sync`'s `units=` is a LIST of the same vocabulary, not a different one.
                for unit in attr_strs(op, "units") {
                    census.units.insert(unit.to_owned());
                }
                // ⛔ A VIA IS A UNIT TOO. `vias=["pe"]` names the hop a transfer passes through, and
                // it is drawn from this same set — so censusing it separately would let a spelling
                // that only ever appears as a via be missing from `Unit`.
                for via in attr_strs(op, "vias") {
                    census.units.insert(via.to_owned());
                }
                if let Some(strategy) = attr_str(op, "strategy") {
                    census.strategies.insert(strategy.to_owned());
                }
                if let Some(memory) = attr_str(op, "memory") {
                    census.memories.insert(memory.to_owned());
                }
                if let Some(computetype) = attr_str(op, "computetype") {
                    census.compute_types.insert(computetype.to_owned());
                }
                if let Some(func) = attr_str(op, "op") {
                    census.opaque_funcs.insert(func.to_owned());
                }
                // ⭐⭐ A CONSTANT'S NAME IS THE JOIN TO THE FRONTEND, so it is a closed set like any
                // other. `ddl_conversion.cpp:706-719` looks a `ddl.get_external_constant` up in
                // `dsc.constantInfo_` BY NAME and aborts the compile if no entry matches — so the
                // spelling here and the spelling the frontend states have to be the same value, and
                // an enum is how that is checked at build time rather than at bake time.
                if matches!(
                    stmt.mnemonic.as_str(),
                    "ddl.define_constant" | "ddl.get_external_constant" | "ddl.operand_constant"
                ) && let Some(name) = attr_str(op, "name")
                {
                    census.const_names.insert(name.to_owned());
                }
                // ⭐⭐ AN OPAQUE'S DICTIONARIES ARE TOKENS, NOT TEXT. Every register name and every
                // parameter key/value comes out of THESE templates, which this build script already
                // walks — so the closed set is knowable here and `Op::Opaque` has no business
                // holding `Vec<(String, String)>`. The parameter keys are the same variables the
                // `.smc` bodies leave open, which `smc::Hole` already models as an enum.
                // ⛔⛔ THE EXPANDED NAMES ARE THE REAL SET, exactly as for the parameter keys below.
                // `p0_unroll` never reaches a dictionary: `insertReg` writes `p0_0 .. p0_{unroll-1}`
                // (`ddc/ddcv1.cpp:3349-3357`), and those are the names the `.smc` bodies carry —
                // `gelufwd` asks the backend for `p0_0`, not for `p0_unroll`. Censusing the raw
                // name built a `RegName` that could not SPELL a single register we have to emit.
                let max_unroll = attr_int(op, "max_unroll_factor").unwrap_or(1).max(1);
                //
                // ⭐ BOTH FORMS ARE NEEDED AND THEY ARE DIFFERENT THINGS. The TAPE carries what the
                // template wrote (`RegName::P0Unroll`, with `OpaqueReg::unrolled` saying it expands);
                // the DICTIONARY carries what the body asks for (`RegName::P00`). `RegName::at_slice`
                // below is the join, generated from this same census so the two cannot drift.
                let mut census_reg = |reg: &str| {
                    census.reg_names.insert(reg.to_owned());
                    if let Some(pos) = reg.find("_unroll") {
                        let base = &reg[..=pos];
                        for i in 0..max_unroll {
                            census.reg_names.insert(format!("{base}{i}"));
                        }
                        census.reg_slices.insert(
                            reg.to_owned(),
                            (0..max_unroll).map(|i| format!("{base}{i}")).collect(),
                        );
                    }
                };
                for reg in attr_strs(op, "internal_registers") {
                    census_reg(reg);
                }
                for reg in attr_strs(op, "input_output_registers") {
                    census_reg(reg);
                }
                // ⛔ THE EXPANDED KEYS ARE THE REAL SET. `in0_unroll` never reaches the emitted
                // dictionary — it expands to `in0_0 .. in0_{max_unroll-1}`
                // (`ddl_conversion.cpp:1605-1615`), and those are the names the `.smc` body's holes
                // carry (`smc::Hole::InputAtSlice`). Censusing the raw key would build an enum
                // missing every variant the emitter actually writes.
                for (key, value) in attr_dict(op, "params") {
                    census.param_values.insert(value.to_owned());
                    match key.find("_unroll") {
                        None => {
                            census.param_keys.insert(key.to_owned());
                        }
                        Some(pos) => {
                            let base = &key[..=pos];
                            for i in 0..max_unroll {
                                census.param_keys.insert(format!("{base}{i}"));
                            }
                        }
                    }
                }
                // ⭐⭐ TWO KEYS ARE THE EMITTER'S, AND THE TEMPLATES NEVER MENTION THEM. `prec`
                // comes from the OP's data format and `unroll` from its internal scratch
                // (`ddc/ddcv1.cpp:3343`, `:3393-3397`) — neither is a template fact, so censusing
                // `params=` alone builds an enum that cannot SPELL what every opaque needs. Every
                // one of them then reaches dcc as "OPAQUE was not provided with value for variable
                // prec" (`ConstructProgIRHelper.cpp:4006`).
                //
                // ⛔ THEY ARE SEEDED HERE RATHER THAN HAND-WRITTEN INTO THE ENUM because the enum
                // is generated; a variant added by hand would be erased by the next build.
                census.param_keys.insert("prec".to_owned());
                census.param_keys.insert("unroll".to_owned());
                // ⭐ AND `prec` HAS EXACTLY TWO VALUES — `"fp32"` for an IEEE_FP32 op and `"fp16"`
                // for every other (`ddcv1.cpp:3393-3396`). It is a two-way split on the format, not
                // the format's own name, so bf16 and int8 ops are all `fp16` here.
                census.param_values.insert("fp16".to_owned());
                census.param_values.insert("fp32".to_owned());
                // ⭐ AND `unroll` IS A COUNT. `param_map_["unroll"] = std::to_string(unrollFactor)`
                // (`ddcv1.cpp:3343`), which `ddcv1.cpp:3340` requires to be a POWER OF TWO and
                // `roundDownUnrollFactor` caps at 8 for a non-reduction — a closed set of four.
                for factor in ["1", "2", "4", "8"] {
                    census.param_values.insert(factor.to_owned());
                }
                if let Some(connect) = attr_str(op, "data_connect") {
                    census.data_connects.insert(connect.to_owned());
                }
                for connect in attr_strs(op, "input_data_connects") {
                    census.data_connects.insert(connect.to_owned());
                }
                for connect in attr_strs(op, "output_data_connects") {
                    census.data_connects.insert(connect.to_owned());
                }
                if let Some(signal) = attr_str(op, "signal_name") {
                    census.sync_signals.insert(signal.to_owned());
                }
                if let Some(label) = attr_str(op, "label") {
                    census.loop_labels.insert(label.to_owned());
                }
                // A `ddl.condition` names the loop it tests rather than declaring one, so its label
                // has to be in the same set — otherwise a condition could name a loop the enum has no
                // variant for.
                if let Some(label) = attr_str(op, "loop_label") {
                    census.loop_labels.insert(label.to_owned());
                }
                if let Some(ty) = attr_str(op, "data_type") {
                    census.data_types.insert(ty.to_owned());
                }
                if let Some(property) = attr_str(op, "dim_property") {
                    census.dim_properties.insert(property.to_owned());
                }
                // ⛔ ONLY FROM `ddl.get_external_datastage`. `ddl.constraint` states a `property=`
                // too, and that one feeds the tile search this crate does not perform — censusing
                // both into one enum would put spellings in it that no emitted statement can name.
                if stmt.mnemonic == "ddl.get_external_datastage"
                    && let Some(property) = attr_str(op, "property")
                {
                    census.ds_properties.insert(property.to_owned());
                }
                for padding in attr_strs(op, "padding_type") {
                    census.padding_types.insert(padding.to_owned());
                }
                for pattern in attr_strs(op, "access_pattern_style") {
                    census.access_patterns.insert(pattern.to_owned());
                }
            }
        }
        census
    }
}

fn walk_every_program(parsed: &[(String, ast::Module)]) -> Vec<Program> {
    let mut programs = Vec::new();
    for (stem, module) in parsed {
        let Some(flow) = find_dataflow(&module.body) else {
            // A template with no `ddl.dataflow` declares types for another one to use.
            continue;
        };
        // Every `ddl.operation_bind` this template declares — what an unresolved reference is checked
        // against, since a walk for one bind legitimately does not bind the others.
        let binds: BTreeSet<String> = dataflow::op_binds(&module.body)
            .into_iter()
            .map(|(bound, _)| bound.to_owned())
            .collect();

        for (bind, op_func) in dataflow::op_binds(&module.body) {
            let walked = dataflow::statements_for(&module.body, flow, &[bind]);
            let stmts: Vec<Stmt> = walked
                .iter()
                .map(|w| Stmt {
                    mnemonic: w.op.name.clone(),
                    depth: w.depth,
                    op: w.op.clone(),
                })
                .collect();

            // ⛔ TWO PASSES, DECLARATIONS FIRST. An operand may name a value the statement above it
            // bound, and interning on encounter would give a forward reference a fresh id — two ids
            // for one value, which is an SSA graph with an edge missing.
            let mut names = Names::default();
            let results: Vec<Vec<u16>> = stmts
                .iter()
                .map(|stmt| {
                    stmt.op
                        .results
                        .iter()
                        .flat_map(|text| names.declare(text))
                        .collect()
                })
                .collect();
            let operands: Vec<Vec<OperandRef>> = stmts
                .iter()
                .map(|stmt| resolve_operands(&stmt.op, &names, &binds))
                .collect();
            let names_for_role = names.index.clone();
            let paths: Vec<Vec<dataflow::Enclosing>> =
                walked.iter().map(|w| w.path.clone()).collect();

            // ⭐⭐ THE JOIN. `ddl.operation_bind([types], [inputs], [outputs], [internals])`
            // (`bmm.ddl:52-59`) states BY POSITION which of the op's tensors each template name is:
            // for `batchmatmulfp8` the inputs are `[%inptensor_fp8, %kertensor_fp8]` and the output
            // `[%outtensor]`. scratchy's own operand order is the same, so input 0 here is the op's
            // first tensor there — which is how an address reaches the right view.
            let mut roles: Vec<(u16, String)> = Vec::new();
            let mut formats: Vec<String> = Vec::new();
            if let Some(bound) = dataflow::bind_tensors_named(&module.body, bind) {
                formats = bound
                    .types
                    .iter()
                    .map(|name| declared_data_type(&module.body, name, stem, op_func))
                    .collect();
                let mut note = |names: &[String], role: &str, base: usize| {
                    for (index, name) in names.iter().enumerate() {
                        if let Some(id) = names_for_role.get(name.as_str()) {
                            roles.push((*id, format!("{role}({})", base + index)));
                        }
                    }
                };
                note(bound.inputs, "Input", 0);
                note(bound.outputs, "Output", 0);
                note(bound.internals, "Internal", 0);
                // ⛔ ONE NAME MAY HOLD TWO ROLES. `argmax` binds `%outtensor` as both input 1 and
                // output 0 — an in-place op — so the list is a MULTIMAP and `role_of` returning the
                // first is a choice. Sorted so which one that is follows the bind's own order
                // (inputs, then outputs, then internals) rather than a hash.
                roles.sort_by_key(|(id, _)| *id);
            }

            programs.push(Program {
                stem: stem.clone(),
                op_func: op_func.to_owned(),
                bind: bind.trim_start_matches('%').to_owned(),
                stmts,
                names: names.order,
                results,
                operands,
                paths,
                roles,
                formats,
            });
        }
    }
    programs
}

/// THE `data_type=` OF A `ddl.type` THE BIND NAMES — `%type_fp16` to `Sen169Fp16`.
///
/// ⛔ A NAME THIS TEMPLATE NEVER DECLARES IS A BUILD FAILURE, not an omission from the format list.
/// An empty format list means "admits every format" (`ddl_conversion.cpp:2137`), so silently
/// dropping an unresolved type would WIDEN the bind — the fp32-only `exp` would start admitting
/// fp16, which is the exact defect this resolution exists to end.
fn declared_data_type(module: &[ast::Operation], name: &str, stem: &str, op_func: &str) -> String {
    for op in module {
        if op.name != "ddl.type" || !op.results.iter().any(|result| result == name) {
            continue;
        }
        return ident_of(expect_str(op, "data_type", "ddl.type"));
    }
    panic!(
        "`{stem}.ddl`'s bind for `{op_func}` names the type `{name}`, which no `ddl.type` in that \
         template declares — the format list would silently widen to \"any\""
    )
}

/// ONE STATEMENT'S OPERANDS, RESOLVED.
///
/// ⛔ AN UNRESOLVED REFERENCE IS DROPPED, AND [`check_every_operand_resolves`] IS WHY THAT IS SAFE:
/// it counts them and fails the build if any survive, so this cannot quietly thin an operand list.
fn resolve_operands(
    op: &ast::Operation,
    names: &Names,
    binds: &BTreeSet<String>,
) -> Vec<OperandRef> {
    let one = |text: &String| match names.reference(text) {
        Some(id) => OperandRef::One(id),
        // ⭐ AN OP-BIND THIS WALK DID NOT ACTIVATE. `ddl.condition_or(%bmm_fp16_op, %mm_fp16_op, ..)`
        // names every matmul bind the template declares, and a walk for one of them correctly
        // excludes the other twenty (`ddl/dataflow.rs`'s `truth_of`). That is the walk working, not a
        // missing declaration — so it is a variant rather than a silent drop.
        None if binds.contains(base_of(text)) => OperandRef::OtherBind,
        None => panic!(
            "operand `{text}` of a {} names no value this walk binds and is not an operation_bind; \
             the declaration it points at was not spliced",
            op.name
        ),
    };
    op.operands
        .iter()
        .map(|operand| match operand {
            ast::Operand::Ref(text) => one(text),
            ast::Operand::RefList(texts) => OperandRef::List(
                texts
                    .iter()
                    .filter_map(|text| names.reference(text))
                    .collect(),
            ),
        })
        .collect()
}

/// `%wrd#0` → `%wrd`. An op-bind is referenced bare, but the same helper serves both.
fn base_of(text: &str) -> &str {
    text.split_once('#').map_or(text, |(base, _)| base)
}

/// 🛑 EVERY OPERAND IN A **LIST** MUST NAME A VALUE THIS PROGRAM'S WALK BOUND.
///
/// A dangling reference means the walk did not yield the statement that declares it — the template
/// was not fully spliced, and the transfer or compute that reads it has an operand pointing at
/// nothing. That is invisible in the emitted IR: the op simply has one fewer operand.
///
/// ⛔ THE BARE-REFERENCE CASE IS CHECKED IN [`resolve_operands`], WHICH PANICS THERE. This covers the
/// LIST form, where `filter_map` would otherwise thin the list silently — the shape that matters most,
/// since `ddl.data_transfer(%src, [%dsts])`'s destination list IS the multicast.
///
/// ⭐ AN OP-BIND THIS WALK DID NOT ACTIVATE IS NOT DANGLING. A `ddl.condition_or` over every matmul
/// bind names twenty values a walk for one of them does not bind, and that is the walk resolving the
/// condition rather than dropping a declaration — 5379 of them across the vendored set.
fn check_every_operand_resolves(
    programs: &[Program],
    binds_per_template: &BTreeMap<String, BTreeSet<String>>,
) {
    let mut dangling: BTreeMap<String, usize> = BTreeMap::new();
    for program in programs {
        let empty = BTreeSet::new();
        let binds = binds_per_template.get(&program.stem).unwrap_or(&empty);
        let mut names = Names::default();
        for stmt in &program.stmts {
            for text in &stmt.op.results {
                names.declare(text);
            }
        }
        for stmt in &program.stmts {
            for operand in &stmt.op.operands {
                let ast::Operand::RefList(list) = operand else {
                    continue;
                };
                for text in list {
                    if names.reference(text).is_none() && !binds.contains(base_of(text)) {
                        *dangling
                            .entry(format!(
                                "{}/{}: {} names `{text}` in an operand list",
                                program.stem, program.bind, stmt.mnemonic
                            ))
                            .or_default() += 1;
                    }
                }
            }
        }
    }
    assert!(
        dangling.is_empty(),
        "these operands name SSA values no statement of their own program binds, so the walk did \
         not splice the declaration they point at ({} distinct):\n{}",
        dangling.len(),
        dangling
            .keys()
            .take(20)
            .map(|what| format!("  {what}"))
            .collect::<Vec<_>>()
            .join("\n")
    );
}

fn find_dataflow(ops: &[ast::Operation]) -> Option<&[ast::Operation]> {
    ops.iter()
        .find(|op| op.name == "ddl.dataflow")
        .map(|op| op.body.as_slice())
}

fn codegen_stmt_kind(out: &mut String, mnemonics: &BTreeSet<String>) {
    out.push_str(
        "/// WHICH `ddl.*` STATEMENT THIS IS — the census of every mnemonic the vendored templates\n\
         /// state. A closed set with no `_` arm, so a template that grows the set breaks the build\n\
         /// rather than being walked past.\n\
         #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]\n\
         pub enum StmtKind {\n",
    );
    for mnemonic in mnemonics {
        let _ = writeln!(out, "    /// `{mnemonic}`.\n    {},", variant_of(mnemonic));
    }
    out.push_str("}\n\nimpl StmtKind {\n    /// The mnemonic this variant was generated from.\n    pub const fn mnemonic(self) -> &'static str {\n        match self {\n");
    for mnemonic in mnemonics {
        let _ = writeln!(
            out,
            "            Self::{} => \"{mnemonic}\",",
            variant_of(mnemonic)
        );
    }
    out.push_str("        }\n    }\n}\n\n");
}

/// The types the generated tables are built out of, which are hand-written rather than censused
/// because their shape is stated by the model rather than by a spelling.
fn codegen_support_types(out: &mut String) {
    out.push_str(
        r#"/// THE HOP A TRANSFER PASSES THROUGH — `vias=`.
///
/// ⛔⛔ THE PATH IS NOT DERIVABLE FROM THE ENDPOINTS, which is why the attribute exists.
/// `constructDataTransfer` takes the first hop as
/// `to_unit = dst.via_.empty() ? dst.loc_.unit_ : dst.via_.front()`
/// (`SNTransferLowering.cpp:2731-2732`), so an LX-to-PT transfer stated `vias=["pe"]` is a send to
/// the PE and NOT to the PT. Dropping it makes every transfer look direct.
///
/// ⛔ AND ONLY A DESTINATION MAY STATE ONE: "Vias can only be speficied for data transfer
/// destinations" (`ddl_conversion.cpp:1035-1039`).
///
/// ⭐ NOT A LIST, BECAUSE THE VENDORED SET IS NOT ONE. Every `vias=` in the templates has exactly one
/// element — `["pe"]` 31 times and `["sfp"]` 9 times — and the C++ reads only `front()`. A `&[Unit]`
/// here would be a shape nothing constructs and a second hop nobody could act on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Via {
    /// No `vias=`: the transfer goes straight to the destination unit.
    Direct,
    /// Through this unit first.
    Through(Unit),
}

/// HOW MANY BUFFERS AN ALLOCATION RESERVES — `num_buffers=`.
///
/// ⛔ THE BUFFER COUNT IS PART OF THE SIZE, so it travels with the memory rather than being looked
/// up later: `getBufferCapacityForNode` multiplies the tile by it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Buffers {
    /// No `num_buffers=` — the allocation is a single buffer.
    Single,
    /// `num_buffers=-1`, the only value the vendored templates state: TWO for the size, and then the
    /// WHOLE memory for the reservation (`ddcv1.cpp:226-227`, `:318-329`). Two facts, one spelling,
    /// which is why this is a named variant and not the integer.
    SizeTwoReserveAll,
}

/// A `ddl.compute`'s `mode=` — the general SRC1/IMM field (`dsc2.h`'s `InstrAttribute::mode_`).
///
/// ⭐ A NEWTYPE OVER THE MODEL'S OWN `int`, not an enum: the vendored values are 0, 1, 6, 7, 8, 9, 10
/// and 12, and what each means is per computetype rather than a vocabulary of its own. Naming eight
/// variants would invent a meaning the templates do not state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Mode(pub i64);

/// HOW FAR AN OPAQUE BODY UNROLLS — `max_unroll_factor=`.
///
/// ⛔ IT SELECTS THE BODY, so it is not decoration: `op="RECIPROCAL"` at unroll 1 is
/// `reciprocalx1.smc` and at 2 is `reciprocalx2.smc`.
///
/// ⛔ AND IT MUST BE 1 WHERE THERE ARE NO INTERNAL REGISTERS: "No need unrolling without internal
/// registers" (`ddl_conversion.cpp:1618-1622`) — checked at emission time here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MaxUnroll(pub u8);

/// ONE OPAQUE REGISTER: the name the `.smc` body holds, and which allocation supplies its address.
///
/// ⛔⛔ THE ADDRESS IS NOT KNOWN HERE, AND THAT IS THE POINT. `dataflow.opaque`'s dictionaries map a
/// register name to `"R<n>"`, where `n` is the START STICK of the allocation backing it
/// (`ddcv1.cpp:3369-3391`). That address is scratchy's, and it arrives with the op — so the table
/// carries the NAME and the emitter substitutes the register once the allocation is placed. A table
/// that tried to state `"R4"` would be stating an address the template does not know.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpaqueReg {
    /// The name the body refers to it by.
    pub name: RegName,
    /// Whether the name ends in `_unroll`, and so expands across [`MaxUnroll`] with an incrementing
    /// register (`ddcv1.cpp:3350-3356`).
    pub unrolled: bool,
}

"#,
    );
}

/// One `ResolvedCond` as a const expression.
fn render_cond(cond: &dataflow::ResolvedCond) -> String {
    match cond {
        dataflow::ResolvedCond::Position { loop_depth, place } => format!(
            "Cond::Position {{ loop_depth: {loop_depth}, place: Place::{} }}",
            match place {
                dataflow::Place::First => "First",
                dataflow::Place::Last => "Last",
            }
        ),
        dataflow::ResolvedCond::All(parts) => format!(
            "Cond::All(&[{}])",
            parts.iter().map(render_cond).collect::<Vec<_>>().join(", ")
        ),
        dataflow::ResolvedCond::Any(parts) => format!(
            "Cond::Any(&[{}])",
            parts.iter().map(render_cond).collect::<Vec<_>>().join(", ")
        ),
        dataflow::ResolvedCond::Not(inner) => {
            format!("Cond::Not(&{})", render_cond(inner))
        }
    }
}

/// One enclosing construct as a const expression.
fn render_enclosing(step: &dataflow::Enclosing) -> String {
    match step {
        dataflow::Enclosing::Loop { stmt } => format!("Enclosing::Loop {{ stmt: {stmt} }}"),
        dataflow::Enclosing::Arm(arm) => format!(
            "Enclosing::Arm {{ cond: {}, then_arm: {} }}",
            render_cond(&arm.cond),
            matches!(arm.arm, dataflow::Arm::Then)
        ),
    }
}

fn codegen_programs(out: &mut String, programs: &[Program]) {
    out.push_str(
        r#"/// WHAT A STATEMENT CARRIES — one variant per `ddl.*` kind that has attributes DataflowIR reads.
///
/// ⛔ THE DECLARATION KINDS ARE FIELDLESS ON PURPOSE. A `ddl.dimension` or a `ddl.tensor` is asked
/// one question by the emitter — WHICH kind bound this name — because that is what separates a
/// register from a port. Giving them fields would be carrying values for transit only.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Attrs {
    /// `ddl.unit` — the most common statement in the templates by a wide margin.
    Unit {
        /// `unit=`.
        unit: Unit,
        /// `data_connect=`.
        data_connect: DataConnect,
        /// `vias=`.
        via: Via,
        /// `stick_replicated_dim_offset_elements=` — an offset, IN ELEMENTS, into the stick-replicated
        /// dimension. Already the unit DataflowIR addresses in (`Dataflow.td:250`), so it needs no
        /// conversion on the way into a view.
        stick_offset: Option<i64>,
    },
    /// `ddl.allocate` — a register FILE or a MEMORY, named by `memory=`.
    Allocate {
        /// `memory=`.
        memory: Memory,
        /// `num_buffers=`.
        buffers: Buffers,
        /// `padding_type=`.
        padding: &'static [PaddingType],
        /// `replication=` — how many times the allocation is replicated.
        replication: Option<i64>,
    },
    /// `ddl.compute`.
    Compute {
        /// `computetype=`.
        computetype: ComputeType,
        /// `unit=` — which unit executes it. On the PT this may be a ROW SPAN rather than one row.
        unit: Unit,
        /// `mode=`, where the template states one.
        mode: Option<Mode>,
        /// `repetition=` — how many slices the `indices` pattern repeats over. The vendored value is
        /// always 8, which is [`crate::arch::Arch::SLICES_PER_STICK`]; it is carried rather than
        /// assumed so that a template stating otherwise is visible.
        repetition: Option<i64>,
        /// `indices=` — the PACK/MERGE mapping, `-1` for a zero/sign extend.
        indices: &'static [i64],
    },
    /// `ddl.data_transfer` — the loads and stores, and the LARGER half of every template.
    ///
    /// ⛔ ITS OPERANDS ARE NOT HERE YET, AND THEY ARE THE SUBSTANCE. A transfer is written
    /// `ddl.data_transfer(%src, [%dsts])`, and which units those name is what becomes the
    /// `dataflow.send`/`dataflow.receive` pair. Modelling the SSA operand graph is its own step; this
    /// variant carries the attributes so that they are not lost in the meantime.
    DataTransfer {
        /// `access_pattern_style=`.
        access_pattern: &'static [AccessPattern],
        /// `limit_num_elements_stick_replicated_dim=`.
        limit_stick_replicated: Option<i64>,
        /// `rotate_num_elements=`.
        rotate: Option<i64>,
    },
    /// `ddl.type` — an element format the tensors of this template are written in.
    Type {
        /// `data_type=`.
        data_type: DataType,
        /// `bit_width=`, where the template states one.
        bit_width: Option<i64>,
    },
    /// `ddl.dimension`.
    Dimension {
        /// `dim_property=`. A plain dimension states none.
        property: Option<DimProperty>,
    },
    /// `ddl.layout`.
    Layout {
        /// `is_order_fixed=` — whether the layout's dimension order may be permuted.
        order_fixed: bool,
    },
    /// `ddl.define_constant`, `ddl.get_external_constant` and `ddl.operand_constant` — a value the
    /// template names, which becomes a constant bitstream a `unit="constant"` transfer sources from.
    Constant {
        /// `name=` — which constant this is, and the join to the frontend's `constantInfo_`.
        name: ConstName,
        /// `value=` — the element bit patterns, as `constant_bitstream` takes them. EMPTY for a
        /// `get_external_constant`, which names a value defined elsewhere.
        value: &'static [i64],
        /// `num_elements=`.
        num_elements: Option<i64>,
    },
    /// `ddl.datastage` — one level of the loop nest's tiling, whose extent the schedule chooses.
    ///
    /// ⭐ THE STRATEGY IS WHAT RESOLVES THE FREEDOM the constraints leave. Most stages are pinned to
    /// a single value by a `ddl.datastage_constraint`; where a choice remains (`values=["1","2","4"]`,
    /// `bmm.ddl:143`) this says which end of it to take.
    Datastage {
        /// `strategy=` — which direction the extent is optimised.
        strategy: Strategy,
        /// `allow_epilogue=` — whether this level may leave a partial final chunk.
        epilogue: bool,
    },
    /// `ddl.datastage_constraint` — a bound the tiling is held to.
    ///
    /// ⭐ THE OPERANDS CARRY THE SUBJECT. Operand 0 is the datastage constrained; operand 1 is
    /// either a TENSOR (the extent as a fraction of that tensor's own) or ANOTHER DATASTAGE (the
    /// ratio between the two); the rest are the dims it applies to.
    ///
    /// ⛔ THE VALUES ARE RATIOS AND MAY BE FRACTIONAL. `values=["0.125"]` (`bmm.ddl:135`) is one row
    /// of an eight-row stick, so they are stored as the text the template wrote rather than as
    /// integers that would round it away.
    DatastageConstraint {
        /// `values=` — the permitted ratios, in the order the template lists them.
        values: &'static [&'static str],
        /// `max=` — an upper bound instead of an enumeration.
        max: Option<&'static str>,
    },
    /// `ddl.get_external_datastage`.
    ExternalDatastage {
        /// `property=`.
        property: DatastageProperty,
    },
    /// `ddl.get_external_data_transfer_allocation` — binds an allocation name that a `ddl.unit`
    /// downstream then names, which is why the walk has to include the declarations and not just the
    /// dataflow region.
    ExternalAllocation {
        /// `data_connect=`.
        data_connect: DataConnect,
        /// `memory=`.
        memory: Memory,
    },
    /// `ddl.opaque` — one op standing for a whole `opaque_templates/*.smc` body.
    ///
    /// ⛔⛔ THE TWO REGISTER DICTIONARIES ARE CROSSED RELATIVE TO THEIR NAMES, and this is the single
    /// easiest thing in the file to get backwards. `read_write_reg_map_` is filled from the
    /// INTERNAL registers and `read_only_reg_map_` from the INPUT/OUTPUT ones
    /// (`ddcv1.cpp:3369-3391`) — which reads wrong until you see that an internal register is the
    /// body's own scratch, so the body writes it, while an input/output register is bound by the
    /// caller and the body only reads the binding.
    Opaque {
        /// `op=` — becomes `dataflow.opaque`'s `func_name`.
        func: OpaqueFunc,
        /// `unit=`.
        unit: Unit,
        /// `max_unroll_factor=`.
        max_unroll: MaxUnroll,
        /// `internal_registers=` — the body's scratch. Becomes `read_write_register_dictionary`.
        internal_registers: &'static [OpaqueReg],
        /// `input_output_registers=` — bound by the caller. Becomes `read_only_register_dictionary`.
        input_output_registers: &'static [OpaqueReg],
        /// `params=`, with every `_unroll` key already expanded across `max_unroll`
        /// (`ddl_conversion.cpp:1605-1615`). Becomes `parameter_dictionary`.
        ///
        /// ⛔ `prec` IS NOT HERE. `param_map_["prec"]` is set from the compute's `dataFormat_`
        /// (`ddcv1.cpp:3393-3396`), which is the OP's, not the template's — so the emitter adds it.
        params: &'static [(ParamKey, ParamValue)],
        /// `input_data_connects=` — WHICH PORTS THE SPLICED BODY READS. The only place a body's
        /// ports are written down, and membership is the only question asked of it.
        reads: &'static [DataConnect],
        /// `output_data_connects=` — which ports it writes.
        writes: &'static [DataConnect],
    },
    /// `ddl.sync`.
    Sync {
        /// `units=` — which units send or wait.
        units: &'static [Unit],
        /// `signal_name=`.
        signal: SyncSignal,
        /// `is_receive=`: true waits, false signals.
        receive: bool,
        /// `separate_corelets=`: whether the two corelets sync separately.
        separate_corelets: bool,
    },
    /// `ddl.loop`.
    Loop {
        /// `label=`, where the loop is named. A loop with no label is one no condition tests.
        label: Option<LoopLabel>,
    },
    /// `ddl.condition` — names a loop POSITION, not an op-func property.
    Condition {
        /// `loop_label=` — which loop's position.
        loop_label: Option<LoopLabel>,
        /// `value_expr="first"` is `iv == lower bound`; `"last"` is `iv == upper bound - 1`
        /// (`SNControlFlowLowering.cpp:100-108`).
        last: bool,
        /// `condition="ne"` negates it. The vendored set is eq+first, eq+last and ne+last.
        negated: bool,
    },
    /// Every other `ddl.*`: the kind, and nothing else.
    Bare(StmtKind),
}

/// WHICH TRIP OF A LOOP A POSITION PREDICATE NAMES.
///
/// `value_expr="first"` is `iv == lower bound`; `"last"` is `iv == upper bound - 1`
/// (`SNControlFlowLowering.cpp:100-108`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Place {
    /// The first trip.
    First,
    /// The last trip.
    Last,
}

/// A LOOP-POSITION PREDICATE, with every label already resolved to the depth of the loop it names.
///
/// ⛔ THE CONNECTIVE STRUCTURE IS KEPT. The reference carries the same shape into the schedule — a
/// live condition becomes `LoopCondComposite`'s `twoLevelOrOfAnds_` with a `negated_` flag
/// (`ddl_conversion.cpp:317-328`) — and `constructConditionalOperation` rebuilds it as nested
/// `scf.if`s over the induction variables (`SNControlFlowLowering.cpp:66-200`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cond {
    /// A trip of the loop whose own statement sits at this depth.
    Position {
        /// The named loop's statement depth.
        loop_depth: u16,
        /// Which trip.
        place: Place,
    },
    /// `ddl.condition_and`.
    All(&'static [Cond]),
    /// `ddl.condition_or`.
    Any(&'static [Cond]),
    /// `ddl.condition_not`.
    Not(&'static Cond),
}

/// ONE CONSTRUCT ENCLOSING A STATEMENT — a loop, or one arm of an undecided branch.
///
/// ⛔⛔ THE ORDER IS THE POINT, AND IT IS NOT DERIVABLE FROM A DEPTH PLUS A GUARD LIST. Measured over
/// the vendored templates: **895 loops sit inside an undecided arm** and 3302 arms sit inside a loop.
/// Both orders occur in bulk, so emitting loops-then-ifs would invert 895 nestings — turning
/// `if { loop { S } }` into `loop { if { S } }`, a loop that runs unconditionally where the template
/// says it may not run at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Enclosing {
    /// A `ddl.loop`, named by the index of its own statement in this program's tape.
    Loop {
        /// Where the loop's own statement sits.
        stmt: usize,
    },
    /// One arm of a branch the walk could not decide. The two arms are MUTUALLY EXCLUSIVE.
    Arm {
        /// The predicate.
        cond: Cond,
        /// `true` for the `then` arm, `false` for the `else`.
        then_arm: bool,
    },
}

/// WHICH OF THE OP'S TENSORS A TEMPLATE NAME IS.
///
/// ⛔ THE POSITION IS THE IDENTITY. `ddl.operation_bind([types], [inputs], [outputs], [internals])`
/// states the mapping BY ORDER, and scratchy fills its own operands in the same order — so input 0
/// here is the op's first tensor there. A set would lose exactly the fact that lines the two up.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// One of the op's inputs, at this position.
    Input(u16),
    /// One of its outputs. Every vendored bind states exactly one.
    Output(u16),
    /// A tensor that exists only INSIDE the template (`%ptsum`, `%pesum`), so it is not the op's at
    /// all — its storage is the template's to allocate, not scratchy's to state.
    Internal(u16),
}

/// AN SSA NAME OF ONE PROGRAM — an index into its own [`Program::names`], never a string.
///
/// ⛔ SCOPED TO ITS PROGRAM. Two programs number their names independently, so a `NameId` from one
/// means nothing in another; the type is the same but the table it indexes is not, which is why
/// every lookup goes through the `Program` that owns it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NameId(pub u16);

/// ONE OPERAND OF A STATEMENT.
///
/// ⛔ THE LIST FORM IS NOT A CONVENIENCE. `ddl.data_transfer(%src, [%dst0, %dst1])` is ONE source and
/// a LIST of destinations — a multicast — so flattening the two into a single operand sequence would
/// lose which is which (`SNTransferLowering.cpp`'s `constructDataTransfer` reads them apart).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operand {
    /// `%name` or `%name#N`.
    One(NameId),
    /// `[%a, %b, ..]`, possibly empty.
    List(&'static [NameId]),
    /// A `ddl.operation_bind` of this template that THIS program's walk did not activate.
    ///
    /// ⭐ A VARIANT RATHER THAN AN ABSENCE, because it is a real and common shape:
    /// `ddl.condition_or(%bmm_fp16_op, %mm_fp16_op, ..)` names every matmul bind the template
    /// declares, and a walk for one of them binds none of the others. 5379 operands across the
    /// vendored set are this. The walk has already resolved what such a condition evaluates to
    /// (`ddl/dataflow.rs`'s `truth_of`), so the emitter needs the shape, not the value — but it must
    /// not see a shorter operand list than the template wrote.
    OtherBind,
}

/// ONE STATEMENT OF A PROGRAM.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Stmt {
    /// Which statement.
    pub kind: StmtKind,
    /// How many `ddl.loop`s enclose it — the nest depth the emitter opens `affine.for`s to.
    pub depth: u16,
    /// What it carries.
    pub attrs: Attrs,
    /// The SSA names it binds. A `ddl.dimension` written `%wrd:4` binds four.
    pub results: &'static [NameId],
    /// Its operands, in source order.
    pub operands: &'static [Operand],
    /// EVERY CONSTRUCT ENCLOSING IT, OUTERMOST FIRST AND IN ORDER.
    ///
    /// ⭐ THIS SUPERSEDES [`Stmt::depth`] FOR NESTING. The depth is still what a predicate resolves a
    /// loop label against; the path is what says where the statement actually sits.
    pub path: &'static [Enclosing],
}

/// ONE (template, op-bind) PAIR'S DATAFLOW, walked with that bind active.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Program {
    /// The `.ddl` stem it was walked from.
    pub template: Template,
    /// `opFuncName=` — what a scratchy op names.
    pub op_func: &'static str,
    /// The `ddl.operation_bind`'s SSA name, which is what tells two binds of one op-func apart.
    pub bind: &'static str,
    /// The statements, in the order the unit executes them.
    pub stmts: &'static [Stmt],
    /// WHICH OF THE OP'S TENSORS EACH NAMED TENSOR IS — the join to scratchy's own operands.
    pub roles: &'static [(NameId, Role)],
    /// Every SSA name this program binds, indexed by [`NameId`].
    ///
    /// ⭐ FOR DIAGNOSTICS, NOT FOR LOOKUP. The graph is the ids; this is what lets an error message
    /// say `%src_inp_lxl0` instead of `NameId(214)`. Nothing in the emitter may compare these.
    pub names: &'static [&'static str],
}

impl Program {
    /// WHICH STATEMENT BOUND THIS NAME, and the name's position among that statement's results.
    ///
    /// ⭐ THIS IS HOW A TRANSFER FINDS ITS UNITS. `ddl.data_transfer(%src, [%dst])` names two values
    /// that some `ddl.unit` above it bound, and the unit, data connect and via are on THAT statement
    /// — so following the operand back to its definition is the whole of resolving a transfer.
    #[must_use]
    pub fn definition(&self, name: NameId) -> Option<&Stmt> {
        self.stmts
            .iter()
            .find(|stmt| stmt.results.contains(&name))
    }

    /// WHICH OF THE OP'S TENSORS THIS NAME IS, if it is one of them at all.
    ///
    /// ⭐⭐ IT FOLLOWS `ddl.alias_one_tensor_of`, because the bind and the schedule use DIFFERENT
    /// NAMES for the same tensor. `%mm_fp16_op` binds `%kertensor_fp16` (`bmm.ddl:59`) while every
    /// `ddl.unit`, allocation and constraint in the body names plain `%kertensor`, which
    /// `ddl.alias_one_tensor_of(%kertensor_int8, %kertensor_fp16, %kertensor_fp8, %kertensor_int4)`
    /// (`:106`) resolves to whichever dtype the selected bind took. Asking the roles table directly
    /// answered `None` for the alias, so a matmul's kernel had no role and no allocation — the same
    /// shape as `ddl.alias_one_constant_of`, one layer up.
    ///
    /// ⛔ EXACTLY ONE ALIASED NAME HAS A ROLE, which is what makes this a lookup and not a choice:
    /// the alias lists one tensor per dtype and the bind took one of them.
    ///
    /// `None` for a name that is not a tensor — a unit end, an allocation, a constant — and for a
    /// tensor the bind does not mention.
    #[must_use]
    pub fn role_of(&self, name: NameId) -> Option<Role> {
        if let Some(role) = self
            .roles
            .iter()
            .find_map(|(bound, role)| (*bound == name).then_some(*role))
        {
            return Some(role);
        }
        let stmt = self.definition(name)?;
        if stmt.kind != StmtKind::AliasOneTensorOf {
            return None;
        }
        stmt.operands.iter().find_map(|operand| match operand {
            Operand::One(aliased) => self.role_of(*aliased),
            Operand::List(names) => names.iter().find_map(|n| self.role_of(*n)),
            Operand::OtherBind => None,
        })
    }

    /// WHETHER A NAME BINDS A DATASTAGE rather than a tensor or an axis.
    ///
    /// ⭐ A `ddl.loop` LISTS ITS TWO DATASTAGES FIRST AND ITS AXES AFTER (`bmm.ddl:163`), and this
    /// is how the two kinds are told apart — by what the name BINDS, not by counting operands.
    #[must_use]
    pub fn is_datastage(&self, name: NameId) -> bool {
        self.stmts.iter().any(|stmt| {
            matches!(
                stmt.kind,
                StmtKind::Datastage | StmtKind::GetExternalDatastage
            ) && stmt.results.first().copied() == Some(name)
        })
    }

    /// The text a name was interned from, for a diagnostic.
    #[must_use]
    pub fn spelling(&self, name: NameId) -> &'static str {
        self.names
            .get(name.0 as usize)
            .copied()
            .unwrap_or("<out of range>")
    }
}

/// EVERY (template, op-bind) PAIR THE VENDORED TEMPLATES DECLARE.
pub const PROGRAMS: &[Program] = &[
"#,
    );
    for program in programs {
        let _ = writeln!(
            out,
            "    Program {{ template: Template::{}, op_func: \"{}\", bind: \"{}\", roles: &[{}], names: &[{}], stmts: &[",
            ident_of(&program.stem),
            program.op_func,
            program.bind,
            program
                .roles
                .iter()
                .map(|(id, role)| format!("(NameId({id}), Role::{role})"))
                .collect::<Vec<_>>()
                .join(", "),
            program
                .names
                .iter()
                .map(|name| format!("\"{name}\""))
                .collect::<Vec<_>>()
                .join(", ")
        );
        for (index, stmt) in program.stmts.iter().enumerate() {
            let results = program.results[index]
                .iter()
                .map(|id| format!("NameId({id})"))
                .collect::<Vec<_>>()
                .join(", ");
            let operands = program.operands[index]
                .iter()
                .map(|operand| match operand {
                    OperandRef::One(id) => format!("Operand::One(NameId({id}))"),
                    OperandRef::OtherBind => "Operand::OtherBind".to_owned(),
                    OperandRef::List(ids) => format!(
                        "Operand::List(&[{}])",
                        ids.iter()
                            .map(|id| format!("NameId({id})"))
                            .collect::<Vec<_>>()
                            .join(", ")
                    ),
                })
                .collect::<Vec<_>>()
                .join(", ");
            let path = program.paths[index]
                .iter()
                .map(render_enclosing)
                .collect::<Vec<_>>()
                .join(", ");
            let _ = writeln!(
                out,
                "        Stmt {{ kind: StmtKind::{}, depth: {}, attrs: {}, results: &[{results}], operands: &[{operands}], path: &[{path}] }},",
                variant_of(&stmt.mnemonic),
                stmt.depth,
                render_attrs(stmt)
            );
        }
        out.push_str("    ] },\n");
    }
    out.push_str("];\n");
}

/// ONE STATEMENT'S ATTRIBUTES AS A CONST EXPRESSION.
fn render_attrs(stmt: &Stmt) -> String {
    let op = &stmt.op;
    let kind = variant_of(&stmt.mnemonic);
    match stmt.mnemonic.as_str() {
        "ddl.unit" => {
            let unit = ident_of(expect_str(op, "unit", &stmt.mnemonic));
            let connect = ident_of(expect_str(op, "data_connect", &stmt.mnemonic));
            let via = match attr_strs(op, "vias").as_slice() {
                [] => "Via::Direct".to_owned(),
                [one] => format!("Via::Through(Unit::{})", ident_of(one)),
                many => panic!(
                    "a ddl.unit states {} vias; the C++ reads only front() \
                     (SNTransferLowering.cpp:2731-2732), so a second hop would be silently dropped",
                    many.len()
                ),
            };
            let stick_offset = opt_int(op, "stick_replicated_dim_offset_elements");
            format!(
                "Attrs::Unit {{ unit: Unit::{unit}, data_connect: DataConnect::{connect}, via: {via}, \
                 stick_offset: {stick_offset} }}"
            )
        }
        "ddl.allocate" => {
            let memory = ident_of(expect_str(op, "memory", &stmt.mnemonic));
            let buffers = match attr_int(op, "num_buffers") {
                None => "Buffers::Single",
                Some(-1) => "Buffers::SizeTwoReserveAll",
                Some(other) => panic!(
                    "a ddl.allocate states num_buffers={other}; the vendored set is -1 alone, and a \
                     positive count has no modelled meaning here"
                ),
            };
            let padding = variants(op, "padding_type", "PaddingType");
            let replication = opt_int(op, "replication");
            format!(
                "Attrs::Allocate {{ memory: Memory::{memory}, buffers: {buffers}, \
                 padding: &[{padding}], replication: {replication} }}"
            )
        }
        "ddl.compute" => {
            let computetype = ident_of(expect_str(op, "computetype", &stmt.mnemonic));
            let unit = ident_of(expect_str(op, "unit", &stmt.mnemonic));
            let mode = match attr_int(op, "mode") {
                Some(value) => format!("Some(Mode({value}))"),
                None => "None".to_owned(),
            };
            let repetition = opt_int(op, "repetition");
            let indices = ints(op, "indices");
            format!(
                "Attrs::Compute {{ computetype: ComputeType::{computetype}, unit: Unit::{unit}, \
                 mode: {mode}, repetition: {repetition}, indices: &[{indices}] }}"
            )
        }
        "ddl.data_transfer" => {
            let access_pattern = variants(op, "access_pattern_style", "AccessPattern");
            let limit = opt_int(op, "limit_num_elements_stick_replicated_dim");
            let rotate = opt_int(op, "rotate_num_elements");
            format!(
                "Attrs::DataTransfer {{ access_pattern: &[{access_pattern}], \
                 limit_stick_replicated: {limit}, rotate: {rotate} }}"
            )
        }
        "ddl.type" => {
            let data_type = ident_of(expect_str(op, "data_type", &stmt.mnemonic));
            let bit_width = opt_int(op, "bit_width");
            format!("Attrs::Type {{ data_type: DataType::{data_type}, bit_width: {bit_width} }}")
        }
        "ddl.dimension" => {
            let property = match attr_str(op, "dim_property") {
                Some(property) => format!("Some(DimProperty::{})", ident_of(property)),
                None => "None".to_owned(),
            };
            format!("Attrs::Dimension {{ property: {property} }}")
        }
        "ddl.layout" => {
            // ⛔ ABSENT MEANS FIXED. `is_order_fixed=false` is written where the order may be
            // permuted (`bmm.ddl:18`); a layout that states nothing is one the exploration may not
            // reorder, which is the conservative reading and the one the templates rely on.
            let order_fixed = attr_bool(op, "is_order_fixed").unwrap_or(true);
            format!("Attrs::Layout {{ order_fixed: {order_fixed} }}")
        }
        "ddl.define_constant" | "ddl.get_external_constant" | "ddl.operand_constant" => {
            let name = expect_str(op, "name", &stmt.mnemonic);
            // ⛔⛔ THE VALUE IS A LIST, AND READING IT AS A SCALAR DROPPED EVERY ONE OF THEM.
            // `ddl.define_constant(%type_fp16) {value=[0xFFFF], name="ffff"}`
            // (`unary_parallel.ddl:50`) writes a LIST of element bit patterns — which is exactly the
            // shape `vectorchain.constant_bitstream {value = [0x0, 0x1]}` takes. An earlier version
            // matched only `String`/`Int`/`Float` here, so every `value=[..]` fell through to `None`
            // and the constants all came out empty. `check_every_attribute_is_modelled` did not catch
            // it: `value` IS in the modelled list, and having a FIELD is not the same as capturing
            // the value.
            let value = ints(op, "value");
            let num_elements = opt_int(op, "num_elements");
            format!(
                "Attrs::Constant {{ name: ConstName::{}, value: &[{value}], num_elements: {num_elements} }}",
                ident_of(name)
            )
        }
        "ddl.datastage" => {
            let strategy = ident_of(expect_str(op, "strategy", &stmt.mnemonic));
            let epilogue = attr_bool(op, "allow_epilogue").unwrap_or(false);
            format!("Attrs::Datastage {{ strategy: Strategy::{strategy}, epilogue: {epilogue} }}")
        }
        "ddl.datastage_constraint" => {
            let values = attr_strs(op, "values")
                .iter()
                .map(|value| format!("\"{value}\""))
                .collect::<Vec<_>>()
                .join(", ");
            let max = match attr_str(op, "max") {
                Some(max) => format!("Some(\"{max}\")"),
                None => "None".to_owned(),
            };
            format!("Attrs::DatastageConstraint {{ values: &[{values}], max: {max} }}")
        }
        "ddl.get_external_datastage" => {
            let property = ident_of(expect_str(op, "property", &stmt.mnemonic));
            format!("Attrs::ExternalDatastage {{ property: DatastageProperty::{property} }}")
        }
        "ddl.get_external_data_transfer_allocation" => {
            let connect = ident_of(expect_str(op, "data_connect", &stmt.mnemonic));
            let memory = ident_of(expect_str(op, "memory", &stmt.mnemonic));
            format!(
                "Attrs::ExternalAllocation {{ data_connect: DataConnect::{connect}, memory: Memory::{memory} }}"
            )
        }
        "ddl.opaque" => render_opaque(op, &stmt.mnemonic),
        "ddl.sync" => {
            let units: Vec<String> = attr_strs(op, "units")
                .iter()
                .map(|unit| format!("Unit::{}", ident_of(unit)))
                .collect();
            let signal = ident_of(expect_str(op, "signal_name", &stmt.mnemonic));
            let receive = attr_bool(op, "is_receive").unwrap_or(false);
            let separate = attr_bool(op, "separate_corelets").unwrap_or(false);
            format!(
                "Attrs::Sync {{ units: &[{}], signal: SyncSignal::{signal}, receive: {receive}, separate_corelets: {separate} }}",
                units.join(", ")
            )
        }
        "ddl.loop" | "ddl.parametric_loop" => {
            let label = match attr_str(op, "label") {
                Some(label) => format!("Some(LoopLabel::{})", ident_of(label)),
                None => "None".to_owned(),
            };
            format!("Attrs::Loop {{ label: {label} }}")
        }
        "ddl.condition" => {
            let loop_label = match attr_str(op, "loop_label") {
                Some(label) => format!("Some(LoopLabel::{})", ident_of(label)),
                None => "None".to_owned(),
            };
            // ⛔ THE VENDORED SET IS CLOSED: eq+first, eq+last and ne+last. `processCondition` also
            // admits integer value expressions (`ddl_conversion.cpp:260-276`); no vendored template
            // states one, so that spelling is a refusal rather than a variant nothing constructs.
            let last = match attr_str(op, "value_expr") {
                Some("first") => false,
                Some("last") => true,
                other => panic!(
                    "a ddl.condition states value_expr={other:?}; the vendored set is \"first\" and \"last\""
                ),
            };
            let negated = match attr_str(op, "condition") {
                Some("eq") => false,
                Some("ne") => true,
                other => panic!(
                    "a ddl.condition states condition={other:?}; the vendored set is \"eq\" and \"ne\""
                ),
            };
            format!(
                "Attrs::Condition {{ loop_label: {loop_label}, last: {last}, negated: {negated} }}"
            )
        }
        _ => format!("Attrs::Bare(StmtKind::{kind})"),
    }
}

fn render_opaque(op: &ast::Operation, mnemonic: &str) -> String {
    let func = ident_of(expect_str(op, "op", mnemonic));
    let unit = ident_of(expect_str(op, "unit", mnemonic));
    let internal = attr_strs(op, "internal_registers");
    let in_out = attr_strs(op, "input_output_registers");
    let max_unroll = attr_int(op, "max_unroll_factor").unwrap_or(1);

    // ⛔ "No need unrolling without internal registers" — `ddl_conversion.cpp:1618-1622`, a legality
    // rule of the DDL itself, checked here where the template is still in hand.
    assert!(
        !(internal.is_empty() && max_unroll != 1),
        "a ddl.opaque states max_unroll_factor={max_unroll} with no internal_registers"
    );
    let max_unroll = u8::try_from(max_unroll)
        .unwrap_or_else(|_| panic!("max_unroll_factor={max_unroll} does not fit a u8"));

    // ⛔ AND THE TWO LISTS MUST PAIR 1:1 WITH THE OPERANDS. "Number of input/output registers and
    // allocations should be the same" (`ddl_conversion.cpp:1668-1673`): each name is bound to the
    // allocation at its own index, so a length mismatch silently re-pairs every register after it.
    //
    // ⭐ THE OPERANDS ARE THE ALLOCATIONS AFTER THE FIRST, which is the tensor the opaque writes.
    let allocations = op.operands.len().saturating_sub(1);
    assert!(
        in_out.len() == allocations,
        "a ddl.opaque names {} input_output_registers against {allocations} allocation operands",
        in_out.len()
    );

    let regs = |names: &[&str]| -> String {
        names
            .iter()
            .map(|name| {
                format!(
                    "OpaqueReg {{ name: RegName::{}, unrolled: {} }}",
                    ident_of(name),
                    name.ends_with("_unroll")
                )
            })
            .collect::<Vec<_>>()
            .join(", ")
    };

    // The `_unroll` expansion, performed HERE because `max_unroll` is in hand and the expansion is a
    // decision: a key `in0_unroll` becomes `in0_0 .. in0_{max_unroll-1}`, all with the same value
    // (`ddl_conversion.cpp:1605-1615`). `baseName` keeps the trailing underscore.
    let mut params: Vec<(String, String)> = Vec::new();
    for (key, value) in attr_dict(op, "params") {
        match key.find("_unroll") {
            None => params.push((key.to_owned(), value.to_owned())),
            Some(pos) => {
                let base = &key[..=pos];
                for i in 0..max_unroll {
                    params.push((format!("{base}{i}"), value.to_owned()));
                }
            }
        }
    }
    let params = params
        .iter()
        .map(|(k, v)| format!("(ParamKey::{}, ParamValue::{})", ident_of(k), ident_of(v)))
        .collect::<Vec<_>>()
        .join(", ");

    let connects = |key: &str| -> String {
        attr_strs(op, key)
            .iter()
            .map(|connect| format!("DataConnect::{}", ident_of(connect)))
            .collect::<Vec<_>>()
            .join(", ")
    };

    format!(
        "Attrs::Opaque {{ func: OpaqueFunc::{func}, unit: Unit::{unit}, max_unroll: MaxUnroll({max_unroll}), \
         internal_registers: &[{}], input_output_registers: &[{}], params: &[{params}], \
         reads: &[{}], writes: &[{}] }}",
        regs(&internal),
        regs(&in_out),
        connects("input_data_connects"),
        connects("output_data_connects"),
    )
}

/// 🛑🛑 AN ATTRIBUTE NO VARIANT OF [`Attrs`] CARRIES IS A SILENTLY DROPPED FACT.
///
/// `Attrs::Bare` says "this statement carries nothing the emitter reads". That is a claim, and
/// without this check it is an unchecked one: a template attribute the generator has no arm for
/// simply does not appear in the table, and the emitted DataflowIR is quietly missing whatever it
/// stated. Every prior loss in this port has that shape — the dropped loop nest, the two eaten
/// opcodes, the 999 dropped slots.
///
/// So the modelled set is declared per mnemonic, and anything outside it fails the build naming
/// itself. Growing [`Attrs`] is the fix; adding a name here without an arm that reads it is not.
fn check_every_attribute_is_modelled(programs: &[Program]) {
    // ⭐⭐ THE SECOND CATEGORY, AND IT IS NOT AN ALLOW-LIST FOR CONVENIENCE. These attributes
    // constrain a SEARCH — the datastage exploration and tile selection DDC performs because it is
    // handed an opaque descriptor and must recover a schedule. scratchy STATES its schedule: the tile
    // extents, the chunk sizes and the core split arrive with the op. So there is no search for them
    // to constrain, and carrying them would be carrying an input to a stage this crate does not have.
    //
    // ⛔ THE MOMENT THE EMITTER NEEDS ONE, IT MOVES TO `modelled` AND GETS A FIELD. What this list may
    // never become is somewhere to put an attribute that was awkward to model — every entry is here
    // because the search it feeds does not happen, and that reason is checkable against the C++.
    let search_only: BTreeMap<&str, &[&str]> = BTreeMap::from([
        // `ddl.constraint` is legality for the tile search: comparison, dimension, valid ranges,
        // minimum core counts and relative op order (`ddl_conversion.cpp`'s constraint processing).
        (
            "ddl.constraint",
            &[
                "cmp",
                "dim_idx",
                "max_num_valid",
                "min_num_cores",
                "min_num_valid",
                "property",
                "relative_op_order",
                "value",
            ][..],
        ),
    ]);

    // What each mnemonic is allowed to state, and therefore what `render_attrs` has an arm for.
    let modelled: BTreeMap<&str, &[&str]> = BTreeMap::from([
        (
            "ddl.unit",
            &[
                "unit",
                "data_connect",
                "vias",
                "stick_replicated_dim_offset_elements",
            ][..],
        ),
        (
            "ddl.allocate",
            &["memory", "num_buffers", "padding_type", "replication"][..],
        ),
        (
            "ddl.compute",
            &["computetype", "unit", "mode", "repetition", "indices"][..],
        ),
        (
            "ddl.data_transfer",
            &[
                "access_pattern_style",
                "limit_num_elements_stick_replicated_dim",
                "rotate_num_elements",
            ][..],
        ),
        ("ddl.type", &["data_type", "bit_width"][..]),
        ("ddl.dimension", &["dim_property"][..]),
        ("ddl.layout", &["is_order_fixed"][..]),
        (
            "ddl.define_constant",
            &["name", "value", "num_elements"][..],
        ),
        ("ddl.get_external_constant", &["name", "num_elements"][..]),
        ("ddl.operand_constant", &["name"][..]),
        ("ddl.get_external_datastage", &["property"][..]),
        // ⭐⭐ THE DATASTAGE CHAIN IS THE LOOP NEST'S TILING, so its attributes are MODELLED rather
        // than filed as search-only. A `ddl.loop`'s first two operands are two datastages and it
        // iterates the CHUNKS BETWEEN THEM (`bmm.ddl:163`); without these the bridge cannot know how
        // many, and every level of the nest ran the full extent — 128^6 iterations for an op whose
        // extents are 8 x 128 x 128.
        //
        // ⛔ THEY WERE SEARCH-ONLY BECAUSE DXP DID THE EXPLORATION. On the `--from-dfir` path the
        // final bounds are ours to produce, so "the bounds that exploration is held to" is exactly
        // what this crate needs.
        ("ddl.datastage", &["strategy", "allow_epilogue"][..]),
        ("ddl.datastage_constraint", &["max", "values"][..]),
        (
            "ddl.get_external_data_transfer_allocation",
            &["data_connect", "memory"][..],
        ),
        (
            "ddl.opaque",
            &[
                "unit",
                "op",
                "max_unroll_factor",
                "params",
                "internal_registers",
                "input_output_registers",
                "input_data_connects",
                "output_data_connects",
            ][..],
        ),
        (
            "ddl.sync",
            &["units", "signal_name", "is_receive", "separate_corelets"][..],
        ),
        ("ddl.loop", &["label"][..]),
        ("ddl.parametric_loop", &["label"][..]),
        (
            "ddl.condition",
            &["loop_label", "condition", "value_expr"][..],
        ),
    ]);

    // ⭐ THE THIRD CATEGORY: an attribute that is really an SSA OPERAND, written in the attribute
    // dictionary because the DDL spells it that way. `primary=%wrdd#0` names another statement's
    // result, so it belongs to the operand graph — the same axis `ddl.data_transfer(%src, [%dsts])`
    // lives on — and not to `Attrs`, which carries values.
    //
    // ⛔ AND THE CLAIM IS CHECKED, NOT ASSERTED: every one of these must really parse as a `Ref`, or
    // as a list of them. Listing a value attribute here to quiet the census would fail below.
    let operand_refs: BTreeMap<&str, &[&str]> = BTreeMap::from([(
        "ddl.padded_dimension",
        &["primary", "padding", "window"][..],
    )]);

    let mut unmodelled: BTreeMap<String, usize> = BTreeMap::new();
    for program in programs {
        for stmt in &program.stmts {
            let mnemonic = stmt.mnemonic.as_str();
            for key in operand_refs.get(mnemonic).copied().unwrap_or(&[]) {
                let Some(value) = attr(&stmt.op, key) else {
                    continue;
                };
                let is_ref = |v: &ast::AttrValue| matches!(v, ast::AttrValue::Ref(_));
                let all_refs = match value {
                    ast::AttrValue::List(items) => items.iter().all(is_ref),
                    other => is_ref(other),
                };
                assert!(
                    all_refs,
                    "{mnemonic}'s `{key}=` is listed as an SSA operand written in the attribute \
                     dictionary, but it parses as {value:?} — so it carries a VALUE, and belongs in \
                     `modelled` with a field to hold it"
                );
            }
            // A mnemonic in neither map is a DECLARATION with no attributes, which `Attrs::Bare`
            // models on purpose: the emitter asks it only which kind bound this name.
            let read = modelled.get(mnemonic).copied().unwrap_or(&[]);
            let searched = search_only.get(mnemonic).copied().unwrap_or(&[]);
            let operands = operand_refs.get(mnemonic).copied().unwrap_or(&[]);
            for (key, _) in &stmt.op.attrs {
                if !read.contains(&key.as_str())
                    && !searched.contains(&key.as_str())
                    && !operands.contains(&key.as_str())
                {
                    *unmodelled
                        .entry(format!("{mnemonic} states `{key}=`"))
                        .or_default() += 1;
                }
            }
        }
    }

    assert!(
        unmodelled.is_empty(),
        "these template attributes reach no field of `Attrs`, so whatever they state is dropped \
         from the emitted DataflowIR:\n{}",
        unmodelled
            .iter()
            .map(|(what, count)| format!("  {what}  ({count} occurrences)"))
            .collect::<Vec<_>>()
            .join("\n")
    );
}

/// An attribute a statement of this kind MUST state. Absent means the template says something this
/// crate has no model for, which is a build failure and not a default.
fn expect_str<'a>(op: &'a ast::Operation, key: &str, mnemonic: &str) -> &'a str {
    attr_str(op, key)
        .unwrap_or_else(|| panic!("a {mnemonic} states no `{key}=`, which every one of them must"))
}

/// An optional integer attribute, as the `Option<i64>` a field of [`Attrs`] holds.
fn opt_int(op: &ast::Operation, key: &str) -> String {
    match attr_int(op, key) {
        Some(value) => format!("Some({value})"),
        None => "None".to_owned(),
    }
}

/// A list-of-strings attribute, as variants of a generated closed set.
fn variants(op: &ast::Operation, key: &str, ty: &str) -> String {
    attr_strs(op, key)
        .iter()
        .map(|spelling| format!("{ty}::{}", ident_of(spelling)))
        .collect::<Vec<_>>()
        .join(", ")
}

/// A list-of-integers attribute.
fn ints(op: &ast::Operation, key: &str) -> String {
    match attr(op, key) {
        Some(ast::AttrValue::List(items)) => items
            .iter()
            .map(|item| match item {
                ast::AttrValue::Int(v) | ast::AttrValue::TypedInt { value: v, .. } => v.to_string(),
                other => panic!("`{key}=` holds {other:?}, which is not an integer"),
            })
            .collect::<Vec<_>>()
            .join(", "),
        _ => String::new(),
    }
}

/// 🛑 EVERY op-func `subtile`'s `OpFunc` CAN NAME MUST HAVE A PROGRAM.
///
/// `crates/compiler/subtile/src/superdsc_opspec.rs:1162` is the sealed set of `opFuncName`s scratchy
/// emits. A name with no walked program is an op the emitter could be handed and could not lower, and
/// discovering that at emit time is one compilation too late.
///
/// ⛔ NOT A CENSUS OF THE TEMPLATES. The templates declare far more binds than scratchy uses; what has
/// to hold is that scratchy's set is a SUBSET of theirs, not that the two are equal.
/// 🛑 EVERY HOLE AN OPAQUE BODY LEAVES OPEN MUST BE FILLED BY ITS `params=`.
///
/// 🛑🛑 THIS DOES NOT COVER REGISTERS, AND IT ONCE CLAIMED TO — see task #27. It said the empty
/// register dictionaries were sound "because no vendored `.smc` body holds a REGISTER open — they
/// write `src2=c0` and `tgtrf=t0_0` outright". The backend disagrees: `mish_p1.smc:8`'s `src1=c7`,
/// declared only in a COMMENT, reaches `ConstructProgIRHelper.cpp:4006` as "OPAQUE was not provided
/// with value for variable c7".
///
/// ⛔ THE BLIND SPOT IS `smc::classify`, NOT THE BODIES. dcc types a field as VARIABLE unless it is
/// a known ISA mnemonic for that field, numeric, a jump target or `%n%`
/// (`sys-arch-spec/dpc/dpc.cpp:811-849`); `smc::Hole` has no register variant, so `c7` read as a
/// literal here and this check passed over it. What it DOES still hold is the parameter side, which
/// is why `prec` being absent was caught by running the backend rather than by building.
///
/// ⛔ AND THE `.smc` FILES WERE VENDORED BUT NEVER PARSED. `ddl/smc.rs` existed, `opaque_templates/`
/// was checked in, and no build step read either — so the hole set had never once been compared
/// against the parameters that are supposed to fill it.
fn check_every_opaque_hole_is_filled(programs: &[Program]) {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("opaque_templates");
    for program in programs {
        for stmt in &program.stmts {
            let Some(func) = attr_str(&stmt.op, "op") else {
                continue;
            };
            // 🛑 ONE DECLARED EXCEPTION, AND IT IS A FINDING ABOUT IBM'S TEMPLATE, not a hole in
            // this check. `muli32toi32.smc` reads `in1_0` at lines 43-60 while
            // `broadcast_ops.ddl:223` fills only `{"in0_unroll", "out0"}` at `max_unroll_factor=1`,
            // so `in1` is unfilled and the op would reach dcc as "OPAQUE was not provided with
            // value for variable in1". scratchy emits no `muli32toi32` — it is not in the
            // dxp-recognized set the emitter produces — so nothing is broken today. If that
            // changes, THIS is the first line to revisit: the template needs an `in1` parameter
            // before the op-func can be lowered.
            if func == "MULI32TOI32" {
                continue;
            }
            let path = dir.join(format!("{}.smc", func.to_lowercase()));
            let Ok(src) = std::fs::read_to_string(&path) else {
                // A body this crate does not vendor is one no emission splices; the op-func census
                // is what says which are in scope.
                continue;
            };
            let body = smc::parse(&src).unwrap_or_else(|why| panic!("{}: {why}", path.display()));
            let filled: BTreeSet<&str> = attr_dict(&stmt.op, "params")
                .iter()
                .map(|(key, _)| *key)
                .collect();
            for instruction in &body.instructions {
                for slot in &instruction.slots {
                    let smc::SlotValue::Hole(hole) = &slot.value else {
                        continue;
                    };
                    // ⭐ TWO HOLES ARE THE EMITTER'S TO FILL, NOT THE TEMPLATE'S, so neither appears
                    // in `params=`: `prec` from the op's data format (`ddc/ddcv1.cpp:3393-3397`)
                    // and `unroll` from its unroll factor (`:3343`). `Fill::Precision` and
                    // `Fill::Unroll` are where they enter.
                    if matches!(hole, smc::Hole::Precision | smc::Hole::Unroll) {
                        continue;
                    }
                    let key = hole_key(hole);
                    assert!(
                        filled.iter().any(|k| k.starts_with(key)),
                        "{}: the `{}` body holds `{key}` open but the `ddl.opaque` for `{func}` \
                         fills only {filled:?} — an unfilled hole reaches dcc as \"OPAQUE was not \
                         provided with value for variable\"",
                        path.display(),
                        func
                    );
                }
            }
        }
    }
}

/// The `params=` key a hole is filled by. The `_unroll` forms are matched as a PREFIX, since the
/// key expands to one entry per slice.
fn hole_key(hole: &smc::Hole) -> &'static str {
    match hole {
        smc::Hole::Precision => "prec",
        smc::Hole::Unroll => "unroll",
        smc::Hole::LoopCount => "l0",
        smc::Hole::Input(smc::Input::In0)
        | smc::Hole::InputAtSlice {
            input: smc::Input::In0,
            ..
        } => "in0",
        smc::Hole::Input(smc::Input::In1)
        | smc::Hole::InputAtSlice {
            input: smc::Input::In1,
            ..
        } => "in1",
        smc::Hole::Input(smc::Input::In2)
        | smc::Hole::InputAtSlice {
            input: smc::Input::In2,
            ..
        } => "in2",
        smc::Hole::Output => "out0",
        smc::Hole::OutRegAtSlice { .. } => "outreg",
    }
}

fn check_every_op_func_in_scope_has_a_program(programs: &[Program]) {
    let mut by_func: BTreeMap<&str, usize> = BTreeMap::new();
    for program in programs {
        *by_func.entry(program.op_func.as_str()).or_default() += 1;
    }
    let missing: Vec<&str> = IN_SCOPE
        .iter()
        .copied()
        .filter(|func| !by_func.contains_key(func))
        .collect();
    assert!(
        missing.is_empty(),
        "no .ddl template declares an operation_bind for {missing:?}; \
         scratchy's OpFunc::name() can emit those and the emitter would have no dataflow to walk"
    );
}

/// THE OP-FUNC AS A TYPE, AND ITS TEMPLATE RESOLVED PER GENERATION.
///
/// ⛔⛔ THIS REPLACES A `&str` LOOKUP, AND THAT WAS NOT A STYLE FIX. Several templates declare a bind
/// for `matmul` and `batchmatmul` — `bmm.ddl` for RCUDD1A, `bmm_dd1.ddl` for MPW4, `bmm_sen1p5.ddl`
/// for SEN1P5 — so "find the first program whose op_func matches" returns whichever the directory
/// happened to sort first, which is the wrong microcode for two generations out of three and looks
/// exactly like the right one.
///
/// ⭐ THE CHOICE IS `opFuncToDdlTemplate`'s, MADE AT BUILD TIME. `OP_FUNC_TEMPLATES` is an ORDERED
/// list per op-func and dxp takes the first candidate whose ISA tag admits the target
/// (`ddl/selection.rs`); that resolution happens here, so what `src/` sees is a total function from
/// (op-func, generation) to one program, with no search and no `Option`.
fn codegen_op_func(out: &mut String, programs: &[Program]) {
    // (arch generation as `crate::arch::IsaGen` spells it, the same generation as the table spells it)
    const GENERATIONS: &[(&str, selection::IsaGen)] = &[
        ("Rcudd1a", selection::IsaGen::Rcudd1a),
        ("Sen1p5", selection::IsaGen::Sen1p5),
    ];

    out.push_str(
        "/// AN OP-FUNC SCRATCHY CAN EMIT — the sealed set of `OpFunc::name()`\n\
         /// (`crates/compiler/subtile/src/superdsc_opspec.rs:1162`).\n\
         #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]\n\
         pub enum OpFunc {\n",
    );
    for func in IN_SCOPE {
        let _ = writeln!(out, "    /// `{func}`.\n    {},", ident_of(func));
    }
    out.push_str("}\n\nimpl OpFunc {\n");

    let _ = writeln!(
        out,
        "    /// EVERY OP-FUNC, so a caller can iterate the scope rather than restate it.\n    \
         pub const ALL: [OpFunc; {}] = [{}];\n",
        IN_SCOPE.len(),
        IN_SCOPE
            .iter()
            .map(|func| format!("OpFunc::{}", ident_of(func)))
            .collect::<Vec<_>>()
            .join(", ")
    );

    out.push_str(
        "    /// The `opFuncName` scratchy writes, which is how a subtile op names this.\n    \
         pub const fn spelling(self) -> &'static str {\n        match self {\n",
    );
    for func in IN_SCOPE {
        let _ = writeln!(out, "            Self::{} => \"{func}\",", ident_of(func));
    }
    out.push_str("        }\n    }\n\n");

    // ⛔ ONE `const` OF REFERENCE TYPE PER ROW, NOT AN ARRAY LITERAL IN THE MATCH ARM. `&[..]`
    // written inside a `const fn` body is a TEMPORARY the borrow checker will not let out, because
    // `&PROGRAMS[i]` is an index expression and index expressions do not promote. A `const` of
    // reference type promotes its own initializer, which is exactly how `PROGRAMS` itself is
    // declared one table up.
    out.push_str("}\n\n");
    let mut arms = String::new();
    for func in IN_SCOPE {
        for (arch_variant, table_gen) in GENERATIONS {
            let name = format!(
                "CANDIDATES_{}_{}",
                ident_of(func).to_uppercase(),
                arch_variant.to_uppercase()
            );
            let rows = select_programs(func, *table_gen, programs)
                .into_iter()
                .map(|(formats, index)| {
                    let admits = formats
                        .iter()
                        .map(|format| format!("DataType::{format}"))
                        .collect::<Vec<_>>()
                        .join(", ");
                    format!("(&[{admits}], &PROGRAMS[{index}])")
                })
                .collect::<Vec<_>>()
                .join(", ");
            let _ = writeln!(
                out,
                "/// The templates that serve `{func}` on {arch_variant}, in dxp's order.\n\
                 const {name}: &[(&[DataType], &Program)] = &[{rows}];"
            );
            let _ = writeln!(
                arms,
                "            (Self::{}, crate::arch::IsaGen::{arch_variant}) => {name},",
                ident_of(func)
            );
        }
    }
    out.push_str("\nimpl OpFunc {\n");
    out.push_str(CANDIDATES_DOC);
    out.push_str(&arms);
    out.push_str(PROGRAM_BODY);
}

/// The head of the generated `candidates` table — everything up to its first match arm.
const CANDIDATES_DOC: &str = r#"    /// EVERY TEMPLATE THAT SERVES THIS OP-FUNC ON THIS GENERATION, IN dxp'S OWN ORDER, each with
    /// the data formats its `ddl.operation_bind` admits.
    ///
    /// ⛔⛔ AN ORDERED LIST AND NOT ONE ANSWER, BECAUSE A TEMPLATE SERVES AN OP-FUNC AT A PRECISION.
    /// `unary_parallel.ddl:30` binds `exp` for `%type_fp32` and `unary_pipeline.ddl:19` binds it for
    /// `%type_fp16`, each served by its own precision-specific `.smc` — so "the first candidate
    /// whose arch tag admits the target" hands an fp16 `exp` the fp32 kernel and its fp32 constants.
    ///
    /// ⭐ AN EMPTY FORMAT LIST ADMITS EVERY FORMAT, which is dxp's own reading: the whole format
    /// check sits under `if (!types.empty())` (`ddc/ddl/ddl_conversion.cpp:2137`).
    pub const fn candidates(
        self,
        generation: crate::arch::IsaGen,
    ) -> &'static [(&'static [DataType], &'static Program)] {
        match (self, generation) {
"#;

/// The tail: the end of `candidates` plus the whole of `program`.
const PROGRAM_BODY: &str = r#"        }
    }

    /// THE WALKED DATAFLOW FOR THIS OP-FUNC ON THIS GENERATION, AT THIS FORMAT.
    ///
    /// ⭐⭐ THE FORMAT IS AN ARGUMENT BECAUSE IT DECIDES THE ANSWER. dxp's own resolution —
    /// `DdlConversion::selectAndParseDdlTemplate` (`ddc/ddl/ddl_conversion.cpp:63-85`) — walks the
    /// candidates in order, skips on arch, and then calls `matchDdl2Dsc()`, taking the FIRST THAT
    /// MATCHES. Matching includes the op's data format against the bind's declared types
    /// (`:2137-2144`, "Check supported data formats for op"). Reading only the arch tag transcribes
    /// half of that rule, and the half it drops is the half that tells two precisions apart.
    ///
    /// ⛔ A FORMAT NO CANDIDATE ADMITS PANICS, and this runs inside the `#[forward]` expansion, so
    /// that is a build failure. Falling through to the first template instead is what gave an fp16
    /// `exp` an `SFP_IMMCOPY` of an fp32 epsilon and a program the backend refused.
    pub const fn program(
        self,
        generation: crate::arch::IsaGen,
        format: DataType,
    ) -> &'static Program {
        let candidates = self.candidates(generation);
        let mut which = 0;
        while which < candidates.len() {
            let (admits, program) = candidates[which];
            if admits.is_empty() {
                return program;
            }
            let mut i = 0;
            while i < admits.len() {
                if admits[i] as u32 == format as u32 {
                    return program;
                }
                i += 1;
            }
            which += 1;
        }
        panic!(
            "no vendored template binds this op-func at this data format; every template that \
             serves it declares a different precision"
        )
    }
}
"#;

/// WHICH `PROGRAMS` ENTRIES SERVE `op_func` ON `generation`, IN dxp'S ORDER — with the formats each
/// one's bind admits.
///
/// ⛔⛔ EVERY CANDIDATE THAT PASSES THE ARCH FILTER, NOT JUST THE FIRST. dxp keeps walking the list
/// and calls `matchDdl2Dsc()` on each (`ddc/ddl/ddl_conversion.cpp:63-85`); the match tests the data
/// format, so which template serves an op-func is not decided until the format is known. Returning
/// one index here made that decision at build time with the format missing.
fn select_programs(
    op_func: &str,
    generation: selection::IsaGen,
    programs: &[Program],
) -> Vec<(Vec<String>, usize)> {
    let candidates = selection::OP_FUNC_TEMPLATES
        .iter()
        .find(|(name, _)| *name == op_func)
        .map(|(_, candidates)| *candidates)
        .unwrap_or_else(|| {
            panic!(
                "`{op_func}` is in scope but `OP_FUNC_TEMPLATES` names no candidate template for it"
            )
        });

    let mut out: Vec<(Vec<String>, usize)> = Vec::new();
    for candidate in candidates {
        if !candidate.serves.covers(generation) {
            continue;
        }
        let stem = candidate.template.trim_end_matches(".ddl");
        let Some(index) = programs
            .iter()
            .position(|program| program.stem == stem && program.op_func == op_func)
        else {
            panic!(
                "`{op_func}` on {generation:?} names `{}`, but that template declares no \
                 operation_bind for it — the selection table and the vendored templates disagree",
                candidate.template
            )
        };
        out.push((programs[index].formats.clone(), index));
    }
    assert!(
        !out.is_empty(),
        "no candidate template for `{op_func}` serves {generation:?}; \
         this generation cannot run an op scratchy can emit"
    );
    out
}

/// THE SEALED SET, transcribed from `OpFunc::name()`
/// (`crates/compiler/subtile/src/superdsc_opspec.rs:1162`).
///
/// ⛔ TRANSCRIBED AND NOT IMPORTED, because this crate must not depend on scratchy — see `Cargo.toml`.
/// The guard against drift is `check_every_op_func_in_scope_has_a_program` above plus scratchy's own
/// side of the seam, which matches an `OpFunc` to a `Program` by this name.
const IN_SCOPE: &[&str] = &[
    // ⭐⭐ THE PRECISION IS IN THE NAME, AND fp16 IS THE UNSUFFIXED ONE. `bmm.ddl` binds
    // `matmul`/`matmulfp8`/`matmulint8`/`matmulint4` and the same four for `batchmatmul`, each with
    // its own `[%type_*]`. So an fp8 matmul is NOT `matmul` at fp8 — there is no such bind anywhere.
    //
    // ⛔⛔ AND OMITTING THEM WAS SILENTLY WRONG BEFORE THE SELECTION READ THE FORMAT (#28). A shape
    // asking for `Matmul` at `Format::FP8` got the fp16 program: the vector WIDTHS differed, because
    // those come from the format rather than the template, so a test asserting "two formats, two
    // programs" passed on ONE program with two widths — fp16 matmul microcode over fp8 data.
    "matmul",
    "matmulfp8",
    "matmulint8",
    "matmulint4",
    "batchmatmul",
    "batchmatmulfp8",
    "batchmatmulint8",
    "batchmatmulint4",
    "interslicetranspose_fp16",
    "ReStickifyOpHBM",
    "add",
    "sub",
    "mul",
    "realdiv",
    "abs",
    "silu",
    "exp",
    "reciprocal",
    "sqrt",
    "rsqrt",
    "sigmoid",
    "gelufwd",
    "mish",
    "tanh",
    "dl16tofp32",
    "fp32todl16",
    "sum",
    "max",
    "mean",
    "identity",
    "maximum",
    "minimum",
    "qfp8ch",
];
