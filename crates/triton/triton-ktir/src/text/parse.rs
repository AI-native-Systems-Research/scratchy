// Copyright (c) 2026 IBM Corporation. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files
// (the "Software"), to deal in the Software without restriction,
// including without limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of the Software,
// and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
// SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

//! A TEST INSTRUMENT: MLIR text -> [`Module`].
//!
//! Line-oriented, and sound for that reason: the input is machine-printed by
//! MLIR, one operation per line, never hand-written. The one construct that
//! breaks the rule is a region, and regions are tracked explicitly by brace
//! depth. Anything it cannot account for is an error NAMING THE LINE -- never a
//! silent skip, because a skipped op becomes a missing op in the census and the
//! diff then reports a difference whose cause is here.
//!
//! Scope: the generic-ish form `make_ttir` and `make_ktir` print. It is not a
//! general MLIR parser and must not become one -- when bridge one lands, the
//! parser's only remaining user is the golden test.

use std::collections::HashMap;

use crate::ir::*;
use crate::{Refusal, Result};

const PASS: &str = "ktir-text-parse";

fn err(line: usize, msg: impl std::fmt::Display) -> Refusal {
    Refusal::new(PASS, format!("line {line}: {msg}"))
}

/// Parse a module. `%name` spellings become [`Ssa`] ids, with the spelling kept
/// in [`Module::hints`] for diagnostics only.
pub fn parse(text: &str) -> Result<Module> {
    let mut p = Parser {
        module: Module::new(),
        scopes: vec![HashMap::new()],
        aliases: HashMap::new(),
    };
    p.collect_aliases(text);
    let cleaned = strip_locs(text);

    // A stack of (op-under-construction, its region ops). The module is the
    // implicit root; an explicit `module {` line pushes a real one we then unwrap,
    // so both `module {` and a bare function parse.
    let mut stack: Vec<Op> = Vec::new();
    let mut roots: Vec<Op> = Vec::new();

    for (i, raw) in cleaned.lines().enumerate() {
        let lineno = i + 1;
        let line = raw.trim();
        if line.is_empty() || line.starts_with("//") || line.starts_with('#') {
            continue;
        }
        if line.starts_with("{-#") {
            break; // the external_resources trailer carries no operations
        }

        // A block label introduces the enclosing region's arguments.
        if let Some(rest) = line.strip_prefix('^') {
            let args = p.parse_block_args(rest, lineno)?;
            let top = stack
                .last_mut()
                .ok_or_else(|| err(lineno, "a block label outside any region"))?;
            let r = top
                .regions
                .last_mut()
                .ok_or_else(|| err(lineno, "a block label on an op with no region"))?;
            r.args = args;
            continue;
        }

        let delta = brace_delta(line);

        if delta < 0 {
            // A closer, possibly closing several regions, possibly carrying the
            // op's trailing result type (`} -> tensor<64xf16>`).
            for _ in 0..(-delta) {
                let mut done = stack
                    .pop()
                    .ok_or_else(|| err(lineno, format!("unbalanced '}}': {line}")))?;
                p.pop_scope();
                p.apply_closer(&mut done, line, lineno)?;
                match stack.last_mut() {
                    Some(parent) => {
                        let r = parent.regions.last_mut().expect("region pushed with op");
                        r.ops.push(done);
                    }
                    None => roots.push(done),
                }
            }
            continue;
        }

        let mut op = p.parse_op(line, lineno, delta > 0)?;

        if delta > 0 {
            // The braces that opened here are this op's region(s). An attribute
            // dictionary is balanced, so a positive delta is always a region.
            //
            // `tt.func` and `scf.for` have ALREADY created their region, because
            // that is where their block arguments (the signature, the induction
            // variable and the iter_args) live. Only top up to `delta`, or those
            // arguments end up on a region the body never lands in -- which reads
            // downstream as a function with no arguments.
            while op.regions.len() < delta as usize {
                op.regions.push(Region::default());
            }
            stack.push(op);
        } else {
            match stack.last_mut() {
                Some(parent) => {
                    let r = parent.regions.last_mut().expect("region pushed with op");
                    r.ops.push(op);
                }
                None => roots.push(op),
            }
        }
    }

    if let Some(open) = stack.last() {
        return Err(Refusal::new(
            PASS,
            format!("input ended with an unclosed region opened by `{}`", open.kind.spelling()),
        ));
    }

    // Unwrap an explicit `module { ... }`.
    let mut module = p.module;
    if roots.len() == 1 && roots[0].kind == OpKind::Module {
        let m = roots.remove(0);
        module.attrs = m.attrs;
        module.ops = m.regions.into_iter().next().unwrap_or_default().ops;
    } else {
        module.ops = roots;
    }
    Ok(module)
}

struct Parser {
    module: Module,
    /// A STACK of name tables, one per open REGION.
    ///
    /// MLIR'S SSA NAMES ARE REGION-SCOPED, and the causal attention body proves it:
    /// its two sibling `scf.for` loops BOTH define `%k`, `%k_25`, `%start_n` and
    /// `%acc_20`. A single flat table interns those to ONE value each, so the two
    /// loops' key tiles become the same SSA value -- and then `count_uses` says 2
    /// where it should say 1, the `tt.trans` fold refuses to fire, and the KV cache
    /// doubles for a reason that looks like a pass bug and is a parser bug.
    ///
    /// A USE searches innermost-outward; a name not found anywhere is a DEFINITION and
    /// lands in the innermost scope. That handles both without the parser having to
    /// know which is which, because a use always names something already defined.
    scopes: Vec<HashMap<String, Ssa>>,
    /// `#map`/`#set`/`#loc` aliases, so an op referring to `#map1` gets the body.
    aliases: HashMap<String, String>,
}

impl Parser {
    /// `#map = affine_map<...>` / `#set = affine_set<...>` lines, gathered before
    /// the ops so a forward reference resolves. MLIR prints `#map` before the
    /// module and `#locN` after it, which is exactly why this is a separate pass
    /// over the text.
    fn collect_aliases(&mut self, text: &str) {
        for line in text.lines() {
            let t = line.trim();
            if !t.starts_with('#') {
                continue;
            }
            if let Some((k, v)) = t.split_once(" = ") {
                self.aliases.insert(k.trim().to_string(), v.trim().to_string());
            }
        }
    }

    fn resolve_alias(&self, s: &str) -> String {
        match self.aliases.get(s) {
            Some(v) => v.clone(),
            None => s.to_string(),
        }
    }

    /// A `%name` -> [`Ssa`]: found in the innermost enclosing scope that has it, or
    /// minted in the innermost scope.
    fn ssa(&mut self, name: &str) -> Ssa {
        let key = name.trim_start_matches('%').to_string();
        for scope in self.scopes.iter().rev() {
            if let Some(v) = scope.get(&key) {
                return *v;
            }
        }
        let v = self.module.fresh_named(&key);
        self.scopes.last_mut().expect("there is always a scope").insert(key, v);
        v
    }

    fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    fn pop_scope(&mut self) {
        // The outermost scope is the module's and is never popped.
        if self.scopes.len() > 1 {
            self.scopes.pop();
        }
    }

    fn parse_block_args(&mut self, rest: &str, lineno: usize) -> Result<BlockArgs> {
        // `bb0(%a: f16, %b: f16):`
        let open = rest.find('(');
        let Some(open) = open else { return Ok(Vec::new()) };
        let close = rest
            .rfind(')')
            .ok_or_else(|| err(lineno, "block label has no closing ')'"))?;
        let mut out = Vec::new();
        for part in split_top(&rest[open + 1..close], ',') {
            let part = part.trim();
            if part.is_empty() {
                continue;
            }
            let (name, ty) = part
                .split_once(':')
                .ok_or_else(|| err(lineno, format!("block argument without a type: {part}")))?;
            out.push((self.ssa(name.trim()), self.parse_type(ty.trim(), lineno)?));
        }
        Ok(out)
    }

    /// The closer line of a region-carrying op: `}`, `} -> tensor<64xf16>`,
    /// `}) : (tensor<64x64xf16>) -> tensor<64xf16>`, `} loc(...)`, and --
    /// crucially -- `} {work_division = array<i64: ...>}`.
    ///
    /// THE CLOSER CARRIES ATTRIBUTES. When an op's `assemblyFormat` puts `attr-dict`
    /// AFTER `$body` -- which `ktdf.corelet_plan` does
    /// (`` `pattern` `=` custom<CoreletPattern>($pattern) $body attr-dict ``) -- every
    /// attribute not named in the custom part prints on the CLOSING line. Reading only
    /// the opening line loses `work_division` entirely, and the golden then appears to
    /// lack a plan field that the C++ does emit and this port also emits: a difference
    /// reported against the pass whose cause is in the reader.
    fn apply_closer(&mut self, op: &mut Op, line: &str, lineno: usize) -> Result<()> {
        // A trailing `-> T` (or `-> (T, T)`) after the brace carries the result
        // type for tt.reduce / linalg.generic, whose results are declared there.
        if let Some(arrow) = line.rfind("->") {
            let tys = line[arrow + 2..].trim();
            if !tys.is_empty() && op.result_types.is_empty() && !op.results.is_empty() {
                for t in split_type_list(tys) {
                    op.result_types.push(self.parse_type(&t, lineno)?);
                }
            }
        }
        // The post-body attribute dictionary. Search AFTER the brace that closed the
        // region, so the `}` itself is not mistaken for a dictionary opener.
        if let Some(brace) = line.find('}') {
            let after = &line[brace + 1..];
            if let Some(start) = find_attr_dict(after) {
                let (inner, _) = balanced(&after[start..], '{', '}')
                    .ok_or_else(|| err(lineno, "post-body attribute dictionary is not balanced"))?;
                let inner = inner.to_string();
                for (k, v) in self.parse_attr_dict(&inner, lineno)? {
                    op.set_attr(k, v);
                }
            }
        }
        Ok(())
    }

    /// `opens_region` says whether the trailing `{` on this line opens a region, so
    /// the op's block arguments and body land in a NEW name scope while its own
    /// results stay in the enclosing one.
    fn parse_op(&mut self, line: &str, lineno: usize, opens_region: bool) -> Result<Op> {
        let (result_names, rhs) = split_results(line);
        let rhs = rhs.trim();

        // The op name: bare (`arith.addf`) or quoted-generic (`"tt.reduce"`).
        let (name, rest) = if let Some(stripped) = rhs.strip_prefix('"') {
            let close = stripped
                .find('"')
                .ok_or_else(|| err(lineno, "unterminated quoted op name"))?;
            (&stripped[..close], stripped[close + 1..].trim())
        } else {
            let end = rhs
                .find(|c: char| c.is_whitespace() || c == '(')
                .unwrap_or(rhs.len());
            (&rhs[..end], rhs[end..].trim())
        };
        if name.is_empty() {
            return Err(err(lineno, format!("cannot read an operation from: {line}")));
        }
        let kind = OpKind::from_spelling(name);
        let mut op = Op::new(kind.clone());

        // Results FIRST, and in the ENCLOSING scope: `%offsetv_y:5 = scf.for ...`
        // defines five values the code AFTER the loop reads.
        for r in result_names {
            if let Some((base, n)) = r.split_once(':') {
                let n: usize = n
                    .trim()
                    .parse()
                    .map_err(|_| err(lineno, format!("bad result count in `{r}`")))?;
                for k in 0..n {
                    let v = self.ssa(&format!("{}#{k}", base.trim()));
                    op.results.push(v);
                }
            } else {
                op.results.push(self.ssa(r));
            }
        }

        // Now open the region's scope, so the induction variable, the iter_args, a
        // combiner's block arguments and everything in the body are region-local.
        if opens_region {
            self.push_scope();
        }

        match kind {
            // `tt.get_program_id x : i32` -- the AXIS IS A BARE KEYWORD, not an
            // attribute. Reading it as an attribute leaves every axis at 0, and
            // DistributeWork then sees a single-axis kernel where there are two: the
            // multi-axis RED-stop never fires and the grid silently linearizes wrong.
            OpKind::TtGetProgramId => {
                let axis = match rest.trim_start().chars().next() {
                    Some('x') => 0,
                    Some('y') => 1,
                    Some('z') => 2,
                    other => {
                        return Err(err(
                            lineno,
                            format!("tt.get_program_id axis must be x, y or z, got {other:?}"),
                        ))
                    }
                };
                op.set_attr(AttrKey::Axis, Attr::Int(axis));
                op.result_types.push(IrType::Scalar(DType::I32));
            }
            // `arith.cmpi slt, %a, %b : i32` / `arith.cmpf oge, ...`. THE PREDICATE IS A
            // BARE KEYWORD, the same shape as `tt.get_program_id`'s axis: no `=`, no
            // dictionary. `arith.cmpf` has no `OpKind` variant, so it arrives as
            // `Other("arith.cmpf")` and is matched on its spelling.
            _ if kind == OpKind::ArithCmpi || kind.spelling() == "arith.cmpf" => {
                let head = rest.trim_start();
                let pred: String = head
                    .chars()
                    .take_while(|c| c.is_ascii_alphabetic())
                    .collect();
                if pred.is_empty() {
                    return Err(err(
                        lineno,
                        format!(
                            "`{}` has no comparison predicate -- MLIR prints it as a bare \
                             keyword before the operands (`slt`, `oge`), and a consumer \
                             that cannot read it would map every comparison to the same \
                             opFuncName: {rest}",
                            kind.spelling()
                        ),
                    ));
                }
                op.set_attr(AttrKey::Predicate, Attr::Str(pred.clone()));
                // The rest parses generically once the keyword is out of the way, so the
                // operands and result type come from one code path.
                let tail = head[pred.len()..].trim_start().trim_start_matches(',');
                self.parse_generic(&mut op, tail, lineno)?;
            }
            // BOTH function containers. `func.func` is what `ToSchedulerKTIR` produces,
            // and routing only `tt.func` here meant a stage-3 KTIR parsed with NO
            // `sym_name` and NO region arguments -- silently, because `parse_generic`
            // accepts the line. No fixture caught it: every `make_ktir` golden is still
            // `tt.func`. Found by hand-writing a `func.func` kernel for a refusal test in
            // triton-superdsc-lower, which then refused with "has no arguments".
            OpKind::TtFunc | OpKind::FuncFunc => self.parse_func(&mut op, rest, lineno)?,
            OpKind::ScfFor => self.parse_scf_for(&mut op, rest, lineno)?,
            OpKind::ArithConstant => self.parse_constant(&mut op, rest, lineno)?,
            OpKind::TtMakeTensorDescriptor => {
                self.parse_make_tensor_descriptor(&mut op, rest, lineno)?
            }
            OpKind::KtdpConstructMemoryView => {
                self.parse_construct_memory_view(&mut op, rest, lineno)?
            }
            // `ktdp.construct_indirect_access_tile %table captures(%r, %c : index, index)
            //    indirect(%ids : memref<256xsi32>) { ^bb0(...): } {attrs} :
            //    memref<49159x4096xf16> -> <128x4096xindex>`
            //
            // THE GENERIC READER GETS THIS ONE WRONG, and silently. Its type tail is
            // `rfind(" : ")` over the opening line, and `indirect(%ids : memref<256xsi32>)`
            // puts a ` : ` INSIDE the operand list -- so the op came out typed
            // `memref<256xsi32>`, the index view's type, and because `apply_closer` only
            // fills `result_types` when it is EMPTY, the real `-> <128x4096xindex>` on the
            // closing line was then never read. A consumer asking for the gathered tile's
            // shape got the index vector's instead.
            OpKind::KtdpConstructIndirectAccessTile => {
                self.parse_indirect_access_tile(&mut op, rest, lineno)?
            }
            OpKind::KtdfCoreletPlan => self.parse_corelet_plan(&mut op, rest, lineno)?,
            OpKind::KtdfCorelet => self.parse_corelet(&mut op, rest, lineno)?,
            _ => self.parse_generic(&mut op, rest, lineno)?,
        }
        Ok(op)
    }

    /// `tt.func public @attn_fwd(%a: !tt.ptr<f16>, ...) attributes {noinline = false} {`
    fn parse_func(&mut self, op: &mut Op, rest: &str, lineno: usize) -> Result<()> {
        let at = rest
            .find('@')
            .ok_or_else(|| err(lineno, format!("{} has no @name", op.kind.spelling())))?;
        let after = &rest[at + 1..];
        let end = after
            .find(|c: char| !(c.is_alphanumeric() || c == '_' || c == '.' || c == '$'))
            .unwrap_or(after.len());
        op.set_attr(AttrKey::SymName, Attr::Str(after[..end].to_string()));

        let open = at + 1 + end;
        let sig = &rest[open..];
        let (inner, _) = balanced(sig, '(', ')')
            .ok_or_else(|| err(lineno, "tt.func signature is not balanced"))?;
        let mut args = Vec::new();
        for part in split_top(inner, ',') {
            let part = part.trim();
            if part.is_empty() {
                continue;
            }
            let (name, ty) = part
                .split_once(':')
                .ok_or_else(|| err(lineno, format!("function argument without a type: {part}")))?;
            args.push((self.ssa(name.trim()), self.parse_type(ty.trim(), lineno)?));
        }
        // The `attributes { ... }` dictionary. `func.func` carries `grid` there -- the
        // 1-D launch extent the scheduler reads -- so skipping the dictionary makes the
        // one attribute this stage exists to set invisible.
        let after_sig = &rest[open..];
        if let Some(at) = after_sig.find("attributes") {
            if let Some((inner, _)) = balanced(&after_sig[at..], '{', '}') {
                let inner = inner.to_string();
                for (k, v) in self.parse_attr_dict(&inner, lineno)? {
                    op.set_attr(k, v);
                }
            }
        }

        // The body region is created HERE, carrying the signature as its block
        // arguments; `parse` tops up to the brace count rather than pushing a
        // second, argument-less one.
        op.regions.push(Region { args, ops: Vec::new() });
        Ok(())
    }

    /// `scf.for %i = %lb to %ub step %st iter_args(%a = %x, ...) -> (T, ...) : i32 {`
    fn parse_scf_for(&mut self, op: &mut Op, rest: &str, lineno: usize) -> Result<()> {
        let eq = rest
            .find('=')
            .ok_or_else(|| err(lineno, "scf.for has no induction variable"))?;
        let iv = self.ssa(rest[..eq].trim());
        let after = &rest[eq + 1..];
        let to = after
            .find(" to ")
            .ok_or_else(|| err(lineno, "scf.for has no ` to `"))?;
        let step = after
            .find(" step ")
            .ok_or_else(|| err(lineno, "scf.for has no ` step `"))?;
        let lb = self.ssa(after[..to].trim());
        let ub = self.ssa(after[to + 4..step].trim());
        let tail = &after[step + 6..];
        // The step operand ends at the next space or `iter_args`.
        let step_end = tail.find(char::is_whitespace).unwrap_or(tail.len());
        let st = self.ssa(tail[..step_end].trim());
        op.operands = vec![lb, ub, st];

        let mut region = Region { args: vec![(iv, IrType::Verbatim("index".into()))], ops: vec![] };

        if let Some(ia) = tail.find("iter_args") {
            let (inner, _) = balanced(&tail[ia..], '(', ')')
                .ok_or_else(|| err(lineno, "iter_args is not balanced"))?;
            for part in split_top(inner, ',') {
                let part = part.trim();
                if part.is_empty() {
                    continue;
                }
                let (arg, init) = part
                    .split_once('=')
                    .ok_or_else(|| err(lineno, format!("iter_arg without an init: {part}")))?;
                let a = self.ssa(arg.trim());
                let v = self.ssa(init.trim());
                region.args.push((a, IrType::Verbatim("?".into())));
                op.operands.push(v);
            }
        }

        // `-> (T, T, ...)` gives both the result types and the iter_arg types.
        if let Some(arrow) = rest.find("->") {
            let after_arrow = &rest[arrow + 2..];
            let tys = match balanced(after_arrow, '(', ')') {
                Some((inner, _)) => split_type_list(inner),
                None => {
                    let end = after_arrow.find(':').unwrap_or(after_arrow.len());
                    split_type_list(after_arrow[..end].trim())
                }
            };
            for t in &tys {
                op.result_types.push(self.parse_type(t, lineno)?);
            }
            // iter_arg types are the result types, in order, after the IV.
            for (i, t) in op.result_types.iter().enumerate() {
                if let Some(slot) = region.args.get_mut(i + 1) {
                    slot.1 = t.clone();
                }
            }
        }
        // The IV's type is the trailing `: i32` (or index when absent).
        if let Some(colon) = rest.rfind(" : ") {
            let t = rest[colon + 3..].trim().trim_end_matches('{').trim();
            if !t.is_empty() {
                region.args[0].1 = self.parse_type(t, lineno)?;
            }
        }
        op.regions.push(region);
        Ok(())
    }

    /// `tt.make_tensor_descriptor %base, [%s0, %s1], [%t0, %t1] : <f16>, <64x128xf16>`
    ///
    /// The type tail holds TWO abbreviated types: the pointee (`<f16>`) and the
    /// descriptor's BLOCK type (`<64x128xf16>`). The block type is the one the
    /// descriptor patterns read -- `ConvertDescriptorLoad` takes the block shape from
    /// the descriptor's type, never the load's result, so that a rank-reduced load
    /// fails verification instead of silently building a mismatched tile.
    fn parse_make_tensor_descriptor(
        &mut self,
        op: &mut Op,
        rest: &str,
        lineno: usize,
    ) -> Result<()> {
        let colon = rest
            .rfind(" : ")
            .ok_or_else(|| err(lineno, "tt.make_tensor_descriptor has no type tail"))?;
        for tok in ssa_tokens(&rest[..colon]) {
            let v = self.ssa(&tok);
            op.operands.push(v);
        }
        let tys = split_type_list(rest[colon + 3..].trim());
        let block = tys
            .last()
            .ok_or_else(|| err(lineno, "tt.make_tensor_descriptor has no block type"))?;
        // The abbreviated `<64x128xf16>` form; `parse_type` resolves it by element
        // type, so a descriptor is never confused with an access tile.
        op.result_types.push(self.parse_type(block, lineno)?);
        Ok(())
    }

    /// `ktdp.construct_memory_view %base, sizes: [512, 128], strides: [128, 1]
    ///  {coordinate_set = #set, memory_space = ...} : memref<512x128xf16>`
    ///
    /// `sizes:` and `strides:` are in the CUSTOM format, NOT the attribute dictionary.
    /// Reading only the dictionary leaves both empty, and `PlanCorelets` then cannot
    /// recover the matmul's full shape -- which surfaces as a missing `work_division`
    /// three passes later.
    fn parse_construct_memory_view(
        &mut self,
        op: &mut Op,
        rest: &str,
        lineno: usize,
    ) -> Result<()> {
        if let Some(at) = rest.find("sizes:") {
            let (inner, _) = balanced(&rest[at..], '[', ']')
                .ok_or_else(|| err(lineno, "`sizes:` list is not balanced"))?;
            op.set_attr(AttrKey::Shape, Attr::IntList(parse_int_list(inner)));
        }
        if let Some(at) = rest.find("strides:") {
            let (inner, _) = balanced(&rest[at..], '[', ']')
                .ok_or_else(|| err(lineno, "`strides:` list is not balanced"))?;
            op.set_attr(AttrKey::Strides, Attr::IntList(parse_int_list(inner)));
        }
        // THE BASE, THEN ANY DYNAMIC EXTENTS. A `%token` inside `sizes: [...]` is a RUNTIME
        // extent passed as an operand -- `sizes: [%a_desc_4]` over a `memref<?xf16>` -- and
        // reading only the base silently dropped it. That is the same class of defect as
        // `construct_indirect_access_tile` being typed with its index vector's type: the
        // reader loses a real operand and nothing errors.
        //
        // The scan stops at `strides:` because a stride is always static in this dialect; a
        // `%token` there would be a form neither side emits, and letting it through as an
        // operand would be a guess.
        let head = rest.find("strides:").unwrap_or(rest.len());
        for tok in ssa_tokens(&rest[..head]) {
            let v = self.ssa(&tok);
            op.operands.push(v);
        }
        if let Some(start) = find_attr_dict(rest) {
            let (inner, _) = balanced(&rest[start..], '{', '}')
                .ok_or_else(|| err(lineno, "attribute dictionary is not balanced"))?;
            let inner = inner.to_string();
            for (k, v) in self.parse_attr_dict(&inner, lineno)? {
                op.set_attr(k, v);
            }
        }
        if let Some(colon) = rest.rfind(" : ") {
            op.result_types.push(self.parse_type(rest[colon + 3..].trim(), lineno)?);
        }
        Ok(())
    }

    /// `ktdf.corelet_plan pattern = "independent_rows" {`
    ///
    /// `pattern` is in the CUSTOM format. Reading only the dictionary leaves the plan
    /// with no pattern at all -- so a diff cannot tell `independent_rows` from
    /// `split`, which is the single most load-bearing field the plan carries.
    fn parse_corelet_plan(&mut self, op: &mut Op, rest: &str, lineno: usize) -> Result<()> {
        if let Some(at) = rest.find("pattern") {
            let after = &rest[at..];
            if let Some(eq) = after.find('=') {
                let v = after[eq + 1..].trim();
                let v = v.trim_start_matches('"');
                let end = v.find('"').unwrap_or(v.len());
                op.set_attr(AttrKey::Pattern, Attr::Str(v[..end].to_string()));
            }
        }
        if let Some(at) = rest.find("work_division") {
            if let Some((inner, _)) = balanced(&rest[at..], '[', ']') {
                op.set_attr(AttrKey::WorkDivision, Attr::IntList(parse_int_list(inner)));
            }
        }
        let _ = lineno;
        Ok(())
    }

    /// `ktdf.corelet 0 {data_bounds = [0, 32]}` -- the INDEX is a bare integer in the
    /// custom format, and the rest is a dictionary.
    fn parse_corelet(&mut self, op: &mut Op, rest: &str, lineno: usize) -> Result<()> {
        let head = rest.split_whitespace().next().unwrap_or("");
        if let Some(i) = parse_int(head) {
            op.set_attr(AttrKey::Index, Attr::Int(i));
        }
        if let Some(start) = find_attr_dict(rest) {
            let (inner, _) = balanced(&rest[start..], '{', '}')
                .ok_or_else(|| err(lineno, "attribute dictionary is not balanced"))?;
            let inner = inner.to_string();
            for (k, v) in self.parse_attr_dict(&inner, lineno)? {
                op.set_attr(k, v);
            }
        }
        Ok(())
    }

    /// `arith.constant 0 : i32`, `arith.constant dense<0.0> : tensor<64xf16>`,
    /// `arith.constant 1.275630e-01 : f16`, `arith.constant 0xFC00 : f16`.
    /// `ktdp.construct_indirect_access_tile`: operands only, type from the CLOSER.
    ///
    /// The operands are read in printed order -- base, the `captures(...)` offsets, then
    /// the `indirect(...)` index view last -- and the result type is deliberately left
    /// empty so [`Parser::apply_closer`] takes it off the `-> T` after the region's brace.
    /// The attribute dictionary is also on the closing line, which `apply_closer` already
    /// handles.
    fn parse_indirect_access_tile(
        &mut self,
        op: &mut Op,
        rest: &str,
        lineno: usize,
    ) -> Result<()> {
        for tok in ssa_tokens(rest) {
            let v = self.ssa(&tok);
            op.operands.push(v);
        }
        if op.operands.len() < 2 {
            return Err(err(
                lineno,
                format!(
                    "ktdp.construct_indirect_access_tile needs at least a base and an \
                     `indirect(...)` index view, found {} operand(s): {rest}",
                    op.operands.len()
                ),
            ));
        }
        Ok(())
    }

    fn parse_constant(&mut self, op: &mut Op, rest: &str, lineno: usize) -> Result<()> {
        let colon = rest
            .rfind(" : ")
            .ok_or_else(|| err(lineno, format!("arith.constant without a type: {rest}")))?;
        let val = rest[..colon].trim();
        let ty = self.parse_type(rest[colon + 3..].trim(), lineno)?;
        let elem = ty.elem().unwrap_or(DType::F32);

        let attr = if let Some(inner) = val.strip_prefix("dense<") {
            let body = inner.trim_end_matches('>').trim();
            // Only a SPLAT dense constant is modelled: it is the only form
            // DecomposeDenseConstants rewrites, and a non-splat is left alone by
            // the C++ too -- so it is kept verbatim rather than half-read.
            if body.contains('[') {
                Attr::Verbatim(val.to_string())
            } else {
                Attr::SplatFloat(parse_float(body, elem, lineno)?)
            }
        } else if elem.is_float() {
            Attr::Float(parse_float(val, elem, lineno)?)
        } else {
            Attr::Int(
                parse_int(val)
                    .ok_or_else(|| err(lineno, format!("cannot read integer constant `{val}`")))?,
            )
        };
        op.set_attr(AttrKey::Value, attr);
        op.result_types.push(ty);
        Ok(())
    }

    /// Everything else: operands are the `%tokens`, the trailing `: ... -> T`
    /// gives the result type, and a `{...}` / `<{...}>` dictionary gives attrs.
    fn parse_generic(&mut self, op: &mut Op, rest: &str, lineno: usize) -> Result<()> {
        // Attribute dictionaries, inline `<{...}>` (properties) and `{...}`.
        let mut body = rest.to_string();
        while let Some(start) = find_attr_dict(&body) {
            let (inner, end) = balanced(&body[start..], '{', '}')
                .ok_or_else(|| err(lineno, "attribute dictionary is not balanced"))?;
            let inner = inner.to_string();
            for (k, v) in self.parse_attr_dict(&inner, lineno)? {
                op.set_attr(k, v);
            }
            let mut cut_end = start + end;
            // Swallow the `>` of a `<{...}>` property list.
            if body[cut_end..].starts_with('>') {
                cut_end += 1;
            }
            let mut cut_start = start;
            if body[..cut_start].ends_with('<') {
                cut_start -= 1;
            }
            body.replace_range(cut_start..cut_end, "");
        }

        // Result types: `-> T` (last arrow), else the trailing `: T`.
        //
        // An op with NO OPERANDS puts the colon first, with nothing before it:
        // `ktdp.get_compute_tile_id : index` leaves `rest` as `": index"`, which has
        // no ` : ` in it at all. Missing that made the landmark parse with no result
        // type -- and the landmark's type is `index`, which is what the work loop's
        // lower bound is.
        let trimmed = body.trim_start();
        let type_tail = if let Some(a) = body.rfind("->") {
            Some(body[a + 2..].trim().to_string())
        } else if let Some(t) = trimmed.strip_prefix(':') {
            Some(t.trim().to_string())
        } else { body.rfind(" : ").map(|c| body[c + 3..].trim().to_string()) };
        if let Some(t) = type_tail {
            let mut t = t.trim_end_matches('{').trim().to_string();
            // `arith.extf %x : tensor<..xf16> to tensor<..xf32>` states the OPERAND
            // type then the RESULT type, separated by ` to `. Taking the whole tail
            // yields an unparseable type that falls back to Verbatim -- and then
            // LegalizeTypes cannot see that the result is f32, so the widening island
            // never collapses. The RESULT is the part after the last ` to `.
            if let Some(at) = t.rfind(" to ") {
                t = t[at + 4..].trim().to_string();
            }
            if !t.is_empty() {
                let tys = match balanced(&t, '(', ')') {
                    Some((inner, _)) if t.starts_with('(') => split_type_list(inner),
                    _ => split_type_list(&t),
                };
                for ty in tys {
                    op.result_types.push(self.parse_type(&ty, lineno)?);
                }
            }
        }
        // Trim the result types to the declared result count when the op declares
        // its operand types in the same tail (`: (A, B) -> C` handled above, but
        // `%x = ktdp.load %t : <64x128xindex> -> tensor<...>` has one of each).
        if op.results.is_empty() {
            // A terminator or a store: every type in the tail is an OPERAND type, and
            // keeping them as result types would make `scf.yield` look like it defines
            // five values.
            op.result_types.clear();
        } else if op.result_types.len() > op.results.len() {
            let extra = op.result_types.len() - op.results.len();
            op.result_types.drain(..extra);
        }

        // `ktdp.construct_access_tile`'s CUSTOM PRINTER ELIDES an identity `base_map`.
        // The op carries one regardless -- the generic form in `3_sched.mlir` shows it
        // -- so synthesize it here, or the golden's op and ours differ on a field
        // neither the C++ nor this port actually disagrees about.
        if op.kind == OpKind::KtdpConstructAccessTile && op.attr(&AttrKey::BaseMap).is_none() {
            let rank = op.result_type().map(|t| t.rank()).unwrap_or(0);
            let d: Vec<String> = (0..rank).map(|i| format!("d{i}")).collect();
            op.set_attr(
                AttrKey::BaseMap,
                Attr::AffineMap(format!("({}) -> ({})", d.join(", "), d.join(", "))),
            );
        }

        // Operands: every `%token` in the pre-type part, in order.
        let cut = body
            .rfind("->")
            .or_else(|| body.rfind(" : "))
            .unwrap_or(body.len());
        for tok in ssa_tokens(&body[..cut]) {
            let v = self.ssa(&tok);
            op.operands.push(v);
        }
        Ok(())
    }

    fn parse_attr_dict(&mut self, inner: &str, lineno: usize) -> Result<Vec<(AttrKey, Attr)>> {
        let mut out = Vec::new();
        for part in split_top(inner, ',') {
            let part = part.trim();
            if part.is_empty() {
                continue;
            }
            let (k, v) = match part.split_once('=') {
                Some((k, v)) => (k.trim(), v.trim()),
                None => (part, ""), // a unit attribute
            };
            let key = AttrKey::from_spelling(k);
            out.push((key, self.parse_attr_value(v, lineno)?));
        }
        Ok(out)
    }

    fn parse_attr_value(&mut self, v: &str, lineno: usize) -> Result<Attr> {
        let v = v.trim();
        if v.is_empty() {
            return Ok(Attr::Unit);
        }
        if v == "true" || v == "false" {
            return Ok(Attr::Bool(v == "true"));
        }
        if let Some(s) = v.strip_prefix('"') {
            return Ok(Attr::Str(s.trim_end_matches('"').to_string()));
        }
        if v.starts_with("#map") || v.starts_with("affine_map<") {
            return Ok(Attr::AffineMap(strip_wrapper(&self.resolve_alias(v), "affine_map<")));
        }
        if v.starts_with("#set") || v.starts_with("affine_set<") {
            return Ok(Attr::AffineSet(strip_wrapper(&self.resolve_alias(v), "affine_set<")));
        }
        // `array<i32: 1, 0>` / `array<i64: 0, 32>`
        if let Some(inner) = v.strip_prefix("array<") {
            let body = inner.trim_end_matches('>');
            let list = body.split_once(':').map(|(_, r)| r).unwrap_or(body);
            let mut out = Vec::new();
            for t in list.split(',') {
                let t = t.trim();
                if t.is_empty() {
                    continue;
                }
                out.push(
                    parse_int(t)
                        .ok_or_else(|| err(lineno, format!("bad array element `{t}`")))?,
                );
            }
            return Ok(Attr::IntList(out));
        }
        // `[...]` -- a list of maps, ints or strings.
        if let Some(inner) = v.strip_prefix('[') {
            let body = inner.trim_end_matches(']');
            let parts: Vec<String> = split_top(body, ',')
                .into_iter()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            if parts.iter().all(|p| p.starts_with("#map") || p.starts_with("affine_map<")) {
                return Ok(Attr::AffineMapList(
                    parts
                        .iter()
                        .map(|p| strip_wrapper(&self.resolve_alias(p), "affine_map<"))
                        .collect(),
                ));
            }
            if parts.iter().all(|p| parse_int(p).is_some()) {
                return Ok(Attr::IntList(parts.iter().filter_map(|p| parse_int(p)).collect()));
            }
            // `[8 : index]` -- a typed integer list, which is how `grid` prints.
            if parts.iter().all(|p| p.split_once(" : ").and_then(|(n, _)| parse_int(n)).is_some())
            {
                return Ok(Attr::IntList(
                    parts
                        .iter()
                        .filter_map(|p| p.split_once(" : ").and_then(|(n, _)| parse_int(n)))
                        .collect(),
                ));
            }
            // `#linalg.iterator_type<parallel>` -> `parallel`. Keeping the wrapper means
            // nothing can ask whether an axis is a reduction, which is the one question
            // an iterator list exists to answer.
            return Ok(Attr::StrList(
                parts.iter().map(|p| unwrap_enum_attr(p)).collect(),
            ));
        }
        // `1 : i32` -- an integer with its type.
        if let Some((num, _ty)) = v.split_once(" : ") {
            if let Some(i) = parse_int(num.trim()) {
                return Ok(Attr::Int(i));
            }
        }
        if let Some(i) = parse_int(v) {
            return Ok(Attr::Int(i));
        }
        // `memory_space = #ktdp.spyre_memory_space<HBM>` -- the SPACE NAME is what the
        // pass sets and what any consumer reads; keeping the whole spelling would make
        // the golden and our own emission differ on formatting rather than meaning.
        if let Some(inner) = v.strip_prefix("#ktdp.spyre_memory_space<") {
            return Ok(Attr::Str(inner.trim_end_matches('>').to_string()));
        }
        Ok(Attr::Verbatim(self.resolve_alias(v)))
    }

    fn parse_type(&mut self, t: &str, lineno: usize) -> Result<IrType> {
        let t = t.trim();
        if let Some(d) = DType::from_spelling(t) {
            return Ok(if t == "index" { IrType::Index } else { IrType::Scalar(d) });
        }
        if let Some(inner) = t.strip_prefix("tensor<") {
            let (dims, elem) = parse_shaped(inner)
                .ok_or_else(|| err(lineno, format!("cannot read tensor type `{t}`")))?;
            let d = DType::from_spelling(&elem)
                .ok_or_else(|| err(lineno, format!("unmodelled element type `{elem}` in `{t}`")))?;
            return Ok(IrType::Tensor { dims, elem: d });
        }
        if let Some(inner) = t.strip_prefix("memref<") {
            let (dims, elem) = parse_shaped(inner)
                .ok_or_else(|| err(lineno, format!("cannot read memref type `{t}`")))?;
            let d = DType::from_spelling(&elem)
                .ok_or_else(|| err(lineno, format!("unmodelled element type `{elem}` in `{t}`")))?;
            return Ok(IrType::MemRef { dims, elem: d });
        }
        if let Some(inner) = t.strip_prefix("!ktdp.access_tile<") {
            let (dims, _elem) = parse_shaped(inner)
                .ok_or_else(|| err(lineno, format!("cannot read access_tile type `{t}`")))?;
            return Ok(IrType::AccessTile { dims });
        }
        // THE ABBREVIATED FORM. `ktdp.load %t : <64x128xindex> -> ...` and
        // `tt.make_tensor_descriptor ... : <f16>, <64x128xf16>` both print a type with
        // the dialect prefix elided. Which type it is follows from the ELEMENT type:
        // `index` means an access tile (an access tile INDEXES rather than holds, so
        // `index` is the only element type it has), anything else means a tensordesc
        // block type.
        if let Some(inner) = t.strip_prefix('<') {
            if let Some((dims, elem)) = parse_shaped(inner) {
                if elem == "index" {
                    return Ok(IrType::AccessTile { dims });
                }
                if let Some(d) = DType::from_spelling(&elem) {
                    return Ok(if dims.is_empty() {
                        IrType::Ptr { elem: d }
                    } else {
                        IrType::TensorDesc { dims, elem: d }
                    });
                }
            }
        }
        if let Some(inner) = t.strip_prefix("!tt.ptr<") {
            let e = inner.trim_end_matches('>');
            let d = DType::from_spelling(e)
                .ok_or_else(|| err(lineno, format!("unmodelled pointee `{e}`")))?;
            return Ok(IrType::Ptr { elem: d });
        }
        if let Some(inner) = t.strip_prefix("!tt.tensordesc<") {
            let (dims, elem) = parse_shaped(inner)
                .ok_or_else(|| err(lineno, format!("cannot read tensordesc type `{t}`")))?;
            let d = DType::from_spelling(&elem)
                .ok_or_else(|| err(lineno, format!("unmodelled element type `{elem}`")))?;
            return Ok(IrType::TensorDesc { dims, elem: d });
        }
        Ok(IrType::Verbatim(t.to_string()))
    }
}

//===----------------------------------------------------------------------===//
// text helpers
//===----------------------------------------------------------------------===//

/// Strip every `loc(...)` (balanced, possibly nested) and drop `#locN = ` lines.
pub fn strip_locs(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for line in text.lines() {
        if line.trim_start().starts_with("#loc") {
            continue;
        }
        let mut s = line;
        let mut buf = String::new();
        loop {
            match find_call(s, "loc(") {
                None => {
                    buf.push_str(s);
                    break;
                }
                Some((a, b)) => {
                    buf.push_str(&s[..a]);
                    s = &s[b..];
                }
            }
        }
        out.push_str(buf.trim_end());
        out.push('\n');
    }
    out
}

/// Byte range of the first `name(...)` in `s`, matching parens, not preceded by
/// an identifier character.
fn find_call(s: &str, name: &str) -> Option<(usize, usize)> {
    let b = s.as_bytes();
    let n = name.len();
    let mut i = 0;
    while i + n <= b.len() {
        if &b[i..i + n] == name.as_bytes()
            && (i == 0 || !(b[i - 1].is_ascii_alphanumeric() || b[i - 1] == b'_' || b[i - 1] == b'.'))
        {
            let mut depth = 0usize;
            let mut j = i + n - 1;
            let mut in_str = false;
            while j < b.len() {
                match b[j] {
                    b'"' if j == 0 || b[j - 1] != b'\\' => in_str = !in_str,
                    b'(' if !in_str => depth += 1,
                    b')' if !in_str => {
                        depth -= 1;
                        if depth == 0 {
                            return Some((i, j + 1));
                        }
                    }
                    _ => {}
                }
                j += 1;
            }
            return None;
        }
        i += 1;
    }
    None
}

/// Net `{` minus `}`, ignoring string literals.
fn brace_delta(line: &str) -> i32 {
    let b = line.as_bytes();
    let mut d = 0i32;
    let mut in_str = false;
    for i in 0..b.len() {
        match b[i] {
            b'"' if i == 0 || b[i - 1] != b'\\' => in_str = !in_str,
            b'{' if !in_str => d += 1,
            b'}' if !in_str => d -= 1,
            _ => {}
        }
    }
    d
}

/// Split at the top-level `=` separating results from the op. Returns the result
/// tokens and the rest.
fn split_results(line: &str) -> (Vec<&str>, &str) {
    if !line.starts_with('%') {
        return (Vec::new(), line);
    }
    let b = line.as_bytes();
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'(' | b'<' | b'{' | b'"' | b'[' => return (Vec::new(), line),
            b'=' if b.get(i + 1) != Some(&b'=') && b.get(i + 1) != Some(&b'>') => {
                let names = line[..i]
                    .split(',')
                    .map(|s| s.trim())
                    .filter(|s| !s.is_empty())
                    .collect();
                return (names, &line[i + 1..]);
            }
            _ => i += 1,
        }
    }
    (Vec::new(), line)
}

/// The contents of the first balanced `open..close` group in `s`, plus the index
/// just past the closer.
fn balanced(s: &str, open: char, close: char) -> Option<(&str, usize)> {
    let b = s.as_bytes();
    let start = s.find(open)?;
    let mut depth = 0i32;
    let mut in_str = false;
    for i in start..b.len() {
        let c = b[i] as char;
        if c == '"' && (i == 0 || b[i - 1] != b'\\') {
            in_str = !in_str;
        }
        if in_str {
            continue;
        }
        // Same `->` trap as `split_top`: an arrow's `>` is not a bracket.
        if close == '>' && c == '>' && i > 0 && b[i - 1] == b'-' {
            continue;
        }
        if c == open {
            depth += 1;
        } else if c == close {
            depth -= 1;
            if depth == 0 {
                return Some((&s[start + 1..i], i + 1));
            }
        }
    }
    None
}

/// Split on `sep` at nesting depth zero (parens, angles, braces, brackets).
///
/// THE `->` TRAP, and it is not hypothetical -- it silently mangled every attribute
/// that followed an affine map. An affine map prints as `affine_map<(d0) -> (d0)>`,
/// so counting every `>` as a closer takes the depth NEGATIVE at the arrow: the
/// comma after the map is then not at depth zero, the split does not happen, and
/// `access_tile_order`'s value swallows `base_map = ...` whole. The op comes out with
/// one attribute where it should have two, and the missing one reappears inside the
/// other's text.
///
/// So a `>` immediately preceded by `-` is part of an arrow, not a bracket.
fn split_top(s: &str, sep: char) -> Vec<&str> {
    let b = s.as_bytes();
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut in_str = false;
    let mut start = 0usize;
    for i in 0..b.len() {
        let c = b[i] as char;
        if c == '"' && (i == 0 || b[i - 1] != b'\\') {
            in_str = !in_str;
        }
        if in_str {
            continue;
        }
        let is_arrow = c == '>' && i > 0 && b[i - 1] == b'-';
        match c {
            '(' | '<' | '{' | '[' => depth += 1,
            '>' if is_arrow => {}
            ')' | '>' | '}' | ']' => depth -= 1,
            _ if c == sep && depth == 0 => {
                out.push(&s[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    out.push(&s[start..]);
    out
}

/// Split a type list, respecting nesting. `(A, B)` should be unwrapped first.
fn split_type_list(s: &str) -> Vec<String> {
    let s = s.trim();
    let s = if s.starts_with('(') && s.ends_with(')') { &s[1..s.len() - 1] } else { s };
    split_top(s, ',')
        .into_iter()
        .map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty())
        .collect()
}

/// The start of an attribute dictionary in `s`, or `None`. A `{` that opens a
/// REGION is at end of line, so a dictionary is one with a matching `}` on the
/// same line.
fn find_attr_dict(s: &str) -> Option<usize> {
    let b = s.as_bytes();
    let mut in_str = false;
    for i in 0..b.len() {
        match b[i] {
            b'"' if i == 0 || b[i - 1] != b'\\' => in_str = !in_str,
            b'{' if !in_str => {
                // Balanced on this line => a dictionary, not a region opener.
                if balanced(&s[i..], '{', '}').is_some() {
                    return Some(i);
                }
                return None;
            }
            _ => {}
        }
    }
    None
}

/// Every `%name` (or `%name#k`) token, in order, duplicates kept.
fn ssa_tokens(s: &str) -> Vec<String> {
    let b = s.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' {
            let start = i;
            i += 1;
            while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_' || b[i] == b'#') {
                i += 1;
            }
            if i > start + 1 {
                out.push(s[start + 1..i].to_string());
            }
        } else {
            i += 1;
        }
    }
    out
}

/// Shape + element type from `64x128xf16>`.
///
/// Peels leading `<digits>x` groups rather than splitting on every `x`:
/// `!ktdp.access_tile<64x512xindex>` has an `x` INSIDE its element type, and a
/// naive split turns `index` into `inde`.
pub fn parse_shaped(body: &str) -> Option<(Vec<i64>, String)> {
    let end = body.find('>')?;
    let inner = &body[..end];
    let mut dims = Vec::new();
    let mut rest = inner;
    loop {
        let run: String = rest.chars().take_while(|c| c.is_ascii_digit() || *c == '?').collect();
        if run.is_empty() || !rest[run.len()..].starts_with('x') {
            break;
        }
        dims.push(if run == "?" { crate::ir::DYNAMIC } else { run.parse::<i64>().ok()? });
        rest = &rest[run.len() + 1..];
    }
    Some((dims, rest.to_string()))
}

/// `#dialect.thing<value>` -> `value`; anything else unchanged.
fn unwrap_enum_attr(s: &str) -> String {
    let t = s.trim();
    // A quoted list element: `"reduction"` -> `reduction`. Keeping the quotes makes
    // every comparison against an iterator name fail silently.
    if let Some(inner) = t.strip_prefix('"') {
        return inner.trim_end_matches('"').to_string();
    }
    if !t.starts_with('#') {
        return t.to_string();
    }
    match (t.find('<'), t.rfind('>')) {
        (Some(a), Some(b)) if b > a => t[a + 1..b].to_string(),
        _ => t.to_string(),
    }
}

fn strip_wrapper(s: &str, prefix: &str) -> String {
    match s.strip_prefix(prefix) {
        Some(r) => r.trim_end_matches('>').to_string(),
        None => s.to_string(),
    }
}

/// `[512, 128]`'s contents.
fn parse_int_list(inner: &str) -> Vec<i64> {
    inner.split(',').filter_map(|t| parse_int(t.trim())).collect()
}

fn parse_int(s: &str) -> Option<i64> {
    let s = s.trim();
    if let Some(h) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        return i64::from_str_radix(h, 16).ok();
    }
    s.parse::<i64>().ok()
}

/// A float literal, including MLIR's hex form (`0xFC00` for -inf as f16).
fn parse_float(s: &str, elem: DType, lineno: usize) -> Result<FloatBits> {
    let s = s.trim();
    let width = match elem {
        DType::F16 => 16,
        _ => 32,
    };
    if let Some(h) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        let bits = u64::from_str_radix(h, 16)
            .map_err(|_| err(lineno, format!("bad hex float `{s}`")))?;
        return Ok(FloatBits { bits, width });
    }
    let v: f32 = s
        .parse()
        .map_err(|_| err(lineno, format!("cannot read float `{s}`")))?;
    Ok(match elem {
        DType::F16 => FloatBits::f16_from_f32(v),
        _ => FloatBits::f32(v),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_post_body_attribute_dictionary_is_read_from_the_closing_line() {
        // `ktdf.corelet_plan`'s assemblyFormat is `pattern = ... $body attr-dict`, so
        // `work_division` prints AFTER the body's closing brace. Reading only the
        // opening line silently loses it.
        let src = "\
module {
  tt.func public @k(%q: !tt.ptr<f16>) attributes {noinline = false} {
    ktdf.corelet_plan pattern = \"independent_subtile\" {
      ktdf.corelet 0 {output_partition = [0, 32], pt_rows = [0, 7], xrf_capacity = 64}
      ktdf.corelet 1 {output_partition = [32, 64], pt_rows = [0, 7], xrf_capacity = 64}
    } {work_division = array<i64: 64, 256, 128, 64, 4, 8, 32, 16, 8, 1>}
    tt.return
  }
}
";
        let m = parse(src).unwrap();
        let plan = m
            .ops_deep()
            .into_iter()
            .find(|o| o.kind == OpKind::KtdfCoreletPlan)
            .expect("the plan is read");
        assert_eq!(
            plan.attr(&AttrKey::Pattern).and_then(|a| a.as_str()),
            Some("independent_subtile")
        );
        assert_eq!(
            plan.attr(&AttrKey::WorkDivision),
            Some(&Attr::IntList(vec![64, 256, 128, 64, 4, 8, 32, 16, 8, 1])),
            "the post-body attr-dict must be read"
        );
        assert_eq!(plan.regions[0].ops.len(), 2, "both corelets");
        assert_eq!(
            plan.regions[0].ops[1].attr(&AttrKey::OutputPartition),
            Some(&Attr::IntList(vec![32, 64]))
        );
        assert_eq!(
            plan.regions[0].ops[1].attr(&AttrKey::XrfCapacity),
            Some(&Attr::Int(64))
        );
    }

    #[test]
    fn an_affine_maps_arrow_does_not_end_the_attribute() {
        // The regression: counting `->`'s `>` as a closing bracket makes the comma
        // after the first map invisible, so `base_map` is swallowed into
        // `access_tile_order`'s value and the op loses an attribute.
        let parts = split_top(
            "access_tile_order = affine_map<(d0) -> (d0)>, base_map = affine_map<(d0) -> (d0)>",
            ',',
        );
        assert_eq!(parts.len(), 2, "got {parts:?}");
        assert!(parts[1].trim().starts_with("base_map"));
    }

    #[test]
    fn a_map_list_splits_into_its_maps() {
        let parts = split_top(
            "affine_map<(d0, d1, d2) -> (d0, d2, 0)>, affine_map<(d0, d1, d2) -> (d2, d1)>",
            ',',
        );
        assert_eq!(parts.len(), 2, "got {parts:?}");
    }

    #[test]
    fn locs_are_stripped_including_nested_callsites() {
        let s = "%m = arith.addf %a, %b : f16 loc(callsite(#loc2 at #loc24))";
        assert_eq!(strip_locs(s).trim(), "%m = arith.addf %a, %b : f16");
    }

    #[test]
    fn an_element_type_containing_x_survives() {
        assert_eq!(parse_shaped("64x512xindex>"), Some((vec![64, 512], "index".into())));
        assert_eq!(parse_shaped("f16>"), Some((vec![], "f16".into())));
        assert_eq!(parse_shaped("?x64xf16>"), Some((vec![DYNAMIC, 64], "f16".into())));
    }

    #[test]
    fn multi_result_scf_for_declares_every_result() {
        let src = "\
module {
  tt.func public @k(%a: !tt.ptr<f16>) attributes {noinline = false} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %r:2 = scf.for %i = %c0 to %c1 step %c1 iter_args(%x = %c0, %y = %c1) -> (i32, i32)  : i32 {
      scf.yield %x, %y : i32, i32
    }
    tt.return
  }
}
";
        let m = parse(src).unwrap();
        let f = m.kernel().unwrap();
        let forr = f.regions[0].ops.iter().find(|o| o.kind == OpKind::ScfFor).unwrap();
        assert_eq!(forr.results.len(), 2, "five iter_args means five results");
        assert_eq!(forr.result_types.len(), 2);
        // lb, ub, step, then the two inits.
        assert_eq!(forr.operands.len(), 5);
        // IV plus two region args.
        assert_eq!(forr.regions[0].args.len(), 3);
    }

    #[test]
    fn a_splat_dense_constant_reads_as_a_splat_and_a_list_does_not() {
        let m = parse(
            "module {\n  tt.func @k() {\n    %a = arith.constant dense<1.000000e+00> : tensor<64xf16>\n    %b = arith.constant dense<[1, 2]> : tensor<2xi32>\n    tt.return\n  }\n}\n",
        )
        .unwrap();
        let ops = &m.kernel().unwrap().regions[0].ops;
        assert!(matches!(ops[0].attr(&AttrKey::Value), Some(Attr::SplatFloat(_))));
        assert!(matches!(ops[1].attr(&AttrKey::Value), Some(Attr::Verbatim(_))));
    }

    #[test]
    fn the_reduce_region_and_its_axis_are_read() {
        let src = "\
module {
  tt.func @k(%in: !tt.ptr<f16>) {
    %x = arith.constant dense<0.000000e+00> : tensor<64x64xf16>
    %m = \"tt.reduce\"(%x) <{axis = 1 : i32}> ({
    ^bb0(%a: f16, %b: f16):
      %c = arith.maxnumf %a, %b : f16
      tt.reduce.return %c : f16
    }) : (tensor<64x64xf16>) -> tensor<64xf16>
    tt.return
  }
}
";
        let m = parse(src).unwrap();
        let red = m
            .ops_deep()
            .into_iter()
            .find(|o| o.kind == OpKind::TtReduce)
            .expect("the reduce is read");
        assert_eq!(red.attr(&AttrKey::Axis), Some(&Attr::Int(1)));
        assert_eq!(red.regions[0].args.len(), 2, "two combiner block args");
        assert_eq!(red.regions[0].ops.len(), 2, "maxnumf + reduce.return");
        assert_eq!(
            red.result_types,
            vec![IrType::Tensor { dims: vec![64], elem: DType::F16 }]
        );
    }

    #[test]
    fn an_unmodelled_op_keeps_its_spelling_so_a_refusal_can_name_it() {
        let m = parse("module {\n  tt.func @k() {\n    %x = tt.histogram %y : i32\n    tt.return\n  }\n}\n")
            .unwrap();
        let names: Vec<String> =
            m.ops_deep().iter().map(|o| o.kind.spelling().to_string()).collect();
        assert!(names.contains(&"tt.histogram".to_string()), "got {names:?}");
    }

    #[test]
    fn unbalanced_input_is_an_error_naming_the_construct() {
        let e = parse("module {\n  tt.func @k() {\n    tt.return\n").unwrap_err();
        assert!(e.message.contains("unclosed region"), "got {e}");
    }
}
