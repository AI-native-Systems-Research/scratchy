//! Parser for `opaque_templates/*.smc` — the opaque ops' bodies, and ISLAND 2'S OTHER SOURCE.
//!
//! A `ddl.opaque` names an op (`op="RECIPROCAL"`) whose body is one of these files. They are already written at
//! the machine's level — `FMA`, `LOGICAL`, `IME`, `SHR`, `SETMASK` with `src0=` / `tgtrf=` / `imm=` fields —
//! but against VIRTUAL registers (`p0_0`, `t4_0`, `c1`, `in0`), which is exactly island 2. So this text does
//! not get lifted to `computetype`; `computetype` gets lowered to meet it.
//!
//! ⛔ `///instr_count:N` IS IGNORED, and it is worth saying why because it reads like a length. IBM uses it for
//! an IBUFF CAPACITY ESTIMATE (`InstructionEstimation.cpp:453-482`) and never compares it against the body it
//! heads, so it has drifted: `idx32toaddr` says 100 for a 44-instruction body and `exp` says 13 for 14. It is
//! neither the length to emit nor a fact to check against, so it is not parsed.
//!
//! # ⭐ A BODY IS A TEMPLATE, AND ITS HOLES LOOK LIKE VALUES
//!
//! `mode=prec` does not state a precision, it says "the op's precision goes here". Same for `unroll=unroll`,
//! `be=be`, `imm=l0`, `src0=in0`. So a slot's right-hand side is classified HERE, at the parse boundary, into
//! [`SlotValue::Stated`] or [`SlotValue::Hole`] — leaving it for later is how a hole reaches an island and some
//! reader treats it as a value. Each hole's evidence is on [`SlotValue`].
//!
//! Build-only, like the rest of `ddl/`: its `String`s cannot reach an island because this file is compiled into
//! `build.rs` and not into the library.

/// One machine instruction: a mnemonic and its `key=value` slots, in source order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Instruction {
    pub mnemonic: String,
    pub slots: Vec<Slot>,
    /// ⭐ THE STATE-RESETTING `SETMASK` AT A BODY'S TAIL, which the splicer DROPS when the loop count
    /// is a multiple of 8 — `SETMASK` wraps after every increment of 8, so the reset is redundant
    /// (`ConstructProgIRHelper.cpp:3949-3954`). It is marked by its COMMENT and nothing else, so the
    /// flag is read here where the comment still exists.
    pub resets_mask: bool,
}

/// One `key=value` on an instruction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Slot {
    pub key: String,
    pub value: SlotValue,
}

/// A slot's right-hand side: something the body STATES, or a HOLE for the invoking `ddl.opaque` to fill.
///
/// | spelling | evidence that it is a hole |
/// |---|---|
/// | `prec` | not a legal `mode` value — `ConstructProgIRHelper.cpp:2040-2042`'s `DT_CHECK` allows only `fp16` and `fp32` |
/// | `unroll` | its value IS its field's name, and `x1` / `x2` / `u0` are the concrete unroll factors |
/// | `be` | the same: value equals field name, while `0` and `1` are concrete |
/// | `l0` | `getOpaqueInstrFromFile` reads it as `param.getNamed("l0")` |
/// | `in0`, `in1`, `in2`, `out0` | `Ddl_OpaqueOp`'s own example is `params={"in0"="lxlu", "out0"="result"}` |
///
/// ⭐ THE LAST ROW RESOLVES SOMETHING THAT LOOKED CONTRADICTORY. Bodies write both `fwdencoding=out0` (38x) and
/// `fwdencoding=result` (13x). If both were values there would be two spellings for one destination; as a hole
/// and a filling they are consistent — `out0` is the hole, `result` is what `params` puts in it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SlotValue {
    /// A value the body states outright: `mask=255`, `mode=fp32`, `tgtrf=t3_0`.
    Stated(String),
    /// A hole the invoking `ddl.opaque` fills.
    Hole(Hole),
}

/// WHICH input a hole names. The three the `params` maps state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Input {
    In0,
    In1,
    In2,
}

/// WHICH unroll slice. The two the bodies write.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Slice {
    S0,
    S1,
}

/// What fills a hole.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Hole {
    /// `prec` — the op's compute precision.
    Precision,
    /// `unroll` — the op's unroll factor.
    Unroll,
    /// `l0` — a loop count from the op's `params`.
    LoopCount,
    /// `in0` / `in1` / `in2` — WHICH input, from the op's `params`.
    Input(Input),
    /// `in0_0` / `in0_1` / `in1_0` / `in1_1` — which input, at which unroll SLICE, from
    /// `params={"in0_unroll"=…}`.
    InputAtSlice { input: Input, slice: Slice },
    /// `out0` — the op's output, from its `params`.
    Output,
    /// `outreg_0` / `outreg_1` — the output register at one unroll slice, from `params={"outreg_unroll"=…}`.
    OutRegAtSlice { slice: Slice },
}

/// Classifies one right-hand side.
///
/// ⛔ AND IT CHECKS THE ONE CLASS IT CAN. A hole gives itself away by being spelled as its own field's name
/// (`unroll=unroll`), so a value equal to its key that is not a classified hole means the bodies grew a
/// placeholder nobody has named — and calling that a stated value would put it in an island. The others
/// (`prec`, `l0`, `in0`) cannot be caught that way, which is why each carries its evidence above instead.
///
/// ⛔⛔ `be=be` IS NOT ONE OF THEM: IT IS THE ISA's OWN LITERAL. The `be` field carries exactly one encode
/// entry, `Enc::Lit("be", 1)` (`isa_fields.rs:257-260`), so the value `be` is that field's spelling for 1 —
/// the same `field=value-name` shape as `soft=no` or `group=GTR0`, and what both of the reference's own
/// disassemblers print for a set block end against `be=0` for a clear one. Reading it as a hole and resolving
/// it to `Off` DROPPED the close of every body that opens a loop: all 15 such bodies state exactly one
/// `MVLOOPCNT` and exactly one active `be=be`, so the body closes its own loop and the template is the
/// authority on its own fixed instruction sequence. The unmatched counters that left are what
/// `bridges::i3`'s `blocks_balance_loop_counters` reports.
fn classify(key: &str, value: &str) -> SlotValue {
    let hole = match value {
        "prec" => Hole::Precision,
        "unroll" => Hole::Unroll,
        "l0" => Hole::LoopCount,
        "in0" => Hole::Input(Input::In0),
        "in1" => Hole::Input(Input::In1),
        "in2" => Hole::Input(Input::In2),
        "out0" => Hole::Output,
        // ⭐ THE PER-SLICE FORMS ARE THE SAME HOLES, and reading them as ordinary names was a live mistake. A
        // `ddl.opaque` states its filler ONCE — `params={"in0_unroll"="lxlu"}` — and the body writes out the
        // slices, so `src0=in0_0` names input 0 at slice 0 rather than a local called `in0_0`.
        //
        // ⛔ ENUMERATED, NOT PARSED. I wrote a `rsplit_once('_')` that returned an `Option` and fell through a
        // `_` arm — a suffix parser guessing at a spelling, in a crate whose world is 36 vendored files. These
        // NINE are every per-slice spelling those files contain; a tenth is a build error at the assert below,
        // which is what a closed world buys.
        "in0_0" => Hole::InputAtSlice {
            input: Input::In0,
            slice: Slice::S0,
        },
        "in0_1" => Hole::InputAtSlice {
            input: Input::In0,
            slice: Slice::S1,
        },
        "in1_0" => Hole::InputAtSlice {
            input: Input::In1,
            slice: Slice::S0,
        },
        "in1_1" => Hole::InputAtSlice {
            input: Input::In1,
            slice: Slice::S1,
        },
        "outreg_0" => Hole::OutRegAtSlice { slice: Slice::S0 },
        "outreg_1" => Hole::OutRegAtSlice { slice: Slice::S1 },
        stated => {
            // ⭐ `be=be` IS THE ONE `key == value` THAT IS A REAL VALUE, and the ISA says so: the `be` field's
            // whole encode list is `Enc::Lit("be", 1)` (`isa_fields.rs:257-260`). Every other field whose value
            // repeats its name is a placeholder, so the guard stays for all of them.
            assert!(
                key != stated || key == "be",
                "an .smc slot `{key}={stated}` is spelled as its own field's name and is neither a classified \
                 hole nor a field whose ISA encode list gives that spelling a value; the bodies grew a \
                 placeholder that would otherwise reach an island as a value"
            );
            return SlotValue::Stated(stated.to_string());
        }
    };
    SlotValue::Hole(hole)
}

/// One `.smc` body.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Body {
    /// `@RegInit:c0:"0x…"` — a constant register's initial value, as `(register, hex)`.
    pub reg_init: Vec<(String, String)>,
    pub instructions: Vec<Instruction>,
}

/// Parses one `.smc` file.
pub fn parse(src: &str) -> Result<Body, String> {
    let mut body = Body {
        reg_init: Vec::new(),
        instructions: Vec::new(),
    };
    for (lineno, raw) in src.lines().enumerate() {
        // A comment may start anywhere, and `@RegInit:` lives INSIDE one — so the comment is read for it
        // before being discarded.
        let (code, comment) = match raw.find("//") {
            Some(at) => (&raw[..at], &raw[at..]),
            None => (raw, ""),
        };
        if let Some(rest) = comment.split("@RegInit:").nth(1) {
            body.reg_init.extend(reg_init_of(rest));
        }
        // A `// c1:"0x…"` continuation line of a `@RegInit:` block, which is how the multi-register form is
        // written: the marker appears once and the remaining registers follow as plain comments.
        if comment.contains(":\"0x") && !comment.contains("@RegInit:") {
            body.reg_init.extend(reg_init_of(comment));
        }
        let code = code.trim();
        if code.is_empty() {
            continue;
        }
        let mut parts = code.split_whitespace();
        let mnemonic = parts
            .next()
            .ok_or_else(|| format!("line {}: no mnemonic", lineno + 1))?
            .to_string();
        let mut slots = Vec::new();
        for part in parts {
            let (key, value) = part
                .split_once('=')
                .ok_or_else(|| format!("line {}: {part:?} is not key=value", lineno + 1))?;
            slots.push(Slot {
                key: key.to_string(),
                value: classify(key, value),
            });
        }
        let resets_mask = mnemonic == "SETMASK" && comment.contains("reset mask");
        body.instructions.push(Instruction {
            mnemonic,
            slots,
            resets_mask,
        });
    }
    Ok(body)
}

/// Every `reg:"0xhex"` pair in one comment.
fn reg_init_of(text: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let mut rest = text;
    while let Some(quote) = rest.find(":\"") {
        let name = rest[..quote]
            .rsplit(|c: char| !(c.is_alphanumeric() || c == '_'))
            .next()
            .unwrap_or("");
        let after = &rest[quote + 2..];
        match after.find('"') {
            Some(end) if !name.is_empty() => {
                out.push((name.to_string(), after[..end].to_string()));
                rest = &after[end + 1..];
            }
            _ => return out,
        }
    }
    out
}
