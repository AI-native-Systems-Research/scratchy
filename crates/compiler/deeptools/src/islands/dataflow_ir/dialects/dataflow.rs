//! `Dataflow.td` — UNITS, VIEWS, THE TRANSFERS BETWEEN UNITS, AND THE OPAQUE BODIES.
//!
//! The dialect declares fourteen operations; the ten here are the ones an emitted program contains.

use std::fmt::Write as _;

use crate::generated::{OpaqueFunc, ParamKey, ParamValue, RegName, SyncSignal};
use crate::islands::dataflow_ir::dialects::Val;
use crate::islands::dataflow_ir::link::{RecvEnd, SendEnd};
use crate::islands::dataflow_ir::print;
use crate::islands::dataflow_ir::ty::{AffineMap, MemRef, Vector};

/// WHERE ONE OF AN OPAQUE'S REGISTERS LIVES — the value its name binds to.
///
/// ⭐⭐ AN ADDRESS, NOT A NAME AND NOT A BLANK. `insertReg(regName, startAddress + n, ...)`
/// (`ddc/ddcv1.cpp:3369-3391`) binds each register name to its allocation's start address, and
/// `ConstructProgIRHelper.cpp:3999-4014` substitutes that value straight into an instruction's
/// operand field.
///
/// ⛔ AN EMPTY STRING PASSED THE CHECK AND SUBSTITUTED NOTHING. `DataflowToSentient.cpp:1986-1997`
/// only asserts the value IS a `StringAttr`, so `String::new()` — which is what this field held —
/// satisfied it and then wrote an empty operand into the instruction. A newtype over the address
/// makes that unrepresentable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct RegAddr(pub u32);

/// WHICH REGISTER FILE a `get_local_unit` names.
///
/// ⭐ THESE ARE THE UNIT-PREFIXED SPELLINGS, and the prefix matters: `arch_enums.h:62-66` keeps a
/// generic `LRFREG` "for DSC-level and arch-level compatibility" but says new IR operations use
/// `PE_LRFREG`, `SFP_LRFREG`, `PT_LRFREG`. The vendored DataflowIR writes `pt_lrfreg` and `ptxrf`
/// (`dcc/test/PT/xrfbmm_int8_fwd.mlir:56,60`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LocalUnit {
    /// `pe_lrfreg`.
    PeLrf,
    /// `sfp_lrfreg`.
    SfpLrf,
    /// `pt_lrfreg`.
    PtLrf,
    /// `ptxrf` — the PT's transposed register file, where a matmul's kernel block lands.
    PtXrf,
    /// `ptarf` — the PT's accumulator register file.
    PtArf,
    /// `l0scale` — the L0's scale region.
    ///
    /// ⛔ A LOCAL UNIT OF A PT ROW, AND ONLY FROM SEN1P5. Each PT row's arm ends with
    /// `if (coreArch >= SEN1P5_ISA) component_to_handler_[L0_SCALE] = createGetLocalUnitOp(..)`
    /// (`DSC2ToDataflowIRUtils.hpp:180-183`), which matches RCUDD1A having no scale region at all
    /// (`Arch::L0_SCALE_CAPACITY` is zero there).
    L0Scale,
    // ⛔ `sfpstate` AND `pestate` ARE NOT HERE, AND THAT IS NOT AN OMISSION. Both are bound with
    // `createGetUnitOp`, not `createGetLocalUnitOp` (`DSC2ToDataflowIRUtils.hpp:369-370, 350-351`) —
    // they are units in their own right that a transfer addresses, not register files a unit owns.
    // An earlier version listed `SfpState` here, which would have emitted a `get_local_unit` where
    // the backend expects a `get_unit`.
}

impl LocalUnit {
    /// The `name=` the op carries.
    #[must_use]
    pub const fn spelling(self) -> &'static str {
        match self {
            Self::PeLrf => "pe_lrfreg",
            Self::SfpLrf => "sfp_lrfreg",
            Self::PtLrf => "pt_lrfreg",
            Self::PtXrf => "ptxrf",
            Self::PtArf => "ptarf",
            Self::L0Scale => "l0scale",
        }
    }
}

/// A VALUE THAT ARRIVED ON THE WIRE — what a `dataflow.receive` binds.
///
/// # 🛑 A VALUE DEFINED INSIDE A PROGRAM UNIT DOES NOT LEAVE IT
///
/// ⛔⛔ MLIR's parser pushes a fresh DEFINITIONS scope for EVERY region, isolated or not:
/// `parseRegionBody` calls `pushSSANameScope(isIsolatedNameScope)` and that always runs
/// `isolatedNameScopes.back().pushSSANameScope()` (`Parser.cpp:2273-2276`, `:944-951`).
/// `IsolatedFromAbove` decides only whether the region can still SEE outward — it never makes the
/// region's own definitions escape. So a view a `dataflow.program_unit` takes is gone at its `}`,
/// and IBM's `lxlu` unit takes the LX view AGAIN rather than reuse the `l3lu`'s
/// (`/tmp/ktir_ref/export/debug/dfir.mlir:73-74` vs `:112-116`).
///
/// ⛔ THE EMITTER HANDED THE MOVER'S VIEWS STRAIGHT TO THE COMPUTE, and dbo-opt refused it twice
/// wearing two faces. First as *"use of undeclared SSA value name"* on `agen.vector_load %9`. Then,
/// once the leaked forward reference survived to the next program, as *"definition of SSA value
/// '%24#0' has type 'memref<2048x2048xf16>'"* — `Parser.cpp:1005-1011`, which fires only when the
/// name was already a forward-reference placeholder, here typed `memref<255x512xf16>` from the
/// PREVIOUS program's compute. One defect, two messages.
///
/// ⭐⭐ SO THE COMPUTE'S OPERANDS ARE WIRES, NOT VIEWS. Only [`Op::Receive`] mints one of these, so
/// a `Val` naming a view cannot reach a compute: the mover loads and sends, the compute receives.
///
/// ⛔ THE `Val` IS PRIVATE. A public field would be a way to launder the brand off at any call site
/// that happened to want a `Val`.
///
/// ⛔⛔⛔ AND IT IS ONE END OF A REUSE EDGE, SO IT IS SPENT ONCE — NOT `Copy`. The receive is the
/// PRODUCER and the compute that reads it is the CONSUMER, and that pairing is a fact the backend
/// checks: `OperandReuse::setReuseInformation` records `data_origins_` entries per USER
/// (`OperandReuse.cpp:17-38`), so a `dataflow.receive` earns an entry ONLY by being an operand of a
/// lowered compute, and `VectorChainToSentientPESFP.cpp:1343` then refuses any op that has none.
///
/// ⛔⛔ IT WAS `Copy`, AND THE TWO ENDS WERE TWO INDEPENDENT COUNTS. The receives were emitted one
/// per node operand; the compute was built over `split_first()` and `rest.first()`. On granite that
/// was SIX received and TWO consumed, and dbo-opt refused the four with "Dangling non-compute op
/// has no use | no OperandReuse entry: never an operand of a lowered compute". Spending the edge —
/// [`Received::operand`] takes `self` — makes a received vector nothing consumed a value still held
/// at the call site, which `#[must_use]` names at the line that made it.
#[must_use = "a received vector that no compute consumes is a `dataflow.receive` with no \
              OperandReuse entry — dbo-opt refuses it as a dangling non-compute op"]
#[derive(Debug, PartialEq, Eq)]
pub struct Received {
    val: Val,
    ty: Vector,
}

impl Received {
    /// EMIT THE `dataflow.receive`, AND BIND WHAT IT CARRIES.
    ///
    /// ⛔ THE ONLY CONSTRUCTOR, and it PUSHES the op rather than taking one to inspect. A
    /// `Received` therefore cannot exist without the receive that produces it, and the two cannot
    /// disagree about the value or its type. Nothing here can fail, so there is no arm to refuse in.
    pub fn receive(into: &mut Vec<super::Op>, result: Val, from: RecvEnd, ty: Vector) -> Received {
        into.push(super::Op::Dataflow(Op::Receive { result, from, ty }));
        Received { val: result, ty }
    }

    /// THE COMPUTE'S OPERAND, ONCE — CONSUMES THE EDGE.
    ///
    /// ⛔⛔ AND `#[must_use]` ALONE DOES NOT ENFORCE THAT. Rust has no linear types: a move-only
    /// value dropped out of a `Vec` triggers neither `must_use` nor any error, so a unit could mint
    /// six edges, spend two and drop four — measured, and dbo-opt refused it exactly as before.
    /// LINEARITY COMES FROM THE ARRAY: a `[Received; N]` destructured as `let [a, b] = edges` binds
    /// EVERY element or does not compile, so "minted and not consumed" is a pattern that cannot be
    /// written. That is why the edges travel as an array and never as a `Vec`.
    #[must_use]
    pub fn operand(self) -> Val {
        self.val
    }

    /// Its type — the wire's width, which the compute's operands are all at.
    #[must_use]
    pub const fn ty(&self) -> Vector {
        self.ty
    }

}

/// THE NUMERIC PRECISION OF A UNIT'S PROGRAM — `dataflow.program_unit`'s `precision` attribute.
///
/// ⛔ IT SELECTS THE MAC OPCODE. "This precision attribute is used to identify the MAC op code used
/// in the units" (`DSC2ToDataflowIR.hpp:137-143`), and it is produced by
/// `stringifyComputePrecision` (`:54-71`) — so it is the COMPUTE's type, not the tensor's.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    /// `int8` — `IMA8`.
    Int8,
    /// `int4` — `IMA4`.
    Int4,
    /// `fp4` — `FMA4`.
    Fp4,
    /// `fp8` — `FMA8`.
    Fp8,
    /// `fp16` — `FMA16`.
    Fp16,
    /// `fp32` — `FMA32`, and also `FNMS`, which `stringifyComputePrecision` maps here too (`:67-68`).
    Fp32,
}

impl Precision {
    /// The string the attribute carries.
    #[must_use]
    pub const fn spelling(self) -> &'static str {
        match self {
            Self::Int8 => "int8",
            Self::Int4 => "int4",
            Self::Fp4 => "fp4",
            Self::Fp8 => "fp8",
            Self::Fp16 => "fp16",
            Self::Fp32 => "fp32",
        }
    }
}

/// ONE `dataflow` OPERATION.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Op {
    /// `dataflow.get_unit {core, corelet, name, type} : index`.
    ///
    /// ⛔⛔ `type` IS LOAD-BEARING AND `name` IS NOT — THEY ARE NOT THE SAME STRING. Only
    /// `StrAttr:$name` and `StrAttr:$type` are declared arguments (`Dataflow.td`, `get_unit`);
    /// `core` and `corelet` ride through as discardable attributes on `attr-dict`. Downstream
    /// identity is taken from `getType()`, the `type` attribute
    /// (`DataflowToSentient.cpp:119-130`), and that string is then fed to
    /// `symbolizeSentientLoadConsumer(..).value()` — which **aborts** on a spelling outside the
    /// sixteen-member `SentientLoadConsumer` set (`SentientTypes.td:556-593`). So `type` is a
    /// censused token, never free text.
    ///
    /// ⭐ THE `name` FOLLOWS THE SCHEDULER'S CONVENTION, `C{core}-{tag}[-CL{corelet}]`, which is
    /// what IBM's own DataflowIR carries (`/tmp/ktir_ref/export/debug/dfir.mlir:45-63`) and what
    /// `UnitMaterializer.cpp:53-159` writes. The ddc translator instead passes
    /// `senComponentsToString.at(comp)` as BOTH name and type
    /// (`DSC2ToDataflowIRUtils.hpp:69-73`, and `dcc/test/Conversion/DataflowToSentient/opaque.mlir`
    /// shows `{name = "pe", type = "pe"}`) — two producers, two conventions. No consumer reading
    /// `name` was found, so this matches the producer whose output the entry point we entered by
    /// was built to accept.
    ///
    /// ⛔ AND THE UNIT IS A [`DfirUnit`], NOT A TEMPLATE [`Unit`]. DataflowIR binds units the `.ddl`
    /// vocabulary has no `unit=` spelling for — `lx`, `l3lu`, `l3su` — and names each PT ROW rather
    /// than a span.
    GetUnit {
        /// The handle it binds.
        result: Val,
        /// WHERE THE UNIT LIVES, which decides both attributes and the name.
        ///
        /// ⛔ THIS WAS `core: u32` PLUS `corelet: Option<u32>` AND THAT PAIR COULD NOT SPELL A
        /// GLOBAL UNIT — `core` was mandatory, so the HBM, which carries neither attribute, had no
        /// representation. See [`crate::units::Residency`].
        residency: crate::units::Residency,
        /// `type=`.
        unit: crate::units::DfirUnit,
    },

    /// `dataflow.get_local_unit %unit {name} : index` — a register file of a unit already held.
    GetLocalUnit {
        /// The handle it binds.
        result: Val,
        /// The unit it belongs to.
        of: Val,
        /// Which file.
        which: LocalUnit,
    },

    /// `dataflow.get_logical_memory_view %unit, %start {layout_map} : index, index, memref<..>`.
    ///
    /// ⛔ `start` IS IN ELEMENTS (`Dataflow.td:250`).
    GetLogicalMemoryView {
        /// The view it binds.
        result: Val,
        /// The memory unit viewed.
        from: Val,
        /// The start address, in elements.
        start: Val,
        /// How the view's indices map onto the linear region.
        layout: AffineMap,
        /// The view's type.
        ty: MemRef,
    },

    /// `dataflow.program_unit %unit {precision} : { .. }` — one unit's whole program.
    ProgramUnit {
        /// The units this program runs on. More than one where the same program is bound across
        /// program time steps (`Dataflow.td:99-104`).
        units: Vec<Val>,
        /// `precision=`, which selects the MAC opcode. Absent on a unit that computes nothing.
        precision: Option<Precision>,
        /// The body.
        body: Vec<super::Op>,
    },

    /// `dataflow.send %to, %data : vector<..>`.
    ///
    /// ⛔ `to` IS THE FIRST HOP, NOT THE DESTINATION, where the transfer states a via:
    /// `to_unit = dst.via_.empty() ? dst.loc_.unit_ : dst.via_.front()`
    /// (`SNTransferLowering.cpp:2731-2732`).
    Send {
        /// The unit sent to — ONE END OF A [`crate::islands::dataflow_ir::link::Link`], not a unit handle a caller chose.
        to: SendEnd,
        /// The data.
        data: Val,
        /// Its type.
        ty: Vector,
    },

    /// `dataflow.receive %from : vector<..>`.
    Receive {
        /// The vector it binds.
        result: Val,
        /// The unit received from — THE OTHER END OF THE SAME [`crate::islands::dataflow_ir::link::Link`] the send spent.
        from: RecvEnd,
        /// Its type.
        ty: Vector,
    },

    /// `dataflow.sync_send %unit : index`.
    SyncSend {
        /// The unit signalled.
        to: Val,
        /// Which signal, for the debug name.
        signal: SyncSignal,
    },

    /// `dataflow.sync_recv %unit : index` — blocking.
    SyncRecv {
        /// The unit waited on.
        from: Val,
        /// Which signal, for the debug name.
        signal: SyncSignal,
    },

    /// `dataflow.implicit_sync_on_streaming_buffer %view, %dst, %size : ..` — synchronisation at the
    /// granularity of a streaming buffer rather than per transfer.
    ImplicitSync {
        /// The buffer.
        view: Val,
        /// The unit synchronised with.
        dst: Val,
        /// The buffer size.
        size: Val,
        /// The buffer's type.
        view_ty: MemRef,
    },

    /// `dataflow.opaque {func_name, read_write_register_dictionary, read_only_register_dictionary,
    /// parameter_dictionary}` — one op standing for a whole `.smc` body, which dcc splices.
    ///
    /// ⭐ THE FIELDS ARE THEIR OWN STRUCT because the rung above forwards ALL of them at once:
    /// `lowerOpaqueOperation` (`DataflowToSentient.cpp:1984-2012`) reads the three dictionaries, the
    /// func name and the `dbgName` and hands every one of them to `sentient::OpaqueOp::create`. A
    /// five-field inline variant makes that one call five arguments; see [`Opaque`].
    Opaque(Opaque),
}

/// WHAT ONE `dataflow.opaque` CARRIES — the whole `.smc` body's binding, in one value.
///
/// ⛔⛔ THE TWO DICTIONARIES ARE CROSSED RELATIVE TO THEIR NAMES. `read_write_reg_map_` is filled
/// from the body's INTERNAL registers and `read_only_reg_map_` from the caller-bound INPUT/OUTPUT
/// ones (`ddcv1.cpp:3369-3391`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Opaque {
    /// `func_name=`.
    pub func: OpaqueFunc,
    /// `read_write_register_dictionary=` — the body's own scratch, from `internal_registers=`.
    pub read_write: Vec<(RegName, RegAddr)>,
    /// `read_only_register_dictionary=` — caller-bound, from `input_output_registers=`.
    pub read_only: Vec<(RegName, RegAddr)>,
    /// `parameter_dictionary=`, including the `prec` the op's own data format sets.
    pub params: Vec<(ParamKey, ParamValue)>,
    /// `dbgName=` — ⛔ CARRIED BECAUSE THE RUNG ABOVE FORWARDS IT.
    ///
    /// `OptionalAttr<StrAttr>:$dbgName` (`Dataflow.td:342`), and `lowerOpaqueOperation` passes
    /// `getDbgNameAttr(opaque_op)` straight into the `sentient.opaque` it builds
    /// (`DataflowToSentient.cpp:2005-2010`). IBM's own input writes it —
    /// `dataflow.opaque {dbgName="opaque_op #1", func_name= "reciprocal", ...}`
    /// (`dcc/test/Conversion/DataflowToSentient/opaque.mlir:32`) — so without this field the port of
    /// that pass has nothing to forward and the lowered op loses the only name a debugger has for it.
    pub dbg_name: Option<String>,
}

/// ONE `dataflow` OP AS TEXT. The caller has already indented the opening line.
pub(crate) fn emit(out: &mut String, op: &Op, depth: usize) {
    match op {
        Op::GetUnit {
            result,
            residency,
            unit,
        } => {
            use crate::units::Residency;

            // Attributes are written in MLIR's own key order, which is alphabetical: core, corelet,
            // name, type. Which of `core`/`corelet` appear is the residency's, and a missing
            // `corelet` is a fact rather than a hole — `ExtendUnitNameToCorelet` errors on one that
            // is absent where it needs it, and reads `0` as the name suffix `"0"`
            // (`DataflowToSentient.cpp:104-117`).
            let spelling = unit.spelling();
            let (attrs, name) = match residency {
                Residency::Global => (String::new(), spelling.to_owned()),
                Residency::Scratchpad { core } => (
                    format!("core = {} : i32, ", core.get()),
                    format!("C{}-{spelling}", core.get()),
                ),
                Residency::CoreWide { core } => (
                    format!("core = {} : i32, corelet = 0 : i32, ", core.get()),
                    format!("C{}-{spelling}", core.get()),
                ),
                Residency::Corelet { core, corelet } => (
                    format!(
                        "core = {} : i32, corelet = {} : i32, ",
                        core.get(),
                        corelet.get()
                    ),
                    format!("C{}-{spelling}-CL{}", core.get(), corelet.get()),
                ),
            };
            let _ = writeln!(
                out,
                "{} = dataflow.get_unit {{{attrs}name = \"{name}\", type = \"{spelling}\"}} : index",
                print::val(*result),
            );
        }
        Op::GetLocalUnit { result, of, which } => {
            let _ = writeln!(
                out,
                "{} = dataflow.get_local_unit {} {{name = \"{}\"}} : index",
                print::val(*result),
                print::val(*of),
                which.spelling()
            );
        }
        Op::GetLogicalMemoryView {
            result,
            from,
            start,
            layout,
            ty,
        } => {
            let _ = writeln!(
                out,
                "{} = dataflow.get_logical_memory_view {}, {} {{layout_map = {}}} : index, index, {}",
                print::val(*result),
                print::val(*from),
                print::val(*start),
                print::affine_map(layout),
                print::memref(ty)
            );
        }
        Op::ProgramUnit {
            units,
            precision,
            body,
        } => {
            let precision = match precision {
                Some(p) => format!(" {{precision = \"{}\"}}", Precision::spelling(*p)),
                None => String::new(),
            };
            let _ = writeln!(
                out,
                "dataflow.program_unit {}{precision} : {{",
                print::vals(units)
            );
            for inner in body {
                print::emit(out, inner, depth + 1);
            }
            print::indent(out, depth);
            out.push_str("}\n");
        }
        Op::Send { to, data, ty } => {
            let _ = writeln!(
                out,
                "dataflow.send {}, {} : {}",
                print::val(to.val()),
                print::val(*data),
                print::vector(*ty)
            );
        }
        Op::Receive { result, from, ty } => {
            let _ = writeln!(
                out,
                "{} = dataflow.receive {} : {}",
                print::val(*result),
                print::val(from.val()),
                print::vector(*ty)
            );
        }
        Op::SyncSend { to, signal } => {
            let _ = writeln!(
                out,
                // ⛔⛔ THE WAIT MODE IS OPTIONAL IN THE DIALECT AND MANDATORY TO LOWER.
                // `Dataflow.td:200` declares it `OptionalAttr<BoolAttr>`, but
                // `DataflowToSentient.cpp:219-222` fails outright when a send has none: "Unknown
                // async transfers modes for sentient".
                //
                // ⭐ `true` IS THE ONLY SAFE ANSWER HERE, and it is what the scheduler writes for
                // every send in its own output. It "controls whether the sender waits for
                // outstanding asynchronous transfers before signalling" (`:191-192`) — and a
                // `ddl.sync` exists to say the data has ARRIVED, so a signal that does not wait can
                // be observed before the transfer it is announcing lands. `false` is an
                // optimisation that needs something else to order the transfers, and no vendored
                // template states one: `ddl.sync` carries `units`, `signal_name` and `receive`, and
                // nothing about async waits.
                "dataflow.sync_send {} {{dbgName = \"{}\", \
                 wait_immediately_for_async_transfers = true}} : index",
                print::val(*to),
                signal.spelling()
            );
        }
        Op::SyncRecv { from, signal } => {
            let _ = writeln!(
                out,
                "dataflow.sync_recv {} {{dbgName = \"{}\"}} : index",
                print::val(*from),
                signal.spelling()
            );
        }
        Op::ImplicitSync {
            view,
            dst,
            size,
            view_ty,
        } => {
            let _ = writeln!(
                out,
                "dataflow.implicit_sync_on_streaming_buffer {}, {}, {} : {}, index, index",
                print::val(*view),
                print::val(*dst),
                print::val(*size),
                print::memref(view_ty)
            );
        }
        Op::Opaque(opaque) => {
            let Opaque {
                func,
                read_write,
                read_only,
                params,
                dbg_name,
            } = opaque;
            // ⛔ `dbgName` SORTS FIRST — `d` before `f`. A `DictionaryAttr` is key-ordered, so the
            // optional attribute is not a tail; it opens the dictionary when it is present.
            let named = dbg_name
                .as_ref()
                .map_or_else(String::new, |name| format!("dbgName = \"{name}\", "));
            let _ = writeln!(
                out,
                "dataflow.opaque {{{}func_name = \"{}\", parameter_dictionary = {}, \
                 read_only_register_dictionary = {}, read_write_register_dictionary = {}}}",
                named,
                func.spelling().to_lowercase(),
                dictionary(params, ParamKey::spelling, |v| ParamValue::spelling(v)
                    .to_owned()),
                // ⛔⛔ THE `R` IS LOAD-BEARING, NOT DECORATION. `insertReg` writes
                // `"R" + std::to_string(startAddress)` (`ddc/ddcv1.cpp:3350`) and the consumer
                // takes it back off by POSITION: `port_str = "lrf" + port_str.erase(0, 1)`
                // (`dcc/src/Dialect/Sentient/Utils.cpp:157`). A bare `"20"` becomes `lrf0` — the
                // wrong register, silently — and a bare `"8"` becomes `"lrf"`, which is no port at
                // all and aborts with "Invalid port".
                dictionary(read_only, RegName::spelling, |a: RegAddr| format!(
                    "R{}",
                    a.0
                )),
                dictionary(read_write, RegName::spelling, |a: RegAddr| format!(
                    "R{}",
                    a.0
                ))
            );
        }
    }
}

/// A `{key = "value", ..}` attribute dictionary, KEY-SORTED.
///
/// ⛔ SORTED BECAUSE MLIR SORTS. A `DictionaryAttr` is stored key-ordered, so a round trip through
/// the parser reorders anything else — and a printer whose output does not survive a round trip
/// cannot be checked against the vendored files.
/// ONE MLIR DICTIONARY ATTRIBUTE, sorted by key so the emission is byte-reproducible.
///
/// ⭐ THE KEYS AND VALUES ARRIVE AS TOKENS and are spelled here, at the one place text is the point.
/// Everything upstream holds enums, so a key that does not exist is a variant that does not exist.
fn dictionary<K: Copy, V: Copy>(
    entries: &[(K, V)],
    key: impl Fn(K) -> &'static str,
    value: impl Fn(V) -> String,
) -> String {
    let mut sorted: Vec<(&'static str, String)> =
        entries.iter().map(|(k, v)| (key(*k), value(*v))).collect();
    sorted.sort_by(|a, b| a.0.cmp(b.0));
    format!(
        "{{{}}}",
        sorted
            .iter()
            .map(|(key, value)| format!("{key} = \"{value}\""))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

#[cfg(test)]
mod tests {
    use crate::islands::dataflow_ir::dialects::dataflow::Op;
    use crate::islands::dataflow_ir::dialects::{self, Val};
    use crate::islands::dataflow_ir::print::emit;
    use crate::units::{Core, Corelet, DfirUnit, Residency};

    /// A core this build's arch has.
    fn core(index: u32) -> Core {
        Core::checked(index).expect("cores 0 and 1 exist on every arch this crate builds for")
    }

    /// A corelet this build's arch has.
    fn corelet(index: u32) -> Corelet {
        Corelet::checked(index).expect("corelets 0 and 1 exist on every arch this crate builds for")
    }

    /// ⭐⭐ IBM'S OWN UNIT BLOCK, REPRODUCED — all four residency classes, carried as TEXT rather
    /// than as a shape.
    ///
    /// The nineteen lines are `/tmp/ktir_ref/export/debug/dfir.mlir:45-63`, the DataflowIR the
    /// stock KTIR pathway produced on a two-core two-corelet run. They are the whole point of
    /// [`Residency`]: `C0-l3lu` carries `corelet = 0`, `C0-lx` carries NO corelet, and `hbm`
    /// carries NEITHER attribute — three distinct attribute shapes, of which the previous
    /// `core: u32, corelet: Option<u32>` pair could spell two.
    ///
    /// ⛔ AN ASSERTION ON THE SHAPE WOULD NOT HAVE CAUGHT THE BUG THIS REPLACES. "The HBM line has
    /// no `core`" passes just as well on an emitter that also drops `core` from the LX. This
    /// compares IBM's bytes in IBM's order, so a wrong attribute set on any one of the four classes
    /// is a diff rather than a silent agreement.
    #[test]
    fn reproduces_ibms_unit_block() {
        let mut ops = Vec::new();
        let mut next = 0u32;
        let mut unit = |residency, unit| {
            let op = dialects::Op::Dataflow(Op::GetUnit {
                result: Val(next),
                residency,
                unit,
            });
            next += 1;
            ops.push(op);
        };

        // Non-parallel compute, declared in the core group: `core` AND `corelet = 0`.
        for which in [DfirUnit::L3lu, DfirUnit::L3su] {
            for c in 0..2 {
                unit(Residency::CoreWide { core: core(c) }, which);
            }
        }
        // Parallel compute, declared in the corelet group. Corelet-major, as IBM emits it.
        for which in [DfirUnit::Lxlu, DfirUnit::Sfp, DfirUnit::Lxsu] {
            for cl in 0..2 {
                for c in 0..2 {
                    unit(
                        Residency::Corelet {
                            core: core(c),
                            corelet: corelet(cl),
                        },
                        which,
                    );
                }
            }
        }
        // Then the memories: the global root first, then one scratchpad per core.
        unit(Residency::Global, DfirUnit::Hbm);
        for c in 0..2 {
            unit(Residency::Scratchpad { core: core(c) }, DfirUnit::Lx);
        }

        let mut got = String::new();
        for op in &ops {
            emit(&mut got, op, 0);
        }

        let want = r#"%0 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-l3lu", type = "l3lu"} : index
%1 = dataflow.get_unit {core = 1 : i32, corelet = 0 : i32, name = "C1-l3lu", type = "l3lu"} : index
%2 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-l3su", type = "l3su"} : index
%3 = dataflow.get_unit {core = 1 : i32, corelet = 0 : i32, name = "C1-l3su", type = "l3su"} : index
%4 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-lxlu-CL0", type = "lxlu"} : index
%5 = dataflow.get_unit {core = 1 : i32, corelet = 0 : i32, name = "C1-lxlu-CL0", type = "lxlu"} : index
%6 = dataflow.get_unit {core = 0 : i32, corelet = 1 : i32, name = "C0-lxlu-CL1", type = "lxlu"} : index
%7 = dataflow.get_unit {core = 1 : i32, corelet = 1 : i32, name = "C1-lxlu-CL1", type = "lxlu"} : index
%8 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-sfp-CL0", type = "sfp"} : index
%9 = dataflow.get_unit {core = 1 : i32, corelet = 0 : i32, name = "C1-sfp-CL0", type = "sfp"} : index
%10 = dataflow.get_unit {core = 0 : i32, corelet = 1 : i32, name = "C0-sfp-CL1", type = "sfp"} : index
%11 = dataflow.get_unit {core = 1 : i32, corelet = 1 : i32, name = "C1-sfp-CL1", type = "sfp"} : index
%12 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-lxsu-CL0", type = "lxsu"} : index
%13 = dataflow.get_unit {core = 1 : i32, corelet = 0 : i32, name = "C1-lxsu-CL0", type = "lxsu"} : index
%14 = dataflow.get_unit {core = 0 : i32, corelet = 1 : i32, name = "C0-lxsu-CL1", type = "lxsu"} : index
%15 = dataflow.get_unit {core = 1 : i32, corelet = 1 : i32, name = "C1-lxsu-CL1", type = "lxsu"} : index
%16 = dataflow.get_unit {name = "hbm", type = "hbm"} : index
%17 = dataflow.get_unit {core = 0 : i32, name = "C0-lx", type = "lx"} : index
%18 = dataflow.get_unit {core = 1 : i32, name = "C1-lx", type = "lx"} : index
"#;

        for (at, (want_line, got_line)) in want.lines().zip(got.lines()).enumerate() {
            assert_eq!(
                want_line, got_line,
                "unit line {at} diverges from IBM's own DataflowIR"
            );
        }
        assert_eq!(
            want.lines().count(),
            got.lines().count(),
            "emitted a different number of unit lines than IBM's block has"
        );
    }
}
