// SPDX-License-Identifier: Apache-2.0
//! THE MoE PORT'S COMPARE/SELECT VOCABULARY, PINNED AGAINST THE VENDOR'S OWN DDL.
//!
//! The gemma-4 MoE lowerings (RouteArgsort's rank vector, RouteTopK's one-hot selector,
//! ExpertCombine's chain) compute with elementwise COMPARES whose result is consumed as
//! arithmetic — 0/1 tiles summed into ranks and matched into indices. This crate had no compare
//! opFunc: the door refused those programs by name, and the vendor `topk.ddl` scan (sfpring ring,
//! k-dim worksplit) is not expressible by this emitter at all. The compare/select family of
//! `broadcast_ops.ddl` is the vocabulary the port rests on, and these tests pin the THREE wire
//! facts a lowering depends on, each read straight from the vendor template:
//!
//! * **The names** (`broadcast_ops.ddl:36-42`): `lesserthan`, `equal`, `notequal`, `where3` —
//!   the exact `opFuncName` strings the `operation_bind`s carry. dxp refuses an unrecognized
//!   opFunc at `designSpaceConfig.cpp:7713` (`DtException Unrecognized opFunc`), the same refusal
//!   the crate already recorded for `"multiply"` vs `"mul"`, so a spelling drift here is a pod
//!   crash, not a build error.
//! * **The unit is sfp** (`broadcast_ops.ddl:297-344`): every one of them is a
//!   `ddl.compute {unit="sfp"}` — `LESSERTHAN`/`EQUAL`/`NOTEQUAL` into the BOOL state reg, then
//!   `SELECT` — so `OpFunc::ex_unit` must say `sfp` and can never say `pt`.
//! * **The result is fp16 0/1, not BOOL** (same lines): the vendor's own SELECT writes
//!   `plus1`/`zero` CONSTANTS into the fp16 OUTPUT tensor, so a compare's wire-level result is
//!   arithmetic-grade fp16 — exactly what a `sum` reduce can consume without a convert. This is
//!   the fact that makes the rank-vector form expressible at all: no BOOL-tensor consumer exists
//!   in this crate and none is needed.
//!
//! ⛔ THE RESULT PRECISION IS THE ODD ONE OUT AND IS NOT PINNED HERE. Whether the on-card
//! `lesserthan` rounds its SELECT through `plus1 = 0x3E00` exactly (it is a SEN169 constant, and
//! the crate's own `sen169_bits` doc records that a plain IEEE f16 encoding silently mis-scales)
//! is a card-track question the tiny26 EMU-vs-card tensordump parity gate answers per lowering —
//! `1.0` vs `1.0009766…` differs the moment a rank multiplies an index. This file pins the
//! VOCABULARY; the numerics gate is the port's own.

use ktir_superdsc::superdsc_opspec::OpFunc;

/// The four `opFuncName`s are the vendor's own strings (`broadcast_ops.ddl:36-42`), and
/// `op_func_from_str` round-trips each — a lowering hands the assembler a string, so a spelling
/// drift between the enum's `name()` and the string door would emit an op dxp refuses on card.
#[test]
fn compare_names_round_trip_through_the_string_door() {
    for (func, name) in [
        (OpFunc::LesserThan, "lesserthan"),
        (OpFunc::Equal, "equal"),
        (OpFunc::NotEqual, "notequal"),
        (OpFunc::Where3, "where3"),
    ] {
        assert_eq!(func.name(), name);
        assert_eq!(ktir_superdsc::emit::op_func_from_str(name), func);
    }
}

/// Every compare/select op runs on the SFP unit (`broadcast_ops.ddl:297-344` — all four are
/// `ddl.compute {unit="sfp"}`), never the PT matrix unit: `ex_unit` is the SOLE producer of
/// [`ExUnit`] and this pins it for the new arms.
#[test]
fn compare_ops_run_on_the_sfp_unit() {
    for func in [
        OpFunc::LesserThan,
        OpFunc::Equal,
        OpFunc::NotEqual,
        OpFunc::Where3,
    ] {
        assert_eq!(func.ex_unit().as_str(), "sfp");
    }
}

/// None of the compare/select ops reads an EXTERNAL sfp const (`broadcast_ops.ddl:146-153`: the
/// `conditional_op` block stages only the vendor's INTERNAL `zero`/`plus1` define_constants for
/// the SELECT) — so `needs_sfp_const_table` must be false, exactly like `add`/`maximum`. A true
/// here would attach the crate's external `SfpConstTable` and shadow the DDL's own
/// `plus1`/`zero` — the measured silu·up under-refinement failure mode, on the selector every
/// routed expert depends on.
#[test]
fn compare_ops_need_no_external_sfp_const_table() {
    for func in [
        OpFunc::LesserThan,
        OpFunc::Equal,
        OpFunc::NotEqual,
        OpFunc::Where3,
    ] {
        assert!(!func.needs_sfp_const_table());
    }
}
