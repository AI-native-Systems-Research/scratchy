// SPDX-License-Identifier: Apache-2.0
//! `ProgIR -> SenProg` — bridge 4, ported from `sys-arch-spec/{dpc,progir,isa}`.
//!
//! ⭐ THE EMISSION'S CLOSURE IS FOUR FILES, NOT ONE. `dpc.cpp` holds only 3 of the 33 units;
//! `progir.h` holds 18 as IN-CLASS definitions, which a `.cpp`-only scan would miss entirely.

pub mod isa_fields;
pub mod progir_inline;
pub mod progir_print;
pub mod senprog_writer;

/// TYPED INPUTS FOR THIS BRIDGE'S TESTS.
///
/// ⛔ A RUNG HAS TO BE STATED HERE because no `Model` or `Workload` impl exists in this crate
/// outside its integration tests, and without one a `Program` cannot be named at all.
#[cfg(test)]
pub(crate) mod test_fixtures {
    use crate::arch::Dd2;
    use crate::islands::progir::ty::OperandValue;
    use crate::islands::progir::{Block, Instruction, OpCode, OperandField, Program, UnitProgram};
    use crate::model::Model;
    use crate::workload::Workload;
    use sys_arch_spec::regfile::Component;

    pub(crate) struct Granite;
    impl Model for Granite {
        const QUERY_HEADS: u32 = 32;
        const KV_HEADS: u32 = 8;
        const HEAD_DIM: u32 = 64;
        const HIDDEN: u32 = 2048;
        const LAYERS: u32 = 40;
        const FFN: u32 = 8192;
        const VOCAB: u32 = 49152;
    }

    pub(crate) struct Decode;
    impl Workload for Decode {
        const ROWS: u32 = 1;
        const ACTIVE_CAP: u32 = 64;
    }

    /// One rung's program.
    pub(crate) type Prog = Program<Dd2, Granite, Decode>;

    /// A single-unit program over `blocks`, filed against the PT.
    pub(crate) fn program(blocks: Vec<Block>) -> Prog {
        Program {
            per_unit: vec![(Component::Pt, UnitProgram { blocks })],
            reg_state: Vec::new(),
            variable_definitions: Vec::new(),
            bound: core::marker::PhantomData,
        }
    }

    /// One instruction carrying `fields` and nothing else.
    pub(crate) fn instr(fields: Vec<(OperandField, OperandValue)>) -> Instruction {
        Instruction {
            opcode: OpCode::IMA8,
            symbolic_opcode: None,
            fields,
            dead: false,
            tag: None,
            comment: None,
        }
    }
}
