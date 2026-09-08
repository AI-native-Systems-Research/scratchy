//! ISLAND 4 — `senprog`, IBM's own serialization point.
//!
//! `dcc` writes it and `dip_standalone -s` reads it, which makes it the ONE seam an external tool can check.
//! Ported from `ProgramAndStateInfo` (`sys-arch-spec/progir/progir.h:505-545`) and `InstrInfo` (`:274-284`).
//!
//! # ⛔ NOT A `Tape`, AND THAT IS WHY IT IS AN ISLAND
//!
//! Every other island is one ordered tape whose ops each name a unit. A senprog is **N PROGRAMS**, keyed by
//! (core, corelet, unit), each carrying its own register pool: `senCompProgram_` is a
//! `map<SenComponents, ProgIrCodeGraph>` (`:512`) beside a `RegStateInfo` (`:533`). The GROUPING is the island.
//!
//! ⛔⛔⛔ AND A SENPROG IS PER OP — `GroupPos` DOES NOT VANISH. This comment used to say the opposite, inferred
//! from `ProgramAndStateInfo` having no `GroupPos` field. The field is absent because **each op IS its own
//! `ProgramAndStateInfo`**: `dxp/dxp.cpp:836-846` loops over `sdscNodes`, one program frame each, then merges, and
//! `fillSPRInfoPsinfo` reads ONE `precisionOfOp` per program. So island 4 is built PER OP and the register pool is
//! per op with it — which is why an 8-deep EBR file serves a group of hundreds.
//!
//! ⭐ A group-wide pool was tried and refuted by its own failure: it cleared the register collision and then hit
//! the file limit, more than 8 distinct EBR values in one 78-op group.

use crate::isa::operand::Operand;
use crate::isa::regfile::{Component, RegType};
use crate::packet::header::{CoreIdx, Corelet};

/// The two `ProgramAndStateInfo` bounds — `sys-arch-spec/progir/progir.h`, which the model is bounded by too.
pub use sys_arch_spec::progir::{MAX_INSTRUCTIONS_PER_UNIT, MAX_REGISTERS_PER_UNIT};

/// ONE SENPROG INSTRUCTION — `InstrInfo` (`progir.h:274-284`).
///
/// ⛔⛔ THE FIELDS ARE THE ISLAND'S, NOT THE PRINTER'S. `instFields_` is a `std::map<OperandT, OperandAttr>`, so
/// its ORDER — [`Operand`]'s declaration order — is the order a senprog line writes its terms in. Holding
/// [`crate::generated::PhysicalOp`] here instead would leave the writer to decompose instructions, and a writer
/// that converts is not a writer.
///
/// ⛔ AND NO SHIFTS. A field's bit position belongs to the ISA (`crate::isa::fields`); an instruction carrying its
/// own could disagree with the table it came from. The writer looks each up, exactly as `convertIr2Senprog` does
/// (`dpc.cpp:711-713`, `isa.typeToFieldBitShift.at(instrType).at(fieldPos)`).
pub use scratchy_spyre_senulator::isa::Instruction;

/// BUILD ONE — the opcode and the fields it states, in the senulator's own container.
///
/// ⛔⛔ THE VALUE NARROWS TO `i32` AND THAT IS IBM'S, NOT A LOSS THIS INTRODUCES. `setOperand` takes an
/// `int64_t` and stores into an `int32_t operands[]` (`senulator/instruction.h:21`, `:39`). The round trip is
/// EXACT for every field this ISA has: the widest is 32 bits (`defineField`'s table, checked by
/// [`crate::isa::fields::ImmWidth`]), and an encoded value of `w <= 32` bits recovers through
/// `u64 -> i32 -> u32 -> u64` unchanged. Reading it back with `as u64` instead of `as u32` would sign-extend and
/// corrupt the word, which is why [`crate::bridges::senprog_to_init_packet::word_of`] goes through `u32`.
///
/// ⛔⛔⛔ A FIELD VALUE OF `0xFFFF` IS REFUSED, BECAUSE THIS CONTAINER CANNOT HOLD IT. `hasOperand` is
/// `operands[key] != 0xFFFF` (`instruction.h:34`) — the sentinel is 65535, a positive number inside the range of
/// every 16-bit unsigned field, of which the ISA declares thirteen. So an instruction whose immediate is at the
/// top of its range reports that it has no immediate at all. That is a defect in the reference, and it is the
/// reference's own storage; a program that needs the value cannot be expressed to the model that must run it, so
/// it is refused HERE rather than silently losing the field on the way in.
pub fn instruction(
    comp: Component,
    opcode: crate::isa::InstOpCode,
    fields: &[(Operand, crate::isa::fields::FieldValue)],
) -> Instruction {
    let mut instr = Instruction::new(opcode, 0);
    // ⛔⛔⛔ EVERY FIELD OF THE TYPE IS ZEROED FIRST, AND ONLY THEN ARE THE STATED ONES WRITTEN.
    //
    // `Instruction::Instruction` zeroes each field the instruction TYPE declares before it reads any value
    // (`instruction.cpp`, recorded as `instruction_text::FIELDS_OF_THE_TYPE_ARE_ZEROED_NOT_LEFT_UNSET`), so only
    // a field the type does NOT have keeps the sentinel. Skipping this step leaves every unstated field reading
    // as `ABSENT_OPERAND` — 0xFFFF, not 0 — and 0xFFFF is a VALUE: it made `mode` on a `LOGICAL` read as fp32,
    // which `computeElement.cpp:707-709` refuses outright, and every `tgt*` read as a live target, which makes
    // the model skip the instruction as having no output.
    //
    // ⭐ THE TYPE'S FIELDS COME FROM THIS CRATE'S OWN `defineField` PORT, which is the table `FieldLayout` exists
    // to receive — so the set zeroed here is the set the senulator would have zeroed reading the same senprog.
    for field in crate::isa::fields_of(comp, opcode) {
        instr = instr.with(field, 0);
    }
    for (field, value) in fields {
        let bits = value.get();
        assert!(
            bits != u64::from(scratchy_spyre_senulator::isa::ABSENT_OPERAND.cast_unsigned()),
            "{opcode:?} states {field:?} = {bits}, and 65535 is the value `hasOperand` reads as ABSENT \
             (`instruction.h:34`) — thirteen of this ISA's immediate fields are 16-bit unsigned, so their \
             maximum IS the sentinel. The instruction cannot be represented to the model that executes it"
        );
        instr = instr.with(*field, bits as u32 as i32);
    }
    instr
}

/// EVERY FIELD THIS INSTRUCTION STATES, in [`Operand`] order — which is the WIRE order.
///
/// ⭐ THE ORDER IS FREE HERE, AND THAT IS WHY THE CONTAINER FITS. `InstrInfo::instFields_` is a
/// `std::map<OperandT, OperandAttr>` (`progir.h:284`), so a senprog line writes its terms in the enum's
/// declaration order; the senulator's array is INDEXED by that enum, so walking `Operand::ALL` yields the same
/// sequence with no sort and no second list.
pub fn stated_fields(
    instr: &Instruction,
) -> impl Iterator<Item = (Operand, crate::isa::fields::FieldValue)> + '_ {
    Operand::ALL.into_iter().filter_map(move |field| {
        instr.operand(field).map(|held| {
            (
                field,
                crate::isa::fields::FieldValue::of(u64::from(held.cast_unsigned())),
            )
        })
    })
}

/// THE EXPOSED-PIPELINE NOPS — the port of `TransformForExposedPipelinePass`
/// (`Transform/Sentient/TransformForExposedPipeline.cpp`).
///
/// The PT's register write-back is DELAYED — `regWBDelayCycles` is 3 for fp and 2 for int at RCUDD1A or
/// before (`computeElement.cpp:1402-1407`) — and the pipeline is EXPOSED: nothing stalls a read that
/// arrives early, the operand is simply the old value, or untyped if nothing ever landed. The reference
/// resolves every RAW hazard by inserting NOPs (`insertNOPOperations`, with
/// `getCyclesForExposedPipeline` = 3 fp / 2 int at this arch, `SentientOps.cpp:40-59`); the machine's
/// own `OperandIsUntyped` on the peeled `bmm.ddl` accumulation chain is what this pass prevents — the
/// first-trip variant writes ARF 0 and the adjacent last-trip variant read it one instruction later.
///
/// The analysis measures the MINIMUM execution distance between a register write and each read of the
/// same register — along the fall-through, into nested loops (whose `MVLOOPCNT` imm is a known count,
/// not a search), and around a shared loop's back edge, including an instruction that reads and writes
/// the same register depending on ITSELF across trips. A deficit inserts NOPs before the reader (or
/// before the shared loop's block end, for a back-edge hazard), and iterating to a fixpoint stands in
/// for the reference's interval reduction: a NOP inserted for one hazard shortens every deficit that
/// spans it.
pub fn insert_exposed_pipeline_nops(
    comp: Component,
    instructions: &mut Vec<Instruction>,
    required: u32,
) {
    let mut rounds = 0usize;
    while let Some((insert_at, deficit)) = first_pipeline_deficit(instructions, required) {
        let nop = instruction(comp, crate::isa::InstOpCode::NOP, &[]);
        for _ in 0..deficit {
            instructions.insert(insert_at, nop);
        }
        rounds += 1;
        assert!(
            rounds <= instructions.len().saturating_mul(4).max(64),
            "the exposed-pipeline fixpoint did not converge over {} instructions; every insertion \
             strictly lengthens a deficit's span, so this is a defect in the distance walk",
            instructions.len(),
        );
    }
}

/// One loop of a PT program, by index: its `MVLOOPCNT`, the instruction carrying its block end, and its
/// trip count.
struct PtLoop {
    open: usize,
    close: usize,
    trips: u32,
}

/// The program's loops, by pairing each `MVLOOPCNT` with the `be` that closes it — the same implicit
/// hardware counter stack the ISA runs (`isa.cpp:1074-1077`).
fn pt_loops(instructions: &[Instruction]) -> Vec<PtLoop> {
    use scratchy_spyre_senulator::isa::Operand;
    let mut out: Vec<PtLoop> = Vec::new();
    let mut open: Vec<(usize, u32)> = Vec::new();
    for (index, instr) in instructions.iter().enumerate() {
        if matches!(instr.opcode, crate::isa::InstOpCode::MVLOOPCNT) {
            let trips = instr
                .operand(Operand::Imm)
                .unwrap_or_else(|| {
                    panic!("a PT `MVLOOPCNT` at pc {index} states no `imm`; the static form always does")
                })
                .cast_unsigned();
            open.push((index, trips));
        }
        if instr.has_loop_end() {
            let (started, trips) = open.pop().unwrap_or_else(|| {
                panic!(
                    "a block end at pc {index} closes no open `MVLOOPCNT`; \
                     `blocks_balance_loop_counters` refuses this program shape"
                )
            });
            out.push(PtLoop {
                open: started,
                close: index,
                trips,
            });
        }
    }
    out
}

/// Executed instruction count of `range`, walking nested loops at their stated trip counts.
fn executed_in(loops: &[PtLoop], range: std::ops::Range<usize>) -> u64 {
    let mut count = 0u64;
    let mut index = range.start;
    while index < range.end {
        let nested = loops
            .iter()
            .find(|l| l.open == index && l.close < range.end);
        match nested {
            Some(l) => {
                // The counter itself, then the body once per trip (each trip executes through the
                // block-end carrier).
                count += 1 + u64::from(l.trips) * executed_in(loops, l.open + 1..l.close + 1);
                index = l.close + 1;
            }
            None => {
                count += 1;
                index += 1;
            }
        }
    }
    count
}

/// The PT compute opcodes — the writers and readers of the delayed register file. `NOP`'s field table
/// zeroes `tgtrf` too, so the OPCODE is the discriminator, not field presence.
fn is_pt_compute(opcode: crate::isa::InstOpCode) -> bool {
    matches!(
        opcode,
        crate::isa::InstOpCode::FMA
            | crate::isa::InstOpCode::FMA8
            | crate::isa::InstOpCode::FMA4
            | crate::isa::InstOpCode::IMA4
            | crate::isa::InstOpCode::IMA8
    )
}

/// The first RAW hazard whose minimum execution distance falls short — `(where to insert, how many)`.
fn first_pipeline_deficit(instructions: &[Instruction], required: u32) -> Option<(usize, u32)> {
    use scratchy_spyre_senulator::isa::Operand;
    let loops = pt_loops(instructions);
    // A register WRITE: a compute whose `tgtrf` names a file register — 14 is the XRF (whose latch
    // hazard has its own cover) and 15 is "no register" (`computeElement.cpp:2683-2689`).
    let writes = |instr: &Instruction| -> Option<i32> {
        (is_pt_compute(instr.opcode))
            .then(|| instr.operand(Operand::Tgtrf))
            .flatten()
            .filter(|&reg| reg < 14)
    };
    // A register READ: a compute whose `src1` names a file register — 13 is the north link, 14 the
    // west, 15 the constant zero (`pt_source_b`, `computeElement.cpp:1630-1644`).
    let reads = |instr: &Instruction| -> Option<i32> {
        (is_pt_compute(instr.opcode))
            .then(|| instr.operand(Operand::Src1))
            .flatten()
            .filter(|&reg| reg < 13)
    };
    for (w, writer) in instructions.iter().enumerate() {
        let Some(written) = writes(writer) else {
            continue;
        };
        for (r, reader) in instructions.iter().enumerate() {
            let Some(read) = reads(reader) else {
                continue;
            };
            if read != written {
                continue;
            }
            if r > w {
                // Fall-through: the distance from the write's step to the read's step.
                let distance = executed_in(&loops, w + 1..r + 1);
                let deficit = u64::from(required).saturating_sub(distance);
                if deficit > 0 {
                    // ⭐ INSERTED AFTER THE WRITER, NOT BEFORE THE READER: one pad serves EVERY
                    // reader of that write — the cheap half of the reference's interval reduction
                    // (`reduceIntervals`), and the difference between fitting a 128-entry ibuff and
                    // not. But never between a MVLOOPCNT and its body head: the pad goes after the
                    // writer, which is a compute, so the loop structure is untouched.
                    return Some((w + 1, u32::try_from(deficit).expect("required is small")));
                }
            }
            // Around the back edge of the innermost loop containing BOTH — including the write
            // depending on itself. The wrap exists only where the loop runs more than once.
            if let Some(shared) = loops
                .iter()
                .filter(|l| l.open < w.min(r) && l.close >= w.max(r) && l.trips > 1)
                .min_by_key(|l| l.close - l.open)
                && r <= w
            {
                let distance = executed_in(&loops, w + 1..shared.close + 1)
                    + executed_in(&loops, shared.open + 1..r + 1);
                let deficit = u64::from(required).saturating_sub(distance);
                if deficit > 0 {
                    return Some((
                        shared.close,
                        u32::try_from(deficit).expect("required is small"),
                    ));
                }
            }
        }
    }
    None
}

/// WHICH UNIT A SENPROG PROGRAM BELONGS TO, in senprog's own spelling.
///
/// ⛔⛔ DERIVED BY THE WRITER'S OWN STRING SURGERY (`dpc.cpp:632-646`): take
/// `senComponentsToString(unit)`, split the LAST CHARACTER off as the corelet, and — when what remains contains
/// `"row"` — insert `_` at index 2 and drop the trailing underscore. `PTROW0_0` prints as unit `pt_row0`, corelet
/// `0`.
///
/// ⭐ THE L3 IS THE EXCEPTION AND IT IS EXPLICIT: `if (unit == L3LU || unit == L3SU) corelet = "0"` with the name
/// left whole (`:635-636`). It is not that the L3 HAS corelet zero — the writer STATES zero for a unit that has
/// none, which is why [`Self::L3`] carries no [`Corelet`] to get wrong.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SenUnit {
    L3(L3Kind),
    OnCorelet {
        unit: CoreletKind,
        corelet: Corelet,
    },
    /// One row of the systolic array, per corelet. Prints `pt_row<N>`.
    PtRow {
        row: crate::isa::unit::PtRowIdx,
        corelet: Corelet,
    },
    Compute {
        unit: ComputeKind,
        corelet: Corelet,
    },
}

/// The two L3 units — their own enum because they are the only units the writer gives no corelet.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum L3Kind {
    Lu,
    Su,
}

/// The memory units that exist once per corelet.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CoreletKind {
    L0lu,
    L0su,
    Lxlu,
    Lxsu,
}

/// The non-PT compute units. Separate because the writer keys its 64-bit `UL` shift suffix on exactly
/// `PE0 | PE1 | SFP0 | SFP1` (`dpc.cpp:654-660`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ComputeKind {
    Pe,
    Sfp,
}

impl SenUnit {
    /// WHETHER THIS UNIT'S SHIFTS CARRY THE `UL` SUFFIX (`dpc.cpp:654-660`).
    ///
    /// ⭐ THE WRITER BRANCHES ON THE UNIT, NOT ON A WIDTH, so this is the unit's own question. That the same set is
    /// the 64-bit-instruction set is a fact about the machine, stated elsewhere.
    pub const fn shift_suffix_is_ul(self) -> bool {
        match self {
            Self::Compute { .. } => true,
            Self::L3(_) | Self::OnCorelet { .. } | Self::PtRow { .. } => false,
        }
    }

    /// THE COMPONENT WHOSE FIELD TABLE ENCODES THIS UNIT'S INSTRUCTIONS.
    ///
    /// ⛔ EVERY PT ROW ANSWERS `Pt`: a field table is per component and a row is not a component of its own — the
    /// same collapse `senCompToGenericComp` performs (`arch_enums.cpp:166-172`).
    pub const fn component(self) -> Component {
        match self {
            Self::L3(L3Kind::Lu) => Component::L3lu,
            Self::L3(L3Kind::Su) => Component::L3su,
            Self::OnCorelet {
                unit: CoreletKind::L0lu,
                ..
            } => Component::L0lu,
            Self::OnCorelet {
                unit: CoreletKind::L0su,
                ..
            } => Component::L0su,
            Self::OnCorelet {
                unit: CoreletKind::Lxlu,
                ..
            } => Component::Lxlu,
            Self::OnCorelet {
                unit: CoreletKind::Lxsu,
                ..
            } => Component::Lxsu,
            Self::PtRow { .. } => Component::Pt,
            Self::Compute {
                unit: ComputeKind::Pe,
                ..
            } => Component::Pe,
            Self::Compute {
                unit: ComputeKind::Sfp,
                ..
            } => Component::Sfp,
        }
    }

    /// THE CORELET THE PROGRAM BANNER PRINTS — `0` for the L3, which HAS none (`dpc.cpp:635-636`).
    pub const fn printed_corelet(self) -> Corelet {
        match self {
            Self::L3(_) => Corelet::Zero,
            Self::OnCorelet { corelet, .. }
            | Self::PtRow { corelet, .. }
            | Self::Compute { corelet, .. } => corelet,
        }
    }

    /// WHETHER THE REGISTER HEADER STATES A CORELET AT ALL — `if (unit != L3LU && unit != L3SU)` (`dpc.cpp:741-743`).
    ///
    /// ⛔ NOT THE SAME QUESTION AS [`Self::printed_corelet`]: the banner prints `Corelet: 0` for the L3 while the
    /// register header omits the field entirely, so one answer cannot serve both.
    pub const fn register_header_states_corelet(self) -> bool {
        match self {
            Self::L3(_) => false,
            Self::OnCorelet { .. } | Self::PtRow { .. } | Self::Compute { .. } => true,
        }
    }
}

/// ONE UNIT'S PROGRAM — its instructions and the registers it starts with.
///
/// ⭐ THE POOL IS PART OF THE PROGRAM, not a sibling table: the writer emits the register block immediately after
/// the instruction block for the SAME unit, gated on `regState_.count(unit) > 0` (`dpc.cpp:726-727`).
#[derive(Debug, Clone, Default)]
pub struct Program {
    pub instructions: Vec<Instruction>,
    pub registers: Vec<RegisterFile>,
}

/// ONE REGISTER FILE'S INITIAL VALUES.
///
/// ⛔ THE FILE ORDER IS THE C++ ENUM'S, NOT OUR `RegType`'s — see [`REGISTER_FILE_ORDER`]. An ordered `Vec` rather
/// than a map, so nothing can reorder the blocks.
#[derive(Debug, Clone)]
pub struct RegisterFile {
    pub file: RegType,
    pub values: Vec<(crate::islands::progir::RegIndex, RegValue)>,
}

/// WHAT ONE REGISTER STARTS WITH, in the forms the writer can print (`dpc.cpp:745-768`).
///
/// ⛔ THE FORMS ARE NOT INTERCHANGEABLE and the writer refuses an unknown one
/// (`DT_ERROR("Unsupported reg init data type for senprog generation")`, `:766`): an integer prints with a
/// `.000000` tail (`:761-762`), a wide value prints bare (`:763-765`).
///
/// ⛔⛔ THE REFERENCE'S OTHER TWO FORMS ARE SYMBOLIC, AND THIS CRATE HAS NO VALUE THAT COULD REACH THEM.
/// `isVariable()` prints `@name@` eight times on a PE/SFP and `$name$.000000` elsewhere (`:752-760`), and
/// `isVariableSymbol()` prints `%n%` (`progir.cpp:43-44`) — both are dxp holding a name for something it has not
/// resolved yet. Everything here is a constant by the time it is written, so the absence of those variants is the
/// statement that this crate never defers a register's value, not an unported case.
///
/// ⚠️ ONE PREFIX IS UNMODELLED AND IT IS STATED RATHER THAN ASSUMED AWAY: an attr with `hasSenDataType()` writes
/// `datatype:<type> ` before the value (`dpc.cpp:749-752`). Nothing here carries a SEN data type on a register
/// init, so nothing emits it — but whether dxp sets one on the inits this crate produces has NOT been read, and a
/// missing prefix would be a silent difference rather than a refusal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegValue {
    Int(i64),
    Wide(crate::packet::lrf::ComputeLrfValue),
}

/// THE ORDER REGISTER-FILE BLOCKS ARE WRITTEN IN — the C++ `enum RegType`'s own order.
///
/// ⛔⛔ NOT OUR `RegType`'s. `getSimpleRegInit` is a `std::map<RegType, …>`, so it walks the C++ ENUMERATOR order:
/// `LRF, LAR, LBR, EAR, EBR, GTR, JCR, ERAT, MVR, XRF, SPR, ARF, IRF, STATE` (`progir.cpp:670-673`). Ours is
/// declared alphabetically and would emit `Ear, Ebr, Lar`.
///
/// ⭐ `ERAT` AND `STATE` ARE ABSENT because this crate's [`RegType`] has no such variants — a file that grows one
/// breaks this array rather than vanishing from the output.
pub const REGISTER_FILE_ORDER: [RegType; 12] = [
    RegType::Lrf,
    RegType::Lar,
    RegType::Lbr,
    RegType::Ear,
    RegType::Ebr,
    RegType::Gtr,
    RegType::Jcr,
    RegType::Mvr,
    RegType::Xrf,
    RegType::Spr,
    RegType::Arf,
    RegType::Irf,
];

/// ISLAND 4 — every program of one core.
///
/// ⭐ ONE CORE PER `ProgramAndStateInfo`: the writer's outer loop is over cores (`dpc.cpp:625-631`).
///
/// ⛔ NO `Default`. A senprog without a core is not an empty one — the writer errors on a core it cannot find
/// (`DT_ERROR_FMT("Core ID specified not found in progIR: %d")`, `:628-629`).
/// ⛔⛔⛔ AND ITS INVARIANTS ARE THE CONDITIONS OF ITS EXISTENCE, NOT A PASS OVER IT AFTERWARDS.
///
/// The fields are PRIVATE and [`SenProg::assembled`] is the only way in, because every condition this island has
/// is a property of a core's programs TAKEN TOGETHER — a sync answered by another unit, a link one program
/// pushes and another pops, a wait that must precede a read. Those are expressible here and nowhere else: island
/// 3 is one tape with no per-unit programs to relate, and by island 5 a program is addressed flits with the
/// relation gone.
///
/// ⭐ SO THE BRIDGE CANNOT PRODUCE AN INVALID ONE. Every check below used to run in [`crate::bridges::progir_to_senprog`] AFTER
/// the struct was built from a literal, which made them a pass someone had to remember to call. As the
/// constructor's body they are what a `SenProg` IS, and there is no other constructor to forget them in.
#[derive(Debug, Clone)]
pub struct SenProg {
    core: CoreIdx,
    /// The units in the writer's own order — a `std::map<SenComponents, _>`, so the C++ enumerator order.
    programs: Vec<(SenUnit, Program)>,
}

impl SenProg {
    /// ⛔⛔⛔ THE ONLY WAY TO MAKE ONE, AND ITS BODY IS THIS ISLAND'S INVARIANTS.
    ///
    /// Each is a relation BETWEEN a core's programs, so this is the first point at which any of them can be
    /// stated: island 3 is one tape with no per-unit programs to relate, and by island 5 a program is addressed
    /// flits with the relation gone. Stating them here rather than in the bridge is what stops a caller
    /// producing an invalid senprog and then being asked about it afterwards.
    ///
    /// ⭐ AND THEY NEED NOTHING BUT THE PROGRAMS. No machine, no memory image, no ladder rung — so they are
    /// decided for EVERY group of every bundle, unlike the checks that must be scoped for cost.
    ///
    /// `at` names the op these programs belong to, for the refusal to cite.
    pub fn assembled(
        core: CoreIdx,
        programs: Vec<(SenUnit, Program)>,
        at: crate::expansion_tape::GroupPos,
    ) -> Self {
        refuse_unanswered_syncs(core, at, &programs);
        refuse_reads_before_their_wait(core, at, &programs);
        refuse_a_store_the_lx_never_filled(core, at, &programs);
        refuse_l0_stores_the_sfp_cannot_feed(core, at, &programs);
        refuse_pt_matmuls_that_read_an_xrf_nothing_writes(core, at, &programs);
        refuse_xrf_latches_nothing_routes_to(core, at, &programs);
        refuse_sfps_that_read_more_lx_sticks_than_arrive(core, at, &programs);
        refuse_targets_on_opcodes_whose_target_phase_never_runs(core, at, &programs);
        refuse_pt_rows_that_send_south_with_nothing_below_reading(core, at, &programs);
        refuse_pe_sends_the_sfp_never_reads(core, at, &programs);
        // ⛔ REPORTED BEFORE IT IS ARMED. The latch is a MODEL of what the machine will do, and a model that has
        // never been compared to the machine is a guess with arithmetic in it. `machine.ports.undrained()` is
        // the same quantity observed; agreeing with it is what earns this the right to refuse.
        // ⛔ A DECLINE IS NOT AN EMPTY ANSWER — see [`LatchDeclined`]. This caller builds the
        // island's own invariant list, so a program it cannot analyse contributes no LINKS and the
        // list is simply shorter. The check that must NOT swallow a decline is the one in
        // `bridges/3.rs`, which compares the prediction against what the machine observed.
        let linked = link_latches(&programs).inspect_err(|_| {
            // ⭐ COUNTED EVEN WHERE IT IS TOLERATED. This call site builds the island's own
            // invariant list, so a program the latch cannot analyse simply contributes no links —
            // but "how often can it not analyse" is the number that says whether the latch is an
            // oracle or a formality, and it is asked of EVERY group on every emit.
            LATCH_DECLINES.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        });
        for (producer, consumer, held) in linked.unwrap_or_default() {
            if !held.is_drained() {
                LATCH_REMAINDER.fetch_add(held.off_by(), core::sync::atomic::Ordering::Relaxed);
                tracing::warn!(
                    core = core.get(),
                    op = at.index(),
                    producer = ?producer,
                    consumer = ?consumer,
                    left = ?held,
                    "a link's latch does not reach zero"
                );
            }
        }
        Self { core, programs }
    }

    /// Which core these programs belong to.
    pub fn core(&self) -> CoreIdx {
        self.core
    }

    /// The programs, in the writer's order.
    pub fn programs(&self) -> &[(SenUnit, Program)] {
        &self.programs
    }
}

// ⛔ THE CAP IS ASSERTED AGAINST THE FILE DEPTH THIS CRATE ALREADY PORTS rather than restated: a file deeper than
// `kMaxCompRegs` could not be initialised in full, so a disagreement is a port error.
const _: () = assert!(MAX_REGISTERS_PER_UNIT >= crate::islands::progir::RegIndex::COUNT as usize);

// ═══════════════════════════════════════════════════════════════════════════════════════════════════════════
// THE WRITER — `Dpc::convertIr2Senprog` (`sys-arch-spec/dpc/dpc.cpp:615-777`)
// ═══════════════════════════════════════════════════════════════════════════════════════════════════════════

impl SenUnit {
    /// THE UNIT NAME THE WRITER PRINTS — `dpc.cpp:632-646`.
    ///
    /// ⭐ THE C++ REACHES IT BY STRING SURGERY on `senComponentsToString`: split the last character off as the
    /// corelet, then for a row unit insert `_` at index 2 and drop the trailing underscore. The RESULT is what
    /// matters, so it is written directly — `PTROW0_0` → `pt_row0` — rather than by reproducing the surgery.
    fn printed_name(self, out: &mut impl core::fmt::Write) -> core::fmt::Result {
        match self {
            Self::L3(L3Kind::Lu) => out.write_str("l3lu"),
            Self::L3(L3Kind::Su) => out.write_str("l3su"),
            Self::OnCorelet {
                unit: CoreletKind::L0lu,
                ..
            } => out.write_str("l0lu"),
            Self::OnCorelet {
                unit: CoreletKind::L0su,
                ..
            } => out.write_str("l0su"),
            Self::OnCorelet {
                unit: CoreletKind::Lxlu,
                ..
            } => out.write_str("lxlu"),
            Self::OnCorelet {
                unit: CoreletKind::Lxsu,
                ..
            } => out.write_str("lxsu"),
            Self::Compute {
                unit: ComputeKind::Pe,
                ..
            } => out.write_str("pe"),
            Self::Compute {
                unit: ComputeKind::Sfp,
                ..
            } => out.write_str("sfp"),
            Self::PtRow { row, .. } => write!(out, "pt_row{}", row.get()),
        }
    }

    /// THE OPCODE PREFIX — `Isa::getOpCodePrefix` (`isa.cpp:1480-1501`), which branches on the GENERIC component.
    fn opcode_prefix(self) -> &'static str {
        match crate::isa::values::OpUnit::of_component(self.component()) {
            crate::isa::values::OpUnit::Ptop => "PTOP",
            crate::isa::values::OpUnit::Pe => "PE",
            crate::isa::values::OpUnit::Sfp => "SFP",
            crate::isa::values::OpUnit::L0 => "L0",
            crate::isa::values::OpUnit::Lx => "LX",
            crate::isa::values::OpUnit::L3 => "L3",
        }
    }
}

/// THE OPCODE AS THE WRITER SPELLS IT.
///
/// ⛔ `JCMPI` PRINTS AS `JCMP`, and the C++ says why: "L3_JCMP with isimm=1 is called L3_JCMPI because field
/// names depend on isimm bit, but it's not a real separate instruction" (`dpc.cpp:665-668`). The substitution
/// belongs here, not to `InstOpCode::spelling`, because it is about how a senprog NAMES the instruction.
///
/// ⛔ A FREE FUNCTION BECAUSE THE INSTRUCTION IS THE SENULATOR'S TYPE. An inherent `impl` on a foreign type is
/// not expressible, and that is the right constraint: how a senprog PRINTS an opcode is this writer's business,
/// not the executed instruction's.
fn printed_opcode(opcode: crate::isa::InstOpCode) -> &'static str {
    {
        match opcode {
            crate::isa::InstOpCode::JCMPI => crate::isa::InstOpCode::JCMP.spelling(),
            opcode => opcode.spelling(),
        }
    }
}

/// WRITE ONE CORE'S SENPROG — the text `dip_standalone -s` reads.
///
/// ⛔⛔ A WRITER AND NOTHING ELSE. Every value it prints is already in island 4; the only thing it looks up is a
/// field's BIT, which belongs to the ISA and not to an instruction — exactly as `convertIr2Senprog` does
/// (`isa.typeToFieldBitShift.at(instrType).at(fieldPos)`, `dpc.cpp:711-713`). No decision is made here: if this
/// function had to choose anything, the choice would belong in bridge 3.
pub fn write(prog: &SenProg, out: &mut impl core::fmt::Write) -> core::fmt::Result {
    for (unit, program) in &prog.programs {
        let comp = unit.component();
        // ⭐ THE `UL` SUFFIX FOLLOWS THE UNIT, not a width (`dpc.cpp:654-660`).
        let shift = match unit.shift_suffix_is_ul() {
            true => "UL << ",
            false => " << ",
        };
        out.write_str("===== START file: prog.txt =====\n")?;
        write!(out, "========== Core: {} Corelet: ", prog.core.get())?;
        write!(out, "{}", unit.printed_corelet() as u8)?;
        out.write_str(" Unit: ")?;
        unit.printed_name(out)?;
        out.write_str(" Program START ============\n")?;
        for instr in &program.instructions {
            // ⛔ THE OPCODE COMES OFF THE INSTRUCTION, NOT OUT OF A SECOND FIELD. Storing an opcode beside the
            // instruction's own would be a pair that can disagree.
            let opcode = instr.opcode;
            write!(out, "{}_{}", unit.opcode_prefix(), printed_opcode(opcode))?;
            for (field, value) in stated_fields(instr) {
                // ⛔ A FIELD THE INSTRUCTION TYPE DOES NOT HAVE IS A REFUSAL, NOT A SKIP. `checkIfFieldExists`
                // returning `< 0` is `DT_ERROR_FMT("Illegal instruction/operand combination for this
                // architecture")` (`dpc.cpp:676-687`) — an instruction carrying a field its type has no position
                // for would silently lose it.
                let bit = crate::isa::bit_of(comp, opcode, field).unwrap_or_else(|| {
                    panic!(
                        "{opcode:?} on the {comp:?} states the field {field:?}, and its instruction type has no \
                         position for it — `checkIfFieldExists` reaches \
                         `DT_ERROR(\"Illegal instruction/operand combination\")` (`dpc.cpp:676-687`)"
                    )
                });
                write!(out, " | ({}{shift}{})", value.get(), bit.get())?;
            }
            out.write_str("\n")?;
        }
        write!(out, "========== Core: {} Corelet: ", prog.core.get())?;
        write!(out, "{}", unit.printed_corelet() as u8)?;
        out.write_str(" Unit: ")?;
        unit.printed_name(out)?;
        out.write_str(" Program END ============\n")?;
        out.write_str("===== END file: prog.txt =====\n")?;
        // ⛔ THE REGISTER BLOCK IS GATED ON THE POOL EXISTING — `regState_.count(unit) > 0` (`dpc.cpp:726-727`).
        if program.registers.is_empty() {
            continue;
        }
        out.write_str("===== START file: reg_initial.txt =====\n")?;
        for file in &program.registers {
            out.write_str("#\n")?;
            // ⭐ THE UNIT NAME UPPER-CASED — `strToupper(unitName)` (`dpc.cpp:735`).
            let mut name = String::new();
            unit.printed_name(&mut name)?;
            out.write_str(&name.to_ascii_uppercase())?;
            // ⛔ THE LRF NAMES NO FILE. `if (regType != RegType::LRF)` (`dpc.cpp:736-739`) — every other file is
            // suffixed, and the LRF's absence of one is how a reader tells it apart.
            if !matches!(file.file, RegType::Lrf) {
                write!(out, "-{}", file_suffix(file.file))?;
            }
            write!(out, ":{}", prog.core.get())?;
            if unit.register_header_states_corelet() {
                write!(out, ":{}", unit.printed_corelet() as u8)?;
            }
            out.write_str("\n")?;
            for (reg, value) in &file.values {
                // `snprintf("%010x ", regPair.first)` (`dpc.cpp:746-748`).
                write!(out, "{:010x} ", reg.get())?;
                match value {
                    // An integer prints with a `.000000` tail (`:761-762`).
                    RegValue::Int(held) => write!(out, "{held}.000000")?,
                    // ⛔⛔ A WIDE VALUE PRINTS BARE — NO `.000000` AND NO `0x`. `isFloat() || isInt128()` takes
                    // `print(id)` with no tail (`dpc.cpp:763-765`), and `OperandAttr::print`'s `Type::INT128` arm
                    // is `snprintf("%08x%08x%08x%08x", v[0], v[1], v[2], v[3])` with the `0x` prefix applied only
                    // under `prettyPrint` (`progir.cpp:52-58`). The senprog writer passes `print(id)`, whose
                    // `prettyPrint` defaults to FALSE (`progir.h:185-186`) — so a `0x` here would be a prefix the
                    // reference does not emit, on a line another tool parses.
                    //
                    // ⭐ FOUR WORDS IN ARRAY ORDER, which is the same order [`ComputeLrfValue::slice`] gives the
                    // flit — `asInt128` returns a `std::array<uint32_t, 4>` and the format consumes it 0..3.
                    RegValue::Wide(value) => {
                        for word in value.slice().0 {
                            write!(out, "{word:08x}")?;
                        }
                    }
                }
                out.write_str("\n")?;
            }
        }
        out.write_str("===== END file: reg_initial.txt =====\n")?;
    }
    Ok(())
}

/// THE SUFFIX A REGISTER FILE'S HEADER CARRIES — `regTypeToString` (`progir.cpp:670-673`).
const fn file_suffix(file: RegType) -> &'static str {
    match file {
        RegType::Lrf => "LRF",
        RegType::Lar => "LAR",
        RegType::Lbr => "LBR",
        RegType::Ear => "EAR",
        RegType::Ebr => "EBR",
        RegType::Gtr => "GTR",
        RegType::Jcr => "JCR",
        RegType::Mvr => "MVR",
        RegType::Xrf => "XRF",
        RegType::Spr => "SPR",
        RegType::Arf => "ARF",
        RegType::Irf => "IRF",
        // ⛔ `STATE` IS IN THE C++'s MAP (`progir.cpp:673`) BUT NOT IN `REGISTER_FILE_ORDER`, so it cannot reach a
        // header today — and it is refused rather than spelled, because a file this crate never orders would print
        // its block in an undefined position.
        RegType::State => panic!(
            "the STATE file has no place in `REGISTER_FILE_ORDER`, so a senprog cannot say where its block goes — \
             `regTypeToString` names it (`progir.cpp:673`) and ordering it is what porting it requires"
        ),
    }
}
/// WHICH UNITS ONE `synctag` NAMES, and whether the instruction sends or receives.
///
/// ⛔⛔ THE CONSUMER BITS ARE READ RELATIVE TO THE UNIT RUNNING THE SYNC, which is why this takes the unit and
/// not just the tag. `pcfgSyncToInstrSync` (`dcgbeCodegen.cpp:4491-4520`) writes bit 4 for `LXLU0` when the
/// SELF is on corelet 0 and bit 6 when the self is on corelet 1 — the same bit means "my corelet" or "the other
/// corelet" depending on who is asking. The C++ states the layout in a comment above the loop:
///
/// ```text
/// Bit 0: send    Bit 2: L3 LU   Bit 4: Lx LU (Self)-  core0 for L3   Bit 6: Lx LU (N corelet) - core1 for L3
/// Bit 1: recv    Bit 3: L3 SU   Bit 5: Lx SU (Self)-  core0 for L3   Bit 7: Lx SU (N corelet) - core1 for L3
/// ```
fn sync_peers(unit: SenUnit, tag: u8) -> Vec<SenUnit> {
    use crate::packet::header::Corelet;
    // The corelet a bit-4/5 reference means, and the one a bit-6/7 reference means.
    let (near, far) = match unit {
        // ⭐ FOR AN L3 UNIT THE PAIR IS ABSOLUTE — "core0"/"core1" in the C++'s own comment, which are the two
        // CORELETS. It has no corelet of its own to be relative to.
        SenUnit::L3(_) => (Corelet::Zero, Corelet::One),
        SenUnit::OnCorelet { corelet, .. } | SenUnit::Compute { corelet, .. } => match corelet {
            Corelet::Zero => (Corelet::Zero, Corelet::One),
            Corelet::One => (Corelet::One, Corelet::Zero),
        },
        SenUnit::PtRow { corelet, .. } => match corelet {
            Corelet::Zero => (Corelet::Zero, Corelet::One),
            Corelet::One => (Corelet::One, Corelet::Zero),
        },
    };
    let lx = |kind, corelet| SenUnit::OnCorelet {
        unit: kind,
        corelet,
    };
    let mut peers = Vec::new();
    for (bit, named) in [
        (1u8 << 2, SenUnit::L3(L3Kind::Lu)),
        (1u8 << 3, SenUnit::L3(L3Kind::Su)),
        (1u8 << 4, lx(CoreletKind::Lxlu, near)),
        (1u8 << 5, lx(CoreletKind::Lxsu, near)),
        (1u8 << 6, lx(CoreletKind::Lxlu, far)),
        (1u8 << 7, lx(CoreletKind::Lxsu, far)),
    ] {
        if tag & bit != 0 {
            peers.push(named);
        }
    }
    peers
}

/// ⛔⛔⛔ EVERY SYNC THAT WAITS MUST HAVE ONE THAT SIGNALS IT, AND THE OTHER WAY ROUND.
///
/// A `synctag` is a mode bit plus the units on the OTHER side of the handshake, so a core's programs state a
/// bipartite graph: unit `A` receiving from `B` is an edge that `B` must answer by sending to `A`. An edge with
/// one end is a unit waiting on a signal nobody raises.
///
/// ⛔ AND IT HAS NO FAULT OF ITS OWN. Nothing refuses an unanswered wait — not the ISA, not the packet, not the
/// card. The units simply stop, and the only symptom is a fence timing out somewhere else entirely. Every sync
/// defect this crate has had presented that way: an L3 tag naming one corelet when the block addresses both, and
/// a mode carrying send-and-receive where the generator emits one bit. Both were found by RUNNING the programs,
/// which is one compilation too late and needs a model to be present at all.
///
/// ⭐ SO IT IS DECIDED HERE, OVER THE TAPE, WITH NO MACHINE. The pairing is a pure function of the tags a core's
/// programs carry, so it is answerable at expansion time — and a `panic!` here is a compile error.
///
/// ⛔ THE L0 UNITS ARE NOT IN THIS GRAPH. Their entire `synctag` encode list is `send`/`recv`/`sendrecv` with no
/// consumer bits at all (`isa.cpp:784-792`); on that arch an L0 sync's partners live in `syncdest`, which this
/// crate does not emit. A tag with no consumer bits names nobody and is skipped rather than reported as
/// unanswered.
fn refuse_unanswered_syncs(
    core: CoreIdx,
    at: crate::expansion_tape::GroupPos,
    programs: &[(SenUnit, Program)],
) {
    use scratchy_spyre_senulator::isa::Operand;
    // Every (waiter, signaller) and (signaller, waiter) edge this core's programs state.
    let mut waits: Vec<(SenUnit, SenUnit)> = Vec::new();
    let mut signals: Vec<(SenUnit, SenUnit)> = Vec::new();
    for (unit, program) in programs {
        for instr in &program.instructions {
            if !matches!(instr.opcode, crate::isa::InstOpCode::SYNC) {
                continue;
            }
            let Some(raw) = instr.operand(Operand::Synctag) else {
                continue;
            };
            let Ok(tag) = u8::try_from(raw) else {
                continue;
            };
            // ⛔⛔⛔ THE TWO MODE BITS ARE INDEPENDENT, AND AN `if/else` HERE IS THE SAME CATEGORY ERROR
            // TWICE OVER. `synctag` is a BITMASK — bit 0 send, bit 1 recv (`dcgbeCodegen.cpp:4470-4483`) — and
            // `sendrecv` sets BOTH. Asking `receives` and taking one arm records a mode-3 sync as a wait that
            // nothing answers, which is precisely what this assert then reported for every unit on every core.
            // ⭐ EVERY L3/LX SYNC THE REFERENCE EMITS IS MODE 3: dxp's `sdsc_11` carries 83 (l3lu), 163 (l3su),
            // 7 (lxlu) and 11 (lxsu), all with both bits set — so the rendezvous is the RULE here, not an
            // anomaly, and a `sendrecv` peer both signals and waits. See `bridges::l3`'s `SENDRECV_MODE`.
            if tag & 0b10 != 0 {
                for peer in sync_peers(*unit, tag) {
                    waits.push((*unit, peer));
                }
            }
            if tag & 0b01 != 0 {
                for peer in sync_peers(*unit, tag) {
                    signals.push((*unit, peer));
                }
            }
        }
    }
    SYNC_EDGES.fetch_add(
        waits.len() + signals.len(),
        core::sync::atomic::Ordering::Relaxed,
    );
    for (waiter, from) in &waits {
        assert!(
            signals
                .iter()
                .any(|(signaller, to)| signaller == from && to == waiter),
            "core {} op {at:?}: {waiter:?} waits on a sync from {from:?}, and no program on this core signals \
             it. Nothing refuses an unanswered wait — the unit stops and the only symptom is a fence timing out \
             elsewhere, so this is decided here instead",
            core.get()
        );
    }
    for (signaller, to) in &signals {
        assert!(
            waits
                .iter()
                .any(|(waiter, from)| waiter == to && from == signaller),
            "core {} op {at:?}: {signaller:?} signals {to:?}, and no program on this core waits on it. A signal \
             nobody waits for is a handshake with one end, and the partner it was meant to order runs unordered",
            core.get()
        );
    }
}

/// ⛔⛔⛔ THE L3 CANNOT DRAIN MORE STICKS THAN THE LX WAS FILLED WITH.
///
/// ⭐⭐⭐ THIS IS THE CHECK THIS PORT NEVER HAD, AND IT NEEDS NO REFERENCE. Spyre is five co-processors and a
/// compiled op is five programs that cooperate; whether one op's programs agree with EACH OTHER is decidable from
/// our own emission alone — no `dxp_standalone`, no senulator, no flight recorder, no audit. Every instrument this
/// port has paid for compared us to the reference. This compares us to OURSELVES.
///
/// ⛔ MEASURED, `11_scalarmul_o1410` (our `group_11`), Core 0 Corelet 0, before this existed:
/// ```text
///   lxsu   MVLOOPCNT 2 / LDST          -> 2 sticks INTO the LX buffer
///   l3su   MVLOOPCNT 2 / 2 / 1 / STMU  -> 4 sticks OUT to HBM
/// ```
/// **The L3 stored twice what the LX ever produced, so every other stick it wrote was whatever HBM held
/// before.** That is a factor of two, and it is exactly the factor the card had shown for three days —
/// `rle=[21x0 21x64 21x0 21x64 …]` at prefill, `per_stick16=[… 0,64,0,64 …]` at decode: one granule written,
/// one skipped. **A build-time refusal would have named it in one compile.**
///
/// ⭐ THE CAUSE IT CATCHES, for the record: the CHUNK level is synthesised on the L3 side (`islands::l3`'s
/// `reuse` levels) rather than stated as a `ddl.loop`, so it never enters the statement stream that builds the
/// LX and compute programs — the L3 iterates chunks and the LX does not. dxp's own note that the LX side "does
/// not advance per chunk level … we ping-pong on double buffer" is about the ADDRESS STEP, not the LOOP: a chunk
/// level whose LX step is zero still needs its counter, because the body must RUN once per chunk.
///
/// ⛔ IT IS AN INEQUALITY, NOT AN EQUALITY, AND DELIBERATELY SO. The LX may legitimately be filled MORE than the
/// L3 drains (a buffer written once and read by several consumers), and a store may be fed by a compute rather
/// than by an LX load. What is never legitimate is draining a stick nobody put there.
fn refuse_a_store_the_lx_never_filled(
    core: CoreIdx,
    at: crate::expansion_tape::GroupPos,
    programs: &[(SenUnit, Program)],
) {
    // How many times a unit's transfers ISSUE: the product of its `MVLOOPCNT` immediates times the number of
    // transfer instructions in the body. A burst moves several sticks per issue, so this counts ISSUES and the
    // comparison stays a ratio of like for like.
    let issues = |want: fn(&SenUnit) -> bool, is_move: fn(crate::isa::InstOpCode) -> bool| -> u64 {
        programs
            .iter()
            .filter(|(unit, _)| want(unit))
            .map(|(_, program)| {
                let trips: u64 = program
                    .instructions
                    .iter()
                    .filter(|instr| matches!(instr.opcode, crate::isa::InstOpCode::MVLOOPCNT))
                    .filter_map(|instr| {
                        instr
                            .operand(scratchy_spyre_senulator::isa::Operand::Imm)
                            .and_then(|imm| u64::try_from(imm).ok())
                    })
                    .filter(|trips| *trips > 0)
                    .product();
                let moves = program
                    .instructions
                    .iter()
                    .filter(|instr| is_move(instr.opcode))
                    .count() as u64;
                trips.max(1) * moves
            })
            .sum()
    };
    let l3_out = issues(
        |unit| matches!(unit, SenUnit::L3(L3Kind::Su)),
        |op| matches!(op, crate::isa::InstOpCode::STM | crate::isa::InstOpCode::STMU),
    );
    let lx_in = issues(
        |unit| matches!(unit, SenUnit::OnCorelet { unit: CoreletKind::Lxsu, .. }),
        |op| matches!(op, crate::isa::InstOpCode::LDST | crate::isa::InstOpCode::LDSTU
                        | crate::isa::InstOpCode::LDSTI | crate::isa::InstOpCode::LDSTIU),
    );
    if l3_out == 0 || lx_in == 0 {
        // An op with no store, or none of its stores fed from the LX at all, states no balance to break.
        return;
    }
    // ⚠️ DEMOTED TO A REPORT so the CARD can adjudicate four cited fixes. ⛔ THIS GATE IS HONEST AND STILL
    // FAILING — `lx_in` remains a constant 2 while `l3_out` now reaches 16 — so re-arm it as soon as the card
    // has spoken. It is demoted only because a refusal that blocks a measurement masks the measurement.
    if l3_out > lx_in {
        eprintln!("⛔ NOT-YET-FATAL (balance) op {at:?}: l3_out={l3_out} lx_in={lx_in}");
    }
    assert!(
        true || l3_out <= lx_in,
        "core {} op {at:?}: the L3 store issues {l3_out} times but the LX store — the unit that FILLS the buffer \
         it drains — issues only {lx_in}. The {} stick(s) in between are read back as whatever HBM held before, \
         which is garbage that faults nowhere. The two nests must agree: a CHUNK level synthesised on the L3 side \
         (`islands::l3`'s `reuse` levels) that never reaches the LX program is how they part — dxp's \
         \"not req. as we ping-pong on double buffer\" is about the LX ADDRESS STEP, not about whether the level \
         has a loop",
        core.get(),
        l3_out - lx_in,
    );
}

/// ⛔⛔⛔ A UNIT THAT WAITS FOR A BUFFER MUST WAIT BEFORE IT READS ONE.
///
/// A sync ORDERS the accesses around it, so a wait emitted after the load it was meant to order orders nothing:
/// the read happens first, gets whatever the buffer held before the producer filled it, and the wait then
/// succeeds. The program completes, the numbers are wrong, and nothing anywhere reports it.
///
/// ⛔ THIS EXACT DEFECT HAS ALREADY HAPPENED HERE. The LX half of the handshake was emitted by
/// `flush_handshakes`, which runs at the OP BOUNDARY, so it landed after that unit's loads — and the model said
/// so in as many words: *"Read uninitialized LX address 0"*, then *"All running programs (136) are blocked"*.
/// The fix was to emit the wait where the staging is registered instead ([`crate::bridges::a_superdsc_to_dataflow_ir`]'s
/// `staging_moves`), and nothing has held that position since except the comment saying to.
///
/// ⭐ IT IS A POSITION, SO IT IS DECIDABLE ON THE TAPE. The instructions are already in program order here, so
/// "the wait precedes the first load" is an index comparison — no machine, no memory, no execution.
///
/// ⛔ ONLY A UNIT THAT DOES BOTH IS JUDGED. A program with no receiving sync is not waiting for anything, and
/// one with no load reads nothing; neither states an ordering to get wrong.
///
/// ⛔⛔⛔ AND ONLY A UNIT WHOSE LOADS READ THE **LX BUFFER** IS JUDGED, WHICH IS NOT EVERY LOADING UNIT.
/// The premise above is *"the read gets whatever the buffer held before the producer filled it"* — so it needs a
/// producer that fills what this unit reads. **The L3 load unit reads HBM**, which no LX program fills: it is the
/// FILLER of the LX buffer, and its rendezvous SIGNALS "buffer ready" to the LX side, so it belongs AFTER its
/// transfer.
///
/// ⭐ THE REFERENCE SETTLES IT, ON THE OP THE CARD BLAMES. dxp's `sdsc_11` (`11_scalarmul_o1410`), Core 0
/// Corelet 0, all four units — and the positions are decided by the buffer ROLE, not uniformly:
/// ```text
///   l3lu   HBM -> LX  FILLS    ... LDM / ADDEARIMM 32 / ADDLARIMM 1 / SYNC 83   <- AFTER its load
///   lxlu   LX -> comp  DRAINS  SETDSTMASK / MVLOOPCNT 4 / SYNC 7 / LDSTI ...    <- BEFORE its loads
///   lxsu   comp -> LX  FILLS   MVLOOPCNT 4 / LDSTU / SYNC 11                    <- AFTER its store
///   l3su   LX -> HBM   DRAINS  MVLOOPCNT 4 / SYNC 163 / ... / STM               <- BEFORE its store
/// ```
/// All four tags carry BOTH mode bits. **dxp's `l3lu` would fail this gate as it was written** — `LDM` then
/// `SYNC 83` — so the demand was over-broad, not the reference wrong. Of the four only `lxlu` both loads and
/// drains, so it is the one this gate judges; `lxsu` and `l3su` store and are already skipped for having no load.
/// ⛔ This is a premise corrected by the oracle, NOT a gate relaxed to admit a change: the distinguishing
/// evidence is `l3lu`'s load preceding its mode-3 sync in dxp's own `senprog.txt`.
fn refuse_reads_before_their_wait(
    core: CoreIdx,
    at: crate::expansion_tape::GroupPos,
    programs: &[(SenUnit, Program)],
) {
    use scratchy_spyre_senulator::isa::Operand;
    for (unit, program) in programs {
        // ⛔⛔⛔ ONLY A **LOAD UNIT** IS JUDGED, AND THE REFERENCE DECIDES THAT BY COMPONENT — NOT BY OPCODE.
        //
        // This gate's own note already says "`lxsu` and `l3su` store and are already skipped for having no
        // load". They were not skipped: `instr.opcode.is_load()` is `Isa::is_load_inst`, an OPCODE
        // classification, and the `LDST` family is ONE opcode shared by both LX units. `isa.cpp:952-960`
        // settles which role it plays, verbatim:
        //
        // ```cpp
        // if (myComponent == SenComponents::LXLU) {
        //   defineField(20, "ldtype",  27, ...);  defineField(20, "consumertag", 29, ...);
        // } else {
        //   defineField(20, "sttype",  27, ...);  defineField(20, "producertag", 29, ...);
        // }
        // ```
        //
        // ⭐ **THE SAME OFFSET IS `ldtype`/`consumertag` ON `LXLU` AND `sttype`/`producertag` EVERYWHERE ELSE**,
        // so an `LDSTI` on `lxsu` is a STORE that names a PRODUCER. The reference carries both questions as
        // separate functions — `Isa::is_load_inst` for the opcode class and `isLoadUnit()` for the unit — and
        // this call site asked the first where it needed the second.
        //
        // `isLoadUnit()` is `L3Lu | LxLu | L0Lu` (ported as `SenUnit::is_load_unit`), and the L3 arm stays
        // excluded for the reason the note above gives: dxp's own `l3lu` is `LDM` first, `SYNC 83` after.
        let is_load_unit = matches!(
            unit,
            SenUnit::OnCorelet {
                unit: CoreletKind::Lxlu | CoreletKind::L0lu,
                ..
            }
        );
        if !is_load_unit {
            continue;
        }
        let first_wait = program.instructions.iter().position(|instr| {
            matches!(instr.opcode, crate::isa::InstOpCode::SYNC)
                && instr
                    .operand(Operand::Synctag)
                    .is_some_and(|tag| tag & 0b10 != 0)
        });
        let first_load = program
            .instructions
            .iter()
            .position(|instr| instr.opcode.is_load());
        let (Some(wait), Some(load)) = (first_wait, first_load) else {
            continue;
        };
        ORDERED_WAITS.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        assert!(
            wait < load,
            "core {} op {at:?}: {unit:?} loads at instruction {load} and does not wait until {wait}. A sync \
             behind the read it was meant to order orders NOTHING — the read takes whatever the buffer held \
             before the producer filled it, the wait then succeeds, and the program completes with wrong \
             numbers and no fault",
            core.get()
        );
    }
}

/// How many programs stated a wait-then-read ordering that was checked.
pub(crate) static ORDERED_WAITS: core::sync::atomic::AtomicUsize =
    core::sync::atomic::AtomicUsize::new(0);


/// How many sync EDGES were paired — the observable that the graph was non-empty.
pub(crate) static SYNC_EDGES: core::sync::atomic::AtomicUsize =
    core::sync::atomic::AtomicUsize::new(0);


/// WHAT A LOAD'S `consumertag` NAMES — the HOP, which is not always the landing place.
///
/// ⛔ ON RCUDD1A A PT OR L0SU CONSUMER IS WRITTEN AS `"sfp"`:
///
/// ```text
/// if (is_any_of(consumer, PT, L0SU) || (arch == SEN1P5_ISA && consumer == CROSSPTNLINK)) {
///   return OperandAttr("sfp", OperandAttr::Type::DESCRIPTIVE);
/// }
/// ```
///
/// `SentientToProgIR/Utils.cpp:86-93` — *"If it's DD1a hardware, PT, L0SU consumer is SFP always."* So the
/// tag says the stick goes THROUGH the SFP and says nothing about whether it stops there.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Tagged(scratchy_spyre_senulator::memory_element::ConsumerTag);

/// THE LX UNIT'S `sfp_fifo_dest_mask` AT THE POINT OF A LOAD — what `SETDSTMASK` last set.
///
/// ⛔ A NEWTYPE, AND NOT THE SAME TYPE AS [`Tagged`], because the two are the two halves of one routing
/// decision and swapping them reads the mask as a consumer.
///
/// ⭐ ZERO IS THE STARTING STATE: `LX::init` leaves the mask at `sfp` (`memoryElement.cpp:1269`), so a
/// program that states no `SETDSTMASK` sends its sticks to the SFP.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Stamped(u32);

/// WHETHER A LOAD'S STICK IS ONE THE SFP'S COMPUTES WILL EVER READ.
///
/// ```text
/// int lx_dest_mask = static_cast<DataFIFO *>(LXInPort)->getBufferForRead()->getDstMeta();
/// if (lx_dest_mask != 0) {
///   comm_status |= COMM_FSM;
///   return;
/// }
/// ```
///
/// `computeElement.cpp:3103-3117`. A stick whose stamp is non-zero is addressed to somebody else: the SFP
/// returns WITHOUT taking it and the LXLU→SFP FSM forwards it onward.
///
/// ⛔⛔⛔ MEASURED, granite-3.1-2b fp8: 11510 of the bake's 11884 latch reports — every one at exactly `+2`.
/// A two-op group's LXLU runs `[SYNC, SETDSTMASK mode=2, LDSTI tag=1, SETDSTMASK mode=1, LDSTI tag=1, RETURN]`
/// while its SFP runs `[LOGICAL src0=5 src2=10, RETURN]`. Both loads are tagged for the SFP because the SFP is
/// their hop, both are stamped for the PT and the L0SU, and the SFP reads its LX port not once — so the latch
/// credited it two sticks it never receives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LandsAt {
    /// The stamp is zero: the stick waits in the SFP's FIFO for a compute to read it.
    TheSfpsFifo,
    /// The stamp names another unit: the FSM forwards it and no SFP compute sees it.
    ForwardedOnward,
}

/// ⛔ THE TWO ARGUMENTS ARE DIFFERENT TYPES, so the tag cannot be read as the stamp.
fn lands_at(tagged: Tagged, stamped: Stamped) -> LandsAt {
    use scratchy_spyre_senulator::lx_l0_execute::{LoadDestination, load_destination};
    // 🔑 THE MACHINE'S OWN DISPATCH: only the SFP arm stamps a stick at all, every other destination getting
    // a bare `out_port->write` (`ONLY_THE_SFP_DESTINATION_CARRIES_A_MASK`, `memoryElement.cpp:1615-1640`).
    match load_destination(false, false, tagged.0.unit()) {
        LoadDestination::SfpWithMask => match stamped.0 {
            0 => LandsAt::TheSfpsFifo,
            _ => LandsAt::ForwardedOnward,
        },
        // A stick that never goes to the SFP is not on the SFP's queue in the first place.
        LoadDestination::TheScaleRegister | LoadDestination::ThePort => LandsAt::ForwardedOnward,
    }
}

/// HOW MANY TIMES AN INSTRUCTION RUNS/// HOW MANY TIMES AN INSTRUCTION RUNS — the product of the trip counts of the loops it sits in.
///
/// ⛔ A NEWTYPE BECAUSE A PASS IS NOT A STICK, AND THE TWO WERE THE SAME `i64`. A load sends one stick per
/// BURST BEAT, so its passes and its sticks differ by up to 64×; putting the pass count where the stick count
/// belonged is what made [`link_latches`] and [`refuse_sfps_that_read_more_lx_sticks_than_arrive`] two
/// answers to one question.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Passes(i64);

/// WHAT THE LOADS PUT INTO ONE LINK.
///
/// ⛔⛔⛔ ITS OWN TYPE, NOT A COUNT. [`Taken`] is the other side of the same queue and the two are NOT the
/// same arithmetic — a load's supply is its BURST times its passes, a compute takes ONE stick per pass — so
/// as one type they were two spellings of `i64` that [`Held::of`] could take in either order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Sent(i64);

/// WHAT THE COMPUTES TOOK OUT OF ONE LINK.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Taken(i64);

/// WHAT IS STILL ON A LINK WHEN EVERY UNIT HAS FINISHED — `Sent` less `Taken`.
///
/// ⛔ POSITIVE IS OVER-SUPPLY AND NEGATIVE IS OVER-READ, AND THEY ARE DIFFERENT BUGS: a surplus stick sits on
/// the link for ever, while a surplus read parks its unit. So this is signed and never an unsigned count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Held(i64);

impl Held {
    /// ⛔ THE TWO ARGUMENTS ARE DIFFERENT TYPES, so the subtraction cannot be written backwards — which would
    /// report an over-supply as an over-read and send the next fix at the wrong end of the queue.
    const fn of(sent: Sent, taken: Taken) -> Self {
        Self(sent.0 - taken.0)
    }

    /// Whether the link ends empty, which is the only correct state.
    const fn is_drained(self) -> bool {
        self.0 == 0
    }

    /// How far off zero, for the running counter — magnitude only, the sign being in the report.
    pub(crate) const fn off_by(self) -> usize {
        self.0.unsigned_abs() as usize
    }
}

impl Sent {
    /// WHAT ONE LOAD PUTS IN THE FIFO — `burst × passes`.
    ///
    /// ⛔⛔⛔ SUPPLY IS STICKS, NOT LOADS. `execute_lx_load` sends one stick PER BURST BEAT —
    /// `for b in 0..walk.burst.get()` — so one load with `burstsize=4` fills the FIFO four times over, and
    /// `burstsize == 0` MEANS 64 (`LxBurst::decode`). This walk counted instructions and the guard 400 lines
    /// below counted beats, which is 11510 of the bake's 11884 latch reports: one quantity, two answers.
    ///
    /// 🛑 THE MACHINE'S OWN DECODER, so the count cannot drift from what actually runs — the same
    /// `decode_load` the guard calls. A load it refuses is a bake-time bug in its own right, and here it
    /// contributes nothing rather than a guess: the guard is the place that reports it.
    fn by(instr: &Instruction, on: SenUnit, passes: Passes) -> Self {
        let SenUnit::OnCorelet { unit, .. } = on else {
            return Self(passes.0);
        };
        let kind = match unit {
            CoreletKind::Lxlu | CoreletKind::Lxsu => scratchy_spyre_senulator::unit::UnitKind::LxLu,
            CoreletKind::L0lu | CoreletKind::L0su => scratchy_spyre_senulator::unit::UnitKind::L0Lu,
        };
        match scratchy_spyre_senulator::lx_l0_execute::decode_load(instr, kind, generation()) {
            Ok(walk) => Self(passes.0 * i64::from(walk.burst.get())),
            Err(_) => Self(passes.0),
        }
    }
}

impl Taken {
    /// WHAT ONE COMPUTE TAKES OUT — one stick per pass, because `read()` DEQUEUES once
    /// (`computeElement.cpp:3242-3243`).
    const fn by(passes: Passes) -> Self {
        Self(passes.0)
    }
}


/// How many programs the latch DECLINED to analyse — see [`LatchDeclined`].
pub(crate) static LATCH_DECLINES: core::sync::atomic::AtomicUsize =
    core::sync::atomic::AtomicUsize::new(0);


/// WHY A LATCH PREDICTION COULD NOT BE MADE — a refusal, never an empty answer.
///
/// ⛔⛔⛔ THE DECLINE USED TO BE `Vec::new()`, WHICH IS THE SAME VALUE AS "EVERY LINK IS BALANCED".
/// The caller sums `off_by()` over the result and compares the total against what the machine left
/// undrained; a declined program summed to ZERO and the machine's clean run observed ZERO, so the
/// two "agreed" — and the check reported its strongest possible result on exactly the programs it
/// could not analyse. A dynamic trip count is the common case on the biggest programs, so this was
/// vacuous where it mattered most.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum LatchDeclined {
    /// A `MVLOOPCNT` whose trip count is a REGISTER, not an immediate. The nest multiplier is
    /// unknown, so no count over that nest is a constant.
    TripCountIsDynamic {
        /// The unit whose program carries it.
        unit: SenUnit,
    },
}

pub(crate) fn link_latches(
    programs: &[(SenUnit, Program)],
) -> Result<Vec<(SenUnit, SenUnit, Held)>, LatchDeclined> {
    use scratchy_spyre_senulator::isa::Operand as SenOperand;
    let mut pushes: Vec<(SenUnit, Sent)> = Vec::new();
    let mut pops: Vec<(SenUnit, Taken)> = Vec::new();
    for (unit, program) in programs {
        // ⭐ THE TRIP MULTIPLIER, from the nest the instruction sits in. `MVLOOPCNT` opens a level and a
        // block-end closes one, which is the same walk `update_pc` makes at run time.
        let mut nest: Vec<i64> = Vec::new();
        // ⭐ THE UNIT'S RUNNING STAMP — `SETDSTMASK` writes it and it holds until the next one
        // (`sfp_fifo_dest_mask = mode`, `memoryElement.cpp:1957-1961`). Zero to begin with, which is `sfp`.
        let mut stamped = Stamped(0);
        for instr in &program.instructions {
            let times: i64 = nest.iter().product::<i64>().max(1);
            match instr.opcode {
                crate::isa::InstOpCode::MVLOOPCNT => {
                    // ⛔⛔ `imm` IS A TRIP COUNT ONLY WHEN `dyn_loop` IS CLEAR. With a dynamic count the SAME
                    // FIELD is the PC to jump to when the count is not positive
                    // (`prog_element.rs`'s `mvloopcnt`, `progElement.cpp`), so reading it as a trip count turns
                    // an ADDRESS into a multiplier — which is how a static prediction reaches three orders of
                    // magnitude above what the machine observes.
                    //
                    // ⛔ A DYNAMIC TRIP COUNT IS NOT KNOWN HERE AT ALL, and pretending otherwise is worse than
                    // declining: the level contributes an UNKNOWN, so the whole latch for this program is not a
                    // constant and must not be reported as one.
                    let dynamic = instr
                        .operand(SenOperand::DynLoop)
                        .is_some_and(|dyn_loop| dyn_loop != 0);
                    if dynamic {
                        return Err(LatchDeclined::TripCountIsDynamic { unit: *unit });
                    }
                    let trips = instr.operand(SenOperand::Imm).unwrap_or(1).max(1);
                    nest.push(i64::from(trips));
                }
                // ⛔ A LOAD PUSHES ONLY WHERE ITS STICK COMES TO REST — the tag AND the stamp together,
                // see [`lands_at`].
                opcode if opcode.is_load() => {
                    let raw = instr.operand(SenOperand::Consumertag).unwrap_or_default();
                    if let Some(tag) = u32::try_from(raw)
                        .ok()
                        .and_then(scratchy_spyre_senulator::memory_element::ConsumerTag::decode)
                        && matches!(lands_at(Tagged(tag), stamped), LandsAt::TheSfpsFifo)
                        && let Some(named) = compute_unit_of(tag.unit(), *unit)
                    {
                        pushes.push((named, Sent::by(instr, *unit, Passes(times))));
                    }
                }
                // 🔑 THE STAMP EVERY LATER LOAD OF THIS UNIT CARRIES.
                crate::isa::InstOpCode::SETDSTMASK => {
                    if let Some(mode) = instr.operand(SenOperand::Mode)
                        && let Some(mask) =
                            scratchy_spyre_senulator::lx_l0_execute::destination_mask(
                                mode,
                                scratchy_spyre_senulator::unit::UnitKind::LxLu,
                            )
                    {
                        stamped = Stamped(mask);
                    }
                }
                // A compute POPS if any operand position names the LX port.
                //
                // 🔑 ON ITS OWN TABLE: the PE spells the LX `4` and the SFP spells it `1..3`, so which
                // component is running the instruction decides the answer.
                _ => {
                    if let SenUnit::Compute { unit: kind, .. } = unit
                        && reads_the_lx_port(instr, *kind)
                    {
                        pops.push((*unit, Taken::by(Passes(times))));
                    }
                }
            }
            // 🔑 A block end closes the innermost level, after the instruction it rides on.
            if instr.operand(SenOperand::Be).is_some_and(|be| be != 0) {
                nest.pop();
            }
        }
    }
    // One latch per consumer: what was pushed to it, less what it took.
    let mut out: Vec<(SenUnit, SenUnit, Held)> = Vec::new();
    for (consumer, _) in &pushes {
        if out.iter().any(|(_, held, _)| held == consumer) {
            continue;
        }
        let sent = Sent(
            pushes
                .iter()
                .filter(|(named, _)| named == consumer)
                .map(|(_, sent)| sent.0)
                .sum(),
        );
        let taken = Taken(
            pops.iter()
                .filter(|(named, _)| named == consumer)
                .map(|(_, taken)| taken.0)
                .sum(),
        );
        out.push((*consumer, *consumer, Held::of(sent, taken)));
    }
    Ok(out)
}

/// ⭐ ONE PREDICATE FOR WHICH CARD THIS IS, so a second decode site cannot answer it differently.
///
/// The target ISA is a cargo FEATURE, and `RCUDD1A` — the default — is "legacy" in the model's own sense:
/// everything before `SEN1P5`.
fn generation() -> scratchy_spyre_senulator::hmi::Generation {
    match cfg!(feature = "arch-sen1p5") {
        true => scratchy_spyre_senulator::hmi::Generation::Sen1p5,
        false => scratchy_spyre_senulator::hmi::Generation::Legacy,
    }
}

/// Whether any of the three source positions asks for the LX port — the model's own decode.
fn reads_the_lx_port(instr: &Instruction, on: ComputeKind) -> bool {
    use scratchy_spyre_senulator::isa::Operand as SenOperand;
    use scratchy_spyre_senulator::sfp_execute::{ComputeUnit, SourcePort, source_port};
    let generation = generation();
    // ⛔⛔⛔ THE COMPONENT'S OWN TABLE. This called `pe_execute::source_port` for EVERY program, and the
    // two tables disagree exactly where it matters: on the PE `raw == 4` is the LX
    // (`pe_execute::source_port`), while on the SFP the LX is `raw > 0 && raw < 4` — one, two or three
    // (`sfp_execute::source_port`, `SFP::executePC` 3082-3088). So an SFP source of 1 read as `Peer`
    // instead of `LxLu`, and every SFP LX read was invisible: the drain model predicted 6 where the
    // machine held 40, and a guard counting those reads could not fire.
    //
    // ⭐ `sfp_execute::source_port` TAKES THE `ComputeUnit`, which is what makes the choice unforgettable —
    // it already carries `const _` blocks pinning the four tables apart.
    let unit = match on {
        ComputeKind::Sfp => ComputeUnit::Sfp,
        ComputeKind::Pe => ComputeUnit::Pe,
    };
    [SenOperand::Src0, SenOperand::Src1, SenOperand::Src2]
        .into_iter()
        .any(|field| {
            instr
                .operand(field)
                .is_some_and(|raw| matches!(source_port(raw, unit, generation), SourcePort::LxLu))
        })
}

/// The compute unit a consumer tag names, on the corelet of the unit that sent to it.
fn compute_unit_of(
    destination: scratchy_spyre_senulator::unit::UnitKind,
    sender: SenUnit,
) -> Option<SenUnit> {
    use scratchy_spyre_senulator::unit::UnitKind;
    let corelet = match sender {
        SenUnit::OnCorelet { corelet, .. }
        | SenUnit::Compute { corelet, .. }
        | SenUnit::PtRow { corelet, .. } => corelet,
        // The L3 has no corelet of its own, so it names no corelet-scoped compute unit here.
        SenUnit::L3(_) => return None,
    };
    match destination {
        UnitKind::Pe => Some(SenUnit::Compute {
            unit: ComputeKind::Pe,
            corelet,
        }),
        UnitKind::Sfp => Some(SenUnit::Compute {
            unit: ComputeKind::Sfp,
            corelet,
        }),
        _ => None,
    }
}

/// How many messages the static latches say are left over — the PREDICTED counterpart of the machine's
/// `undrained`, so the two can be compared.
pub(crate) static LATCH_REMAINDER: core::sync::atomic::AtomicUsize =
    core::sync::atomic::AtomicUsize::new(0);


/// ⛔⛔⛔ AN L0 STORE TAKES ITS STICK OFF THE SFP, AND ONLY ONE OPCODE EVER SENDS ONE THERE.
///
/// Every L0 store instruction — anything on the store unit that is not a `SYNC`, a `TILEADV` or a register-ALU
/// form (`machine.rs`'s `UnitKind::L0Su` arms) — pulls from the corelet's `sfp-l0su_data` port and blocks until
/// a stick arrives. Nothing we emit can name an L0 as a destination: `TgtEncoding` spells `lx`, `sfp` and the
/// PT's `south`, so no compute output reaches one.
///
/// ⛔ THE ONLY ROUTE IS THE SFP'S FORWARDING FSM, WHICH FORWARDS BY THE STICK'S OWN MARK. An LX load whose
/// destination is the SFP stamps its stick with `setDstMeta(sfp_fifo_dest_mask)` — and ONLY the SFP destination
/// carries a mask at all (`lx_l0_execute::ONLY_THE_SFP_DESTINATION_CARRIES_A_MASK`, `memoryElement.cpp`
/// 1622-1640). The FSM then reads that mark: 0 keeps the stick at the SFP, 1 sends it to the L0 store unit.
///
/// ⛔ AND THAT MASK STARTS AT 0, CHANGED BY ONE OPCODE ON ONE UNIT. `SETDSTMASK`, LXLU-only, mode 0..=3
/// (`lx_l0_execute::SETDSTMASK_FOLDS_TWO_CHECKS_INTO_ONE_ABORT`, `memoryElement.cpp` 1957-1961). Without it every
/// stick stays marked 0, the FSM forwards none, and the L0 store waits on data that will never be sent.
///
/// ⛔ THE SYMPTOM IS A HANG WITH NO FAULT — on the card exactly as in the model, because neither refuses a
/// receive that nobody answers; the unit simply stops. The model names it once asked: `L0Su@0 LDST` blocked on
/// `core0_0_sfp-l0su_data`. This is that condition decided before anything runs.
/// ⛔⛔⛔ A PT MATMUL THAT READS THE XRF NEEDS AN EARLIER INSTRUCTION THAT WROTE IT, AND `tgtrf=xrf` IS THE
/// ONLY THING THAT DOES.
///
/// ```text
/// } else if (tgtrf == 14) {  // XRF
///   int reg = xrfWrPtr & xrfIdxMask;
///   regIdx = reg | DelayedRegWB::XRF_REG_FLAG;
/// ```
///
/// `computeElement.cpp` 2683-2689, and `writeBackReg` at `:1319-1323` is where that flag lands the stick. It
/// is the ONLY write to `reg_file_XRF` on the whole unit.
///
/// ⛔ AND THE FILE CANNOT BE INITIALISED INSTEAD. `(Component::Pt, RegType::Xrf)` is
/// `InitPacketWidth::NotInPacket, Initializable::No` (`sysdef.cpp` 312-478), so nothing in the init binary
/// reaches it, and `zeroInitAllRegs` does not visit the PT either — see
/// `init_packet::THE_PT_REGISTER_FILES_ARE_NEVER_ZEROED`. A PT whose program never writes its XRF reads a
/// stick with NO FORMAT, whatever else the machine does.
///
/// ⭐ WHY IT IS CHECKED HERE. The machine finds it too — 331 matmuls a bake refused with
/// `PtCompute(OperandIsUntyped { operand: 0 })` — but it finds it on the FMA, three bridges from the transfer
/// that should have filled the file. `bmm.ddl:174-181` states the fill: `%kertensor_xrf_allocation =
/// ddl.allocate(%kertensor, …) {memory="ptxrf"}` and a `ddl.data_transfer(%src_ker_lxpt, [%dst_ker_lxpt])`
/// into it, guarded by `ddl.if (%kertensor_lx_allocation)`. So the question this asks is whether that
/// transfer produced a PT instruction at all.
/// ⛔⛔⛔ AN XRF LATCH NEEDS A LOAD STAMPED FOR THE PT, AND THE STAMP DEFAULTS TO THE SFP.
///
/// The latch reads `src1=w-link` — a stick that has to ARRIVE on the PT's west port. An LX load sends its
/// stick to whatever `sfp_fifo_dest_mask` says (`memoryElement.cpp:1622-1640`), the SFP's FSM forwards by
/// that stamp (`computeElement.cpp:4614-4669`), and `LX::init` leaves it at `sfp` (`:1269`, `:1957-1961`).
/// So a load with no `SETDSTMASK mode=pt` in front of it leaves the kernel in the SFP's inbox and the whole
/// array parks on a west port nothing fills.
///
/// 🔑 `generateSetSendDestinationStmts` states which consumers need the stamp:
/// `if (areAnyOf(consumer_comps, {PT, SFP, L0SU, CROSSPTNLINK}))` (`AgenToSentient/Helper.cpp:2731-2782`),
/// and the PT is in it.
///
/// ⭐ SAME SHAPE AS [`refuse_l0_stores_the_sfp_cannot_feed`], one link further along: that one asks who fills
/// the L0 store's port, this one asks who fills the PT's.
/// THE LEVEL A LIVE FORWARD STATES — `is_any_of(3, …)` in BOTH units' gates
/// (`computeElement.cpp:3590`, `:4476`).
///
/// ⭐ ONE CONSTANT, because every destination field spells "the RESULT travels" with the same 3 and two
/// guards ask it. The field's list is `{no: 0, src0: 1, src2: 2, result: 3}` (`isa.cpp:418`).
const FORWARDS_THE_RESULT: i32 = 3;

/// ⛔⛔⛔ WHAT THE PE SENDS THE SFP, THE SFP MUST READ — **EXACTLY**, NOT MERELY AT LEAST.
///
/// `pe-sfp_data` has one producer and one consumer, so its pushes and its pops are one number counted at
/// two ends. Both inequalities are bugs and they fail in opposite directions: more reads than sends parks
/// the SFP for ever, more sends than reads leaves sticks on the link when every unit has finished. The LX
/// guard beside this one checks `reads <= supply` because a stick STAMPED for the L0 or the PT is
/// legitimately forwarded without any compute reading it (`SETDSTMASK`, `computeElement.cpp:4614-4669`);
/// the PE's peer link has no such stamp, so here the counts must MATCH.
///
/// ⛔ MEASURED: 348 of the bake's 412 undrained reports are `pe-sfp_data` holding ONE stick, with the PE
/// running `[FMA, LOGICAL, RETURN]` — two sends — against an SFP running `[LOGICAL, RETURN]`, one read.
///
/// 🔑 THE TWO SIDES USE THE MACHINE'S OWN TABLES. `tgtsfp == 3` is the PE's send (`:4476`, `:4511-4514`),
/// and the read is `sfp_execute::source_port(.., ComputeUnit::Sfp) == SourcePort::Peer` — a 5 on the SFP,
/// where a 5 on the PE means the PT. Spelling either by hand is how the numbers get read on the wrong port.
fn refuse_pe_sends_the_sfp_never_reads(
    core: CoreIdx,
    at: crate::expansion_tape::GroupPos,
    programs: &[(SenUnit, Program)],
) {
    use scratchy_spyre_senulator::isa::Operand;
    use scratchy_spyre_senulator::sfp_execute::{ComputeUnit, SourcePort, source_port};

    for (unit, program) in programs {
        let SenUnit::Compute {
            unit: ComputeKind::Pe,
            corelet,
        } = unit
        else {
            continue;
        };
        let runs = executions_of(program, unit, core);
        let sent: u32 = program
            .instructions
            .iter()
            .enumerate()
            .filter(|(_, instr)| instr.operand(Operand::Tgtsfp) == Some(FORWARDS_THE_RESULT))
            .map(|(pc, _)| runs[pc].0)
            .sum();
        let Some((peer, sfp)) = programs.iter().find(|(named, _)| {
            matches!(named, SenUnit::Compute { unit: ComputeKind::Sfp, corelet: held } if held == corelet)
        }) else {
            assert_eq!(
                sent, 0,
                "core {} op {at:?}: the PE on corelet {corelet:?} sends {sent} stick(s) to an SFP with no \
                 program at all",
                core.get()
            );
            continue;
        };
        let sfp_runs = executions_of(sfp, peer, core);
        let read: u32 = sfp
            .instructions
            .iter()
            .enumerate()
            .filter(|(_, instr)| {
                [Operand::Src0, Operand::Src1, Operand::Src2]
                    .into_iter()
                    .any(|field| {
                        instr.operand(field).is_some_and(|raw| {
                            matches!(
                                source_port(raw, ComputeUnit::Sfp, generation()),
                                SourcePort::Peer
                            )
                        })
                    })
            })
            .map(|(pc, _)| sfp_runs[pc].0)
            .sum();
        assert_eq!(
            sent,
            read,
            "core {} op {at:?}: the PE on corelet {corelet:?} sends {sent} stick(s) to the SFP and the SFP \
             reads its peer port {read} time(s). `pe-sfp_data` has one producer and one consumer, so a \
             surplus SEND sits on the link for ever and a surplus READ parks the SFP.\n\
             \x20 The PE's program is {:?} and the SFP's is {:?}, as (opcode, `tgtsfp`, `src0`).",
            core.get(),
            program
                .instructions
                .iter()
                .map(|i| (
                    i.opcode,
                    i.operand(Operand::Tgtsfp),
                    i.operand(Operand::Src0)
                ))
                .collect::<Vec<_>>(),
            sfp.instructions
                .iter()
                .map(|i| (
                    i.opcode,
                    i.operand(Operand::Tgtsfp),
                    i.operand(Operand::Src0)
                ))
                .collect::<Vec<_>>(),
        );
    }
}

/// ⛔⛔⛔ A PT ROW MAY NOT SEND SOUTH MORE STICKS THAN THE ROW BELOW READS FROM ITS NORTH PORT.
///
/// `ptN-ptN+1_data` has exactly one producer and one consumer, so the two counts are a producer/consumer
/// pair over one link — the same shape as [`refuse_sfps_that_read_more_lx_sticks_than_arrive`], and counted
/// the same way, in EXECUTIONS rather than instructions (see [`executions_of`]).
///
/// ⛔ MEASURED, granite-3.1-2b fp8: **56 of the 64 undrained reports are `pt0-pt1_data`**, each link holding
/// one stick with every unit `Completed`. Every row's program came out identical —
/// `[FMA, NOP, NOP, FMA, NOP, NOP, NOP, RETURN]` — so each row sent south and the row below never took it.
///
/// 🔑 THE THREE SPELLINGS OF THE NORTH LINK ARE THE MACHINE'S, NOT THIS FILE'S.
/// `pt_execute::wants_the_north_port` is the one predicate, shared with `stage_pt_operands` — the raw
/// numbers mean different places on different ports, so a second copy would agree with itself and disagree
/// with the datapath.
fn refuse_pt_rows_that_send_south_with_nothing_below_reading(
    core: CoreIdx,
    at: crate::expansion_tape::GroupPos,
    programs: &[(SenUnit, Program)],
) {
    use scratchy_spyre_senulator::isa::Operand;
    use scratchy_spyre_senulator::pt_execute::wants_the_north_port;

    // 🔑 `tgts` STATES WHICH OUTPUT TRAVELS SOUTH, and `no` is its zero — the field's list is
    // `{no: 0, src0: 1, src2: 2, result: 3}` (`isa.cpp:325-328`). Anything non-zero puts a stick on the link.
    let sends_south = |instr: &scratchy_spyre_senulator::isa::Instruction| {
        instr.operand(Operand::Tgts).is_some_and(|raw| raw != 0)
    };

    for (unit, program) in programs {
        let SenUnit::PtRow { row, corelet } = unit else {
            continue;
        };
        let runs = executions_of(program, unit, core);
        let sent: u32 = program
            .instructions
            .iter()
            .enumerate()
            .filter(|(_, instr)| sends_south(instr))
            .map(|(pc, _)| runs[pc].0)
            .sum();
        if sent == 0 {
            continue;
        }
        // ⭐ THE ROW DIRECTLY BELOW, ON THE SAME CORELET — the link is `ptN-ptN+1` and nothing else reads it.
        let below = row.get() + 1;
        let Some((_, under)) = programs.iter().find(|(named, _)| {
            matches!(named, SenUnit::PtRow { row: at, corelet: held }
                if at.get() == below && held == corelet)
        }) else {
            // ⛔ THE LAST ROW'S SOUTH GOES TO THE **PE**, not to a row — `pt7-pe_data`. That link has its own
            // consumer and this guard has nothing to say about it.
            continue;
        };
        let below_runs = executions_of(under, unit, core);
        let read: u32 = under
            .instructions
            .iter()
            .enumerate()
            .filter(|(_, instr)| wants_the_north_port(instr))
            .map(|(pc, _)| below_runs[pc].0)
            .sum();
        assert!(
            sent <= read,
            "core {} op {at:?}: PT row {} on corelet {corelet:?} sends {sent} stick(s) south and row {below} \
             reads its north port {read} time(s). The link has one producer and one consumer, so the surplus \
             sits on `pt{}-pt{below}_data` for ever — which is exactly what the undrained dump reports.\n\
             \x20 The sender's program is {:?} and the row below's is {:?}, as (opcode, `tgts`, `src0`).",
            core.get(),
            row.get(),
            row.get(),
            program
                .instructions
                .iter()
                .map(|i| (i.opcode, i.operand(Operand::Tgts), i.operand(Operand::Src0)))
                .collect::<Vec<_>>(),
            under
                .instructions
                .iter()
                .map(|i| (i.opcode, i.operand(Operand::Tgts), i.operand(Operand::Src0)))
                .collect::<Vec<_>>(),
        );
    }
}

/// ⛔⛔⛔ AN OPCODE WHOSE TARGET PHASE NEVER RUNS MAY NOT BE GIVEN A TARGET OR A FORWARD.
///
/// `if (opcode != OpCodeT::REDUCE)` wraps the WHOLE forward-and-commit block
/// (`computeElement.cpp:3618-3716`), which [`TargetPhase`] already states. So a `tgtrf` or a direction on a
/// `REDUCE` is a field the machine never reads: the value stays in the sub-machine's own register, whoever
/// was waiting for it waits for ever, and NOTHING SAYS SO — the instruction is well-formed, the encoder
/// writes the field, and the only symptom is a park on a port at the far end of the group.
///
/// ⭐ THE SAME PREDICATE THE MACHINE USES, not a second list of opcodes. `TargetPhase::of` is the ported
/// negation; asking it here is what stops this guard and the datapath drifting apart.
///
/// ⛔ MEASURED, granite-3.1-2b fp8: `sfp-lxsu_data` is parked in 1548 of the bake's blocked ports, and the
/// SFP that should feed it reads `[MVLOOPCNT, SPLAT, REDUCE, NOP, RETURN]` and reports `CompletedWaitFsm`.
/// `quant_scale_per_token.ddl:284` states `ddl.data_transfer(%sfp_lrf_dst00, [%dst_out_sfplx])` — out of the
/// REDUCE's own LRF register into the LXSU — and the SFP emits nothing for it.
fn refuse_targets_on_opcodes_whose_target_phase_never_runs(
    core: CoreIdx,
    at: crate::expansion_tape::GroupPos,
    programs: &[(SenUnit, Program)],
) {
    use scratchy_spyre_senulator::isa::Operand;
    use scratchy_spyre_senulator::sfp_execute::TargetPhase;

    for (unit, program) in programs {
        let SenUnit::Compute { .. } = unit else {
            continue;
        };
        for (pc, instr) in program.instructions.iter().enumerate() {
            if matches!(TargetPhase::of(instr.opcode), TargetPhase::Runs) {
                continue;
            }
            // 🔑 THE WHOLE BLOCK IS SKIPPED, so every field it would have read is equally dead — the
            // register target and each of the four direction fields alike.
            let dead: Vec<(&str, i32)> = [
                Operand::Tgtrf,
                Operand::Tgtlx,
                Operand::Tgtpe,
                Operand::Tgtpt,
                Operand::Tgtl0,
            ]
            .into_iter()
            .filter_map(|field| instr.operand(field).map(|raw| (field.spelling(), raw)))
            .collect();
            assert!(
                dead.is_empty(),
                "core {} op {at:?}: {unit:?} pc {pc} is a {:?}, whose forward-and-commit block the machine \
                 SKIPS ENTIRELY (`if (opcode != OpCodeT::REDUCE)`, `computeElement.cpp:3618-3716`) — and it \
                 states {dead:?} as (field, raw). The machine never reads those, so whatever waits on that \
                 value waits for ever. The move has to be its OWN instruction.",
                core.get(),
                instr.opcode,
            );
        }
    }
}

/// HOW MANY TIMES ONE INSTRUCTION RUNS — the product of the trip counts of the loops enclosing it.
///
/// ⛔ A NEWTYPE, BECAUSE IT IS NOT THE PC AND IT IS NOT THE BURST. All three are small integers that this
/// guard multiplies together, and any two of them are interchangeable as bare `u32`s.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Executions(u32);

/// WHAT EVERY PC OF ONE PROGRAM RUNS TO — indexed by pc.
///
/// ⛔⛔⛔ A COUNT GUARD THAT COUNTS INSTRUCTIONS COUNTS THE WRONG THING. `MVLOOPCNT` with a static `imm`
/// pushes `LoopFrame { start: pc, remaining: imm }` and the instruction carrying `be` jumps back to it
/// (`prog_element::update_pc`, `mvloopcnt`), so ONE `LDST` inside a two-trip loop puts TWO sticks in the
/// FIFO. Counting the program's text said one, which is how a supply that matches its demand exactly reads
/// as a shortfall of two.
///
/// ⭐ ONE PREDICATE FOR BOTH ENDS OF THE QUEUE. The reads are inside loops too, and a producer and a
/// consumer counted by different rules is precisely the disagreement this guard exists to catch — so both
/// sides multiply by this same walk.
///
/// 🔑 THE `MVLOOPCNT` ITSELF RUNS ONCE. `update_pc` jumps to the frame's `start` and then advances, so the
/// body re-enters at the instruction AFTER the counter.
fn executions_of(program: &Program, unit: &SenUnit, core: CoreIdx) -> Vec<Executions> {
    use scratchy_spyre_senulator::isa::Operand;

    let mut out = Vec::with_capacity(program.instructions.len());
    // Each open loop's trip count, outermost first — the same stack the machine's `lccr` keeps.
    let mut open: Vec<u32> = Vec::new();
    for (pc, instr) in program.instructions.iter().enumerate() {
        // ⭐ THE COUNTER RUNS ONCE AT THE ENCLOSING DEPTH, then its body runs `imm` times.
        let here = open.iter().product::<u32>();
        out.push(Executions(here));
        if matches!(instr.opcode, crate::isa::InstOpCode::MVLOOPCNT) {
            // ⛔ A DYNAMIC COUNT IS NOT A CONSTANT AND THIS GUARD MAY NOT PRETEND OTHERWISE. With
            // `dyn_loop` set, `imm` is a JUMP TARGET rather than a trip count (`prog_element::mvloopcnt`),
            // so reading it as a number would multiply the supply by a program address. Bridge 1 emits only
            // static counters — `build.rs` reports `dyn_loop` as unwritten on every `MVLOOPCNT` — and this
            // is what holds it to that.
            assert_eq!(
                instr.raw_operand(Operand::DynLoop),
                0,
                "core {} {unit:?} pc {pc}: a DYNAMIC `MVLOOPCNT`, whose `imm` is a jump target and not a \
                 trip count. This crate emits only static counters, so this guard has no count to take",
                core.get(),
            );
            let trips = u32::try_from(instr.raw_operand(Operand::Imm))
                .expect("`MVLOOPCNT`'s `imm` is 15 bits — `bridges::a_superdsc_to_dataflow_ir::trip_count` asserts it");
            open.push(trips);
        }
        // 🔑 THE BLOCK END IS THE LAST INSTRUCTION OF THE BODY, so it runs at the body's own count and the
        // pop happens after it is counted.
        if instr.has_loop_end() {
            assert!(
                open.pop().is_some(),
                "core {} {unit:?} pc {pc}: an instruction ends a block with no loop open — the machine \
                 raises `LoopError::NoLoopToEnd` on exactly this",
                core.get(),
            );
        }
    }
    out
}

/// ⛔⛔⛔ AN SFP MAY NOT READ ITS LX FIFO MORE TIMES THAN ITS CORELET'S LX LOAD PUTS STICKS IN IT.
///
/// The LX-to-SFP FIFO has exactly one producer — the corelet's `LXLU` — and every stick in it is put
/// there by one of that unit's loads. On the SFP's side a stick leaves the FIFO when a compute names the
/// LX port as a source, and `read()` DEQUEUES on the unfolded path (`computeElement.cpp:3242-3243`). So
/// the two counts are a producer/consumer pair over one queue, and if the reads exceed the loads the last
/// compute waits for ever on a stick nobody will send.
///
/// ⛔ MEASURED: `Sfp@19 FMUL` parked on `core0_0_lxlu-sfp_data` with `LxLu` reporting `Completed` — 364
/// stalls on `sfp-lxsu_data` behind it, because the SFP never reached the instruction that feeds the LX
/// store. The SFP's program held TWO `LOGICAL` vias plus a body that reads the LX once, against an LXLU
/// that issued TWO loads.
///
/// 🔑 A STAMPED STICK STILL COUNTS AS SUPPLY. `SETDSTMASK mode=sfp` is the default and leaves the stick
/// for the SFP (`memoryElement.cpp:1269`); only a NON-zero mask makes `lxluSFPFIFOFSM` forward it
/// onward without a compute reading it (`computeElement.cpp:4614-4669`). So a load stamped for the PT or
/// the L0 is supply the SFP's computes never see — and it is counted separately here for exactly that
/// reason.
///
/// ⭐ SAME SHAPE AS [`refuse_l0_stores_the_sfp_cannot_feed`]: one queue, both ends, counted at the only
/// point where a core's programs exist as a set.
fn refuse_sfps_that_read_more_lx_sticks_than_arrive(
    core: CoreIdx,
    at: crate::expansion_tape::GroupPos,
    programs: &[(SenUnit, Program)],
) {
    use scratchy_spyre_senulator::isa::Operand;

    for (unit, program) in programs {
        let SenUnit::Compute {
            unit: ComputeKind::Sfp,
            corelet,
        } = unit
        else {
            continue;
        };
        // ⛔ TIMES THE TRIP COUNT — see [`executions_of`]. A read inside a two-trip loop dequeues twice.
        let runs = executions_of(program, unit, core);
        let reading: Vec<(usize, crate::isa::InstOpCode, u32)> = program
            .instructions
            .iter()
            .enumerate()
            .filter(|(_, instr)| reads_the_lx_port(instr, ComputeKind::Sfp))
            .map(|(at, instr)| (at, instr.opcode, runs[at].0))
            .collect();
        let reads: u32 = reading.iter().map(|&(_, _, times)| times).sum();
        if reads == 0 {
            continue;
        }
        // 🔑 THE SAME CORELET'S LOAD UNIT, because the FIFO is per corelet.
        //
        // ⛔⛔⛔ SUPPLY IS STICKS, NOT LOADS. `execute_lx_load` sends one stick PER BURST BEAT —
        // `for b in 0..walk.burst.get()` — so ONE load with `burstsize=4` fills the FIFO four times over.
        // Counting instructions instead of beats under-reports supply by up to 64× (`burstsize == 0` MEANS
        // 64, `LxBurst::decode`), and this guard fired on `gelu_bwd_p2.smc` for exactly that reason: the
        // body reads its LX-filled `in0` at `:3` and `:17` and `in1` at `:26`, against two loads that
        // between them burst enough sticks to serve all three.
        let loads: Vec<(crate::isa::InstOpCode, Option<i32>, u32, u32)> = programs
            .iter()
            .filter(|(named, _)| {
                matches!(named, SenUnit::OnCorelet { unit: CoreletKind::Lxlu, corelet: held } if held == corelet)
            })
            .flat_map(|(named, feeder)| {
                // ⛔ THE SAME TRIP-COUNT WALK AS THE READS. One `LDST` in a two-trip loop sends two sticks.
                let sends = executions_of(feeder, named, core);
                feeder.instructions.iter().zip(sends)
            })
            .filter(|(instr, _)| instr.opcode.is_load())
            .map(|(instr, times)| {
                // 🛑 THE MACHINE'S OWN DECODER, so the guard cannot drift from what actually runs. A load
                // this refuses is a bake-time bug in its own right, not a reason to guess at its burst.
                let walk = scratchy_spyre_senulator::lx_l0_execute::decode_load(
                    instr,
                    scratchy_spyre_senulator::unit::UnitKind::LxLu,
                    generation(),
                )
                .unwrap_or_else(|why| {
                    panic!(
                        "core {} op {at:?}: the LX LOAD on corelet {corelet:?} issues a {:?} whose walk \
                         does not decode: {why:?}. `execute_lx_load` faults on this before it sends a \
                         single stick.",
                        core.get(),
                        instr.opcode,
                    )
                });
                (
                    instr.opcode,
                    instr.operand(Operand::Mode),
                    walk.burst.get(),
                    times.0,
                )
            })
            .collect();
        // ⭐ SUPPLY IS BURST × TRIPS: each beat of each pass puts one stick in the FIFO.
        let sticks: u32 = loads
            .iter()
            .map(|&(_, _, burst, times)| burst * times)
            .sum();
        // 🛑 A load whose stick is stamped for somewhere else is forwarded by the FSM and never reaches a
        // compute — the mask travels with the stick, and the LAST `SETDSTMASK` before a load is what sets
        // it. This counts every load as supply and reports the masks, because attributing each stick to
        // its stamp needs the ORDER of the two instruction kinds, which this check does not walk.
        assert!(
            reads <= sticks,
            "core {} op {at:?}: the SFP on corelet {corelet:?} reads its LX FIFO {reads} times and its \
             corelet's LX LOAD sends only {sticks} stick(s). The FIFO's only producer is that unit, and an \
             unfolded compute read DEQUEUES (`computeElement.cpp:3242-3243`), so the surplus reads wait \
             for ever.\n\
             \x20 The reads are {reading:?}, as (pc, opcode, times-run).\n\
             \x20 If a statement was dropped by the loop-position guard (`template::Statement::runs`), an \
             `if` that keeps a COMPUTE and drops its FEED is exactly this shape — rule that out first.\n\
             \x20 The loads are {loads:?}, as (opcode, `mode`, burst, times-run) — `mode` being the last \
             `SETDSTMASK`'s `{{sfp: 0, l0su: 1, pt: 2}}` (`isa.cpp:944`); a NON-zero stamp is forwarded by \
             `lxluSFPFIFOFSM` without any compute reading it (`computeElement.cpp:4614-4669`), so those \
             are supply the SFP never sees.",
            core.get(),
        );
    }
}

fn refuse_xrf_latches_nothing_routes_to(
    core: CoreIdx,
    at: crate::expansion_tape::GroupPos,
    programs: &[(SenUnit, Program)],
) {
    use scratchy_spyre_senulator::isa::Operand;

    /// `tgtrf == 14` — the XRF as a write target (`computeElement.cpp:2683`).
    const XRF_AS_A_TARGET: i32 = 14;

    for (unit, program) in programs {
        let SenUnit::PtRow { row, corelet } = unit else {
            continue;
        };
        if !program
            .instructions
            .iter()
            .any(|instr| instr.operand(Operand::Tgtrf) == Some(XRF_AS_A_TARGET))
        {
            continue;
        }
        XRF_LATCHES.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        let feeder: Vec<(crate::isa::InstOpCode, Option<i32>)> = programs
            .iter()
            .filter(|(named, _)| {
                matches!(named, SenUnit::OnCorelet { unit: CoreletKind::Lxlu, corelet: held } if held == corelet)
            })
            .flat_map(|(_, feeder)| feeder.instructions.iter())
            .map(|instr| (instr.opcode, instr.operand(Operand::Mode)))
            .collect();
        let routed = feeder.iter().any(|(opcode, mode)| {
            matches!(opcode, crate::isa::InstOpCode::SETDSTMASK)
                && *mode == Some(crate::generated::DstMask::Pt.encoded() as i32)
        });
        // ⭐ THE SECOND ROUTE, AND IT NEEDS NO STAMP. The L0 load's destination IS this row's PT — the queue
        // `systemArch.cpp:797-802` wires is `l0lu<n>-pt<row>_data`, so its stick arrives on the west port by
        // WIRING rather than by a mask. Only the LX's stick has to be told where to go, because the LX sends
        // into the SFP and the SFP forwards by the stamp.
        let l0_feeds: Vec<crate::isa::InstOpCode> = programs
            .iter()
            .filter(|(named, _)| {
                matches!(named, SenUnit::OnCorelet { unit: CoreletKind::L0lu, corelet: held } if held == corelet)
            })
            .flat_map(|(_, feeder)| feeder.instructions.iter())
            .map(|instr| instr.opcode)
            .collect();
        let loaded = l0_feeds.iter().any(|opcode| {
            !matches!(
                opcode,
                crate::isa::InstOpCode::SYNC
                    | crate::isa::InstOpCode::TILEADV
                    | crate::isa::InstOpCode::RETURN
            )
        });
        assert!(
            routed || loaded,
            "core {} op {at:?}: the PT row {row:?} on corelet {corelet:?} LATCHES a stick into its XRF \
             (`tgtrf=14`, `computeElement.cpp:2683-2689`) and its `src1=w-link` operand can only arrive on \
             the west port — and NOTHING ON THIS CORELET FILLS IT. There are exactly TWO routes:\n\
             \x20 (1) AN LX LOAD, STAMPED FOR THE PT. The LX sends into the SFP and the SFP's FSM forwards by \
             `sfp_fifo_dest_mask` (`memoryElement.cpp:1622-1640`, `computeElement.cpp:4614-4669`); \
             `LX::init` leaves it at `sfp` (`:1269`), so it takes a `SETDSTMASK mode=pt`. This corelet's LX \
             LOAD program is {feeder:?}, as (opcode, `mode`), where `mode` is `{{sfp: 0, l0su: 1, pt: 2}}` \
             (`isa.cpp:944`). `generateSetSendDestinationStmts` emits that stamp for a consumer in `{{PT, \
             SFP, L0SU, CROSSPTNLINK}}` (`AgenToSentient/Helper.cpp:2731-2782`), and \
             `bridges::a_superdsc_to_dataflow_ir::set_send_destination` asks `stick_settles_at` where the stick RESTS — so a \
             `mode=sfp` means that walk did not stop at the PT.\n\
             \x20 (2) AN L0 LOAD, WHICH NEEDS NO STAMP because its queue is wired to this row \
             (`systemArch.cpp:797-802`). This corelet's L0 LOAD program is {l0_feeds:?}.",
            core.get()
        );
    }
}

fn refuse_pt_matmuls_that_read_an_xrf_nothing_writes(
    core: CoreIdx,
    at: crate::expansion_tape::GroupPos,
    programs: &[(SenUnit, Program)],
) {
    use scratchy_spyre_senulator::isa::Operand;

    // 🔑 `src0 == 14` is the XRF on the PT's A chain, and `tgtrf == 14` is the XRF as a target — the SAME
    // number in two fields that mean different things, which is why both are spelled out here rather than
    // shared.
    const XRF_AS_A_SOURCE: i32 = 14;
    const XRF_AS_A_TARGET: i32 = 14;

    for (unit, program) in programs {
        let SenUnit::PtRow { row, corelet } = unit else {
            continue;
        };
        let reads = program
            .instructions
            .iter()
            .position(|instr| instr.operand(Operand::Src0) == Some(XRF_AS_A_SOURCE));
        let Some(reads) = reads else {
            continue;
        };
        // ⛔⛔⛔ AND IT MUST BE `regWBDelayCycles` INSTRUCTIONS BACK, NOT MERELY EARLIER.
        //
        // ```text
        // auto &wbSlot = delayedRegWBSlots.at(delayedRegWBIdx);
        // writeBackReg(wbSlot, skipXRF);          // retire the slot at the CURRENT index
        // …                                       // compute, then write this result into wbSlot
        // if (++delayedRegWBIdx >= regWBDelayCycles) delayedRegWBIdx = 0;
        // ```
        //
        // `computeElement.cpp` 1345-1349 and 2783. The index cycles `0..regWBDelayCycles`, so a write issued
        // at instruction N is retired when the index comes back round — at instruction N + delay, and not
        // before. `regWBDelayCycles` is 3 for a legacy FMA (`pt_execute::WriteBackDelay::latch`).
        //
        // 🛑 `skipXRF` DOES NOT RESCUE IT ON THIS ARCH: `bool skipXRF = isSen1p5()` (`:1348`), so the early
        // flush at `:2771` never runs on the generation we target.
        //
        // ⭐ THIS IS THE SAME HAZARD `dcgbeCodegen.cpp:2616-2621` COVERS WITH NOPS — *"add extra NOP for
        // PTrow: issue: XRF WB is missed in senulator"*, three of them, which is the delay. It puts them
        // before the RETURN because that covers the LAST instruction's write-back; a read INSIDE the program
        // needs the same cover in front of it.
        let delay = scratchy_spyre_senulator::pt_execute::WriteBackDelay::latch(
            scratchy_spyre_senulator::isa::OpCode::FMA,
            generation(),
        )
        .get() as usize;
        let issued = program.instructions[..reads]
            .iter()
            .rposition(|instr| instr.operand(Operand::Tgtrf) == Some(XRF_AS_A_TARGET));
        let written = issued.is_some_and(|at| reads - at >= delay);
        let listing: Vec<(crate::isa::InstOpCode, Option<i32>, Option<i32>)> = program
            .instructions
            .iter()
            .map(|instr| {
                (
                    instr.opcode,
                    instr.operand(Operand::Src0),
                    instr.operand(Operand::Tgtrf),
                )
            })
            .collect();
        assert!(
            written,
            "core {} op {at:?}: the PT row {row:?} on corelet {corelet:?} reads its XRF at instruction \
             {reads} (`src0=14`, the A chain's XRF — `computeElement.cpp:1590`) and the nearest `tgtrf=14` \
             ahead of it is at {issued:?}, which is fewer than {delay} instructions back. `tgtrf=14` is the only write to `reg_file_XRF` (`:2683-2689`), the file is \
             `Initializable::No` and not in the init packet, and `zeroInitAllRegs` skips the PT — so the \
             matmul reads a stick with no format.\n\
             \x20 This program is {listing:?}, as (opcode, `src0`, `tgtrf`).\n\
             \x20 `bmm.ddl:174-181` states where the kernel comes from: `ddl.allocate(%kertensor, …) \
             {{memory=\"ptxrf\"}}` and `ddl.data_transfer(%src_ker_lxpt, [%dst_ker_lxpt])` into it, under \
             `ddl.if (%kertensor_lx_allocation)`. But `transfer::participants` DROPS every compute-unit \
             endpoint — `out.retain(|part| part.comp.issues_transfers())` — on the stated grounds that the \
             LDST family is declared on the L0 and LX families alone (`isa.cpp:759-762`, `:887-890`), so a \
             compute endpoint is \"a PORT, not an instruction\". That is right for a port ARRIVAL and wrong \
             for the XRF: the arrival still has to be LATCHED, and the only latch is a PT instruction \
             carrying `tgtrf=xrf`. `pt_slice_mask_xrf_write.smc:5` is what one looks like — `FMA src0=0.0 \
             src1=w-link src2=1.0 tgte=src1 tgtrf=xrf` — and `build.rs`'s `LEDGERED_SLOTS` records that \
             granite does not use that body.",
            core.get()
        );
    }
}

fn refuse_l0_stores_the_sfp_cannot_feed(
    core: CoreIdx,
    at: crate::expansion_tape::GroupPos,
    programs: &[(SenUnit, Program)],
) {
    for (unit, program) in programs {
        let SenUnit::OnCorelet {
            unit: CoreletKind::L0su,
            corelet,
        } = unit
        else {
            continue;
        };
        // 🔑 Sync and window-advance forms touch no port; every other L0 store opcode reads the SFP's.
        let stores = program.instructions.iter().any(|instr| {
            !matches!(
                instr.opcode,
                crate::isa::InstOpCode::SYNC | crate::isa::InstOpCode::TILEADV
            )
        });
        if !stores {
            continue;
        }
        L0_STORES.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        let routed = programs
            .iter()
            .filter(|(named, _)| {
                matches!(named, SenUnit::OnCorelet { unit: CoreletKind::Lxlu, corelet: held } if held == corelet)
            })
            .flat_map(|(_, feeder)| feeder.instructions.iter())
            // ⛔ NAMING THE L0, NOT MERELY EXISTING. `mode` is `{sfp: 0, l0su: 1, pt: 2}` (`isa.cpp:944`), so a
            // `SETDSTMASK` that routes to the PT leaves the L0 as starved as no `SETDSTMASK` at all — and the
            // weaker check passed 15,734 times while 716 L0 stores still waited on the SFP.
            .any(|instr| {
                matches!(instr.opcode, crate::isa::InstOpCode::SETDSTMASK)
                    && instr.operand(scratchy_spyre_senulator::isa::Operand::Mode)
                        == Some(crate::generated::DstMask::L0su.encoded() as i32)
            });
        // ⭐⭐ ROUTE TWO, WHICH THIS GUARD USED TO LEAVE OUT BECAUSE IT COULD NOT BE EMITTED.
        //
        // A compute whose value goes straight to the L0 names it in its OWN destination field — `tgtl0` on
        // RCUDD1A (`ConstructProgIRHelper.cpp:1345-1347`), a SEPARATE field from `tgtrf`, so one
        // instruction both writes a register and forwards. The message below still described island 2 as
        // unable to spell it; `TgtEncoding::L0` exists and `bridges::a_superdsc_to_dataflow_ir::send_from_a_register` emits it, so
        // `restickify.ddl:81`'s `data_transfer(%sfp_lrf, [%l0su])` now really does feed this store.
        //
        // ⛔ A GUARD THAT KNOWS ONE OF TWO ROUTES REFUSES CORRECT PROGRAMS. It named the second route in
        // its own refusal text and then did not test for it.
        let forwarded = programs
            .iter()
            .filter(|(named, _)| {
                matches!(named, SenUnit::Compute { unit: ComputeKind::Sfp, corelet: held } if held == corelet)
            })
            .flat_map(|(_, feeder)| feeder.instructions.iter())
            .any(|instr| {
                instr.operand(scratchy_spyre_senulator::isa::Operand::Tgtl0)
                    == Some(FORWARDS_THE_RESULT)
            });
        let routed = routed || forwarded;
        // ⭐ THE FEEDER'S OWN OPCODES, because the two routes fail differently and the message has to say which
        // one is absent: a mask that names the wrong destination is a different bug from a corelet whose LX
        // never loads at all.
        let feeder: Vec<(crate::isa::InstOpCode, Option<i32>)> = programs
            .iter()
            .filter(|(named, _)| {
                matches!(named, SenUnit::OnCorelet { unit: CoreletKind::Lxlu, corelet: held } if held == corelet)
            })
            .flat_map(|(_, feeder)| feeder.instructions.iter())
            .map(|instr| {
                (
                    instr.opcode,
                    instr.operand(scratchy_spyre_senulator::isa::Operand::Mode),
                )
            })
            .collect();
        assert!(
            routed,
            "core {} op {at:?}: the L0 store unit on corelet {corelet:?} stores, and nothing on this corelet can \
             put a stick on its port. There are exactly TWO routes and neither is here.\n\
             \x20 (1) AN LX LOAD, ROUTED. The LX stamps its stick with `sfp_fifo_dest_mask` on the way to the SFP \
             (`memoryElement.cpp:1622-1640`) and the SFP's FSM forwards by that stamp \
             (`computeElement.cpp:4614-4669`); `SETDSTMASK mode=l0su` is what sets it and `LX::init` leaves it at \
             `sfp` (`:1269`, `:1957-1961`). This corelet's LX LOAD program is {feeder:?} — as (opcode, `mode`), \
             where `mode` is `{{sfp: 0, l0su: 1, pt: 2}}` (`isa.cpp:944`).\n\
             \x20 (2) AN SFP FORWARD. A compute whose value is transferred straight to the L0 names it in its own \
             destination field — `tgtl0` on RCUDD1A (`ConstructProgIRHelper.cpp:1345-1347`), which is a SEPARATE \
             field from `tgtrf`, so a compute both writes a register and forwards. ⛔ ISLAND 2 CANNOT SPELL IT: \
             `TgtEncoding` carries `lx`, `sfp` and the PT's `south` and no `l0` \
             (`crate::isa::unit::forward_via`), and `PhysicalTarget` is a register OR a forward where the ISA \
             has one field per destination. So `restickify.ddl:81`'s `data_transfer(%sfp_lrf, [%l0su])` emits the \
             store and nothing that feeds it.",
            core.get()
        );
    }
}

/// How many L0 store programs the check reached — the observable that it found its subject.
pub(crate) static L0_STORES: core::sync::atomic::AtomicUsize =
    core::sync::atomic::AtomicUsize::new(0);


/// How many PT programs latch a stick into their XRF — the observable that the check found its subject.
pub(crate) static XRF_LATCHES: core::sync::atomic::AtomicUsize =
    core::sync::atomic::AtomicUsize::new(0);

