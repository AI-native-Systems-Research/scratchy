//! BRIDGE 1, THE PIVOTED PATH — scratchy's SuperDSC handed to the RUST port.
//!
//! ```text
//! SubtileTape ──lower_subtile_tape_to_superdsc──► SuperDSC (Dsc / SdscOp)
//!             ──deeptools::bridges::superdsc_to_dataflow_ir──► DataflowIR
//!             ──dbo-opt --from-dfir──► init_binary
//! ```
//!
//! ⭐⭐ THIS REPLACES [`crate::lower_subtile_tape_to_dataflow_ir`], WHICH IS ABANDONED. That module
//! hand-wrote the lowering; the audit found it emitting an HBM offset as an LX address
//! (`dbo-opt`: *"Register initialization out of boundary: lxsu0 : LRF0 : 2416640"*, which is exactly
//! `1208320 * 2` for the `get_logical_memory_view %C0-lx, 1208320` it emitted on the `lxsu`), with
//! `Residence::Lx` never constructed outside a test and the store side hardcoding `from: lx` where
//! the input side matches on residence. The port is the same conversion taken from the C++ instead:
//! 23,646 lines, 110/110 functions carrying a `Replaces: eNNN_name` anchor.
//!
//! ⛔⛔ WHAT THIS FILE STILL OWES, STATED SO IT CANNOT BE MISTAKEN FOR DONE. The port consumes a
//! **scheduled** DSC and scratchy builds an **unscheduled** one. The C++ `ScheduleNode::NodeType` is
//! `{BLOCK, LOOP, TRANSFER, COMPUTE, SYNC, CONDITION, ALLOCATE, STICKMASK}` (`dsc/dsc2.h:446-456`)
//! and the port's `Statement` is those less `ALLOCATE` (`superdsc_to_dataflow_ir/driver.rs:467`) —
//! but [`crate::lower_subtile_tape_to_superdsc::Dsc`]'s `scheduleTree_` is a `Vec<AllocNode>` whose
//! every `nodeType_` is `"allocate"`, with the computes in a separate flat `computeOp_` list. The
//! LOOP/TRANSFER/SYNC/CONDITION nodes are built by `ddc` — `ddl/ddl_conversion.cpp:1065` mints the
//! `dsc2::LoopNode` from the DDL's `ddl.loop`, names it `loop_ds<num>_ds<den>` and registers it in
//! `ddlInterface.loop_labels_`.
//!
//! ⭐ SO THIS GROWS ONE STATEMENT KIND AT A TIME, and until a kind lands its absence is an ABSENT
//! STATEMENT rather than a substituted one — which the driver already handles, since a component the
//! schedule names no root for binds nothing (the port's entry 108). While [`Schedule::roots`] yields
//! nothing, `convert_v3` binds no unit and [`lower_superdsc_to_dataflow_ir`] returns [`None`]: the
//! traits are satisfied over scratchy's own types and NO PROGRAM IS PRODUCED YET. Anything that
//! reports otherwise is reporting a stub as a bake.
//!
//! ⛔ AND NOTHING HERE REFUSES. No `Err`, and no panic on a shape it has not met.

use deeptools::arch::Arch;
use deeptools::bridges::superdsc_to_dataflow_ir::driver::{
    Dsc as PortDsc, ScheduleView, Scheduled, Translated, UnitHandles, Uniformize, Viewed, Viewing,
    run_translator,
};
use deeptools::bridges::superdsc_to_dataflow_ir::control_flow::PrimaryDim;
use deeptools::islands::dataflow_ir::ty::GenericComp;
use deeptools::bridges::superdsc_to_dataflow_ir::driver::{Emitted, TransferStatement};
use deeptools::bridges::superdsc_to_dataflow_ir::dsc_lowering::{
    Bound, Component, DataLocation, Handlers, constant_index,
};
use deeptools::bridges::superdsc_to_dataflow_ir::transfer::{
    ContiguousSticks, ContiguousTransfer, DataTransfer, DstFormats, DstVia, EndFormat, LoadSource,
    LoadAndStoreSource, LoadAndStoreTransfer, Replication, StickCounts, TransferRead,
    Uniformization, UnitTimeChunks, ViewLoops, ViewSize,
    generate_load_and_send_from_data_transfer_node,
    generate_load_and_store_from_data_transfer_node,
};
use deeptools::bridges::superdsc_to_dataflow_ir::utils::DscKind;
use deeptools::generated::DataType;
use deeptools::islands::dataflow_ir::dialects::{Op, Val};
use deeptools::islands::dataflow_ir::link::{DynLink, RecvEnd};
use deeptools::islands::dataflow_ir::ty::{ElemType, TensorCategory, Vector};

/// A TRANSFER THAT IS NOT REPLICATED — the default factor, which `Replication` will only build
/// through its checked constructor.
const REPLICATION_ONE_SRC: i64 = 1;
use deeptools::islands::dataflow_ir::{Grid, KernelName, ProgramName, Run, Values, print};
use deeptools::units::{Core, Corelet, DfirUnit, NumFolds};

use crate::lower_subtile_tape_to_superdsc::Dsc as SuperDsc;

/// ONE SCRATCHY `Dsc`, IN THE SHAPE THE PORT ASKS FOR.
///
/// ⛔ IT BORROWS RATHER THAN OWNS. The port's `Dsc<'c>` hands out `Viewing<'c, Self>` and the
/// `Scheduled<'s>` a view yields borrows for `'s` with `'c: 's`, so everything a statement points at
/// has to outlive the walk. Holding the scratchy `Dsc` by reference is what makes that true without
/// a clone per component.
pub struct OneDsc<'c> {
    /// The unscheduled SuperDSC this program was built from.
    dsc: &'c SuperDsc,
    /// Which cores it occupies, as the port's newtype.
    cores: Vec<Core>,
    /// WHERE EVERY BORROWED STATEMENT PAYLOAD LIVES.
    ///
    /// ⛔⛔ THE STATEMENTS CANNOT OWN THEIR PAYLOADS AND `roots` CANNOT KEEP THEM. The port's
    /// `Emitted::Compute` holds `ctx: &'s OperandContext<'s>`, `Statement::Transfer` holds
    /// `&'s TransferStatement<'s>` whose `send`/`store`/`receive` are `&'s dyn Fn(..)`, and
    /// [`ScheduleView::roots`] takes `self` BY VALUE — so anything built inside `roots` is dropped
    /// at its return and the borrow cannot outlive it.
    ///
    /// ⛔⛔ AND IT IS BORROWED, NOT OWNED, WHICH THE SIGNATURE FORCES. [`PortDsc::view`] takes
    /// `&self` and returns `Viewing<'c, Self>`, so a payload it hands out must live for `'c` — but
    /// `&self.arena` on an OWNED field is only good for the anonymous borrow of that call, which is
    /// shorter. Holding `&'c Bump` makes the field `Copy` at `'c` and the return well-formed. The
    /// caller owns the `Bump` and drops it after the walk.
    arena: &'c bumpalo::Bump,
}

impl<'c> OneDsc<'c> {
    /// WRAP ONE SCRATCHY `Dsc`.
    ///
    /// ⛔ THE CORE LIST IS `coreIdsUsed_`, NOT `0..numCoresUsed_`. A bundle may occupy a
    /// non-contiguous set, and the port indexes handles by the core id it is given.
    #[must_use]
    pub fn of(dsc: &'c SuperDsc, arena: &'c bumpalo::Bump) -> OneDsc<'c> {
        let cores = dsc
            .coreIdsUsed_
            .iter()
            .filter_map(|id| Core::checked(*id))
            .collect();
        OneDsc { dsc, cores, arena }
    }
}

/// ONE COMPONENT'S ROOTS.
pub struct Schedule<'c> {
    /// Which component the driver asked about.
    comp: DfirUnit,
    /// The SuperDSC the statements are read out of.
    dsc: &'c SuperDsc,
    /// Where this component's statement payloads are allocated — see [`OneDsc::arena`].
    arena: &'c bumpalo::Bump,
}

impl<'c> ScheduleView<'c> for Schedule<'c> {
    /// ⛔⛔ YIELDS NOTHING YET, AND THAT IS THE OWED WORK, NOT A DESIGN. The statements belong to
    /// `ddc`'s expansion of this component's DDL template against its allocations; building a
    /// `Statement` from anything else is the invention this pivot exists to remove. The component and
    /// the DSC are both in hand here, so this is the one function the ported expansion plugs into.
    fn roots<'s>(self, _vals: &mut Values, handles: UnitHandles<'s>) -> Vec<Scheduled<'s>>
    where
        'c: 's,
    {
        // ⛔ ONE COMPONENT, ONE STATEMENT KIND. Only the LX load unit yields anything yet, and only
        // a single TRANSFER. Every other component is an ABSENT root, which the driver already
        // handles (entry 108: a component the schedule names no root for binds nothing).
        //
        // ⛔⛔ AND IT IS `Lxlu`, NOT `L3lu`, BECAUSE `sen_components()` DOES NOT WALK THE L3 UNITS
        // (`driver.rs:995-1009`: the PT rows, `Pe`, `Sfp`, `L0lu`, `L0su`, `Lxlu`, `Lxsu`). An
        // earlier version put this on `L3lu` and produced NO program at all — `view()` is never
        // asked about a component the walk does not name. That matches the reference: the L3
        // programs are `runDcg`'s ("DCG to generate L3 programs",
        // `dbo/src/Utils/sdsc_bundle/SchedulerStages.cpp:45-57`), a stage BEFORE
        // `sdscToDataflowIR`, so they do not come out of this walk.
        if self.comp != DfirUnit::Lxlu {
            return Vec::new();
        }
        let own = handles.units.first();
        let replication_one = match Replication::checked(REPLICATION_ONE_SRC) {
            Some(one) => one,
            // Unreachable: 1 is in range by construction. An absent root is the honest answer.
            None => return Vec::new(),
        };
        let core = match self.dsc.coreIdsUsed_.first().copied().and_then(Core::checked) {
            Some(core) => core,
            // A DSC that occupies no core schedules nothing on one.
            None => return Vec::new(),
        };

        // ⭐ THE HANDLERS THE PORTED ENTRIES INDEX BY COMPONENT — this unit's own handle, which the
        // driver has already minted and passed in. `own_lrf`/`pt_xrf` belong to the compute units
        // and this transfer never reaches them.
        let handlers: &'s Handlers = self.arena.alloc(Handlers {
            units: vec![(
                DfirUnit::Lxlu,
                Bound::Unit {
                    handle: own,
                    corelet: None,
                },
            )],
            own_lrf: own,
            pt_xrf: own,
        });

        // ⛔ THE EXTENTS AND THE FORMAT ARE THE ONLY THINGS THE SCHEDULE DOES NOT STATE, and they
        // come from the SuperDSC, not from here: one stick of fp16 is 64 elements
        // (`sysdef` stickSize_), which is what `labeledDs_` records as `wordLength` 2.
        let view_sizes: &'s [ViewSize] = self.arena.alloc_slice_copy(&[ViewSize {
            dim: PrimaryDim::Out,
            size: 64,
        }]);
        let chunks: &'s [deeptools::arch::Elements] =
            self.arena.alloc_slice_copy(&[deeptools::arch::Elements(64)]);

        // ⛔⛔ A TRANSFER WITH NO `vias` EMITS NOTHING AT ALL. Every one of
        // `construct_data_transfer`'s three closures is reached only from a loop over
        // `transfer.vias` (`transfer.rs:5228-5289`): the sends walk it, and `store`/`receive` fire
        // from `if dst.unit == transfer.comp`. An earlier version passed `&[]` and got a bound unit
        // with an EMPTY `affine.for` — the statement was walked and lowered to nothing.
        //
        // ⭐ ONE DESTINATION, THE COMPUTE UNIT — copied from the port's own fixture shape
        // (`transfer.rs:8288-8300`). `dst.unit != comp`, so the store arm is skipped and the SEND
        // loop is what runs.
        let vias: &'s [DstVia<'s>] = self.arena.alloc_slice_clone(&[DstVia {
            unit: Component::Unit(DfirUnit::Sfp),
            storage: Component::SfpLrf,
            vias: &[],
            fusable_parent: None,
        }]);

        // ⛔⛔ AND THE DESTINATION CANNOT BE THIS UNIT ITSELF. Pointing a via back at `comp` with
        // `src_unit == comp` reaches `transfer.rs:5234`, which is a `todo!` guarding the reference's
        // own `DT_CHECK(transfer_->src_.unit_ != comp_ || dst.loc_.storage_ == LXLUSCALEREG)`
        // (`:2786-2787`) — a unit that is its own destination must store into the LX SCALE register,
        // and anything else is a shape the reference asserts against. Measured: it panics with
        // *"comp_ is its own destination and does not store into the LX scale register"*.
        //
        // ⭐⭐ SO AN `lxlu`'s TRANSFER IS A LOAD-AND-SEND, WHICH IS WHAT AN LX LOAD UNIT DOES: it
        // reads the LX and hands the vector to a compute unit. The via names `Sfp`/`SfpLrf`, `store`
        // never fires, and the closure to wire is `send` -> entry 090
        // (`generate_load_and_send_from_data_transfer_node`), which is exactly the shape of the
        // port's own fixture at `transfer.rs:8302-8355`. Entry 090 needs four more description
        // structs — `TransferRead`, `ViewLoops`, `ContiguousTransfer`, `LoadSource` — and that is
        // the next step. Until it is written, `vias` stays empty: the statement is walked, its unit
        // is bound and its loop opened, and NOTHING is emitted inside it.

        let transfer = DataTransfer {
            comp: Component::Unit(DfirUnit::Lxlu),
            // ⭐ ITS OWN UNIT, so nothing leaves on a wire and the data STAYS — which is what makes
            // `store` the closure that fires rather than `send`.
            src_unit: Component::Unit(DfirUnit::Lxlu),
            src: EndFormat::Labeled(DataType::Sen169Fp16, TensorCategory::Regular),
            // ⛔ NO LEADING DESTINATION: this transfer lands once, in the LX, with no via to
            // convert through. `last` is what types the result.
            dsts: DstFormats {
                leading: &[],
                last: EndFormat::Labeled(DataType::Sen169Fp16, TensorCategory::Regular),
            },
            chunks,
            num_chunks: 1,
            vias,
            fusable_src: None,
            core,
            corelet: None,
            // ⛔ MATCHES THE `Uniformize::Disabled` THIS CALLER PASSES `run_translator`; the two
            // disagreeing would ask corelet 0 for blocks a non-uniformized walk never fixes up.
            uniformized: Uniformization::Disabled,
            node: "lxlu_load",
            name: "t0",
            view_sizes,
            replication: replication_one,
        };

        let statement: &'s TransferStatement<'s> = self.arena.alloc(TransferStatement {
            handlers,
            transfer,
            own,
            // One stick, no epilogue: the steady count IS the whole transfer here.
            blocks: self.arena.alloc(|_corelet: Option<Corelet>| {
                vec![(
                    PrimaryDim::Out,
                    StickCounts {
                        steady: 1,
                        epilogue: 1,
                    },
                )]
            }),
            // ⛔ NEITHER END IS A WIRE. This transfer is memory-to-memory on one unit, so the
            // reference's own test panics in exactly these two arms (`transfer.rs:8349-8355`)
            // rather than returning a stand-in.
            // ⭐⭐ ENTRY 090, THE LOAD-AND-SEND — what an LX load unit does: read the LX and hand
            // the vector to the compute. The `TransferRead` is the port's own fixture
            // (`transfer.rs:7156-7192`) with this unit's component and location substituted, and the
            // call is its own at `:7931-7943`.
            send: self.arena.alloc(
                move |vals: &mut Values,
                      ops: &mut Vec<Op>,
                      sticks: &mut ContiguousSticks,
                      to: Val| {
                    let (to_end, _) = DynLink::between(own, to).ends();
                    let read = TransferRead {
                        storage: Component::Unit(DfirUnit::Lx),
                        src: GenericComp::Lxlu,
                        core,
                        corelet: None,
                        comp: GenericComp::Sfp,
                        location: DataLocation::LxluLx,
                        src_location: DataLocation::LxluLx,
                        name: "t0",
                        node: "lxlu_load",
                        view_sizes,
                        elem: ElemType::F16,
                        chunk_sizes: &[],
                        chunk_stride: None,
                        num_chunks: 1,
                        result_ty: Vector {
                            len: 64,
                            elem: ElemType::F16,
                        },
                        dst_result_ty: Vector {
                            len: 64,
                            elem: ElemType::F16,
                        },
                        dst_prec: DataType::Sen169Fp16,
                        precision: DataType::Sen169Fp16,
                        replication: replication_one,
                        rotate: None,
                        latches: &[],
                        to: to_end,
                    };
                    generate_load_and_send_from_data_transfer_node(
                        vals,
                        handlers,
                        &read,
                        &ViewLoops {
                            outer: &[],
                            composite: &[],
                        },
                        sticks,
                        ContiguousTransfer::Absent,
                        |_| None,
                        LoadSource::Zero,
                    )

                },
            ),
            store: self.arena.alloc(
                move |vals: &mut Values, ops: &mut Vec<Op>, storage: Component| {
                    let made = generate_load_and_store_from_data_transfer_node(
                        vals,
                        handlers,
                        &LoadAndStoreTransfer {
                            storage,
                            core,
                            corelet: None,
                            // ⭐⭐ THE PAIR THE GRANULARITY FACTOR IS DERIVED FROM, and the reason
                            // the address below is handed over RAW: entry 025 computes
                            // `scale * 8 / bits` from THIS and the precision
                            // (`SNDSCLowering.cpp:156`, table `sysdef.cpp:531-550`), so scaling
                            // here would apply it twice.
                            location: DataLocation::LxluLx,
                            precision: DataType::Sen169Fp16,
                            name: "t0",
                            node: "lxlu_load",
                            view_sizes,
                            elem: ElemType::F16,
                            outer_loops: &[],
                            // ⛔ ONE CHUNK, NO STRIDE. The time chunking is the schedule's and
                            // arrives with the loop nest; a single-stick transfer walks no time
                            // dimension at all.
                            chunks: UnitTimeChunks {
                                sizes: &[],
                                stride: None,
                                num_strides: 1,
                            },
                            // The wire this load's data leaves on: one stick of fp16 is 64 lanes.
                            result_ty: Vector {
                                len: 64,
                                elem: ElemType::F16,
                            },
                        },
                        LoadAndStoreSource::Zero,
                        // ⛔ THE PLACEMENT, UNSCALED. `Factor::apply` is the port's, and the
                        // address it scales is the `AllocNode`'s own start — zero until the
                        // placements are read out of `scheduleTree_`.
                        // ⛔ THE PLACEMENT SCALED BY THE PORT'S OWN FACTOR, then bound as an
                        // index. `Factor::scale` is entry 025's `int(address * factor)`; the `0`
                        // is the `AllocNode` start address, which is what step 2b reads out of
                        // `scheduleTree_`.
                        |vals, ops, factor| constant_index(vals, ops, factor.scale(0)),
                    );
                    made
                },
            ),
            receive: self.arena.alloc(
                |_: &mut Values,
                 _: &mut Vec<Op>,
                 _: &mut ContiguousSticks,
                 _: usize,
                 _: RecvEnd| {
                    panic!("this LX load is no destination of a wire transfer")
                },
            ),
        });

        vec![Scheduled::Leaf(Emitted::Transfer(statement))]
    }
}

impl<'c> PortDsc<'c> for OneDsc<'c> {
    /// No transfer carries a latch until the TRANSFER statements land.
    type Latch = ();
    /// No transfer carries a stick mask yet.
    type Mask = ();
    /// No buffer switching yet.
    type Switch = ();
    type View = Schedule<'c>;

    fn cores(&self) -> &[Core] {
        &self.cores
    }

    /// ⭐ THE FRONTEND CONSTANT, READ OFF THE `Dsc` — `numCoreletsUsed_`, which scratchy sets from
    /// `ACTIVE_CORELETS`.
    fn num_corelets_used(&self) -> u32 {
        self.dsc.numCoreletsUsed_
    }

    /// ⭐⭐ `Dsc2` IFF IT HAS A COMPUTE, which is the reference's own test: `getTranslatorVersion`
    /// sends a compute-less DSC down its `failure()` path and a DSC1.0 down the trailing `else`
    /// (the port's entry 110). `computeOp_` empty is exactly the compute-less case, and scratchy
    /// never builds a DSC1.0.
    fn kind(&self) -> DscKind {
        if self.dsc.computeOp_.is_empty() {
            DscKind::NoComputeOp
        } else {
            DscKind::Dsc2
        }
    }

    /// ⛔ FOLDS COLLAPSED. `SdscFolds` carries the fold data scratchy writes for dxp, but the
    /// uniformized address construction that reads it is a later stage of this wiring; `false`
    /// brings every component back with one fold, which is what the port's own fixture asserts.
    fn folds_needed(&self, _comp: DfirUnit) -> bool {
        false
    }

    fn view(&self, comp: DfirUnit, _at: Viewed) -> Viewing<'c, Self> {
        Viewing {
            transfers: Vec::new(),
            head_children: Vec::new(),
            roots: Schedule {
                comp,
                dsc: self.dsc,
                arena: self.arena,
            },
        }
    }
}

/// LOWER ONE GROUP'S SUPERDSC TO DATAFLOWIR TEXT.
///
/// ⭐ `None` WHEN THE WALK BOUND NO UNIT — which is every call until [`Schedule::roots`] yields
/// statements. The caller stages nothing in that case.
#[must_use]
pub fn lower_superdsc_to_dataflow_ir<A: Arch>(
    dsc: &SuperDsc,
    name: ProgramName,
    grid: Grid,
) -> Option<String> {
    let mut vals = Values::default();
    // ⭐ OUTLIVES THE WALK AND NOTHING ELSE. Every statement payload the views allocate lives here,
    // and the whole arena is dropped when this function returns — after `print::run` has turned the
    // program into characters, which is the last reader of anything borrowed.
    let arena = bumpalo::Bump::new();
    let one = OneDsc::of(dsc, &arena);
    let ran: Translated<A> = run_translator(
        &mut vals,
        // ⛔ DISABLED UNTIL THE FOLDS ARE READ. `Uniformize::Enabled` is the port's entry 109 — one
        // view per component over every (core, corelet) pair at once — and it is only correct once
        // `folds_needed` answers from `sdscFolds_`.
        Uniformize::Disabled,
        name,
        grid,
        &[NumFolds::ONE],
        core::slice::from_ref(&one),
    );
    match ran {
        Translated::Ran(converted) => converted.program.map(|program| {
            // ⭐ THE KERNEL IS THE GROUP, which `ProgramName` already carries: one group's programs
            // are one kernel, and `print::run` writes the declaration module that names it.
            print::run(&Run {
                kernel: KernelName(program.name.group),
                programs: vec![program],
            })
        }),
        // ⭐ NEITHER IS AN ERROR: they are the two states the reference runs no driver for.
        Translated::Dsc1 | Translated::NoComputeOp => None,
    }
}


/// LOWER ONE SUPERDSC PROGRAM, NAMED BY ITS GROUP — the call the build makes.
///
/// ⭐ IT NAMES NO `deeptools` TYPE IN ITS SIGNATURE, AND THAT IS THE POINT. `scratchy-forward-
/// compiler-macro` does not depend on `deeptools`, so a call site there cannot spell `ProgramName`,
/// `GroupId` or `Grid`. Keeping the wrapper here also keeps the `Dd2` choice and the single-element
/// grid in one place instead of at every caller.
#[must_use]
pub fn lower_group(dsc: &SuperDsc, group: u32) -> Option<String> {
    lower_superdsc_to_dataflow_ir::<deeptools::arch::Dd2>(
        dsc,
        ProgramName {
            group: deeptools::islands::dataflow_ir::GroupId(group),
            index: deeptools::islands::dataflow_ir::OpIndex(0),
            // ⛔ THE OP-FUNC IS THE SCHEDULE'S QUESTION AND IT IS STILL A PLACEHOLDER. It belongs to
            // the `computeOp_[i].opFuncName` of the `Dsc` being lowered; naming `Mul` here is a
            // stand-in that only affects the printed program NAME, not what is emitted, and it goes
            // when `Schedule::roots` selects its DDL program by op-func.
            func: deeptools::generated::OpFunc::Mul,
        },
        Grid::single(),
    )
}
