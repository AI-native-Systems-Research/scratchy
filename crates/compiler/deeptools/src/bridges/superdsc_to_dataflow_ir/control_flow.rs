//! THE LOOP NEST AND THE CONDITIONALS — DSC loops, blocks and conds becoming the scf/affine nest.
//!
//! 14 units. Every citation resolves against the authority tree
//! `/Users/nickm/git/deeptools-src` at revision `a0d29abbed`.
//!
//! | unit | level | LoC | authority |
//! |---|---|---|---|
//! | `e017_getCmpIPredicate_dup` | 0 | 25 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNControlFlowLowering.cpp:21` |
//! | `e018_getMLIRLoopFromSNLoopNode` | 0 | 16 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNControlFlowLowering.cpp:49` |
//! | `e019_getParentLoop` | 0 | 26 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNControlFlowLowering.cpp:381` |
//! | `e020_propagateBufferSwitchLoopsToRootRecursively` | 0 | 50 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNControlFlowLowering.cpp:519` |
//! | `e021_resetIterArguments` | 0 | 15 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNControlFlowLowering.cpp:738` |
//! | `e054_constructConditionalOperation` | 1 | 211 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNControlFlowLowering.cpp:66` |
//! | `e055_constructConditionalsForSAMV` | 1 | 91 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNControlFlowLowering.cpp:284` |
//! | `e056_getBufferingOrStreamingMode` | 1 | 54 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNControlFlowLowering.cpp:408` |
//! | `e057_propagateBufferSwitchLoopsToRoot` | 1 | 15 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNControlFlowLowering.cpp:575` |
//! | `e058_constructLoopForADim` | 1 | 101 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNControlFlowLowering.cpp:754` |
//! | `e074_getBlockingOrStreamingBufferLoopLocations` | 2 | 46 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNControlFlowLowering.cpp:467` |
//! | `e075_constructLoopIterArgs` | 2 | 91 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNControlFlowLowering.cpp:595` |
//! | `e088_constructLoopsRecursive` | 3 | 324 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNControlFlowLowering.cpp:862` |
//! | `e096_constructLoops` | 4 | 27 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNControlFlowLowering.cpp:1191` |

// ⛔ ONE `crustify:todo:` PER SCHEDULED UNIT. Replace each with the ported function
// carrying `/// Replaces: eNNN_name`. A surviving TODO is open work.

use crate::islands::dataflow_ir::dialects::arith::CmpIPredicate;
use crate::islands::dataflow_ir::dialects::{Op as DfirOp, Val, affine, scf};

/// A DIMENSION THE SCHEDULE IS WRITTEN OVER — `PrimaryDimTypes` (`dsc/dims.h:34-48`).
///
/// ⛔ TWELVE, NOT THIRTEEN: `PrimaryDimTypesCount` is the reference's count-and-unset sentinel, the
/// value `PrimaryDimAndKind` defaults to (`dims.h:77`). Absence is an [`Option`] here, so a dim that
/// was never set cannot be compared equal to a real one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PrimaryDim {
    /// `IN` — input features.
    In,
    /// `OUT` — output features.
    Out,
    /// `IJ` — the output image, rows and cols together.
    Ij,
    /// `MB` — the minibatch.
    Mb,
    /// `X` — a repeat dim that adds no reuse.
    X,
    /// `Y` — a kernel reuse dim.
    Y,
    /// `KIJ` — the kernel, rows and cols together.
    Kij,
    /// `I` — output image rows.
    I,
    /// `J` — output image cols.
    J,
    /// `KI` — kernel rows.
    Ki,
    /// `KJ` — kernel cols.
    Kj,
    /// `X1` — a second repeat dim.
    X1,
}

/// WHICH COMPARISON A SCHEDULE CONDITION MAKES — `CondOp` (`dsc/dscdefn.h:95-107`).
///
/// ⭐ ELEVEN, AND ONLY SIX OF THEM COMPARE. The other five are how a condition says *always*,
/// *never*, *flip*, *do not evaluate me* and *whatever the default is* — see
/// [`CondOp::cmp_predicate`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CondOp {
    /// `==`.
    Eq,
    /// `!=`.
    Ne,
    /// `<`.
    Lt,
    /// `<=`.
    Le,
    /// `>`.
    Gt,
    /// `>=`.
    Ge,
    /// `TOGGLE` — flip on every visit.
    Toggle,
    /// `ALWAYS`.
    Always,
    /// `NEVER`.
    Never,
    /// `CONST` — "used when the condition should not be evaluated" (`dscdefn.h:105`).
    Const,
    /// `DEFAULT`.
    Default,
}

impl CondOp {
    /// Replaces: e017_getCmpIPredicate_dup
    ///
    /// THE `arith.cmpi` PREDICATE A CONDITION LOWERS TO, or [`None`] for a condition that is not a
    /// comparison at all — the reference's `default: return LogicalResult::failure()`
    /// (`SNControlFlowLowering.cpp:41-42`), whose caller then emits no compare.
    ///
    /// ⛔ SIGNED, EVERY ONE: `LT` is `slt` and not `ult` (`:30-31`), which is the same six
    /// [`CmpIPredicate`] holds and the reason it holds six rather than MLIR's ten.
    #[must_use]
    pub const fn cmp_predicate(self) -> Option<CmpIPredicate> {
        match self {
            CondOp::Eq => Some(CmpIPredicate::Eq),
            CondOp::Ne => Some(CmpIPredicate::Ne),
            CondOp::Lt => Some(CmpIPredicate::Slt),
            CondOp::Le => Some(CmpIPredicate::Sle),
            CondOp::Gt => Some(CmpIPredicate::Sgt),
            CondOp::Ge => Some(CmpIPredicate::Sge),
            CondOp::Toggle | CondOp::Always | CondOp::Never | CondOp::Const | CondOp::Default => {
                None
            }
        }
    }
}

/// Replaces: e018_getMLIRLoopFromSNLoopNode
///
/// WHICH EMITTED LOOP CARRIES A NODE'S DIM, or [`None`] for a dim this node does not walk.
///
/// ⛔⛔ THE LOOPS ARE INDEXED IN REVERSE — dim `i` of `dims_` is loop `len - i - 1`
/// (`SNControlFlowLowering.cpp:56-58`) — and `len` is the EMITTED count, not the dim count, so the
/// reference's unchecked `record->second[size - i - 1]` reads out of bounds on a node whose dims
/// outnumber its loops. That is [`None`] here. The map lookup is mechanism and stays the caller's.
pub fn mlir_loop_from_sn_loop_node<'l, L>(
    dims: &[PrimaryDim],
    loops: &'l [L],
    dim: PrimaryDim,
) -> Option<&'l L> {
    let index = dims.iter().position(|walked| *walked == dim)?;
    loops.get(loops.len().checked_sub(index + 1)?)
}

/// A DIM OF ONE NODE AND THE SLICE RANGE ONE COMPONENT SEES OVER IT.
///
/// ⭐ THE TWO VALUES ARE `ss_` AND `el_` READ THROUGH THE COMPONENT — the start-slice and end-slice
/// data stages of `dsc_->dataStageParam_.at(numId_)`, each asked
/// `dataStageDimToVal_compView_st(dim, comp_)` (`SNControlFlowLowering.cpp:388-393`). Reaching them
/// is mechanism; that they are asked PER COMPONENT is not, which is why they are carried and not
/// recomputed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DimSlice {
    /// Which dim.
    pub dim: PrimaryDim,
    /// `ss_`'s value for it — the first slice.
    pub start: i32,
    /// `el_`'s value for it — the last slice.
    pub end: i32,
}

/// ONE ENCLOSING LOOP, AS [`parent_loop`] READS IT.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ancestor<'a, L> {
    /// `parent->dims_`, in the order the node declares them.
    pub dims: &'a [DimSlice],
    /// The loops emitted for it, indexed exactly as [`mlir_loop_from_sn_loop_node`] indexes them.
    pub loops: &'a [L],
}

/// Replaces: e019_getParentLoop
///
/// THE NEAREST ENCLOSING LOOP WALKING `dim` AS A SINGLE SLICE, or [`None`] when no ancestor does.
///
/// ⛔ SINGLE SLICE IS THE WHOLE TEST: an ancestor qualifies only where its start and end slice for
/// `dim` are EQUAL (`SNControlFlowLowering.cpp:394`), so an ancestor that walks several slices of the
/// dim is passed over and the walk continues outward.
///
/// The parent chain is mechanism — `getPrev()`'s null root and the `getOwnerLoop()` walk — so
/// `ancestors` IS that chain, NEAREST FIRST, and empty for a node with no enclosing loop.
pub fn parent_loop<'l, L>(ancestors: &'l [Ancestor<'l, L>], dim: PrimaryDim) -> Option<&'l L> {
    for parent in ancestors {
        for (index, slice) in parent.dims.iter().enumerate() {
            if slice.dim == dim && slice.start == slice.end {
                return parent.loops.get(parent.loops.len().checked_sub(index + 1)?);
            }
        }
    }
    None
}

/// WHICH KIND OF SCHEDULE NODE — `ScheduleNode::NodeType` (`dsc/dsc2.h:445-456`), as far as the
/// buffer-switch propagation tells them apart.
///
/// ⭐ THE THREE THAT CARRY CHILDREN ARE EXACTLY `isBlockNode()`'s three (`dsc2.h:481-483`); every
/// other node type is a leaf the walk does not descend into.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeKind {
    /// `LOOP` — and the only kind that contributes switches of its own.
    Loop,
    /// `CONDITION`.
    Condition,
    /// `BLOCK`.
    Block,
    /// A transfer, compute, sync, allocate or stick-mask node — no children, no switches.
    Leaf,
}

/// A SCHEDULE NODE AS THE BUFFER-SWITCH PROPAGATION READS AND WRITES IT.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SwitchNode<T> {
    /// Which kind it is.
    pub kind: NodeKind,
    /// `getNextView(comp_, corelet_id_, core_id_)` — the children RELEVANT to this component,
    /// corelet and core. ⭐ ALREADY FILTERED: which children a component sees is mechanism, and the
    /// reference calls the view rather than the raw child list (`SNControlFlowLowering.cpp:524`).
    pub children: Vec<SwitchNode<T>>,
    /// `dsc_loops_to_buffers_switch_map_[loop]` — the transfers whose buffers THIS loop switches.
    /// Read only for a [`NodeKind::Loop`], because that map is keyed by `LoopNode*` (`:558-560`).
    pub own: Vec<T>,
    /// OUTPUT: `(*dsc_all_parent_loops_to_buffers_switch_map_)[node]` — every switching transfer at
    /// or below this node. ⭐ THE REAL RESULT: entry 057 discards the vector it passes in and keeps
    /// this map (`:575-589`).
    pub all: Vec<T>,
}

/// Replaces: e020_propagateBufferSwitchLoopsToRootRecursively
///
/// EVERY BUFFER-SWITCHING TRANSFER AT OR BELOW A NODE, WRITTEN ONTO THAT NODE and APPENDED to
/// `node_contrib` — appended, not assigned, which is how a parent accumulates its children's.
///
/// ⭐ THE THREE `dynamic_cast` ARMS ARE ONE ARM, and that is the reference's own note: "the code for
/// all the bodies are same" (`SNControlFlowLowering.cpp:526-529`). A child that is none of the three
/// contributes nothing and is not descended into.
///
/// ⛔ A LOOP'S OWN SWITCHES COME **AFTER** ITS CHILDREN'S, in both outputs (`:557-567`); the
/// reference's `LogicalResult` has no failing path to carry.
pub fn propagate_buffer_switch_loops_to_root_recursively<T: Clone>(
    node: &mut SwitchNode<T>,
    node_contrib: &mut Vec<T>,
) {
    let mut below: Vec<T> = Vec::new();
    for child in &mut node.children {
        let mut child_contrib: Vec<T> = Vec::new();
        match child.kind {
            NodeKind::Loop | NodeKind::Condition | NodeKind::Block => {
                propagate_buffer_switch_loops_to_root_recursively(child, &mut child_contrib);
            }
            NodeKind::Leaf => {}
        }
        below.append(&mut child_contrib);
    }
    if node.kind == NodeKind::Loop {
        below.extend(node.own.iter().cloned());
    }
    node_contrib.extend(below.iter().cloned());
    node.all.extend(below);
}

/// Replaces: e021_resetIterArguments
///
/// THE REGION ITER ARGUMENTS OF A LOOP — `getRegionIterArgs()` of whichever loop op it is, built
/// fresh, which is the reference's `iter_args.clear()` and then its push loop
/// (`SNControlFlowLowering.cpp:739-750`).
///
/// ⛔ [`None`] IS THE `llvm_unreachable("Unknown loop")`, moved where a caller cannot observe it:
/// the two `dyn_cast`s are the whole dispatch and an op that is neither loop has no iter arguments
/// to reset. Same shape as `tf_loop_unroll_for_shuffle_op::Loop::of`.
#[must_use]
pub fn reset_iter_arguments(loop_op: &DfirOp) -> Option<Vec<Val>> {
    match loop_op {
        DfirOp::Affine(affine::Op::For { carried, .. })
        | DfirOp::Scf(scf::Op::For { carried, .. }) => {
            Some(carried.iter().map(|value| value.arg).collect())
        }
        DfirOp::Affine(_)
        | DfirOp::Scf(_)
        | DfirOp::Arith(_)
        | DfirOp::Dataflow(_)
        | DfirOp::Agen(_)
        | DfirOp::Vector(_)
        | DfirOp::VectorChain(_)
        | DfirOp::Uniform(_)
        | DfirOp::Symbol(_) => None,
    }
}

// crustify:todo: e054_constructConditionalOperation
// crustify:todo: e055_constructConditionalsForSAMV
// crustify:todo: e056_getBufferingOrStreamingMode
// crustify:todo: e057_propagateBufferSwitchLoopsToRoot
// crustify:todo: e058_constructLoopForADim
// crustify:todo: e074_getBlockingOrStreamingBufferLoopLocations
// crustify:todo: e075_constructLoopIterArgs
// crustify:todo: e088_constructLoopsRecursive
// crustify:todo: e096_constructLoops

#[cfg(test)]
mod unit_tests {
    use super::{
        Ancestor, CondOp, DimSlice, NodeKind, PrimaryDim, SwitchNode, mlir_loop_from_sn_loop_node,
        parent_loop, propagate_buffer_switch_loops_to_root_recursively, reset_iter_arguments,
    };
    use crate::islands::dataflow_ir::dialects::arith::CmpIPredicate;
    use crate::islands::dataflow_ir::dialects::{Op as DfirOp, Val, affine, scf};

    /// ⛔⛔ SIGNED, AND THE FIVE NON-COMPARISONS ARE NOT PREDICATES AT ALL.
    ///
    /// A condition lowered with `ult` where `slt` belongs compares two negative offsets the wrong
    /// way round and still parses, so the six are checked by name; and `ALWAYS`/`NEVER`/`TOGGLE`/
    /// `CONST`/`DEFAULT` reaching a predicate at all would emit an `arith.cmpi` for a condition the
    /// reference refuses to build one for (`SNControlFlowLowering.cpp:41-42`).
    #[test]
    fn six_signed_predicates_and_five_conditions_that_are_not_comparisons() {
        assert_eq!(
            [
                CondOp::Eq.cmp_predicate(),
                CondOp::Ne.cmp_predicate(),
                CondOp::Lt.cmp_predicate(),
                CondOp::Le.cmp_predicate(),
                CondOp::Gt.cmp_predicate(),
                CondOp::Ge.cmp_predicate(),
            ],
            [
                Some(CmpIPredicate::Eq),
                Some(CmpIPredicate::Ne),
                Some(CmpIPredicate::Slt),
                Some(CmpIPredicate::Sle),
                Some(CmpIPredicate::Sgt),
                Some(CmpIPredicate::Sge),
            ],
        );
        for not_a_comparison in [
            CondOp::Toggle,
            CondOp::Always,
            CondOp::Never,
            CondOp::Const,
            CondOp::Default,
        ] {
            assert_eq!(not_a_comparison.cmp_predicate(), None);
        }
    }

    /// ⛔⛔ THE FIRST DIM IS THE **LAST** LOOP, and a node walking one dim cannot show it.
    ///
    /// `dims_` is declared outermost first while the loop vector is appended to as the nest is
    /// built, so `IN`, the outermost dim, is the loop at index `len - 1`
    /// (`SNControlFlowLowering.cpp:56-58`). Read forward, every dim of a three-deep node but the
    /// middle one pairs with the wrong loop — which is a legal program addressing the wrong axis.
    #[test]
    fn a_dim_reads_the_loop_at_its_mirrored_index() {
        let dims = [PrimaryDim::In, PrimaryDim::Out, PrimaryDim::Ij];
        let loops = ["innermost", "middle", "outermost"];

        assert_eq!(
            [
                mlir_loop_from_sn_loop_node(&dims, &loops, PrimaryDim::In),
                mlir_loop_from_sn_loop_node(&dims, &loops, PrimaryDim::Out),
                mlir_loop_from_sn_loop_node(&dims, &loops, PrimaryDim::Ij),
                mlir_loop_from_sn_loop_node(&dims, &loops, PrimaryDim::Mb),
            ],
            [
                Some(&"outermost"),
                Some(&"middle"),
                Some(&"innermost"),
                None,
            ],
        );
        // ⛔ AND A DIM DEEPER THAN THE LOOPS EMITTED IS THE REFERENCE'S OUT-OF-BOUNDS READ, not a
        // wrong loop: `loops.size() - index - 1` underflows `size_t` there and indexes the vector
        // with it, while the outermost dim still reads loop 0 in both.
        assert_eq!(
            [
                mlir_loop_from_sn_loop_node(&dims, &loops[..1], PrimaryDim::In),
                mlir_loop_from_sn_loop_node(&dims, &loops[..1], PrimaryDim::Ij),
            ],
            [Some(&"innermost"), None],
        );
    }

    /// ⛔⛔ AN ANCESTOR WALKING SEVERAL SLICES OF THE DIM IS PASSED OVER, NOT ANSWERED WITH.
    ///
    /// The test is `ss_ == el_` for the dim on this component (`SNControlFlowLowering.cpp:394`): the
    /// nearest ancestor here declares `MB` across three slices and must be skipped in favour of the
    /// outer one that pins it, and the loop returned is that outer node's mirrored index. Answering
    /// with the nearest match would hoist a value to a loop that changes it.
    #[test]
    fn the_nearest_single_slice_ancestor_wins_and_a_multi_slice_one_is_skipped() {
        let near_dims = [DimSlice {
            dim: PrimaryDim::Mb,
            start: 0,
            end: 2,
        }];
        let far_dims = [
            DimSlice {
                dim: PrimaryDim::Mb,
                start: 4,
                end: 4,
            },
            DimSlice {
                dim: PrimaryDim::In,
                start: 0,
                end: 7,
            },
        ];
        let near_loops = ["near-mb"];
        let far_loops = ["far-in", "far-mb"];
        let ancestors = [
            Ancestor {
                dims: &near_dims,
                loops: &near_loops,
            },
            Ancestor {
                dims: &far_dims,
                loops: &far_loops,
            },
        ];

        assert_eq!(parent_loop(&ancestors, PrimaryDim::Mb), Some(&"far-mb"));
        assert_eq!(parent_loop(&ancestors, PrimaryDim::In), None, "0 != 7");
        assert_eq!(parent_loop::<u32>(&[], PrimaryDim::Mb), None, "no parent");
    }

    /// ⛔⛔ A LOOP'S OWN SWITCHES COME AFTER ITS CHILDREN'S, AND BOTH OUTPUTS SEE THE SAME ORDER.
    ///
    /// The reference pushes each child's contribution into the caller's vector AND the node's map
    /// entry, then its own (`SNControlFlowLowering.cpp:551-567`), so a node's entry reads
    /// inner-to-outer. It is also the whole point that the map is written at EVERY level: the inner
    /// loop's own entry holds only what is at or below it, and reversing the two appends would make a
    /// parent switch a buffer before the child that fills it.
    #[test]
    fn a_loops_own_switches_follow_its_childrens_at_every_level() {
        let mut root = SwitchNode {
            kind: NodeKind::Loop,
            children: vec![
                SwitchNode {
                    kind: NodeKind::Block,
                    children: vec![SwitchNode {
                        kind: NodeKind::Loop,
                        children: Vec::new(),
                        own: vec!["inner"],
                        all: Vec::new(),
                    }],
                    own: vec!["a block is never asked for its own"],
                    all: Vec::new(),
                },
                SwitchNode {
                    kind: NodeKind::Leaf,
                    children: vec![SwitchNode {
                        kind: NodeKind::Loop,
                        children: Vec::new(),
                        own: vec!["below a leaf"],
                        all: Vec::new(),
                    }],
                    own: vec!["nor is a leaf"],
                    all: Vec::new(),
                },
            ],
            own: vec!["root"],
            all: Vec::new(),
        };
        let mut node_contrib = vec!["what the caller already had"];

        propagate_buffer_switch_loops_to_root_recursively(&mut root, &mut node_contrib);

        assert_eq!(
            node_contrib,
            vec!["what the caller already had", "inner", "root"],
            "appended, and the loop's own last",
        );
        assert_eq!(root.all, vec!["inner", "root"]);
        assert_eq!(
            root.children[0].all,
            vec!["inner"],
            "the block sees only below"
        );
        assert_eq!(root.children[0].children[0].all, vec!["inner"]);
        assert_eq!(
            root.children[1].all,
            Vec::<&str>::new(),
            "a leaf is not descended into",
        );
    }

    /// ⭐ BOTH LOOP OPS ANSWER WITH THEIR REGION ARGUMENTS, AND IT IS THE `arg` OF THE THREE.
    ///
    /// `getRegionIterArgs()` is the name the BODY reads the carried value through, not the init it
    /// starts from nor the result the loop binds — handing back an init would build a body that reads
    /// a value defined outside the region it is verified in. An op that is neither loop has no iter
    /// arguments, which is where the reference's `llvm_unreachable` went.
    #[test]
    fn both_loops_reset_to_their_region_arguments() {
        let carried = vec![
            affine::Carried {
                init: Val(1),
                arg: Val(2),
                result: Val(3),
            },
            affine::Carried {
                init: Val(4),
                arg: Val(5),
                result: Val(6),
            },
        ];
        let affine_for = DfirOp::Affine(affine::Op::For {
            iv: Val(0),
            lo: affine::Bound::Const(0),
            hi: affine::Bound::Const(8),
            carried: carried.clone(),
            body: Vec::new(),
            dbg_name: None,
        });
        let scf_for = DfirOp::Scf(scf::Op::For {
            iv: Val(0),
            lo: Val(10),
            hi: Val(11),
            step: Val(12),
            carried,
            body: Vec::new(),
            dbg_name: None,
        });
        let not_a_loop = DfirOp::Scf(scf::Op::Yield {
            operands: vec![Val(2)],
        });

        assert_eq!(
            reset_iter_arguments(&affine_for),
            Some(vec![Val(2), Val(5)])
        );
        assert_eq!(reset_iter_arguments(&scf_for), Some(vec![Val(2), Val(5)]));
        assert_eq!(reset_iter_arguments(&not_a_loop), None);
    }
}
