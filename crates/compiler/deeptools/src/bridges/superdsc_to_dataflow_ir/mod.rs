//! BRIDGE 1 — scratchy's SuperDSC meeting DataflowIR's vocabulary.
//!
//! ```text
//! SuperDSC (SdscOp / SdscFolds)  ──here──►  islands::dataflow_ir::Run<A>  ──►  dbo-opt
//! ```
//!
//! ⭐ THE DDL TEMPLATE IS THE SCHEDULE (units, transfers, loop nest, computes) — the same for
//! every op of its op-func, and it knows no extents. THE NODE IS THE SHAPE. Two sides, both
//! needed; a bridge that derives one from the other has invented it.
//!
//! ⛔ THIS REPLACES [`super::subtile_to_dataflow_ir`], which stays in the tree until the
//! acceptance build is green on BOTH granite-3.1-2b and -8b. Switching the call site is a
//! separate, deliberate step — do not edit that module from here.
//!
//! Ported from the authority tree `/Users/nickm/git/deeptools-src` at revision `a0d29abbed`;
//! 110 functions, each carrying a `/// Replaces: eNNN_name` anchor citing its original.

/// THE TRANSFER STATEMENTS — load, store and send, and how each is walked over the AGEN time axis.
pub mod transfer;

/// THE VECTOR CHAINS — mac, binary and unary computes, and the precision they run at.
pub mod compute;

/// THE LOOP NEST AND THE CONDITIONALS — DSC loops, blocks and conds becoming the scf/affine nest.
pub mod control_flow;

/// THE PER-COMPONENT LOWERING — unit, corelet and core identity, and the component handler.
pub mod dsc_lowering;

/// THE CONVERSION DRIVER — runTranslator, convertV3, convertV4, and the module scaffolding they build.
pub mod driver;

/// THE SHARED SHAPE AND FOLD HELPERS every lowering below stands on.
pub mod utils;

/// DATAFLOWIR CONSTRUCTION — building the island's types and attributes.
pub mod construction;

/// THE DDL COMPILER'S DERIVATIONS — and the reason this campaign exists.
pub mod shape_constraints;

/// THE SYNC STATEMENTS.
pub mod sync;

/// THE STICK MASK — the SAMV set-transfer-mask-state op.
pub mod stick_mask;
