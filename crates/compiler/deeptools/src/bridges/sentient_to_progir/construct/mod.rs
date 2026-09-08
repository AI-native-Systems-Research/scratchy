//! `ConstructProgIRHelper.cpp` — 46 units, one submodule per family.

/// THE COMPUTE INSTRUCTIONS — FMA, binary, unary and ternary, and the operand plumbing that
pub mod compute;

/// THE MASK, SPLAT AND SAMV INSTRUCTIONS — set-dest-mask, set-dest, splat, splat-pad, and the
pub mod mask_and_splat;

/// THE OPAQUE-TEMPLATE INSTRUCTION — `ConstructOpaqueInstr` and the string trim it uses.
pub mod opaque;

/// WHAT MUST BE IN A REGISTER BEFORE ANYTHING READS IT — the reg-init accumulation and the
pub mod reg_init;

/// THE SCALAR INSTRUCTIONS — the jumps, the compares, the adds and subs, and the sync.
pub mod scalar;

/// THE TRANSFER INSTRUCTIONS — the L3 and non-L3 loads and stores, the burst normalisation and
pub mod transfer;
