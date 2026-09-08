//! `LowerSentientHelper.cpp` — 34 units, one submodule per family.

/// THE COMPUTE OPERATION HANDLERS — mac, binary, unary, ternary, add, sub, splat, opaque, samv
pub mod compute;

/// THE CONTROL FLOW AND THE UNIFORM REGIONS — for, yield, return, sync, nop, set-send-dest, and
pub mod control;

/// THE LABELS, THE CODE GRAPH AND THE REGISTER IMMEDIATES — AddToLabelsMap,
pub mod labels_and_regs;

/// THE TRANSFER OPERATION HANDLERS — load_and_send, receive_and_store, load_and_store, the
pub mod transfer;
