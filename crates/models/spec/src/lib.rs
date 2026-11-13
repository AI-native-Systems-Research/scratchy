//! Speculative-decoding build scope. This crate carries no code: its features (`mtp`) are reached
//! as `spec/<drafter>` from scratchy-cli and forward to scratchy-models, whose scope the targets'
//! models crates compile the drafters of. See `Cargo.toml`.
