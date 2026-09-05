//! ONE MODULE PER IR — its types, its invariants, its printer, and nothing else.
//!
//! ⛔ AN ISLAND NAMES NO PRODUCER AND NO CONSUMER. The moment one does, its invariants start
//! being whatever the current emitter happens to satisfy, and checking it against the reference
//! files stops meaning anything.
