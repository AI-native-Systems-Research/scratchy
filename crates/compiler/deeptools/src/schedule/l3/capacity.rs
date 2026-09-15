//! HOW MANY BYTES A BUFFER HOLDS — `DesignSpaceConfig::getBufferCapacityForNode`
//! (`dsc/dsc2.cpp:3977`) and the closure beneath it.
//!
//! This is the one `dsc/` call the L3 scheduler stops on: `try_alloc_l3`
//! ([`super::dl_ops`]) asks `L3Placement::buffer_capacity_even_sticks` for the capacity of the
//! allocation it is about to commit, and every program reaches that question. ⛔ **A FABRICATED
//! CAPACITY COMMITS A FABRICATED PLACEMENT** — it compiles, it bakes, and the card reads memory
//! nothing filled. A stop here is faithful; a plausible number is not.
//!
//! ⭐ THE CLOSURE LIVES IN ITS OWN FILE SO NONE OF ITS THREE CALLERS OWNS IT. The same reference
//! call stands behind three carriers with three different signatures —
//! `L3Placement::buffer_capacity_even_sticks` (`stages/carriers.rs`),
//! `v1::Placement::buffer_capacity` (`stages/ddc_reads.rs`) and `conv::DdlSizes::buffer_capacity`
//! (`stages/ddc_sites.rs`) — and answering it inside any one of them would leave the other two to
//! reinvent it. `DesignSpaceConfig::lx_chunk_capacity` ([`super::dsc`]) is this same call already
//! made for the CHUNK stage at one fixed site: reusing it answers a different question with the
//! same number.
//!
//! ⛔ THE HOME IS NOT `impl DesignSpaceConfig`. That type carries no allocate nodes and no schedule
//! tree; ours live in a separate `v1::AllocArena` (`BTreeMap<AllocId, AllocateNode>`) handed to
//! `try_alloc_l3` as its own parameter. The units here take the node, the labelled DS and the
//! ancestor loop chain as arguments — the loop chain in particular, because the reference's
//! `getOwnerLoop()`/`getPrev()` walk is over a chain every caller already holds.
//!
//! ⛔ THE AUTHORITY IS `/Users/nickm/git/deeptools-src/<file>:<line>`, NEVER A COMMENT IN THIS
//! CRATE ABOUT IT. Eight defects in this stack were found by reading IBM's source instead of our
//! own prose, and every one was self-documented as deliberate; two were citations off by 410 and 48
//! lines. Note one drift you will meet: `stages/carriers.rs` cites `dsc/dsc2.cpp:3755` for
//! `getBufferCapacityForNodePerDimCustomLocation`, whose definition begins at **3754**.
//!
//! Five units land here, in dependency order — see `crustify-capacity/UNITS.tsv` for each one's
//! `rust_home` and callees, and `crustify-capacity/AGENT-BRIEF.md` §5 for the arm-by-arm fixture
//! measurement that says which arms to port and which to defer with a citation:
//!
//! | unit | authority | L |
//! |---|---|---|
//! | `e015_getSizeDataStageForNode` | `dsc/dsc2.cpp:3616-3752` | 137 |
//! | `e016_getSizeDataStageForNode_2arg` | `dsc/dsc2.cpp:3611-3614` | 4 |
//! | `e017_getBufferCapacityForNodePerDimCustomLocation` | `dsc/dsc2.cpp:3754-3964` | 211 |
//! | `e018_getBufferCapacityForNodePerDim` | `dsc/dsc2.cpp:3966-3975` | 10 |
//! | `e019_getBufferCapacityForNode` | `dsc/dsc2.cpp:3977-4005` | 29 |
//!
//! ⛔ INTENTIONALLY EMPTY. No signature shells and no `todo!` bodies: `todo!` is capped in this
//! crate and ratchets DOWN only, and a signature is something a porter derives from the C++ — not
//! something to inherit from whoever created the file.
