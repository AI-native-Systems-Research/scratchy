// SPDX-License-Identifier: Apache-2.0
//! ⭐⭐⭐ A BUNDLE'S NAME IS ITS CONTENT — SHA-256 over the layout it ships, then each launch group's
//! shifts and compiled input, in launch order.
//!
//! The name is a bundle's identity everywhere downstream: a sibling names it, the bake keys its
//! registration on it, and the runtime looks the bundle up by it. So two bundles may share a name
//! exactly when they are the same bundle, and nothing else may decide that.
//!
//! ⛔ IT USED TO BE A HASH OF THE LOWERING'S INPUT, and that was a defect. The emitter hashed the ops
//! BEFORE the expansion (name, trip count, KTIR, bindings) and returned early on a name it had already
//! registered. The expansion also reads the bundle's layout (every descriptor's addresses, the LX plan
//! that picks divisions) and its row kind (`BundleAttnParams::rows_are_requests`, which picks rope's
//! row frame), and the hash covered neither. So a bundle with equal ops under another layout or row
//! kind took the first one's name and was handed the first one's programs. One name carried 222
//! descriptors in one build and 235 in another.
//!
//! Hashed AFTER the expansion, the name covers everything the expansion read without listing any of
//! it: a fact that changes what the bundle runs changes the bytes, and a fact that does not, does not.
//! There is no second statement of the bundle's inputs here to drift from the lowering.
//!
//! ⭐ WHAT IS HASHED IS WHAT SHIPS, READ THROUGH THE ONE WALK THE MACRO EMITS FROM
//! (`bundle_code::facts`). The layout and each group's shifts are walked field by field into a
//! [`DigestSink`], so the name covers every field the binary carries, and a field added to either
//! is in the name the moment it is in the binary. A group's programs enter as the compiler's input:
//! on a card that is dxp's input, digested as the dxp store keys it ([`crate::dxp_cache::GroupDigest`]),
//! and on the emulator, where the KTIR itself runs, the KTIR programs and their bindings.
//!
//! ⚠️ ONE FIELD MOVES AFTER THE NAME IS TAKEN: a re-rolled body's `segment_bytes`, which
//! `cover_ladder_allocation` widens to the high-water mark of every sibling its session launches.
//! That is the session's allocation rather than the bundle's content (a body addresses nothing past
//! its own extents, and over-allocating a segment is inert for it), so the name is taken over the
//! extents the bundle itself declares, and a name two sessions share carries the wider cover of both.

use sha2::{Digest, Sha256};

use crate::bundle_code::collective::{Fragment, Rank, Tp, WindowCut};
use crate::bundle_code::facts::{FactSink, Facts, ListShape, Path};
use crate::bundle_code::{BundleLayout, KvShifts, LaunchGroup, PlaceId};

/// What one launch group compiles from.
pub enum GroupContent<'a> {
    /// A card's group: the SHA-256 of dxp's input (every `sdsc_*.json` and `bundle.mlir`, names
    /// included, and the per-group compile setting) — the dxp store's own key.
    DxpInput([u8; 32]),
    /// An emulator's group: the KTIR programs it runs, and what each binds.
    Programs(&'a [crate::bundle_code::LaunchProgram<'a>]),
}

/// A bundle's name: its layout, then each group's shifts and content, in launch order.
///
/// `layout` is `None` for a bundle emitted with no memory plan, which is named as having none —
/// never as a defaulted plan. The `Option` goes with the residency stack, which makes the plan
/// required.
pub fn name<'a, T: Tp>(
    layout: Option<&BundleLayout<'_>>,
    groups: impl IntoIterator<Item = (&'a KvShifts<T>, GroupContent<'a>)>,
) -> String {
    let mut d = DigestSink(Sha256::new());
    match layout {
        Some(l) => {
            d.tag(b"layout");
            l.visit(&mut d);
        }
        None => d.tag(b"no layout"),
    }
    for (i, (kv, content)) in groups.into_iter().enumerate() {
        d.tag(b"group");
        d.word(i as u64);
        kv.visit(&mut d);
        match content {
            GroupContent::DxpInput(key) => {
                d.tag(b"dxp");
                d.0.update(key);
            }
            GroupContent::Programs(programs) => {
                d.tag(b"ktir");
                d.word(programs.len() as u64);
                for p in programs {
                    program(&mut d, p);
                }
            }
        }
    }
    let h: [u8; 32] = d.0.finalize().into();
    h[..8].iter().map(|b| format!("{b:02x}")).collect()
}

/// The name of an emulator bundle, whose groups are complete launches.
pub fn of_groups<T: Tp>(
    layout: Option<&BundleLayout<'_>>,
    groups: &[LaunchGroup<'_, T>],
) -> String {
    name(
        layout,
        groups
            .iter()
            .map(|g| (&g.kv, GroupContent::Programs(&g.programs))),
    )
}

/// One KTIR program into the digest: its function, its kind and its bindings, each through its own
/// structural `Hash` streamed whole into the SHA-256 ([`Structural`]), then its pin window.
fn program(d: &mut DigestSink, p: &crate::bundle_code::LaunchProgram<'_>) {
    use std::hash::Hash;
    let crate::bundle_code::LaunchProgram {
        func,
        program,
        args,
        lx_pin_window_bytes,
    } = p;
    d.tag(b"program");
    func.hash(&mut Structural(&mut d.0));
    program.hash(&mut Structural(&mut d.0));
    args.hash(&mut Structural(&mut d.0));
    lx_pin_window_bytes.visit(d);
}

/// ⭐ A TYPE'S OWN `Hash`, STREAMED LOSSLESSLY INTO THE SHA-256 — every byte the derived (or written)
/// `Hash` produces goes into the digest, so the value is hashed by its structure, as it is defined,
/// with no second rendering and no 64-bit fold in between. `finish` is never the name.
struct Structural<'a>(&'a mut Sha256);

impl std::hash::Hasher for Structural<'_> {
    fn write(&mut self, bytes: &[u8]) {
        self.0.update(bytes);
    }
    fn finish(&self) -> u64 {
        unreachable!("a structural stream is read as the digest it feeds, never as a u64")
    }
}

/// ⭐ THE DIGEST SINK OF THE BUNDLE'S ONE WALK. The walk is a fold, so each value reaches the sink
/// after its parts: every leaf is written as a tag and a fixed-width payload, and every record,
/// variant, option, pair and list as a tag and its arity (field names and count, element count) after
/// them. That is postfix notation with every arity stated, which parses exactly one way, so no two
/// different walks write one byte stream.
struct DigestSink(Sha256);

impl DigestSink {
    fn word(&mut self, v: u64) {
        self.0.update(v.to_le_bytes());
    }
    /// A length-prefixed byte string.
    fn tag(&mut self, b: &[u8]) {
        self.word(b.len() as u64);
        self.0.update(b);
    }
    fn path(&mut self, path: Path) {
        self.word(path.len() as u64);
        for s in path {
            self.tag(s.as_bytes());
        }
    }
}

impl FactSink for DigestSink {
    type Out = ();
    fn record(&mut self, path: Path, fields: Vec<(&'static str, ())>) {
        // The fields' values were written as the walk visited them; the record closes them.
        self.tag(b"}");
        self.path(path);
        self.word(fields.len() as u64);
        for (name, ()) in fields {
            self.tag(name.as_bytes());
        }
    }
    fn variant(&mut self, path: Path, variant: &'static str, fields: Vec<(&'static str, ())>) {
        self.tag(b"|");
        self.path(path);
        self.tag(variant.as_bytes());
        self.word(fields.len() as u64);
        for (name, ()) in fields {
            self.tag(name.as_bytes());
        }
    }
    fn option(&mut self, v: Option<()>) {
        self.tag(if v.is_some() { b"some" } else { b"none" });
    }
    fn list(&mut self, shape: ListShape, vs: Vec<()>) {
        let shape: &[u8] = match shape {
            ListShape::Array => b"array",
            ListShape::Slice => b"slice",
            ListShape::PlaceTable => b"places",
        };
        self.tag(shape);
        self.word(vs.len() as u64);
    }
    fn pair(&mut self, (): (), (): ()) {
        self.tag(b"pair");
    }
    fn u32(&mut self, v: u32) {
        self.tag(b"u32");
        self.word(u64::from(v));
    }
    fn u64(&mut self, v: u64) {
        self.tag(b"u64");
        self.word(v);
    }
    fn bool(&mut self, v: bool) {
        self.tag(b"bool");
        self.word(u64::from(v));
    }
    fn f32(&mut self, v: f32) {
        self.tag(b"f32");
        self.word(u64::from(v.to_bits()));
    }
    fn fold_scale(&mut self, v: f32) {
        self.tag(b"fold_scale");
        self.word(u64::from(v.to_bits()));
    }
    fn place(&mut self, id: PlaceId) {
        use std::hash::Hash;
        self.tag(b"place");
        id.hash(&mut Structural(&mut self.0));
    }
    fn fragment(&mut self, f: Fragment) {
        self.tag(b"fragment");
        for v in [f.first_stick(), f.sticks(), f.rows()] {
            self.word(u64::from(v));
        }
    }
    fn rank<W: Tp>(&mut self, r: Rank<W>) {
        self.tag(b"rank");
        self.word(u64::from(r.get()));
    }
    fn cut(&mut self, c: WindowCut) {
        self.tag(b"cut");
        self.word(c.index() as u64);
    }
}
