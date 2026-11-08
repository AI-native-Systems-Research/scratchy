// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! ⭐ THE CONSTRAINT SOLVE'S ONE UNION-FIND — components of values joined by stated
//! constraints, each component optionally ANCHORED to one absolute label, unanchored
//! components resolving to a caller's default. Generic over the WEIGHT a constraint
//! carries (a group: the frame pass states equalities only — the unit group — while
//! the element-arrangement solve states mod-4 lane shifts) and over the LABEL a
//! component can be anchored to (the frame pass's `World`; the arrangement's own
//! powers).
//!
//! # THE LAWS
//!
//! - **ONE structure.** Every constraint-bearing solve in this crate lowers through
//!   this module — no second union-find, no side table of classes. A reader that
//!   needs the answer asks [`UnionFind::resolve`]; it never re-derives a class.
//! - **THE ANCHOR IS A SIDE TABLE, NOT A TREE NODE.** Z4's arrangement solve kept its
//!   anchor as node 0 and never reparented it, because a `pin` welding a second
//!   anchor under the first would close a cycle and overflow `find`. Anchors here are
//!   keyed by root instead: no cycle is possible, a component can be anchored from
//!   any of its nodes, and the anchor MIGRATES to the new root when its component is
//!   attached under another.
//! - **FIRST ANCHOR WINS.** A component keeps the first anchor stated for it; a
//!   second, different anchor is the bake error, and the message names the op that
//!   stated EACH — the stored label's own `why`, never an iteration order.
//! - **UNANCHORED IS NOT UNDETERMINED.** A component no anchor was stated for
//!   resolves through the caller's one default — the frame pass's "a class the
//!   program states no column reader for is a row", the arrangement's "a component
//!   the program states no channel for reads k = 0 at its root". There is no second
//!   default and no third case.
//! - **DETERMINISM.** Joins attach the second side's root under the first's, always;
//!   classes and refusals are decided by the caller's call order (the program's own
//!   order), never by a hash. The anchor map is a lookup, never an iteration.

use std::collections::HashMap;
use std::fmt::Debug;

/// The WEIGHT a join carries — a group: the weight from `a` to `b` stated by one
/// constraint, composed along paths, inverted to translate anchors across an attach.
/// The frame pass states equalities only (`()`); the element-arrangement solve states
/// mod-4 lane shifts.
pub trait Group: Copy + Eq + Debug {
    fn identity() -> Self;
    fn compose(self, other: Self) -> Self;
    fn inverse(self) -> Self;
}

/// Equality only — the frame pass's joins.
impl Group for () {
    fn identity() {}
    fn compose(self, (): ()) {}
    fn inverse(self) {}
}

/// What a component can be ANCHORED to, and how the anchor rides the component's
/// weights: the value a root labelled `self` gives the node whose weight-to-root is
/// `w`. The unit weight cannot move a label (`World`); a mod-4 weight translates a
/// power along the lane shift.
pub trait Anchor<W: Group>: Copy + Eq + Debug {
    fn rides(self, w: W) -> Self;
}

/// One constraint solve: a forest of nodes (the caller's values, indexed `0..n`), the
/// weighted edges its constraints stated, and the anchors its absolutes pinned.
pub struct UnionFind<W: Group, A: Anchor<W>> {
    parent: Vec<u32>,
    /// The weight from a node to its parent.
    off: Vec<W>,
    /// The constraint that last named the node — what a two-paths conflict quotes.
    why: Vec<String>,
    /// The component's anchor, BY ROOT. The first anchor stated for a class; never
    /// iterated.
    anchors: HashMap<u32, (A, String)>,
}

impl<W: Group, A: Anchor<W>> UnionFind<W, A> {
    /// A solve over `n` nodes, no constraints stated.
    pub fn with_len(n: usize) -> Self {
        Self {
            parent: (0..n as u32).collect(),
            off: vec![W::identity(); n],
            why: vec![String::new(); n],
            anchors: HashMap::new(),
        }
    }

    /// `find` with path compression: the component's root, and the weight from `x`
    /// to it (the value at `x` is its root's anchor composed with this).
    fn find(&mut self, x: u32) -> (u32, W) {
        let p = self.parent[x as usize];
        if p == x {
            return (x, W::identity());
        }
        let (root, off_to_root) = self.find(p);
        self.parent[x as usize] = root;
        self.off[x as usize] = self.off[x as usize].compose(off_to_root);
        (root, self.off[x as usize])
    }

    /// ⭐ STATE ONE CONSTRAINT — the weight from `a` to `b` is `w`, labelled by the op
    /// that stated it. A constraint that CONTRADICTS one already in the same
    /// component is the bake error, both paths named; so is a join that would weld
    /// two differently-anchored components into one. The second side's root is
    /// attached under the first's, always.
    pub fn join(&mut self, a: u32, b: u32, w: W, why: &str) -> Result<(), String> {
        let (ra, oa) = self.find(a);
        let (rb, ob) = self.find(b);
        if ra == rb {
            // The paths already state the weight from a to b: oa⁻¹ ∘ ob. The
            // constraint agrees or it is the two-paths error.
            if ob != oa.compose(w) {
                return Err(format!(
                    "two paths into one component disagree: `{}` leaves node {a} at {oa:?} \
                     from the root while `{}` leaves node {b} at {ob:?}, and `{why}` states the \
                     two meet at {w:?} — there is no one weight satisfying all three",
                    self.why[a as usize], self.why[b as usize],
                ));
            }
            self.why[a as usize] = why.to_string();
            self.why[b as usize] = why.to_string();
            return Ok(());
        }
        // Attach rb's component under ra: the weight from rb to ra that the stated
        // constraint implies, oa ∘ w ∘ ob⁻¹.
        let edge = oa.compose(w).compose(ob.inverse());
        // ⛔ TWO ANCHORED COMPONENTS JOIN ONLY WHEN THE ANCHORS AGREE through the
        // edge — otherwise the one join states two absolutes for one class, and the
        // error names the op that stated EACH.
        if let (Some((va, wa)), Some((vb, wb))) = (self.anchors.get(&ra), self.anchors.get(&rb))
            && va.rides(edge) != *vb
        {
            return Err(format!(
                "a join of two anchored classes: `{wa}` anchors one side at {va:?} and \
                 `{wb}` anchors the other at {vb:?}, but `{why}` states they are one value — \
                 there is no one anchor satisfying all three",
            ));
        }
        self.parent[rb as usize] = ra;
        self.off[rb as usize] = edge;
        self.why[rb as usize] = why.to_string();
        self.why[a as usize] = why.to_string();
        self.why[b as usize] = why.to_string();
        // The attached side's anchor MIGRATES to the new root, translated across the
        // edge — unless the surviving root already has one (they agreed above; the
        // first stated keeps the class).
        if !self.anchors.contains_key(&ra)
            && let Some((vb, wb)) = self.anchors.remove(&rb)
        {
            self.anchors.insert(ra, (vb.rides(edge.inverse()), wb));
        } else {
            self.anchors.remove(&rb);
        }
        Ok(())
    }

    /// ⭐ ANCHOR A NODE — the component's absolute states that the node's value is
    /// `val`, labelled by the op that stated it. A component already anchored
    /// differently is the bake error, naming the op that stated EACH anchor. The
    /// first anchor stated for a class is the one it keeps.
    pub fn anchor(&mut self, x: u32, val: A, why: &str) -> Result<(), String> {
        let (root, ox) = self.find(x);
        // The label at the root that gives x `val` through ox.
        let at_root = val.rides(ox.inverse());
        if let Some((v, w)) = self.anchors.get(&root)
            && *v != at_root
        {
            return Err(format!(
                "a class anchored two ways: `{w}` anchors it at {v:?} and `{why}` anchors it at \
                 {at_root:?} — there is no one value for the class; the op that stated each is \
                 named",
            ));
        }
        self.anchors
            .entry(root)
            .or_insert_with(|| (at_root, why.to_string()));
        Ok(())
    }

    /// ⭐ The node's SOLVED label: its component's anchor, or the caller's one
    /// default at the root for a component no anchor was stated for. There is no
    /// other case and no second default.
    pub fn resolve(&mut self, x: u32, default: A) -> A {
        let (root, ox) = self.find(x);
        match self.anchors.get(&root) {
            Some((v, _)) => v.rides(ox),
            None => default.rides(ox),
        }
    }
}

#[cfg(test)]
mod union_find_tests {
    use super::*;

    /// The frame pass's anchor: which of the two worlds a class is held in.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum World {
        Column,
        Row,
    }

    impl Anchor<()> for World {
        fn rides(self, (): ()) -> Self {
            self
        }
    }

    /// The element-arrangement solve's weight: a mod-4 lane shift.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct Mod4(i8);

    impl Group for Mod4 {
        fn identity() -> Self {
            Mod4(0)
        }
        fn compose(self, other: Self) -> Self {
            Mod4((self.0 + other.0).rem_euclid(4))
        }
        fn inverse(self) -> Self {
            Mod4((-self.0).rem_euclid(4))
        }
    }

    impl Anchor<Mod4> for Mod4 {
        fn rides(self, w: Mod4) -> Self {
            self.compose(w)
        }
    }

    #[test]
    fn an_equality_join_shares_one_anchor() {
        let mut uf: UnionFind<(), World> = UnionFind::with_len(3);
        uf.join(0, 1, (), "the exp of the folded max").unwrap();
        uf.anchor(0, World::Column, "the stick fold writes a column")
            .unwrap();
        assert_eq!(uf.resolve(1, World::Row), World::Column);
        // The unjoined node resolves the default — the one row-once function.
        assert_eq!(uf.resolve(2, World::Row), World::Row);
    }

    #[test]
    fn an_anchor_rides_the_weights_to_every_node() {
        let mut uf: UnionFind<Mod4, Mod4> = UnionFind::with_len(3);
        uf.join(0, 1, Mod4(1), "one lane shift").unwrap();
        uf.join(1, 2, Mod4(1), "another lane shift").unwrap();
        uf.anchor(0, Mod4(0), "the standard-order channel").unwrap();
        assert_eq!(uf.resolve(1, Mod4(0)), Mod4(1));
        assert_eq!(uf.resolve(2, Mod4(0)), Mod4(2));
    }

    #[test]
    fn a_weighted_cycle_that_closes_is_silent() {
        let mut uf: UnionFind<Mod4, Mod4> = UnionFind::with_len(3);
        uf.join(0, 1, Mod4(1), "edge one").unwrap();
        uf.join(1, 2, Mod4(1), "edge two").unwrap();
        // The third edge states what the first two already imply.
        uf.join(0, 2, Mod4(2), "edge three").unwrap();
    }

    #[test]
    fn a_weighted_cycle_that_does_not_close_is_refused_naming_both_paths() {
        let mut uf: UnionFind<Mod4, Mod4> = UnionFind::with_len(2);
        uf.join(0, 1, Mod4(1), "the first op's shift").unwrap();
        let Err(e) = uf.join(0, 1, Mod4(2), "the second op's shift") else {
            panic!("a cycle that does not close must refuse");
        };
        assert!(
            e.contains("the first op's shift"),
            "names the first path: {e}"
        );
        assert!(e.contains("the second op's shift"), "names the second: {e}");
    }

    #[test]
    fn two_anchored_classes_join_only_when_the_anchors_agree() {
        // Agreeing anchors: one class.
        let mut uf: UnionFind<(), World> = UnionFind::with_len(3);
        uf.anchor(0, World::Column, "a col spray of the source")
            .unwrap();
        uf.anchor(1, World::Column, "another col spray").unwrap();
        uf.join(0, 1, (), "the exp lane").unwrap();
        assert_eq!(uf.resolve(0, World::Row), World::Column);
        // Disagreeing anchors: the join is refused, both named.
        uf.anchor(2, World::Row, "the matmul writes a row").unwrap();
        let Err(e) = uf.join(0, 2, (), "a lane between them") else {
            panic!("a join of two differently-anchored classes must refuse");
        };
        assert!(
            e.contains("a col spray of the source") && e.contains("the matmul writes a row"),
            "names the op that stated each anchor: {e}"
        );
    }

    #[test]
    fn a_second_anchor_on_one_class_refuses_by_name() {
        let mut uf: UnionFind<(), World> = UnionFind::with_len(2);
        uf.join(0, 1, (), "the shape op keeps the frame").unwrap();
        uf.anchor(0, World::Column, "the first anchor").unwrap();
        let Err(e) = uf.anchor(1, World::Row, "the second anchor") else {
            panic!("a second, different anchor on one class must refuse");
        };
        assert!(
            e.contains("the first anchor") && e.contains("the second anchor"),
            "names both: {e}"
        );
        // The same anchor stated twice is silent.
        uf.anchor(1, World::Column, "the same anchor again")
            .unwrap();
    }

    #[test]
    fn an_anchor_migrates_to_the_new_root_when_its_component_is_attached() {
        let mut uf: UnionFind<(), World> = UnionFind::with_len(2);
        uf.anchor(1, World::Column, "the spray's source is a column")
            .unwrap();
        // The join attaches 1's root under 0's — the anchor must travel with it.
        uf.join(0, 1, (), "the reduce keeps the operand's frame")
            .unwrap();
        assert_eq!(uf.resolve(0, World::Row), World::Column);
    }

    #[test]
    fn an_anchor_migrates_across_the_weights_it_rides() {
        let mut uf: UnionFind<Mod4, Mod4> = UnionFind::with_len(3);
        uf.join(1, 2, Mod4(2), "two lanes over").unwrap();
        uf.anchor(2, Mod4(3), "the channel reads k = 3 at node 2")
            .unwrap();
        // Attaching 1's component (2 is one shift back... two lanes from 1) under 0:
        // the anchor translates across the edge the join states.
        uf.join(0, 1, Mod4(1), "one lane over").unwrap();
        // k(2) = 3 is the anchor; the joins state k(2) = k(1) + 2 and k(1) = k(0) + 1,
        // so the one solution is k(0) = 0, k(1) = 1, k(2) = 3.
        assert_eq!(uf.resolve(0, Mod4(0)), Mod4(0));
        assert_eq!(uf.resolve(1, Mod4(0)), Mod4(1));
        assert_eq!(uf.resolve(2, Mod4(0)), Mod4(3));
    }

    #[test]
    fn an_unanchored_component_resolves_the_one_default() {
        let mut uf: UnionFind<(), World> = UnionFind::with_len(3);
        uf.join(0, 1, (), "one equality").unwrap();
        uf.join(1, 2, (), "another").unwrap();
        for x in [0u32, 1, 2] {
            assert_eq!(uf.resolve(x, World::Row), World::Row);
        }
    }
}
