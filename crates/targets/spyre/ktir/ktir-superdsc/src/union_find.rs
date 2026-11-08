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
//! EQUALITIES, each component optionally ANCHORED to one absolute label, unanchored
//! components resolving to a caller's one default.
//!
//! Unit weight only: the frame pass states equalities and absolutes, nothing more.
//! A solve that needs a weight on each edge (the element-arrangement solve's mod-4
//! lane shifts) brings its own group machinery with the pass that uses it — a
//! generic `Group`/`Anchor` algebra with no consumer cannot state its own laws
//! (notably whether `compose` commutes), so it is not carried ahead of need.
//!
//! # THE LAWS
//!
//! - **ONE structure.** Every constraint-bearing solve in this crate lowers through
//!   this module — no second union-find, no side table of classes. A reader that
//!   needs the answer asks [`UnionFind::resolve`]; it never re-derives a class.
//! - **THE ANCHOR IS A SIDE TABLE, NOT A TREE NODE.** The anchor is keyed by root,
//!   not stored as node 0 or any node at all: no cycle is possible, a component
//!   can be anchored from any of its nodes, and the anchor MIGRATES to the new
//!   root when its component is attached under another.
//! - **FIRST ANCHOR WINS.** A component keeps the first anchor stated for it; a
//!   second, different anchor is the bake error, and the message names the op that
//!   stated EACH — the stored label's own `why`, never an iteration order.
//! - **UNANCHORED IS NOT UNDETERMINED.** A component no anchor was stated for
//!   resolves through the caller's one default — the frame pass's "a class the
//!   program states no column reader for is a row". There is no second default
//!   and no third case.
//! - **DETERMINISM.** Joins attach the second side's root under the first's, always;
//!   classes and refusals are decided by the caller's call order (the program's own
//!   order), never by a hash. The anchor map is a lookup, never an iteration.

use std::collections::HashMap;
use std::fmt::Debug;

/// One constraint solve: a forest of nodes (the caller's values, indexed `0..n`), the
/// equalities its constraints stated, and the anchors its absolutes pinned. `A` is
/// the label a component can be anchored to — the frame pass's [`World`]-like
/// classification.
pub struct UnionFind<A: Copy + Eq + Debug> {
    parent: Vec<u32>,
    /// The constraint that last named the node — what a two-anchors clash quotes.
    why: Vec<String>,
    /// The component's anchor, BY ROOT. The first anchor stated for a class; never
    /// iterated.
    anchors: HashMap<u32, (A, String)>,
}

impl<A: Copy + Eq + Debug> UnionFind<A> {
    /// A solve over `n` nodes, no constraints stated.
    pub fn with_len(n: usize) -> Self {
        Self {
            parent: (0..n as u32).collect(),
            why: vec![String::new(); n],
            anchors: HashMap::new(),
        }
    }

    /// `find` with path compression: the component's root.
    fn find(&mut self, x: u32) -> u32 {
        let p = self.parent[x as usize];
        if p == x {
            return x;
        }
        let root = self.find(p);
        self.parent[x as usize] = root;
        root
    }

    /// ⭐ STATE ONE CONSTRAINT — `a` and `b` are one value, labelled by the op that
    /// stated it. A join that would weld two differently-anchored components into one
    /// is the bake error, both anchors named. The second side's root is attached
    /// under the first's, always.
    pub fn join(&mut self, a: u32, b: u32, why: &str) -> Result<(), String> {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            self.why[a as usize] = why.to_string();
            self.why[b as usize] = why.to_string();
            return Ok(());
        }
        // ⛔ TWO ANCHORED COMPONENTS JOIN ONLY WHEN THE ANCHORS AGREE — otherwise the
        // one join states two absolutes for one class, and the error names the op
        // that stated EACH.
        if let (Some((va, wa)), Some((vb, wb))) = (self.anchors.get(&ra), self.anchors.get(&rb))
            && va != vb
        {
            return Err(format!(
                "a join of two anchored classes: `{wa}` anchors one side at {va:?} and \
                 `{wb}` anchors the other at {vb:?}, but `{why}` states they are one value — \
                 there is no one anchor satisfying all three",
            ));
        }
        self.parent[rb as usize] = ra;
        self.why[rb as usize] = why.to_string();
        self.why[a as usize] = why.to_string();
        self.why[b as usize] = why.to_string();
        // The attached side's anchor MIGRATES to the new root — unless the surviving
        // root already has one (they agreed above; the first stated keeps the class).
        if !self.anchors.contains_key(&ra)
            && let Some(anchor) = self.anchors.remove(&rb)
        {
            self.anchors.insert(ra, anchor);
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
        let root = self.find(x);
        if let Some((v, w)) = self.anchors.get(&root)
            && *v != val
        {
            return Err(format!(
                "a class anchored two ways: `{w}` anchors it at {v:?} and `{why}` anchors it at \
                 {val:?} — there is no one value for the class; the op that stated each is \
                 named",
            ));
        }
        self.anchors
            .entry(root)
            .or_insert_with(|| (val, why.to_string()));
        Ok(())
    }

    /// ⭐ The node's SOLVED label: its component's anchor, or the caller's one
    /// default for a component no anchor was stated for. There is no other case and
    /// no second default.
    pub fn resolve(&mut self, x: u32, default: A) -> A {
        let root = self.find(x);
        match self.anchors.get(&root) {
            Some((v, _)) => *v,
            None => default,
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

    #[test]
    fn an_equality_join_shares_one_anchor() {
        let mut uf: UnionFind<World> = UnionFind::with_len(3);
        uf.join(0, 1, "the exp of the folded max").unwrap();
        uf.anchor(0, World::Column, "the stick fold writes a column")
            .unwrap();
        assert_eq!(uf.resolve(1, World::Row), World::Column);
        // The unjoined node resolves the default — the one row-once function.
        assert_eq!(uf.resolve(2, World::Row), World::Row);
    }

    #[test]
    fn a_repeated_equality_is_silent() {
        let mut uf: UnionFind<World> = UnionFind::with_len(2);
        uf.join(0, 1, "the first statement").unwrap();
        // The second statement states what the first already did.
        uf.join(1, 0, "the second statement").unwrap();
        assert_eq!(uf.resolve(0, World::Row), uf.resolve(1, World::Row));
    }

    #[test]
    fn two_anchored_classes_join_only_when_the_anchors_agree() {
        // Agreeing anchors: one class.
        let mut uf: UnionFind<World> = UnionFind::with_len(3);
        uf.anchor(0, World::Column, "a col spray of the source")
            .unwrap();
        uf.anchor(1, World::Column, "another col spray").unwrap();
        uf.join(0, 1, "the exp lane").unwrap();
        assert_eq!(uf.resolve(0, World::Row), World::Column);
        // Disagreeing anchors: the join is refused, both named.
        uf.anchor(2, World::Row, "the matmul writes a row").unwrap();
        let Err(e) = uf.join(0, 2, "a lane between them") else {
            panic!("a join of two differently-anchored classes must refuse");
        };
        assert!(
            e.contains("a col spray of the source") && e.contains("the matmul writes a row"),
            "names the op that stated each anchor: {e}"
        );
    }

    #[test]
    fn a_second_anchor_on_one_class_refuses_by_name() {
        let mut uf: UnionFind<World> = UnionFind::with_len(2);
        uf.join(0, 1, "the shape op keeps the frame").unwrap();
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
        let mut uf: UnionFind<World> = UnionFind::with_len(2);
        uf.anchor(1, World::Column, "the spray's source is a column")
            .unwrap();
        // The join attaches 1's root under 0's — the anchor must travel with it.
        uf.join(0, 1, "the reduce keeps the operand's frame")
            .unwrap();
        assert_eq!(uf.resolve(0, World::Row), World::Column);
    }

    #[test]
    fn an_unanchored_component_resolves_the_one_default() {
        let mut uf: UnionFind<World> = UnionFind::with_len(3);
        uf.join(0, 1, "one equality").unwrap();
        uf.join(1, 2, "another").unwrap();
        for x in [0u32, 1, 2] {
            assert_eq!(uf.resolve(x, World::Row), World::Row);
        }
    }
}
