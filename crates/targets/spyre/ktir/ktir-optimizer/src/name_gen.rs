//! Fresh SSA identities for a rewrite.

use ktir_core::ir::{IRFunction, Ssa};

/// A monotonic SSA name generator (collision-free within one rewritten body): mints the values a
/// rewrite introduces. An id cannot collide with one already in the function because the function
/// states its own high-water mark.
pub struct NameGen {
    next: u32,
}

impl NameGen {
    /// Start above every value `func` already defines, so a minted id is fresh by construction.
    pub fn after(func: &IRFunction<'_>) -> NameGen {
        let high = func
            .operations
            .iter()
            .filter_map(|op| op.result)
            .chain(func.arguments.iter().map(|(v, _)| *v))
            .map(|v| v.0)
            .max()
            .map(|m| m + 1)
            .unwrap_or(0);
        NameGen { next: high }
    }

    /// A fresh identity, guaranteed distinct from every value already seeded past.
    pub fn mint(&mut self) -> Ssa {
        let v = Ssa(self.next);
        self.next += 1;
        v
    }
}
