// SPDX-License-Identifier: Apache-2.0
//! One Triton kernel compiled to KTIR, once per key: the module memo, the handoff memo, and the mint
//! of a [`KtirNode`] over the handed-off function.

use ktir_core::arena::Arena;
use scratchy_spyre_bundle::program::{BufferId, KtirNode};

/// One memoized compile: the module every caller of the same key shares, paired with that key.
/// The only mint is [`ModuleMemo::compile_under`], so a key that did not produce its module cannot
/// be written.
pub struct CompiledKernel {
    module: std::sync::Arc<triton_ktir::ir::Module>,
    key: Option<String>,
}

impl std::ops::Deref for CompiledKernel {
    type Target = triton_ktir::ir::Module;
    fn deref(&self) -> &Self::Target {
        &self.module
    }
}

/// One key's module: computed at most once, by whichever caller gets there first.
type ModuleCell =
    std::sync::Arc<std::sync::OnceLock<Result<std::sync::Arc<triton_ktir::ir::Module>, String>>>;

/// One compile per key: a bake compiles the same kernel once per layer, rung and world, and the
/// compile is a pure function of its key, so every later caller shares the first result. A failure
/// is memoized like a success.
pub struct ModuleMemo {
    cells: std::sync::Mutex<std::collections::BTreeMap<String, ModuleCell>>,
}

impl Default for ModuleMemo {
    fn default() -> Self {
        Self::new()
    }
}

impl ModuleMemo {
    pub const fn new() -> ModuleMemo {
        ModuleMemo {
            cells: std::sync::Mutex::new(std::collections::BTreeMap::new()),
        }
    }

    /// The compile under `key`, computed once. A `None` key is never memoized.
    pub fn compile_under(
        &self,
        key: Option<String>,
        compile: impl FnOnce() -> Result<triton_ktir::ir::Module, String>,
    ) -> Result<CompiledKernel, String> {
        let Some(key) = key else {
            return Ok(CompiledKernel {
                module: std::sync::Arc::new(compile()?),
                key: None,
            });
        };
        let cell = {
            let mut cells = self.cells.lock().unwrap_or_else(|p| p.into_inner());
            std::sync::Arc::clone(cells.entry(key.clone()).or_default())
        };
        match cell.get_or_init(|| compile().map(std::sync::Arc::new)) {
            Ok(module) => Ok(CompiledKernel {
                module: std::sync::Arc::clone(module),
                key: Some(key),
            }),
            Err(e) => Err(e.clone()),
        }
    }
}

/// The node over a module, its handoff lowered on every call.
pub fn node_for(
    m: &triton_ktir::ir::Module,
    program: ktir_core::program::Program,
) -> Result<KtirNode, String> {
    Ok(mint_node(lower_handoff(m)?, program))
}

/// The node over a memoized compile, its handoff lowered once per key: every borrow in the
/// function comes from the process-lived, append-only [`Arena::global`], so a cached function
/// stays valid, and `IRFunction` is `Copy` over those borrows.
pub fn node_for_compiled(
    compiled: &CompiledKernel,
    program: ktir_core::program::Program,
) -> Result<KtirNode, String> {
    let func = match &compiled.key {
        Some(key) => cached_handoff(&compiled.module, key)?,
        None => lower_handoff(&compiled.module)?,
    };
    Ok(mint_node(func, program))
}

fn lower_handoff(
    m: &triton_ktir::ir::Module,
) -> Result<ktir_core::ir::IRFunction<'static>, String> {
    triton_ktir::passes::to_ktir_emit::lower(m, Arena::global())
        .map_err(|e| format!("handoff: {e}"))
}

type HandoffCell =
    std::sync::Arc<std::sync::OnceLock<Result<ktir_core::ir::IRFunction<'static>, String>>>;

fn cached_handoff(
    m: &triton_ktir::ir::Module,
    key: &str,
) -> Result<ktir_core::ir::IRFunction<'static>, String> {
    static CELLS: std::sync::Mutex<std::collections::BTreeMap<String, HandoffCell>> =
        std::sync::Mutex::new(std::collections::BTreeMap::new());
    let cell = {
        let mut cells = CELLS.lock().unwrap_or_else(|p| p.into_inner());
        std::sync::Arc::clone(cells.entry(key.to_string()).or_default())
    };
    match cell.get_or_init(|| lower_handoff(m)) {
        Ok(func) => Ok(*func),
        Err(e) => Err(e.clone()),
    }
}

/// Parameter position is the buffer number until the caller rebinds it.
fn mint_node(
    func: ktir_core::ir::IRFunction<'static>,
    program: ktir_core::program::Program,
) -> KtirNode {
    let bindings = (0..func.arguments.len())
        .map(|i| BufferId::new(i as u32))
        .collect();
    KtirNode {
        func,
        program,
        bindings,
    }
}
