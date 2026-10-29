//! THE SLAB SCATTER — a descriptor STORE whose row offset is DATA, Python source to KTIR.
//!
//! The store-side mirror of the slab gather (`expert_slab_gather.rs`): a kernel that writes a
//! block at a row it read from memory,
//!
//! ```text
//!   row = tl.load(rows + r)                                  tt.addptr + tt.load (SCALAR)
//!   pool.store([row, 0], x)                                  tt.descriptor_store
//! ```
//!
//! addresses memory through a value it LOADED, so the access is INDIRECT: it must lower to a
//! `ktdp.store` through a `ktdp.construct_indirect_access_tile` whose row dim is addressed
//! through the row table's memory view, every other dim direct. It is the spelling Triton leaves
//! for a write under `desc.scatter`'s floor of 8 rows (`paged_kv_write.py`'s row-at-a-time arm).

use std::collections::HashMap;

use triton_frontend::codegen::{self, ArgSpec, KernelSpec};
use triton_frontend::opt;
use triton_frontend::semantic::Val;
use triton_frontend::target::Target;

use triton_ktir::from_ttir;
use triton_ktir::ir::{Attr, AttrKey, Module, OpKind};

const SIG: &[(&str, &str)] = &[
    ("rows", "*i32"),
    ("x", "*fp16"),
    ("pool", "*fp16"),
    ("N", "constexpr"),
];

/// An ad-hoc kernel through the ladder, up to KTDP.
fn to_ktdp(src: &str) -> Result<Module, String> {
    let mut signature = HashMap::new();
    let mut constexprs = HashMap::new();
    for (k, v) in SIG {
        signature.insert(k.to_string(), ArgSpec::parse(v).expect("signature"));
        if *v == "constexpr" {
            constexprs.insert(k.to_string(), Val::Int(64));
        }
    }
    let spec = KernelSpec {
        kernel: "k".to_string(),
        signature,
        constexprs,
        file: "slab_scatter_test.py".to_string(),
    };
    let mut tt = codegen::compile(src, &spec, Target::spyre()).map_err(|e| e.to_string())?;
    opt::make_ttir(&mut tt).map_err(|e| e.to_string())?;
    let mut m = from_ttir::convert(&tt).map_err(|e| e.to_string())?;
    triton_ktir::make_ktir(&mut m, &[1]).map_err(|e| e.to_string())?;
    Ok(m)
}

fn count(m: &Module, k: OpKind) -> usize {
    m.ops_deep().into_iter().filter(|o| o.kind == k).count()
}

/// Two rows, each stored at the row the table names: two indirect tiles, each indirect on the
/// row axis and direct on the column axis, each the tile of a `ktdp.store` — and the scalar row
/// reads they subsumed are gone.
#[test]
fn a_store_at_a_loaded_row_is_an_indirect_store_over_the_row_table() {
    let src = r#"
import triton
import triton.language as tl
@triton.jit
def k(rows, x, pool, N: tl.constexpr):
    xd = tl.make_tensor_descriptor(x, shape=[2, N], strides=[N, 1], block_shape=[1, N])
    pd = tl.make_tensor_descriptor(pool, shape=[64, N], strides=[N, 1], block_shape=[1, N])
    for r in tl.static_range(2):
        row = tl.load(rows + r)
        pd.store([row, 0], xd.load([r, 0]))
"#;
    let m = to_ktdp(src).unwrap_or_else(|e| panic!("the slab-scatter kernel did not lower: {e}"));
    assert_eq!(
        count(&m, OpKind::KtdpConstructIndirectAccessTile),
        2,
        "one indirect tile per stored row"
    );
    assert_eq!(
        count(&m, OpKind::TtLoad),
        0,
        "every consumed row read is erased"
    );
    assert_eq!(
        count(&m, OpKind::TtAddptr),
        0,
        "and the pointer offsets it read through"
    );
    let ops = m.ops_deep();
    let indirect: Vec<_> = ops
        .iter()
        .filter(|o| o.kind == OpKind::KtdpConstructIndirectAccessTile)
        .collect();
    for t in &indirect {
        let Some(Attr::StrList(kinds)) = t.attr(&AttrKey::Other("per_dim_subscript_kinds".into()))
        else {
            panic!("an indirect tile without per_dim_subscript_kinds");
        };
        assert_eq!(
            kinds,
            &["true", "false"],
            "the row indirect, the columns direct"
        );
        let tile = t.result().expect("the tile defines a value");
        assert!(
            ops.iter()
                .any(|o| o.kind == OpKind::KtdpStore && o.operands.get(1) == Some(&tile)),
            "an indirect tile no ktdp.store writes through"
        );
    }
}

/// A loaded value may select ONE row block; a store several rows tall on the indirect axis would
/// need one index per row, which is `tt.descriptor_scatter`.
#[test]
fn a_data_offset_on_a_multi_row_store_is_refused_by_name() {
    let src = r#"
import triton
import triton.language as tl
@triton.jit
def k(rows, x, pool, N: tl.constexpr):
    xd = tl.make_tensor_descriptor(x, shape=[N, N], strides=[N, 1], block_shape=[N, N])
    pd = tl.make_tensor_descriptor(pool, shape=[8 * N, N], strides=[N, 1], block_shape=[N, N])
    row = tl.load(rows)
    pd.store([row, 0], xd.load([0, 0]))
"#;
    let Err(msg) = to_ktdp(src) else {
        panic!("the ladder accepted a data offset on a multi-row store");
    };
    assert!(
        msg.contains("tt.descriptor_store") && msg.contains("selects ONE slab"),
        "refused for the wrong reason: {msg}"
    );
}
