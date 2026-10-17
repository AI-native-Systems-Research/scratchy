//! THE SLAB GATHER — a rank-3 descriptor load whose expert offset is DATA, Python source to
//! KTIR.
//!
//! `test-fixtures/expert_slab.py` is upstream's mixture-of-experts matmul idiom
//! (`triton_kernels/matmul_details/_p_matmul.py:423`):
//!
//! ```text
//!   expt = tl.load(ids + p)                                  tt.addptr + tt.load (SCALAR)
//!   w    = tl.reshape(W.load([expt, 0, 0]), [N, K])          tt.descriptor_load + tt.reshape
//!   o    = tl.dot(x, w.T, acc)                               tt.trans + tt.dot
//! ```
//!
//! The address of `W`'s block depends on a value READ FROM MEMORY, so the access is
//! INDIRECT: it must lower to `ktdp.construct_indirect_access_tile`, whose expert dim is
//! addressed through the ids memory view at the pair's own element, every other dim direct.
//! A DIRECT access tile with the loaded scalar as an `index` operand would be well-formed
//! KTIR that no consumer can address (the card's gather reads its index from memory, the
//! emulator's indirect load reads it from the index view), which is exactly the
//! silent-misread class this test exists to rule out.

use std::collections::HashMap;

use triton_frontend::codegen::{self, ArgSpec, KernelSpec};
use triton_frontend::opt;
use triton_frontend::semantic::Val;
use triton_frontend::target::Target;

use triton_ktir::from_ttir;
use triton_ktir::ir::{Module, OpKind};

fn fixture_src() -> (String, String) {
    let p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../test-fixtures/expert_slab.py");
    let src =
        std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("cannot read {}: {e}", p.display()));
    (src, p.to_string_lossy().to_string())
}

fn spec(file: String) -> KernelSpec {
    let mut signature = HashMap::new();
    for (k, v) in [
        ("desc_ids", "*i32"),
        ("desc_w", "*fp16"),
        ("desc_x", "*fp16"),
        ("desc_o", "*fp16"),
        ("P", "constexpr"),
        ("E", "constexpr"),
        ("N", "constexpr"),
        ("K", "constexpr"),
    ] {
        signature.insert(k.to_string(), ArgSpec::parse(v).expect("signature"));
    }
    let mut constexprs = HashMap::new();
    for (k, v) in [("P", 8), ("E", 8), ("N", 64), ("K", 128)] {
        constexprs.insert(k.to_string(), Val::Int(v));
    }
    KernelSpec {
        kernel: "expert_slab_fwd".to_string(),
        signature,
        constexprs,
        file,
    }
}

/// The ladder exactly as the splice drives it (`compile_kernel_uncached`), up to KTDP.
fn to_ktdp() -> Result<Module, String> {
    let (src, file) = fixture_src();
    let mut tt = codegen::compile(&src, &spec(file), Target::spyre()).map_err(|e| e.to_string())?;
    opt::make_ttir(&mut tt).map_err(|e| e.to_string())?;
    let mut m = from_ttir::convert(&tt).map_err(|e| e.to_string())?;
    triton_ktir::make_ktir(&mut m, &[1]).map_err(|e| e.to_string())?;
    Ok(m)
}

fn count(m: &Module, k: OpKind) -> usize {
    m.ops_deep().into_iter().filter(|o| o.kind == k).count()
}

/// Every one of the eight pairs' slab loads is an INDIRECT tile, the scalar id reads it
/// subsumed are gone, and no other op still reads the ids pointer.
#[test]
fn every_slab_load_is_an_indirect_tile_over_the_ids_view() {
    let m = to_ktdp().unwrap_or_else(|e| panic!("the slab-gather fixture did not lower: {e}"));
    assert_eq!(
        count(&m, OpKind::KtdpConstructIndirectAccessTile),
        8,
        "one indirect tile per routed pair"
    );
    assert_eq!(
        count(&m, OpKind::TtLoad),
        0,
        "every consumed id read is erased"
    );
    assert_eq!(
        count(&m, OpKind::TtAddptr),
        0,
        "and the pointer offsets it read through"
    );
    // Each tile states exactly one indirect dim -- the expert axis, dim 0.
    for t in m
        .ops_deep()
        .into_iter()
        .filter(|o| o.kind == OpKind::KtdpConstructIndirectAccessTile)
    {
        let Some(triton_ktir::ir::Attr::StrList(kinds)) = t.attr(&triton_ktir::ir::AttrKey::Other(
            "per_dim_subscript_kinds".into(),
        )) else {
            panic!("an indirect tile without per_dim_subscript_kinds");
        };
        assert_eq!(
            kinds,
            &["true", "false", "false"],
            "dim 0 indirect, N and K direct"
        );
    }
}

/// Compile an ad-hoc kernel through the ladder and return the refusal text.
fn refusal(src: &str, sig: &[(&str, &str)]) -> String {
    let mut signature = HashMap::new();
    let mut constexprs = HashMap::new();
    for (k, v) in sig {
        signature.insert(k.to_string(), ArgSpec::parse(v).expect("signature"));
        if *v == "constexpr" {
            constexprs.insert(k.to_string(), Val::Int(64));
        }
    }
    let spec = KernelSpec {
        kernel: "k".to_string(),
        signature,
        constexprs,
        file: "refusal_test.py".to_string(),
    };
    let mut tt = codegen::compile(src, &spec, Target::spyre()).expect("the front end admits it");
    opt::make_ttir(&mut tt).expect("make_ttir");
    let mut m = from_ttir::convert(&tt).expect("from_ttir");
    match triton_ktir::make_ktir(&mut m, &[1]) {
        Ok(()) => panic!("the ladder accepted a kernel it must refuse"),
        Err(e) => e.to_string(),
    }
}

const SIG: &[(&str, &str)] = &[
    ("ids", "*i32"),
    ("w", "*fp16"),
    ("o", "*fp16"),
    ("N", "constexpr"),
];

/// A loaded value may select ONE slab; a block several rows tall on the indirect axis would
/// need one index per row, which is `tt.descriptor_gather`.
#[test]
fn a_data_offset_on_a_multi_row_block_axis_is_refused_by_name() {
    let src = r#"
import triton
import triton.language as tl
@triton.jit
def k(ids, w, o, N: tl.constexpr):
    wd = tl.make_tensor_descriptor(w, shape=[8 * N, N], strides=[N, 1], block_shape=[N, N])
    od = tl.make_tensor_descriptor(o, shape=[N, N], strides=[N, 1], block_shape=[N, N])
    e = tl.load(ids)
    od.store([0, 0], wd.load([e, 0]))
"#;
    let msg = refusal(src, SIG);
    assert!(
        msg.contains("selects ONE slab"),
        "refused for the wrong reason: {msg}"
    );
}

/// Two data offsets would need two index views on one tile.
#[test]
fn two_data_offsets_are_refused_by_name() {
    let src = r#"
import triton
import triton.language as tl
@triton.jit
def k(ids, w, o, N: tl.constexpr):
    wd = tl.make_tensor_descriptor(w, shape=[8, 8, N], strides=[8 * N, N, 1],
                                   block_shape=[1, 1, N])
    od = tl.make_tensor_descriptor(o, shape=[1, 1, N], strides=[N, N, 1],
                                   block_shape=[1, 1, N])
    a = tl.load(ids)
    b = tl.load(ids + 1)
    od.store([0, 0, 0], wd.load([a, b, 0]))
"#;
    let msg = refusal(src, SIG);
    assert!(
        msg.contains("data-dependent offsets"),
        "refused for the wrong reason: {msg}"
    );
}
