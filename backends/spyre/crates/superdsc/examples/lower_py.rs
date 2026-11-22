// SPDX-License-Identifier: Apache-2.0
//! Lower one Triton kernel end to end and write its dxp group directory.
//!
//! ```bash
//! cargo run -p scratchy-spyre-superdsc --example lower_py -- \
//!   <kernels/FILE.py> <kernel> <out-dir> <grid> name:*fp16 … NAME=value …
//! ```
//! `grid` is `x[,y[,z]]`; a `name:type` argument is a pointer parameter, `NAME=value` a constexpr
//! (an integer, else a float).

use std::collections::HashMap;

use ktir_core::arena::Arena;
use ktir_core::ddl_binds::Baked;
use triton_frontend::codegen::{ArgSpec, KernelSpec};
use triton_frontend::semantic::Val;
use triton_frontend::target::Target;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [file, kernel, out, grid, rest @ ..] = &args[..] else {
        eprintln!("use: lower_py <file.py> <kernel> <out-dir> <grid> name:*type… NAME=value…");
        std::process::exit(2);
    };
    let grid: Vec<i64> = grid.split(',').map(|g| g.parse().expect("grid")).collect();
    let mut signature = HashMap::new();
    let mut staged: Vec<usize> = Vec::new();
    let mut constexprs = HashMap::new();
    for a in rest {
        if let Some(list) = a.strip_prefix("staged=") {
            staged = list
                .split(',')
                .map(|p| p.parse().expect("staged param"))
                .collect();
        } else if let Some((k, v)) = a.split_once('=') {
            signature.insert(
                k.to_string(),
                ArgSpec::parse("constexpr").expect("constexpr"),
            );
            let val = v
                .parse::<i128>()
                .map(Val::Int)
                .unwrap_or_else(|_| Val::Float(v.parse().expect("value")));
            constexprs.insert(k.to_string(), val);
        } else if let Some((k, t)) = a.split_once(':') {
            signature.insert(k.to_string(), ArgSpec::parse(t).expect("type"));
        }
    }
    let spec = KernelSpec {
        kernel: kernel.clone(),
        signature,
        constexprs,
        file: file.clone(),
    };
    let src = std::fs::read_to_string(file).expect("read kernel");
    let mut tt = triton_frontend::codegen::compile(&src, &spec, Target::spyre()).expect("codegen");
    triton_frontend::opt::make_ttir(&mut tt).expect("make_ttir");
    let mut m = triton_ktir::from_ttir::convert(&tt).expect("from_ttir");
    triton_ktir::make_ktir(&mut m, &grid).expect("make_ktir");
    triton_ktir::passes::to_ktir::run(&mut m, &grid).expect("to_ktir");
    let f = triton_ktir::passes::to_ktir_emit::lower(&m, Arena::global()).expect("handoff");
    if rest.iter().any(|a| a == "dump") {
        fn dump(ops: &[ktir_core::ir::Operation<'_>], depth: usize) {
            for o in ops {
                let attrs: Vec<String> = o
                    .attributes
                    .iter()
                    .map(|(k, v)| format!("{k:?}={v:?}"))
                    .collect();
                println!(
                    "{:w$}{:?} = {:?}({:?}) {}",
                    "",
                    o.result,
                    o.op_type,
                    o.operands,
                    attrs.join(" ").chars().take(400).collect::<String>(),
                    w = depth * 2
                );
                for r in o.regions {
                    dump(r, depth + 1);
                }
            }
        }
        println!("args {:?} grid {:?}", f.arguments, f.grid);
        dump(f.operations, 0);
    }
    let params: Vec<spyre_superdsc::DevAddr> = (0..f.arguments.len() as u64)
        .map(|i| spyre_superdsc::DevAddr::new(0x4_0000_0000 + i * 0x4000_0000))
        .collect();
    let residency: Vec<spyre_superdsc::Residency> = (0..f.arguments.len())
        .map(|p| {
            if staged.contains(&p) {
                spyre_superdsc::Residency::Staged
            } else {
                spyre_superdsc::Residency::Device
            }
        })
        .collect();
    let layouts = spyre_superdsc::layouts::<Baked>(&f, &residency).expect("layouts");
    println!("layouts {layouts:?}");
    let natural = spyre_superdsc::lift::lift::<Baked>(&f, &layouts).expect("natural lift");
    let relocated = spyre_superdsc::layout::misaligned::<Baked>(&natural);
    let mut g = match spyre_superdsc::lift::lift_relocating::<Baked>(&f, &layouts, &relocated) {
        Ok(g) => g,
        Err(e) => {
            println!("REFUSED {e:?}");
            std::process::exit(1);
        }
    };
    spyre_superdsc::layout::restick::<Baked>(&mut g);
    spyre_superdsc::layout::index_inner::<Baked>(&mut g);
    for (i, n) in g.nodes.iter().enumerate() {
        println!(
            "node {i}: {:?} space {:?} ins {:?} out {:?}",
            n.func,
            n.space,
            n.inputs
                .iter()
                .map(|o| (
                    o.store,
                    o.view.offset,
                    o.view
                        .axes
                        .iter()
                        .map(|a| (a.extent, a.stride))
                        .collect::<Vec<_>>()
                ))
                .collect::<Vec<_>>(),
            (
                n.output.store,
                n.output.view.offset,
                n.output
                    .view
                    .axes
                    .iter()
                    .map(|a| (a.extent, a.stride))
                    .collect::<Vec<_>>()
            )
        );
    }
    let placed: Vec<(spyre_superdsc::graph::ConstStick, spyre_superdsc::DevAddr)> = g
        .consts
        .iter()
        .enumerate()
        .map(|(i, c)| {
            (
                *c,
                spyre_superdsc::DevAddr::new(0x2000_0000 + i as u64 * 128),
            )
        })
        .collect();
    let b = spyre_superdsc::Bindings {
        params: &params,
        consts: &placed,
        scratch: Some(spyre_superdsc::DevAddr::new(0x1000_0000)),
    };
    let fp = spyre_superdsc::emit::footprint::<Baked>(&g);
    match spyre_superdsc::emit::emit::<Baked>(&g, &fp, &b) {
        Ok(ops) => {
            let files = spyre_superdsc::emit::group_files(&ops).expect("render");
            let _ = std::fs::remove_dir_all(out);
            std::fs::create_dir_all(out).expect("mkdir");
            for (name, body) in files {
                std::fs::write(std::path::Path::new(out).join(name), body).expect("write");
            }
            println!(
                "LOWERED {} ops, scratch {} B, consts {:?}",
                ops.len(),
                fp.scratch_bytes,
                g.consts
            );
        }
        Err(e) => {
            println!("REFUSED {e:?}");
            std::process::exit(1);
        }
    }
}
