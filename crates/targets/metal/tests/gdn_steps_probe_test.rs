// SPDX-License-Identifier: Apache-2.0
//! A Gated-DeltaNet layer's decode-time cost, by the step it serves, at Qwen3.6-35B-A3B's geometry
//! (16 key heads, 32 value heads, 128x128, conv 8192 wide, kernel 4): the one-command decode of a
//! plain step's row (`gdn_decode`), the conv, scan and norm chain of the same row, and the chain
//! of a verify step's 3 rows (2 drafts) replaying the previous verify step's kept rows (1 or 3)
//! from its draft lane. Beside them, a near-empty dispatch: what one more dispatch a barrier apart
//! costs. Each cell is µs a layer over one command buffer of `LAYERS` x `REPEATS` layers, each
//! dispatch a barrier apart, the f32 states rotated over `LAYERS` copies (60 MB, past the system
//! cache), best of 3.
//!
//! Run: cargo test -p scratchy-target-metal --release --test gdn_steps_probe_test -- \
//!     --ignored --nocapture

#![cfg(target_os = "macos")]

use objc2_metal::{MTL4CommandEncoder as _, MTL4VisibilityOptions, MTLSize, MTLStages};
use scratchy_target_metal::aot::baked_build;
use scratchy_target_metal::detect_device;
use scratchy_target_metal::gdn_state::{
    CheckpointRows, DraftLanes, GdnSeqSlots, GdnSlot, GdnStart, GdnStep, RecordArea,
};
use scratchy_target_metal::mtl4_dispatch::{Buffer, Mtl4DispatchBatch, Pipeline, shared_slice};
use scratchy_target_metal::specialized_pipeline_cache::{
    ConstantValue, PipelineKey, SpecializedPipelineCache,
};

/// Qwen3.6's GDN geometry.
const NK: usize = 16;
const NV: usize = 32;
const HK: usize = 128;
const HV: usize = 128;
const KERNEL: usize = 4;
const CONV_DIM: usize = 2 * NK * HK + NV * HV;
const DRAFTS: u8 = 2;
/// The conv's token block (`GDN_CONV_BLOCK`).
const CONV_BLOCK: u32 = 32;

/// A step's GDN layers (Qwen3.6's linear-attention layers), each its own state.
const LAYERS: usize = 30;
const REPEATS: usize = 4;

fn fill(n: usize, by: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i + by) as f32 * 0.1).sin() * 0.05)
        .collect()
}

fn size((width, height, depth): (usize, usize, usize)) -> MTLSize {
    MTLSize {
        width,
        height,
        depth,
    }
}

/// One layer's state: its slot's conv and ssm entries and its draft lane.
struct Layer {
    conv: Buffer,
    ssm: Buffer,
    conv_lanes: Buffer,
    ssm_lanes: Buffer,
}

/// One dispatch: its pipeline, bindings (`None`: the layer's state, by role), grid and threads.
struct Dispatch<'a> {
    pso: &'a Pipeline,
    binds: Vec<(Bind<'a>, usize)>,
    grid: (usize, usize, usize),
    threads: (usize, usize, usize),
}

enum Bind<'a> {
    Shared(&'a Buffer),
    Conv,
    Ssm,
    ConvLanes,
    SsmLanes,
}

#[test]
#[ignore = "measurement probe, not an assertion: run with --ignored --nocapture"]
fn gdn_layer_by_step() {
    let Some(di) = detect_device() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let device = di.device.clone();
    let cache = SpecializedPipelineCache::new(device.clone(), &[]).expect("pipeline cache");
    let build = |lib: &'static str, name: &'static str, c: Vec<ConstantValue>| {
        baked_build(&cache, &PipelineKey::new(lib, name, c)).expect(name)
    };
    let value_dim = NV * HV;
    let conv_slot = CONV_DIM * (KERNEL - 1);
    let rows = CheckpointRows(DRAFTS);
    let lane = DraftLanes::new(rows, 1)
        .and_then(|l| l.lanes().next())
        .expect("a lane");
    let record = CONV_DIM + 2 * NV;
    let layers: Vec<Layer> = (0..LAYERS)
        .map(|l| Layer {
            conv: shared_slice(&device, &fill(conv_slot, l)),
            ssm: shared_slice(&device, &fill(NV * HV * HK, l)),
            conv_lanes: shared_slice(&device, &fill(rows.rows() * conv_slot, l)),
            ssm_lanes: shared_slice(&device, &fill(2 * rows.rows() * record, l)),
        })
        .collect();
    let t_max = rows.rows();
    let qkv = shared_slice(&device, &fill(t_max * CONV_DIM, 7));
    let z = shared_slice(&device, &fill(t_max * value_dim, 13));
    let a = shared_slice(&device, &fill(t_max * NV, 1));
    let b = shared_slice(&device, &fill(t_max * NV, 3));
    let conv_w = shared_slice(&device, &fill(CONV_DIM * KERNEL, 19));
    let a_log = shared_slice(&device, &fill(NV, 2));
    let dt_bias = shared_slice(&device, &fill(NV, 5));
    let norm_w = shared_slice(&device, &fill(HV, 17));
    let unread = shared_slice(&device, &[0f32]);
    let conv_out = shared_slice(&device, &vec![0f32; t_max * CONV_DIM]);
    let o = shared_slice(&device, &vec![0f32; t_max * value_dim]);
    let out = shared_slice(&device, &vec![0f32; t_max * value_dim]);
    let si = shared_slice(&device, &GdnSeqSlots::plain(GdnSlot::new(0)).words());
    let cu = |t: i32| shared_slice(&device, &[0i32, t]);
    let (cu1, cu3) = (cu(1), cu(t_max as i32));
    let code = |start, checkpoint_rows| {
        let mut step = GdnStep {
            start,
            checkpoint_rows,
            records: RecordArea::First,
            lane: None,
        };
        step.lane = (!step.is_plain()).then_some(lane);
        shared_slice(&device, &[step.encode()])
    };
    let plain = code(GdnStart::Slot, CheckpointRows::NONE);
    let verify = |kept: u8| code(GdnStart::Checkpoint(kept), rows);
    let (verify1, verify3) = (verify(0), verify(DRAFTS));

    let scale = (HK as f32).powf(-0.5);
    let geometry = || {
        vec![
            ConstantValue::uint(0, NK as u32),
            ConstantValue::uint(1, NV as u32),
            ConstantValue::uint(2, HK as u32),
            ConstantValue::uint(3, HV as u32),
            ConstantValue::float(4, scale),
        ]
    };
    let decode = build("gdn_decode", "gdn_decode_f32", {
        let mut c = geometry();
        c.extend([
            ConstantValue::uint(5, KERNEL as u32),
            ConstantValue::float(6, 1e-6),
        ]);
        c
    });
    let conv = |drafts: u8| {
        build(
            "gdn_conv1d_varlen",
            "gdn_conv1d_varlen_f32",
            vec![
                ConstantValue::uint(0, CONV_DIM as u32),
                ConstantValue::uint(1, KERNEL as u32),
                ConstantValue::uint(5, u32::from(drafts)),
                ConstantValue::uint(6, CONV_BLOCK),
            ],
        )
    };
    let scan = |drafts: u8| {
        build(
            "gdn_scan_varlen",
            "gdn_scan_simd_f32",
            [vec![ConstantValue::uint(8, u32::from(drafts))], geometry()].concat(),
        )
    };
    let norm = |rows: usize| {
        build(
            "gdn_rms_norm_gated",
            "gdn_rms_norm_gated_f32",
            vec![
                ConstantValue::uint(0, HV as u32),
                ConstantValue::uint(1, rows as u32),
                ConstantValue::float(2, 1e-6),
            ],
        )
    };
    let (conv0, conv2, scan0, scan2) = (conv(0), conv(DRAFTS), scan(0), scan(DRAFTS));
    let (norm1, norm3, norm_one) = (norm(NV), norm(t_max * NV), norm(1));

    use Bind::*;
    let decode_cmd = || Dispatch {
        pso: &decode,
        binds: vec![
            (Shared(&out), 0),
            (Shared(&qkv), 1),
            (Shared(&z), 2),
            (Shared(&a), 3),
            (Shared(&b), 4),
            (Shared(&conv_w), 5),
            (Conv, 6),
            (Ssm, 7),
            (Shared(&cu1), 8),
            (Shared(&si), 9),
            (Shared(&plain), 10),
            (Shared(&a_log), 11),
            (Shared(&dt_bias), 12),
            (Shared(&norm_w), 13),
        ],
        grid: (1, NK, 1),
        threads: (1024, 1, 1),
    };
    // The conv, scan and norm of a plain row (`kept` None) or of a verify step's rows replaying
    // the previous verify step's `kept + 1` rows; a verify step's kernels take the lanes.
    let chain_cmds = |kept: Option<u8>| -> Vec<Dispatch> {
        let drafted = kept.is_some();
        let (t, cu, fresh) = match kept {
            None => (1, &cu1, &plain),
            Some(0) => (t_max, &cu3, &verify1),
            Some(_) => (t_max, &cu3, &verify3),
        };
        let (conv_pso, scan_pso) = match drafted {
            true => (&conv2, &scan2),
            false => (&conv0, &scan0),
        };
        let mut conv_binds = vec![
            (Shared(&conv_out), 0),
            (Shared(&qkv), 1),
            (Shared(&conv_w), 2),
            (Conv, 3),
            (Shared(cu), 4),
            (Shared(&si), 5),
            (Shared(fresh), 6),
        ];
        let mut scan_binds = vec![
            (Shared(&o), 0),
            (Shared(&conv_out), 1),
            (Shared(&a), 2),
            (Shared(&b), 3),
            (Ssm, 4),
            (Shared(cu), 5),
            (Shared(&si), 6),
            (Shared(fresh), 7),
            (Shared(&a_log), 8),
            (Shared(&dt_bias), 9),
        ];
        if drafted {
            conv_binds.extend([(Shared(&unread), 7), (ConvLanes, 8)]);
            scan_binds.extend([(Shared(&unread), 10), (SsmLanes, 11)]);
        }
        vec![
            Dispatch {
                pso: conv_pso,
                binds: conv_binds,
                grid: (1, CONV_DIM.div_ceil(256), t.div_ceil(CONV_BLOCK as usize)),
                threads: (1, 256, 1),
            },
            Dispatch {
                pso: scan_pso,
                binds: scan_binds,
                grid: (1, NV * HV / 4, 1),
                threads: (32, 4, 1),
            },
            Dispatch {
                pso: if t == 1 { &norm1 } else { &norm3 },
                binds: vec![
                    (Shared(&out), 0),
                    (Shared(&o), 1),
                    (Shared(&z), 2),
                    (Shared(&norm_w), 3),
                ],
                grid: (t * NV, 1, 1),
                threads: (256, 1, 1),
            },
        ]
    };
    let empty = vec![Dispatch {
        pso: &norm_one,
        binds: vec![
            (Shared(&out), 0),
            (Shared(&o), 1),
            (Shared(&z), 2),
            (Shared(&norm_w), 3),
        ],
        grid: (1, 1, 1),
        threads: (32, 1, 1),
    }];

    let time = |cmds: &[Dispatch]| -> f64 {
        let run = || {
            let mut batch = Mtl4DispatchBatch::begin(&device).expect("mtl4 batch");
            for i in 0..LAYERS * REPEATS {
                let layer = &layers[i % LAYERS];
                for d in cmds {
                    let binds: Vec<(&Buffer, usize)> = (d.binds.iter())
                        .map(|(b, ix)| {
                            let buf = match b {
                                Shared(buf) => *buf,
                                Conv => &layer.conv,
                                Ssm => &layer.ssm,
                                ConvLanes => &layer.conv_lanes,
                                SsmLanes => &layer.ssm_lanes,
                            };
                            (buf, *ix)
                        })
                        .collect();
                    batch.encode(d.pso, &binds, &[], &[], &[], size(d.grid), size(d.threads));
                    batch.barrier();
                }
            }
            let t0 = std::time::Instant::now();
            batch.commit(true);
            t0.elapsed().as_secs_f64() * 1e6 / (LAYERS * REPEATS) as f64
        };
        run();
        (0..3).map(|_| run()).fold(f64::MAX, f64::min)
    };

    println!("µs a GDN layer (x{LAYERS} layers a step), Qwen3.6 geometry, f32");
    let cells: [(&str, Vec<Dispatch>); 5] = [
        ("plain row, gdn_decode (1 dispatch)", vec![decode_cmd()]),
        ("plain row, conv+scan+norm (3)", chain_cmds(None)),
        ("verify 3 rows, 1 kept replayed (3)", chain_cmds(Some(0))),
        (
            "verify 3 rows, 3 kept replayed (3)",
            chain_cmds(Some(DRAFTS)),
        ),
        ("near-empty dispatch (1)", empty),
    ];
    for (name, cmds) in &cells {
        let us = time(cmds);
        println!(
            "  {name:<38} {us:8.1} µs a layer  {:7.2} ms a step",
            us * LAYERS as f64 / 1e3
        );
    }

    // What the fence before a dispatch costs, by the tape's classes: the near-empty dispatch
    // with none, an ordering-only barrier (`Fence::Ordered`) and a device-visible one
    // (`Fence::Coherent`, every fence of Qwen3.6's decode tapes).
    println!("µs a near-empty dispatch, by the fence before it ({DISPATCHES} a command buffer)");
    for (name, visibility) in [
        ("no fence", None),
        (
            "Ordered (visibility None)",
            Some(MTL4VisibilityOptions::None),
        ),
        (
            "Coherent (visibility Device)",
            Some(MTL4VisibilityOptions::Device),
        ),
    ] {
        let run = || {
            let mut batch = Mtl4DispatchBatch::begin(&device).expect("mtl4 batch");
            let binds = [(&out, 0), (&o, 1), (&z, 2), (&norm_w, 3)];
            for _ in 0..DISPATCHES {
                if let Some(v) = visibility {
                    batch
                        .encoder()
                        .barrierAfterEncoderStages_beforeEncoderStages_visibilityOptions(
                            MTLStages::Dispatch,
                            MTLStages::Dispatch,
                            v,
                        );
                }
                batch.encode(
                    &norm_one,
                    &binds,
                    &[],
                    &[],
                    &[],
                    size((1, 1, 1)),
                    size((32, 1, 1)),
                );
            }
            let t0 = std::time::Instant::now();
            batch.commit(true);
            t0.elapsed().as_secs_f64() * 1e6 / DISPATCHES as f64
        };
        run();
        let us = (0..3).map(|_| run()).fold(f64::MAX, f64::min);
        println!("  {name:<38} {us:8.2} µs");
    }
}

/// Near-empty dispatches a command buffer, for the fence cells.
const DISPATCHES: usize = 600;
