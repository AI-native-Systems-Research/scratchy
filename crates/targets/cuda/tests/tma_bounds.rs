// SPDX-License-Identifier: Apache-2.0
//! What a ThunderKittens TMA tile load and store do with a 64×64 bf16 box that runs past its
//! array's edge, swizzled and plain (`kernels/tma_box_probe.cu`), on the H100: the GKE probe's R1
//! cases (`/tmp/tk-probe/probe.cu` at c8d80fda; AI-native-Systems-Research/scratchy#846), the same
//! shapes, tile coordinates, sentinels and labels. The GEMV's and the narrow GEMM's choice of tiles
//! rests on these facts (`kernels/gemv_bf16.cu`), so this is the port's first gate.
//!
//! It asserts the measured behaviour, not a wish: the 8 cases that passed in the probe give 0 in
//! every count, and the 4 swizzled cases with columns past the edge give a non-zero
//! `past_edge_not_zero` (load) or `sentinel_overwritten` (store), with every in-bounds count 0. A
//! swizzled tile's tensor map rounds a row up to the 64-element swizzle chunk, so it reads and
//! writes past a row narrower than that. If the hardware or the pin ever changes either half, a
//! line says `DIFFERS FROM PROBE`.
//!
//! Each array sits at element offset `PAD` of a sentinel-filled allocation, as the probe's do, so a
//! swizzled overflow touches only the test's own padding. Built only under `nvcc`
//! (`required-features`); no GPU fails rather than skips.

use scratchy_target_cuda::kernels::{tma_box_probe_load, tma_box_probe_store};
use scratchy_target_cuda::runtime::{CudaDevice, DeviceBuffer};

/// A NaN sentinel around each array in global memory.
const SENT: u16 = 0x7FC1;
/// A NaN poison loaded into the tile before the box under test.
const POISON: u16 = 0x7FC2;
/// The box: 64×64 bf16.
const BOX: usize = 64;
/// Sentinel elements before and after each array.
const PAD: usize = 2 * BOX * BOX;

/// A finite, non-zero source value.
fn val_in(i: usize) -> u16 {
    0x4000 + (i % 0x3F00) as u16
}
/// An array element before a store.
fn val_old(i: usize) -> u16 {
    0x2000 + (i % 0x1000) as u16
}
/// The tile being stored, at `(r, c)`.
fn val_box(r: usize, c: usize) -> u16 {
    0x3000 + (r * BOX + c) as u16
}

fn h100() -> CudaDevice {
    CudaDevice::open()
        .unwrap_or_else(|e| panic!("the `nvcc` feature's tests need an sm_90 GPU (H100): {e}"))
}

fn up(device: &CudaDevice, host: &[u16]) -> DeviceBuffer {
    DeviceBuffer::upload(device, host).expect("upload")
}

/// One R1 case: the probe's label, shape and tile, and what the probe measured.
struct Case {
    swizzled: bool,
    rows: usize,
    cols: usize,
    tile: (usize, usize),
    label: &'static str,
    /// The probe saw a non-zero count past the edge (and 0 in bounds); otherwise every count 0.
    overflows: bool,
}

const PREDICTS: &str = "[header predicts FAIL: swizzled TMA rounds the inner dim up to 64 elements, types/global/tma.cuh]";

const LOADS: [Case; 6] = [
    Case {
        swizzled: true,
        rows: 1,
        cols: 1024,
        tile: (0, 3),
        label: "GEMV A row, k=1024",
        overflows: false,
    },
    Case {
        swizzled: true,
        rows: 100,
        cols: 64,
        tile: (1, 0),
        label: "partial M tile, NaN sentinels in memory past row 100",
        overflows: false,
    },
    Case {
        swizzled: true,
        rows: 1,
        cols: 8,
        tile: (0, 0),
        label: "GEMV test shape k=8",
        overflows: true,
    },
    Case {
        swizzled: true,
        rows: 40,
        cols: 24,
        tile: (0, 0),
        label: "rows and cols past the edge",
        overflows: true,
    },
    Case {
        swizzled: false,
        rows: 1,
        cols: 8,
        tile: (0, 0),
        label: "GEMV test shape k=8",
        overflows: false,
    },
    Case {
        swizzled: false,
        rows: 40,
        cols: 24,
        tile: (0, 0),
        label: "rows and cols past the edge",
        overflows: false,
    },
];

const STORES: [Case; 6] = [
    Case {
        swizzled: true,
        rows: 1,
        cols: 512,
        tile: (0, 2),
        label: "GEMV C row, n=512",
        overflows: false,
    },
    Case {
        swizzled: true,
        rows: 100,
        cols: 64,
        tile: (1, 0),
        label: "partial M tile, NaN sentinels in memory past row 100",
        overflows: false,
    },
    Case {
        swizzled: true,
        rows: 1,
        cols: 8,
        tile: (0, 0),
        label: "n=8",
        overflows: true,
    },
    Case {
        swizzled: true,
        rows: 40,
        cols: 24,
        tile: (0, 0),
        label: "rows and cols past the edge",
        overflows: true,
    },
    Case {
        swizzled: false,
        rows: 1,
        cols: 8,
        tile: (0, 0),
        label: "n=8",
        overflows: false,
    },
    Case {
        swizzled: false,
        rows: 40,
        cols: 24,
        tile: (0, 0),
        label: "rows and cols past the edge",
        overflows: false,
    },
];

/// The case's name, as the probe printed it.
fn name(kind: &str, c: &Case) -> String {
    let predicts = if c.overflows {
        format!(" {PREDICTS}")
    } else {
        String::new()
    };
    format!(
        "R1 {kind} {} [{},{}] tile ({},{}) {}{predicts}",
        if c.swizzled { "swizzled" } else { "plain" },
        c.rows,
        c.cols,
        c.tile.0,
        c.tile.1,
        c.label
    )
}

/// A `[rows, cols]` array at offset `PAD` of a sentinel-filled host buffer.
fn padded(rows: usize, cols: usize, fill: fn(usize) -> u16) -> Vec<u16> {
    let mut host = vec![SENT; PAD + rows * cols + PAD];
    for (i, v) in host[PAD..][..rows * cols].iter_mut().enumerate() {
        *v = fill(i);
    }
    host
}

/// Whether the measured counts match the probe's: 0 everywhere, or (when it overflows) 0 in bounds
/// and non-zero past the edge.
fn matches(c: &Case, in_bounds: usize, past: usize, other: usize) -> bool {
    in_bounds == 0 && other == 0 && if c.overflows { past > 0 } else { past == 0 }
}

fn run_load(device: &CudaDevice, c: &Case) -> bool {
    let host = padded(c.rows, c.cols, val_in);
    let src = up(device, &host);
    let poison = up(device, &vec![POISON; BOX * BOX]);
    let dump = up(device, &vec![0xEEEE; BOX * BOX]);
    // SAFETY: `src` holds PAD + rows·cols + PAD bf16 and the array starts at PAD, so a swizzled
    // overflow of its last row stays inside the allocation; `poison` and `dump` hold 4096 bf16;
    // cudaMalloc pointers are 256-byte aligned, and PAD·2 bytes keeps the array 16-byte aligned;
    // null is the default stream, which the download below waits for.
    let launched = unsafe {
        tma_box_probe_load(
            (
                src.ptr().cast::<u16>().add(PAD).cast(),
                poison.ptr(),
                dump.ptr(),
            ),
            (c.rows as u32, c.cols as u32),
            (c.tile.0 as u32, c.tile.1 as u32),
            c.swizzled,
            std::ptr::null_mut(),
        )
    };
    let got = dump.download().expect("download the dump");
    if let Err(e) = launched {
        eprintln!("{}: launch failed: {e} DIFFERS FROM PROBE", name("load", c));
        return false;
    }
    let (mut in_bad, mut past, mut poison_left) = (0usize, 0usize, 0usize);
    let mut first = String::new();
    for r in 0..BOX {
        for col in 0..BOX {
            let (gr, gc) = (c.tile.0 * BOX + r, c.tile.1 * BOX + col);
            let inside = gr < c.rows && gc < c.cols;
            let want = if inside {
                host[PAD + gr * c.cols + gc]
            } else {
                0
            };
            let v = got[r * BOX + col];
            poison_left += usize::from(v == POISON);
            if v != want {
                if inside {
                    in_bad += 1;
                } else {
                    past += 1;
                }
                if first.is_empty() {
                    first = format!(" first=({r},{col} got 0x{v:04X} want 0x{want:04X})");
                }
            }
        }
    }
    let ok = matches(c, in_bad, past, poison_left);
    eprintln!(
        "{}: in_bounds_wrong={in_bad} past_edge_not_zero={past} smem_poison_left={poison_left}\
         {first} {}",
        name("load", c),
        if ok {
            "MATCHES PROBE"
        } else {
            "DIFFERS FROM PROBE"
        }
    );
    ok
}

fn run_store(device: &CudaDevice, c: &Case) -> bool {
    let host = padded(c.rows, c.cols, val_old);
    let dst = up(device, &host);
    let pattern: Vec<u16> = (0..BOX * BOX).map(|i| val_box(i / BOX, i % BOX)).collect();
    let pattern = up(device, &pattern);
    // SAFETY: as `run_load`'s: the array starts at PAD of its sentinel-padded allocation, so a
    // swizzled overflow stays inside it; `pattern` holds 4096 bf16.
    let launched = unsafe {
        tma_box_probe_store(
            (pattern.ptr(), dst.ptr().cast::<u16>().add(PAD).cast()),
            (c.rows as u32, c.cols as u32),
            (c.tile.0 as u32, c.tile.1 as u32),
            c.swizzled,
            std::ptr::null_mut(),
        )
    };
    let got = dst.download().expect("download the array");
    if let Err(e) = launched {
        eprintln!(
            "{}: launch failed: {e} DIFFERS FROM PROBE",
            name("store", c)
        );
        return false;
    }
    let (mut box_bad, mut changed, mut sentinel) = (0usize, 0usize, 0usize);
    let mut first = String::new();
    for (i, &v) in got.iter().enumerate() {
        let array = (PAD..PAD + c.rows * c.cols).contains(&i);
        let (want, kind) = if !array {
            (SENT, 0)
        } else {
            let a = i - PAD;
            let (r, col) = (a / c.cols, a % c.cols);
            let (br, bc) = (
                r as isize - (c.tile.0 * BOX) as isize,
                col as isize - (c.tile.1 * BOX) as isize,
            );
            if (0..BOX as isize).contains(&br) && (0..BOX as isize).contains(&bc) {
                (val_box(br as usize, bc as usize), 2)
            } else {
                (host[i], 1)
            }
        };
        if v != want {
            match kind {
                0 => sentinel += 1,
                1 => changed += 1,
                _ => box_bad += 1,
            }
            if first.is_empty() {
                let off = i as isize - PAD as isize;
                first = format!(" first=(elem {off} got 0x{v:04X} want 0x{want:04X})");
            }
        }
    }
    let ok = matches(c, box_bad, sentinel, changed);
    eprintln!(
        "{}: box_in_bounds_wrong={box_bad} array_outside_box_changed={changed} \
         sentinel_overwritten={sentinel}{first} {}",
        name("store", c),
        if ok {
            "MATCHES PROBE"
        } else {
            "DIFFERS FROM PROBE"
        }
    );
    ok
}

/// The probe's 12 R1 cases, every line printed before the test fails.
#[test]
fn tma_boxes_past_the_edge_match_the_probe() {
    let device = h100();
    let mut pass = true;
    for c in &LOADS {
        pass &= run_load(&device, c);
    }
    for c in &STORES {
        pass &= run_store(&device, c);
    }
    assert!(
        pass,
        "a TMA box case differs from the GKE probe's R1: see the lines above (STOP)"
    );
}
