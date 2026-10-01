// SPDX-License-Identifier: Apache-2.0
//! TCP-based store for inter-process NCCL initialization.
//!
//! When using an external launcher (torchrun, mpirun, SLURM), each process
//! needs to exchange the NCCL unique ID. This module provides a simple
//! synchronous TCP store for that purpose.
//!
//! Rank 0 acts as the server:
//! - Generates the NCCL unique ID
//! - Listens on `MASTER_ADDR:MASTER_PORT`
//! - Sends the 128-byte ID to each connecting rank
//!
//! Ranks 1..N-1 connect to rank 0 and receive the ID.
//!
//! The rest of the multi-node rendezvous (memory all-reduce, control channel)
//! is device-free and lives in `scratchy_serving_transport::tcp_store`.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};

use anyhow::{Context, Result};
use tracing::info;

/// Size of an NCCL unique ID in bytes.
const NCCL_ID_BYTES: usize = 128;

/// Exchange the NCCL unique ID across processes via TCP.
///
/// - Rank 0: generates a new NCCL ID, accepts `world_size - 1` connections,
///   and sends the ID to each.
/// - Rank N > 0: connects to `master_addr:master_port` and receives the ID.
///
/// Returns the raw 128-byte NCCL unique ID (same value on all ranks).
pub fn exchange_nccl_id(
    rank: usize,
    world_size: usize,
    master_addr: &str,
    master_port: u16,
) -> Result<[core::ffi::c_char; NCCL_ID_BYTES]> {
    if rank == 0 {
        exchange_nccl_id_rank0(world_size, master_addr, master_port)
    } else {
        exchange_nccl_id_worker(master_addr, master_port)
    }
}

/// Rank 0: generate NCCL ID and distribute to all other ranks.
fn exchange_nccl_id_rank0(
    world_size: usize,
    master_addr: &str,
    master_port: u16,
) -> Result<[core::ffi::c_char; NCCL_ID_BYTES]> {
    // Generate the NCCL unique ID.
    let nccl_id = crate::NcclId::new().context("failed to generate NCCL unique ID")?;
    let id_bytes = *nccl_id.raw();

    if world_size == 1 {
        return Ok(id_bytes);
    }

    let bind_addr = format!("{master_addr}:{master_port}");
    let listener = TcpListener::bind(&bind_addr)
        .with_context(|| format!("rank 0: failed to bind to {bind_addr}"))?;
    info!(
        "Rank 0: listening on {} for {} worker(s)",
        bind_addr,
        world_size - 1
    );

    // Accept connections from ranks 1..world_size-1.
    // Cast c_char bytes to u8 for network I/O.
    let id_as_u8: &[u8; NCCL_ID_BYTES] = unsafe { &*(&id_bytes as *const _ as *const _) };
    for i in 1..world_size {
        let (mut stream, peer_addr) = listener
            .accept()
            .with_context(|| format!("rank 0: failed to accept connection {i}"))?;
        info!(
            "Rank 0: accepted connection from {} (rank {})",
            peer_addr, i
        );
        stream
            .write_all(id_as_u8)
            .with_context(|| format!("rank 0: failed to send NCCL ID to rank {i}"))?;
    }

    Ok(id_bytes)
}

/// Rank N > 0: connect to rank 0 and receive the NCCL unique ID.
fn exchange_nccl_id_worker(
    master_addr: &str,
    master_port: u16,
) -> Result<[core::ffi::c_char; NCCL_ID_BYTES]> {
    let addr = format!("{master_addr}:{master_port}");

    // Retry connection with backoff — rank 0 may not be listening yet.
    let mut stream = None;
    for attempt in 0..30 {
        match TcpStream::connect(&addr) {
            Ok(s) => {
                stream = Some(s);
                break;
            }
            Err(e) => {
                if attempt < 29 {
                    let delay = std::time::Duration::from_millis(500);
                    info!(
                        "Worker: connection to {} failed (attempt {}): {}, retrying in {:?}",
                        addr,
                        attempt + 1,
                        e,
                        delay
                    );
                    std::thread::sleep(delay);
                } else {
                    return Err(e).with_context(|| {
                        format!("worker: failed to connect to {addr} after 30 attempts")
                    });
                }
            }
        }
    }
    let mut stream = stream.unwrap();

    let mut buf = [0u8; NCCL_ID_BYTES];
    stream
        .read_exact(&mut buf)
        .context("worker: failed to read NCCL ID from rank 0")?;

    // Reinterpret u8 bytes as c_char.
    let id_bytes: [core::ffi::c_char; NCCL_ID_BYTES] = unsafe { std::mem::transmute(buf) };
    Ok(id_bytes)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn pick_unused_port() -> u16 {
        std::net::TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port()
    }

    #[test]
    fn test_nccl_id_exchange_simulated() {
        // We can't call real NCCL ID generation without CUDA, so test the TCP
        // plumbing with a mock: rank 0 serves fixed bytes, workers receive them.
        let world_size = 3;
        let port: u16 = pick_unused_port();

        // Simulate by directly testing the TCP layer with raw bytes.
        let test_bytes = [42i8; NCCL_ID_BYTES];

        let handles: Vec<_> = (0..world_size)
            .map(|rank| {
                std::thread::spawn(move || -> [u8; NCCL_ID_BYTES] {
                    if rank == 0 {
                        // Server: send test_bytes to each worker.
                        let bind_addr = format!("127.0.0.1:{port}");
                        let listener = TcpListener::bind(&bind_addr).unwrap();
                        let id_as_u8: &[u8; NCCL_ID_BYTES] =
                            unsafe { &*(&test_bytes as *const _ as *const _) };
                        for _ in 1..world_size {
                            let (mut stream, _) = listener.accept().unwrap();
                            stream.write_all(id_as_u8).unwrap();
                        }
                        unsafe { std::mem::transmute::<[i8; 128], [u8; 128]>(test_bytes) }
                    } else {
                        // Worker: connect and receive.
                        let addr = format!("127.0.0.1:{port}");
                        let mut stream = loop {
                            match TcpStream::connect(&addr) {
                                Ok(s) => break s,
                                Err(_) => std::thread::sleep(std::time::Duration::from_millis(50)),
                            }
                        };
                        let mut buf = [0u8; NCCL_ID_BYTES];
                        stream.read_exact(&mut buf).unwrap();
                        buf
                    }
                })
            })
            .collect();

        let expected: [u8; NCCL_ID_BYTES] = unsafe { std::mem::transmute([42i8; NCCL_ID_BYTES]) };
        for handle in handles {
            let result = handle.join().unwrap();
            assert_eq!(result, expected);
        }
    }
}
