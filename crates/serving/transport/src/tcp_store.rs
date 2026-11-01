// SPDX-License-Identifier: Apache-2.0
//! TCP rendezvous and control plane for multi-node serving.
//!
//! Plain `std::net` TCP with no device code. Rank 0 listens; ranks 1..N-1
//! connect to `MASTER_ADDR`:
//! - [`allreduce_min`] (`master_port + 1`): each rank sends its available
//!   memory to rank 0, which computes the minimum and broadcasts it back.
//! - [`TcpControlChannel`] (`master_port + 2`): persistent length-prefixed
//!   frames from rank 0 to every follower.
//!
//! `master_port` itself carries the NCCL unique-ID exchange, which needs the
//! CUDA runtime and lives in `scratchy_target_cuda::tcp_store`.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};

use anyhow::{Context, Result};
use tracing::info;

/// [`allreduce_min`] listens on `master_port + ALLREDUCE_PORT_OFFSET`. The
/// NCCL-ID exchange owns `master_port` itself.
pub const ALLREDUCE_PORT_OFFSET: u16 = 1;

/// [`TcpControlChannel`] listens on `master_port + CONTROL_CHANNEL_PORT_OFFSET`.
pub const CONTROL_CHANNEL_PORT_OFFSET: u16 = 2;

/// All-reduce MIN of a `usize` value across all ranks via TCP.
///
/// - Rank 0: accepts `world_size - 1` connections, reads each rank's value,
///   computes the minimum (including its own), and sends the result back.
/// - Rank N > 0: connects to rank 0, sends its value, receives the minimum.
///
/// Returns the minimum value across all ranks.
pub fn allreduce_min(
    rank: usize,
    world_size: usize,
    value: usize,
    master_addr: &str,
    master_port: u16,
) -> Result<usize> {
    if world_size == 1 {
        return Ok(value);
    }

    let port = master_port + ALLREDUCE_PORT_OFFSET;

    if rank == 0 {
        allreduce_min_rank0(world_size, value, master_addr, port)
    } else {
        allreduce_min_worker(value, master_addr, port)
    }
}

/// Rank 0: collect values, compute min, broadcast result.
fn allreduce_min_rank0(
    world_size: usize,
    local_value: usize,
    master_addr: &str,
    port: u16,
) -> Result<usize> {
    let bind_addr = format!("{master_addr}:{port}");
    let listener = TcpListener::bind(&bind_addr)
        .with_context(|| format!("rank 0: failed to bind to {bind_addr} for allreduce"))?;

    let mut min_value = local_value;
    let mut streams = Vec::with_capacity(world_size - 1);

    for i in 1..world_size {
        let (mut stream, _) = listener
            .accept()
            .with_context(|| format!("rank 0: failed to accept allreduce connection {i}"))?;

        let mut buf = [0u8; 8];
        stream
            .read_exact(&mut buf)
            .with_context(|| format!("rank 0: failed to read value from rank {i}"))?;
        let remote_value = usize::from_le_bytes(buf);
        min_value = min_value.min(remote_value);
        streams.push(stream);
    }

    info!(
        "Rank 0: allreduce MIN = {} bytes ({:.1} GB)",
        min_value,
        min_value as f64 / (1024.0 * 1024.0 * 1024.0)
    );

    // Broadcast the minimum back to all workers.
    let result_bytes = min_value.to_le_bytes();
    for (i, mut stream) in streams.into_iter().enumerate() {
        stream
            .write_all(&result_bytes)
            .with_context(|| format!("rank 0: failed to send min to rank {}", i + 1))?;
    }

    Ok(min_value)
}

/// Rank N > 0: send value to rank 0, receive the minimum.
fn allreduce_min_worker(value: usize, master_addr: &str, port: u16) -> Result<usize> {
    let addr = format!("{master_addr}:{port}");

    let mut stream = connect_within(&addr, CONNECT_BUDGET)
        .with_context(|| format!("worker: failed to connect to {addr} for allreduce"))?;

    // Send our value.
    stream
        .write_all(&value.to_le_bytes())
        .context("worker: failed to send value for allreduce")?;

    // Receive the minimum.
    let mut buf = [0u8; 8];
    stream
        .read_exact(&mut buf)
        .context("worker: failed to read allreduce result")?;

    Ok(usize::from_le_bytes(buf))
}

// ---------------------------------------------------------------------------
// Persistent TCP control channel (multi-node scheduler output broadcast)
// ---------------------------------------------------------------------------

/// Persistent TCP control channel for broadcasting scheduler output from rank 0
/// to all follower nodes. Unlike the one-shot NCCL-ID exchange and
/// [`allreduce_min`], this maintains persistent connections for the entire
/// lifetime of the inference server.
///
/// Listens on `master_port + CONTROL_CHANNEL_PORT_OFFSET`.
pub struct TcpControlChannel {
    /// Rank 0 holds connections to all followers.
    /// Followers hold a single connection to rank 0.
    role: ChannelRole,
}

enum ChannelRole {
    /// Rank 0: one TCP stream per follower (indexed 0..world_size-2).
    Leader { streams: Vec<TcpStream> },
    /// Rank > 0: single TCP stream to rank 0.
    Follower { stream: TcpStream },
}

impl TcpControlChannel {
    /// Establish the control channel.
    ///
    /// - Rank 0: listens and accepts `world_size - 1` connections.
    /// - Rank > 0: connects to rank 0 with retry.
    pub fn establish(
        rank: usize,
        world_size: usize,
        master_addr: &str,
        master_port: u16,
    ) -> Result<Self> {
        let port = master_port + CONTROL_CHANNEL_PORT_OFFSET;

        if rank == 0 {
            let bind_addr = format!("{master_addr}:{port}");
            let listener = TcpListener::bind(&bind_addr).with_context(|| {
                format!("rank 0: failed to bind control channel on {bind_addr}")
            })?;
            info!(
                "Control channel: rank 0 listening on {} for {} follower(s)",
                bind_addr,
                world_size - 1
            );

            // ⭐ SEATED BY THE RANK EACH FOLLOWER STATES, NOT BY ARRIVAL. Followers connect in
            // whatever order they start, so `streams[r - 1]` is rank `r` only because each one says
            // which rank it is ([`HELLO`]); in accept order, every per-follower fact (a ready
            // frame's landing address) would belong to whichever rank happened to connect first.
            let mut seats: Vec<Option<TcpStream>> = (1..world_size).map(|_| None).collect();
            for i in 1..world_size {
                let (mut stream, peer) = listener.accept().with_context(|| {
                    format!("rank 0: failed to accept control channel connection {i}")
                })?;
                // Disable Nagle's algorithm for low-latency framing.
                stream.set_nodelay(true).ok();
                let mut hello = [0u8; HELLO_FRAME];
                stream.read_exact(&mut hello).with_context(|| {
                    format!("rank 0: the follower from {peer} never said its rank")
                })?;
                let word = |at: usize| {
                    u32::from_le_bytes([hello[at], hello[at + 1], hello[at + 2], hello[at + 3]])
                        as usize
                };
                let (their_rank, their_world) = (word(1), word(5));
                if hello[0] != HELLO {
                    anyhow::bail!(
                        "rank 0: the follower from {peer} opened with {:#04x}, not its rank",
                        hello[0]
                    );
                }
                if their_world != world_size {
                    anyhow::bail!(
                        "rank 0: the follower from {peer} runs a world of {their_world}, this one \
                         is {world_size}"
                    );
                }
                let seat = their_rank
                    .checked_sub(1)
                    .and_then(|s| seats.get_mut(s))
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "rank 0: the follower from {peer} says it is rank {their_rank}, not a \
                             follower of a world of {world_size}"
                        )
                    })?;
                if seat.is_some() {
                    anyhow::bail!("rank 0: two followers say they are rank {their_rank}");
                }
                info!(
                    "Control channel: accepted follower rank {} from {}",
                    their_rank, peer
                );
                *seat = Some(stream);
            }
            let streams = seats
                .into_iter()
                .collect::<Option<Vec<TcpStream>>>()
                .expect("every follower seat filled: world_size - 1 accepts, no rank twice");

            Ok(Self {
                role: ChannelRole::Leader { streams },
            })
        } else {
            let addr = format!("{master_addr}:{port}");
            let mut stream = connect_within(&addr, CONNECT_BUDGET)?;
            stream.set_nodelay(true).ok();
            let mut hello = [0u8; HELLO_FRAME];
            hello[0] = HELLO;
            hello[1..5].copy_from_slice(
                &u32::try_from(rank)
                    .context("control channel: a rank past u32")?
                    .to_le_bytes(),
            );
            hello[5..9].copy_from_slice(
                &u32::try_from(world_size)
                    .context("control channel: a world past u32")?
                    .to_le_bytes(),
            );
            stream
                .write_all(&hello)
                .with_context(|| format!("rank {rank}: failed to say its rank to {addr}"))?;
            info!("Control channel: rank {} connected to {}", rank, addr);
            Ok(Self {
                role: ChannelRole::Follower { stream },
            })
        }
    }

    /// Broadcast a length-prefixed frame from rank 0 to all followers.
    ///
    /// Only callable on rank 0 (leader). Sends a 4-byte big-endian length
    /// prefix followed by the payload to each follower.
    pub fn broadcast(&mut self, data: &[u8]) -> Result<()> {
        match &mut self.role {
            ChannelRole::Leader { streams } => {
                let len = (data.len() as u32).to_be_bytes();
                for (i, stream) in streams.iter_mut().enumerate() {
                    stream.write_all(&len).with_context(|| {
                        format!("control channel: failed to write len to follower {}", i + 1)
                    })?;
                    stream.write_all(data).with_context(|| {
                        format!(
                            "control channel: failed to write data to follower {}",
                            i + 1
                        )
                    })?;
                    stream.flush().with_context(|| {
                        format!("control channel: failed to flush to follower {}", i + 1)
                    })?;
                }
                Ok(())
            }
            ChannelRole::Follower { .. } => {
                anyhow::bail!("broadcast() called on follower (rank > 0)")
            }
        }
    }

    /// Receive a length-prefixed frame from rank 0.
    ///
    /// Only callable on followers (rank > 0). Blocks until a full frame is
    /// received. Returns the raw payload bytes.
    pub fn recv(&mut self) -> Result<Vec<u8>> {
        match &mut self.role {
            ChannelRole::Follower { stream } => {
                let mut len_buf = [0u8; 4];
                stream
                    .read_exact(&mut len_buf)
                    .context("control channel: failed to read frame length")?;
                let len = u32::from_be_bytes(len_buf) as usize;

                let mut buf = vec![0u8; len];
                stream
                    .read_exact(&mut buf)
                    .context("control channel: failed to read frame data")?;
                Ok(buf)
            }
            ChannelRole::Leader { .. } => {
                anyhow::bail!("recv() called on leader (rank 0)")
            }
        }
    }
}

/// The byte a follower's ready frame opens with ([`TcpControlChannel::report_ready`]).
const READY: u8 = 0x52;

/// A ready frame: [`READY`], whether a landing address follows (0 or 1), and the address (LE).
const READY_FRAME: usize = 10;

impl TcpControlChannel {
    /// A follower tells its leader it is ready — the one frame that flows follower → leader.
    ///
    /// A tensor-parallel rank whose collectives run card to card (spyre's p2p RDMA) must not
    /// issue its first send before every peer has mapped the link and programmed the window it
    /// lands in; the leader holds its first step until every follower has said so.
    ///
    /// `landing` is the device address this rank's peers land their sends at, for a world whose
    /// collectives land in a receive window: the leader checks every rank names the same one
    /// (each send names its own window's address for its peer's). `None` for a world without.
    pub fn report_ready(&mut self, landing: Option<u64>) -> Result<()> {
        let mut frame = [0u8; READY_FRAME];
        frame[0] = READY;
        if let Some(a) = landing {
            frame[1] = 1;
            frame[2..].copy_from_slice(&a.to_le_bytes());
        }
        match &mut self.role {
            ChannelRole::Follower { stream } => stream
                .write_all(&frame)
                .and_then(|()| stream.flush())
                .context("control channel: failed to report ready"),
            ChannelRole::Leader { .. } => {
                anyhow::bail!("report_ready() called on leader (rank 0)")
            }
        }
    }

    /// The leader blocks until every follower has [`report_ready`](Self::report_ready), and
    /// returns each follower's landing address, in rank order (rank 1 first).
    pub fn await_ready(&mut self) -> Result<Vec<Option<u64>>> {
        match &mut self.role {
            ChannelRole::Leader { streams } => {
                let mut landings = Vec::with_capacity(streams.len());
                for (i, stream) in streams.iter_mut().enumerate() {
                    let mut frame = [0u8; READY_FRAME];
                    stream.read_exact(&mut frame).with_context(|| {
                        format!("control channel: follower {} never reported ready", i + 1)
                    })?;
                    if frame[0] != READY {
                        anyhow::bail!(
                            "control channel: follower {} sent {:#04x}, not ready",
                            i + 1,
                            frame[0]
                        );
                    }
                    let mut address = [0u8; 8];
                    address.copy_from_slice(&frame[2..]);
                    landings.push(match frame[1] {
                        0 => None,
                        1 => Some(u64::from_le_bytes(address)),
                        other => anyhow::bail!(
                            "control channel: follower {}'s ready frame says {other:#04x} for \
                             whether a landing address follows",
                            i + 1
                        ),
                    });
                }
                Ok(landings)
            }
            ChannelRole::Follower { .. } => {
                anyhow::bail!("await_ready() called on follower (rank > 0)")
            }
        }
    }
}

/// A follower's first frame: this tag, then its rank and its world size, each a little-endian u32.
const HELLO: u8 = 0x48;
const HELLO_FRAME: usize = 9;

/// ⭐ HOW LONG A FOLLOWER WAITS FOR RANK 0 TO LISTEN. A follower is started before rank 0 (rank 0
/// waits for all of them), so it must outlast rank 0's whole launch-to-listen time plus the launch
/// stagger, at every world size — a fixed attempt count was 15 s, and measured on a 4-card pod
/// (#1009 run 17) rank 1 gave up 0.5 s before rank 0's listener opened. Long enough for any startup;
/// a rank 0 that never comes still ends the follower, by name, instead of leaving it waiting forever.
const CONNECT_BUDGET: std::time::Duration = std::time::Duration::from_secs(300);

/// How often a waiting follower retries.
const CONNECT_POLL: std::time::Duration = std::time::Duration::from_millis(500);

/// Connect to `addr`, retrying every [`CONNECT_POLL`] until it accepts or `budget` has passed.
fn connect_within(addr: &str, budget: std::time::Duration) -> Result<TcpStream> {
    let start = std::time::Instant::now();
    let mut attempt = 0u32;
    loop {
        attempt += 1;
        match TcpStream::connect(addr) {
            Ok(s) => return Ok(s),
            Err(e) if start.elapsed() + CONNECT_POLL < budget => {
                info!(
                    "Control channel: connection to {} failed (attempt {}): {}, retrying in {:?}",
                    addr, attempt, e, CONNECT_POLL
                );
                std::thread::sleep(CONNECT_POLL);
            }
            Err(e) => {
                return Err(e).with_context(|| {
                    format!(
                        "control channel: failed to connect to {addr} within {budget:?} \
                         ({attempt} attempts) — rank 0 never listened"
                    )
                });
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// A follower started before rank 0 connects once rank 0 listens, however late inside the
    /// budget — here rank 0 binds two polls after the follower starts trying.
    #[test]
    fn a_follower_waits_for_a_late_listener() {
        let port = TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        let addr = format!("127.0.0.1:{port}");
        let late = std::thread::spawn(move || {
            std::thread::sleep(CONNECT_POLL * 2);
            let l = TcpListener::bind(("127.0.0.1", port)).unwrap();
            l.accept().map(|_| ())
        });
        connect_within(&addr, std::time::Duration::from_secs(10)).unwrap();
        late.join().unwrap().unwrap();
    }

    /// Followers are seated by the rank they state, whatever order they connect in: rank 3
    /// connects first here, and its ready frame is still the third entry.
    #[test]
    fn followers_are_seated_by_rank_not_arrival() {
        let port = master_port_binding(CONTROL_CHANNEL_PORT_OFFSET);
        let landing = |rank: usize| Some(0x1000 * rank as u64);
        let leader = std::thread::spawn(move || {
            let mut ch = TcpControlChannel::establish(0, 4, "127.0.0.1", port).unwrap();
            ch.await_ready().unwrap()
        });
        for rank in [3, 1, 2] {
            let mut ch = TcpControlChannel::establish(rank, 4, "127.0.0.1", port).unwrap();
            ch.report_ready(landing(rank)).unwrap();
            std::mem::forget(ch);
        }
        assert_eq!(leader.join().unwrap(), [landing(1), landing(2), landing(3)]);
    }

    /// A follower of another world size is refused by name — the bake and the load must agree.
    #[test]
    fn a_follower_of_another_world_is_refused() {
        let port = master_port_binding(CONTROL_CHANNEL_PORT_OFFSET);
        let leader = std::thread::spawn(move || {
            TcpControlChannel::establish(0, 2, "127.0.0.1", port)
                .err()
                .map(|e| format!("{e:#}"))
        });
        let _follower = TcpControlChannel::establish(1, 4, "127.0.0.1", port).unwrap();
        let err = leader
            .join()
            .unwrap()
            .expect("a world-of-4 follower refused");
        assert!(err.contains("world of 4"), "{err}");
    }

    /// A rank 0 that never listens ends the wait at the budget, by name.
    #[test]
    fn a_follower_gives_up_at_its_budget() {
        let port = TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        let err = connect_within(&format!("127.0.0.1:{port}"), CONNECT_POLL * 2).unwrap_err();
        assert!(
            format!("{err:#}").contains("rank 0 never listened"),
            "{err:#}"
        );
    }

    /// The leader's `await_ready` returns only after every follower reported ready, with each
    /// follower's landing address in rank order, and a broadcast still reaches each follower
    /// afterwards.
    #[test]
    fn the_leader_waits_for_every_follower_to_report_ready() {
        let port = master_port_binding(CONTROL_CHANNEL_PORT_OFFSET);
        let landing = |rank: usize| (rank == 1).then_some(0x4_c842_2800_u64);
        let followers: Vec<_> = (1..3)
            .map(|rank| {
                std::thread::spawn(move || {
                    let mut ch = TcpControlChannel::establish(rank, 3, "127.0.0.1", port).unwrap();
                    ch.report_ready(landing(rank)).unwrap();
                    assert!(ch.await_ready().is_err(), "a follower awaits no one");
                    ch.recv().unwrap()
                })
            })
            .collect();
        let mut leader = TcpControlChannel::establish(0, 3, "127.0.0.1", port).unwrap();
        assert_eq!(leader.await_ready().unwrap(), [landing(1), landing(2)]);
        assert!(
            leader.report_ready(None).is_err(),
            "a leader reports to no one"
        );
        leader.broadcast(b"step").unwrap();
        for f in followers {
            assert_eq!(f.join().unwrap(), b"step");
        }
    }

    /// A `master_port` whose `master_port + offset`, the port actually bound,
    /// is free. Probing the base instead collides across parallel tests: the
    /// OS hands out adjacent ports, and one test's base is another's `+ 1`.
    fn master_port_binding(offset: u16) -> u16 {
        std::net::TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port()
            - offset
    }

    #[test]
    fn test_allreduce_min_single_rank() {
        let result = allreduce_min(0, 1, 42, "127.0.0.1", 0).unwrap();
        assert_eq!(result, 42);
    }

    #[test]
    fn test_allreduce_min_multi_rank() {
        let world_size = 4;
        let values = [100usize, 50, 200, 25];
        let expected_min = 25;

        let port: u16 = master_port_binding(ALLREDUCE_PORT_OFFSET);

        let handles: Vec<_> = (0..world_size)
            .map(|rank| {
                let value = values[rank];
                std::thread::spawn(move || {
                    allreduce_min(rank, world_size, value, "127.0.0.1", port).unwrap()
                })
            })
            .collect();

        for handle in handles {
            let result = handle.join().unwrap();
            assert_eq!(result, expected_min);
        }
    }

    #[test]
    fn test_allreduce_min_two_ranks() {
        let port: u16 = master_port_binding(ALLREDUCE_PORT_OFFSET);
        let handles: Vec<_> = (0..2)
            .map(|rank| {
                let value = if rank == 0 { 1000 } else { 500 };
                std::thread::spawn(move || {
                    allreduce_min(rank, 2, value, "127.0.0.1", port).unwrap()
                })
            })
            .collect();
        for handle in handles {
            assert_eq!(handle.join().unwrap(), 500);
        }
    }

    #[test]
    fn test_allreduce_min_identical_values() {
        let port: u16 = master_port_binding(ALLREDUCE_PORT_OFFSET);
        let world_size = 3;
        let handles: Vec<_> = (0..world_size)
            .map(|rank| {
                std::thread::spawn(move || {
                    allreduce_min(rank, world_size, 42, "127.0.0.1", port).unwrap()
                })
            })
            .collect();
        for handle in handles {
            assert_eq!(handle.join().unwrap(), 42);
        }
    }

    #[test]
    fn test_allreduce_min_rank0_has_minimum() {
        // Ensure rank 0's own value is included in the min computation.
        let port: u16 = master_port_binding(ALLREDUCE_PORT_OFFSET);
        let world_size = 3;
        // Rank 0 has the smallest value.
        let values = [1usize, 100, 200];
        let handles: Vec<_> = (0..world_size)
            .map(|rank| {
                let value = values[rank];
                std::thread::spawn(move || {
                    allreduce_min(rank, world_size, value, "127.0.0.1", port).unwrap()
                })
            })
            .collect();
        for handle in handles {
            assert_eq!(handle.join().unwrap(), 1);
        }
    }

    #[test]
    fn test_allreduce_min_large_world_size() {
        let world_size = 8;
        let port: u16 = master_port_binding(ALLREDUCE_PORT_OFFSET);
        let handles: Vec<_> = (0..world_size)
            .map(|rank| {
                // Rank 5 has the minimum.
                let value = if rank == 5 { 7 } else { 1000 + rank };
                std::thread::spawn(move || {
                    allreduce_min(rank, world_size, value, "127.0.0.1", port).unwrap()
                })
            })
            .collect();
        for handle in handles {
            assert_eq!(handle.join().unwrap(), 7);
        }
    }

    #[test]
    fn test_allreduce_min_zero_value() {
        let port: u16 = master_port_binding(ALLREDUCE_PORT_OFFSET);
        let handles: Vec<_> = (0..2)
            .map(|rank| {
                let value = if rank == 0 { 0 } else { 100 };
                std::thread::spawn(move || {
                    allreduce_min(rank, 2, value, "127.0.0.1", port).unwrap()
                })
            })
            .collect();
        for handle in handles {
            assert_eq!(handle.join().unwrap(), 0);
        }
    }

    #[test]
    fn test_allreduce_min_max_usize() {
        // Edge case: usize::MAX should be representable.
        let port: u16 = master_port_binding(ALLREDUCE_PORT_OFFSET);
        let handles: Vec<_> = (0..2)
            .map(|rank| {
                let value = if rank == 0 {
                    usize::MAX
                } else {
                    usize::MAX - 1
                };
                std::thread::spawn(move || {
                    allreduce_min(rank, 2, value, "127.0.0.1", port).unwrap()
                })
            })
            .collect();
        for handle in handles {
            assert_eq!(handle.join().unwrap(), usize::MAX - 1);
        }
    }

    #[test]
    fn test_allreduce_sequential_calls() {
        // Two sequential allreduce rounds on different ports (simulating
        // NCCL ID exchange then memory coordination).
        let port1: u16 = master_port_binding(ALLREDUCE_PORT_OFFSET);
        let port2: u16 = master_port_binding(ALLREDUCE_PORT_OFFSET);
        let handles: Vec<_> = (0..2)
            .map(|rank| {
                std::thread::spawn(move || {
                    let v1 = allreduce_min(rank, 2, 100 + rank, "127.0.0.1", port1).unwrap();
                    let v2 = allreduce_min(rank, 2, 200 + rank, "127.0.0.1", port2).unwrap();
                    (v1, v2)
                })
            })
            .collect();
        for handle in handles {
            let (v1, v2) = handle.join().unwrap();
            assert_eq!(v1, 100); // min(100, 101)
            assert_eq!(v2, 200); // min(200, 201)
        }
    }

    #[test]
    fn test_control_channel_basic() {
        let port: u16 = master_port_binding(CONTROL_CHANNEL_PORT_OFFSET);
        let world_size = 2;

        let leader = std::thread::spawn(move || {
            let mut ch = TcpControlChannel::establish(0, world_size, "127.0.0.1", port).unwrap();
            ch.broadcast(b"hello followers").unwrap();
            ch.broadcast(b"second message").unwrap();
        });

        let follower = std::thread::spawn(move || {
            let mut ch = TcpControlChannel::establish(1, world_size, "127.0.0.1", port).unwrap();
            let msg1 = ch.recv().unwrap();
            let msg2 = ch.recv().unwrap();
            (msg1, msg2)
        });

        leader.join().unwrap();
        let (msg1, msg2) = follower.join().unwrap();
        assert_eq!(msg1, b"hello followers");
        assert_eq!(msg2, b"second message");
    }

    #[test]
    fn test_control_channel_multiple_followers() {
        let port: u16 = master_port_binding(CONTROL_CHANNEL_PORT_OFFSET);
        let world_size = 4;
        let payload = b"broadcast to all";

        let leader = std::thread::spawn(move || {
            let mut ch = TcpControlChannel::establish(0, world_size, "127.0.0.1", port).unwrap();
            ch.broadcast(payload).unwrap();
        });

        let followers: Vec<_> = (1..world_size)
            .map(|rank| {
                std::thread::spawn(move || {
                    let mut ch =
                        TcpControlChannel::establish(rank, world_size, "127.0.0.1", port).unwrap();
                    ch.recv().unwrap()
                })
            })
            .collect();

        leader.join().unwrap();
        for f in followers {
            assert_eq!(f.join().unwrap(), payload);
        }
    }

    #[test]
    fn test_control_channel_large_payload() {
        let port: u16 = master_port_binding(CONTROL_CHANNEL_PORT_OFFSET);
        let world_size = 2;
        // 1MB payload to test chunked reads.
        let payload: Vec<u8> = (0..1_000_000).map(|i| (i % 256) as u8).collect();
        let payload_clone = payload.clone();

        let leader = std::thread::spawn(move || {
            let mut ch = TcpControlChannel::establish(0, world_size, "127.0.0.1", port).unwrap();
            ch.broadcast(&payload_clone).unwrap();
        });

        let follower = std::thread::spawn(move || {
            let mut ch = TcpControlChannel::establish(1, world_size, "127.0.0.1", port).unwrap();
            ch.recv().unwrap()
        });

        leader.join().unwrap();
        let received = follower.join().unwrap();
        assert_eq!(received, payload);
    }

    #[test]
    fn test_control_channel_empty_payload() {
        let port: u16 = master_port_binding(CONTROL_CHANNEL_PORT_OFFSET);
        let world_size = 2;

        let leader = std::thread::spawn(move || {
            let mut ch = TcpControlChannel::establish(0, world_size, "127.0.0.1", port).unwrap();
            ch.broadcast(b"").unwrap();
        });

        let follower = std::thread::spawn(move || {
            let mut ch = TcpControlChannel::establish(1, world_size, "127.0.0.1", port).unwrap();
            ch.recv().unwrap()
        });

        leader.join().unwrap();
        let received = follower.join().unwrap();
        assert!(received.is_empty());
    }
}
