//! ⭐ WHERE A CARD'S HOST THREADS RUN: the card's NUMA node, read from sysfs when the card opens.
//!
//! A card's response ring is polled, and its doorbell rung, by host threads: the response
//! worker (`RspWrk0`, which busy-polls, as the vendor's does), the per-step submit thread of its
//! rank, and the bring-up thread that first touches its host buffers. A thread that runs on the
//! other socket reaches the card's ring across the socket interconnect on every poll and every
//! doorbell. Measured on a tp/8 decode (granite-3.1-2b fp8, 8 cards on node 1 of a 2-node host,
//! unpinned): 3 of 6 runs stalled between steps (ITL p50 9.85–11.98 ms) against 6 of 6 clean at
//! p50 ≈ 7.4 ms with the process confined to node 1.
//!
//! So a card's node is a DEVICE FACT, read once at open:
//! `/sys/bus/pci/devices/<bdf>/numa_node`, then that node's
//! `/sys/devices/system/node/node<N>/cpulist`, intersected with the CPUs this process may use
//! (`sched_getaffinity`). The result is a [`CardCpus`], owned by the card's runtime, and each of
//! those threads pins itself with [`CardCpus::pin_current_thread`] as its first act.
//!
//! A machine can have no node to pin to: sysfs reports `numa_node = -1`, or none of the node's
//! CPUs is in the process's allowed set. Then [`CardCpus`] says so ([`NoNode`]) and nothing pins.
//! That is how the machine is built, not a fallback: on such a machine there is no nearer CPU to
//! choose. A sysfs file that cannot be read or parsed, or an affinity call the kernel refuses, is
//! an error by name ([`PlacementError`], [`PinError`]); nothing here is skipped silently.

use std::fmt;

/// A NUMA node id, as the kernel numbers it in sysfs (`node<N>`). A device's node is read by
/// [`NumaNode::parse_sysfs`]; there is no other constructor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NumaNode(u32);

impl NumaNode {
    /// The node's number.
    pub fn get(self) -> u32 {
        self.0
    }

    /// Parse a PCI device's `numa_node` attribute: a node id, or `-1`, which the kernel writes
    /// for a device the platform assigns to no node (`None`). One trailing newline is accepted,
    /// as sysfs writes it. Anything else is refused by name.
    pub fn parse_sysfs(text: &str) -> Result<Option<Self>, NumaNodeError> {
        let t = text.strip_suffix('\n').unwrap_or(text);
        if t == "-1" {
            return Ok(None);
        }
        let malformed = || NumaNodeError {
            text: text.to_string(),
        };
        if t.is_empty() || !t.bytes().all(|b| b.is_ascii_digit()) {
            return Err(malformed());
        }
        t.parse::<u32>()
            .map(|n| Some(Self(n)))
            .map_err(|_| malformed())
    }
}

impl fmt::Display for NumaNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A `numa_node` attribute that is neither a node id nor `-1`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NumaNodeError {
    /// The attribute's text, as read.
    pub text: String,
}

impl fmt::Display for NumaNodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "numa_node {:?} is neither a node id nor -1", self.text)
    }
}
impl std::error::Error for NumaNodeError {}

/// A set of CPU ids, held as sorted, disjoint, non-adjacent inclusive ranges, the shape of the
/// kernel's cpulist format (`0-35,72-107`). Built by [`CpuList::parse`]; a range is never
/// expanded, so a list's size is its number of ranges, whatever CPU ids it names.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CpuList {
    ranges: Vec<(usize, usize)>,
}

impl CpuList {
    /// Parse the kernel's cpulist format: comma-separated items, each a CPU id `N` or an
    /// inclusive range `A-B` with `A <= B`, ids in decimal digits only. One trailing newline is
    /// accepted, as sysfs writes it. An empty list is a node with no CPUs (a memory-only node).
    /// Items may overlap or come in any order; the list is normalized. Anything else (an empty
    /// item, a sign, whitespace, a reversed range, an id past `usize`) is refused by name.
    pub fn parse(text: &str) -> Result<Self, CpuListError> {
        let t = text.strip_suffix('\n').unwrap_or(text);
        if t.is_empty() {
            return Ok(Self::default());
        }
        let mut ranges = Vec::new();
        for item in t.split(',') {
            let (lo, hi) = match item.split_once('-') {
                Some((a, b)) => (cpu_id(a, item)?, cpu_id(b, item)?),
                None => {
                    let c = cpu_id(item, item)?;
                    (c, c)
                }
            };
            if lo > hi {
                return Err(CpuListError::ReversedRange {
                    item: item.to_string(),
                });
            }
            ranges.push((lo, hi));
        }
        Ok(Self::normalized(ranges))
    }

    /// Sort, then merge every overlapping or adjacent pair.
    fn normalized(mut ranges: Vec<(usize, usize)>) -> Self {
        ranges.sort_unstable();
        let mut out: Vec<(usize, usize)> = Vec::with_capacity(ranges.len());
        for (lo, hi) in ranges {
            match out.last_mut() {
                Some(last) if lo <= last.1.saturating_add(1) => last.1 = last.1.max(hi),
                _ => out.push((lo, hi)),
            }
        }
        Self { ranges: out }
    }

    /// Whether `cpu` is in the list.
    pub fn contains(&self, cpu: usize) -> bool {
        self.ranges.iter().any(|&(lo, hi)| lo <= cpu && cpu <= hi)
    }

    /// Whether the list names no CPU.
    pub fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }

    /// Every CPU id in the list, ascending.
    pub fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.ranges.iter().flat_map(|&(lo, hi)| lo..=hi)
    }
}

/// One CPU id of `item`: decimal digits only.
fn cpu_id(s: &str, item: &str) -> Result<usize, CpuListError> {
    if s.is_empty() {
        return Err(CpuListError::EmptyItem {
            item: item.to_string(),
        });
    }
    let not_a_cpu = || CpuListError::NotACpu {
        item: item.to_string(),
    };
    if !s.bytes().all(|b| b.is_ascii_digit()) {
        return Err(not_a_cpu());
    }
    s.parse::<usize>().map_err(|_| not_a_cpu())
}

impl fmt::Display for CpuList {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, &(lo, hi)) in self.ranges.iter().enumerate() {
            if i > 0 {
                write!(f, ",")?;
            }
            if lo == hi {
                write!(f, "{lo}")?;
            } else {
                write!(f, "{lo}-{hi}")?;
            }
        }
        Ok(())
    }
}

/// A cpulist the kernel's format does not allow.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CpuListError {
    /// An item, or one end of a range, is empty (`1,,2`, `3-`, `-1`).
    EmptyItem { item: String },
    /// An item is not a decimal CPU id or a range of them (`a`, `+1`, ` 1`, an id past `usize`).
    NotACpu { item: String },
    /// A range runs backwards (`3-1`).
    ReversedRange { item: String },
}

impl fmt::Display for CpuListError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyItem { item } => write!(f, "cpulist item {item:?} has an empty CPU id"),
            Self::NotACpu { item } => write!(f, "cpulist item {item:?} is not a CPU id or range"),
            Self::ReversedRange { item } => write!(f, "cpulist range {item:?} runs backwards"),
        }
    }
}
impl std::error::Error for CpuListError {}

/// Why a card's threads have no node to run on. Each is a fact of the machine or the build, and
/// with any of them there is nothing nearer to pin to, so nothing pins.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoNode {
    /// sysfs reports `numa_node = -1`: the platform assigns the card to no node (a single-node
    /// machine, or firmware that names none). Every CPU is as near as every other.
    FirmwareReportsNone,
    /// The card's node has no CPU this process may run on: the process's allowed set leaves the
    /// node out, or the node has no CPUs (a memory-only node).
    NoAllowedCpu(NumaNode),
    /// Hardware with no card behind it (a test double).
    NoDevice,
    /// This build's target has no per-thread CPU affinity: only Linux sets one, and only Linux
    /// drives a card.
    TargetHasNoAffinity,
}

impl fmt::Display for NoNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FirmwareReportsNone => write!(f, "sysfs reports numa_node -1"),
            Self::NoAllowedCpu(n) => {
                write!(f, "node {n} has no CPU in this process's allowed set")
            }
            Self::NoDevice => write!(f, "no device"),
            Self::TargetHasNoAffinity => write!(f, "this target sets no thread affinity"),
        }
    }
}

/// ⭐ THE CPUS ONE CARD'S HOST THREADS RUN ON: the card's NUMA node and that node's CPUs this
/// process may use, read once when the card opens ([`CardCpus::read`]) and owned by the card's
/// runtime. A thread pins itself with [`CardCpus::pin_current_thread`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CardCpus(Placement);

#[derive(Debug, Clone, PartialEq, Eq)]
enum Placement {
    /// The card sits on `node`; `cpus` is that node's CPUs in this process's allowed set, never
    /// empty, and every id in it is one the kernel's affinity mask holds (it came from one).
    Node { node: NumaNode, cpus: CpuList },
    /// There is no node to run on, and why.
    NoNode(NoNode),
}

impl CardCpus {
    /// Hardware with no card behind it: no node, nothing pins.
    pub const NO_DEVICE: Self = Self(Placement::NoNode(NoNode::NoDevice));

    /// The card's node, when it has one this process can run on.
    pub fn node(&self) -> Option<NumaNode> {
        match &self.0 {
            Placement::Node { node, .. } => Some(*node),
            Placement::NoNode(_) => None,
        }
    }

    /// The CPUs the card's threads run on, when it has a node.
    pub fn cpus(&self) -> Option<&CpuList> {
        match &self.0 {
            Placement::Node { cpus, .. } => Some(cpus),
            Placement::NoNode(_) => None,
        }
    }

    /// Why the card has no node, when it has none.
    pub fn no_node(&self) -> Option<NoNode> {
        match &self.0 {
            Placement::Node { .. } => None,
            Placement::NoNode(why) => Some(*why),
        }
    }

    /// Read the placement of the card at PCI address `pci` (`DDDD:BB:DD.F`): its
    /// `/sys/bus/pci/devices/<pci>/numa_node`, then that node's
    /// `/sys/devices/system/node/node<N>/cpulist`, intersected with the CPUs this process may
    /// run on. On a target with no thread affinity there is nothing to read: no node.
    ///
    /// # Errors
    /// A sysfs file that cannot be read or does not parse, or a `sched_getaffinity` the kernel
    /// refuses, each by name.
    pub fn read(pci: &str) -> Result<Self, PlacementError> {
        let Some(allowed) = process_allowed()? else {
            return Ok(Self(Placement::NoNode(NoNode::TargetHasNoAffinity)));
        };
        let node_path = format!("/sys/bus/pci/devices/{pci}/numa_node");
        let node = NumaNode::parse_sysfs(&read_sysfs(&node_path)?).map_err(|err| {
            PlacementError::NumaNode {
                path: node_path,
                err,
            }
        })?;
        Self::from_facts(
            node,
            |n| {
                let path = format!("/sys/devices/system/node/node{n}/cpulist");
                CpuList::parse(&read_sysfs(&path)?)
                    .map_err(|err| PlacementError::CpuList { path, err })
            },
            &allowed,
        )
    }

    /// The placement the facts give: no node for `-1`; otherwise the node's CPUs (`node_cpus`,
    /// read only when there is a node) that are in `allowed`, or no node when none is.
    fn from_facts(
        node: Option<NumaNode>,
        node_cpus: impl FnOnce(NumaNode) -> Result<CpuList, PlacementError>,
        allowed: &CpuList,
    ) -> Result<Self, PlacementError> {
        let Some(node) = node else {
            return Ok(Self(Placement::NoNode(NoNode::FirmwareReportsNone)));
        };
        let node_cpus = node_cpus(node)?;
        let cpus = CpuList::normalized(
            allowed
                .iter()
                .filter(|&c| node_cpus.contains(c))
                .map(|c| (c, c))
                .collect(),
        );
        Ok(Self(if cpus.is_empty() {
            Placement::NoNode(NoNode::NoAllowedCpu(node))
        } else {
            Placement::Node { node, cpus }
        }))
    }

    /// ⭐ RUN THE CALLING THREAD ON THE CARD'S NODE: `sched_setaffinity` to the card's CPUs, the
    /// first act of each thread that drives the card. With no node it does nothing; there is
    /// nothing nearer to run on.
    ///
    /// # Errors
    /// The kernel refused the affinity call, by name.
    pub fn pin_current_thread(&self) -> Result<(), PinError> {
        match &self.0 {
            Placement::NoNode(_) => Ok(()),
            Placement::Node { node, cpus } => set_affinity(cpus).map_err(|errno| PinError {
                node: *node,
                cpus: cpus.clone(),
                errno,
            }),
        }
    }
}

impl fmt::Display for CardCpus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.0 {
            Placement::Node { node, cpus } => write!(f, "NUMA node {node}, CPUs {cpus}"),
            Placement::NoNode(why) => write!(f, "no NUMA node ({why})"),
        }
    }
}

/// The CPUs this process may run on: the main thread's `sched_getaffinity` (the thread whose id
/// is the process id), read once and kept. The first read precedes every pin in the process (a
/// pin needs a [`CardCpus`], and reading one reads this first), so what is kept is the mask the
/// process started with. Not the calling thread's mask: a card opened from a thread spawned by a
/// pinned one would intersect with what that pin left it. The error is the kernel's errno.
#[cfg(target_os = "linux")]
fn process_allowed() -> Result<Option<CpuList>, PlacementError> {
    use rustix::thread::{CpuSet, Pid};
    static ALLOWED: std::sync::OnceLock<Result<CpuList, i32>> = std::sync::OnceLock::new();
    ALLOWED
        .get_or_init(|| {
            let main = i32::try_from(std::process::id())
                .ok()
                .and_then(Pid::from_raw)
                .ok_or(libc::ESRCH)?;
            let mask =
                rustix::thread::sched_getaffinity(Some(main)).map_err(|e| e.raw_os_error())?;
            Ok(CpuList::normalized(
                (0..CpuSet::MAX_CPU)
                    .filter(|&c| mask.is_set(c))
                    .map(|c| (c, c))
                    .collect(),
            ))
        })
        .clone()
        .map(Some)
        .map_err(|errno| PlacementError::AllowedCpus { errno })
}

/// Only Linux sets a per-thread CPU affinity, and only Linux drives a card.
#[cfg(not(target_os = "linux"))]
fn process_allowed() -> Result<Option<CpuList>, PlacementError> {
    Ok(None)
}

/// `sched_setaffinity(0, cpus)`: the calling thread runs on `cpus` from here on. Every id is
/// below `CpuSet::MAX_CPU`: a placement's CPUs are a subset of [`process_allowed`], which the
/// kernel reported in a `CpuSet`. The error is the kernel's errno.
#[cfg(target_os = "linux")]
fn set_affinity(cpus: &CpuList) -> Result<(), i32> {
    let mut mask = rustix::thread::CpuSet::new();
    for c in cpus.iter() {
        mask.set(c);
    }
    rustix::thread::sched_setaffinity(None, &mask).map_err(|e| e.raw_os_error())
}

/// No placement has a node on this target ([`process_allowed`] reads none), so this is never
/// reached; were it, the target has no such call.
#[cfg(not(target_os = "linux"))]
fn set_affinity(_cpus: &CpuList) -> Result<(), i32> {
    Err(libc::ENOSYS)
}

fn read_sysfs(path: &str) -> Result<String, PlacementError> {
    std::fs::read_to_string(path).map_err(|e| PlacementError::Read {
        path: path.to_string(),
        err: e.to_string(),
    })
}

/// Why a card's placement could not be read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlacementError {
    /// A sysfs file could not be read.
    Read { path: String, err: String },
    /// The card's `numa_node` is neither a node id nor `-1`.
    NumaNode { path: String, err: NumaNodeError },
    /// The node's `cpulist` is not in the kernel's format.
    CpuList { path: String, err: CpuListError },
    /// `sched_getaffinity` refused (the kernel's errno).
    AllowedCpus { errno: i32 },
}

impl fmt::Display for PlacementError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Read { path, err } => write!(f, "read {path}: {err}"),
            Self::NumaNode { path, err } => write!(f, "{path}: {err}"),
            Self::CpuList { path, err } => write!(f, "{path}: {err}"),
            Self::AllowedCpus { errno } => write!(
                f,
                "sched_getaffinity: {}",
                std::io::Error::from_raw_os_error(*errno)
            ),
        }
    }
}
impl std::error::Error for PlacementError {}

/// The kernel refused to run a thread on its card's node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PinError {
    /// The card's node.
    pub node: NumaNode,
    /// The CPUs the thread was to run on.
    pub cpus: CpuList,
    /// `sched_setaffinity`'s errno.
    pub errno: i32,
}

impl fmt::Display for PinError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "could not run this thread on NUMA node {}'s CPUs {}: sched_setaffinity: {}",
            self.node,
            self.cpus,
            std::io::Error::from_raw_os_error(self.errno)
        )
    }
}
impl std::error::Error for PinError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn list(text: &str) -> CpuList {
        CpuList::parse(text).unwrap()
    }

    fn cpus(l: &CpuList) -> Vec<usize> {
        l.iter().collect()
    }

    #[test]
    fn cpulist_parses_a_range() {
        assert_eq!(cpus(&list("0-3")), vec![0, 1, 2, 3]);
    }

    #[test]
    fn cpulist_parses_singletons_and_commas() {
        assert_eq!(cpus(&list("5")), vec![5]);
        assert_eq!(cpus(&list("0-1,4,6-7")), vec![0, 1, 4, 6, 7]);
    }

    #[test]
    fn cpulist_takes_sysfs_trailing_newline() {
        let l = list("36-71,108-143\n");
        assert_eq!(l.iter().count(), 72);
        assert!(l.contains(36) && l.contains(71) && l.contains(108) && l.contains(143));
        assert!(!l.contains(35) && !l.contains(72) && !l.contains(107) && !l.contains(144));
        assert_eq!(l.to_string(), "36-71,108-143");
    }

    #[test]
    fn cpulist_normalizes_overlap_order_and_adjacency() {
        assert_eq!(list("4,0-2,3").to_string(), "0-4");
        assert_eq!(list("8-9,1-5,2-3").to_string(), "1-5,8-9");
        assert_eq!(list("7,7").to_string(), "7");
    }

    #[test]
    fn cpulist_empty_is_a_cpuless_node() {
        assert!(list("").is_empty());
        assert!(list("\n").is_empty());
    }

    #[test]
    fn cpulist_refuses_malformed_input_by_name() {
        use CpuListError::*;
        let cases: &[(&str, CpuListError)] = &[
            ("1,,2", EmptyItem { item: "".into() }),
            ("3-", EmptyItem { item: "3-".into() }),
            ("-1", EmptyItem { item: "-1".into() }),
            ("1,", EmptyItem { item: "".into() }),
            ("a", NotACpu { item: "a".into() }),
            ("+1", NotACpu { item: "+1".into() }),
            (" 1", NotACpu { item: " 1".into() }),
            (
                "1-2-3",
                NotACpu {
                    item: "1-2-3".into(),
                },
            ),
            ("1\n\n", NotACpu { item: "1\n".into() }),
            (
                "99999999999999999999999",
                NotACpu {
                    item: "99999999999999999999999".into(),
                },
            ),
            ("3-1", ReversedRange { item: "3-1".into() }),
        ];
        for (text, want) in cases {
            assert_eq!(CpuList::parse(text).as_ref(), Err(want), "input {text:?}");
        }
    }

    #[test]
    fn numa_node_parses_an_id_or_minus_one() {
        assert_eq!(NumaNode::parse_sysfs("1\n"), Ok(Some(NumaNode(1))));
        assert_eq!(NumaNode::parse_sysfs("0"), Ok(Some(NumaNode(0))));
        assert_eq!(NumaNode::parse_sysfs("-1\n"), Ok(None));
    }

    #[test]
    fn numa_node_refuses_anything_else_by_name() {
        for text in ["", "\n", "-2\n", "x", " 1", "+1", "1\n\n", "99999999999"] {
            assert_eq!(
                NumaNode::parse_sysfs(text),
                Err(NumaNodeError { text: text.into() }),
                "input {text:?}"
            );
        }
    }

    #[test]
    fn no_node_pins_nothing() {
        assert_eq!(CardCpus::NO_DEVICE.pin_current_thread(), Ok(()));
        assert_eq!(CardCpus::NO_DEVICE.node(), None);
        assert_eq!(CardCpus::NO_DEVICE.no_node(), Some(NoNode::NoDevice));
    }

    fn place(node: &str, node_cpus: &str, allowed: &str) -> CardCpus {
        let node = NumaNode::parse_sysfs(node).unwrap();
        let node_cpus = list(node_cpus);
        CardCpus::from_facts(node, |_| Ok(node_cpus), &list(allowed)).unwrap()
    }

    #[test]
    fn a_node_inside_the_allowed_set_keeps_all_its_cpus() {
        let p = place("1\n", "36-71,108-143\n", "0-143");
        assert_eq!(p.node(), Some(NumaNode(1)));
        assert_eq!(
            p.cpus().map(ToString::to_string).as_deref(),
            Some("36-71,108-143")
        );
    }

    #[test]
    fn the_allowed_set_narrows_the_node() {
        let p = place("1\n", "36-71,108-143\n", "40-50,100-110");
        assert_eq!(p.node(), Some(NumaNode(1)));
        assert_eq!(
            p.cpus().map(ToString::to_string).as_deref(),
            Some("40-50,108-110")
        );
    }

    #[test]
    fn an_empty_intersection_is_no_node() {
        let p = place("1\n", "36-71,108-143\n", "0-35,72-107");
        assert_eq!(p.no_node(), Some(NoNode::NoAllowedCpu(NumaNode(1))));
        assert_eq!(p.cpus(), None);
        assert_eq!(p.pin_current_thread(), Ok(()));
    }

    #[test]
    fn a_cpuless_node_is_no_node() {
        let p = place("2\n", "\n", "0-143");
        assert_eq!(p.no_node(), Some(NoNode::NoAllowedCpu(NumaNode(2))));
    }

    #[test]
    fn minus_one_is_no_node_and_reads_no_cpulist() {
        let node = NumaNode::parse_sysfs("-1\n").unwrap();
        let p = CardCpus::from_facts(
            node,
            |n| panic!("node {n}'s cpulist read for a card with no node"),
            &list("0-143"),
        )
        .unwrap();
        assert_eq!(p.no_node(), Some(NoNode::FirmwareReportsNone));
        assert_eq!(p.pin_current_thread(), Ok(()));
    }

    #[test]
    fn a_cpulist_read_error_is_refused_by_name() {
        let err = CardCpus::from_facts(
            Some(NumaNode(1)),
            |_| {
                CpuList::parse("3-1").map_err(|err| PlacementError::CpuList {
                    path: "node1/cpulist".into(),
                    err,
                })
            },
            &list("0-143"),
        );
        assert_eq!(
            err,
            Err(PlacementError::CpuList {
                path: "node1/cpulist".into(),
                err: CpuListError::ReversedRange { item: "3-1".into() },
            })
        );
    }

    /// The pin itself, against the kernel: a thread placed on one CPU it may run on runs on
    /// exactly that CPU afterwards.
    #[cfg(target_os = "linux")]
    #[test]
    fn a_pin_sets_the_threads_mask_to_the_cards_cpus() {
        let allowed = process_allowed().unwrap().unwrap();
        let first = allowed.iter().next().unwrap();
        std::thread::spawn(move || {
            let one = CpuList::normalized(vec![(first, first)]);
            let p = CardCpus::from_facts(Some(NumaNode(0)), |_| Ok(one), &allowed).unwrap();
            p.pin_current_thread().unwrap();
            let now = rustix::thread::sched_getaffinity(None).unwrap();
            let mut want = rustix::thread::CpuSet::new();
            want.set(first);
            assert_eq!(now, want);
        })
        .join()
        .unwrap();
    }
}
