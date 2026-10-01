//! Every MTL4 commit reports its own completion or failure through
//! `mtl4_dispatch::commit_and_wait`: the one wait every commit whose results a caller reads goes
//! through (the worker pool's forwards included). A command buffer the GPU did not complete still
//! reaches its event signal, so without that report a failed forward returns its stale outputs as
//! if they were its results.

use objc2::rc::Retained;
use objc2::runtime::{AnyObject, ProtocolObject};
use objc2_foundation::NSString;
use objc2_metal::{
    MTL4ArgumentTable, MTL4ArgumentTableDescriptor, MTL4CommandAllocator, MTL4CommandBuffer,
    MTL4CommandEncoder, MTL4CommandQueue, MTL4CommitFeedback, MTL4CommitOptions,
    MTL4ComputeCommandEncoder, MTLBuffer, MTLComputePipelineState, MTLDevice, MTLLibrary,
    MTLResourceOptions, MTLSharedEvent, MTLSize,
};
use scratchy_target_metal::mtl4_dispatch::{
    CommandFailure, Device, commit_and_wait, read_slice, shared_slice,
};
use scratchy_target_metal::residency::{MetalResidencySet, Pinned};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

const ECHO: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void echo(device uint* out [[buffer(0)]], device const uint* in [[buffer(1)]]) {
  out[0] = in[0];
}
"#;

/// Commits per test.
const COMMITS: u32 = 1000;

/// One queue, allocator and event, as the pool holds them, and the `echo` kernel: each command
/// buffer copies the word the host wrote into `input` to `out`.
struct Rig {
    device: Device,
    queue: Retained<ProtocolObject<dyn MTL4CommandQueue>>,
    allocator: Retained<ProtocolObject<dyn MTL4CommandAllocator>>,
    event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
    echo: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    residency: MetalResidencySet,
    table: Retained<ProtocolObject<dyn MTL4ArgumentTable>>,
    out: Pinned,
    input: Pinned,
    signalled: u64,
}

impl Rig {
    fn new() -> Option<Self> {
        let dev = scratchy_target_metal::detect_device()
            .filter(|_| scratchy_target_metal::metal4_available())?;
        let device = dev.device;
        let queue = device.newMTL4CommandQueue()?;
        let function = device
            .newLibraryWithSource_options_error(
                &NSString::from_str(ECHO),
                Some(&objc2_metal::MTLCompileOptions::new()),
            )
            .expect("compile echo")
            .newFunctionWithName(&NSString::from_str("echo"))
            .expect("echo");
        let echo = device
            .newComputePipelineStateWithFunction_error(&function)
            .expect("echo pipeline");
        let residency = MetalResidencySet::new(&device);
        let out = residency.pin(shared_slice(&device, &[0u32]));
        let input = residency.pin(shared_slice(&device, &[0u32]));
        residency.commit();
        let descriptor = MTL4ArgumentTableDescriptor::new();
        descriptor.setMaxBufferBindCount(2);
        let table = device
            .newArgumentTableWithDescriptor_error(&descriptor)
            .expect("argument table");
        unsafe {
            table.setAddress_atIndex(out.gpuAddress(), 0);
            table.setAddress_atIndex(input.gpuAddress(), 1);
        }
        Some(Self {
            allocator: device.newCommandAllocator().expect("allocator"),
            event: device.newSharedEvent().expect("event"),
            device,
            queue,
            echo,
            residency,
            table,
            out,
            input,
            signalled: 0,
        })
    }

    /// One ended command buffer holding one single-thread `echo` of `word`, declaring the rig's
    /// residency set and `extra`.
    fn command_buffer(
        &self,
        word: u32,
        extra: Option<&MetalResidencySet>,
    ) -> Retained<ProtocolObject<dyn MTL4CommandBuffer>> {
        unsafe { *(self.input.contents().as_ptr() as *mut u32) = word };
        let cb = self.device.newCommandBuffer().expect("command buffer");
        cb.beginCommandBufferWithAllocator(&self.allocator);
        for set in std::iter::once(&self.residency).chain(extra) {
            unsafe { set.attach_to_mtl4_command_buffer(Retained::as_ptr(&cb) as *mut AnyObject) };
        }
        let encoder = cb.computeCommandEncoder().expect("encoder");
        encoder.setComputePipelineState(&self.echo);
        encoder.setArgumentTable(Some(&self.table));
        let one = MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        };
        encoder.dispatchThreadgroups_threadsPerThreadgroup(one, one);
        encoder.endEncoding();
        cb.endCommandBuffer();
        cb
    }

    /// Commit `cb` through `commit_and_wait`.
    fn commit(&mut self, cb: &ProtocolObject<dyn MTL4CommandBuffer>) -> Result<(), CommandFailure> {
        self.signalled += 1;
        let done = commit_and_wait(&self.queue, cb, &self.event, self.signalled);
        self.allocator.reset();
        done
    }

    /// Commit `cb` with `options` (or none) and wait for the event alone — the completion
    /// `commit_and_wait` replaces.
    fn commit_event_only(
        &mut self,
        cb: &ProtocolObject<dyn MTL4CommandBuffer>,
        options: Option<&MTL4CommitOptions>,
    ) {
        self.signalled += 1;
        let mut cbs = [std::ptr::NonNull::from(cb)];
        let at = std::ptr::NonNull::from(&mut cbs[0]);
        match options {
            Some(options) => unsafe { self.queue.commit_count_options(at, 1, options) },
            None => unsafe { self.queue.commit_count(at, 1) },
        }
        self.queue
            .signalEvent_value(ProtocolObject::from_ref(&*self.event), self.signalled);
        assert!(
            self.event
                .waitUntilSignaledValue_timeoutMS(self.signalled, 60_000),
            "a one-thread dispatch ran over a minute"
        );
        self.allocator.reset();
    }

    fn out(&self) -> u32 {
        read_slice::<u32>(&self.out, 1)[0]
    }
}

/// `commit_and_wait` hears from every commit: each of [`COMMITS`] commits echoes its own word
/// and returns `Ok` only once its own feedback came. The same number of commits sharing ONE
/// `MTL4CommitOptions` and handler — what the pool did — hears from only some of them (Metal calls
/// a handler for the first commit of its options), which is why every commit carries its own.
/// Prints what the report costs per commit over waiting for the event alone.
#[test]
fn every_commit_reports_its_completion() {
    let Some(mut rig) = Rig::new() else {
        eprintln!("skipping: no Metal 4 GPU");
        return;
    };
    let mut reported = Vec::with_capacity(COMMITS as usize);
    let mut event_only = Vec::with_capacity(COMMITS as usize);
    for i in 1..=COMMITS {
        let cb = rig.command_buffer(i, None);
        let started = Instant::now();
        rig.commit(&cb)
            .unwrap_or_else(|failure| panic!("commit {i}: {failure}"));
        reported.push(started.elapsed());
        assert_eq!(rig.out(), i, "commit {i} returned before its dispatch ran");
        let cb = rig.command_buffer(i, None);
        let started = Instant::now();
        rig.commit_event_only(&cb, None);
        event_only.push(started.elapsed());
    }

    let heard = Arc::new(AtomicUsize::new(0));
    let shared = MTL4CommitOptions::new();
    let handler = {
        let heard = Arc::clone(&heard);
        block2::RcBlock::new(
            move |_: std::ptr::NonNull<ProtocolObject<dyn MTL4CommitFeedback>>| {
                heard.fetch_add(1, Ordering::AcqRel);
            },
        )
    };
    unsafe { shared.addFeedbackHandler(block2::RcBlock::as_ptr(&handler) as _) };
    for i in 1..=COMMITS {
        let cb = rig.command_buffer(i, None);
        rig.commit_event_only(&cb, Some(&shared));
    }
    // One more reported commit: every feedback of the commits before it is due by its own.
    let cb = rig.command_buffer(0, None);
    rig.commit(&cb).expect("the last commit");
    let shared_heard = heard.load(Ordering::Acquire);

    let median = |mut v: Vec<Duration>| {
        v.sort();
        v[v.len() / 2]
    };
    let (reported, event_only) = (median(reported), median(event_only));
    println!(
        "{COMMITS} commits: every one reported; median commit-to-return {reported:?} with its report, \
         {event_only:?} on the event alone ({:+.1} us). {COMMITS} commits sharing one options \
         object: its handler ran {shared_heard} times.",
        (reported.as_secs_f64() - event_only.as_secs_f64()) * 1e6,
    );
    assert!(
        shared_heard < COMMITS as usize,
        "a handler on shared commit options ran for all {COMMITS} commits: Metal now reports every \
         commit of reused options, and commit_and_wait's doc is out of date"
    );
}

/// A command buffer whose declared working set the GPU cannot hold resident is not run, reaches
/// its signal anyway with its output left as it was, and `commit_and_wait` returns its failure;
/// the next command buffer on the queue completes. The working set is one GiB of host memory
/// declared again and again through aliases (`bytesNoCopy` buffers over the same pages): it
/// outgrows `recommendedMaxWorkingSetSize` while the host's memory does not.
#[test]
#[ignore = "declares more memory than the GPU may hold resident, on purpose; run it alone"]
fn a_command_buffer_the_gpu_cannot_hold_resident_is_an_error() {
    let Some(mut rig) = Rig::new() else {
        eprintln!("skipping: no Metal 4 GPU");
        return;
    };
    const GIB: usize = 1 << 30;
    let layout = std::alloc::Layout::from_size_align(GIB, 1 << 14).expect("layout");
    let region = unsafe { std::alloc::alloc(layout) };
    assert!(!region.is_null(), "one GiB of host memory");
    unsafe { std::ptr::write_bytes(region, 0x5a, GIB) };
    let aliases = rig.device.recommendedMaxWorkingSetSize() as usize / GIB + 2;
    let hog = MetalResidencySet::new_unwired(&rig.device);
    let pins: Vec<Pinned> = (0..aliases)
        .map(|_| {
            let alias = unsafe {
                rig.device
                    .newBufferWithBytesNoCopy_length_options_deallocator(
                        std::ptr::NonNull::new(region.cast()).expect("region"),
                        GIB,
                        MTLResourceOptions::StorageModeShared,
                        None,
                    )
            };
            hog.pin(alias.expect("alias buffer"))
        })
        .collect();
    hog.commit();

    let cb = rig.command_buffer(7, Some(&hog));
    let failed = rig.commit(&cb);
    let out = rig.out();
    drop(pins);
    hog.commit();
    let cb = rig.command_buffer(8, None);
    let after = rig.commit(&cb);
    println!(
        "{aliases} GiB declared: {failed:?}, the echo {}; the next commit: {after:?}",
        if out == 7 { "ran" } else { "did not run" },
    );
    let failure =
        failed.expect_err("a command buffer declaring more than the GPU may hold completed");
    assert_eq!(failure, CommandFailure::OutOfMemory);
    assert_ne!(out, 7, "the failed command buffer ran");
    after.expect("the commit after the failed one");
    assert_eq!(rig.out(), 8);
    unsafe { std::alloc::dealloc(region, layout) };
}
