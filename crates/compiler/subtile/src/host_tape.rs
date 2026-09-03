// SPDX-License-Identifier: Apache-2.0
//! THE HOST TAPE — the program, as data.
//!
//! A compiled bundle is not a program. It is a set of KERNELS. What actually
//! runs a model is the host-side sequence between them: gather the embedding
//! row, upload it, launch a device program, read the logits back, fill a mask,
//! launch again. Today that sequence exists as hand-written control flow
//! (`spyre_worker::superdsc_forward_chunk` — ~970 lines of `acts.push(...)`,
//! `run_step(...)`, `read_tensor(...)` interleaved with `if` on facts the bake
//! already decided). This module is that sequence as DATA.
//!
//! ⛔ NAMING, BECAUSE THIS REPO ALREADY HAS THREE "TAPE"S:
//!   - `subtile_tape::SubtileTape` — the rerolled per-wavefront op tape (compiler IR)
//!   - `targets/metal/src/tape/` — metal's INSTRUCTION tape
//!   - THIS — the HOST tape: what the host does, in order, to run a forward.
//!
//! # Every step is a kernel; the site says who launches it
//!
//! Host-side compute is not a different KIND of thing from device compute — it
//! is a kernel that happens to run on the CPU. So a step names a kernel and the
//! SITE it runs at, and the player hands it to that site's launcher. The
//! protocol differs per site, and that is the launcher's business, not the
//! player's:
//!
//! | site | transfers | launch |
//! |---|---|---|
//! | [`Site::Device`] (a baked `init_binary`) | yes | yes |
//! | [`Site::Host`] (a CPU kernel: embed gather, rotary, mask fill) | no | a direct call |
//! | a weight load ([`Site::Host`], output `ToDevice`) | h2d only | none |
//!
//! That last row is why weight loading is not a special phase: it is a kernel
//! whose launcher happens to do an h2d and nothing else. There is no privileged
//! "PrepareModel" moment for a constant to be uploaded outside of.
//!
//! # What is on the tape vs inside a kernel is a LOWERING decision
//!
//! Moving the mask fill from the CPU into a device program changes that step's
//! [`Step::site`] and [`Step::kernel`] and NOTHING ELSE — not this type, not the
//! player. That is the whole point of the split: the boundary can move as the
//! lowering improves, and the runtime never learns about it.
//!
//! # Transfers are STATED, never inferred
//!
//! Each operand carries the movement it needs ([`Operand::transfer`]), baked by
//! the lowering. The launcher obeys; it does not consult a residency model, or
//! compare anything, or decide. A residency mistake is then a WRONG CONSTANT
//! visible in generated code — not emergent runtime behaviour. (Chosen over
//! inference deliberately: today's `dev_off_stk` episode is what runtime-derived
//! addressing costs when it is wrong.)

use core::fmt;

/// A tensor's launch-ABI id — the same `t{id}` space the bundle's placements use.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct TensorId(pub u32);

/// Which kernel a step runs. Opaque to the player: it is an index into whatever
/// table the target's launcher keeps (a baked `init_binary`, a CPU fn).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct KernelId(pub u32);

/// Where a step runs. The player dispatches on this and does nothing else with it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Site {
    /// A CPU kernel — embedding gather, rotary table, mask fill. No transfers,
    /// no explicit launch; the launcher calls it.
    Host,
    /// A baked device program. Its launcher performs the transfers its operands
    /// declare, then launches.
    Device,
}

/// The movement an operand needs, decided at bake.
///
/// ⛔ NOT A RESIDENCY QUERY. This says what MUST MOVE, not where the bytes
/// currently are — so the launcher transcribes rather than plans.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Transfer {
    /// Already where it is needed.
    None,
    /// Upload before the kernel runs.
    ToDevice,
    /// Read back after the kernel runs.
    ToHost,
}

/// One operand of a step.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Operand {
    pub tensor: TensorId,
    pub transfer: Transfer,
}

impl Operand {
    /// An operand already in place — no movement.
    pub const fn resident(tensor: TensorId) -> Self {
        Self {
            tensor,
            transfer: Transfer::None,
        }
    }
}

/// One step: a kernel, where it runs, and its operands.
#[derive(Clone, Copy, Debug)]
pub struct Step<'a> {
    pub kernel: KernelId,
    pub site: Site,
    /// Operands read. A `ToDevice` input is uploaded before the launch.
    pub inputs: &'a [Operand],
    /// Operands written. After the kernel runs, each output's stated transfer
    /// is applied: `ToHost` reads it back, `ToDevice` uploads it. The second is
    /// what makes a host-produced value (a constant, a loaded weight) a single
    /// step rather than a step plus a separate uploader.
    pub outputs: &'a [Operand],
}

/// The program: steps in execution order.
///
/// ⭐ THE GENERATED FORM IS `HostTape<'static>` — `#[forward]` emits the steps as
/// a `static`, which is the end state and what the lifetime parameter exists to
/// express. It is a PARAMETER rather than a hard `'static` so a tape can also be
/// assembled at LOAD from baked bundle facts during the port, without either
/// weakening the generated form or inviting a second tape type. What is ruled
/// out is a tape rebuilt per FORWARD — that would be the runtime analysis this
/// replaces, and it is ruled out by where the tape is constructed, not by this
/// lifetime.
#[derive(Clone, Copy, Debug)]
pub struct HostTape<'a> {
    pub steps: &'a [Step<'a>],
}

/// What a target must provide. FOUR primitives, and every tape player in the
/// system is the same loop over them — metal, spyre and cuda differ only here.
///
/// ⛔ A GENERIC BOUND, NOT `dyn`. The player is monomorphised per target; there
/// is no runtime dispatch and no plugin registry. scratchy generates and
/// specialises, it does not abstract.
pub trait Launcher {
    type Error: fmt::Debug;

    /// Upload `tensor` to the device.
    fn h2d(&mut self, tensor: TensorId) -> Result<(), Self::Error>;
    /// Read `tensor` back to the host.
    fn d2h(&mut self, tensor: TensorId) -> Result<(), Self::Error>;
    /// Run a baked device program.
    fn launch(&mut self, kernel: KernelId) -> Result<(), Self::Error>;
    /// Run a CPU kernel.
    fn host_call(&mut self, kernel: KernelId) -> Result<(), Self::Error>;
}

/// THE TAPE PLAYER. This is the whole thing.
///
/// It knows the ORDER and the protocol shape; it knows nothing about any model,
/// any opcode, or what a kernel computes. Adding a target does not touch it.
pub fn play<L: Launcher>(tape: &HostTape<'_>, l: &mut L) -> Result<(), L::Error> {
    for step in tape.steps {
        for op in step.inputs {
            if op.transfer == Transfer::ToDevice {
                l.h2d(op.tensor)?;
            }
        }
        match step.site {
            Site::Host => l.host_call(step.kernel)?,
            Site::Device => l.launch(step.kernel)?,
        }
        // ⛔ OUTPUTS ARE SYMMETRIC WITH INPUTS, and the asymmetry was a bug.
        // A HOST kernel that fills a buffer the device must then read (a
        // constant, a gathered row, a loaded weight) declares its output
        // `ToDevice` — so "run it, then upload the result" is ONE step. Without
        // this, a weight load could not be a step at all: it would need a host
        // step plus a separate uploader, which is the ad-hoc shape this type
        // exists to remove.
        for op in step.outputs {
            match op.transfer {
                Transfer::ToDevice => l.h2d(op.tensor)?,
                Transfer::ToHost => l.d2h(op.tensor)?,
                Transfer::None => {}
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Records what the player asked for, in order.
    #[derive(Default)]
    struct Trace(Vec<String>);

    impl Launcher for Trace {
        type Error = ();
        fn h2d(&mut self, t: TensorId) -> Result<(), ()> {
            self.0.push(format!("h2d {}", t.0));
            Ok(())
        }
        fn d2h(&mut self, t: TensorId) -> Result<(), ()> {
            self.0.push(format!("d2h {}", t.0));
            Ok(())
        }
        fn launch(&mut self, k: KernelId) -> Result<(), ()> {
            self.0.push(format!("launch {}", k.0));
            Ok(())
        }
        fn host_call(&mut self, k: KernelId) -> Result<(), ()> {
            self.0.push(format!("host {}", k.0));
            Ok(())
        }
    }

    /// ⭐ THE SHAPE OF A REAL FORWARD, as the tape sees it: a CPU kernel
    /// produces the embedding row, a device kernel consumes it and writes
    /// logits, and the logits come back. This is exactly the sequence
    /// `superdsc_forward_chunk` spells as control flow.
    ///
    /// What it pins: a HOST step performs NO transfer and NO launch (it is a
    /// direct call), and a DEVICE step moves only the operands that say so, in
    /// the order inputs → launch → outputs.
    #[test]
    fn a_host_kernel_then_a_device_kernel_plays_in_protocol_order() {
        static EMBED_OUT: &[Operand] = &[Operand::resident(TensorId(0))];
        static DEC_IN: &[Operand] = &[
            Operand {
                tensor: TensorId(0),
                transfer: Transfer::ToDevice,
            },
            // A weight already staged — declared, and deliberately NOT moved.
            Operand::resident(TensorId(7)),
        ];
        static DEC_OUT: &[Operand] = &[Operand {
            tensor: TensorId(9),
            transfer: Transfer::ToHost,
        }];
        static STEPS: &[Step<'static>] = &[
            Step {
                kernel: KernelId(100),
                site: Site::Host,
                inputs: &[],
                outputs: EMBED_OUT,
            },
            Step {
                kernel: KernelId(200),
                site: Site::Device,
                inputs: DEC_IN,
                outputs: DEC_OUT,
            },
        ];

        let mut t = Trace::default();
        play(&HostTape { steps: STEPS }, &mut t).expect("plays");
        assert_eq!(
            t.0,
            vec![
                "host 100",   // CPU kernel: no h2d, no launch
                "h2d 0",      // the gathered row goes up
                "launch 200", // then the device program
                "d2h 9",      // then the logits come back
            ],
            "protocol order is inputs -> run -> outputs, and only STATED transfers happen"
        );
    }

    /// The property that lets work move between host and device without the
    /// player changing: flipping a step's `site` changes WHICH launcher runs and
    /// nothing else about the loop.
    #[test]
    fn moving_a_step_between_sites_changes_only_the_launcher() {
        static OUT: &[Operand] = &[Operand::resident(TensorId(3))];
        // Statics, because `HostTape::steps` is `&'static` BY DESIGN: a tape
        // assembled at runtime would be the analysis this type removes. The
        // two differ in ONE field.
        static AS_HOST: &[Step<'static>] = &[Step {
            kernel: KernelId(42),
            site: Site::Host,
            inputs: &[],
            outputs: OUT,
        }];
        static AS_DEVICE: &[Step<'static>] = &[Step {
            kernel: KernelId(42),
            site: Site::Device,
            inputs: &[],
            outputs: OUT,
        }];

        let mut host = Trace::default();
        play(&HostTape { steps: AS_HOST }, &mut host).unwrap();
        let mut dev = Trace::default();
        play(&HostTape { steps: AS_DEVICE }, &mut dev).unwrap();

        assert_eq!(host.0, vec!["host 42"]);
        assert_eq!(dev.0, vec!["launch 42"]);
    }

    /// A weight load is a kernel whose launcher does an h2d and NO launch —
    /// expressed with the SAME `Step`, not a special case. Pinning it because
    /// "weight loading is a step on the tape" is the claim that removes the
    /// privileged `PrepareModel` phase: if load is a step, there is no phase for
    /// a constant to be uploaded outside of.
    ///
    /// It is a HOST kernel (it reads the checkpoint into host memory) whose
    /// OUTPUT is `ToDevice`. That is the case the first cut of this player could
    /// not express — outputs only handled `ToHost` — and it is why they are
    /// symmetric now.
    #[test]
    fn a_weight_load_is_a_host_kernel_whose_output_goes_to_the_device() {
        static W: &[Operand] = &[Operand {
            tensor: TensorId(2),
            transfer: Transfer::ToDevice,
        }];
        static STEPS: &[Step<'static>] = &[Step {
            kernel: KernelId(1),
            site: Site::Host,
            inputs: &[],
            outputs: W,
        }];
        let mut t = Trace::default();
        play(&HostTape { steps: STEPS }, &mut t).unwrap();
        assert_eq!(
            t.0,
            vec!["host 1", "h2d 2"],
            "read the checkpoint, then upload it — one step, no launch"
        );
    }

    /// A baked CONSTANT (`0.5`, `±448`, the identity, the ones vector, the rope
    /// permutation) is the same shape as a weight load, and that is the whole
    /// reason it can be hoisted: today those are rebuilt and re-uploaded on
    /// EVERY forward inside `superdsc_forward_chunk`. As a step with a
    /// `ToDevice` output it belongs to whichever tape it is placed on, and
    /// putting it on the LOAD tape uploads it once.
    #[test]
    fn a_constant_upload_has_the_same_shape_as_a_weight_load() {
        static C: &[Operand] = &[Operand {
            tensor: TensorId(4095),
            transfer: Transfer::ToDevice,
        }];
        static STEPS: &[Step<'static>] = &[Step {
            kernel: KernelId(77),
            site: Site::Host,
            inputs: &[],
            outputs: C,
        }];
        let mut t = Trace::default();
        play(&HostTape { steps: STEPS }, &mut t).unwrap();
        assert_eq!(t.0, vec!["host 77", "h2d 4095"]);
    }
}
