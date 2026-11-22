// SPDX-License-Identifier: Apache-2.0
//! KTIR → SuperDSC: Spyre's lowering of one KTIR function to the descriptors `dxp` compiles.
//!
//! The function's dataflow is run with every index a constant ([`lift`]): loads are views of
//! their buffers, shape ops re-describe views, and each tensor op becomes one device op over
//! strided views of a parameter buffer, an intermediate or a constant stick ([`graph`]). Each op
//! then becomes one descriptor ([`emit`]): its space labelled with the DDL's dimensions, divided
//! across the cores ([`divide`]), and every operand's layout read straight off its view.
//!
//! What an op is on the device — its name, its unit, how its operands are sticked — is the
//! [`opfunc`] table; the device's own facts are the generation's (`ktir_core::ddl_binds`).

pub mod dim;
pub mod divide;
pub mod emit;
pub mod format;
pub mod graph;
pub mod layout;
pub mod lift;
pub mod opfunc;
pub mod view;
pub mod wire;

use ktir_core::ddl_binds::DeviceGen;
use ktir_core::ir::IRFunction;

pub use emit::{Bindings, EmitError, Footprint};
pub use layout::Residency;
pub use lift::{Held, LiftError, ParamLayout};

/// Why a function did not lower.
#[derive(Clone, Debug, PartialEq)]
pub enum Error {
    Lift(LiftError),
    Emit(EmitError),
}

/// A function ready to place: its graph, how each parameter's buffer is held, and the storage its
/// descriptors reach.
#[derive(Clone, Debug)]
pub struct Planned {
    pub graph: graph::Graph,
    /// For each parameter, the layout its buffer is held in (`None`: as its view states). Only a
    /// [`Residency::Staged`] parameter is ever re-laid-out — its stager writes it so.
    pub layouts: Vec<ParamLayout>,
    pub footprint: Footprint,
}

impl Planned {
    /// The descriptors, in launch order, against `bindings`.
    pub fn emit<G: DeviceGen>(
        &self,
        bindings: &Bindings<'_>,
    ) -> Result<Vec<(opfunc::OpFunc, wire::Sdsc)>, Error> {
        emit::emit::<G>(&self.graph, &self.footprint, bindings).map_err(Error::Emit)
    }
}

/// The layouts `f`'s staged parameters must be held in: each re-laid-out so every op reads it on
/// its own stick, and held as wide as every op's division pads it.
pub fn layouts<G: DeviceGen>(
    f: &IRFunction<'_>,
    residency: &[Residency],
) -> Result<Vec<ParamLayout>, Error> {
    let natural = lift::lift::<G>(f, &vec![None; f.arguments.len()]).map_err(Error::Lift)?;
    let mut held = layout::staged_layouts(&natural, residency);
    let relaid = lift::lift::<G>(f, &held).map_err(Error::Lift)?;
    for (p, width) in emit::staged_widths::<G>(&relaid).into_iter().enumerate() {
        if let (Some(Some(h)), Some(w)) = (held.get_mut(p), width) {
            h.inner = h.inner.max(w);
        }
    }
    Ok(held)
}

/// Plan `f` for generation `G`, each parameter's buffer held as `layouts` states.
pub fn plan<G: DeviceGen>(f: &IRFunction<'_>, layouts: &[ParamLayout]) -> Result<Planned, Error> {
    let natural = lift::lift::<G>(f, layouts).map_err(Error::Lift)?;
    let relocated = layout::misaligned::<G>(&natural);
    let mut graph = if relocated.is_empty() {
        natural
    } else {
        lift::lift_relocating::<G>(f, layouts, &relocated).map_err(Error::Lift)?
    };
    layout::restick::<G>(&mut graph);
    layout::index_inner::<G>(&mut graph);
    let footprint = emit::footprint::<G>(&graph);
    Ok(Planned {
        graph,
        layouts: layouts.to_vec(),
        footprint,
    })
}
