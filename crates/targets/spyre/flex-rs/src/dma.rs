//! Port of `DmaParams`/`FillParams` construction and validation from
//! `flex/include/flex/runtime_stream/runtime_submission_params.hpp` +
//! `flex/src/runtime_stream/runtime_submission_params.cpp`.
//!
//! `ComputeParams` lives in `crate::compute` (sibling file, same C++ header).
//! The remaining `RuntimeOperationParams` subclasses in that header
//! (`P2PDataParams`, `DataExchangeParams`, `P2PRdmaSendParams`,
//! `P2PRdmaWaitParams`, `P2PFirmwareSignalParams`, `P2PFirmwareWaitParams`,
//! `HostCallbackParams`) are out of this phase's assigned scope (DMA/compute
//! params + control_blocks/); they belong with whichever phase owns
//! multi-device P2P/RDMA submission.
//!
//! Nothing in this file calls senlib: `DmaParams` is a pure parameter/value
//! object built and validated entirely on the host side. The C++ factory
//! functions (`createDmaParams`/`destroyDmaParams`) are a manual new/delete
//! pair for a C ABI that doesn't exist once this is native Rust — ordinary
//! construction (returning `Result`) replaces them, and `Drop` is automatic.

use std::sync::Arc;

use crate::address::{ByteSize, CompositeAddress};

/// Raw host pointer for a DMA transfer, before IOMMU mapping. Newtype over
/// the C++ `void* hmva`. Never dereferenced on the Rust side of this crate —
/// carried opaquely until it crosses into the (out-of-scope) scheduler's
/// IOMMU-mapping/submit path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostVirtualAddress(pub usize);

/// Pre-mapped IO virtual address, supplied by the caller when it wants to
/// skip the backend's RAII shadow-buffer copy. Port of the C++ `void* iova`
/// on `DmaParams`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IovaAddress(pub usize);

/// Optional node-name prefix for debugging/profiling. Port of `std::string
/// op_name`; empty string in C++ means "use the default op name", modeled
/// here as `None`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct OpName(pub Option<String>);

impl OpName {
    pub const fn none() -> Self {
        Self(None)
    }
}

/// Opaque handle to a `data_conversion_info` descriptor. The type itself is
/// forward-declared in `runtime_submission_params.hpp` (`struct
/// data_conversion_info;`) and defined in the external
/// `spyrecode-host-functions/sendataconvert/sen_data_convert.h` header — it
/// is not a `flex::` type, so this phase does not port its internals, only
/// carries the handle through `DmaParams` exactly as the C++
/// `std::shared_ptr<data_conversion_info>` does.
#[derive(Debug, Clone)]
pub struct DataConversionHandle(pub Arc<DataConversionInfoOpaque>);

/// Placeholder body for the externally-defined conversion descriptor.
#[derive(Debug)]
pub struct DataConversionInfoOpaque {
    _private: (),
}

/// Port of `flex::RuntimeOperationType`'s DMA-relevant variants
/// (`H2D`/`D2H`), folded into `DmaDirection` since `DmaParams::getType()`
/// derives the tag from `to_device` — an enum here removes the possibility
/// of the tag and the bool ever disagreeing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DmaDirection {
    /// Host to device (`RuntimeOperationType::H2D`).
    HostToDevice,
    /// Device to host (`RuntimeOperationType::D2H`).
    DeviceToHost,
}

/// Which hardware pipeline shape a DMA is submitted with. Port of
/// `DmaParams::use_compute_pipeline`: `false` submits on
/// `PipelineId::ASYNC_DMAI`/`ASYNC_DMAO` with the normal DMA-graph CB shape;
/// `true` submits on `PipelineId::COMPUTE` with a graph-free CB shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DmaPipelineShape {
    /// Normal async DMA pipeline, DMA-graph CB shape.
    #[default]
    AsyncDma,
    /// Compute pipeline, graph-free CB shape.
    ComputePipeline,
}

/// Error returned when constructing `DmaParams` fails. Port of the
/// `nullDeviceAddress` throw path in `runtime_submission_params.cpp:27-37`
/// (`validatedTotalSize`/`validatedDeviceAddress`, both of which call
/// `nullDeviceAddress(dev_addr, "DmaParams")` before doing anything else).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DmaParamsError {
    /// `device_address` was null in the C++ source
    /// (`runtime_stream_utils.hpp:118`'s `device_address == nullptr` check).
    /// Since `CompositeAddress` in this port has no nullable/empty
    /// representation of its own (`CompositeAddress::from_chunks` asserts
    /// at least one chunk, so an `Arc<CompositeAddress>` is never "empty"),
    /// the nullable C++ `const CompositeAddress*` is ported as
    /// `Option<Arc<CompositeAddress>>` instead — this variant is returned
    /// when the caller passes `None`.
    NullDeviceAddress,
}

/// Port of `flex::DmaParams`.
///
/// Of the base class fields (`pipeline_barrier`, `callback`,
/// `callback_user_data`) from `RuntimeOperationParams`
/// (`runtime_submission_params.hpp:71-73`), only `pipeline_barrier` is
/// flattened in here as a concrete field. `callback`/`callback_user_data`
/// are NOT represented: in the C++ they are read one level up, by
/// `RuntimeStream` when it builds the operation object
/// (`runtime_stream.cpp:148` / `:178`:
/// `makeCompletionCallback(params->callback, params->callback_user_data)`),
/// and never by anything that consumes a `DmaParams` — the scheduler
/// receives the resulting `CompletionCallback` as a separate `submitDma`
/// argument, not through `DmaParams`. Known omission (`RuntimeStream` is not
/// ported in this phase); it carries no size/offset/CB-encoding meaning, so
/// nothing in the control-block path depends on it.
#[derive(Debug, Clone)]
pub struct DmaParams {
    hmva: HostVirtualAddress,
    dma_size: ByteSize,
    direction: DmaDirection,
    device_address: Arc<CompositeAddress>,
    dci: Option<DataConversionHandle>,
    op_name: OpName,
    iova: Option<IovaAddress>,
    pipeline_shape: DmaPipelineShape,
    pipeline_barrier: bool,
}

/// The secondary/optional shaping parameters shared by both `DmaParams`
/// constructors — bundled so neither constructor exceeds a reasonable
/// argument count, without dropping any field the C++ constructors take.
#[derive(Debug, Clone, Default)]
pub struct DmaShaping {
    pub dci: Option<DataConversionHandle>,
    pub op_name: OpName,
    pub iova: Option<IovaAddress>,
    pub pipeline_shape: DmaPipelineShape,
    pub pipeline_barrier: bool,
}

impl DmaParams {
    /// Port of the first `DmaParams` constructor
    /// (`runtime_submission_params.hpp:141-155`), where `dma_size` is derived
    /// from `device_address->total_size()` via `validatedTotalSize` — the
    /// caller must not compute the DMA size itself; it is always the total
    /// size of the device allocation.
    ///
    /// This is the constructor the *scheduler* uses when turning a
    /// `RuntimeOperationH2D`/`RuntimeOperationD2H` into params
    /// (`runtime_scheduler.cpp:214` and `:290`:
    /// `DmaParams(hmva, true/false, device_address, op.getDataConversionInfo(),
    /// op.getOpName(), op.getIova(), op.useComputePipeline(),
    /// op.getPipelineBarrier())`). It is NOT the one behind the public
    /// `createDmaParams` factory — that factory takes an explicit `dma_size`
    /// and calls the *other* constructor; see `with_explicit_size`.
    ///
    /// `device_address` is `Option`-wrapped to carry the C++ raw-pointer's
    /// nullability (see `DmaParamsError::NullDeviceAddress`); this mirrors
    /// `validatedTotalSize` (`runtime_submission_params.cpp:27-31`), which
    /// calls `nullDeviceAddress` before `dev_addr->total_size()`.
    ///
    /// # Errors
    /// Returns [`DmaParamsError::NullDeviceAddress`] if `device_address` is
    /// `None`.
    pub fn from_device_allocation(
        hmva: HostVirtualAddress,
        direction: DmaDirection,
        device_address: Option<Arc<CompositeAddress>>,
        shaping: DmaShaping,
    ) -> Result<Self, DmaParamsError> {
        let device_address = device_address.ok_or(DmaParamsError::NullDeviceAddress)?;
        let dma_size = device_address.total_size();
        Ok(Self {
            hmva,
            dma_size,
            direction,
            device_address,
            dci: shaping.dci,
            op_name: shaping.op_name,
            iova: shaping.iova,
            pipeline_shape: shaping.pipeline_shape,
            pipeline_barrier: shaping.pipeline_barrier,
        })
    }

    /// Port of the second `DmaParams` constructor
    /// (`runtime_submission_params.hpp:158-172`), where the device-side DMA
    /// byte count is already known and need not equal
    /// `device_address->total_size()`; only the address itself is validated,
    /// via `validatedDeviceAddress`.
    ///
    /// Two C++ call sites use it:
    /// * the public `createDmaParams` factory
    ///   (`runtime_submission_params.cpp:67-72`), which forwards the caller's
    ///   `dma_size` and hard-codes `op_name = ""` (the default op name)
    ///   while passing `dci`/`iova`/`use_compute_pipeline`/`pipeline_barrier`
    ///   through — so external callers land here, not on
    ///   `from_device_allocation`;
    /// * `RuntimeScheduler::handleDmaDataConversion`
    ///   (`runtime_scheduler.cpp:910`):
    ///   `DmaParams raw(staging->Pointer(), dma.dma_size, true,
    ///   dma.device_address)`, which re-uses the original `dma_size` for the
    ///   staging buffer and leaves every shaping field at its default
    ///   (no `dci`, no `iova`, no compute pipeline, no barrier) — model that
    ///   with `DmaShaping::default()`.
    ///
    /// `device_address` is `Option`-wrapped for the same reason as in
    /// `from_device_allocation`; this mirrors `validatedDeviceAddress`
    /// (`runtime_submission_params.cpp:33-37`), which also calls
    /// `nullDeviceAddress` up front.
    ///
    /// # Errors
    /// Returns [`DmaParamsError::NullDeviceAddress`] if `device_address` is
    /// `None`.
    pub fn with_explicit_size(
        hmva: HostVirtualAddress,
        dma_size: ByteSize,
        direction: DmaDirection,
        device_address: Option<Arc<CompositeAddress>>,
        shaping: DmaShaping,
    ) -> Result<Self, DmaParamsError> {
        let device_address = device_address.ok_or(DmaParamsError::NullDeviceAddress)?;
        Ok(Self {
            hmva,
            dma_size,
            direction,
            device_address,
            dci: shaping.dci,
            op_name: shaping.op_name,
            iova: shaping.iova,
            pipeline_shape: shaping.pipeline_shape,
            pipeline_barrier: shaping.pipeline_barrier,
        })
    }

    pub fn hmva(&self) -> HostVirtualAddress {
        self.hmva
    }
    pub fn dma_size(&self) -> ByteSize {
        self.dma_size
    }
    pub fn direction(&self) -> DmaDirection {
        self.direction
    }
    pub fn device_address(&self) -> &Arc<CompositeAddress> {
        &self.device_address
    }
    pub fn data_conversion(&self) -> Option<&DataConversionHandle> {
        self.dci.as_ref()
    }
    pub fn op_name(&self) -> &OpName {
        &self.op_name
    }
    pub fn iova(&self) -> Option<IovaAddress> {
        self.iova
    }
    pub fn pipeline_shape(&self) -> DmaPipelineShape {
        self.pipeline_shape
    }
    pub fn pipeline_barrier(&self) -> bool {
        self.pipeline_barrier
    }
}

/// 32-bit pattern used to fill device memory. Newtype over the C++ `uint32_t
/// fill_pattern`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FillPattern(pub u32);

/// Which async-DMA pipeline a `FillParams` submits on. Port of `FillParams`'
/// `bool use_dmai` (`true` = DMAI, `false` = DMAO).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FillPipeline {
    Dmai,
    Dmao,
}

/// Port of `flex::FillParams` (`runtime_submission_params.hpp:278-291`): fill
/// a region of device memory with a 32-bit pattern. No host buffer or IOMMU
/// mapping is involved, so this has none of `DmaParams`' host-address fields.
///
/// Note `FillParams` is the one submission-params struct in that header that
/// does *not* derive from `RuntimeOperationParams`, hence no
/// `pipeline_barrier`/`callback` fields and no `getType()`.
#[derive(Debug, Clone)]
pub struct FillParams {
    device_address: Arc<CompositeAddress>,
    size: ByteSize,
    fill_pattern: FillPattern,
    pipeline: FillPipeline,
}

impl FillParams {
    /// Port of the sole `FillParams` constructor
    /// (`runtime_submission_params.hpp:287-290`): a plain member-init list
    /// with no validation at all — no `nullDeviceAddress`, no size or
    /// alignment check. `size` is taken from the caller verbatim and is
    /// *not* derived from `device_address->total_size()` (contrast
    /// `DmaParams::from_device_allocation`), so no `Result` and no derivation
    /// here either.
    pub fn new(
        device_address: Arc<CompositeAddress>,
        size: ByteSize,
        fill_pattern: FillPattern,
        pipeline: FillPipeline,
    ) -> Self {
        Self {
            device_address,
            size,
            fill_pattern,
            pipeline,
        }
    }

    pub fn device_address(&self) -> &Arc<CompositeAddress> {
        &self.device_address
    }
    pub fn size(&self) -> ByteSize {
        self.size
    }
    pub fn fill_pattern(&self) -> FillPattern {
        self.fill_pattern
    }
    pub fn pipeline(&self) -> FillPipeline {
        self.pipeline
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::address::{ByteOffset, Chunk, LogicalAddress, RegionId};
    use crate::memory_region::DomainId;

    fn dummy_addr() -> Arc<CompositeAddress> {
        let chunk = Chunk::new(
            LogicalAddress::new(RegionId(1), ByteOffset(0)),
            ByteSize(4096),
            DomainId(0),
        );
        Arc::new(CompositeAddress::from_chunk(chunk))
    }

    #[test]
    fn from_device_allocation_rejects_null_device_address() {
        // Port of `validatedTotalSize` (`runtime_submission_params.cpp:27-31`),
        // which calls `nullDeviceAddress(dev_addr, "DmaParams")` before
        // `dev_addr->total_size()`.
        let err = DmaParams::from_device_allocation(
            HostVirtualAddress(0x1000),
            DmaDirection::HostToDevice,
            None,
            DmaShaping::default(),
        )
        .unwrap_err();
        assert_eq!(err, DmaParamsError::NullDeviceAddress);
    }

    #[test]
    fn with_explicit_size_rejects_null_device_address() {
        // Port of `validatedDeviceAddress` (`runtime_submission_params.cpp:33-37`).
        let err = DmaParams::with_explicit_size(
            HostVirtualAddress(0x1000),
            ByteSize(4096),
            DmaDirection::DeviceToHost,
            None,
            DmaShaping::default(),
        )
        .unwrap_err();
        assert_eq!(err, DmaParamsError::NullDeviceAddress);
    }

    #[test]
    fn from_device_allocation_accepts_valid_device_address() {
        let params = DmaParams::from_device_allocation(
            HostVirtualAddress(0x1000),
            DmaDirection::HostToDevice,
            Some(dummy_addr()),
            DmaShaping::default(),
        )
        .unwrap();
        assert_eq!(params.dma_size(), ByteSize(4096));
    }
}
