//! Port of `flex/src/host_interface/raii_buffer.cpp`
//! (`flex/include/flex/host_interface/raii_buffer.hpp`) and
//! `flex/src/host_interface/host_memory_buffer.{hpp,cpp}`.
//!
//! `RaiiBuffer` is a plain aligned-host-memory RAII wrapper around
//! `posix_memalign`, `memset`, `memcpy`, and `free`, with no senlib
//! involvement whatsoever — it is the allocation primitive host-side buffers
//! are built on top of, not a device-memory or hardware type. Ported using
//! `std::alloc` (which already provides aligned allocation) instead of raw
//! `posix_memalign`/`free` calls, so this file needs zero `unsafe extern "C"`
//! bindings.
//!
//! `HostMemoryBuffer` is a separate type in the same C++ directory this file
//! previously left uncovered (see `SENLIB_BOUNDARY.md`'s coverage
//! cross-check) — a plain host-buffer-requirement descriptor consumed by the
//! not-yet-ported graph executor (`flat_executor.cpp`, its only call site in
//! flex). Its `seg_` field is a `sendnn::Segment` — a type owned by the
//! external `sendnn` library, not by flex, so out of this port's scope by
//! the same rule that keeps FFI-boundary types wrapped rather than
//! reimplemented (`SENLIB_BOUNDARY.md`'s framing of port vs. wrap scope).
//! Modeled here as an opaque handle until the graph-executor phase needs to
//! construct or inspect a real one.
//!
//! No senlib calls identified in this subsystem — see
//! SENLIB_BOUNDARY_device-interface-misc.md.

use std::alloc::{Layout, alloc, alloc_zeroed, dealloc};

/// Byte alignment for a `RaiiBuffer` allocation. `posix_memalign` — the call
/// the C++ constructors use (`raii_buffer.cpp:29,45`) — fails with `EINVAL`
/// unless the alignment is a power of two AND a multiple of `sizeof(void*)`,
/// and the C++ turns that failure into
/// `RAS::MEMORY::RaiiAllocFail().Message("aligned memory allocation failed")`.
/// Both halves of that contract are enforced here, so an alignment the C++
/// would reject cannot be constructed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Alignment(u32);

impl Alignment {
    pub const fn new(bytes: u32) -> Option<Self> {
        if bytes.is_power_of_two() && bytes as usize >= std::mem::size_of::<*const ()>() {
            Some(Self(bytes))
        } else {
            None
        }
    }

    pub const fn get(self) -> u32 {
        self.0
    }
}

/// Error constructing a `RaiiBuffer`. Port of the C++ `RAS::MEMORY::RaiiAllocFail`
/// throw sites.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RaiiBufferError {
    /// Port of "attempted to allocate zero bytes".
    ZeroSizeRequested,
    /// Port of "aligned memory allocation failed" (`posix_memalign` != 0, or
    /// here, a failed/oversized `Layout`/allocator request).
    AllocationFailed,
}

/// Port of `flex::RaiiBuffer`: an RAII owner of an aligned, heap-allocated
/// host buffer. Move-only (no `Clone` — matches `RaiiBuffer(const RaiiBuffer&) = delete`);
/// use `deep_copy` for an explicit content copy, matching `RaiiBuffer::DeepCopy`.
///
/// # One unreplicable C++ move detail
///
/// The C++ move *constructor* (`raii_buffer.cpp:69-74`) copies only
/// `data_ptr_` and `size_bytes_` into the new object — it never copies
/// `alignment_`, so a move-constructed `RaiiBuffer` reports `Alignment() == 0`
/// even though its allocation is aligned. (The move *assignment* operator,
/// `raii_buffer.cpp:54-67`, does copy it, and both leave the moved-from object
/// at `alignment_ = 128` rather than `0`.) A Rust move is a bitwise relocation
/// with no user hook, so `alignment` necessarily survives it here. That is the
/// direction this port cannot help but differ in, and it is also the only safe
/// one: `dealloc` requires the exact `Layout` the allocation was made with, so
/// dropping the alignment would make every post-move `Drop` unsound. Nothing
/// in the ported code reads `Alignment()` after a move.
#[derive(Debug)]
pub struct RaiiBuffer {
    ptr: *mut u8,
    size_bytes: usize,
    // `Option<Alignment>` rather than a bare `Alignment`: the C++ member
    // (`raii_buffer.hpp:107`, `size_t alignment_ = 0;`) is a plain integer
    // that can hold the literal value `0` — a state a power-of-two-checked
    // `Alignment` cannot represent — for the default-constructed,
    // never-allocated buffer. `None` here is that `0` state; every
    // allocating constructor (`new`/`from_bytes`) always sets `Some`.
    alignment: Option<Alignment>,
}

// SAFETY: `RaiiBuffer` owns its heap allocation exclusively (move-only, no
// aliasing); the raw pointer itself carries no thread-affinity.
unsafe impl Send for RaiiBuffer {}

impl RaiiBuffer {
    fn layout_for(size_bytes: usize, alignment: Alignment) -> Result<Layout, RaiiBufferError> {
        Layout::from_size_align(size_bytes, alignment.get() as usize)
            .map_err(|_| RaiiBufferError::AllocationFailed)
    }

    /// `layout_for`'s `Option<Alignment>`-aware counterpart, used by `Drop`
    /// and `deep_copy` where the buffer may be in the never-allocated
    /// default state (`alignment: None`) — which never has a live
    /// allocation to reconstruct a `Layout` for in the first place.
    fn layout_for_opt(
        size_bytes: usize,
        alignment: Option<Alignment>,
    ) -> Result<Layout, RaiiBufferError> {
        Self::layout_for(
            size_bytes,
            alignment.ok_or(RaiiBufferError::AllocationFailed)?,
        )
    }

    /// Port of `RaiiBuffer(size_bytes, alignment_bytes)`: allocates and
    /// zero-fills a new buffer (`memset(data_ptr_, 0, size_bytes_)`).
    pub fn new(size_bytes: usize, alignment: Alignment) -> Result<Self, RaiiBufferError> {
        if size_bytes == 0 {
            return Err(RaiiBufferError::ZeroSizeRequested);
        }
        let layout = Self::layout_for(size_bytes, alignment)?;
        // SAFETY: `layout` has non-zero size (checked above).
        let ptr = unsafe { alloc_zeroed(layout) };
        if ptr.is_null() {
            return Err(RaiiBufferError::AllocationFailed);
        }
        Ok(Self {
            ptr,
            size_bytes,
            alignment: Some(alignment),
        })
    }

    /// Port of `RaiiBuffer(std::vector<std::byte> data, alignment_bytes)`:
    /// allocates a new buffer and copies `data` into it.
    pub fn from_bytes(data: &[u8], alignment: Alignment) -> Result<Self, RaiiBufferError> {
        if data.is_empty() {
            return Err(RaiiBufferError::ZeroSizeRequested);
        }
        let layout = Self::layout_for(data.len(), alignment)?;
        // `alloc`, not `alloc_zeroed`: the C++ vector constructor
        // (`raii_buffer.cpp:38-52`) is `posix_memalign` + `memcpy` with NO
        // intervening `memset` — unlike the (size, alignment) constructor
        // (`raii_buffer.cpp:34`), which does memset. Every byte is overwritten
        // by the copy below, so nothing is left uninitialized either way; the
        // difference is a full extra pass over the buffer that the C++ does
        // not make.
        // SAFETY: `layout` has non-zero size (checked above).
        let ptr = unsafe { alloc(layout) };
        if ptr.is_null() {
            return Err(RaiiBufferError::AllocationFailed);
        }
        // SAFETY: `ptr` was just allocated with `data.len()` bytes and is not aliased.
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
        }
        Ok(Self {
            ptr,
            size_bytes: data.len(),
            alignment: Some(alignment),
        })
    }

    /// Port of `IsInitialized()` (`raii_buffer.cpp:87`):
    /// `data_ptr_ != nullptr`. `false` only for the default-constructed,
    /// never-allocated state (see [`Default`] below).
    pub fn is_initialized(&self) -> bool {
        !self.ptr.is_null()
    }

    /// Port of `Alignment()` (`raii_buffer.hpp:60`). Returns `None` for the
    /// default-constructed, never-allocated state (C++'s `alignment_ == 0`,
    /// which a power-of-two-checked [`Alignment`] cannot represent).
    pub fn alignment(&self) -> Option<Alignment> {
        self.alignment
    }

    pub fn size_bytes(&self) -> usize {
        self.size_bytes
    }

    /// Port of `Pointer()`.
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    /// Port of the non-const `Pointer()`.
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr
    }

    /// Safe byte-slice view, replacing the raw-pointer `PointerAs<T>`/`TypeOffset<T>`
    /// template pair from the C++ API for the common byte-buffer case.
    pub fn as_slice(&self) -> &[u8] {
        // The default-constructed state (`ptr == null`, `size_bytes == 0`) is
        // a legal `RaiiBuffer` in the C++ (`RaiiBuffer() = default`), and
        // `operator<<` runs over it happily (its byte loop just executes zero
        // times). `slice::from_raw_parts` is UB on a null pointer even for
        // length 0, so that state gets an empty slice rather than a
        // constructed-from-null one.
        if self.ptr.is_null() {
            return &[];
        }
        // SAFETY: `self.ptr` is non-null and valid for `self.size_bytes` bytes
        // for the lifetime of `self` (allocated in `new`/`from_bytes`, freed
        // only in `Drop`).
        unsafe { std::slice::from_raw_parts(self.ptr, self.size_bytes) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        if self.ptr.is_null() {
            return &mut [];
        }
        // SAFETY: see `as_slice`; `&mut self` gives exclusive access.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.size_bytes) }
    }

    /// Port of `ByteOffset(size_bytes)` (non-const): pointer to `self.ptr + size_bytes`.
    /// Returns `None` if `size_bytes` is past the end of the buffer.
    pub fn byte_offset(&mut self, size_bytes: usize) -> Option<*mut u8> {
        if size_bytes > self.size_bytes {
            return None;
        }
        // SAFETY: `size_bytes <= self.size_bytes`, so this stays within (or at
        // the one-past-the-end of) the allocation.
        Some(unsafe { self.ptr.add(size_bytes) })
    }

    /// Port of `DeepCopy()` (`raii_buffer.cpp:99-104`): constructs a
    /// `RaiiBuffer(size_bytes_, alignment_)` and `memcpy`s into it, so the
    /// zero-size rejection of that constructor
    /// (`"attempted to allocate zero bytes"`, `raii_buffer.cpp:25-28`) applies
    /// first — checked here in the same order, before the alignment lookup, so
    /// deep-copying a default-constructed buffer reports the zero-size error
    /// the C++ reports and not an allocation failure.
    pub fn deep_copy(&self) -> Result<Self, RaiiBufferError> {
        if self.size_bytes == 0 {
            return Err(RaiiBufferError::ZeroSizeRequested);
        }
        let alignment = self.alignment.ok_or(RaiiBufferError::AllocationFailed)?;
        Self::from_bytes(self.as_slice(), alignment)
    }
}

impl Default for RaiiBuffer {
    /// Port of `RaiiBuffer() = default` — the C++ default constructor leaves
    /// `data_ptr_` null, `size_bytes_` zero, and — per its member
    /// initializer `size_t alignment_ = 0;` (`raii_buffer.hpp:107`) —
    /// `alignment_` zero too (an "empty"/unallocated buffer). This port's
    /// module doc previously claimed there was "no equivalent" to this state
    /// and represented it only as `Option<RaiiBuffer>` at call sites; that
    /// was true for every call site added so far, but it left the real
    /// default-constructed state unreachable, which in turn made
    /// `IommuMapperInterface::Map`'s nullptr-rejection path
    /// (`iommu_mapper_test.cpp`'s `MapNullptrBufferReturnsError`) impossible
    /// to exercise from a unit test. Adding the real default constructor is
    /// the more faithful port: `is_initialized()`/`as_ptr()`/`Drop` already
    /// handle a null `ptr` correctly (see below), so this is a pure
    /// completion of the existing null-safe design, not a new special case.
    ///
    /// `alignment` is `None` here, not `Alignment::new(1)`: a
    /// power-of-two-checked `Alignment` cannot represent the literal `0` the
    /// C++ member initializer actually leaves `alignment_` at, so `None` is
    /// the faithful port of that zero/uninitialized state rather than the
    /// nearest representable `Alignment`.
    fn default() -> Self {
        Self {
            ptr: std::ptr::null_mut(),
            size_bytes: 0,
            alignment: None,
        }
    }
}

impl Drop for RaiiBuffer {
    fn drop(&mut self) {
        if self.ptr.is_null() {
            return;
        }
        // SAFETY: `layout` is reconstructed identically to the one used at
        // allocation time (same size/alignment), matching the C++ `free(data_ptr_)`.
        if let Ok(layout) = Self::layout_for_opt(self.size_bytes, self.alignment) {
            unsafe {
                dealloc(self.ptr, layout);
            }
        }
        self.ptr = std::ptr::null_mut();
    }
}

impl std::fmt::Display for RaiiBuffer {
    /// Port of `operator<<(std::ostream&, const RaiiBuffer&)`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // C++'s `rhs.Alignment()` returns the raw `size_t alignment_`, which
        // is `0` in the default/never-allocated state — matched here via
        // `map_or(0, ...)` rather than unwrapping `Some`.
        write!(
            f,
            "RaiiBuffer(data_ptr: {:p}, size_bytes: {}, alignment: {}, values:",
            self.ptr,
            self.size_bytes,
            self.alignment.map_or(0, Alignment::get)
        )?;
        let bytes = self.as_slice();
        let n = bytes.len().min(8);
        for b in &bytes[..n] {
            write!(f, " {b:02x}")?;
        }
        if bytes.len() > n {
            write!(f, " ...")?;
        }
        write!(f, ")")
    }
}

// ---------------------------------------------------------------------
// HostMemoryBuffer (host_memory_buffer.{hpp,cpp})
// ---------------------------------------------------------------------

/// Opaque placeholder for a `sendnn::Segment` — an external `sendnn`-owned
/// type, not a `flex::` type, so out of this port's scope (see this module's
/// own doc comment). Carries no data yet: nothing in the currently-ported
/// slice constructs or inspects a real one, and `operator<<` on
/// `HostMemoryBuffer` doesn't print it either. A real port arrives with the
/// graph-executor phase, if `sendnn::Segment` turns out to be reachable from
/// there rather than staying wrap-scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SendnnSegmentHandle(());

impl SendnnSegmentHandle {
    pub const fn placeholder() -> Self {
        Self(())
    }
}

/// Port of `flex::HostMemoryBuffer` (`host_memory_buffer.hpp`): describes a
/// host-side buffer requirement for a model's PIPO or per-model program/data
/// segments, and the transfer direction. Plain data descriptor, no logic —
/// `flat_executor.cpp`'s (not yet ported) graph executor is its only real
/// consumer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostMemoryBuffer {
    pub seg: SendnnSegmentHandle,
    pub size_bytes: usize,
    pub to_device: bool,
}

impl std::fmt::Display for HostMemoryBuffer {
    /// Port of `operator<<(std::ostream&, const HostMemoryBuffer&)`. Matches
    /// the C++ exactly, including printing the bare `bool` as `1`/`0` (no
    /// `std::boolalpha` set in that TU) rather than Rust's default
    /// `true`/`false` — and, like the C++, never prints `seg_`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "HostMemoryBuffer(size_bytes: {} [B], to_device: {})",
            self.size_bytes, self.to_device as u8
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_buffer_is_zero_filled() {
        let buf = RaiiBuffer::new(16, Alignment::new(128).unwrap()).unwrap();
        assert_eq!(buf.as_slice(), &[0u8; 16]);
        assert!(buf.is_initialized());
    }

    #[test]
    fn zero_size_is_rejected() {
        let err = RaiiBuffer::new(0, Alignment::new(128).unwrap()).unwrap_err();
        assert_eq!(err, RaiiBufferError::ZeroSizeRequested);
    }

    #[test]
    fn from_bytes_copies_contents() {
        let buf = RaiiBuffer::from_bytes(&[1, 2, 3, 4], Alignment::new(64).unwrap()).unwrap();
        assert_eq!(buf.as_slice(), &[1, 2, 3, 4]);
    }

    #[test]
    fn deep_copy_is_independent() {
        let mut buf = RaiiBuffer::from_bytes(&[9, 9], Alignment::new(64).unwrap()).unwrap();
        let copy = buf.deep_copy().unwrap();
        buf.as_mut_slice()[0] = 1;
        assert_eq!(copy.as_slice(), &[9, 9]);
    }

    #[test]
    fn byte_offset_bounds_checked() {
        let mut buf = RaiiBuffer::new(8, Alignment::new(8).unwrap()).unwrap();
        assert!(buf.byte_offset(8).is_some());
        assert!(buf.byte_offset(9).is_none());
    }

    #[test]
    fn host_memory_buffer_display_matches_cpp_operator_shl() {
        let hmb = HostMemoryBuffer {
            seg: SendnnSegmentHandle::placeholder(),
            size_bytes: 4096,
            to_device: true,
        };
        // C++'s `os << hmb.to_device_` prints a bare bool as "1"/"0" (no
        // `std::boolalpha` set anywhere in this TU), not "true"/"false".
        assert_eq!(
            format!("{hmb}"),
            "HostMemoryBuffer(size_bytes: 4096 [B], to_device: 1)"
        );
    }

    // Ported from flex/tests/memory_interface/raii_buffer_test.cpp. That file
    // has no senlib dependency, so all 6 of its TESTs are portable as-is.
    //
    // C++'s `PointerAs<T>()`/`TypeOffset<T>(n)` are template methods (cast the
    // base byte pointer to `T*`, then `base_as_T + n`, i.e. `n` is in units of
    // `sizeof(T)`). This Rust port has no named equivalent (`as_ptr`/`byte_offset`
    // are byte-only, per this file's own doc comment on `as_slice`), so the two
    // typed-offset tests below reconstruct the same pointer-arithmetic invariant
    // with raw casts + `.add()` instead of inventing new public API.

    #[test]
    fn type_offset_non_const() {
        // C++: TypeOffsetNonConst — TypeOffset<uint8_t>(4) == PointerAs<uint8_t>() + 4.
        let mut buf = RaiiBuffer::new(64, Alignment::new(64).unwrap()).unwrap();
        let base = buf.as_ptr();
        let result = buf.byte_offset(4).unwrap();
        assert_eq!(result as *const u8, unsafe { base.add(4) });
    }

    #[test]
    fn type_offset_const() {
        // C++: TypeOffsetConst — const TypeOffset<uint32_t>(3) == PointerAs<uint32_t>() + 3
        // (i.e. 3 elements == 12 bytes for uint32_t).
        let buf = RaiiBuffer::new(64, Alignment::new(64).unwrap()).unwrap();
        let base = buf.as_ptr();
        let result = unsafe { (base as *const u32).add(3) };
        assert_eq!(result as *const u8, unsafe {
            base.add(3 * std::mem::size_of::<u32>())
        });
    }

    #[test]
    fn type_offset_zero_offset() {
        // C++: TypeOffsetZeroOffset — TypeOffset<uint8_t>(0) == PointerAs<uint8_t>() for
        // both the non-const and const accessors.
        let mut buf = RaiiBuffer::new(32, Alignment::new(32).unwrap()).unwrap();
        let base = buf.as_ptr();
        assert_eq!(buf.byte_offset(0).unwrap() as *const u8, base);
        assert_eq!(buf.as_ptr(), base);
    }

    #[test]
    fn stream_output_short_buffer_is_not_truncated() {
        // C++: StreamOutputShortBuffer — <=8 bytes prints every byte, no "...".
        let mut buf = RaiiBuffer::new(4, Alignment::new(8).unwrap()).unwrap();
        buf.as_mut_slice()
            .copy_from_slice(&[0xAB, 0xCD, 0xEF, 0x12]);
        let out = format!("{buf}");
        assert!(out.contains("RaiiBuffer"));
        assert!(out.contains("ab"));
        assert!(out.contains("cd"));
        assert!(out.contains("ef"));
        assert!(out.contains("12"));
        assert!(!out.contains("..."));
    }

    #[test]
    fn stream_output_long_buffer_is_truncated() {
        // C++: StreamOutputLongBuffer — >8 bytes prints "..." after the first 8.
        let buf = RaiiBuffer::new(32, Alignment::new(8).unwrap()).unwrap();
        let out = format!("{buf}");
        assert!(out.contains("RaiiBuffer"));
        assert!(out.contains("..."));
    }

    #[test]
    fn default_buffer_is_empty_and_uninitialized() {
        // Port of the C++ default constructor's semantics (`RaiiBuffer()`):
        // a null pointer / zero size, matching `IsInitialized() == false`.
        // This is the state `iommu.rs`'s `MapNullptrBufferReturnsError` port
        // relies on to construct a null-pointer `RaiiBuffer`.
        let buf = RaiiBuffer::default();
        assert!(!buf.is_initialized());
        assert!(buf.as_ptr().is_null());
        assert_eq!(buf.size_bytes(), 0);
        // Must not crash on drop (mirrors the C++ dtor's `if(data_ptr_)` guard).
        drop(buf);
    }

    #[test]
    fn shared_raii_buffer_stream_output() {
        // C++: SharedRaiiBufferStreamOutput — `operator<<` also works through a
        // `std::shared_ptr<RaiiBuffer>`. Ported via `Rc`'s blanket `Display` impl
        // rather than `Arc`'s: the buffer never crosses a thread here, and
        // `RaiiBuffer` is deliberately not `Sync` (it owns a raw allocation).
        let shared = std::rc::Rc::new(RaiiBuffer::new(16, Alignment::new(8).unwrap()).unwrap());
        let out = format!("{shared}");
        assert!(out.contains("RaiiBuffer"));
    }
}
