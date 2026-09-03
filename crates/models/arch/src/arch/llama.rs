include!(concat!(env!("OUT_DIR"), "/llama.rs"));

// Also gated on the `tinyllama-1.1b` model feature specifically (model
// selection is fully opt-in, no default) — this test hardcodes that one
// model's emitted module path, so it can only compile when that model was
// actually selected.
#[cfg(all(test, feature = "metal", feature = "tinyllama-1.1b"))]
mod metal_emission_tests {
    /// Sanity-check that the macro emits the per-canonical metal
    /// surface (real Weights struct + accessor methods + load fn +
    /// METAL_BUCKETS static + metal_pool fn) for at least one model
    /// in this arch. The check is structural — it doesn't run the
    /// loader, just asserts the symbols exist and resolve to the
    /// expected types.
    #[test]
    fn tinyllama_metal_symbols_resolve() {
        // METAL_BUCKETS is a non-empty `&[MetalBucketSpec]`.
        let buckets: &[::scratchy_target_metal::interpreter::metal::MetalBucketSpec] =
            super::tinyllama_1_1b::METAL_BUCKETS;
        assert!(
            !buckets.is_empty(),
            "TinyLlama-1.1B emits at least one bucket"
        );
        // metal_pool resolves as an `fn(...) -> Result<MetalWorkerPool, PoolBuildError>`.
        // We don't call it (no Device available in unit-test ctx); just
        // taking the fn-pointer proves the symbol + signature compiled.
        let _ctor: fn(
            ::std::sync::Arc<::scratchy_target_metal::interpreter::metal::__re::Device>,
            &super::tinyllama_1_1b::Weights,
            ::std::sync::Arc<::scratchy_target_metal::MetalAllocator>,
            ::scratchy_target_metal::interpreter::metal::RuntimeFactory,
            usize,
            usize,
        ) -> ::core::result::Result<
            ::scratchy_target_metal::interpreter::metal::MetalWorkerPool<
                super::tinyllama_1_1b::Weights,
            >,
            ::scratchy_target_metal::interpreter::metal::PoolBuildError,
        > = super::tinyllama_1_1b::metal_pool;
        // load() resolves as a stream-free fn returning `Result<Weights>`.
        // Same fn-pointer-only check; no GpuWeights instance available in
        // unit-test ctx.
        let _loader: fn(
            &mut ::scratchy_target_metal::weights::GpuWeights,
            ::scratchy_target_metal::CUstream,
            usize,
            u8,
        ) -> ::anyhow::Result<super::tinyllama_1_1b::Weights> = super::tinyllama_1_1b::load;
    }
}
