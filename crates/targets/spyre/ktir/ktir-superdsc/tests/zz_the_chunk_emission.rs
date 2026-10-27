// SPDX-License-Identifier: Apache-2.0
//! The LX core-local pin's division law.
//!
//! (The split-KV chunk's LX-resident form and its table builders — `ChunkScratch::lx_form`,
//! `chunk_gather_index_table` — were deleted with the copy-gather composers, and their tests with
//! them: the live chunk kernels gather straight from HBM through the block-entry tables.)

#[test]
fn a_core_local_tensor_refuses_a_second_division() {
    use ktir_superdsc::place::PlaceId;
    use ktir_superdsc::placement::BundleLayout;
    let l = BundleLayout::default();
    l.synth_lx_core_local(PlaceId::Act(7), 4096);
    let name = PlaceId::Act(7).to_string();
    assert!(l.lx_core_local(&name));
    assert_eq!(l.lx_pin(&name).map(|(_, fp)| fp), Some(4096));
    assert_eq!(
        l.record_lx_division(&name, "esub", "split {out: 32}".into()),
        Ok(())
    );
    assert_eq!(
        l.record_lx_division(&name, "exp", "split {out: 32}".into()),
        Ok(())
    );
    assert!(
        l.record_lx_division(&name, "sum", "split {mb: 32}".into())
            .is_err()
    );
}
