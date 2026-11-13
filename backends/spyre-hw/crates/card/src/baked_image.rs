//! The bake's packed device images, decoded — the load half of the metal AOT precedent
//! (`crates/targets/metal/src/tape/lowered.rs`, `BakedMetallib::decompress`): the dictionary is
//! digested once and shared by every frame of its build, each frame decompresses to its baked
//! length, and a mismatch names both numbers.
//!
//! The bytes live as data in `scratchy-spyre-bundle` ([`BakedImages`]): one blob of frames, one
//! dictionary, one row table, `include_bytes!`'d by the macro. Decoding is card-runtime behavior,
//! so it lives here — the bundle crate is a leaf both ends of the pipeline name, and neither end
//! owns a decoder.

use scratchy_spyre_bundle::BakedImages;
use std::collections::HashMap;
use std::sync::Mutex;

/// One prepared dictionary per `BakedImages` static, keyed by the dictionary's address: a `static`
/// outlives the process, so its address is its identity. Leaked once, shared by every frame of that
/// build — metal's `BakedDictionary` holds the same one-digest-per-build behind its `OnceLock`,
/// keyed the only way a leaf-of-bytes table can offer.
static DICTS: Mutex<Option<HashMap<usize, &'static zstd::dict::DecoderDictionary<'static>>>> =
    Mutex::new(None);

/// The dictionary `bytes` name, prepared for decoding. First use digests it; every later frame of
/// the same build shares it.
fn prepared(bytes: &'static [u8]) -> &'static zstd::dict::DecoderDictionary<'static> {
    let mut dicts = DICTS.lock().expect("the baked-image dictionary table is not poisoned");
    dicts
        .get_or_insert_with(HashMap::new)
        .entry(bytes.as_ptr() as usize)
        .or_insert_with(|| Box::leak(Box::new(zstd::dict::DecoderDictionary::copy(bytes))))
}

/// The image at `row`, decompressed. The row and its table are both generated facts — the emission's
/// interner mints them together — so a row past the table is unconstructible and indexes directly.
/// The baked `len` is the check, exactly as metal's is: a frame that decompresses to any other size
/// names both numbers and refuses.
pub fn decompress(images: &BakedImages, row: u32) -> Result<Vec<u8>, String> {
    let frame = &images.frames[row as usize];
    let dictionary = prepared(images.dictionary);
    let mut decompressor = zstd::bulk::Decompressor::with_prepared_dictionary(dictionary)
        .map_err(|e| format!("baked image dictionary: {e}"))?;
    let bytes = decompressor
        .decompress(
            &images.blob[frame.offset as usize..frame.offset as usize + frame.frame_len as usize],
            frame.len as usize,
        )
        .map_err(|e| format!("baked image frame {row}: {e}"))?;
    if bytes.len() != frame.len as usize {
        return Err(format!(
            "baked image {row}: {} bytes decompressed, {} baked",
            bytes.len(), frame.len
        ));
    }
    Ok(bytes)
}
