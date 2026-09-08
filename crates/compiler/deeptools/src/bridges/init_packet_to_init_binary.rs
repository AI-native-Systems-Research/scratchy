//! BRIDGE 4 — island 4 → bytes. **SERIALISATION**, into 128-byte flits.
//!
//! ⛔ NO DECISIONS. Every value written here was decided upstream, so a fault in the bytes is a fault in bridge 2
//! or 3 and must be fixed there. A serialiser that "fixes up" a value is a second place that value is decided,
//! and the one it disagrees with is whichever the reader did not look at.
//!
//! The one thing it does compute is each packet's flit COUNT, and that is derived from the flits rather than taken
//! from a field — see [`crate::islands::init_packet::Packet::header_with_count`]. The packet chain is walked by that
//! number, so a stale one does not corrupt a packet, it desynchronises every packet after it.
//!
//! `dip.cpp:60-123` — the QGI header, the flit walk, the zero padding, and the size handed to the runtime.

use crate::islands::init_packet::{Flit, PAD_TO_FLITS, Packets, QgiHeader, Slice};

/// THE BYTES. One `init_binary.bin` for one group's packets.
///
/// ⭐ A CONCATENATION, WHICH IS THE WHOLE OF IT. Each packet contributes its header flit followed by its own
/// flits, and the header states how many follow — so the file is walkable forwards without a table of contents,
/// and `f + myflits + 1` lands on the next packet.///
/// ⭐ SPECIALISED ON THE WHOLE MODEL AND THE MACHINE, LIKE EVERY OTHER BRIDGE. `M` carries every shape a
/// `config.json` declares as associated consts, and `PT_ROWS`/`STICK_BYTES` are the two the architecture fixes.
/// A bridge that lacked them while its passes had them would be the one place a model fact had to become a
/// runtime argument again — which is the whole failure this threading exists to prevent.
pub fn serialise<A: crate::consts::Arch>(
    packets: &Packets,
    cores: crate::bridges::senprog_to_init_packet::CoreCount,
) -> Vec<u8> {
    let mut out = Vec::with_capacity(flit_total(packets) * Flit::BYTES);
    for packet in packets {
        out.extend_from_slice(&packet.header_with_count().flit().to_le_bytes());
        for flit in &packet.flits {
            out.extend_from_slice(&flit.to_le_bytes());
        }
    }
    // ⭐ THE FILE IS ROUNDED UP TO THE INIT ALLOCATION GRANULARITY WITH ZERO FLITS — `dip.cpp:70-75` computes
    // `gran = progInitAddrGranularity / bytesPerStick` and `:112-119` writes zero flits up to it, which is also the
    // size handed to the runtime (`:123`). So the padding is part of the artefact, not slack the writer may omit.
    //
    // ⛔ AND IT IS AN ARCH FACT: 128 bytes on RCUDD1A and 256 on SEN1P5 (`sysdef.cpp:538-540`) against a 128-byte
    // stick (`:181`), so this is a no-op on one arch and pads to an even flit count on the other. Testing only
    // RCUDD1A would never show it.
    let pad = PAD_TO_FLITS * Flit::BYTES;
    let over = out.len() % pad;
    if over != 0 {
        out.resize(out.len() + (pad - over), 0);
    }
    assert!(
        out.len().is_multiple_of(pad),
        "an init binary is a whole number of {PAD_TO_FLITS}-flit allocation units"
    );
    // ⛔⛔ THE CHAIN IS WALKED, NOT ASSUMED, AND THIS IS WHY IT MATTERS NOW. The image is ONE program per op
    // (`autopilot` concatenates them, `dxp.cpp:836-870`), and the load state machine finds the next program at
    // `f + myflits + 1` — so a single wrong count makes it walk past the sentinel and never reach it. The card
    // accepts such a control block and never completes it, which is a HANG rather than an error: the host waits on
    // a completion fence that cannot arrive.
    //
    // ⭐ AND THE READER IS THE WRITER'S INVERSE, so this catches a disagreement between a header's count and its
    // body — the one defect that leaves every byte round-tripping exactly.
    let walked = packet_starts(&out);
    assert!(
        walked.len() == packets.len(),
        "the image holds {} programs and its chain walks {} of them: a header's flit count disagrees with its          body, so the load state machine cannot reach the sentinel that ends the control block",
        packets.len(),
        walked.len()
    );
    // ⛔⛔⛔ AND THE IMAGE MUST BE ONE `dip` CAN WALK — see [`refuse_an_image_dip_cannot_walk`]. The chain
    // check above walks PACKETS, which is our own reading of our own bytes; this walks BLOCKS the way
    // IBM's reader does, and it is the only check here whose rules this crate did not write.
    if let Err(defect) = refuse_an_image_dip_cannot_walk(&out) {
        panic!("{defect}");
    }
    // ⛔⛔⛔ AND A PATCH IMAGE MUST BE ONE `reversePatchInit` CAN APPLY — see the function's own doc. The walk
    // above RETURNS at the first patch header, faithfully, which leaves a patch image structurally unchecked;
    // this is the check that covers it, and it is the one IBM's reader used to reject every malformed image
    // this crate produced while patch init was being ported.
    if let Err(defect) = refuse_a_patch_image_the_reader_cannot_apply(&out, cores) {
        panic!("{defect}");
    }
    out
}

/// ⛔⛔⛔ WHAT `dip` ITSELF REFUSES — a port of `Dip::isPatchInit` (`patchinit.cpp:1253-1279`), read as a
/// VALIDATOR rather than as the predicate it is spelled as.
///
/// ⭐⭐⭐ THIS IS THE ONLY CHECK IN THE CRATE WHOSE RULES WE DID NOT WRITE. Everything else that reads these
/// bytes — `packet_starts`, the senulator, the ISA tables — is OUR reading of OUR encoding, so an emitter bug
/// and a reader bug that agree are invisible to both. `isPatchInit` walks the block chain of every slice with
/// IBM's own field offsets and ends `DT_CHECK(rowIdx == numFlits)`: an image whose blocks do not tile its flits
/// exactly cannot be walked, and the card's init state machine walks it the same way.
///
/// ⛔⛔ A REGULAR BLOCK TARGETS EXACTLY ONE CORE. `if (targetCore & (targetCore - 1)) return true;`
/// (`:1268-1269`) — the not-a-power-of-two test, commented *"targeting more than one core"* — means a
/// multi-core mask IS a patch-init image BY DEFINITION, not a compact spelling of a regular one. dxp's size win
/// comes from emitting patch-init, an encoding with its own patch headers and LX-address bookkeeping
/// (`reversePatchInit`, `:1075-1130`); a multi-core mask on a REGULAR block is that encoding's signature
/// without its content, and `updateLxAddress`'s `DT_CHECK(check.size() == 1)` (`:31`) is where the reader
/// discovers it.
///
/// # Errors
/// [`ImageDefect`] — the condition, the slice and the flit, so the refusal names where in the image it is.
pub fn refuse_an_image_dip_cannot_walk(bytes: &[u8]) -> Result<(), ImageDefect> {
    let flits = bytes.len() / Flit::BYTES;
    // `initslicepart[j]` of slice `s` in flit `f` — the layout `loadInitFromMem` fills
    // (`dip.cpp:1030-1038`: `initBin[flitNum * 32 + i * 4 + j]`).
    let word = |flit: usize, slice: usize, j: usize| -> u32 {
        let at = flit * Flit::BYTES + slice * Slice::BYTES + j * 4;
        u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
    };
    // What the walk learns about the image as a whole, judged after it.
    let mut shared: Option<(usize, usize, u32)> = None;
    for slice in 0..Flit::SLICES {
        // Row 0 is the QG header, which `isPatchInit` leaves out (`:1256`).
        let mut row = 1;
        while row < flits {
            let v = word(row, slice, 2);
            // ⛔⛔⛔ A PATCH HEADER ENDS THE WALK, AND THAT IS `isPatchInit`'s OWN CONTROL FLOW.
            // `if (slice_id == 0 && (v & 1) == 1 && (v >> 21) == 0) return true;` (`patchinit.cpp:1259-1261`)
            // RETURNS — slice 0 is walked first, so a patch image never reaches slices 1..7 and its
            // `DT_CHECK(rowIdx == numFlits)` (`:1275`) never runs. The per-slice tiling law is the REGULAR
            // image's; a patch flit's slices 1..7 hold replacement DATA, and walking them as block headers
            // reads a data word as a header. MEASURED, when this walked on: a constant five-flit overrun
            // ("stopped at flit 457 of 452") on every group, which is those data slices being stepped over.
            if slice == 0 && (v & 1) == 1 && (v >> 21) == 0 {
                return Ok(());
            }
            // "null header if bits 127:71 is all 0" (`:1262-1265`) — also how the zero padding is consumed.
            if v & 0xFFFF_FF80 == 0 {
                row += 1;
                continue;
            }
            // ⭐ A MULTI-CORE MASK IS THE SHARED BASE OF A PATCH-INIT IMAGE — what dxp emits, and what this
            // crate now emits. It is a defect only when the image carries NO patch flit to complete it, which
            // is judged once the whole image has been walked.
            let target_core = word(row, slice, 3);
            if target_core & target_core.wrapping_sub(1) != 0 && shared.is_none() {
                shared = Some((row, slice, target_core));
            }
            // bit 77:73, 84:78 and 69:64 (`:1270-1272`).
            let lrf = ((v >> 9) & 0x1F) as usize;
            let ibuff = ((v >> 14) & 0x7F) as usize;
            let corr = (v & 0x3F) as usize;
            row += 2 + lrf + ibuff + corr;
        }
        // `DT_CHECK(rowIdx == numFlits)` (`:1275`) — the walk must land exactly on the end.
        if row != flits {
            return Err(ImageDefect::WalkOverran {
                slice,
                stopped_at: row,
                flits,
            });
        }
    }
    // ⛔⛔⛔ A SHARED BASE WITHOUT PATCHES IS THE SHAPE THAT FENCED THE CARD. `isPatchInit` reads a mask naming
    // more than one core as a PATCH-INIT image BY DEFINITION (`patchinit.cpp:1268-1269`), so an image carrying
    // one OWES the patch flits that complete it; without them IBM's reader enters `reversePatchInit`, finds no
    // patch bodies, and dies in `updateLxAddress`'s `DT_CHECK(check.size() == 1)` (`:31`).
    // ⭐ REACHING HERE MEANS NO PATCH HEADER WAS FOUND — the walk above returns as soon as one is. So a
    // multi-core mask seen on the way is a shared base with nothing to complete it, which is the shape that
    // fenced the card.
    if let Some((flit, slice, core_mask)) = shared {
        return Err(ImageDefect::SharedBaseWithoutPatches {
            flit,
            slice,
            core_mask,
            cores: core_mask.count_ones(),
        });
    }
    Ok(())
}

/// ⛔⛔⛔ WHAT `reversePatchInit` REFUSES — a port of `updateLxAddress`'s `DT_CHECK(check.size() == 1)`
/// (`patchinit.cpp:21-32`), read as a validator.
///
/// ⭐⭐ THIS IS THE CHECK THAT CAUGHT EVERY MALFORMED IMAGE THIS CRATE PRODUCED. While patch init was being
/// ported, IBM's reader rejected each wrong shape here and nowhere else: a shared base with no patches, a base
/// one flit short, a base one flit long. Every one of those was found by running `reverse_dip_standalone`
/// BY HAND. On the compilation path the bake finds them itself.
///
/// THE INVARIANT: every core is at lx address ONE to begin with — `std::vector<int>
/// coreslxAdds(initcores.size(), 1)` (`:1087`) — and the image moves them. A BLOCK advances only the cores its
/// own header names, by its own length (`pushFlit`, `:1192-1197`). An ALL-NULL row is a separator that advances
/// every core it names by one (`:1146-1156`). So when a separator names a set of cores, THEY MUST ALREADY
/// AGREE: `check.size() == 1`. A base that leaves one core's column longer than another's breaks exactly this.
///
/// ⛔ A PATCH FLIT IS STEPPED OVER (`rowIdx += 1`, `:1137`) and does not move any core.
///
/// # Errors
/// [`PatchDefect`] — which flit, which cores, and the addresses they had reached.
pub fn refuse_a_patch_image_the_reader_cannot_apply(
    bytes: &[u8],
    cores: crate::bridges::senprog_to_init_packet::CoreCount,
) -> Result<(), PatchDefect> {
    let flits = bytes.len() / Flit::BYTES;
    let word = |flit: usize, slice: usize, j: usize| -> u32 {
        let at = flit * Flit::BYTES + slice * Slice::BYTES + j * 4;
        u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
    };
    // `isNullHeader` (`patchinit.cpp:14-18`): words 0..2 zero AND a non-zero core mask.
    let null_slice = |flit: usize, slice: usize| -> bool {
        word(flit, slice, 0) == 0
            && word(flit, slice, 1) == 0
            && word(flit, slice, 2) == 0
            && word(flit, slice, 3) != 0
    };

    // ⛔⛔⛔ THE CORE SET IS GIVEN, NOT RECOVERED FROM THE BYTES. `reversePatchInit` is handed `initcores` and
    // starts every one of them at one — `std::vector<int> coreslxAdds(initcores.size(), 1)`
    // (`patchinit.cpp:1087`). This used to SCAN the image instead, seeding a core for any bit set in any slice's
    // word 3 of any flit, which is wrong twice over: a PATCH flit's slices 1..7 hold replacement DATA rather than
    // block headers (see [`refuse_an_image_dip_cannot_walk`]), so a data word could invent a core; and the scan
    // ran to `u32::BITS` while `updateLxAddress` bounds its loop by `initcores.size()` (`:25`) and never looks at
    // the bits above it.
    //
    // ⭐ A CHECK WHOSE RULES WE WROTE IS NOT THE CHECK THIS IS FOR. Its whole value is being IBM's reader applied
    // to our bytes; two of its rules were ours, and both are gone.
    let mut reached: std::collections::BTreeMap<u32, u32> =
        (0..cores.get()).map(|core| (core, 1u32)).collect();

    let mut row = 1;
    while row < flits {
        let v = word(row, 0, 2);
        // A patch flit moves nothing (`:1137`).
        if (v & 1) == 1 && (v >> 21) == 0 {
            row += 1;
            continue;
        }
        if (0..Flit::SLICES).all(|slice| null_slice(row, slice)) {
            // A null separator: advance the cores it names, and they must have agreed (`:21-32`).
            let mask = word(row, 0, 3);
            let mut seen: Vec<(u32, u32)> = Vec::new();
            for (core, at) in &mut reached {
                if mask & (1 << core) != 0 {
                    *at += 1;
                    seen.push((*core, *at));
                }
            }
            let mut addresses: Vec<u32> = seen.iter().map(|(_, at)| *at).collect();
            addresses.sort_unstable();
            addresses.dedup();
            // ⛔⛔ `DT_CHECK(check.size() == 1)` FAILS AT ZERO TOO, and reading it as "more than one is bad"
            // drops half of it. `check` is a `std::set` of the addresses of the cores the separator NAMED, so an
            // empty one means the separator named no core this image has — with the walk correctly bounded by
            // `initcores.size()`, a mask whose only bits sit above that bound reaches exactly this state.
            match addresses.len() {
                1 => {}
                0 => {
                    return Err(PatchDefect::SeparatorNamesNoCore {
                        flit: row,
                        core_mask: mask,
                        cores: cores.get(),
                    });
                }
                _ => {
                    return Err(PatchDefect::CoresOutOfStep {
                        flit: row,
                        core_mask: mask,
                        lowest: *addresses.first().expect("addresses is not empty"),
                        highest: *addresses.last().expect("addresses is not empty"),
                    });
                }
            }
            row += 1;
            continue;
        }
        // A block: its length is the LONGEST of its slices (`getNumFlitsFromHeader`, `:42-48`), and it advances
        // only the cores its own header names (`:1188-1209`).
        let mut span = 0usize;
        let mut mask = 0u32;
        for slice in 0..Flit::SLICES {
            if null_slice(row, slice) {
                continue;
            }
            let w = word(row, slice, 2);
            let here =
                ((w >> 9) & 0x1F) as usize + ((w >> 14) & 0x7F) as usize + (w & 0x3F) as usize;
            span = span.max(here);
            if w >> 21 != 0 && word(row, slice, 3) != 0 {
                mask = word(row, slice, 3);
            }
        }
        let span = span + 2;
        for (core, at) in &mut reached {
            if mask & (1 << core) != 0 {
                *at += u32::try_from(span).expect("a block span that fits");
            }
        }
        row += span;
    }
    Ok(())
}

// ⛔⛔⛔ THE `core_initFlits % 4` RULE DOES **NOT** SURVIVE PATCH INIT — DO NOT CHECK IT HERE.
// `dip.cpp:2176-2193` pads every core's REGULAR init to a multiple of four *"so that QGI can inject the next
// header when datasync = 0"*, and `lay_blocks` emits that pad. It is tempting to require the same of each
// core's run as `reversePatchInit` reconstructs it. MEASURED AGAINST dip's OWN OUTPUT: its golden for a
// 28-core group reconstructs to SEVEN flits per core, so such a check rejects IBM's own artefact. Whatever
// the rule is on the far side of the patch transform, it is not this one.

/// WHY `reversePatchInit` COULD NOT APPLY THE IMAGE.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PatchDefect {
    /// A null separator names cores that are not at the same lx address.
    CoresOutOfStep {
        flit: usize,
        core_mask: u32,
        lowest: u32,
        highest: u32,
    },
    /// A null separator names no core this image has — the other half of `check.size() == 1`.
    SeparatorNamesNoCore {
        flit: usize,
        core_mask: u32,
        cores: u32,
    },
}

impl std::fmt::Display for PatchDefect {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CoresOutOfStep {
                flit,
                core_mask,
                lowest,
                highest,
            } => write!(
                f,
                "the null separator at flit {flit} names cores {core_mask:#x}, and they are NOT at the same \
                 lx address — some have reached {lowest} and others {highest}. `updateLxAddress` advances \
                 every core a separator names and then requires them to agree, `DT_CHECK(check.size() == 1)` \
                 (`patchinit.cpp:21-32`); they disagree when the shared base leaves one core's column longer \
                 than another's, so this image is one IBM's own reader would reject"
            ),
            Self::SeparatorNamesNoCore {
                flit,
                core_mask,
                cores,
            } => write!(
                f,
                "the null separator at flit {flit} names cores {core_mask:#x}, and this image is for {cores} \
                 cores — so it names NONE of them. `updateLxAddress` collects the addresses of the cores a \
                 separator names and requires EXACTLY one, `DT_CHECK(check.size() == 1)` \
                 (`patchinit.cpp:21-32`); an empty set fails that check just as a disagreeing one does, and its \
                 loop stops at `initcores.size()` so bits above the core count name nothing at all"
            ),
        }
    }
}

/// WHY `dip` COULD NOT WALK THE IMAGE.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageDefect {
    /// A shared base under a multi-core mask with no patch flit to complete it.
    SharedBaseWithoutPatches {
        flit: usize,
        slice: usize,
        core_mask: u32,
        cores: u32,
    },
    /// The block chain did not tile the image exactly.
    WalkOverran {
        slice: usize,
        stopped_at: usize,
        flits: usize,
    },
}

impl std::fmt::Display for ImageDefect {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SharedBaseWithoutPatches {
                flit,
                slice,
                core_mask,
                cores,
            } => write!(
                f,
                "the block at flit {flit} slice {slice} is addressed to {cores} cores at once \
                 (core_mask {core_mask:#x}) and this image carries NO patch flit. `isPatchInit` reads a mask \
                 naming more than one core as a PATCH-INIT image by definition (`patchinit.cpp:1268-1269`), so \
                 an image carrying one owes the patches that complete it — without them IBM's reader enters \
                 `reversePatchInit`, finds no patch bodies, and dies in `updateLxAddress`'s \
                 `DT_CHECK(check.size() == 1)` (`patchinit.cpp:31`)"
            ),
            Self::WalkOverran {
                slice,
                stopped_at,
                flits,
            } => write!(
                f,
                "slice {slice}'s block chain stopped at flit {stopped_at} of {flits}: the blocks do not tile \
                 the image, so every block after the first wrong flit count is read at the wrong offset \
                 (`patchinit.cpp:1275`)"
            ),
        }
    }
}

/// THE IMAGE AS `dip`'s OWN `init.txt` — one line per flit, a port of `exportInit` (`dip.cpp:1098-1112`).
///
/// ⛔ SLICE 7 FIRST, AND WORD 3 FIRST WITHIN EACH SLICE. `for (int slice_id = 7; slice_id >= 0; slice_id--)` then
/// `for (int j = 3; j >= 0; j--)`, each printed `std::setw(8) << std::setfill('0')` in `std::hex` — so a line reads
/// the flit as one 1024-bit number, most significant slice first. That is NOT the order the image holds its bytes
/// in, and printing our own order would make every line differ for a reason that is about the dump.
///
/// ⭐ IT WALKS THE BYTES, NOT THE PACKETS, so what it prints is exactly the artefact — the padding flits included —
/// and there is no second walk that could disagree with [`serialise`].
///
/// ⭐ WHY IT EXISTS: `dip_standalone -s <senprog> -c <cores>` writes an `init.txt` from island 4, and this writes
/// one from island 5. The two are then diffed BYTE FOR BYTE, which is the only checkable seam this crate has.
/// ⛔ IT VALIDATES ENCODING, NOT SEMANTICS — dip lays out a wrong instruction as happily as a right one.
pub fn export_init(bytes: &[u8], out: &mut impl core::fmt::Write) -> core::fmt::Result {
    assert!(
        bytes.len().is_multiple_of(Flit::BYTES),
        "an init binary is whole flits: {} bytes is not a multiple of {}",
        bytes.len(),
        Flit::BYTES
    );
    // ⭐ `as_chunks` RATHER THAN `chunks_exact`, because the flit's width is a CONSTANT: the chunk is a
    // `[u8; Flit::BYTES]` and every index below is bounded by the type rather than by the assert above.
    let (flits, remainder) = bytes.as_chunks::<{ Flit::BYTES }>();
    debug_assert!(
        remainder.is_empty(),
        "the assert above already refused a partial flit"
    );
    for flit in flits {
        for slice in (0..Flit::SLICES).rev() {
            for word in (0..Slice::WORDS).rev() {
                let at = slice * Slice::BYTES + word * 4;
                let value =
                    u32::from_le_bytes([flit[at], flit[at + 1], flit[at + 2], flit[at + 3]]);
                write!(out, "{value:08x}")?;
            }
        }
        writeln!(out)?;
    }
    Ok(())
}

/// How many flits the whole file holds — every packet's header plus its body.
fn flit_total(packets: &Packets) -> usize {
    packets
        .iter()
        .map(|packet| 1 + packet.flits.len())
        .sum::<usize>()
}

/// WHERE EACH PACKET BEGINS, by walking the chain the way the runtime does.
///
/// ⛔ THIS IS A READER, AND IT EXISTS TO CATCH THE MISREADING THAT WAS INVISIBLE. An inner QGI header has
/// `word2 == 0`, so a classifier keyed on `unit_bits == 0` treats it as a Null header and re-serialises it
/// byte-for-byte — the round trip stays exact while every packet boundary after it is wrong. Walking `myflits`
/// and requiring the walk to land exactly on the end is what says the boundaries are right, and comparing bytes
/// is what cannot.
pub fn packet_starts(bytes: &[u8]) -> Vec<usize> {
    assert!(
        bytes.len().is_multiple_of(Flit::BYTES),
        "an init binary is whole flits: {} bytes is not a multiple of {}",
        bytes.len(),
        Flit::BYTES
    );
    let flits = bytes.len() / Flit::BYTES;
    let mut starts = Vec::new();
    let mut at = 0;
    while at < flits {
        // ⭐ THE ZERO PADDING IS NOT A PACKET. `dip.cpp:112-119` fills up to the allocation granularity with zero
        // flits, and a zero word 0 has neither the `0xdd` tag nor a count — so the walk ends at the first one
        // rather than reading it as a header of length 1 and marching off the end.
        let tagged = bytes[at * Flit::BYTES] == 0xDD;
        if !tagged {
            break;
        }
        starts.push(at);
        // ⭐ THE WRITER'S OWN INVERSE, not a second reading of the layout. A reader that masked a different width
        // would desynchronise every packet after the first large one and still round-trip byte-exactly.
        let word0 = u32::from_le_bytes([
            bytes[at * Flit::BYTES],
            bytes[at * Flit::BYTES + 1],
            bytes[at * Flit::BYTES + 2],
            bytes[at * Flit::BYTES + 3],
        ]);
        let myflits = QgiHeader::myflits_of(word0).get();
        at += 1 + usize::try_from(myflits).expect("a flit count that fits an index");
    }
    // ⛔ THE WALK MUST LAND ON THE END, ALLOWING ONLY THE PAD. Anything more than one allocation unit of slack
    // means a header's count disagrees with its body, which does not corrupt one packet — it desynchronises every
    // packet after it, while the bytes still round-trip exactly.
    let slack = flits - at;
    assert!(
        slack < PAD_TO_FLITS.max(1) || at == flits,
        "the packet chain stopped at flit {at} of {flits}, leaving {slack} unaccounted for — more than the \
         allocation padding, so a header's count disagrees with its body"
    );
    for byte in &bytes[at * Flit::BYTES..] {
        assert_eq!(
            *byte, 0,
            "the bytes after the last packet are not zero padding, so a packet boundary is being read where there \
             is none"
        );
    }
    starts
}
