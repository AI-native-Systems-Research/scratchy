//! Bridge 3 (`TiledOp -> SdscOp`), pointwise/broadcast family: the PRIMITIVE (one `TileOp` in, one
//! `OpSpec`/`EmittedOp` out, no internal decomposition) TileIR→SdscOp entry points for elementwise
//! and broadcast pointwise ops. See `super`'s module doc for why this is its own small file rather
//! than folded into a bigger one.

use crate::emit::{EmittedOp, EwOperand, op_func_from_str};
use crate::ir::island::tile_op::TileOp;
use crate::placement::BundleLayout;
use crate::sdsc_abstract::{KindTag, StickKind, Stk};
use crate::superdsc_opspec::{
    Allocation, AnyTensorArg, DataFormat, Df, Fp16, MAX_CORES, MaxCores, OpFunc, OpInfo, OpSpec,
    Role, Scale, SdscFoldSet, TensorArg,
};
use crate::work::distribute_cores;

/// ⭐⭐⭐⭐⭐ THE GATHER'S OWN OP — a KERNEL-LESS `identity` that copies a PAGED source through an int32
/// index into a CONTIGUOUS destination. This is the only op shape deeptools will schedule a gather on.
///
/// ## Why a separate op at all, rather than gathering the matmul's operand
/// MEASURED as a bake refusal on the card at both possible index `memOrg_` values (see the
/// KERNEL-guard in `emit_sdsc`, which now refuses it at build time): `hasDimensionReuse`
/// (`L3DlOpsScheduler.cpp:303-321`) is `primaryDsInfo_.size() > 1 && count(DsTypes::KERNEL)`, and a
/// reuse op runs the arithmetic-intensity explorer whose `calculateFlopPerByte` demands an LX allocate
/// node for every HBM-pinned labeledDs — which `allocAllMem` never gives an index, because an index is
/// read into the IBR rather than staged. No declaration escapes that; only a KERNEL-less op does.
///
/// ## The shape, read off `dxp/test/test_gather_1core/sdsc_1.json`
/// ```text
/// primaryDsInfo_  { OUTPUT, KERNEL_IDX }        <- NO KERNEL, NO INPUT
/// lds 0 Tensor0  OUTPUT      wl 2  SEN169_FP16  <- the gathered SOURCE, typed OUTPUT
/// lds 1 Tensor1  KERNEL_IDX  wl 4  SENUINT32    <- the index
/// lds 2 Tensor2  OUTPUT      wl 2  SEN169_FP16  <- the contiguous DESTINATION
/// computeOp_  exUnit sfp  opFuncName identity
///             inputLabeledDs ["Tensor0-idx0"]  outputLabeledDs ["Tensor2-idx2"]
///             indirectAccessIndexLabeledDs ["Tensor1-idx1"]
/// ```
/// Both data operands typed `OUTPUT` is what keeps `KERNEL` out of `primaryDsInfo_` — and
/// [`pointwise_opspec_from_tile`] already builds every operand that way, so this is that builder plus
/// the declaration, not a new operand law.
///
/// ## ⭐ WHY THE DESTINATION BEING CONTIGUOUS IS THE POINT
/// `matmul/opspec.rs` records that a 3-D (`mb`-declaring) kernel "makes dxp treat the weight as
/// per-batch → garbage". That warning is about a per-batch stride **dxp DERIVES** and the KV pool does
/// not have. A purpose-built contiguous scratch has EXACTLY the stride dxp derives — so once the copy
/// has landed each row's keys at a uniform pitch, the score matmul can read it per-batch with no
/// gather and no index at all. The gather resolves the pool's free-list irregularity; the matmul then
/// sees a regular buffer.
///
/// ## ⛔ `rows` IS SUB-ROWS OF **ONE STICK**, AND `cols` MUST BE ONE STICK
/// A `[mb, out]` operand sticked on a MULTI-STICK `out` is `RowBlocked` (stick-major), whose address law
/// `(c/64)*(rows*64) + r*64 + c%64` scatters each gathered block across the whole buffer — off by `rows`
/// from the contiguous `[hd, 64]` block the score kernel then reads, from a clean bake. At one stick of
/// `out` the same law collapses to `r*64 + c`, i.e. ROW-MAJOR, and a pool block is `hd` consecutive
/// sub-rows. `GatherScratch::sub_rows`/`entry_page` carry that split, and the gather's `page` is what
/// keeps ONE ENTRY equal to one whole block.
pub fn gather_copy_opspec(
    src: &str,
    dst: &str,
    rows: u32,
    cols: u32,
    gather: crate::superdsc_opspec::GatherIndex,
    // ⭐⭐⭐⭐⭐ THE DESTINATION ENTRY THIS RUN STARTS AT — separate from `gather.first_entry`, and it has to
    // be. See `GatherCopy::dest_entry`: the two were one number while a run's entries and its destination
    // rows advanced together (the window-granular cut, where one index stick IS 32 scratch rows), and a
    // page-granular op breaks that — it names ONE entry but must sit at a stick boundary, so entries
    // advance 32 per op while rows advance 1. Both are minted from the same request index by `copies()`,
    // so they cannot drift; they simply are not equal any more.
    dest_entry: crate::superdsc_opspec::DestEntry,
) -> Result<OpSpec, String> {
    let cols_ext = crate::superdsc_opspec::StickExtent::<Fp16>::new(cols)?;
    if cols != Fp16::ELEMS_PER_STICK {
        return Err(format!(
            "gather_copy_opspec('{src}' -> '{dst}'): `out` is {cols} elements, and a gathered copy's \
             `out` must be exactly one fp16 stick ({}). Wider than one stick, the operand classifies \
             stick-major and each gathered block is scattered `rows*64` apart instead of contiguous — \
             which bakes clean and hands the score kernel another request's slots. Declare the block as \
             `hd` one-stick sub-rows (`GatherScratch::sub_rows`) with the entry `page` covering them.",
            Fp16::ELEMS_PER_STICK,
        ));
    }
    // ⛔ THE PAGE MUST DIVIDE THE ROWS, or the last entry covers a partial block. Refused rather than
    // rounded: a partial entry is an address, and the caller's `rows` comes from `sub_rows` which is a
    // whole number of pages by construction — so this can only fire on a caller that built its own.
    let page = gather.page.get();
    // ⛔⛔⛔ THE INDEX MUST FIT **ONE STICK**, AND THIS IS THE RAW DOOR'S HALF OF THAT LAW.
    // `GatherScratch::copies` cannot mint a longer run (that is the point of the type), but this
    // function also takes hand-built extents — and above one stick of entries the card does not fault
    // and does not refuse: the IBR is loaded one stick at a time and each core's read offset inside it
    // is taken modulo that stick, so the cores past the wrap silently read another core's page
    // addresses. That is rung 8's corruption, and a build-time `Err` is the only honest answer to it.
    if let Some(entries) = crate::superdsc_opspec::PageExtent::of_positions(page).entries_in(rows) {
        let cap = crate::sdsc_abstract::CopyDims::ENTRIES_PER_OP;
        if entries > cap {
            return Err(format!(
                "gather_copy_opspec('{src}' -> '{dst}'): {entries} index entries ({rows} `mb` \
                 positions / a {page}-position page), and one gather op's index is ONE {cap}-entry \
                 stick — dxp loads the IBR in a single stick transfer and takes each core's read \
                 offset inside it modulo that stick, so entries past the first {cap} WRAP and those \
                 cores gather another core's pages with a clean bake. Cut the pass into one op per \
                 stick (`GatherScratch::copies`)."
            ));
        }
    }
    // ⭐ THE ELEMENTS ONE INDEX ENTRY COVERS IN THE DESTINATION — `page` one-stick sub-rows, i.e. one
    // whole pool block. Named once, here, because it is also what `skip_addr` is (`page × the per-core
    // extents of the unpinned dims`, and `out` is the only unpinned one) — so the distance an index step
    // moves in the SOURCE and the distance a run's base moves in the DESTINATION are one number.
    let entry_elems = page.saturating_mul(cols_ext.elems());
    if page == 0 || !rows.is_multiple_of(page) {
        return Err(format!(
            "gather_copy_opspec('{src}' -> '{dst}'): {rows} `mb` positions do not divide into whole \
             {page}-position entries, so the last index entry would cover a partial block."
        ));
    }
    let dim = |name: &'static str, size: u32, is_stick: bool| crate::superdsc_opspec::ItDim {
        name,
        size,
        is_reduction: false,
        is_stick,
        df: Df::Fp16,
    };
    let tile_op = TileOp {
        // Two operands: the gathered source and the contiguous destination. The index is APPENDED by
        // `attach_gather_index`, which is also what declares it — one call, so an operand cannot be
        // pushed without the declaration that makes it an index.
        kind: crate::ir::island::tile_op::TileOpKind::PointwiseOrReduce { n_operands: 2 },
        dims: vec![
            dim("mb", rows, false),
            dim("out", cols_ext.elems(), true),
            dim("y", 1, false),
        ],
        df: Df::Fp16,
    };
    let mut op = pointwise_opspec_from_tile_split(
        &tile_op,
        OpFunc::Identity,
        &[src],
        dst,
        // ⛔⛔⛔ RANK-3, AND THIS IS A CARD-MEASURED REQUIREMENT, NOT A PRESENTATION CHOICE. This flag was
        // `false`, which puts a `rows > 1` op into [`pointwise_opspec_from_tile`]'s STICK-MAJOR rank-2
        // form — and that form OMITS `y` from `layoutDimOrder_` entirely rather than sizing it 1. The
        // sibling builder records the consequence from a real pod `dxp_standalone` crash
        // (`vector::_M_range_check`): "dxp's own dim-handling consistently assumes a 3-axis (mb/out/y)
        // structure elsewhere, and a rank-2 op whose layout genuinely lacks `y` is exactly what breaks
        // that assumption", which is why IT restricts stick-major to `cols > 64`. This op's `cols` is
        // exactly one stick, so it is precisely the regime that crash was found in — MEASURED here as a
        // launch-time `PrepZeroFlitCnt` on the job binary at `job_bin_ptr + 0x400` (939524104 flits, i.e.
        // a garbage program length), granite-3.1-2b fp8 at bs=8.
        //
        // ⭐ AND IT COSTS NOTHING IN ADDRESSING. `StickLayout::for_view_df` FOLDS a trailing unit dim, so
        // rank-3 `[mb, out, y=1]` classifies exactly as rank-2 `[mb, out]` does — and at one stick of
        // `out` that law collapses to `r*64 + c`, i.e. row-major either way. Only the declared axis COUNT
        // differs, which is the half dxp needs.
        true,
        // ⛔⛔⛔ `mb` ONLY — `out` MUST STAY WHOLE, AND THIS IS THE ONE NUMBER THE GATHER'S CORRECTNESS
        // HANGS ON. dxp DERIVES `skip_addr` as `page × the PER-CORE extents of the unpinned dims`
        // (`getBufferCapacityForNodePerDim`, reproduced locally by
        // `zz_skip_addr_computed_from_our_own_emission.rs`), so a split `out` divides the entry by the core
        // count: at bs=8/nb=1 `mb` takes all 32 cores and `out` stays whole by luck, while at bs=2 `mb`
        // takes 16 and `out` splits 2 ways — `skip_addr` 2048 against a host staging block numbers in units
        // of 4096. That is a clean bake whose every index step lands half a block short, i.e. fluent text
        // from inside the previous block. A distribution that cannot split `out` makes the entry unit a
        // property of the DECLARATION rather than of the batch width.
        //
        // ⛔ AND IT DIVIDES BY WHOLE ENTRIES, not by `mb` positions: `skip_addr` clamps the pinned axis to
        // `min(per_core_extent, page)`, so a core owning fewer than `page` positions SHRINKS the entry. At
        // bs=2 an unrestricted split hands each of 32 cores 32 sub-rows against a 64-position page and
        // halves `skip_addr` — the same clean-bake wrong-address class as a split `out`.
        //
        // It costs no parallelism worth having: the entries are `nkvh * nb * mq` — at least `nkvh` (8), and
        // 64 at bs=8 — so the copy still spreads over 8-32 cores.
        |dims: &[crate::superdsc_opspec::ItDim], max_cores: u32| {
            gather_copy_cores(dims, max_cores, page)
        },
        // ⭐⭐⭐ THE DESTINATION'S BASE IS **DERIVED**, NOT PASSED. This op serves the run of scratch rows
        // starting at the index's own `first_entry`, so there is exactly one legal destination base and
        // it is that entry count times the elements ONE ENTRY covers.
        //
        // ⛔ AS A PARAMETER IT WAS A SECOND KNOB FOR ONE FACT, and the two ways of getting it wrong are
        // both silent: a base ahead of the index's run overwrites the previous run's blocks and leaves
        // this run's rows holding whatever the allocator handed out, and a base of zero on every run
        // makes every op write run 0's rows — each one a valid, contiguous, wrongly-addressed gather
        // with a clean bake. Derived here, the run the index reads and the rows the copy writes cannot
        // be different runs.
        //
        // ⛔⛔⛔ AND IT IS `page * out` PER ENTRY, NOT `out`. This read `first_entry * cols`, and `cols`
        // here is the op's `out` EXTENT — ONE STICK — not the scratch row's width. A scratch row is
        // `page` one-stick SUB-ROWS (`GatherScratch::sub_rows`, which is the whole reason `mb` is counted
        // in sub-rows), so `cols` alone is short by exactly `page` = `hd` = 64×. MEASURED on the card
        // before this line was right: every run's base landed inside run 0's rows, so rows past the
        // first run were never written and were read as keys — one run `solo=EXACT`, two runs the first
        // token right then divergence, four runs wrong from the first token. The emission-side assertion
        // is `every_run_of_the_cut_pass_addresses_its_own_entries_and_rows`, which measures this exact
        // delta off the per-core start addresses rather than trusting the arithmetic.
        // ⛔ `dest_entry`, NOT `gather.first_entry`. This read `gather.first_entry.entries()`, which is
        // the INDEX's base in entries — equal to the destination's only while one index stick is 32
        // consecutive destination rows. A page-granular op's index base is `32 * request` while its rows
        // start at `request`, so reusing it lands every op's rows 32× too far out, past the scratch and
        // into the next intermediate's bytes. Read as keys, clean bake, no fault.
        entry_elems
            .checked_mul(dest_entry.entries())
            .ok_or_else(|| {
                format!(
                    "gather_copy_opspec('{src}' -> '{dst}'): the destination base for the run at entry \
                     {} overflows a 32-bit element offset at {entry_elems} elements per entry",
                    dest_entry.entries()
                )
            })?,
    )?;
    // ⛔ OPERAND 0 IS THE GATHERED ONE, AND IT IS NOT THE OUTPUT. `attach_gather_index` refuses the
    // output slot itself (gathering INTO a destination is a scatter), so this cannot silently become
    // one; the `?` below turns a shape it cannot express into this function's own `Err`.
    op.attach_gather_index(gather, 0).ok_or_else(|| {
        format!(
            "gather_copy_opspec('{src}' -> '{dst}'): the index declaration does not fit this op — \
             its dims are [mb, out, y], so a paged axis must be `mb` or `out`, and the two paged \
             axes must differ"
        )
    })?;
    Ok(op)
}

/// ⭐ THE GATHER COPY'S WORK DIVISION — split `mb` and NOTHING ELSE.
///
/// See [`gather_copy_opspec`]'s call site for why `out` staying whole is a correctness requirement and
/// not a tuning choice: it is the factor dxp derives `skip_addr` from, so splitting it changes what one
/// index entry MEANS as a function of the batch width.
///
/// Spelled as its own distributor rather than as a flag on [`distribute_cores`] because it is a LAW of
/// this one op shape — a distributor that may not touch a dim is not a policy the general splitter can
/// carry without every other caller having to say it does not want it.
fn gather_copy_cores(
    dims: &[crate::superdsc_opspec::ItDim],
    max_cores: u32,
    // Positions of `mb` ONE INDEX ENTRY covers. The division is over ENTRIES, so no core owns a fraction
    // of one — see the note at the call site for what a fractional entry does to `skip_addr`.
    entry_page: u32,
) -> std::collections::BTreeMap<&'static str, u32> {
    let mut splits: std::collections::BTreeMap<&'static str, u32> = Default::default();
    if let Some(mb) = dims
        .iter()
        .find(|d| d.name == "mb" && !d.is_reduction && !d.is_stick)
        .filter(|_| entry_page > 0)
    {
        // The BASIS is the entry count, and the split of `mb` is that same number — so every core gets a
        // whole number of entries and `min(per_core, page)` is always the page.
        let split = crate::work::core_split(mb.size / entry_page, max_cores);
        if split > 1 {
            splits.insert(mb.name, split);
        }
    }
    splits
}

/// [`gather_copy_opspec`] + `emit_sdsc_tiled` — the ONE assembler for the gather's copy, so the op the
/// fold launches and the descriptor the tests read are the same construction.
pub fn assemble_gather_copy(
    op_name: &str,
    src: &str,
    dst: &str,
    // ⛔ ONE RUN OF THE PASS, NOT THE PASS. It was the whole [`crate::sdsc_abstract::GatherScratch`], so
    // the op it built declared an index as long as the pass had rows — which above one stick of entries
    // is the rung-8 corruption (`GatherScratch::copies`). A [`crate::sdsc_abstract::GatherCopy`] carries
    // the run's extents AND its two bases together, so an op cannot be built for a run longer than dxp
    // loads or at a base that disagrees with its index.
    copy: crate::sdsc_abstract::GatherCopy,
    index: &str,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> EmittedOp {
    // ⛔ ALL THREE EXTENTS FROM THE SCRATCH, AS ONE VALUE, NOT THREE REACHES. `mb` is the copy's own
    // extent, the index's live entry count and the scratch's row count; `out` is the pool's stick; the
    // `page` the walk pins is the `page` the index declares. They were three separate calls here
    // (`sub_rows()`, `POOL_STICK`, `of_positions(entry_page())`) — so a reservation could be sized
    // against a different count than the descriptor declares (the "512 reserved against 2048 declared"
    // defect, whose overrun is another page's keys read as block numbers) and the two pins could drift
    // apart with nothing refusing it. See `sdsc_abstract::CopyDims`.
    let d = copy.dims();
    let op = gather_copy_opspec(
        src,
        dst,
        d.mb(),
        d.out(),
        crate::superdsc_opspec::GatherIndex::of_scratch_rows(
            index.to_string(),
            d.page(),
            // THE INDEX's base, in whole index STICKS — dxp loads the IBR one stick at a time.
            copy.index_base(),
        ),
        // ⛔ AND THE DESTINATION's base SEPARATELY. It used to be derived from the index base above;
        // see `GatherCopy::dest_entry` for the measurement that separated them (a page-granular op names
        // one entry at stick `r` but writes run `r`, so the two advance at different rates). Both come
        // from `copies()`, which mints them from one request index.
        crate::superdsc_opspec::DestEntry::of_entries(copy.dest_entry()),
    )
    .unwrap_or_else(|e| panic!("assemble_gather_copy {op_name}: {e}"));
    let folds = SdscFoldSet::new(op.iter.cores_used());
    crate::emit::emit_sdsc_tiled(op_name, &op, &folds, sym_id_base, layout)
        .unwrap_or_else(|e| panic!("assemble_gather_copy {op_name}: {e}"))
}

pub fn pointwise_opspec_from_tile(
    tile_op: &TileOp,
    op: OpFunc,
    in_names: &[&str],
    o_name: &str,
    head_major: bool,
) -> Result<OpSpec, String> {
    pointwise_opspec_from_tile_split(
        tile_op,
        op,
        in_names,
        o_name,
        head_major,
        distribute_cores,
        // Whole-tensor output — every caller but the gather copy, whose runs land at a base.
        0,
    )
}

/// [`pointwise_opspec_from_tile`] with the WORK DIVISION as a parameter — for the one op whose entry
/// unit is a function of which dims got split (see [`gather_copy_cores`]). Every other caller takes
/// [`distribute_cores`] through the wrapper above and is byte-identical.
fn pointwise_opspec_from_tile_split(
    tile_op: &TileOp,
    op: OpFunc,
    in_names: &[&str],
    o_name: &str,
    head_major: bool,
    cores: impl Fn(
        &[crate::superdsc_opspec::ItDim],
        u32,
    ) -> std::collections::BTreeMap<&'static str, u32>,
    // The OUTPUT's base, in elements — see [`gather_copy_opspec`]'s `dst_off`. 0 = whole-tensor output.
    out_offset: u32,
) -> Result<OpSpec, String> {
    let rows = tile_op
        .dims
        .iter()
        .find(|d| d.name == "mb")
        .map(|d| d.size)
        .unwrap_or(1);
    let tiled = tile_op
        .tile(MaxCores::<MAX_CORES>, cores, "out")
        .map_err(|e| e.0)?;
    let (plan, time_tile) = (tiled.plan, tiled.time_tile);
    let device_dims = plan.iter_syms(["mb", "out", "y"]);
    // ── STICK-MAJOR RESIDUAL ── a whole-tensor residual pointwise op writes FLAT (rank-3
    // [mb,out,y=1]) but the consuming matmul reads STICK-MAJOR (rank-2 [mb,out]); byte-identical at
    // rows==1, scrambled at rows>1 AND cols>64. Present rank-2 [mb,out] so producer + consumer
    // agree. head_major (per-head attn ops) stays rank-3.
    let stickmajor = !head_major && rows > 1;
    let device_dims_r2 = plan.iter_syms(["mb", "out"]);

    // rank-2 stick-major (`stickmajor`) or the legacy rank-3 flat operand (drops the size-1 `y`).
    // `off` is the operand's own base in elements: 0 for every input (see `gather_copy_opspec`'s
    // `dst_off` for why the gathered source must not take one) and `out_offset` for the output.
    let mk = |is_input: bool, name: &str, off: u32| -> Result<AnyTensorArg, String> {
        if stickmajor {
            Ok(AnyTensorArg::R2(
                TensorArg::<2>::new(
                    is_input,
                    name.to_string(),
                    Role::Output,
                    [Scale::Active, Scale::Active],
                    device_dims_r2,
                    ["mb", "out"],
                    "out",
                    Allocation::Hbm,
                )
                .map_err(|e| e.0)?
                .with_offset(off),
            ))
        } else {
            Ok(AnyTensorArg::R3(
                TensorArg::<3>::new(
                    is_input,
                    name.to_string(),
                    Role::Output,
                    [Scale::Active, Scale::Active, Scale::Active],
                    device_dims,
                    ["mb", "out", "y"],
                    "out",
                    Allocation::Hbm,
                )
                .map_err(|e| e.0)?
                .with_offset(off),
            ))
        }
    };
    let mut args = Vec::with_capacity(in_names.len() + 1);
    for name in in_names {
        args.push(mk(true, name, 0)?);
    }
    args.push(mk(false, o_name, out_offset)?);

    let tiled_symbols = time_tile.map(|t| vec![t.dim()]).unwrap_or_default();
    Ok(OpSpec {
        op,
        is_reduction: false,
        iter: plan,
        args,
        op_info: if op.needs_sfp_const_table() {
            OpInfo::SfpConstTable
        } else {
            OpInfo::None
        },
        tiled_symbols,
        time_tile,
        indirect: None,
    })
}

/// TileIR → SdscOp for the [`EwOperand`]-broadcast pointwise family: takes an ALREADY-LOWERED
/// `TileOp` (from `pointwise_broadcast_opspec_df` in the live emitter, or directly from a live
/// per-node lowering via `node_to_single_tile_op`) and runs the tiler + TensorArg assembly.
/// `rows`/`cols` stay EXPLICIT parameters (not re-derived from `tile_op.dims`) because this
/// builder's `stickmajor`/block-alignment logic needs the untouched `rows·cols` product, not the
/// stick-rounded `out` extent `TileOp` carries.
#[allow(clippy::too_many_arguments)]
pub fn pointwise_broadcast_opspec_from_tile(
    tile_op: &TileOp,
    rows: u32,
    cols: u32,
    op: OpFunc,
    inputs: &[EwOperand<'_>],
    o_name: &str,
    out_offset: u32,
    head_major: bool,
) -> Result<OpSpec, String> {
    let df = tile_op.df;
    let tiled = tile_op
        .tile(MaxCores::<MAX_CORES>, distribute_cores, "out")
        .map_err(|e| e.0)?;
    let (plan, time_tile) = (tiled.plan, tiled.time_tile);
    let device_dims = plan.iter_syms(["mb", "out", "y"]);
    // ── STICK-MAJOR RESIDUAL + PER-HEAD ATTN ── The m>1 seam fix: a residual whole-tensor
    // pointwise op writes FLAT (rank-3 [mb,out,y=1]) but the consuming matmul reads STICK-MAJOR
    // (rank-2 [mb,out]); byte-identical at rows==1, scrambled at rows>1 AND cols>64. Present the op
    // as rank-2 [mb,out] (stick-major) so producer + consumer agree. head_major stays rank-3.
    //
    // BLOCK-ALIGNED per-head offsets also qualify: a per-head attn softmax op processes head h's
    // whole [rows,cols] SUB-BLOCK at base offset h·(rows·cols) — a base shift of a STANDALONE
    // stick-major [rows,cols] tensor (`with_offset` bumps the flat base, `per_core_addr` then
    // addresses the block stick-major), byte-identical to the score matmul that WROTE the block
    // per-head. So an offset that is an exact multiple of rows·cols is stick-major-compatible. A
    // per-row-scalar (out-broadcast) operand collapses its column addressing to c=0, so its own
    // per-head row-block offset (e.g. rc's h·rows·stick) is compatible regardless of alignment to
    // rows·cols. A NON-block-aligned COLUMN slice (RoPE rotate-half `half`; an MLP column chunk
    // `cols_start`) is NOT a whole-block shift — a column slice of a stick-major tensor is NOT
    // contiguous — so it stays rank-3 flat.
    let block = (rows as u64) * (cols as u64);
    let block_aligned = |off: u32| block != 0 && (off as u64).is_multiple_of(block);
    // `cols > 64` (one fp16 stick) is the OTHER half of this fn's own documented precondition —
    // the comment above says outright "byte-identical at rows==1, scrambled at rows>1 AND
    // cols>64" — but the condition below never checked `cols`, so it dropped to rank-2 for EVERY
    // rows>1 op regardless of width, including `cols<=64` ops the flat/rank-3 form was already
    // correct for. That rank-2 form OMITS the `y` dim from `layoutDimOrder_` entirely (not just
    // sizes it 1) — confirmed against a real pod dxp_standalone crash (`vector::_M_range_check`,
    // fp8-dynamic granite prefill's `attn_onorm`/`attn_sp`, both `cols=head_dim=64`): dxp's own
    // dim-handling (`designSpaceConfig.cpp`/`dsc2.cpp` in the dxp source, `dxp-rs/deeptools`)
    // consistently assumes a 3-axis (mb/out/y) structure elsewhere, and a rank-2 op whose layout
    // genuinely lacks `y` is exactly what breaks that assumption. Restricting stickmajor to
    // `cols > 64` — the ONLY regime the introducing comment claims needs it — leaves every
    // `cols<=64` op (every per-head attention op here) at the already-proven rank-3 form, while
    // preserving the real fix for wide (`cols>64`) ops it was actually written for.
    let stickmajor = !head_major
        && rows > 1
        && cols > Fp16::ELEMS_PER_STICK
        && block_aligned(out_offset)
        && inputs
            .iter()
            .all(|i| i.out_broadcast() || block_aligned(i.col_offset()));
    let device_dims_r2 = plan.iter_syms(["mb", "out"]);

    if time_tile.is_some() && inputs.iter().any(|i| i.out_broadcast()) {
        return Err(format!(
            "pointwise_broadcast '{o_name}': a broadcast-over-`out` operand in a TIME-TILED op \
             is not yet supported — concrete_trips would wrongly advance the broadcast operand's \
             address per trip (silently-wrong). Refusing (build guard). [rows={rows} cols={cols}]"
        ));
    }

    // Build a rank-2 stick-major (`stickmajor`) or the legacy rank-3 flat operand. rank-2 drops the
    // size-1 `y` dim + its scale so `for_view_df` classifies it RowBlocked (stick-major), matching
    // the matmul I/O.
    let mk =
        |is_input: bool, name: &str, scale: [Scale; 3], off: u32| -> Result<AnyTensorArg, String> {
            if stickmajor {
                Ok(AnyTensorArg::R2(
                    TensorArg::<2>::new(
                        is_input,
                        name.to_string(),
                        Role::Output,
                        [scale[0], scale[1]],
                        device_dims_r2,
                        ["mb", "out"],
                        "out",
                        Allocation::Hbm,
                    )
                    .map_err(|e| e.0)?
                    .with_offset(off)
                    .with_df(df)
                    .with_row_blocked(matches!(df, Df::Fp32)),
                ))
            } else {
                Ok(AnyTensorArg::R3(
                    TensorArg::<3>::new(
                        is_input,
                        name.to_string(),
                        Role::Output,
                        scale,
                        device_dims,
                        ["mb", "out", "y"],
                        "out",
                        Allocation::Hbm,
                    )
                    .map_err(|e| e.0)?
                    .with_offset(off)
                    .with_df(df)
                    .with_row_blocked(matches!(df, Df::Fp32)),
                ))
            }
        };
    let mut args = Vec::with_capacity(inputs.len() + 1);
    for inp in inputs {
        args.push(mk(true, inp.name(), inp.scale(), inp.col_offset())?);
    }
    // The OUTPUT may write to a COLUMN SLICE of its tensor (`out_offset` cols in →
    // `out_offset·2` byte offset): RoPE's rotate-half writes each head-half to a
    // distinct half of the roped output. `per_core_addr` adds this on top of the
    // global placement (task #55). 0 = whole-tensor output (every existing caller;
    // and the `stickmajor` predicate requires out_offset==0, so rank-2 is whole-tensor).
    args.push(mk(
        false,
        o_name,
        [Scale::Active, Scale::Active, Scale::Active],
        out_offset,
    )?);
    let tiled_symbols = time_tile.map(|t| vec![t.dim()]).unwrap_or_default();
    Ok(OpSpec {
        op,
        is_reduction: false,
        iter: plan,
        args,
        op_info: if op.needs_sfp_const_table() {
            OpInfo::SfpConstTable
        } else {
            OpInfo::None
        },
        tiled_symbols,
        time_tile,
        indirect: None,
    })
}

/// `assemble_pointwise_broadcast_off` TAKING AN ALREADY-LOWERED `TileOp` — the TileIR→SdscOp entry
/// point for a live per-node caller that built its `TileOp` via `node_to_single_tile_op` (straight
/// off the `SubtileNode`) instead of re-deriving `rows`/`cols` first. `rows`/`cols` stay explicit
/// params (needed for the SFP-transcendental guard and the stickmajor block-alignment calc — see
/// [`pointwise_broadcast_opspec_from_tile`]'s doc for why they are not re-derived from
/// `tile_op.dims`).
#[allow(clippy::too_many_arguments)]
pub fn assemble_pointwise_broadcast_off_from_tile<O: KindTag>(
    op_name: &str,
    tile_op: &TileOp,
    op_func: &'static str,
    rows: u32,
    cols: u32,
    inputs: &[EwOperand<'_>],
    o: &Stk<O>,
    out_offset: u32,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> EmittedOp {
    let o_name = o.name();
    let head_major = O::kind() == StickKind::Flat;
    // ⛔ ONE SHARED LIST, NOT A SECOND COPY. This was a verbatim duplicate of the `const` in
    // `emit::assemble_pointwise_broadcast_off`, so a fix had to land in both — and `"mish"` had been
    // added to neither. See [`crate::emit::SFP_SPLIT_TRANSCENDENTALS`] for the geometry, the dxp
    // `map::at` observation, and why the list cannot be keyed on `OpFunc` yet.
    if crate::emit::SFP_SPLIT_TRANSCENDENTALS.contains(&op_func)
        && cols > Fp16::ELEMS_PER_STICK
        && inputs.iter().any(|i| i.out_broadcast())
    {
        panic!(
            "[superdsc sfp-transcendental-guard] {op_name}: SFP `{op_func}` emitted at \
             multi-stick width (cols={cols} > {}) with an OUT-BROADCAST (reduction) input — \
             the full-width-transcendental geometry that CRASHES the dxp compile (map::at in \
             ddc's per-core SFP split). A transcendental on a reduced scalar MUST be 1-stick \
             (cols={}); broadcast in the following multiply. (guard-every-crash-at-build-time)",
            Fp16::ELEMS_PER_STICK,
            Fp16::ELEMS_PER_STICK,
        );
    }
    let op = pointwise_broadcast_opspec_from_tile(
        tile_op,
        rows,
        cols,
        op_func_from_str(op_func),
        inputs,
        o_name,
        out_offset,
        head_major,
    )
    .unwrap_or_else(|e| panic!("assemble_pointwise_broadcast {op_name}: {e}"));
    let folds = SdscFoldSet::new(op.iter.cores_used());
    crate::emit::emit_sdsc_tiled(op_name, &op, &folds, sym_id_base, layout)
        .unwrap_or_else(|e| panic!("assemble_pointwise_broadcast {op_name}: {e}"))
}

/// `assemble_pointwise_seeded` TAKING AN ALREADY-LOWERED `TileOp` instead of raw `rows`/`cols` — the
/// real TileIR→SdscOp entry point for the live per-node path: a caller that already built its
/// `TileOp` via `subtile_tape_to_tile_ir::node_to_single_tile_op` (straight off the
/// `SubtileNode`, no re-derivation of shape) hands it here instead of unpacking it back into
/// `rows`/`cols` first.
pub fn assemble_pointwise_seeded_from_tile(
    op_name: &str,
    tile_op: &TileOp,
    op_func: &'static str,
    in_names: &[&str],
    o_name: &str,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> EmittedOp {
    let op =
        pointwise_opspec_from_tile(tile_op, op_func_from_str(op_func), in_names, o_name, false)
            .unwrap_or_else(|e| panic!("assemble_pointwise {op_name}: {e}"));
    let folds = SdscFoldSet::new(op.iter.cores_used());
    crate::emit::emit_sdsc_tiled(op_name, &op, &folds, sym_id_base, layout)
        .unwrap_or_else(|e| panic!("assemble_pointwise {op_name}: {e}"))
}
