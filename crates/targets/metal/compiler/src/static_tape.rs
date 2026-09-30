//! The macro-side metal tape bake.
//!
//! Runs `scratchy_target_metal::tape::lowering::lower_subtile_tape_to_metal`
//! at expansion, once per (generation class × chunked-addressing) variant, and emits
//! the results as `ClassedTape` statics via the generic serializer in
//! [`crate::const_tokens`]. Nothing here re-implements lowering logic —
//! zero per-op surface.
//!
//! The two runtime-only inputs are handled by construction:
//! - **device generation** → one variant per [`GenClass`], deduped when
//!   the lowering is generation-invariant (the common case);
//! - **block-table capacity** → the lowering runs at four probe
//!   capacities; every constant/scratch value that moves must fit
//!   `max(floor, base + slope·cap)` exactly (verified at the fourth
//!   probe) and is emitted as a patch the pool substitutes at load.
//!   A value that doesn't fit the model REFUSES the bake loudly.

use proc_macro2::TokenStream;
use quote::quote;
use scratchy_target_metal::tape::lowered::{
    CapPatch, GatedCommand, GenClass, LoweredMetalTape, PatchTarget, ScratchField, ScratchPatch,
};

/// ⭐ EVERY DISTINCT COMMAND OF ONE MODEL'S BAKED TAPES, SPELLED ONCE.
///
/// A bucket bakes up to six `(gen class × chunked)` variants, and the variants of one bucket —
/// and the buckets of one model — share almost every command: the chunked variant differs in a
/// single `ATTN_BLOCKS_PER_CHUNK` constant on the attention readers, the M5 variant in its GEMMs.
/// Deduping whole bodies only helps when NOTHING differs, so one constant forked a full copy of
/// the tape and rustc's single-threaded front end paid for every token of it.
///
/// Each distinct command is one `const` in the model's `__tape_cmds` module; a body's command
/// list is references into it. The runtime types are untouched — the statics still hold
/// `&'static [GatedCommand]`.
#[derive(Default)]
pub struct CommandPool {
    index: std::collections::HashMap<String, usize>,
    consts: Vec<TokenStream>,
}

impl CommandPool {
    fn intern(&mut self, cmd: &GatedCommand) -> Result<TokenStream, BakeDefect> {
        let toks = crate::const_tokens::const_tokens(cmd)
            .map_err(|e| BakeDefect(format!("serialize command: {e}")))?;
        let next = self.consts.len();
        let ix = *self.index.entry(toks.to_string()).or_insert_with(|| {
            let id = quote::format_ident!("C{next}");
            self.consts
                .push(quote! { pub(super) const #id: __tl::GatedCommand = #toks; });
            next
        });
        let id = quote::format_ident!("C{ix}");
        Ok(quote! { __tape_cmds::#id })
    }

    /// The `__tape_cmds` module the model's tape statics reference. Emit it once, beside them.
    pub fn into_tokens(self) -> TokenStream {
        let aliases = crate::const_tokens::alias_preamble();
        let consts = self.consts;
        quote! {
            #[cfg(feature = "metal")]
            mod __tape_cmds {
                #aliases
                #(#consts)*
            }
        }
    }
}
use scratchy_target_metal::tape::lowering as tl;
use scratchy_target_metal::tape::model_consts::MetalModelConsts;
use scratchy_target_metal::tape::step::{MetalStepTape, RotaryTables};
use scratchy_target_metal::tape::targets::MetalTargetProfile;

/// Why a bucket's tape could not bake — a defect; the caller panics with it.
pub struct BakeDefect(pub String);

impl std::fmt::Display for BakeDefect {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Probe capacities. `CAP_ZERO` bakes the floors; A/B derive the slope;
/// `CAP_CHECK` (off the A–B line) verifies the max-affine model.
const CAP_ZERO: u32 = 0;
const CAP_A: u32 = 1024;
const CAP_B: u32 = 2048;
const CAP_CHECK: u32 = 1536;

/// One bucket's lowering input: its step tape, minus the two runtime values.
pub struct BucketLowerInput<'a> {
    pub steps: &'a MetalStepTape,
    pub bucket_m: u32,
    pub num_arena_slots: u32,
    /// The model's class rotary tables (rope-on-read), from its source manifest.
    pub rotary: Option<RotaryTables>,
}

/// A bucket's MTL4 barrier flags as a `static`.
pub fn emit_bucket_barriers_static(static_ident: &syn::Ident, barriers: &[bool]) -> TokenStream {
    quote! {
        #[cfg(feature = "metal")]
        static #static_ident: &[bool] = &[ #(#barriers),* ];
    }
}

fn profile_for(class: GenClass) -> MetalTargetProfile {
    use scratchy_target_metal::tape::targets::{M1_MAX, M4_10CORE, M5_10CORE};
    match class {
        GenClass::M1 => M1_MAX,
        GenClass::Mid => M4_10CORE,
        GenClass::M5 => M5_10CORE,
    }
}

fn run_lower(
    mc: &MetalModelConsts,
    input: &BucketLowerInput<'_>,
    cap: u32,
    profile: &MetalTargetProfile,
    chunked: bool,
) -> Result<tl::Lowered, BakeDefect> {
    let at = tl::BakePoint {
        chunked,
        bucket_m: input.bucket_m,
        num_arena_slots: input.num_arena_slots,
        rotary: input.rotary,
        block_cap: cap,
        profile: Some(profile),
    };
    tl::lower_subtile_tape_to_metal(input.steps, mc, at)
        .map_err(|e| BakeDefect(format!("bucket_m={}: {e}", input.bucket_m)))
}

/// The fitted rational capacity model:
/// `value(cap) = max(floor, base + (num·cap)/den)` with floor/ceil div.
pub struct CapFit {
    pub floor: u32,
    pub base: i64,
    pub num: i64,
    pub den: u32,
    pub round_up: bool,
}

/// Fit the rational model through four observations. `Ok(None)` =
/// capacity-invariant. `Err` = the value moves but not in a shape the
/// model covers — the bake must refuse.
fn fit(f0: u32, fa: u32, fb: u32, fc: u32, what: &str) -> Result<Option<CapFit>, String> {
    if f0 == fa && fa == fb && fb == fc {
        return Ok(None);
    }
    let (fa_i, fb_i, fc_i, f0i) = (fa as i64, fb as i64, fc as i64, f0 as i64);
    let d = (CAP_B - CAP_A) as i64;
    let diff = fb_i - fa_i;
    // Reduce (diff / d) to num/den.
    let g = {
        let mut a = diff.abs().max(1);
        let mut b = d;
        while b != 0 {
            let t = a % b;
            a = b;
            b = t;
        }
        a.max(1)
    };
    let num = diff / g;
    let den = (d / g) as u32;
    let deni = den.max(1) as i64;
    for round_up in [false, true] {
        let q = |cap: u32| {
            let prod = num * cap as i64;
            if round_up {
                prod.div_euclid(deni) + if prod.rem_euclid(deni) != 0 { 1 } else { 0 }
            } else {
                prod.div_euclid(deni)
            }
        };
        let base = fa_i - q(CAP_A);
        let floor = f0i;
        let model = |cap: u32| floor.max(base + q(cap));
        if [
            (CAP_ZERO, f0i),
            (CAP_A, fa_i),
            (CAP_B, fb_i),
            (CAP_CHECK, fc_i),
        ]
        .iter()
        .all(|&(cap, v)| model(cap) == v)
        {
            return Ok(Some(CapFit {
                floor: f0,
                base,
                num,
                den,
                round_up,
            }));
        }
    }
    Err(format!(
        "{what}: capacity dependence does not fit max(floor, base + num*cap/den) \
         (f(0)={f0}, f({CAP_A})={fa}, f({CAP_B})={fb}, f({CAP_CHECK})={fc})"
    ))
}

/// Diff the four probe tapes into (floor tape, patches). Every field
/// other than constant bits and the scratch sizes must be identical
/// across probes — anything else moving is an unmodeled capacity
/// dependence and refuses the bake.
fn diff_probes(
    t0: LoweredMetalTape,
    ta: &LoweredMetalTape,
    tb: &LoweredMetalTape,
    tc: &LoweredMetalTape,
) -> Result<(LoweredMetalTape, Vec<CapPatch>, Vec<ScratchPatch>), String> {
    let n = t0.commands.len();
    if ta.commands.len() != n || tb.commands.len() != n || tc.commands.len() != n {
        return Err(format!(
            "command count varies with block capacity ({n} vs {} vs {} vs {})",
            ta.commands.len(),
            tb.commands.len(),
            tc.commands.len()
        ));
    }
    let mut const_patches = Vec::new();
    for i in 0..n {
        let (c0, ca, cb, cc) = (
            &t0.commands[i],
            &ta.commands[i],
            &tb.commands[i],
            &tc.commands[i],
        );
        // Everything except `constants` must match bit-for-bit.
        // `GatedCommand` is `Copy + PartialEq` all the way down, so the
        // comparison is the struct itself; only the REFUSAL path pays for
        // a rendered form (see `render` below), and that path fails the
        // build anyway.
        let strip = |c: &scratchy_target_metal::tape::lowered::GatedCommand| {
            let mut c = *c;
            c.command.constants = &[];
            c.command.bindings = &[];
            c.command.dispatch.threadgroups = (0, 0, 0);
            c.command.dispatch.threads_per_threadgroup = (0, 0, 0);
            if let Some(ms) = c.command.dispatch.m_scaling.as_mut() {
                ms.bucket_m = scratchy_target_metal::tape::ids::BucketM(0);
            }
            c
        };
        let render = |c: &scratchy_target_metal::tape::lowered::GatedCommand| {
            crate::const_tokens::const_tokens(c)
                .expect("serialize probe command")
                .to_string()
        };
        let g0 = strip(c0);
        for (other, cap) in [(ca, CAP_A), (cb, CAP_B), (cc, CAP_CHECK)] {
            let go = strip(other);
            if go != g0 {
                // Name the first token-level divergence so the refusal
                // says WHICH field moved, not just that one did.
                let s0 = render(&g0);
                let so = render(&go);
                let a: Vec<&str> = s0.split(' ').collect();
                let b: Vec<&str> = so.split(' ').collect();
                let k = a.iter().zip(&b).take_while(|(x, y)| x == y).count();
                let ctx_a: String = a[k.saturating_sub(6)..(k + 6).min(a.len())].join(" ");
                let ctx_b: String = b[k.saturating_sub(6)..(k + 6).min(b.len())].join(" ");
                return Err(format!(
                    "command {i} ({}) differs beyond constants at capacity {cap} — \
                     unmodeled block-capacity dependence; first divergence:\n  cap0: …{ctx_a}…\n  cap{cap}: …{ctx_b}…",
                    c0.command.function
                ));
            }
        }
        let nc = c0.command.constants.len();
        if ca.command.constants.len() != nc
            || cb.command.constants.len() != nc
            || cc.command.constants.len() != nc
        {
            return Err(format!(
                "command {i}: constants length varies with capacity"
            ));
        }
        for j in 0..nc {
            let (k0, ka, kb, kc) = (
                c0.command.constants[j],
                ca.command.constants[j],
                cb.command.constants[j],
                cc.command.constants[j],
            );
            if (k0.index, k0.ty) != (ka.index, ka.ty)
                || (k0.index, k0.ty) != (kb.index, kb.ty)
                || (k0.index, k0.ty) != (kc.index, kc.ty)
            {
                return Err(format!(
                    "command {i} constant {j}: slot/type varies with capacity"
                ));
            }
            if let Some(f) = fit(
                k0.bits,
                ka.bits,
                kb.bits,
                kc.bits,
                &format!("command {i} ({}) constant {j}", c0.command.function),
            )? {
                const_patches.push(CapPatch {
                    cmd_idx: i as u32,
                    target: PatchTarget::Constant(j as u32),
                    floor: f.floor,
                    base: f.base,
                    num: f.num,
                    den: f.den,
                    round_up: f.round_up,
                });
            }
        }
        // Bindings: AttnUnfusedScratch offsets may scale with capacity;
        // every other binding must be identical across probes.
        let nb = c0.command.bindings.len();
        if ca.command.bindings.len() != nb
            || cb.command.bindings.len() != nb
            || cc.command.bindings.len() != nb
        {
            return Err(format!("command {i}: binding count varies with capacity"));
        }
        for bi in 0..nb {
            use scratchy_target_metal::tape::lowered::Binding as B;
            let strip_b = |b: &B| {
                let mut b = *b;
                if let B::AttnUnfusedScratch { offset, .. } = &mut b {
                    *offset = 0;
                }
                format!("{b:?}")
            };
            let (b0, ba, bb2, bc) = (
                &c0.command.bindings[bi],
                &ca.command.bindings[bi],
                &cb.command.bindings[bi],
                &cc.command.bindings[bi],
            );
            let s0b = strip_b(b0);
            if strip_b(ba) != s0b || strip_b(bb2) != s0b || strip_b(bc) != s0b {
                return Err(format!(
                    "command {i} ({}) binding {bi} varies with capacity beyond the \
                     scratch offset: {b0:?} vs {ba:?}",
                    c0.command.function
                ));
            }
            let off = |b: &B| match b {
                B::AttnUnfusedScratch { offset, .. } => Some(*offset),
                _ => None,
            };
            if let (Some(o0), Some(oa), Some(ob), Some(oc)) = (off(b0), off(ba), off(bb2), off(bc))
                && let Some(f) = fit(
                    o0,
                    oa,
                    ob,
                    oc,
                    &format!(
                        "command {i} ({}) binding {bi} scratch offset",
                        c0.command.function
                    ),
                )?
            {
                const_patches.push(CapPatch {
                    cmd_idx: i as u32,
                    target: PatchTarget::AttnScratchOffset(bi as u32),
                    floor: f.floor,
                    base: f.base,
                    num: f.num,
                    den: f.den,
                    round_up: f.round_up,
                });
            }
        }
        // Dispatch scalars — same fit, different target.
        type DGet = fn(&scratchy_target_metal::tape::lowered::GatedCommand) -> u32;
        let scalars: [(PatchTarget, DGet); 7] = [
            (PatchTarget::Threadgroups(0), |c| {
                c.command.dispatch.threadgroups.0
            }),
            (PatchTarget::Threadgroups(1), |c| {
                c.command.dispatch.threadgroups.1
            }),
            (PatchTarget::Threadgroups(2), |c| {
                c.command.dispatch.threadgroups.2
            }),
            (PatchTarget::ThreadsPerThreadgroup(0), |c| {
                c.command.dispatch.threads_per_threadgroup.0
            }),
            (PatchTarget::ThreadsPerThreadgroup(1), |c| {
                c.command.dispatch.threads_per_threadgroup.1
            }),
            (PatchTarget::ThreadsPerThreadgroup(2), |c| {
                c.command.dispatch.threads_per_threadgroup.2
            }),
            (PatchTarget::MScalingBucketM, |c| {
                c.command
                    .dispatch
                    .m_scaling
                    .map(|m| m.bucket_m.0)
                    .unwrap_or(0)
            }),
        ];
        for (target, get) in scalars {
            if let Some(f) = fit(
                get(c0),
                get(ca),
                get(cb),
                get(cc),
                &format!("command {i} ({}) dispatch {target:?}", c0.command.function),
            )? {
                const_patches.push(CapPatch {
                    cmd_idx: i as u32,
                    target,
                    floor: f.floor,
                    base: f.base,
                    num: f.num,
                    den: f.den,
                    round_up: f.round_up,
                });
            }
        }
    }
    if t0.barrier_before != ta.barrier_before
        || t0.barrier_before != tb.barrier_before
        || t0.barrier_before != tc.barrier_before
    {
        return Err("barrier flags vary with block capacity".into());
    }
    let mut scratch_patches = Vec::new();
    type Get = fn(&LoweredMetalTape) -> u32;
    let fields: [(ScratchField, Get); 4] = [
        (ScratchField::SplitK, |t| t.splitk_scratch_bytes),
        (ScratchField::Moe, |t| t.moe_scratch_bytes),
        (ScratchField::RopedK, |t| t.roped_k_scratch_bytes),
        (ScratchField::AttnUnfused, |t| t.attn_unfused_scratch_bytes),
    ];
    for (field, get) in fields {
        if let Some(f) = fit(
            get(&t0),
            get(ta),
            get(tb),
            get(tc),
            &format!("{field:?} scratch bytes"),
        )? {
            scratch_patches.push(ScratchPatch {
                field,
                floor: f.floor,
                base: f.base,
                num: f.num,
                den: f.den,
                round_up: f.round_up,
            });
        }
    }
    Ok((t0, const_patches, scratch_patches))
}

/// Bake every `(gen class × chunked)` variant of one bucket's tape and
/// emit the `&'static [ClassedTape]` expression. Variants that lower
/// identically are deduped (shared entry, both flags listed — the pool
/// matches on class+flag, so dedup means emitting the SAME tape body
/// under each label; the serializer output is compared for equality).
pub fn bake_bucket_tapes(
    mc: &MetalModelConsts,
    input: &BucketLowerInput<'_>,
    pool: &mut CommandPool,
) -> Result<TokenStream, BakeDefect> {
    let classes = [GenClass::M1, GenClass::Mid, GenClass::M5];
    let mut entries: Vec<TokenStream> = Vec::new();
    let mut body_statics: Vec<TokenStream> = Vec::new();
    // Dedupe: identical (tape, patches) bodies share ONE hoisted static
    // — six labelled entries per bucket would otherwise repeat the full
    // command tape six times and rustc drowns in tokens (a 10-family
    // build OOM-killed the compiler before this).
    //
    // Keyed on the LOWERED VALUES, not on their rendered token text: the
    // tape is megabytes of tokens and `classes × chunked` is 6, so a
    // linear scan over `PartialEq` beats stringifying every body.
    type Body = (LoweredMetalTape, Vec<CapPatch>, Vec<ScratchPatch>);
    let mut seen: Vec<Body> = Vec::new();
    let uniq = format!("M{}", input.bucket_m);
    for class in classes {
        let profile = profile_for(class);
        for chunked in [false, true] {
            let l0 = run_lower(mc, input, CAP_ZERO, &profile, chunked)?;
            let ta = run_lower(mc, input, CAP_A, &profile, chunked)?.tape;
            let tb = run_lower(mc, input, CAP_B, &profile, chunked)?.tape;
            let tc = run_lower(mc, input, CAP_CHECK, &profile, chunked)?.tape;
            let body = diff_probes(l0.tape, &ta, &tb, &tc).map_err(BakeDefect)?;
            let class_toks = crate::const_tokens::const_tokens(&class)
                .map_err(|e| BakeDefect(format!("serialize class: {e}")))?;
            // THE DECODE MEGAKERNEL: the bucket-1 tape of the M5 direct-addressing variant.
            let megakernel = match (class, chunked, input.bucket_m) {
                (GenClass::M5, false, 1) => {
                    let (tape, const_patches, _) = &body;
                    let mk = crate::megakernel_bake::bake_megakernel(
                        tape,
                        const_patches,
                        &l0.row_commands,
                        input.steps,
                    )?;
                    let toks = crate::const_tokens::const_tokens(&mk.as_slice())
                        .map_err(|e| BakeDefect(format!("serialize megakernel: {e}")))?;
                    quote! { megakernel: #toks, }
                }
                _ => TokenStream::new(),
            };
            let body_ix = match seen.iter().position(|b| *b == body) {
                Some(ix) => ix,
                None => {
                    let (tape, const_patches, scratch_patches) = &body;
                    let cmd_refs = tape
                        .commands
                        .iter()
                        .map(|c| pool.intern(c))
                        .collect::<Result<Vec<_>, _>>()?;
                    let rest_toks = crate::const_tokens::const_tokens(&LoweredMetalTape {
                        commands: &[],
                        ..*tape
                    })
                    .map_err(|e| BakeDefect(format!("serialize tape: {e}")))?;
                    let tape_toks = quote! {
                        __tl::LoweredMetalTape { commands: &[ #(#cmd_refs),* ], ..#rest_toks }
                    };
                    let cp_toks = crate::const_tokens::const_tokens(&const_patches.as_slice())
                        .map_err(|e| BakeDefect(format!("serialize const patches: {e}")))?;
                    let sp_toks = crate::const_tokens::const_tokens(&scratch_patches.as_slice())
                        .map_err(|e| BakeDefect(format!("serialize scratch patches: {e}")))?;
                    let ix = body_statics.len();
                    let ident = quote::format_ident!("__TAPE_BODY_{uniq}_{ix}");
                    body_statics.push(quote! {
                        const #ident: __tl::ClassedTape = __tl::ClassedTape {
                                // Placeholder label; entries override below.
                                gen_class: __tl::GenClass::M1,
                                chunked: false,
                                tape: #tape_toks,
                                const_patches: #cp_toks,
                                scratch_patches: #sp_toks,
                                megakernel: &[],
                            };
                    });
                    seen.push(body);
                    ix
                }
            };
            let ident = quote::format_ident!("__TAPE_BODY_{uniq}_{body_ix}");
            entries.push(quote! {
                __tl::ClassedTape {
                    gen_class: #class_toks,
                    chunked: #chunked,
                    #megakernel
                    ..#ident
                },
            });
        }
    }
    // ⭐ THE ALIASES THE WHOLE BAKED TAPE RESOLVES THROUGH, opened once for this block. Every
    // literal below names its type through `__tl`/`__tc`/`__ti` instead of spelling
    // `::scratchy_target_metal::tape::<module>::` on the order of a million times — see
    // `const_tokens::alias_preamble`.
    let aliases = crate::const_tokens::alias_preamble();
    Ok(quote! {
        {
            #aliases
            #(#body_statics)*
            &[ #(#entries)* ]
        }
    })
}
