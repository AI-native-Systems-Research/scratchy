// SPDX-License-Identifier: Apache-2.0
//! The weight bundles a program passes whole to one op — a MoE's experts, a Gated-DeltaNet or
//! Mamba-2 mixer's weights (`moe_block(x, mlp[layer])`) — each derived from that op and the
//! model's config, never declared: the op names the bundle's kind, and the config's own keys
//! (the bounds the bridge's MoE expansion reads) name its roles' dims. The typed-Rust emit pass
//! types each role at its derived shape and passes them to the kind's constructor, whose signature
//! relates them to each other and to the op's activations; the bake sizes the bundle's loader by
//! the same derivation; a test holds it to real checkpoints.

use crate::classified::{Expr, OpKind, Program, Stmt, WeightId};
use crate::dim::{Dim, Shape};
pub use scratchy_tensors::bundle_roles::{BundleKind, BundleRole, ExpertNaming};
use std::collections::BTreeMap;

/// One weight bundle: its kind, how its expert bank is named in a checkpoint (the bank's kinds
/// only), and its roles' shapes in [`BundleKind::roles`]' order. A matrix is `[in, out]` (the
/// manifest's orientation), an expert bank `[experts, in, out]`, a conv `[channels, taps]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Bundle {
    pub kind: BundleKind,
    pub naming: Option<ExpertNaming>,
    pub roles: Vec<Shape>,
}

/// One argument of a bundle kind's constructor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BundleArg {
    /// A role, passed as itself.
    Role(BundleRole),
    /// A part: the constructor named first, over these roles.
    Part(&'static str, &'static [BundleRole]),
}

/// The constructor `kind`'s bundle is built by in the signatures.
pub const fn constructor(kind: BundleKind) -> &'static str {
    match kind {
        BundleKind::RoutedExperts => "routed_experts",
        BundleKind::CorrectedExperts => "corrected_experts",
        BundleKind::GatedSharedExperts => "gated_shared_experts",
        BundleKind::UngatedExperts => "ungated_experts",
        BundleKind::BiasedExperts => "biased_experts",
        BundleKind::ScaledRouter => "scaled_router",
        BundleKind::ExpertBank => "expert_bank",
        BundleKind::GatedDeltaNet => "gated_delta_net_mixer",
        BundleKind::Mamba2 => "mamba2_mixer",
    }
}

/// The constructor's arguments: each a role, or a part its own constructor builds from roles — in
/// the order of [`BundleKind::roles`] (a part's roles in place; a test holds the two equal).
pub const fn args(kind: BundleKind) -> &'static [BundleArg] {
    use BundleArg::{Part, Role};
    use BundleRole::*;
    match kind {
        BundleKind::RoutedExperts => &[Role(Router), Role(Gate), Role(Up), Role(Down)],
        BundleKind::CorrectedExperts => &[
            Role(Router),
            Role(Correction),
            Role(Gate),
            Role(Up),
            Role(Down),
        ],
        BundleKind::GatedSharedExperts => &[
            Role(Router),
            Role(Gate),
            Role(Up),
            Role(Down),
            Part("swiglu", &[SharedGate, SharedUp, SharedDown]),
            Role(SharedScale),
        ],
        BundleKind::UngatedExperts => &[Role(Router), Role(Correction), Role(Up), Role(Down)],
        BundleKind::BiasedExperts => &[
            Role(GptOssRouter),
            Role(RouterBias),
            Role(Gate),
            Role(Up),
            Role(Down),
            Part("expert_biases", &[GateBias, UpBias, DownBias]),
        ],
        BundleKind::ScaledRouter => &[Role(Proj), Role(Scale), Role(PerExpertScale)],
        BundleKind::ExpertBank => &[Role(Gate), Role(Up), Role(Down)],
        BundleKind::GatedDeltaNet => &[Role(Conv), Role(ALog), Role(DtBias), Role(Norm)],
        BundleKind::Mamba2 => &[
            Role(Conv),
            Role(ConvBias),
            Role(ALog),
            Role(D),
            Role(DtBias),
            Role(Norm),
        ],
    }
}

/// The activation a MoE's routed experts apply to their in-projection, as the config declares it:
/// `mlp_hidden_act` (Nemotron-H's key) else `hidden_act`, else HF's `PretrainedConfig` default,
/// `"silu"`. Which expert body a MoE block lowers to, and which bundle kind it reads, is this
/// fact's — a gated SiLU bank, or Nemotron-H's ungated relu² `SwitchMLP` — so a body the config
/// does not declare can never be picked silently; an activation no expert body realizes is refused
/// when a MoE op reads it (a dense model never does).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExpertActivation {
    Silu,
    Relu2,
    /// Declared, but no expert body realizes it — the declared name, for the refusal.
    Unrealized(String),
}

impl ExpertActivation {
    /// The activation `json` (a model's config) declares for its experts.
    pub fn of(json: &serde_json::Value) -> Self {
        let act = (json.get("mlp_hidden_act"))
            .or_else(|| json.get("hidden_act"))
            .and_then(|v| v.as_str());
        match act {
            None | Some("silu") => Self::Silu,
            Some("relu2") => Self::Relu2,
            Some(other) => Self::Unrealized(other.to_string()),
        }
    }
}

/// What a model's bundles derive from: the bounds its config states (normalized), and its experts'
/// activation.
#[derive(Clone, Copy, Debug)]
pub struct BundleConfig<'a> {
    pub bounds: &'a BTreeMap<String, u64>,
    pub activation: &'a ExpertActivation,
}

/// How a MoE routes a token to its experts: what its config's spelling of the expert count names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Routing {
    /// `num_local_experts`: the top-k logits, then a softmax over the chosen (Mixtral).
    TopKThenSoftmax,
    /// `n_routed_experts`: sigmoid scores plus a per-expert correction pick the top-k, the
    /// uncorrected sigmoids their weights (GLM-4's, Nemotron-H's `noaux_tc` at one group).
    CorrectedSigmoid,
    /// `num_experts`: a softmax over every expert, then the top-k (Qwen).
    SoftmaxThenTopK,
}

/// The config keys that count a MoE's routed experts, each with the routing it names, in the
/// order they are read: the first the config states counts them.
const EXPERT_COUNTS: [(&str, Routing); 3] = [
    ("num_local_experts", Routing::TopKThenSoftmax),
    ("n_routed_experts", Routing::CorrectedSigmoid),
    ("num_experts", Routing::SoftmaxThenTopK),
];

/// The config keys that size one routed expert's inner width, in the order they are read.
const EXPERT_WIDTHS: [&str; 2] = ["moe_intermediate_size", "intermediate_size"];

/// The key that counts a MoE's routed experts in `bounds` (a config's), and the routing it names:
/// the one source of both for the bundle's roles and the bridge's routing ops.
pub fn routed_experts(bounds: &BTreeMap<String, u64>) -> Option<(&'static str, Routing)> {
    EXPERT_COUNTS
        .into_iter()
        .find(|(k, _)| bounds.contains_key(*k))
}

/// The key that sizes one routed expert's inner width in `bounds` (a config's).
pub fn expert_width(bounds: &BTreeMap<String, u64>) -> Option<&'static str> {
    EXPERT_WIDTHS.into_iter().find(|k| bounds.contains_key(*k))
}

/// The expert count's key and routing, refused when the config states none.
fn count(config: &BundleConfig<'_>) -> Result<(&'static str, Routing), String> {
    routed_experts(config.bounds).ok_or_else(|| {
        let keys: Vec<&str> = EXPERT_COUNTS.iter().map(|(k, _)| *k).collect();
        format!("the config states no routed expert count (none of {keys:?})")
    })
}

/// The expert width's key, refused when the config states none.
fn width(config: &BundleConfig<'_>) -> Result<&'static str, String> {
    expert_width(config.bounds)
        .ok_or_else(|| format!("the config states no expert width (none of {EXPERT_WIDTHS:?})"))
}

/// Every bundle `program` passes whole to an op, keyed by its weight path (`mlp`), derived from
/// that op and `config`. Refused when a bundle's op and config name no kind, when its config lacks
/// a key its roles need, or when two calls derive one bundle two ways.
pub fn derive(
    program: &Program,
    config: &BundleConfig<'_>,
) -> Result<BTreeMap<String, Bundle>, String> {
    let mut calls = Vec::new();
    walk(&program.statements, &mut |op, args| {
        calls.push((op, args.to_vec()))
    });
    let mut out: BTreeMap<String, Bundle> = BTreeMap::new();
    for (op, args) in calls {
        let weights: Vec<WeightId> = (args.iter())
            .filter_map(|a| match a {
                Expr::Weight { id, .. } => Some(*id),
                _ => None,
            })
            .collect();
        let activations = args.len() - weights.len();
        let derived: Vec<(WeightId, Bundle)> = match (op, weights.as_slice()) {
            (OpKind::Moe, [w]) => vec![(*w, moe(program, *w, activations, config)?)],
            (OpKind::GptOssMoe, [w]) => vec![(*w, biased_experts(config)?)],
            (OpKind::GemmaMoe, [router, experts]) => vec![
                (*router, scaled_router(config)?),
                (*experts, expert_bank(config)?),
            ],
            (OpKind::GatedDeltaNet, [w]) => vec![(*w, gated_delta_net())],
            (OpKind::Mamba2, [w]) => vec![(*w, mamba2())],
            (
                OpKind::Moe
                | OpKind::GptOssMoe
                | OpKind::GemmaMoe
                | OpKind::GatedDeltaNet
                | OpKind::Mamba2,
                _,
            ) => {
                return Err(format!(
                    "`{op:?}` passes {} weights, not the bundles its op reads",
                    weights.len()
                ));
            }
            _ => continue,
        };
        for (id, bundle) in derived {
            let path = program.weights.path(id).join(".");
            match out.get(&path) {
                Some(seen) if *seen != bundle => {
                    return Err(format!(
                        "`{path}` is read as two bundles: {seen:?} and {bundle:?}"
                    ));
                }
                _ => {
                    out.insert(path, bundle);
                }
            }
        }
    }
    Ok(out)
}

/// Every op call in `stmts`, its own operands' before it.
fn walk<'a>(stmts: &'a [Stmt], f: &mut impl FnMut(OpKind, &'a [Expr])) {
    fn expr<'a>(e: &'a Expr, f: &mut impl FnMut(OpKind, &'a [Expr])) {
        match e {
            Expr::Call { op, args } => {
                args.iter().for_each(|a| expr(a, f));
                f(*op, args);
            }
            Expr::Mul { lhs, rhs } => {
                expr(lhs, f);
                expr(rhs, f);
            }
            _ => {}
        }
    }
    for s in stmts {
        match s {
            Stmt::Assign { value, .. } | Stmt::AssignTuple { value, .. } => expr(value, f),
            Stmt::For { body, .. } => walk(body, f),
            Stmt::If {
                then_body,
                else_body,
                ..
            } => {
                walk(then_body, f);
                walk(else_body, f);
            }
        }
    }
}

fn bound(name: &str) -> Dim {
    Dim::Bound(name.to_string())
}

fn stated(config: &BundleConfig<'_>, key: &'static str) -> Result<&'static str, String> {
    match config.bounds.contains_key(key) {
        true => Ok(key),
        false => Err(format!("the config states no `{key}`")),
    }
}

/// A routed expert bank's `gate` / `up` / `down` (`[E, D, I]`, `[E, D, I]`, `[E, I, D]`).
fn bank(e: &str, d: &str, i: &str) -> [Shape; 3] {
    let gate = vec![bound(e), bound(d), bound(i)];
    [gate.clone(), gate, vec![bound(e), bound(i), bound(d)]]
}

/// A `moe_block` bundle: its kind is the routing the config's expert count names, the shared
/// expert the config declares and the carrier does not state as ops of its own (a carrier that
/// reads any weight under the bundle's own path owns it), and the experts' activation; its experts
/// read the block's input, or, with two activations, a latent (`moe_latent_size`). Its bank is
/// named as its routing's checkpoints name it: Mixtral's `w1`/`w3`/`w2`, the HF `*_proj` of every
/// other, mlx's `SwitchMLP` `fc1`/`fc2` for un-gated experts.
fn moe(
    program: &Program,
    w: WeightId,
    activations: usize,
    config: &BundleConfig<'_>,
) -> Result<Bundle, String> {
    let (e, routing) = count(config)?;
    let i = width(config)?;
    let hidden = stated(config, "hidden_size")?;
    let width = match activations {
        1 => hidden,
        2 => stated(config, "moe_latent_size")?,
        n => return Err(format!("`moe_block` over {n} activations")),
    };
    let router = vec![bound(hidden), bound(e)];
    let correction = vec![bound(e)];
    match config.activation {
        ExpertActivation::Relu2 => {
            if routing != Routing::CorrectedSigmoid {
                return Err(format!(
                    "relu² experts counted by `{e}`: un-gated experts route by a corrected \
                     sigmoid (`n_routed_experts`)"
                ));
            }
            let [_, up, down] = bank(e, width, i);
            Ok(Bundle {
                kind: BundleKind::UngatedExperts,
                naming: Some(ExpertNaming::Ungated),
                roles: vec![router, correction, up, down],
            })
        }
        ExpertActivation::Unrealized(act) => Err(format!(
            "experts of activation `{act}`: no expert body realizes it"
        )),
        ExpertActivation::Silu => {
            if width != hidden {
                return Err("SwiGLU experts over a latent: no bundle kind holds them".into());
            }
            let [gate, up, down] = bank(e, hidden, i);
            // A carrier that reads any weight under the bundle's own path states that part of
            // the block as ops of its own (Qwen3.5-MoE's shared expert): the bundle holds no more.
            let path = program.weights.path(w);
            let reads_under = (program.weights.ids()).any(|id| {
                let p = program.weights.path(id);
                p.len() > path.len() && p[..path.len()] == *path
            });
            let shared = (config.bounds.get("shared_expert_intermediate_size"))
                .is_some_and(|&s| s > 0)
                && !reads_under;
            let naming = Some(match routing {
                Routing::TopKThenSoftmax => ExpertNaming::Mixtral,
                Routing::CorrectedSigmoid | Routing::SoftmaxThenTopK => ExpertNaming::Qwen,
            });
            Ok(match (routing, shared) {
                (Routing::CorrectedSigmoid, false) => Bundle {
                    kind: BundleKind::CorrectedExperts,
                    naming,
                    roles: vec![router, correction, gate, up, down],
                },
                (Routing::SoftmaxThenTopK, true) => {
                    let s = "shared_expert_intermediate_size";
                    Bundle {
                        kind: BundleKind::GatedSharedExperts,
                        naming,
                        roles: vec![
                            router,
                            gate,
                            up,
                            down,
                            vec![bound(hidden), bound(s)],
                            vec![bound(hidden), bound(s)],
                            vec![bound(s), bound(hidden)],
                            vec![bound(hidden), Dim::Lit(1)],
                        ],
                    }
                }
                (_, false) => Bundle {
                    kind: BundleKind::RoutedExperts,
                    naming,
                    roles: vec![router, gate, up, down],
                },
                (_, true) => {
                    return Err(format!(
                        "a shared expert beside experts counted by `{e}`: no bundle kind holds them"
                    ));
                }
            })
        }
    }
}

/// gpt-oss's experts: a biased router and SwiGLU banks with per-expert biases.
fn biased_experts(config: &BundleConfig<'_>) -> Result<Bundle, String> {
    let (e, _) = count(config)?;
    let i = width(config)?;
    let d = stated(config, "hidden_size")?;
    let [gate, up, down] = bank(e, d, i);
    Ok(Bundle {
        kind: BundleKind::BiasedExperts,
        naming: Some(ExpertNaming::GptOss),
        roles: vec![
            vec![bound(d), bound(e)],
            vec![bound(e)],
            gate,
            up,
            down,
            vec![bound(e), bound(i)],
            vec![bound(e), bound(i)],
            vec![bound(e), bound(d)],
        ],
    })
}

/// Gemma 4's router: its projection, the input's scale, a per-expert output scale.
fn scaled_router(config: &BundleConfig<'_>) -> Result<Bundle, String> {
    let (e, _) = count(config)?;
    let d = stated(config, "hidden_size")?;
    Ok(Bundle {
        kind: BundleKind::ScaledRouter,
        naming: None,
        roles: vec![vec![bound(d), bound(e)], vec![bound(d)], vec![bound(e)]],
    })
}

/// Gemma 4's experts, which its router routes.
fn expert_bank(config: &BundleConfig<'_>) -> Result<Bundle, String> {
    let (e, _) = count(config)?;
    let i = width(config)?;
    let d = stated(config, "hidden_size")?;
    Ok(Bundle {
        kind: BundleKind::ExpertBank,
        naming: Some(ExpertNaming::SwitchGlu),
        roles: bank(e, d, i).to_vec(),
    })
}

/// A Gated-DeltaNet mixer's conv over its packed q/k/v channels, its per-value-head decay and
/// step bias, and its per-value-dim output norm.
fn gated_delta_net() -> Bundle {
    let heads = vec![bound("linear_num_value_heads")];
    Bundle {
        kind: BundleKind::GatedDeltaNet,
        naming: None,
        roles: vec![
            vec![bound("gdn_conv_dim"), bound("linear_conv_kernel_dim")],
            heads.clone(),
            heads,
            vec![bound("linear_value_head_dim")],
        ],
    }
}

/// A Mamba-2 mixer's conv over its x/B/C channels and its bias, its per-head decay, skip and step
/// bias, and its inner-width norm.
fn mamba2() -> Bundle {
    let heads = vec![bound("mamba_num_heads")];
    Bundle {
        kind: BundleKind::Mamba2,
        naming: None,
        roles: vec![
            vec![bound("mamba_conv_dim"), bound("conv_kernel")],
            vec![bound("mamba_conv_dim")],
            heads.clone(),
            heads.clone(),
            heads,
            vec![bound("mamba_d_inner")],
        ],
    }
}

/// A checkpoint's tensors' shapes by name, as its safetensors headers state them.
#[cfg(any(test, feature = "test-support"))]
pub type Header = BTreeMap<String, Vec<u64>>;

/// A checkpoint's MLX-affine group sizes, from its config's `quantization`: the default, and each
/// module's own where it states one (`{"model.layers.0.mlp.gate": {"group_size": 64, ..}}`).
#[cfg(any(test, feature = "test-support"))]
#[derive(Clone, Debug, Default)]
pub struct GroupSizes {
    pub default: Option<u64>,
    pub modules: BTreeMap<String, u64>,
}

#[cfg(any(test, feature = "test-support"))]
impl GroupSizes {
    /// The group size `module` (a tensor's name without its `.weight` / `.scales`) is quantized at.
    pub fn of(&self, module: &str) -> Option<u64> {
        self.modules.get(module).copied().or(self.default)
    }
}

/// A role as a checkpoint holds it, in the derived shape's orientation; a pre-stacked expert bank a
/// conversion flattened to 2-D (`[E·out, in]`, gpt-oss's) keeps that, as the loader reads it.
#[cfg(any(test, feature = "test-support"))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Held {
    Shape(Vec<u64>),
    /// `[E·out, in]` for a derived `[E, in, out]`.
    FlatBank {
        rows: u64,
        cols: u64,
    },
}

#[cfg(any(test, feature = "test-support"))]
impl Held {
    fn is(&self, want: &[u64]) -> bool {
        match (self, want) {
            (Self::Shape(s), _) => s == want,
            (Self::FlatBank { rows, cols }, [e, i, o]) => *rows == e * o && cols == i,
            (Self::FlatBank { .. }, _) => false,
        }
    }
}

#[cfg(any(test, feature = "test-support"))]
impl Bundle {
    /// Check the derived bundle against a checkpoint: each role read from the tensors under
    /// `prefix` (one layer's bundle, e.g. `model.layers.0.mlp`) by [`role_in_checkpoint`] — the
    /// bundle table's names and storage, the loaders' own — against its derived shape evaluated
    /// by `eval` (a config's bounds). Every role that differs, or that the checkpoint does not
    /// hold, is named.
    pub fn check(
        &self,
        header: &Header,
        groups: &GroupSizes,
        prefix: &str,
        eval: impl Fn(&Dim) -> Option<u64>,
    ) -> Result<(), Vec<String>> {
        let mut wrong = Vec::new();
        for (role, derived) in self.kind.roles().iter().zip(&self.roles) {
            let want: Option<Vec<u64>> = derived.iter().map(&eval).collect();
            let name = role.name();
            match (
                want,
                role_in_checkpoint(*role, self.naming, header, groups, prefix),
            ) {
                (Some(want), Ok(held)) if held.is(&want) => {}
                (Some(want), Ok(held)) => wrong.push(format!(
                    "`{name}`: derived {derived:?} = {want:?}, the checkpoint holds {held:?}"
                )),
                (None, _) => wrong.push(format!(
                    "`{name}`: {derived:?} names a bound the config lacks"
                )),
                (_, Err(why)) => wrong.push(format!("`{name}`: {why}")),
            }
        }
        match wrong.is_empty() {
            true => Ok(()),
            false => Err(wrong),
        }
    }
}

/// A bundle role's shape as a checkpoint holds it under `prefix`, read by the bundle table's
/// layout for it ([`BundleRole::layout`]: the names and storage the loaders take it by). A linear
/// is dense `.weight` `[out, in]`, or MLX-affine — its `.scales` `[out, in / group]` give its
/// input width — and is read `[in, out]`; a conv `[C, 1, K]` or `[C, K, 1]` is read `[C, K]`; an
/// expert projection is the MLX-affine bank the loader stacks — pre-stacked under its naming's
/// infix (`[E, out, in / group]` scales, or 2-D `[E·out, ..]`), else each expert's own
/// `experts.{e}` copy — read `[E, in, out]`; a shared expert's gate and up may be fused.
#[cfg(any(test, feature = "test-support"))]
pub fn role_in_checkpoint(
    role: BundleRole,
    naming: Option<ExpertNaming>,
    header: &Header,
    groups: &GroupSizes,
    prefix: &str,
) -> Result<Held, String> {
    use scratchy_tensors::bundle_roles::{Layout, PER_EXPERT_INFIX, SHARED_GATE_UP_FUSED};
    let at = |name: &str| header.get(name).cloned();
    let need = |name: &str| at(name).ok_or_else(|| format!("no `{name}` in the checkpoint"));
    let group = |module: &str| {
        groups.of(module).ok_or_else(|| {
            format!("`{module}` is MLX-affine and the checkpoint states no group size")
        })
    };
    // `[out, in]`: dense, or MLX-affine through its scales.
    let linear = |module: &str| -> Result<(u64, u64), String> {
        match at(&format!("{module}.scales")).as_deref() {
            Some([out, g]) => Ok((*out, g * group(module)?)),
            Some(other) => Err(format!("`{module}.scales` of shape {other:?}")),
            None => match need(&format!("{module}.weight"))?.as_slice() {
                [out, inp] => Ok((*out, *inp)),
                other => Err(format!("`{module}.weight` of shape {other:?}")),
            },
        }
    };
    let layout = role
        .layout(naming)
        .ok_or_else(|| format!("{role:?} is an expert projection and the bundle names no bank"))?;
    match layout {
        Layout::Tensor(leaf) => Ok(Held::Shape(need(&format!("{prefix}.{leaf}"))?)),
        Layout::Linear(leaf) => {
            let fused = format!("{prefix}.{SHARED_GATE_UP_FUSED}");
            let shared_half = matches!(role, BundleRole::SharedGate | BundleRole::SharedUp);
            if shared_half && at(&format!("{fused}.weight")).is_some() {
                let (two_i, inp) = linear(&fused)?;
                return Ok(Held::Shape(vec![inp, two_i / 2]));
            }
            let (out, inp) = linear(&format!("{prefix}.{leaf}"))?;
            Ok(Held::Shape(vec![inp, out]))
        }
        Layout::Conv(leaf) => match need(&format!("{prefix}.{leaf}.weight"))?.as_slice() {
            [c, 1, k] | [c, k, 1] => Ok(Held::Shape(vec![*c, *k])),
            other => Err(format!(
                "a conv of shape {other:?}, not [C, 1, K] or [C, K, 1]"
            )),
        },
        Layout::ExpertProj(naming, role) => {
            let proj = naming
                .proj(role)
                .ok_or_else(|| format!("a {naming:?} bank holds no {role:?} projection"))?;
            let stacked = match naming.stacked_infix() {
                Some(infix) => format!("{prefix}.{infix}.{proj}"),
                None => format!("{prefix}.{proj}"),
            };
            if let Some(scales) = at(&format!("{stacked}.scales")) {
                let g = group(&stacked)?;
                return match scales.as_slice() {
                    [e, out, groups] => Ok(Held::Shape(vec![*e, groups * g, *out])),
                    [rows, groups] => Ok(Held::FlatBank {
                        rows: *rows,
                        cols: groups * g,
                    }),
                    other => Err(format!("`{stacked}.scales` of shape {other:?}")),
                };
            }
            // Each expert's own copy, as the loader stacks it: its own leaf, unless the
            // conversion kept the per-expert HF name (an un-gated bank's).
            let per_expert = |leaf: &str| format!("{prefix}.{PER_EXPERT_INFIX}.0.{leaf}.scales");
            let leaf = match naming.per_expert(role) {
                Some(other) if at(&per_expert(proj)).is_none() => other,
                _ => proj,
            };
            let experts: Vec<Vec<u64>> = (0..)
                .map_while(|e| at(&format!("{prefix}.{PER_EXPERT_INFIX}.{e}.{leaf}.scales")))
                .collect();
            let Some(first) = experts.first() else {
                return Err(format!(
                    "no MLX-affine `{stacked}` bank, and no `{}`",
                    per_expert(leaf)
                ));
            };
            if let Some(other) = experts.iter().find(|s| *s != first) {
                return Err(format!("experts of two shapes, {first:?} and {other:?}"));
            }
            let module = format!("{prefix}.{PER_EXPERT_INFIX}.0.{leaf}");
            match first.as_slice() {
                [out, groups] => Ok(Held::Shape(vec![
                    experts.len() as u64,
                    groups * group(&module)?,
                    *out,
                ])),
                other => Err(format!("`{module}.scales` of shape {other:?}")),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each kind's constructor takes exactly its roles, in the table's order: the typed carrier's
    /// fields and the loaders' names are one list.
    #[test]
    fn every_constructor_takes_its_kinds_roles_in_order() {
        use BundleKind::*;
        for kind in [
            RoutedExperts,
            CorrectedExperts,
            GatedSharedExperts,
            UngatedExperts,
            BiasedExperts,
            ScaledRouter,
            ExpertBank,
            GatedDeltaNet,
            Mamba2,
        ] {
            let flat: Vec<BundleRole> = (args(kind).iter())
                .flat_map(|arg| match arg {
                    BundleArg::Role(role) => std::slice::from_ref(role),
                    BundleArg::Part(_, roles) => roles,
                })
                .copied()
                .collect();
            assert_eq!(flat, kind.roles(), "{kind:?}");
            let names: std::collections::BTreeSet<&str> =
                kind.roles().iter().map(|r| r.name()).collect();
            assert_eq!(
                names.len(),
                kind.roles().len(),
                "{kind:?}: a role name twice"
            );
        }
    }
}
