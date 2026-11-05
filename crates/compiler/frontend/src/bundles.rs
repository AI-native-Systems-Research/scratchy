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
use std::collections::BTreeMap;

/// One weight bundle: its kind, and its roles' shapes in the kind's role order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Bundle {
    pub kind: BundleKind,
    /// Each role's shape, in [`BundleKind::roles`]' order.
    pub roles: Vec<Shape>,
}

/// What a weight bundle holds: the roles an op reads from it, each a tensor of the checkpoint. A
/// matrix is `[in, out]` (the manifest's orientation), an expert bank `[experts, in, out]`, a conv
/// `[channels, taps]` (the on-disk singleton axis dropped, as the loader drops it).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BundleKind {
    /// Routed SwiGLU experts: the router and each expert's gate, up and down.
    RoutedExperts,
    /// [`Self::RoutedExperts`] whose routing adds a per-expert score correction
    /// (`e_score_correction_bias`).
    CorrectedExperts,
    /// Routed experts and a shared SwiGLU expert, scaled by its own sigmoid gate.
    GatedSharedExperts,
    /// Routed un-gated (relu²) experts with a score correction, on their own width (a latent MoE's
    /// latent, or the block's input).
    UngatedExperts,
    /// Routed SwiGLU experts with a router bias and per-expert biases.
    BiasedExperts,
    /// A router of its own: its projection, the input's scale and a per-expert output scale.
    ScaledRouter,
    /// Unrouted expert banks (a [`Self::ScaledRouter`] routes them).
    ExpertBank,
    /// A Gated-DeltaNet mixer's non-projection weights.
    GatedDeltaNet,
    /// A Mamba-2 mixer's non-projection weights.
    Mamba2,
}

/// One argument of a bundle kind's constructor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BundleArg {
    /// A role, passed as itself.
    Role(&'static str),
    /// A part: the constructor named first, over these roles.
    Part(&'static str, &'static [&'static str]),
}

impl BundleKind {
    /// Its constructor's name in the signatures.
    pub const fn name(self) -> &'static str {
        match self {
            Self::RoutedExperts => "routed_experts",
            Self::CorrectedExperts => "corrected_experts",
            Self::GatedSharedExperts => "gated_shared_experts",
            Self::UngatedExperts => "ungated_experts",
            Self::BiasedExperts => "biased_experts",
            Self::ScaledRouter => "scaled_router",
            Self::ExpertBank => "expert_bank",
            Self::GatedDeltaNet => "gated_delta_net_mixer",
            Self::Mamba2 => "mamba2_mixer",
        }
    }

    /// Its constructor's arguments: each a role, or a part its own constructor builds from roles.
    pub const fn args(self) -> &'static [BundleArg] {
        use BundleArg::{Part, Role};
        match self {
            Self::RoutedExperts => &[Role("router"), Role("gate"), Role("up"), Role("down")],
            Self::CorrectedExperts => &[
                Role("router"),
                Role("correction"),
                Role("gate"),
                Role("up"),
                Role("down"),
            ],
            Self::GatedSharedExperts => &[
                Role("router"),
                Role("gate"),
                Role("up"),
                Role("down"),
                Part("swiglu", &["shared_gate", "shared_up", "shared_down"]),
                Role("shared_scale"),
            ],
            Self::UngatedExperts => &[Role("router"), Role("correction"), Role("up"), Role("down")],
            Self::BiasedExperts => &[
                Role("router"),
                Role("router_bias"),
                Role("gate"),
                Role("up"),
                Role("down"),
                Part("expert_biases", &["gate_bias", "up_bias", "down_bias"]),
            ],
            Self::ScaledRouter => &[Role("proj"), Role("scale"), Role("per_expert_scale")],
            Self::ExpertBank => &[Role("gate"), Role("up"), Role("down")],
            Self::GatedDeltaNet => &[Role("conv"), Role("a_log"), Role("dt_bias"), Role("norm")],
            Self::Mamba2 => &[
                Role("conv"),
                Role("conv_bias"),
                Role("a_log"),
                Role("d"),
                Role("dt_bias"),
                Role("norm"),
            ],
        }
    }

    /// Its roles, in its constructor's argument order (a part's in place).
    pub fn roles(self) -> Vec<&'static str> {
        (self.args().iter())
            .flat_map(|arg| match arg {
                BundleArg::Role(role) => std::slice::from_ref(role),
                BundleArg::Part(_, roles) => roles,
            })
            .copied()
            .collect()
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
/// expert the config declares and the carrier does not state as ops of its own (a carrier reading
/// `<bundle>.shared_expert.*` owns it), and the experts' activation; its experts read the block's
/// input, or, with two activations, a latent (`moe_latent_size`).
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
            let path = program.weights.path(w);
            let owns_shared = (program.weights.ids()).any(|id| {
                let p = program.weights.path(id);
                p.len() > path.len() && p[..path.len()] == *path && p[path.len()] == "shared_expert"
            });
            let shared = (config.bounds.get("shared_expert_intermediate_size"))
                .is_some_and(|&s| s > 0)
                && !owns_shared;
            Ok(match (routing, shared) {
                (Routing::CorrectedSigmoid, false) => Bundle {
                    kind: BundleKind::CorrectedExperts,
                    roles: vec![router, correction, gate, up, down],
                },
                (Routing::SoftmaxThenTopK, true) => {
                    let s = "shared_expert_intermediate_size";
                    Bundle {
                        kind: BundleKind::GatedSharedExperts,
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
        roles: bank(e, d, i).to_vec(),
    })
}

/// A Gated-DeltaNet mixer's conv over its packed q/k/v channels, its per-value-head decay and
/// step bias, and its per-value-dim output norm.
fn gated_delta_net() -> Bundle {
    let heads = vec![bound("linear_num_value_heads")];
    Bundle {
        kind: BundleKind::GatedDeltaNet,
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

#[cfg(any(test, feature = "test-support"))]
impl Bundle {
    /// Check the derived bundle against a checkpoint: each role read from the tensors under
    /// `prefix` (one layer's bundle, e.g. `model.layers.0.mlp`) by [`role_in_checkpoint`], against
    /// its derived shape evaluated by `eval` (a config's bounds). Every role that differs, or that
    /// the checkpoint does not hold, is named.
    pub fn check(
        &self,
        header: &Header,
        prefix: &str,
        eval: impl Fn(&Dim) -> Option<u64>,
    ) -> Result<(), Vec<String>> {
        let mut wrong = Vec::new();
        for (role, derived) in self.kind.roles().into_iter().zip(&self.roles) {
            let want: Option<Vec<u64>> = derived.iter().map(&eval).collect();
            match (want, role_in_checkpoint(role, header, prefix)) {
                (Some(want), Ok(held)) if want == held => {}
                (Some(want), Ok(held)) => wrong.push(format!(
                    "`{role}`: derived {derived:?} = {want:?}, the checkpoint holds {held:?}"
                )),
                (None, _) => wrong.push(format!(
                    "`{role}`: {derived:?} names a bound the config lacks"
                )),
                (_, Err(why)) => wrong.push(format!("`{role}`: {why}")),
            }
        }
        match wrong.is_empty() {
            true => Ok(()),
            false => Err(wrong),
        }
    }
}

/// A bundle role's shape as a checkpoint holds it under `prefix`, in the manifest's orientation:
/// the layouts the loader reads it from. A matrix is stored `[out, in]` and read `[in, out]`; an
/// expert bank is stacked `[experts, in, out]` from per-expert `experts.{e}.<proj>.weight`
/// (`gate_proj` / `up_proj` / `down_proj`, Mixtral's `w1` / `w3` / `w2`), from a pre-stacked
/// `experts.<proj>` `[E, out, in]`, from a fused `experts.gate_up_proj` `[E, 2·I, in]`, or from
/// gpt-oss's MXFP4 `experts.<proj>_blocks` `[E, out, in / 32, 16]`; a conv `[C, 1, K]` is read
/// `[C, K]`, its singleton axis dropped.
#[cfg(any(test, feature = "test-support"))]
pub fn role_in_checkpoint(role: &str, header: &Header, prefix: &str) -> Result<Vec<u64>, String> {
    let at = |leaf: &str| header.get(&format!("{prefix}.{leaf}")).cloned();
    let first = |leaves: &[&str]| -> Result<Vec<u64>, String> {
        (leaves.iter()).find_map(|l| at(l)).ok_or_else(|| {
            let names: Vec<String> = leaves.iter().map(|l| format!("{prefix}.{l}")).collect();
            format!("none of {} in the checkpoint", names.join(", "))
        })
    };
    let transposed = |s: Vec<u64>| match s.as_slice() {
        [out, inp] => Ok(vec![*inp, *out]),
        _ => Err(format!("a matrix of rank {}", s.len())),
    };
    match role {
        "router" => transposed(first(&["gate.weight", "router.weight"])?),
        "correction" => first(&["gate.e_score_correction_bias"]),
        "router_bias" => first(&["router.bias"]),
        "proj" => transposed(first(&["proj.weight"])?),
        "scale" => first(&["scale"]),
        "per_expert_scale" => first(&["per_expert_scale"]),
        "shared_gate" | "shared_up" | "shared_down" => {
            let proj = &role["shared_".len()..];
            transposed(first(&[
                &format!("shared_expert.{proj}_proj.weight"),
                &format!("shared_experts.{proj}_proj.weight"),
            ])?)
        }
        "shared_scale" => transposed(first(&["shared_expert_gate.weight"])?),
        "gate_bias" | "up_bias" => {
            let fused = first(&["experts.gate_up_proj_bias"])?;
            match fused.as_slice() {
                [e, two_i] if two_i % 2 == 0 => Ok(vec![*e, two_i / 2]),
                _ => Err(format!("a fused gate/up bias of shape {fused:?}")),
            }
        }
        "down_bias" => first(&["experts.down_proj_bias"]),
        "gate" | "up" | "down" => stacked_bank(role, header, prefix),
        "conv" => match first(&["conv1d.weight"])?.as_slice() {
            [c, 1, k] => Ok(vec![*c, *k]),
            other => Err(format!(
                "a conv of shape {other:?}, not [channels, 1, taps]"
            )),
        },
        "conv_bias" => first(&["conv1d.bias"]),
        "a_log" => first(&["A_log"]),
        "d" => first(&["D"]),
        "dt_bias" => first(&["dt_bias"]),
        "norm" => first(&["norm.weight"]),
        other => Err(format!("no checkpoint layout for the role `{other}`")),
    }
}

/// An expert bank's stacked `[experts, in, out]` ([`role_in_checkpoint`]'s layouts).
#[cfg(any(test, feature = "test-support"))]
fn stacked_bank(role: &str, header: &Header, prefix: &str) -> Result<Vec<u64>, String> {
    let names: &[&str] = match role {
        "gate" => &["gate_proj", "w1"],
        "up" => &["up_proj", "w3"],
        _ => &["down_proj", "w2"],
    };
    // Per-expert: every expert's `[out, in]`, one shape.
    for name in names {
        let experts: Vec<&Vec<u64>> = (0..)
            .map_while(|e| header.get(&format!("{prefix}.experts.{e}.{name}.weight")))
            .collect();
        if let Some(first) = experts.first() {
            if let Some(other) = experts.iter().find(|s| *s != first) {
                return Err(format!("experts of two shapes, {first:?} and {other:?}"));
            }
            return match first.as_slice() {
                [out, inp] => Ok(vec![experts.len() as u64, *inp, *out]),
                _ => Err(format!("an expert matrix of shape {first:?}")),
            };
        }
    }
    let stacked = |leaf: &str| header.get(&format!("{prefix}.experts.{leaf}"));
    // Pre-stacked, one projection.
    for name in names {
        if let Some(s) = stacked(name) {
            return match s.as_slice() {
                [e, out, inp] => Ok(vec![*e, *inp, *out]),
                _ => Err(format!("a stacked bank of shape {s:?}")),
            };
        }
    }
    // MXFP4 blocks: 32 codes a block.
    let blocks = |leaf: &str| match stacked(leaf).map(Vec::as_slice) {
        Some([e, out, nb, 16]) => Some((*e, *out, nb * 32)),
        _ => None,
    };
    let (fused, fused_blocks) = (stacked("gate_up_proj"), blocks("gate_up_proj_blocks"));
    match role {
        "gate" | "up" => {
            if let Some([e, two_i, inp]) = fused.map(Vec::as_slice) {
                return Ok(vec![*e, *inp, two_i / 2]);
            }
            if let Some((e, two_i, inp)) = fused_blocks {
                return Ok(vec![e, inp, two_i / 2]);
            }
        }
        _ => {
            if let Some((e, out, inp)) = blocks("down_proj_blocks") {
                return Ok(vec![e, inp, out]);
            }
        }
    }
    Err(format!("no `{role}` expert bank under {prefix}.experts"))
}
