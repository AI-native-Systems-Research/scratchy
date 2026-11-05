// SPDX-License-Identifier: Apache-2.0
//! The tensors of a weight bundle — a group of checkpoint tensors one op reads whole (a MoE's
//! experts, a Gated-DeltaNet or Mamba-2 mixer's weights) — and where each sits in a checkpoint,
//! under the bundle's prefix: ONE table, read by the loaders that take them and by the test that
//! holds the compiler's derived bundle shapes to real checkpoints. A loader spells no bundle
//! tensor's name of its own.

/// What a weight bundle holds (each kind's roles: [`BundleKind::roles`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BundleKind {
    /// Routed SwiGLU experts: the router and each expert's gate, up and down.
    RoutedExperts,
    /// [`Self::RoutedExperts`] whose routing adds a per-expert score correction.
    CorrectedExperts,
    /// Routed experts and a shared SwiGLU expert, scaled by its own sigmoid gate.
    GatedSharedExperts,
    /// Routed un-gated (relu²) experts with a score correction.
    UngatedExperts,
    /// Routed SwiGLU experts with a biased router and per-expert biases (gpt-oss).
    BiasedExperts,
    /// A router of its own: its projection, the input's scale, a per-expert output scale.
    ScaledRouter,
    /// Unrouted expert banks (a [`Self::ScaledRouter`] routes them).
    ExpertBank,
    /// A Gated-DeltaNet mixer's non-projection weights.
    GatedDeltaNet,
    /// A Mamba-2 mixer's non-projection weights.
    Mamba2,
}

/// One tensor of a bundle.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BundleRole {
    /// The router's projection, at `gate`.
    Router,
    /// gpt-oss's router's projection, at `router`.
    GptOssRouter,
    /// gpt-oss's router's bias.
    RouterBias,
    /// The per-expert score correction a sigmoid router adds (`e_score_correction_bias`).
    Correction,
    Gate,
    Up,
    Down,
    GateBias,
    UpBias,
    DownBias,
    SharedGate,
    SharedUp,
    SharedDown,
    /// The shared expert's sigmoid gate.
    SharedScale,
    /// A scaled router's projection.
    Proj,
    /// A scaled router's input scale.
    Scale,
    PerExpertScale,
    Conv,
    ConvBias,
    ALog,
    D,
    DtBias,
    Norm,
}

impl BundleRole {
    /// Its name: a typed carrier's field is `<bundle>_<name>`.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Router | Self::GptOssRouter => "router",
            Self::RouterBias => "router_bias",
            Self::Correction => "correction",
            Self::Gate => "gate",
            Self::Up => "up",
            Self::Down => "down",
            Self::GateBias => "gate_bias",
            Self::UpBias => "up_bias",
            Self::DownBias => "down_bias",
            Self::SharedGate => "shared_gate",
            Self::SharedUp => "shared_up",
            Self::SharedDown => "shared_down",
            Self::SharedScale => "shared_scale",
            Self::Proj => "proj",
            Self::Scale => "scale",
            Self::PerExpertScale => "per_expert_scale",
            Self::Conv => "conv",
            Self::ConvBias => "conv_bias",
            Self::ALog => "a_log",
            Self::D => "d",
            Self::DtBias => "dt_bias",
            Self::Norm => "norm",
        }
    }

    /// Where it sits under its bundle's prefix, and how it is stored. An expert projection's is
    /// its bank's naming's; `None` for one asked without a naming.
    pub const fn layout(self, naming: Option<ExpertNaming>) -> Option<Layout> {
        Some(match self {
            Self::Router => Layout::Linear("gate"),
            Self::GptOssRouter => Layout::Linear("router"),
            Self::RouterBias => Layout::Tensor("router.bias"),
            Self::Correction => Layout::Tensor("gate.e_score_correction_bias"),
            Self::Gate | Self::Up | Self::Down => match naming {
                Some(naming) => Layout::ExpertProj(naming, self),
                None => return None,
            },
            Self::GateBias => Layout::Tensor("experts.gate_proj.bias"),
            Self::UpBias => Layout::Tensor("experts.up_proj.bias"),
            Self::DownBias => Layout::Tensor("experts.down_proj.bias"),
            Self::SharedGate => Layout::Linear("shared_expert.gate_proj"),
            Self::SharedUp => Layout::Linear("shared_expert.up_proj"),
            Self::SharedDown => Layout::Linear("shared_expert.down_proj"),
            Self::SharedScale => Layout::Linear("shared_expert_gate"),
            Self::Proj => Layout::Linear("proj"),
            Self::Scale => Layout::Tensor("scale"),
            Self::PerExpertScale => Layout::Tensor("per_expert_scale"),
            Self::Conv => Layout::Conv("conv1d"),
            Self::ConvBias => Layout::Tensor("conv1d.bias"),
            Self::ALog => Layout::Tensor("A_log"),
            Self::D => Layout::Tensor("D"),
            Self::DtBias => Layout::Tensor("dt_bias"),
            Self::Norm => Layout::Tensor("norm.weight"),
        })
    }

    /// Its leaf under its bundle's prefix: every role but an expert projection (whose leaf is its
    /// bank's naming's, [`ExpertNaming::proj`]) has one.
    pub const fn leaf(self) -> Option<&'static str> {
        match self.layout(None) {
            Some(Layout::Tensor(l) | Layout::Linear(l) | Layout::Conv(l)) => Some(l),
            Some(Layout::ExpertProj(..)) | None => None,
        }
    }
}

/// `role`'s leaf, for a `const` item: a role without one (an expert projection, whose leaf is its
/// bank's naming's) fails the build there.
pub const fn const_leaf(role: BundleRole) -> &'static str {
    match role.leaf() {
        Some(leaf) => leaf,
        None => panic!("an expert projection's leaf is its bank's naming's"),
    }
}

/// The infix each expert's own copy of a projection sits under: `{prefix}.experts.{e}.{proj}`.
pub const PER_EXPERT_INFIX: &str = "experts";

/// The leaf of a shared expert's gate and up projections fused into one (`[2·I, in]`), which a
/// checkpoint may hold in place of [`BundleRole::SharedGate`] and [`BundleRole::SharedUp`].
pub const SHARED_GATE_UP_FUSED: &str = "shared_expert.gate_up_proj";

impl BundleKind {
    /// Its roles, in its constructor's argument order.
    pub const fn roles(self) -> &'static [BundleRole] {
        use BundleRole::*;
        match self {
            Self::RoutedExperts => &[Router, Gate, Up, Down],
            Self::CorrectedExperts => &[Router, Correction, Gate, Up, Down],
            Self::GatedSharedExperts => &[
                Router,
                Gate,
                Up,
                Down,
                SharedGate,
                SharedUp,
                SharedDown,
                SharedScale,
            ],
            Self::UngatedExperts => &[Router, Correction, Up, Down],
            Self::BiasedExperts => &[
                GptOssRouter,
                RouterBias,
                Gate,
                Up,
                Down,
                GateBias,
                UpBias,
                DownBias,
            ],
            Self::ScaledRouter => &[Proj, Scale, PerExpertScale],
            Self::ExpertBank => &[Gate, Up, Down],
            Self::GatedDeltaNet => &[Conv, ALog, DtBias, Norm],
            Self::Mamba2 => &[Conv, ConvBias, ALog, D, DtBias, Norm],
        }
    }
}

/// How an expert bank's projections are named in its checkpoint.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ExpertNaming {
    /// Mixtral's pre-Llama-2 `w1` (gate), `w3` (up), `w2` (down).
    Mixtral,
    /// `gate_proj`, `up_proj`, `down_proj` (Qwen-MoE, GLM-4, every other HF MoE).
    Qwen,
    /// mlx `SwitchMLP`'s un-gated `fc1` (up) and `fc2` (down); a conversion keeping the per-expert
    /// HF names spells them `up_proj` / `down_proj`.
    Ungated,
    /// Gemma 4's SwitchGLU: `gate_proj` / `up_proj` / `down_proj`, pre-stacked directly under the
    /// bundle's prefix (the `experts.switch_glu` infix is the prefix's own).
    SwitchGlu,
    /// gpt-oss's: `gate_proj` / `up_proj` / `down_proj`, pre-stacked under `experts`.
    GptOss,
}

impl ExpertNaming {
    /// The leaf of the projection `role` (`Gate`, `Up` or `Down`); `None` for a role the bank does
    /// not hold (an un-gated bank's gate) or one that is no projection.
    pub const fn proj(self, role: BundleRole) -> Option<&'static str> {
        match (self, role) {
            (Self::Mixtral, BundleRole::Gate) => Some("w1"),
            (Self::Mixtral, BundleRole::Up) => Some("w3"),
            (Self::Mixtral, BundleRole::Down) => Some("w2"),
            (Self::Ungated, BundleRole::Up) => Some("fc1"),
            (Self::Ungated, BundleRole::Down) => Some("fc2"),
            (Self::Ungated, _) => None,
            (_, BundleRole::Gate) => Some("gate_proj"),
            (_, BundleRole::Up) => Some("up_proj"),
            (_, BundleRole::Down) => Some("down_proj"),
            _ => None,
        }
    }

    /// The leaf each expert's own copy of the projection `role` takes (`{prefix}.experts.{e}.<leaf>`):
    /// [`Self::proj`]'s, but an un-gated bank's, whose per-expert files keep the HF `up_proj` /
    /// `down_proj` where its pre-stacked ones are mlx's `fc1` / `fc2`.
    pub const fn per_expert(self, role: BundleRole) -> Option<&'static str> {
        match (self, role) {
            (Self::Ungated, BundleRole::Up) => Some("up_proj"),
            (Self::Ungated, BundleRole::Down) => Some("down_proj"),
            _ => self.proj(role),
        }
    }

    /// The infix a pre-stacked bank's projections sit under, below the bundle's prefix.
    pub const fn stacked_infix(self) -> Option<&'static str> {
        match self {
            Self::SwitchGlu => None,
            Self::GptOss => Some("experts"),
            Self::Mixtral | Self::Qwen | Self::Ungated => Some("switch_mlp"),
        }
    }
}

/// Where a bundle role sits under its bundle's prefix, and how it is stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layout {
    /// One tensor, `{prefix}.{leaf}`, read as stored.
    Tensor(&'static str),
    /// A linear at `{prefix}.{leaf}`: dense `.weight` `[out, in]`, or MLX-affine (`.weight`
    /// packed, with `.scales` / `.biases` `[out, in / group]`); read `[in, out]`.
    Linear(&'static str),
    /// A depthwise conv `{prefix}.{leaf}.weight`, stored `[C, 1, K]` (or mlx's `[C, K, 1]`, the
    /// same bytes); read `[C, K]`.
    Conv(&'static str),
    /// An expert bank's projection: pre-stacked `{prefix}[.{infix}].{proj}` (`[E, out, in]`, or
    /// 2-D `[E·out, in]`), else each expert's own `{prefix}.experts.{e}.{proj}`; MLX-affine, read
    /// `[E, in, out]`. See [`ExpertNaming`].
    ExpertProj(ExpertNaming, BundleRole),
}
