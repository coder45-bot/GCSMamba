"""
HybridTopoMamba — GCSMamba implementation for model.py
=======================================================
Drop-in class following the same (dataset, args) convention as
GCN_Net, GAT_Net, GCNII, GPRGNN, GCN_mamba_Net.

Two model variants controlled by args.use_gce (bool):
  - use_gce=True  : Full GCSMamba with Graph Context Encoder (GCE)
                    topology-conditioned delta and A modulation
  - use_gce=False : Ablation — GCE disabled, standard SSM dynamics
                    (no topology conditioning, just depth-fused diffusion)

args fields required:
    args.hidden        int   — hidden dimension
    args.d_state       int   — SSM state size (default 16)
    args.ctx_dim       int   — graph context embedding dim (default 8)
    args.layer_num     int   — propagation depth / scan steps
    args.dropout       float — dropout probability
    args.dataset       str   — 'arxiv' uses adj_t, others use edge_index
    args.use_gce       bool  — True=full model, False=ablation without GCE

Install requirement: pip install torch-sparse
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module, Linear, Parameter, Dropout
from torch_sparse import SparseTensor


# ─────────────────────────────────────────────────────────────
# 1. RMSNorm
# ─────────────────────────────────────────────────────────────

class RMSNorm(Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = Parameter(torch.ones(d_model))

    def forward(self, x: Tensor) -> Tensor:
        return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
                * self.weight)


# ─────────────────────────────────────────────────────────────
# 2. Graph Context Encoder (GCE)
#    Extracts per-node topology descriptors:
#    degree (scalar) + neighbor mean (d,) + neighbor max (d,)
#    → compact ctx_dim embedding → sigmoid gate γ_i ∈ (0,1)
#    → modulation vectors Δ_g and A_g
# ─────────────────────────────────────────────────────────────

class GraphContextEncoder(Module):
    def __init__(self, in_ch: int, ctx_dim: int, d_model: int):
        """
        in_ch   : input feature dimension (same as d_model after projection)
        ctx_dim : compact context embedding dimension
        d_model : output dimension for modulation vectors Δ_g and A_g
        """
        super().__init__()
        # Projects [deg_feat(1) | neighbor_mean(in_ch) | neighbor_max(in_ch)]
        self.ctx_proj   = Linear(2 * in_ch + 1, ctx_dim)
        # Gate: scalar γ_i per node
        self.gate_proj  = Linear(ctx_dim, 1)
        # Topology modulation for delta and A
        self.delta_proj = Linear(ctx_dim, d_model)
        self.A_proj     = Linear(ctx_dim, d_model)

    def reset_parameters(self):
        self.ctx_proj.reset_parameters()
        self.gate_proj.reset_parameters()
        self.delta_proj.reset_parameters()
        self.A_proj.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor, num_nodes: int):
        """
        Returns:
            gamma   : (N, 1)        sigmoid gate per node
            delta_g : (N, d_model)  topology modulation for step size
            A_g     : (N, d_model)  topology modulation for decay
        """
        row, col = edge_index

        # ── Node degree (log-scaled for numerical stability) ──
        deg = torch.zeros(num_nodes, device=x.device)
        deg.scatter_add_(0, col, torch.ones(col.size(0), device=x.device))
        deg_feat = torch.log1p(deg).unsqueeze(-1)           # (N, 1)

        # ── Neighbor mean ──────────────────────────────────────
        neighbor_sum = torch.zeros_like(x)
        neighbor_sum.scatter_add_(
            0, col.unsqueeze(1).expand(-1, x.size(1)), x[row])
        neighbor_mean = neighbor_sum / deg.clamp(min=1.0).unsqueeze(-1)

        # ── Neighbor max ───────────────────────────────────────
        neighbor_max = torch.full_like(x, float("-inf"))
        neighbor_max.scatter_reduce_(
            0, col.unsqueeze(1).expand(-1, x.size(1)),
            x[row], reduce="amax", include_self=True)
        neighbor_max = torch.where(
            torch.isinf(neighbor_max),
            torch.zeros_like(neighbor_max), neighbor_max)

        # ── Context embedding ──────────────────────────────────
        ctx_input = torch.cat([deg_feat, neighbor_mean, neighbor_max], dim=-1)
        g = F.relu(self.ctx_proj(ctx_input))                # (N, ctx_dim)

        gamma   = torch.sigmoid(self.gate_proj(g))          # (N, 1)
        delta_g = self.delta_proj(g)                        # (N, d_model)
        A_g     = self.A_proj(g)                            # (N, d_model)

        return gamma, delta_g, A_g


# ─────────────────────────────────────────────────────────────
# 3. Sparse normalized adjacency builder
# ─────────────────────────────────────────────────────────────

def build_norm_adj(edge_index: Tensor, num_nodes: int,
                   device) -> SparseTensor:
    """
    Sparse D^{-1/2}(A+I)D^{-1/2} as SparseTensor — O(E) memory.
    """
    row, col = edge_index
    self_idx = torch.arange(num_nodes, device=device)
    full_row = torch.cat([row, col, self_idx])
    full_col = torch.cat([col, row, self_idx])
    full_val = torch.ones(full_row.size(0), device=device)

    adj = SparseTensor(
        row=full_row, col=full_col, value=full_val,
        sparse_sizes=(num_nodes, num_nodes),
    ).coalesce()

    deg = adj.sum(dim=1).to(torch.float)
    d_inv_sqrt = deg.pow(-0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    return d_inv_sqrt.view(-1, 1) * adj * d_inv_sqrt.view(1, -1)


# ─────────────────────────────────────────────────────────────
# 4. Core scan: topology-conditioned SSM + depth-fused diffusion
# ─────────────────────────────────────────────────────────────

class TopoConditionedPropagationScan(Module):
    """
    Selective SSM over propagation depth with depth-fused graph diffusion.

    When use_gce=True:
        Δ_t = Δ_base + γ_i ⊙ Δ_g          (topology-conditioned step size)
        A_t = A_base − γ_i ⊙ softplus(A_g) (topology-conditioned decay)

    When use_gce=False (ablation):
        Δ_t = Δ_base                        (standard Mamba dynamics)
        A_t = A_base                        (no topology modulation)

    In both cases, after every state update:
        S ← Â S                             (depth-fused graph diffusion)
    """
    def __init__(self, d_model: int, d_state: int, num_steps: int,
                 dropout: float = 0.0, use_gce: bool = True):
        super().__init__()
        self.d_model   = d_model
        self.d_state   = d_state
        self.num_steps = num_steps
        self.use_gce   = use_gce

        self.in_proj   = Linear(d_model, d_model * 2)
        self.x_proj    = Linear(d_model, d_state * 2)   # → B, C
        self.dt_proj   = Linear(d_model, d_model)        # → Δ_base
        self.out_proj  = Linear(d_model, d_model)

        # Learnable diagonal A (log-parameterised, always negative)
        A_init = torch.arange(1, d_state + 1, dtype=torch.float32
                              ).unsqueeze(0).repeat(d_model, 1)
        self.A_log = Parameter(torch.log(A_init))
        self.D     = Parameter(torch.ones(d_model))     # skip connection

        self.drop = Dropout(dropout)

    def reset_parameters(self):
        for layer in [self.in_proj, self.x_proj, self.dt_proj, self.out_proj]:
            layer.reset_parameters()

    def forward(self, x: Tensor, norm_adj: SparseTensor,
                gamma=None, delta_g=None, A_g=None) -> Tensor:
        """
        x        : (N, d_model)
        norm_adj : sparse normalized adjacency
        gamma    : (N, 1)        — from GCE; None when use_gce=False
        delta_g  : (N, d_model) — from GCE; None when use_gce=False
        A_g      : (N, d_model) — from GCE; None when use_gce=False
        """
        N = x.size(0)

        # Expand node features along depth dimension
        x_seq = x.unsqueeze(1).expand(N, self.num_steps,
                                       self.d_model).contiguous()

        # Input and residual branches
        x_and_res = self.in_proj(x_seq)                    # (N, T, 2D)
        x_in, res = x_and_res.split(self.d_model, dim=-1)
        x_in = F.relu(x_in)

        # SSM projections
        x_dbl  = self.x_proj(x_in)                        # (N, T, 2*S)
        B, C   = x_dbl.split(self.d_state, dim=-1)        # (N, T, S) each
        delta_base = F.softplus(self.dt_proj(x_in))       # (N, T, D)

        A_base = -torch.exp(self.A_log)                    # (D, S)

        state = torch.zeros(N, self.d_model, self.d_state,
                            device=x.device, dtype=x.dtype)
        ys = []

        for t in range(self.num_steps):
            u_t     = x_in[:, t, :]                        # (N, D)
            B_t     = B[:, t, :]                           # (N, S)
            C_t     = C[:, t, :]                           # (N, S)
            delta_t = delta_base[:, t, :]                  # (N, D)

            # ── Topology-conditioned modulation ────────────────
            if self.use_gce and gamma is not None:
                # Eq. Δ_t = Δ_base + γ_i ⊙ Δ_g
                delta_t = delta_t + gamma * delta_g        # (N, D)
                # Eq. A_t = A_base − γ_i ⊙ softplus(A_g)
                A_t = (A_base.unsqueeze(0)
                       - (gamma * F.softplus(A_g)
                          ).unsqueeze(-1))                  # (N, D, S)
            else:
                # Ablation: standard Mamba, no topology modulation
                A_t = A_base.unsqueeze(0).expand(
                    N, -1, -1)                              # (N, D, S)

            # ── ZOH discretisation ─────────────────────────────
            dA  = torch.exp(delta_t.unsqueeze(-1) * A_t)   # (N, D, S)
            dBu = (delta_t.unsqueeze(-1)
                   * B_t.unsqueeze(1)
                   * u_t.unsqueeze(-1))                     # (N, D, S)

            # ── State update ───────────────────────────────────
            state = dA * state + dBu

            # ── Output ─────────────────────────────────────────
            y_t = torch.einsum('nds,ns->nd', state, C_t)   # (N, D)

            # ── Depth-fused graph diffusion ────────────────────
            # S ← Â S  (one hop, inside every recurrence step)
            state = self.drop(state)
            state_flat = state.reshape(N, self.d_model * self.d_state)
            state_flat = norm_adj @ state_flat
            state = state_flat.reshape(N, self.d_model, self.d_state)

            ys.append(y_t.unsqueeze(1))

        y = torch.cat(ys, dim=1)                           # (N, T, D)
        y = y + x_in * self.D                             # skip connection
        y = y * F.relu(res)                               # gated residual
        return self.out_proj(y)[:, -1, :]                 # (N, D)


# ─────────────────────────────────────────────────────────────
# 5. Full model
# ─────────────────────────────────────────────────────────────

class HybridTopoMamba_Net(Module):
    """
    Full GCSMamba model.

    Set args.use_gce = True  → full model with Graph Context Encoder
    Set args.use_gce = False → ablation without GCE (no topo conditioning)

    All other args / forward() interface identical to GCN_mamba_Net.
    """
    def __init__(self, dataset, args):
        super().__init__()
        self.dataset  = args.dataset
        self.dropout  = args.dropout
        self.use_gce  = getattr(args, "use_gce", True)

        d_state  = getattr(args, "d_state",  16)
        ctx_dim  = getattr(args, "ctx_dim",   8)
        hidden   = args.hidden

        # ── Input projection ───────────────────────────────────
        self.input_proj = Linear(dataset.num_features, hidden)
        self.norm_in    = RMSNorm(hidden)

        # ── Graph Context Encoder (conditional) ────────────────
        if self.use_gce:
            self.ctx_encoder = GraphContextEncoder(
                in_ch=hidden, ctx_dim=ctx_dim, d_model=hidden)
        else:
            self.ctx_encoder = None  # ablation: encoder not instantiated

        # ── Core scan ──────────────────────────────────────────
        self.scan = TopoConditionedPropagationScan(
            d_model=hidden, d_state=d_state,
            num_steps=args.layer_num,
            dropout=args.dropout,
            use_gce=self.use_gce,
        )

        # ── Output ─────────────────────────────────────────────
        self.norm_out    = RMSNorm(hidden)
        self.output_proj = Linear(hidden, dataset.num_classes)

        # Adjacency cache (avoids recomputing every forward)
        self._cached_adj = None
        self._cached_key = None

        self.reset_parameters()

    def reset_parameters(self):
        self.input_proj.reset_parameters()
        self.output_proj.reset_parameters()
        self.scan.reset_parameters()
        if self.ctx_encoder is not None:
            self.ctx_encoder.reset_parameters()

    def _get_norm_adj(self, edge_index: Tensor,
                      num_nodes: int) -> SparseTensor:
        key = (edge_index.data_ptr(), num_nodes)
        if self._cached_key != key:
            self._cached_adj = build_norm_adj(
                edge_index, num_nodes, edge_index.device)
            self._cached_key = key
        return self._cached_adj

    def forward(self, data):
        x = data.x
        edge_index = (data.adj_t if self.dataset in ['arxiv']
                      else data.edge_index)
        N = x.size(0)

        norm_adj = self._get_norm_adj(edge_index, N)

        # ── Input projection ───────────────────────────────────
        h = F.dropout(x, p=self.dropout, training=self.training)
        h = F.relu(self.input_proj(h))
        h_normed = self.norm_in(h)

        # ── Graph Context Encoder ──────────────────────────────
        if self.use_gce and self.ctx_encoder is not None:
            gamma, delta_g, A_g = self.ctx_encoder(
                h_normed, edge_index, N)
        else:
            gamma = delta_g = A_g = None   # ablation path

        # ── Topology-conditioned scan + depth-fused diffusion ──
        scan_out = self.scan(h_normed, norm_adj, gamma, delta_g, A_g)

        # ── Residual + norm + dropout ──────────────────────────
        h = h + scan_out
        h = self.norm_out(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        return F.log_softmax(self.output_proj(h), dim=-1)