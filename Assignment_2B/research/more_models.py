"""Non-graph spatio-temporal models to compare with the GNNs. Same interface:
x [batch, 2, nodes, 12] (normalised flow + time of day in [0, 1)) -> [batch, horizon, nodes].

- STID       (Shao et al., 2022) MLP with learned site and time-of-day embeddings, no graph
- STAEformer (Liu et al., 2023)  transformer over time then over sites, with a learned
                                 per-(time step, site) "adaptive embedding"

Day-of-week embeddings from the papers are left out so every model gets the same inputs.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

STEPS_PER_DAY = 96      # 15-minute intervals


def tod_index(x):
    """Time-of-day channel -> integer slot 0..95, shape [batch, time] (same for every site)."""
    return (x[:, 1, 0, :] * STEPS_PER_DAY).round().long().clamp(0, STEPS_PER_DAY - 1)


# ---------------------------------------------------------------- STID

class ResidualMLP(nn.Module):
    def __init__(self, dim, dropout=0.15):
        super().__init__()
        self.fc1, self.fc2 = nn.Conv2d(dim, dim, 1), nn.Conv2d(dim, dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.fc2(self.dropout(F.relu(self.fc1(x))))


class STID(nn.Module):
    def __init__(self, num_nodes, horizon=4, in_steps=12, dim=32, layers=3):
        super().__init__()
        self.series = nn.Conv2d(in_steps, dim, 1)              # embed the 12-step history
        self.node_emb = nn.Parameter(torch.empty(num_nodes, dim))
        self.tod_emb = nn.Parameter(torch.empty(STEPS_PER_DAY, dim))
        nn.init.xavier_uniform_(self.node_emb); nn.init.xavier_uniform_(self.tod_emb)
        hidden = 3 * dim
        self.encoder = nn.Sequential(*[ResidualMLP(hidden) for _ in range(layers)])
        self.head = nn.Conv2d(hidden, horizon, 1)

    def forward(self, x):
        b, _, n, _ = x.shape
        series = self.series(x[:, 0].transpose(1, 2).unsqueeze(-1))          # [B, D, N, 1]
        node = self.node_emb.T[None, :, :, None].expand(b, -1, -1, -1)
        tod = self.tod_emb[tod_index(x)[:, -1]][:, :, None, None].expand(-1, -1, n, -1)
        h = self.encoder(torch.cat([series, node, tod], dim=1))
        return self.head(h).squeeze(-1)                                     # [B, horizon, N]


# ---------------------------------------------------------------- STAEformer

class SelfAttentionLayer(nn.Module):
    """Multi-head self-attention + feed-forward, both with residual and layer norm.
    Attends along dimension `dim` of a [B, T, N, D] tensor (1 = time, 2 = sites)."""

    def __init__(self, d_model, heads=4, ff=256, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, ff), nn.ReLU(), nn.Linear(ff, d_model))
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, dim):
        x = x.transpose(dim, -2)                                 # attend over the second-last axis
        *lead, length, d = x.shape
        q, k, v = self.qkv(x).reshape(*lead, length, 3, self.heads, d // self.heads).unbind(-3)
        q, k, v = (t.transpose(-2, -3) for t in (q, k, v))       # [..., heads, length, d_head]
        a = F.scaled_dot_product_attention(q, k, v).transpose(-2, -3).reshape(*lead, length, d)
        x = self.norm1(x + self.drop(self.proj(a)))
        x = self.norm2(x + self.drop(self.ff(x)))
        return x.transpose(dim, -2)


class STAEformer(nn.Module):
    def __init__(self, num_nodes, horizon=4, in_steps=12, input_dim=24, tod_dim=24,
                 adaptive_dim=80, layers=3, heads=4, ff=256, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(1, input_dim)
        self.tod_emb = nn.Embedding(STEPS_PER_DAY, tod_dim)
        self.adaptive = nn.Parameter(torch.empty(in_steps, num_nodes, adaptive_dim))
        nn.init.xavier_uniform_(self.adaptive)
        d = input_dim + tod_dim + adaptive_dim
        self.temporal = nn.ModuleList(SelfAttentionLayer(d, heads, ff, dropout) for _ in range(layers))
        self.spatial = nn.ModuleList(SelfAttentionLayer(d, heads, ff, dropout) for _ in range(layers))
        self.out = nn.Linear(in_steps * d, horizon)

    def forward(self, x):
        b, _, n, t = x.shape
        flow = self.input_proj(x[:, 0].permute(0, 2, 1).unsqueeze(-1))         # [B, T, N, Din]
        tod = self.tod_emb(tod_index(x))[:, :, None, :].expand(-1, -1, n, -1)
        h = torch.cat([flow, tod, self.adaptive.expand(b, -1, -1, -1)], dim=-1)  # [B, T, N, D]
        for layer in self.temporal:
            h = layer(h, dim=1)
        for layer in self.spatial:
            h = layer(h, dim=2)
        h = h.permute(0, 2, 1, 3).reshape(b, n, -1)                             # [B, N, T*D]
        return self.out(h).transpose(1, 2)                                       # [B, horizon, N]
