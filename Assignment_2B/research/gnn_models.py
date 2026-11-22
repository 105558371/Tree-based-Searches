"""Other spatio-temporal GNNs to compare against Graph WaveNet. Every model takes
x [batch, 2, nodes, 12] (normalised flow + time of day) and returns [batch, horizon, nodes].

- STGCN  (Yu et al., 2018)  fixed distance graph, Chebyshev graph conv + gated temporal conv
- AGCRN  (Bai et al., 2020) learned graph, GRU whose weights are generated per site
- MTGNN  (Wu et al., 2020)  learned directed graph, mix-hop propagation + dilated inception conv
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- STGCN

def scaled_laplacian(adj):
    """L~ = 2L/lambda_max - I, with L the normalised Laplacian of the distance graph."""
    a = adj.numpy().astype(np.float64)
    d = a.sum(1)
    d_inv_sqrt = np.where(d > 0, d ** -0.5, 0)
    lap = np.eye(len(a)) - d_inv_sqrt[:, None] * a * d_inv_sqrt[None, :]
    lam = np.linalg.eigvalsh(lap).max()
    return torch.tensor(2 * lap / lam - np.eye(len(a)), dtype=torch.float32)


class TemporalGLU(nn.Module):
    """Gated temporal conv: (P + residual) * sigmoid(Q); shortens time by kt - 1."""

    def __init__(self, c_in, c_out, kt):
        super().__init__()
        self.conv = nn.Conv2d(c_in, 2 * c_out, (1, kt))
        self.align = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()
        self.kt = kt

    def forward(self, x):
        p, q = self.conv(x).chunk(2, dim=1)
        res = self.align(x)[..., self.kt - 1:]
        return (p + res) * torch.sigmoid(q)


class ChebConv(nn.Module):
    """Chebyshev graph convolution of order K over the scaled Laplacian."""

    def __init__(self, c_in, c_out, lap, k=3):
        super().__init__()
        self.register_buffer("lap", lap)
        self.weight = nn.Parameter(torch.empty(k, c_in, c_out))
        self.bias = nn.Parameter(torch.zeros(c_out))
        self.align = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):                                    # [B, C, N, T]
        terms = [x, torch.einsum("nm,bcmt->bcnt", self.lap, x)]
        for _ in range(2, self.weight.size(0)):
            terms.append(2 * torch.einsum("nm,bcmt->bcnt", self.lap, terms[-1]) - terms[-2])
        h = sum(torch.einsum("bcnt,co->bont", t, w) for t, w in zip(terms, self.weight))
        h = h + self.bias[None, :, None, None]
        return F.relu(h + self.align(x))


class STBlock(nn.Module):
    """Temporal GLU -> Chebyshev graph conv -> temporal GLU -> layer norm (the 'sandwich')."""

    def __init__(self, c_in, c_spatial, c_out, lap, num_nodes, kt=3, dropout=0.3):
        super().__init__()
        self.t1 = TemporalGLU(c_in, c_out, kt)
        self.s = ChebConv(c_out, c_spatial, lap)
        self.t2 = TemporalGLU(c_spatial, c_out, kt)
        self.norm = nn.LayerNorm([num_nodes, c_out])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.t2(self.s(self.t1(x)))
        x = self.norm(x.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        return self.dropout(x)


class STGCN(nn.Module):
    def __init__(self, num_nodes, adj, in_dim=2, horizon=4, in_steps=12, kt=3):
        super().__init__()
        lap = scaled_laplacian(adj)
        self.blocks = nn.Sequential(STBlock(in_dim, 16, 64, lap, num_nodes, kt),
                                    STBlock(64, 16, 64, lap, num_nodes, kt))
        t_left = in_steps - 2 * 2 * (kt - 1)                 # each block uses 2 temporal convs
        self.out_t = TemporalGLU(64, 128, t_left)            # collapse remaining time to 1
        self.out_norm = nn.LayerNorm([num_nodes, 128])
        self.fc = nn.Sequential(nn.Conv2d(128, 128, 1), nn.ReLU(), nn.Conv2d(128, horizon, 1))

    def forward(self, x):
        x = self.out_t(self.blocks(x))
        x = self.out_norm(x.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        return self.fc(x).squeeze(-1)


# ---------------------------------------------------------------- AGCRN

class AVWGCN(nn.Module):
    """Adaptive graph conv where each site gets its own weights, generated from its embedding."""

    def __init__(self, c_in, c_out, emb_dim, cheb_k=2):
        super().__init__()
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(torch.empty(emb_dim, cheb_k, c_in, c_out))
        self.bias_pool = nn.Parameter(torch.zeros(emb_dim, c_out))
        nn.init.xavier_uniform_(self.weights_pool)

    def forward(self, x, emb):                               # x [B, N, C], emb [N, D]
        n = emb.size(0)
        adj = F.softmax(F.relu(emb @ emb.T), dim=1)
        supports = [torch.eye(n, device=x.device), adj]
        for _ in range(2, self.cheb_k):
            supports.append(2 * adj @ supports[-1] - supports[-2])
        supports = torch.stack(supports)                     # [K, N, N]
        weights = torch.einsum("nd,dkio->nkio", emb, self.weights_pool)
        bias = emb @ self.bias_pool
        xg = torch.einsum("knm,bmc->bnkc", supports, x)
        return torch.einsum("bnki,nkio->bno", xg, weights) + bias


class AGCRNCell(nn.Module):
    def __init__(self, c_in, hidden, emb_dim):
        super().__init__()
        self.hidden = hidden
        self.gate = AVWGCN(c_in + hidden, 2 * hidden, emb_dim)
        self.update = AVWGCN(c_in + hidden, hidden, emb_dim)

    def forward(self, x, h, emb):
        zr = torch.sigmoid(self.gate(torch.cat([x, h], -1), emb))
        z, r = zr.chunk(2, dim=-1)
        cand = torch.tanh(self.update(torch.cat([x, z * h], -1), emb))
        return r * h + (1 - r) * cand


class AGCRN(nn.Module):
    def __init__(self, num_nodes, in_dim=2, horizon=4, hidden=64, layers=2, emb_dim=10):
        super().__init__()
        self.emb = nn.Parameter(torch.randn(num_nodes, emb_dim))
        self.cells = nn.ModuleList(AGCRNCell(in_dim if i == 0 else hidden, hidden, emb_dim)
                                   for i in range(layers))
        self.hidden = hidden
        self.out = nn.Conv2d(1, horizon, (1, hidden))

    def forward(self, x):                                    # [B, C, N, T]
        seq = x.permute(0, 3, 2, 1)                          # [B, T, N, C]
        for cell in self.cells:
            h = seq.new_zeros(seq.size(0), seq.size(2), self.hidden)
            outs = []
            for t in range(seq.size(1)):
                h = cell(seq[:, t], h, self.emb)
                outs.append(h)
            seq = torch.stack(outs, 1)
        return self.out(seq[:, -1:]).squeeze(-1)              # [B, horizon, N]


# ---------------------------------------------------------------- MTGNN

class GraphLearner(nn.Module):
    """Learns a directed, sparse graph: A = relu(tanh(a(M1 M2^T - M2 M1^T))), top-k per row."""

    def __init__(self, num_nodes, k, dim=40, alpha=3.0):
        super().__init__()
        self.emb1, self.emb2 = nn.Embedding(num_nodes, dim), nn.Embedding(num_nodes, dim)
        self.lin1, self.lin2 = nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.k, self.alpha = k, alpha
        self.register_buffer("idx", torch.arange(num_nodes))

    def forward(self):
        m1 = torch.tanh(self.alpha * self.lin1(self.emb1(self.idx)))
        m2 = torch.tanh(self.alpha * self.lin2(self.emb2(self.idx)))
        adj = F.relu(torch.tanh(self.alpha * (m1 @ m2.T - m2 @ m1.T)))
        keep = torch.zeros_like(adj).scatter_(1, adj.topk(self.k, dim=1).indices, 1.0)
        return adj * keep


class MixProp(nn.Module):
    """Mix-hop propagation: h_k = b*x + (1-b)*A h_(k-1); concatenate all hops, then 1x1 conv."""

    def __init__(self, c_in, c_out, depth=2, beta=0.05, dropout=0.3):
        super().__init__()
        self.mlp = nn.Conv2d((depth + 1) * c_in, c_out, 1)
        self.depth, self.beta, self.dropout = depth, beta, dropout

    def forward(self, x, adj):
        adj = adj + torch.eye(adj.size(0), device=x.device)
        adj = adj / adj.sum(1, keepdim=True)
        h, out = x, [x]
        for _ in range(self.depth):
            h = self.beta * x + (1 - self.beta) * torch.einsum("ncwl,vw->ncvl", h, adj)
            out.append(h)
        return self.mlp(torch.cat(out, 1))


class DilatedInception(nn.Module):
    """Parallel temporal convs with kernel sizes 2, 3, 6, 7, cropped to equal length."""

    def __init__(self, c_in, c_out, dilation=1, kernels=(2, 3, 6, 7)):
        super().__init__()
        self.convs = nn.ModuleList(nn.Conv2d(c_in, c_out // len(kernels), (1, k), dilation=(1, dilation))
                                   for k in kernels)

    def forward(self, x):
        outs = [conv(x) for conv in self.convs]
        t = outs[-1].size(3)
        return torch.cat([o[..., -t:] for o in outs], 1)


class MTGNN(nn.Module):
    def __init__(self, num_nodes, in_dim=2, horizon=4, in_steps=12, layers=3, conv=32,
                 residual=32, skip=64, end=128, k=20, dropout=0.3):
        super().__init__()
        self.graph = GraphLearner(num_nodes, k=min(k, num_nodes))
        self.receptive_field = layers * (7 - 1) + 1
        self.dropout = dropout
        t = max(in_steps, self.receptive_field)
        self.start = nn.Conv2d(in_dim, residual, 1)
        self.skip0 = nn.Conv2d(in_dim, skip, (1, t))
        self.filters, self.gates, self.skips, self.g1, self.g2, self.norms = (
            nn.ModuleList() for _ in range(6))
        for _ in range(layers):
            t -= 6
            self.filters.append(DilatedInception(residual, conv))
            self.gates.append(DilatedInception(residual, conv))
            self.skips.append(nn.Conv2d(conv, skip, (1, t)))
            self.g1.append(MixProp(conv, residual, dropout=dropout))
            self.g2.append(MixProp(conv, residual, dropout=dropout))
            self.norms.append(nn.LayerNorm([residual, num_nodes, t]))
        self.skip_end = nn.Conv2d(residual, skip, (1, t))
        self.end1, self.end2 = nn.Conv2d(skip, end, 1), nn.Conv2d(end, horizon, 1)

    def forward(self, x):
        if x.size(3) < self.receptive_field:
            x = F.pad(x, (self.receptive_field - x.size(3), 0))
        adj = self.graph()
        skip = self.skip0(F.dropout(x, self.dropout, self.training))
        x = self.start(x)
        for i in range(len(self.filters)):
            res = x
            x = torch.tanh(self.filters[i](x)) * torch.sigmoid(self.gates[i](x))
            x = F.dropout(x, self.dropout, self.training)
            skip = skip + self.skips[i](x)
            x = self.g1[i](x, adj) + self.g2[i](x, adj.T)
            x = self.norms[i](x + res[..., -x.size(3):])
        skip = skip + self.skip_end(x)
        return self.end2(F.relu(self.end1(F.relu(skip)))).squeeze(-1)
