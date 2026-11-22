"""Graph WaveNet (Wu et al., 2019, "Graph WaveNet for Deep Spatial-Temporal Graph
Modeling"), following the authors' reference design (github.com/nnzhan/Graph-WaveNet).

Shapes use the paper's convention: x is [batch, channels, nodes, time].
Each layer = gated dilated convolution over time, then graph convolution over sites.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def transition(adj):
    """Row-normalise an adjacency matrix into a random-walk transition matrix."""
    deg = adj.sum(1, keepdim=True).clamp(min=1e-8)
    return adj / deg


class GraphConv(nn.Module):
    """Diffusion graph convolution: mixes each site with its 1..order-hop neighbours
    under every support matrix, then a 1x1 conv combines the results."""

    def __init__(self, c_in, c_out, n_supports, order=2, dropout=0.3):
        super().__init__()
        self.order = order
        self.mlp = nn.Conv2d(c_in * (1 + n_supports * order), c_out, kernel_size=1)
        self.dropout = dropout

    def forward(self, x, supports):
        out = [x]
        for a in supports:
            h = x
            for _ in range(self.order):
                h = torch.einsum("ncvl,vw->ncwl", h, a)
                out.append(h)
        h = self.mlp(torch.cat(out, dim=1))
        return F.dropout(h, self.dropout, training=self.training)


class GraphWaveNet(nn.Module):
    def __init__(self, num_nodes, supports, in_dim=2, horizon=4, residual=32, dilation=32,
                 skip=256, end=512, blocks=4, layers=2, kernel=2, emb_dim=10, dropout=0.3):
        super().__init__()
        # fixed graph supports (forward and backward transition of the distance graph)
        self.register_buffer("fixed", torch.stack(supports))
        # adaptive adjacency: learned node embeddings, A = softmax(relu(E1 @ E2))
        self.e1 = nn.Parameter(torch.randn(num_nodes, emb_dim))
        self.e2 = nn.Parameter(torch.randn(emb_dim, num_nodes))
        n_supports = len(supports) + 1

        self.start = nn.Conv2d(in_dim, residual, kernel_size=1)
        self.filters, self.gates, self.skips, self.gconvs, self.norms = (
            nn.ModuleList() for _ in range(5))
        self.receptive_field = 1
        for _ in range(blocks):
            d = 1
            for _ in range(layers):
                self.filters.append(nn.Conv2d(residual, dilation, (1, kernel), dilation=d))
                self.gates.append(nn.Conv2d(residual, dilation, (1, kernel), dilation=d))
                self.skips.append(nn.Conv2d(dilation, skip, kernel_size=1))
                self.gconvs.append(GraphConv(dilation, residual, n_supports, dropout=dropout))
                self.norms.append(nn.BatchNorm2d(residual))
                self.receptive_field += (kernel - 1) * d
                d *= 2
        self.end1 = nn.Conv2d(skip, end, kernel_size=1)
        self.end2 = nn.Conv2d(end, horizon, kernel_size=1)

    def adaptive_adj(self):
        return F.softmax(F.relu(self.e1 @ self.e2), dim=1)

    def forward(self, x):
        # x: [batch, in_dim, nodes, time]; left-pad time to the receptive field
        if x.size(3) < self.receptive_field:
            x = F.pad(x, (self.receptive_field - x.size(3), 0))
        supports = list(self.fixed) + [self.adaptive_adj()]
        x = self.start(x)
        skip = 0
        for i in range(len(self.filters)):
            res = x
            x = torch.tanh(self.filters[i](res)) * torch.sigmoid(self.gates[i](res))
            s = self.skips[i](x)
            skip = s + (skip[..., -s.size(3):] if torch.is_tensor(skip) else 0)
            x = self.gconvs[i](x, supports)
            x = self.norms[i](x + res[..., -x.size(3):])
        out = self.end2(F.relu(self.end1(F.relu(skip))))
        return out.squeeze(-1)      # [batch, horizon, nodes]
