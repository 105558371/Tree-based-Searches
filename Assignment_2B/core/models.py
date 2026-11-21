"""Model definitions shared by tune.py and the per-model training scripts."""
import torch
import torch.nn as nn

# How each model wants its input (see trainer.to_input)
LAYOUT = {"lstm": "site", "gru": "site",                # one site at a time (no spatial info)
        #   "lstm_multi": "multi", "gru_multi": "multi",  # all sites at once (implicit spatial info)
          "gnn": "graph"}                               # all sites + road-distance graph


class RNN(nn.Module):
    """LSTM/GRU. Per-site: in_dim=out_dim=1. Multivariate: in_dim=out_dim=n_sites."""

    def __init__(self, kind, in_dim, hidden, layers, out_dim, dropout=0.1):
        super().__init__()
        cell = nn.LSTM if kind == "lstm" else nn.GRU
        self.rnn = cell(in_dim, hidden, layers, batch_first=True,
                        dropout=dropout if layers > 1 else 0.0)
        self.fc = nn.Linear(hidden, out_dim)

    def forward(self, x):                      # x: (batch, lookback, in_dim)
        out, _ = self.rnn(x)
        return self.fc(out[:, -1])


class GCN(nn.Module):
    """Graph convolutions over the SCATS sites. Node features = each site's last L counts."""

    def __init__(self, adjacency, lookback, hidden, layers, dropout=0.1):
        super().__init__()
        self.register_buffer("A", torch.as_tensor(adjacency, dtype=torch.float32))
        dims = [lookback] + [hidden] * layers
        self.convs = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x):                      # x: (batch, n_sites, lookback)
        for conv in self.convs:
            x = self.drop(torch.relu(self.A @ conv(x)))
        return self.out(x).squeeze(-1)         # (batch, n_sites)


def build_model(name, ds, hidden, layers, dropout=0.1):
    """name in LAYOUT; ds supplies n_sites, lookback and the adjacency matrix."""
    if name == "gnn":
        return GCN(ds.adjacency, ds.lookback, hidden, layers, dropout)
    kind = name.split("_")[0]
    n = ds.n_sites if name.endswith("_multi") else 1
    return RNN(kind, n, hidden, layers, n, dropout)
