"""LSTM, GRU and 1D-CNN predictors of the next 15-minute flow (PyTorch)."""
import numpy as np
import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, kind, in_size=3, hidden=64):
        super().__init__()
        self.kind = kind
        if kind == "lstm":
            self.core = nn.LSTM(in_size, hidden, batch_first=True)
        elif kind == "gru":
            self.core = nn.GRU(in_size, hidden, batch_first=True)
        elif kind == "cnn":
            self.core = nn.Sequential(nn.Conv1d(in_size, hidden, 3, padding=1), nn.ReLU(),
                                      nn.Conv1d(hidden, hidden, 3, padding=1), nn.ReLU())
        else:
            raise ValueError(f"Unknown model kind: {kind}")
        self.out = nn.Linear(hidden, 1)

    def forward(self, x):                                  # x: (batch, window, features)
        h = self.core(x.transpose(1, 2))[:, :, -1] if self.kind == "cnn" else self.core(x)[0][:, -1]
        return self.out(h).squeeze(-1)


def predict(net, x, batch=4096):
    net.eval()
    x = torch.as_tensor(x, dtype=torch.float32)
    with torch.no_grad():
        return torch.cat([net(x[i:i + batch]) for i in range(0, len(x), batch)]).numpy()


def train_model(kind, x, y, xv, yv, hidden=64, lr=1e-3, epochs=10, seed=0, batch=256):
    """Returns (net with the best validation weights, history of (train_loss, val_loss))."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    net, lossf = Net(kind, x.shape[2], hidden), nn.MSELoss()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    xt, yt = torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32)
    best, best_state, history = float("inf"), None, []
    for _ in range(epochs):
        net.train()
        perm, total = torch.randperm(len(xt)), 0.0
        for i in range(0, len(perm), batch):
            b = perm[i:i + batch]
            opt.zero_grad()
            loss = lossf(net(xt[b]), yt[b])
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        val = float(np.mean((predict(net, xv) - yv) ** 2))
        history.append((total / len(xt), val))
        if val < best:
            best, best_state = val, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    return net, history


def count_params(net):
    return sum(p.numel() for p in net.parameters())
