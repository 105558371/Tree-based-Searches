import torch.nn as nn

class RNNForecaster(nn.Module):
    """LSTM/GRU shared across sites (sites are folded into the batch). x [B, C, N, T] -> [B, horizon, N]."""
    def __init__(self, kind, in_dim=2, horizon=4, hidden=64, layers=1, dropout=0.0):
        super().__init__()
        rnn = nn.LSTM if kind == "lstm" else nn.GRU
        self.rnn = rnn(in_dim, hidden, num_layers=layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.fc = nn.Linear(hidden, horizon)

    def forward(self, x):
        B, C, N, T = x.shape
        out, _ = self.rnn(x.permute(0, 2, 3, 1).reshape(B * N, T, C))
        return self.fc(out[:, -1]).view(B, N, -1).permute(0, 2, 1)
