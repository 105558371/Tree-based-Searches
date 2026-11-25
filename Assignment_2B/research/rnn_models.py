import torch.nn as nn

class RNNForecaster(nn.Module):
    """LSTM/GRU shared across sites: sites are folded into the batch dimension."""
    def __init__(self, kind, horizon=4, in_dim=2, hidden=64):
        super().__init__()
        self.rnn = (nn.LSTM if kind == "lstm" else nn.GRU)(in_dim, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, horizon)

    def forward(self, x):                                    # x [B, C, N, T]
        B, C, N, T = x.shape
        out, _ = self.rnn(x.permute(0, 2, 3, 1).reshape(B * N, T, C))
        return self.fc(out[:, -1]).view(B, N, -1).permute(0, 2, 1)   # [B, horizon, N]
