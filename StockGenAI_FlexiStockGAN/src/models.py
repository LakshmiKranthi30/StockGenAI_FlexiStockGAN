from __future__ import annotations
import torch
import torch.nn as nn


class GRUGenerator(nn.Module):
    def __init__(self, input_dim, horizon, noise_dim=16, hidden_dim=64, dropout=0.2, large_model=False):
        super().__init__()
        if large_model:
            h1, h2, h3 = 1024, 512, 256
        else:
            h1, h2, h3 = hidden_dim, hidden_dim, hidden_dim
        self.noise_dim = noise_dim
        self.gru = nn.GRU(input_size=input_dim, hidden_size=h1, num_layers=1, batch_first=True)
        self.proj1 = nn.Linear(h1 + noise_dim, h2)
        self.proj2 = nn.Linear(h2, h3)
        self.out = nn.Linear(h3, horizon)
        self.act = nn.LeakyReLU(0.2)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, z=None):
        b = x.size(0)
        if z is None:
            z = torch.randn(b, self.noise_dim, device=x.device)
        _, h = self.gru(x)
        h = h[-1]
        h = torch.cat([h, z], dim=1)
        h = self.drop(self.act(self.proj1(h)))
        h = self.drop(self.act(self.proj2(h)))
        return self.out(h)


class CNNCritic(nn.Module):
    def __init__(self, horizon):
        super().__init__()
        # Linear output; no sigmoid because WGAN-GP uses raw critic scores.
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(128 * horizon, 220),
            nn.LeakyReLU(0.2),
            nn.Linear(220, 220),
            nn.LeakyReLU(0.2),
            nn.Linear(220, 1),
        )

    def forward(self, y):
        if y.dim() == 2:
            y = y.unsqueeze(1)
        return self.net(y).view(-1)


class SequenceRegressor(nn.Module):
    def __init__(self, input_dim, horizon, kind="gru", hidden_dim=64, dropout=0.2):
        super().__init__()
        self.kind = kind.lower()
        if self.kind == "lstm":
            self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=2, dropout=dropout)
        else:
            self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True, num_layers=2, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, horizon)
        )

    def forward(self, x):
        _, h = self.rnn(x)
        if self.kind == "lstm":
            h = h[0]
        return self.head(h[-1])
