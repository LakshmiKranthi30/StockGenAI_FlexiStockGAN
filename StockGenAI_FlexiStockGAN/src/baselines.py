from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from pathlib import Path
from .models import SequenceRegressor
from .dataset import StockWindowDataset


def train_svr_baseline(X_train, y_train, X_test):
    model = MultiOutputRegressor(SVR(kernel="rbf", C=10.0, epsilon=0.01, gamma="scale"))
    model.fit(X_train.reshape(len(X_train), -1), y_train)
    return model.predict(X_test.reshape(len(X_test), -1))


def train_rnn_baseline(X_train, y_train, X_val, y_val, X_test, input_dim, horizon, kind="gru", hidden_dim=64, epochs=40, batch_size=32, lr=1e-3, device="cpu"):
    device = torch.device(device)
    model = SequenceRegressor(input_dim, horizon, kind=kind, hidden_dim=hidden_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    train_loader = DataLoader(StockWindowDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    best = float("inf"); best_state = None; patience = 8; wait = 0
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = F.mse_loss(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        val_loss = _eval_loss(model, X_val, y_val, device)
        if val_loss < best:
            best = val_loss; best_state = model.state_dict(); wait = 0
        else:
            wait += 1
        if wait >= patience:
            break
    if best_state:
        model.load_state_dict(best_state)
    return _predict(model, X_test, device)


def _eval_loss(model, X, y, device):
    model.eval(); losses=[]
    with torch.no_grad():
        for i in range(0, len(X), 512):
            xb = torch.tensor(X[i:i+512], dtype=torch.float32, device=device)
            yb = torch.tensor(y[i:i+512], dtype=torch.float32, device=device)
            losses.append(F.mse_loss(model(xb), yb).item())
    return float(np.mean(losses))


def _predict(model, X, device):
    model.eval(); preds=[]
    with torch.no_grad():
        for i in range(0, len(X), 512):
            xb = torch.tensor(X[i:i+512], dtype=torch.float32, device=device)
            preds.append(model(xb).cpu().numpy())
    return np.vstack(preds)
