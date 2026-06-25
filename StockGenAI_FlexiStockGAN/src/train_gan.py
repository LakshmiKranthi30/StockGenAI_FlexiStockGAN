from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from .models import GRUGenerator, CNNCritic
from .dataset import StockWindowDataset
from .evaluate import rmse


def gradient_penalty(critic, real, fake, device, gp_lambda=10.0):
    b = real.size(0)
    eps = torch.rand(b, 1, device=device)
    interp = eps * real + (1 - eps) * fake
    interp.requires_grad_(True)
    score = critic(interp)
    grads = torch.autograd.grad(
        outputs=score, inputs=interp,
        grad_outputs=torch.ones_like(score),
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grads = grads.view(b, -1)
    gp = gp_lambda * ((grads.norm(2, dim=1) - 1.0) ** 2).mean()
    return gp


def evaluate_generator(generator, X, y, device, noise_dim):
    generator.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), 512):
            xb = torch.tensor(X[i:i+512], dtype=torch.float32, device=device)
            z = torch.randn(xb.size(0), noise_dim, device=device)
            preds.append(generator(xb, z).cpu().numpy())
    pred = np.vstack(preds)
    return rmse(y, pred), pred


def train_flexistockgan(X_train, y_train, X_val, y_val, input_dim, horizon, cfg, out_dir="outputs/models"):
    device = torch.device(cfg.effective_device)
    train_ds = StockWindowDataset(X_train, y_train)
    loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    G = GRUGenerator(input_dim, horizon, cfg.noise_dim, cfg.hidden_dim, cfg.dropout).to(device)
    D = CNNCritic(horizon).to(device)
    opt_g = torch.optim.Adam(G.parameters(), lr=cfg.g_lr, betas=(0.5, 0.9))
    opt_d = torch.optim.Adam(D.parameters(), lr=cfg.d_lr, betas=(0.5, 0.9))
    best_val = float("inf")
    best_state = None
    patience_count = 0
    history = []

    if cfg.ablation == "generator_only":
        for epoch in range(1, cfg.epochs + 1):
            G.train(); losses=[]
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = G(xb)
                loss = F.l1_loss(pred, yb)
                opt_g.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(G.parameters(), 5.0)
                opt_g.step(); losses.append(loss.item())
            val_rmse, _ = evaluate_generator(G, X_val, y_val, device, cfg.noise_dim)
            history.append({"epoch": epoch, "g_loss": float(np.mean(losses)), "d_loss": 0.0, "val_rmse": val_rmse})
            if val_rmse < best_val:
                best_val = val_rmse; best_state = G.state_dict(); patience_count = 0
            else:
                patience_count += 1
            if patience_count >= cfg.patience:
                break
        G.load_state_dict(best_state)
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        torch.save(G.state_dict(), Path(out_dir) / "flexistockgan_generator.pt")
        return G, None, history

    for epoch in range(1, cfg.epochs + 1):
        G.train(); D.train()
        g_losses, d_losses = [], []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            for _ in range(cfg.critic_steps):
                z = torch.randn(xb.size(0), cfg.noise_dim, device=device)
                fake = G(xb, z).detach()
                d_real = D(yb).mean()
                d_fake = D(fake).mean()
                gp = gradient_penalty(D, yb, fake, device, cfg.gp_lambda)
                d_loss = d_fake - d_real + gp
                opt_d.zero_grad(); d_loss.backward(); opt_d.step()
            z = torch.randn(xb.size(0), cfg.noise_dim, device=device)
            fake = G(xb, z)
            adv_loss = -D(fake).mean()
            sup_loss = F.l1_loss(fake, yb)
            g_loss = adv_loss + cfg.supervised_weight * sup_loss
            opt_g.zero_grad(); g_loss.backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), 5.0)
            opt_g.step()
            g_losses.append(g_loss.item()); d_losses.append(d_loss.item())
        val_rmse, _ = evaluate_generator(G, X_val, y_val, device, cfg.noise_dim)
        history.append({"epoch": epoch, "g_loss": float(np.mean(g_losses)), "d_loss": float(np.mean(d_losses)), "val_rmse": val_rmse})
        print(f"Epoch {epoch:03d} | G {history[-1]['g_loss']:.4f} | D {history[-1]['d_loss']:.4f} | Val RMSE {val_rmse:.4f}")
        if val_rmse < best_val:
            best_val = val_rmse
            best_state = {"G": G.state_dict(), "D": D.state_dict()}
            patience_count = 0
        else:
            patience_count += 1
        if patience_count >= cfg.patience:
            print("Early stopping.")
            break
    if best_state:
        G.load_state_dict(best_state["G"]); D.load_state_dict(best_state["D"])
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    torch.save(G.state_dict(), Path(out_dir) / "flexistockgan_generator.pt")
    torch.save(D.state_dict(), Path(out_dir) / "flexistockgan_critic.pt")
    return G, D, history
