from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    ticker: str = "AAPL"
    start: str = "2010-01-01"
    end: str = "2022-12-31"
    window_size: int = 30
    horizon: int = 5
    batch_size: int = 32
    epochs: int = 80
    patience: int = 15
    g_lr: float = 1e-4
    d_lr: float = 1e-4
    gp_lambda: float = 10.0
    critic_steps: int = 5
    noise_dim: int = 16
    hidden_dim: int = 64
    dropout: float = 0.2
    supervised_weight: float = 10.0
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    seed: int = 42
    device: str = "auto"
    demo: bool = False
    use_indices: bool = True
    use_sentiment: bool = True
    news_csv: str = ""
    ablation: str = "none"  # none, no_sentiment, no_technical, no_indices, generator_only
    output_dir: Path = Path("outputs")

    @property
    def effective_device(self):
        if self.device != "auto":
            return self.device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
