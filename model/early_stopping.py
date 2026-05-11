import numpy as np
import torch
from pathlib import Path

class EarlyStopping:
    """
    Early stops the training if a monitored metric doesn't improve after a given patience.
    """
    def __init__(self, patience: int = 10, mode: str = "max", min_delta: float = 0.0001, save_path: Path = None):
        """
        Args:
            patience (int): How many epochs to wait after last time metric improved.
            mode (str): "max" to maximize a metric (e.g., AUROC), "min" to minimize (e.g., Loss).
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            save_path (Path): Directory where the best model weights will be saved.
        """
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.save_path = save_path
        self.counter = 0
        self.best_score = -np.inf if mode == "max" else np.inf
        self.early_stop = False

    def __call__(self, current_score: float, model: torch.nn.Module):
        if self.mode == "max":
            is_better = current_score > self.best_score + self.min_delta
        else:
            is_better = current_score < self.best_score - self.min_delta

        if is_better:
            self.best_score = current_score
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model: torch.nn.Module):
        """Saves model when metric improves."""
        print(f"Metric improved to {self.best_score:.4f}. Saving best model...")
        if self.save_path is not None:
            model.save_model(self.save_path)