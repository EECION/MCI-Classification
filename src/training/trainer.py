"""Single-run trainer (paper Section 2.4 / S5).

Multiclass cross-entropy (S18) + AdamW (S19); warm-up + cosine schedule;
early stopping on validation loss (patience 15); best-val-loss checkpoint;
automatic mixed precision.
"""
import copy
import time

import numpy as np
import torch
import torch.nn as nn

from src.training.scheduler import build_warmup_cosine


class Trainer:
    def __init__(self, model, device, training_cfg, logger=None, use_amp=None):
        self.model = model.to(device)
        self.device = device
        self.logger = logger
        self.lr = float(training_cfg.get("lr", 3e-4))
        self.weight_decay = float(training_cfg.get("weight_decay", 1e-4))
        self.epochs = int(training_cfg.get("epochs", 100))
        self.warmup_epochs = int(training_cfg.get("warmup_epochs", 5))
        self.min_lr = float(training_cfg.get("min_lr", 1e-6))
        self.patience = int(training_cfg.get("early_stopping_patience", 15))
        if use_amp is None:
            use_amp = bool(training_cfg.get("amp", True))
        self.use_amp = use_amp and torch.cuda.is_available()
        if self.use_amp:
            self.scaler = torch.amp.GradScaler("cuda")
        else:
            self.scaler = None

    def fit(self, train_loader, val_loader):
        """Train; return (best_state_dict on CPU, epoch history)."""
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr,
                                      weight_decay=self.weight_decay)
        scheduler = build_warmup_cosine(optimizer, self.warmup_epochs,
                                        self.epochs, self.min_lr)
        best_val, best_state, wait, history = float("inf"), None, 0, []

        for epoch in range(self.epochs):
            t0 = time.time()
            train_loss, train_acc = self._train_epoch(train_loader, criterion, optimizer)
            val_loss, val_acc, _, _ = self.evaluate(val_loader)
            lr_now = optimizer.param_groups[0]["lr"]
            scheduler.step()
            history.append({"epoch": epoch, "train_loss": train_loss,
                            "train_acc": train_acc, "val_loss": val_loss,
                            "val_acc": val_acc, "lr": lr_now,
                            "seconds": round(time.time() - t0, 2)})
            improved = val_loss < best_val - 1e-6
            if improved:
                best_val = val_loss
                best_state = copy.deepcopy(
                    {k: v.detach().cpu() for k, v in self.model.state_dict().items()})
                wait = 0
            else:
                wait += 1
            if self.logger:
                self.logger.info(
                    f"epoch {epoch:3d} | lr {lr_now:.2e} | train {train_loss:.4f}/{train_acc:.3f}"
                    f" | val {val_loss:.4f}/{val_acc:.3f} | {'*' if improved else ''}"
                    f"{f' | early stop ({wait}/{self.patience})' if not improved else ''}")
            if wait >= self.patience:
                if self.logger:
                    self.logger.info(f"early stopping at epoch {epoch} (best val loss {best_val:.4f})")
                break

        if best_state is None:  # only reachable with epochs == 0
            best_state = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
        return best_state, history

    def _train_epoch(self, loader, criterion, optimizer):
        self.model.train()
        total, correct, losses = 0, 0, []
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            optimizer.zero_grad()
            with torch.autocast(self.device.split(":")[0], enabled=self.use_amp):
                logits = self.model(x)
                loss = criterion(logits, y)
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss.backward()
                optimizer.step()
            losses.append(loss.item() * len(y))
            correct += (logits.argmax(1) == y).sum().item()
            total += len(y)
        return float(np.sum(losses) / total), correct / total

    @torch.no_grad()
    def evaluate(self, loader):
        """Return (loss, acc, probs (N,3), labels (N,))."""
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        losses, probs, labels = [], [], []
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            logits = self.model(x)
            losses.append(criterion(logits, y).item() * len(y))
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
            labels.append(y.cpu().numpy())
        probs = np.concatenate(probs) if probs else np.zeros((0, 3))
        labels = np.concatenate(labels) if labels else np.zeros(0, dtype=int)
        loss = float(np.sum(losses) / max(1, len(labels)))
        acc = float((probs.argmax(1) == labels).mean()) if len(labels) else 0.0
        return loss, acc, probs, labels
