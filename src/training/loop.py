"""Reusable training-loop pieces for WingLSTM: a masked loss (sequences are
zero-padded to max_len, so raw MSE would be skewed by the padded steps) plus
one epoch each of train/eval that `train.py` wires together."""

import torch


def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """MSE over real (non-padded) steps only."""
    max_len = target.size(1)
    mask = torch.arange(max_len, device=lengths.device)[None, :] < lengths[:, None]
    mask = mask.unsqueeze(-1).expand_as(target)
    return (pred[mask] - target[mask]).pow(2).mean()


def train_one_epoch(model, loader, optimizer, device) -> float:
    model.train()
    total_loss, total_count = 0.0, 0
    for static, t, targets, lengths in loader:
        static, t = static.to(device), t.to(device)
        targets, lengths = targets.to(device), lengths.to(device)

        optimizer.zero_grad()
        pred = model(static, t, lengths)
        loss = masked_mse_loss(pred, targets, lengths)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * static.size(0)
        total_count += static.size(0)
    return total_loss / total_count


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    total_loss, total_count = 0.0, 0
    for static, t, targets, lengths in loader:
        static, t = static.to(device), t.to(device)
        targets, lengths = targets.to(device), lengths.to(device)

        pred = model(static, t, lengths)
        loss = masked_mse_loss(pred, targets, lengths)

        total_loss += loss.item() * static.size(0)
        total_count += static.size(0)
    return total_loss / total_count
