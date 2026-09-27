"""
Train a steady-state surrogate, the 7 design inputs scaled to [0, 1]^7 ->
steady CL/CD(/Cm), on data/processed/wing_steady.pt (built by
src/datasets/preprocess.py).

  --arch bern   DeepBern-Net (BernMLP), the surrogate used for reachability
  --arch relu   ReLU MLP of the same widths, a baseline

Targets are standardized with the training-split statistics. The weights with
the lowest validation loss are kept; a BernMLP is saved in eval mode, so its
stored Bernstein input intervals belong to the saved weights.

Usage:
    python src/training/train_steady.py
    python src/training/train_steady.py --arch relu --output checkpoints/wing_steady_relu.pt
"""

import argparse
import copy
import sys
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "models"))

from bern_net import BernMLP  # noqa: E402


def build_model(arch, in_dim, out_dim, hidden, degree):
    if arch == "bern":
        return BernMLP(in_dim, out_dim, hidden, degree)
    layers, sizes = [], [in_dim, *hidden]
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), nn.ReLU()]
    layers.append(nn.Linear(sizes[-1], out_dim))
    return nn.Sequential(*layers)


def load_checkpoint(path):
    """-> (model in eval mode, checkpoint dict); model maps unit-box inputs to
    standardized targets (see ckpt["y_mean"], ckpt["y_std"])."""
    ckpt = torch.load(path, weights_only=False, map_location="cpu")
    model = build_model(ckpt["arch"], len(ckpt["input_columns"]), len(ckpt["target_columns"]),
                        ckpt["hidden"], ckpt["degree"]).double()
    model.load_state_dict(ckpt["model_state_dict"])
    return model.eval(), ckpt


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "data/processed/wing_steady.pt")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "checkpoints/wing_steady_bern.pt")
    parser.add_argument("--arch", choices=("bern", "relu"), default="bern")
    parser.add_argument("--hidden", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--degree", type=int, default=8, help="Bernstein polynomial degree (bern only)")
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    data = torch.load(args.data, weights_only=False)
    x, y, split = data["x"], data["y"], data["split"]
    x_train, x_val = x[split == 0], x[split == 1]
    y_mean, y_std = y[split == 0].mean(0), y[split == 0].std(0)
    y_train, y_val = (y[split == 0] - y_mean) / y_std, (y[split == 1] - y_mean) / y_std

    model = build_model(args.arch, x.shape[1], y.shape[1], args.hidden, args.degree).double()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # Full-batch training: the sweep has a few hundred samples
    best = {"val_loss": float("inf")}
    pbar = tqdm(range(1, args.epochs + 1), desc=f"training {args.arch}")
    for epoch in pbar:
        model.train()
        optimizer.zero_grad()
        loss = nn.functional.mse_loss(model(x_train), y_train)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                val_loss = nn.functional.mse_loss(model(x_val), y_val).item()
            pbar.set_postfix(train_loss=f"{loss.item():.2e}", val_loss=f"{val_loss:.2e}")
            if val_loss < best["val_loss"]:
                best = {"val_loss": val_loss, "epoch": epoch,
                        "model_state_dict": copy.deepcopy(model.state_dict())}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        **best,
        "arch": args.arch,
        "hidden": args.hidden,
        "degree": args.degree,
        "input_columns": data["input_columns"],
        "target_columns": data["target_columns"],
        "design_box": data["design_box"],
        "y_mean": y_mean,
        "y_std": y_std,
        "data": str(args.data),
    }, args.output)
    print(f"Best val loss {best['val_loss']:.3e} (standardized MSE) at epoch {best['epoch']} -> {args.output}")


if __name__ == "__main__":
    main()
