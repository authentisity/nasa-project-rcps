"""
Train a steady-state surrogate, the 7 design inputs scaled to [0, 1]^7 ->
steady CL/CD(/Cm), on data/processed/wing_steady.pt (built by
src/datasets/preprocess.py).

  --arch bern   DeepBern-Net (BernMLP), the surrogate used for reachability
  --arch relu   ReLU MLP of the same widths, a baseline

  --base CKPT   multi-fidelity: learn the correction y - base(x) on top of a
                BernMLP trained on a lower-fidelity sweep of the same design
                box and targets (select them with --targets). The saved
                checkpoint holds both networks; load_checkpoint returns their
                sum as one model. A target the low-fidelity sweep lacks (Cm)
                gets its own model: --targets Cm without --base. Learning it
                inside the correction network was worse: early stopping on
                the small CL/CD correction stops before Cm is fit.

Targets are standardized with the training-split statistics. The weights with
the lowest validation loss are kept; a BernMLP is saved in eval mode, so its
stored Bernstein input intervals belong to the saved weights.

Usage:
    python src/training/train_steady.py
    python src/training/train_steady.py --arch relu --output checkpoints/wing_steady_relu.pt
    python src/training/train_steady.py --data data/processed/wing_steady_hifi.pt --targets CL CD \
        --base checkpoints/wing_steady_bern.pt --output checkpoints/wing_steady_mf.pt
    python src/training/train_steady.py --data data/processed/wing_steady_hifi.pt --targets Cm \
        --output checkpoints/wing_steady_cm.pt
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


class Corrected(nn.Module):
    """A lower-fidelity base surrogate plus a learned correction, in the
    standardized units of the correction (prediction = model(x) * y_std +
    y_mean with the correction checkpoint's statistics). Both are BernMLPs on
    the same input box, so Bern-IBP bounds of the sum are the sums of their
    bounds."""

    def __init__(self, base, base_ckpt, correction, ckpt):
        super().__init__()
        self.base, self.correction = base, correction
        self.register_buffer("base_mean", base_ckpt["y_mean"])
        self.register_buffer("base_std", base_ckpt["y_std"])
        self.register_buffer("y_std", ckpt["y_std"])

    @property
    def input_bounds(self):
        return self.correction.input_bounds

    def forward(self, x):
        base = self.base(x) * self.base_std + self.base_mean
        return self.correction(x) + base / self.y_std

    @torch.no_grad()
    def output_bounds(self, box):
        base = self.base.output_bounds(box) * self.base_std[:, None] + self.base_mean[:, None]
        return self.correction.output_bounds(box) + base / self.y_std[:, None]


def load_checkpoint(path):
    """-> (model in eval mode, checkpoint dict); model maps unit-box inputs to
    standardized targets (see ckpt["y_mean"], ckpt["y_std"])."""
    return model_from_checkpoint(torch.load(path, weights_only=False, map_location="cpu"))


def model_from_checkpoint(ckpt):
    model = build_model(ckpt["arch"], len(ckpt["input_columns"]), len(ckpt["target_columns"]),
                        ckpt["hidden"], ckpt["degree"]).double()
    model.load_state_dict(ckpt["model_state_dict"])
    if "base" in ckpt:
        base, _ = model_from_checkpoint(ckpt["base"])
        model = Corrected(base, ckpt["base"], model, ckpt)
    return model.eval(), ckpt


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "data/processed/wing_steady.pt")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "checkpoints/wing_steady_bern.pt")
    parser.add_argument("--arch", choices=("bern", "relu"), default="bern")
    parser.add_argument("--hidden", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--degree", type=int, default=8, help="Bernstein polynomial degree (bern only)")
    parser.add_argument("--base", type=Path, help="lower-fidelity BernMLP checkpoint to correct (bern only)")
    parser.add_argument("--targets", nargs="+", help="fit only these target columns (default: all)")
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    data = torch.load(args.data, weights_only=False)
    x, split = data["x"], data["split"]
    targets = args.targets or data["target_columns"]
    if not set(targets) <= set(data["target_columns"]):
        parser.error(f"--targets must be among {data['target_columns']}")
    y = data["y"][:, [data["target_columns"].index(c) for c in targets]]
    base_ckpt = None
    if args.base:
        base, base_ckpt = load_checkpoint(args.base)
        if args.arch != "bern" or base_ckpt["arch"] != "bern":
            parser.error("--base needs BernMLP base and correction (Bern-IBP of the sum)")
        if not torch.equal(base_ckpt["design_box"], data["design_box"]) \
                or base_ckpt["target_columns"] != targets:
            parser.error(f"{args.base} has a different design box or targets than {targets}")
        with torch.no_grad():
            y = y - (base(x) * base_ckpt["y_std"] + base_ckpt["y_mean"])
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
        "target_columns": targets,
        "design_box": data["design_box"],
        "y_mean": y_mean,
        "y_std": y_std,
        "data": str(args.data),
        **({"base": base_ckpt} if base_ckpt else {}),
    }, args.output)
    print(f"Best val loss {best['val_loss']:.3e} (standardized MSE) at epoch {best['epoch']} -> {args.output}")


if __name__ == "__main__":
    main()
