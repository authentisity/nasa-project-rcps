"""
Score steady-state surrogates on the held-out designs of wing_steady.pt:
R^2, MAE and max |error| of each target, in physical units.

Checkpoints from src/training/train_steady.py are evaluated directly, on the
targets they predict (e.g. a low-fidelity CL/CD model scored on high-fidelity
CL/CD/Cm data, the baseline of a --base corrected model). A WingLSTM
checkpoint (src/training/train.py) is scored as a steady model by averaging
its predicted transient over the same tail window preprocess.py used for the
steady targets.

For a BernMLP (or corrected BernMLP) checkpoint, the Bern-IBP output bounds
are also checked for soundness: on random sub-boxes of the design box, every
sampled prediction must lie inside the bounds.

Usage:
    python src/inference/eval_steady.py checkpoints/wing_steady_bern.pt checkpoints/wing_steady_relu.pt \
        --lstm checkpoints/wing_lstm.pt
"""

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "models"))
sys.path.insert(0, str(REPO_ROOT / "src" / "training"))
sys.path.insert(0, str(REPO_ROOT / "src" / "inference"))

from bern_net import BernMLP  # noqa: E402
from forward import load_model, predict  # noqa: E402
from train_steady import Corrected, load_checkpoint  # noqa: E402

SPLIT_IDS = {"train": 0, "val": 1, "test": 2}


def metrics(pred, true):
    err = pred - true
    r2 = 1 - err.pow(2).sum(0) / (true - true.mean(0)).pow(2).sum(0)
    return {"R2": r2, "MAE": err.abs().mean(0), "max": err.abs().max(0).values}


def lstm_steady(ckpt_path, ts_path, sample_id, target_columns, tail_frac):
    """Tail mean of the WingLSTM transient for each requested sample."""
    model, t_mean, t_std, _, lstm_targets = load_model(ckpt_path, torch.device("cpu"))
    ts = torch.load(ts_path, weights_only=False)
    rows = [(ts["sample_id"] == s).nonzero().item() for s in sample_id.tolist()]
    pred = predict(model, ts["static"][rows], ts["t"][rows], t_mean, t_std)
    cols = [lstm_targets.index(c) for c in target_columns]
    out = []
    for i, r in enumerate(rows):
        n = ts["lengths"][r].item()
        n_tail = max(1, round(tail_frac * n))
        out.append(pred[i, n - n_tail:n, cols].double().mean(0))
    return torch.stack(out)


@torch.no_grad()
def check_bounds(model, n_boxes, n_samples, seed):
    """Bern-IBP soundness on random sub-boxes of the unit design box (the first
    is the whole box). -> (number of sampled outputs outside their box's
    bounds, bound width on the whole box, mean bound width on the sub-boxes)."""
    g = torch.Generator().manual_seed(seed)
    d = model.input_bounds.shape[0]
    lo = torch.rand(n_boxes, d, generator=g, dtype=torch.float64)
    width = torch.rand(n_boxes, d, generator=g, dtype=torch.float64) * (1 - lo) * 0.5
    box = torch.stack((lo, lo + width), -1)
    box[0] = model.input_bounds
    bounds = model.output_bounds(box)
    s = torch.rand(n_boxes, n_samples, d, generator=g, dtype=torch.float64)
    x = box[:, None, :, 0] + s * (box[:, None, :, 1] - box[:, None, :, 0])
    y = model(x)
    outside = (y < bounds[:, None, :, 0] - 1e-9) | (y > bounds[:, None, :, 1] + 1e-9)
    width = bounds[..., 1] - bounds[..., 0]
    return outside.sum().item(), width[0], width[1:].mean(0)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoints", type=Path, nargs="*")
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "data/processed/wing_steady.pt")
    parser.add_argument("--lstm", type=Path, help="WingLSTM checkpoint to score as a steady model")
    parser.add_argument("--lstm-data", type=Path, default=REPO_ROOT / "data/processed/wing_dataset.pt")
    parser.add_argument("--split", choices=list(SPLIT_IDS), default="test")
    parser.add_argument("--n-boxes", type=int, default=256)
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data = torch.load(args.data, weights_only=False)
    mask = data["split"] == SPLIT_IDS[args.split]
    x, y, target_columns = data["x"][mask], data["y"][mask], data["target_columns"]
    y_range = data["y"].max(0).values - data["y"].min(0).values
    print(f"{args.split} split: {mask.sum().item()} designs, targets {target_columns}")

    results = {}
    for path in args.checkpoints:
        model, ckpt = load_checkpoint(path)
        if not set(ckpt["target_columns"]) <= set(target_columns) \
                or not torch.equal(ckpt["design_box"], data["design_box"]):
            parser.error(f"{path} predicts targets not in the data or has a different design box")
        cols = [target_columns.index(c) for c in ckpt["target_columns"]]
        with torch.no_grad():
            results[path.name] = (model(x) * ckpt["y_std"] + ckpt["y_mean"], cols)
        if isinstance(model, (BernMLP, Corrected)):
            n_out, full, sub = check_bounds(model, args.n_boxes, args.n_samples, args.seed)
            print(f"{path.name}: Bern-IBP on {args.n_boxes} boxes x {args.n_samples} samples:"
                  f" {n_out} outside the bounds")
            for label, width in (("whole design box", full), ("mean over sub-boxes", sub)):
                rel = width * ckpt["y_std"] / y_range[cols]
                print(f"  bound width / data range, {label}: "
                      + ", ".join(f"{c} {w:.2f}" for c, w in zip(ckpt["target_columns"], rel.tolist())))
    if args.lstm:
        pred = lstm_steady(args.lstm, args.lstm_data, data["sample_id"][mask],
                           target_columns, data["tail_frac"])
        results[args.lstm.name] = (pred, list(range(len(target_columns))))

    print(f"\n{'model':<28}{'target':>7}{'R2':>10}{'MAE':>12}{'max err':>12}")
    for name, (pred, cols) in results.items():
        m = metrics(pred, y[:, cols])
        for j, c in enumerate(target_columns[i] for i in cols):
            print(f"{name:<28}{c:>7}{m['R2'][j]:10.5f}{m['MAE'][j]:12.2e}{m['max'][j]:12.2e}")


if __name__ == "__main__":
    main()
