"""
Run forward inference with a trained WingLSTM checkpoint: given a wing
design/flow condition and a time grid, predict the target (e.g. CL/CD/Cm)
trajectory. Column names/order come from the checkpoint (saved by
src/training/train.py from the processed dataset's static_columns /
target_columns), since preprocess.py infers them from the data rather than
hardcoding them.

Usage:
    # see what static/target columns this checkpoint expects
    python src/inference/forward.py --list-columns

    # supply a design point directly, in static_columns order
    python src/inference/forward.py --static 5.0 8.0 0.4 25.0 3.0 -2.0 30.0 450.0 8.0 0.5 0.2 500000 \
        --t-max 2.0 --n-steps 200

    # or replay a trajectory already in the processed dataset
    python src/inference/forward.py --sample-id 3
"""

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "models"))

from forward_net import WingLSTM  # noqa: E402


def load_model(checkpoint_path: Path, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = WingLSTM(
        static_size=ckpt["static_size"],
        target_size=ckpt["target_size"],
        hidden_size=ckpt["hidden_size"],
        num_layers=ckpt["num_layers"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Older checkpoints predate column-name metadata; fall back to generic names.
    static_columns = ckpt.get("static_columns", [f"static_{i}" for i in range(ckpt["static_size"])])
    target_columns = ckpt.get("target_columns", [f"target_{i}" for i in range(ckpt["target_size"])])

    # Older checkpoints were trained on unnormalized static features.
    model.static_mean = ckpt.get("static_mean", torch.zeros(ckpt["static_size"])).to(device)
    model.static_std = ckpt.get("static_std", torch.ones(ckpt["static_size"])).to(device)

    return model, ckpt["target_mean"].to(device), ckpt["target_std"].to(device), static_columns, target_columns


@torch.no_grad()
def predict(model, static: torch.Tensor, t: torch.Tensor, target_mean, target_std) -> torch.Tensor:
    """static: (B, static_size) raw values, t: (B, max_len) -> (B, max_len, target_size) in physical units."""
    pred = model((static - model.static_mean) / model.static_std, t)
    return pred * target_std + target_mean


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "checkpoints/wing_lstm.pt")
    parser.add_argument(
        "--list-columns", action="store_true",
        help="Print this checkpoint's expected static/target column names and exit",
    )
    parser.add_argument(
        "--static", type=float, nargs="+", metavar="VALUE",
        help="Design/flow condition values, in static_columns order (see --list-columns)",
    )
    parser.add_argument("--sample-id", type=int, help="Pull static+t from the processed dataset instead of --static")
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "data/processed/wing_dataset.pt")
    parser.add_argument("--t-max", type=float, default=2.0, help="Only used with --static")
    parser.add_argument("--n-steps", type=int, default=200, help="Only used with --static")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, target_mean, target_std, static_columns, target_columns = load_model(args.checkpoint, device)

    if args.list_columns:
        print("static_columns:", static_columns)
        print("target_columns:", target_columns)
        return

    if args.static is None and args.sample_id is None:
        parser.error("provide either --static <values> or --sample-id (or --list-columns to see the expected order)")
    if args.static is not None and len(args.static) != len(static_columns):
        parser.error(f"expected {len(static_columns)} static values ({', '.join(static_columns)}), got {len(args.static)}")

    if args.sample_id is not None:
        dataset = torch.load(args.data, weights_only=False)
        matches = (dataset["sample_id"] == args.sample_id).nonzero(as_tuple=True)[0]
        if len(matches) == 0:
            parser.error(f"sample_id {args.sample_id} not found in {args.data}")
        idx = matches.item()
        length = dataset["lengths"][idx].item()
        static = dataset["static"][idx : idx + 1].to(device)
        t = dataset["t"][idx : idx + 1, :length].to(device)
    else:
        static = torch.tensor([args.static], dtype=torch.float32, device=device)
        t = torch.linspace(0.0, args.t_max, args.n_steps, device=device).unsqueeze(0)

    pred = predict(model, static, t, target_mean, target_std)

    t_np = t.squeeze(0).cpu().numpy()
    pred_np = pred.squeeze(0).cpu().numpy()
    print(f"{'t':>10}" + "".join(f"{c:>12}" for c in target_columns))
    for row_t, row_pred in zip(t_np, pred_np):
        print(f"{row_t:10.4f}" + "".join(f"{v:12.5f}" for v in row_pred))


if __name__ == "__main__":
    main()
