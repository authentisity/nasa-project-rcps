"""
Train WingLSTM on the preprocessed wing time-series dataset
(data/processed/wing_dataset.pt, built by src/datasets/preprocess.py).

Usage:
    python src/training/train.py
    python src/training/train.py --epochs 300 --batch-size 64 --lr 5e-4
"""

import argparse
import sys
from pathlib import Path

import torch
from torch import optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "models"))

from forward_net import WingLSTM  # noqa: E402
from loop import train_one_epoch, evaluate  # noqa: E402


def build_dataloaders(data_path: Path, batch_size: int, val_frac: float, seed: int):
    dataset = torch.load(data_path, weights_only=False)

    static = dataset["static"]
    t = dataset["t"]
    lengths = dataset["lengths"]

    # Normalize targets; padded steps end up non-zero here but are excluded
    # by the length mask in masked_mse_loss, so it doesn't matter.
    targets = (dataset["targets"] - dataset["target_mean"]) / dataset["target_std"]

    full = TensorDataset(static, t, targets, lengths)

    n_val = max(1, int(round(val_frac * len(full))))
    n_train = len(full) - n_val
    train_set, val_set = random_split(
        full, [n_train, n_val], generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "data/processed/wing_dataset.pt")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "checkpoints/wing_lstm.pt")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, dataset = build_dataloaders(
        args.data, args.batch_size, args.val_frac, args.seed
    )

    static_size = dataset["static"].shape[1]
    target_size = dataset["targets"].shape[-1]

    model = WingLSTM(
        static_size=static_size,
        target_size=target_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    pbar = tqdm(range(1, args.epochs + 1), desc="training")
    for epoch in pbar:
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)
        pbar.set_postfix(train_loss=f"{train_loss:.4f}", val_loss=f"{val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "static_size": static_size,
                    "target_size": target_size,
                    "hidden_size": args.hidden_size,
                    "num_layers": args.num_layers,
                    "target_mean": dataset["target_mean"],
                    "target_std": dataset["target_std"],
                    "static_columns": dataset["static_columns"],
                    "target_columns": dataset["target_columns"],
                    "epoch": epoch,
                    "val_loss": best_val_loss,
                },
                args.output,
            )

    print(f"Best val loss: {best_val_loss:.4f} -> saved to {args.output}")


if __name__ == "__main__":
    main()
