"""
Convert a FLOWUnsteady sweep CSV (data/raw/) into a padded tensor dataset
cached as a .pt file (data/processed/) for RNN training. Works for any sweep
that follows the long-format schema below (wing, Vahana, ...); static vs.
target fields are inferred from the data rather than hardcoded.

Usage:
    python src/datasets/preprocess.py
    python src/datasets/preprocess.py --input path/to.csv --output path/to.pt
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]

ID_COLUMN = "sample_id"
STEP_COLUMN = "step"
TIME_COLUMN = "t"
CONVERGED_COLUMN = "converged"
META_COLUMNS = {ID_COLUMN, STEP_COLUMN, TIME_COLUMN, CONVERGED_COLUMN}


def build_dataset(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    df = df[df[CONVERGED_COLUMN] == 1]

    groups = [g.sort_values(STEP_COLUMN) for _, g in df.groupby(ID_COLUMN, sort=True)]

    # A field constant within every sample is a static/conditioning feature;
    # anything else is part of the per-step target sequence.
    by_sample = df.groupby(ID_COLUMN)
    static_columns = [c for c in df.columns if c not in META_COLUMNS and by_sample[c].nunique().eq(1).all()]
    target_columns = [c for c in df.columns if c not in META_COLUMNS and c not in static_columns]

    lengths = torch.tensor([len(g) for g in groups], dtype=torch.long)
    n_samples = len(groups)
    max_len = int(lengths.max())

    sample_id = torch.zeros(n_samples, dtype=torch.long)
    static = torch.zeros(n_samples, len(static_columns))
    t = torch.zeros(n_samples, max_len)
    targets = torch.zeros(n_samples, max_len, len(target_columns))

    for i, g in enumerate(groups):
        n = len(g)
        sample_id[i] = int(g[ID_COLUMN].iloc[0])
        static[i] = torch.from_numpy(g.iloc[0][static_columns].to_numpy(dtype=np.float32))
        t[i, :n] = torch.from_numpy(g[TIME_COLUMN].to_numpy(dtype=np.float32))
        targets[i, :n] = torch.from_numpy(g[target_columns].to_numpy(dtype=np.float32))

    static_mean, static_std = static.mean(0), static.std(0).clamp_min(1e-8)
    target_mean, target_std = _masked_mean_std(targets, lengths)

    return {
        "static": static,
        "t": t,
        "targets": targets,
        "lengths": lengths,
        "sample_id": sample_id,
        "static_columns": static_columns,
        "target_columns": target_columns,
        "static_mean": static_mean,
        "static_std": static_std,
        "target_mean": target_mean,
        "target_std": target_std,
    }


def _masked_mean_std(targets: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Ignore padded steps so they don't skew the stats.
    mask = torch.arange(targets.size(1))[None, :] < lengths[:, None]
    valid = targets[mask]
    return valid.mean(0), valid.std(0).clamp_min(1e-8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=REPO_ROOT / "data/raw/wing_timeseries_data.csv")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "data/processed/wing_dataset.pt")
    args = parser.parse_args()

    dataset = build_dataset(args.input)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, args.output)

    n_samples, max_len = dataset["static"].shape[0], dataset["t"].shape[1]
    print(f"Saved {n_samples} converged samples (max {max_len} steps) -> {args.output}")


if __name__ == "__main__":
    main()
