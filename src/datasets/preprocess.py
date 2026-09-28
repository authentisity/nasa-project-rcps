"""
Convert FLOWUnsteady sweep CSVs (data/raw/, long format: one row per
sample+timestep, see data/collection/wing_timeseries_sweep.jl) into cached
tensor datasets (data/processed/):

  wing_steady.pt   steady-state CL/CD/Cm per design, the training data for
                   the DeepBern-Net surrogate (src/training/train_steady.py)
  wing_dataset.pt  padded time series for the WingLSTM baseline
                   (src/training/train.py); static vs. target fields are
                   inferred from the data rather than hardcoded

Both carry the same sample-level train/val/test split, so the baselines are
scored on the same held-out designs.

Sweeps written before the reference-quantity fix (no `mac` column) are
converted: CL/CD are rescaled from the b^2/ar reference area to the planform
area, and Cm is dropped because it was taken about the wrong point.

Usage:
    python src/datasets/preprocess.py
    python src/datasets/preprocess.py --input data/raw/wing_timeseries_data_shard*.csv
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

# Design inputs and the box they are sampled from in wing_timeseries_sweep.jl.
# The steady surrogate takes them scaled to [0, 1]^6; its Bernstein layers are
# only defined on that box, which is therefore also the reachability domain.
# The sweep also samples magVinf, but it is not an input: with a fixed-Re polar
# and an inviscid VPM the coefficients do not depend on it (docs, section 4).
DESIGN_BOX = {
    "AOA": (0.0, 12.0),
    "ar": (3.0, 10.0),
    "tr": (0.3, 1.0),
    "lambda": (0.0, 50.0),
    "gamma": (-5.0, 10.0),
    "twist_tip": (-5.0, 5.0),
}
STEADY_TARGETS = ["CL", "CD", "Cm"]

SPLIT_NAMES = ("train", "val", "test")


def load_sweep(paths) -> pd.DataFrame:
    frames = []
    for path in paths:
        df = pd.read_csv(path)
        frames.append(df if "mac" in df.columns else _upgrade_legacy(df))
    df = pd.concat(frames, ignore_index=True)
    # A resumed sweep retries failed samples: keep a single placeholder row
    # only for samples that never succeeded
    failed = df[CONVERGED_COLUMN] == 0
    retried = df[ID_COLUMN].isin(df.loc[~failed, ID_COLUMN])
    df = df[~(failed & (retried | df.duplicated(ID_COLUMN)))]
    if df.duplicated([ID_COLUMN, STEP_COLUMN]).any():
        raise ValueError("duplicate (sample_id, step) rows: a file listed twice, or sweeps "
                         "with the same sample ids (e.g. legacy and new, or low and high fidelity)?")
    return df


def _upgrade_legacy(df: pd.DataFrame) -> pd.DataFrame:
    """Old schema: CL/CD normalized by q*b^2/ar (S_ref column = b*c_tip) and
    Cm about the root quarter chord from the control points."""
    df = df.copy()
    b = df["c_tip"] * df["ar"]
    S = 0.5 * b * (df["c_root"] + df["c_tip"])
    scale = df["S_ref"] / S
    df["CL"] *= scale
    df["CD"] *= scale
    df["Cm"] = np.nan
    df["S_ref"] = S
    tr = df["tr"]
    df["mac"] = 2 / 3 * df["c_root"] * (1 + tr + tr**2) / (1 + tr)
    # Re_approx was based on the mean geometric chord
    df["Re_mac"] = df.pop("Re_approx") * df["mac"] / (0.5 * (df["c_root"] + df["c_tip"]))
    return df


def assign_split(sample_ids, val_frac: float, test_frac: float, seed: int) -> np.ndarray:
    """0/1/2 = train/val/test, drawn independently per sample_id so a sample
    keeps its split as more samples of a running sweep arrive."""
    u = np.array([np.random.default_rng([seed, int(s)]).random() for s in sample_ids])
    return np.where(u < test_frac, 2, np.where(u < test_frac + val_frac, 1, 0))


def build_steady_dataset(df: pd.DataFrame, tail_frac: float, split: dict) -> dict:
    """Steady-state value of each target = mean over the last `tail_frac` of
    the transient. `drift` is the change from the preceding window of the same
    length, a convergence check."""
    df = df[df[CONVERGED_COLUMN] == 1]
    input_columns = list(DESIGN_BOX)
    # Converted legacy sweeps have no Cm at all; a target missing in only some
    # samples means a run diverged without raising
    missing = df[STEADY_TARGETS].isna()
    partial = [c for c in STEADY_TARGETS if missing[c].any() and not missing[c].all()]
    if partial:
        bad = sorted(df.loc[missing[partial].any(axis=1), ID_COLUMN].unique().tolist())
        raise ValueError(f"NaN {partial} in converged samples {bad}")
    target_columns = [c for c in STEADY_TARGETS if not missing[c].all()]

    x_raw, y, drift, tail_std, sample_id = [], [], [], [], []
    for sid, g in df.groupby(ID_COLUMN, sort=True):
        g = g.sort_values(STEP_COLUMN)
        n_tail = max(1, round(tail_frac * len(g)))
        values = g[target_columns].to_numpy()
        tail, prev = values[-n_tail:], values[-2 * n_tail:-n_tail]
        x_raw.append(g.iloc[0][input_columns].to_numpy(dtype=np.float64))
        y.append(tail.mean(0))
        drift.append(np.abs(tail.mean(0) - prev.mean(0)))
        tail_std.append(tail.std(0))
        sample_id.append(sid)

    box = torch.tensor([DESIGN_BOX[c] for c in input_columns], dtype=torch.float64)
    x_raw = torch.tensor(np.array(x_raw))
    if ((x_raw < box[:, 0]) | (x_raw > box[:, 1])).any():
        raise ValueError("design inputs outside DESIGN_BOX; update it to match the sweep")

    return {
        "x": (x_raw - box[:, 0]) / (box[:, 1] - box[:, 0]),
        "x_raw": x_raw,
        "y": torch.tensor(np.array(y)),
        "drift": torch.tensor(np.array(drift)),
        "tail_std": torch.tensor(np.array(tail_std)),
        "sample_id": torch.tensor(sample_id, dtype=torch.long),
        "split": torch.tensor([split[s] for s in sample_id], dtype=torch.long),
        "input_columns": input_columns,
        "target_columns": target_columns,
        "design_box": box,
        "tail_frac": tail_frac,
    }


def build_dataset(df: pd.DataFrame, split: dict) -> dict:
    df = df[df[CONVERGED_COLUMN] == 1]
    df = df.dropna(axis=1, how="all")  # e.g. Cm of converted legacy sweeps

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

    sample_split = torch.tensor([split[int(s)] for s in sample_id], dtype=torch.long)
    train = sample_split == 0  # normalization statistics from the training split only
    static_mean, static_std = static[train].mean(0), static[train].std(0).clamp_min(1e-8)
    target_mean, target_std = _masked_mean_std(targets[train], lengths[train])

    return {
        "static": static,
        "t": t,
        "targets": targets,
        "lengths": lengths,
        "sample_id": sample_id,
        "split": sample_split,
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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, nargs="+",
                        default=sorted((REPO_ROOT / "data/raw").glob("wing_timeseries_data*.csv")))
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "data/processed/wing_dataset.pt")
    parser.add_argument("--steady-output", type=Path, default=REPO_ROOT / "data/processed/wing_steady.pt")
    parser.add_argument("--tail-frac", type=float, default=0.1,
                        help="Fraction of each transient averaged for the steady-state value")
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not args.input:
        parser.error("no input CSVs found; pass --input")

    df = load_sweep(args.input)
    n_failed = df.loc[df[CONVERGED_COLUMN] == 0, ID_COLUMN].nunique()
    ids = df.loc[df[CONVERGED_COLUMN] == 1, ID_COLUMN].unique()
    split = dict(zip(ids.tolist(), assign_split(ids, args.val_frac, args.test_frac, args.seed).tolist()))

    steady = build_steady_dataset(df, args.tail_frac, split)
    dataset = build_dataset(df, split)
    for path, data in ((args.steady_output, steady), (args.output, dataset)):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, path)

    counts = ", ".join(f"{n} {(steady['split'] == i).sum().item()}" for i, n in enumerate(SPLIT_NAMES))
    print(f"{len(args.input)} file(s): {len(ids)} converged samples, {n_failed} failed  ({counts})")
    print(f"Steady targets {steady['target_columns']} (mean of last {args.tail_frac:.0%} of steps):")
    rel_drift = steady["drift"] / steady["y"].abs().clamp_min(1e-3)
    for j, c in enumerate(steady["target_columns"]):
        print(f"  {c}: range [{steady['y'][:, j].min():.4f}, {steady['y'][:, j].max():.4f}]"
              f"  median drift {steady['drift'][:, j].median():.2e}"
              f"  samples with drift > 1%: {(rel_drift[:, j] > 0.01).sum().item()}")
    print(f"Saved -> {args.steady_output}")
    print(f"Saved time series (max {dataset['t'].shape[1]} steps) -> {args.output}")


if __name__ == "__main__":
    main()
