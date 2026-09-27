"""Checks of src/datasets/preprocess.py: conversion of legacy sweeps to the
planform reference quantities, steady-state extraction, and the split.

Usage:
    python -m unittest discover tests
"""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "datasets"))

from preprocess import DESIGN_BOX, _upgrade_legacy, assign_split, build_steady_dataset, load_sweep  # noqa: E402


def sweep(n_samples=3, n_steps=50):
    """Synthetic long-format sweep in the current schema."""
    rng = np.random.default_rng(0)
    rows = []
    for sid in range(1, n_samples + 1):
        design = {c: rng.uniform(lo, hi) for c, (lo, hi) in DESIGN_BOX.items()}
        for k in range(1, n_steps + 1):
            rows.append({"sample_id": sid, **design, "step": k, "t": 0.01 * k,
                         "CL": sid * (1 - np.exp(-k / 5)), "CD": 0.01 * sid, "Cm": -0.1 * sid,
                         "converged": 1})
    return pd.DataFrame(rows)


class TestPreprocess(unittest.TestCase):
    def test_legacy_rescaled_to_planform(self):
        b, ar, tr = 2.489, 6.0, 0.5
        c_tip = b / ar
        c_root = c_tip / tr
        df = pd.DataFrame([{"ar": ar, "tr": tr, "c_root": c_root, "c_tip": c_tip,
                            "S_ref": b**2 / ar, "CL": 0.5, "CD": 0.02, "Cm": -0.3,
                            "Re_approx": 1e6}])
        new = _upgrade_legacy(df).iloc[0]
        S = b * (c_root + c_tip) / 2
        self.assertAlmostEqual(new["S_ref"], S)
        # Same force, different reference area
        self.assertAlmostEqual(new["CL"] * new["S_ref"], 0.5 * b**2 / ar)
        self.assertAlmostEqual(new["CD"] * new["S_ref"], 0.02 * b**2 / ar)
        self.assertTrue(np.isnan(new["Cm"]))
        # MAC = (2/S) * integral of c(y)^2 over the half span, c linear root -> tip
        y = np.linspace(0, b / 2, 100001)
        c = c_root + (c_tip - c_root) * y / (b / 2)
        self.assertAlmostEqual(new["mac"], 2 / S * np.trapezoid(c**2, y), places=8)
        self.assertAlmostEqual(new["Re_mac"], 1e6 * new["mac"] / ((c_root + c_tip) / 2))

    def test_steady_is_tail_mean(self):
        df = sweep()
        split = dict.fromkeys(df.sample_id.unique().tolist(), 0)
        data = build_steady_dataset(df, tail_frac=0.1, split=split)
        self.assertEqual(data["target_columns"], ["CL", "CD", "Cm"])
        k = np.arange(46, 51)
        for i, sid in enumerate(data["sample_id"].tolist()):
            self.assertAlmostEqual(data["y"][i, 0].item(), sid * (1 - np.exp(-k / 5)).mean())
            self.assertAlmostEqual(data["y"][i, 2].item(), -0.1 * sid)
            self.assertLess(data["drift"][i, 0].item(), 1e-3 * sid)
        self.assertTrue(((data["x"] >= 0) & (data["x"] <= 1)).all())

    def test_failed_samples_and_nan_targets_dropped(self):
        df = sweep()
        df.loc[df.sample_id == 2, "converged"] = 0
        df["Cm"] = np.nan
        split = dict.fromkeys([1, 3], 0)
        data = build_steady_dataset(df, tail_frac=0.1, split=split)
        self.assertEqual(data["sample_id"].tolist(), [1, 3])
        self.assertEqual(data["target_columns"], ["CL", "CD"])

    def test_partial_nan_target_raises(self):
        df = sweep()
        df.loc[(df.sample_id == 2) & (df.step == 50), "CL"] = np.nan
        with self.assertRaisesRegex(ValueError, r"\['CL'\].*\[2\]"):
            build_steady_dataset(df, 0.1, dict.fromkeys([1, 2, 3], 0))

    def test_retried_samples_merged(self):
        df = sweep().assign(mac=1.0)  # current schema (a legacy file has no mac)
        placeholder = df[df.step == 1].assign(step=0, CL=np.nan, CD=np.nan, Cm=np.nan, converged=0)
        # Sample 1 failed twice, sample 2 failed once and then succeeded
        first_run = pd.concat([df[df.sample_id == 3], placeholder[placeholder.sample_id <= 2]])
        rerun = pd.concat([placeholder[placeholder.sample_id == 1], df[df.sample_id == 2]])
        with tempfile.TemporaryDirectory() as tmp:
            paths = [Path(tmp) / "a.csv", Path(tmp) / "b.csv"]
            first_run.to_csv(paths[0], index=False)
            rerun.to_csv(paths[1], index=False)
            merged = load_sweep(paths)
        self.assertEqual(merged[merged.converged == 0].sample_id.tolist(), [1])
        self.assertEqual(sorted(merged[merged.converged == 1].sample_id.unique()), [2, 3])
        self.assertEqual(len(merged), 1 + 2 * 50)

    def test_rejects_inputs_outside_design_box(self):
        df = sweep()
        df.loc[df.sample_id == 1, "AOA"] = 20.0
        with self.assertRaises(ValueError):
            build_steady_dataset(df, 0.1, dict.fromkeys([1, 2, 3], 0))

    def test_split_stable_as_sweep_grows(self):
        small = assign_split(np.arange(1, 101), 0.1, 0.1, seed=0)
        large = assign_split(np.arange(1, 501), 0.1, 0.1, seed=0)
        np.testing.assert_array_equal(small, large[:100])
        frac = np.bincount(large, minlength=3) / len(large)
        np.testing.assert_allclose(frac, [0.8, 0.1, 0.1], atol=0.05)


if __name__ == "__main__":
    unittest.main()
