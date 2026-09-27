"""Checks of the multi-fidelity model of src/training/train_steady.py: a
checkpoint trained with --base loads as base + correction, in the correction's
standardized units, with sound Bern-IBP bounds.

Usage:
    python -m unittest discover tests
"""

import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "models"))
sys.path.insert(0, str(REPO_ROOT / "src" / "training"))

from bern_net import BernMLP  # noqa: E402
from test_bern_net import sample_box  # noqa: E402
from train_steady import Corrected, model_from_checkpoint  # noqa: E402

INPUTS = ["AOA", "ar", "tr", "lambda", "gamma", "twist_tip", "magVinf"]


def checkpoint(targets, seed):
    torch.manual_seed(seed)
    model = BernMLP(len(INPUTS), len(targets), (16, 16), 4).double()
    return {"arch": "bern", "hidden": [16, 16], "degree": 4,
            "input_columns": INPUTS, "target_columns": targets,
            "design_box": torch.zeros(len(INPUTS), 2, dtype=torch.float64),
            "y_mean": torch.randn(len(targets), dtype=torch.float64),
            "y_std": torch.rand(len(targets), dtype=torch.float64) + 0.1,
            "model_state_dict": model.state_dict()}


class TestCorrected(unittest.TestCase):
    def setUp(self):
        self.base_ckpt = checkpoint(["CL", "CD"], seed=0)
        self.ckpt = {**checkpoint(["CL", "CD"], seed=1), "base": self.base_ckpt}
        self.model, _ = model_from_checkpoint(self.ckpt)
        self.base, _ = model_from_checkpoint(self.base_ckpt)
        self.correction, _ = model_from_checkpoint({k: v for k, v in self.ckpt.items() if k != "base"})
        self.g = torch.Generator().manual_seed(2)

    def test_prediction_is_base_plus_correction(self):
        self.assertIsInstance(self.model, Corrected)
        x = torch.rand(100, len(INPUTS), dtype=torch.float64, generator=self.g)
        with torch.no_grad():
            pred = self.model(x) * self.ckpt["y_std"] + self.ckpt["y_mean"]
            base = self.base(x) * self.base_ckpt["y_std"] + self.base_ckpt["y_mean"]
            corr = self.correction(x) * self.ckpt["y_std"] + self.ckpt["y_mean"]
        torch.testing.assert_close(pred, base + corr)

    def test_output_bounds_sound(self):
        lo = torch.rand(32, len(INPUTS), dtype=torch.float64, generator=self.g)
        width = torch.rand(32, len(INPUTS), dtype=torch.float64, generator=self.g) * (1 - lo)
        box = torch.stack((lo, lo + width), -1)
        box[0] = self.model.input_bounds
        bounds = self.model.output_bounds(box)
        with torch.no_grad():
            y = self.model(sample_box(box, 2000, self.g))
        self.assertTrue((y >= bounds[:, None, :, 0] - 1e-9).all())
        self.assertTrue((y <= bounds[:, None, :, 1] + 1e-9).all())
        # Bounds of a sum: the sum of the bounds of the parts
        base = self.base.output_bounds(box) * self.base_ckpt["y_std"][:, None] + self.base_ckpt["y_mean"][:, None]
        torch.testing.assert_close(bounds - self.correction.output_bounds(box),
                                   base / self.ckpt["y_std"][:, None])


if __name__ == "__main__":
    unittest.main()
