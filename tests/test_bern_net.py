"""Network-level checks of BernMLP: the stored Bernstein input intervals enclose
every reachable pre-activation after training steps (Algorithm 1 of
DeepBern-Nets, arXiv:2305.13508), and Bern-IBP output bounds are sound.

Usage:
    python -m unittest discover tests
"""

import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "models"))

from bern_net import BernMLP  # noqa: E402


def preactivations(model, x):
    """Inputs of every Bernstein layer for the batch x."""
    acts = []
    for layer in model.net:
        if not isinstance(layer, nn.Linear):
            acts.append(x)
        x = layer(x)
    return acts


def sample_box(box, n, generator):
    """n uniform samples from each box: (B, d, 2) -> (B, n, d), corners included."""
    B, d, _ = box.shape
    s = torch.rand(B, n, d, generator=generator, dtype=box.dtype)
    s[:, 0], s[:, 1] = 0, 1
    return box[:, None, :, 0] + s * (box[:, None, :, 1] - box[:, None, :, 0])


class TestBernMLP(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.g = torch.Generator().manual_seed(1)
        self.model = BernMLP(in_dim=7, out_dim=3, hidden=(32, 32), degree=6).double()
        # Take a few optimizer steps so the weights are not at initialization
        opt = torch.optim.Adam(self.model.parameters(), lr=1e-2)
        x = torch.rand(256, 7, dtype=torch.float64)
        y = torch.stack((x.sum(-1), x[:, 0] * x[:, 1], torch.sin(3 * x[:, 2])), -1)
        self.model.train()
        for _ in range(20):
            opt.zero_grad()
            nn.functional.mse_loss(self.model(x), y).backward()
            opt.step()
        self.model.eval()

    def test_eval_refreshes_stale_bounds(self):
        x = torch.rand(4096, 7, dtype=torch.float64, generator=self.g)
        bern_layers = [m for m in self.model.net if not isinstance(m, nn.Linear)]

        def n_outside():
            return sum(((z < layer.input_bounds[:, 0]) | (z > layer.input_bounds[:, 1])).sum().item()
                       for layer, z in zip(bern_layers, preactivations(self.model, x)))

        # A weight update large enough to leave the stored intervals stale
        self.model.train()
        with torch.no_grad():
            self.model.net[0].weight.mul_(3)
        self.assertGreater(n_outside(), 0)
        self.model.eval()
        self.assertEqual(n_outside(), 0)

    def test_output_bounds_sound_on_random_boxes(self):
        lo = torch.rand(64, 7, dtype=torch.float64, generator=self.g)
        width = torch.rand(64, 7, dtype=torch.float64, generator=self.g) * (1 - lo)
        width[:8] = 0                   # point boxes
        width[8:16] = 1 - lo[8:16]      # boxes touching the upper face
        box = torch.stack((lo, lo + width), -1)
        box[16] = self.model.input_bounds
        bounds = self.model.output_bounds(box)
        y = self.model(sample_box(box, 2000, self.g))       # (B, n, out)
        tol = 1e-10
        self.assertTrue((bounds[:, None, :, 0] <= y + tol).all())
        self.assertTrue((y <= bounds[:, None, :, 1] + tol).all())

    def test_full_box_matches_propagated_bounds(self):
        full = self.model.output_bounds(self.model.input_bounds[None])[0]
        torch.testing.assert_close(full, self.model.update_bounds())

    def test_point_box_is_exact(self):
        x = torch.rand(10, 7, dtype=torch.float64, generator=self.g)
        bounds = self.model.output_bounds(torch.stack((x, x), -1))
        torch.testing.assert_close(bounds[..., 0], self.model(x))
        torch.testing.assert_close(bounds[..., 1], self.model(x))

    def test_bounds_shrink_with_box(self):
        c = torch.full((1, 7), 0.5, dtype=torch.float64)
        widths = []
        for r in (0.5, 0.25, 0.1, 0.01):
            b = self.model.output_bounds(torch.stack((c - r, c + r), -1))
            widths.append((b[..., 1] - b[..., 0]).max().item())
        self.assertEqual(widths, sorted(widths, reverse=True))

    def test_rejects_box_outside_domain(self):
        box = torch.tensor([[[0.0, 1.1]] * 7], dtype=torch.float64)
        with self.assertRaises(ValueError):
            self.model.output_bounds(box)


if __name__ == "__main__":
    unittest.main()
