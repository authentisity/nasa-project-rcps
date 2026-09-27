"""Checks BernsteinLayer against the properties it relies on in Khedr & Shoukry,
"DeepBern-Nets" (arXiv:2305.13508): the activation definition (eq. 1-3),
partition of unity / positivity (Props. 3-4), range enclosure (Prop. 1) and
subdivision (Prop. 2).

Usage:
    python -m unittest discover tests
"""

import math
import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "models" / "layers"))

from bernstein import BernsteinLayer, _de_casteljau_split  # noqa: E402


def make_layer(width=5, degree=6, lo=-1.0, hi=2.0, dtype=torch.float64):
    layer = BernsteinLayer([width], degree).to(dtype)
    layer.bern_coeffs.data = torch.randn(width, degree + 1, dtype=dtype)
    layer.input_bounds = torch.tensor([[lo, hi]] * width, dtype=dtype)
    return layer


def dense_range(layer, lo, hi, n=20001):
    """Brute-force min/max of each neuron's activation on [lo, hi] (per-neuron tensors)."""
    s = torch.linspace(0, 1, n, dtype=lo.dtype)[:, None]
    y = layer(lo + s * (hi - lo))
    return y.min(0).values, y.max(0).values


class TestBernsteinLayer(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_forward_matches_definition(self):
        layer = make_layer()
        l, u = layer.input_bounds[0]
        n = layer.degree
        x = l + (u - l) * torch.rand(64, 5, dtype=torch.float64)
        expected = torch.zeros_like(x)
        for k in range(n + 1):
            b_nk = math.comb(n, k) * (x - l) ** k * (u - x) ** (n - k) / (u - l) ** n
            expected += layer.bern_coeffs[:, k] * b_nk
        torch.testing.assert_close(layer(x), expected)

    def test_basis_partition_of_unity_and_positivity(self):
        layer = make_layer(degree=40, lo=0.0, hi=1.0)
        basis = layer.bern_basis(torch.rand(128, 5, dtype=torch.float64))
        self.assertTrue((basis >= 0).all())
        torch.testing.assert_close(basis.sum(-1), torch.ones(128, 5, dtype=torch.float64))

    def test_range_enclosure(self):
        layer = make_layer(degree=10)
        lo, hi = layer.input_bounds[:, 0], layer.input_bounds[:, 1]
        y_min, y_max = dense_range(layer, lo, hi)
        c_bounds = layer.bern_bounds
        self.assertTrue((c_bounds[:, 0] <= y_min).all() and (y_max <= c_bounds[:, 1]).all())

    def test_split_is_exact_reparametrization(self):
        n = 8
        c = torch.randn(3, n + 1, dtype=torch.float64)
        t = torch.tensor([[0.0], [0.37], [1.0]], dtype=torch.float64)
        left, right = _de_casteljau_split(c, t)

        def bern_eval(coeffs, s):
            s = s[..., None]
            k = torch.arange(n + 1, dtype=torch.float64)
            binom = torch.tensor([math.comb(n, i) for i in range(n + 1)], dtype=torch.float64)
            return (coeffs * binom * s ** k * (1 - s) ** (n - k)).sum(-1)

        for s in torch.linspace(0, 1, 11, dtype=torch.float64):
            # p(t*s) on [0,t] and p(t + (1-t)*s) on [t,1]
            torch.testing.assert_close(bern_eval(left, s), bern_eval(c, t[:, 0] * s))
            torch.testing.assert_close(bern_eval(right, s), bern_eval(c, t[:, 0] + (1 - t[:, 0]) * s))

    def test_subinterval_bounds_sound_and_finite(self):
        layer = make_layer(degree=12)
        lo, hi = -1.0, 2.0
        # Includes degenerate intervals at both ends of [l, u] (previously NaN
        # at the lower end because of an alpha/beta division).
        intervals = [(lo, hi), (lo, lo), (hi, hi), (lo, lo + 1e-3), (hi - 1e-3, hi), (0.3, 0.3)]
        g = torch.Generator().manual_seed(1)
        for _ in range(50):
            a, b = sorted((lo + (hi - lo) * torch.rand(2, generator=g, dtype=torch.float64)).tolist())
            intervals.append((a, b))
        for a, b in intervals:
            box = torch.tensor([[[a, b]] * 5], dtype=torch.float64)
            out = layer.subinterval_bounds(box)[0]
            self.assertTrue(torch.isfinite(out).all(), (a, b))
            y_min, y_max = dense_range(layer, torch.tensor(a, dtype=torch.float64), torch.tensor(b, dtype=torch.float64))
            self.assertTrue((out[:, 0] <= y_min + 1e-12).all(), (a, b))
            self.assertTrue((y_max <= out[:, 1] + 1e-12).all(), (a, b))

    def test_subinterval_bounds_tighten_to_point(self):
        layer = make_layer()
        x = torch.tensor([[0.7] * 5], dtype=torch.float64)
        out = layer.subinterval_bounds(torch.stack((x, x), -1))[0]
        torch.testing.assert_close(out[:, 0], layer(x)[0])
        torch.testing.assert_close(out[:, 1], layer(x)[0])

    def test_float32_forward(self):
        layer = make_layer(dtype=torch.float32)
        self.assertEqual(layer(torch.rand(4, 5)).dtype, torch.float32)
        # A freshly constructed (default-dtype) layer must not promote either
        fresh = BernsteinLayer([5], 40)
        fresh.input_bounds = torch.tensor([[0.0, 1.0]] * 5)
        self.assertEqual(fresh(torch.rand(4, 5)).dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
