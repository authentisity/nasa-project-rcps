# BSD 3-Clause License
#
# Copyright (c) 2023, Haitham Khedr
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import torch
import torch.nn as nn


class BernsteinLayer(nn.Module):
    def __init__(self, in_shape, degree: int):
        super().__init__()
        self.degree = degree
        self.in_shape = in_shape
        _basis_indices_tensor = torch.arange(degree + 1).reshape((-1, degree + 1))
        _deg_tensor = torch.tensor([degree]).reshape(-1, 1)
        nCk_tensor = self.binom(_deg_tensor, _basis_indices_tensor)
        input_bounds = torch.zeros((*in_shape, 2))
        self.register_buffer("input_bounds", input_bounds)
        self.register_buffer("_basis_indices", _basis_indices_tensor)
        self.register_buffer("_deg_tensor", _deg_tensor)
        self.register_buffer("nCk", nCk_tensor)
        bern_coeffs = torch.ones(*in_shape, degree + 1)
        init_std = torch.ones_like(bern_coeffs) / torch.tensor(in_shape).prod()
        self.bern_coeffs = nn.Parameter(
            bern_coeffs * torch.normal(torch.zeros_like(bern_coeffs), init_std)
        )

    @property
    def bern_bounds(self):
        lb, _ = self.bern_coeffs.min(axis=-1, keepdim=True)
        ub, _ = self.bern_coeffs.max(axis=-1, keepdim=True)
        return torch.concat((lb, ub), dim=-1)

    def subinterval_bounds(self, bounds):
        """Compute Bernstein coeffs for interval [alpha,beta]"""
        in_bounds = self.input_bounds.unsqueeze(0)
        bounds = (bounds - in_bounds[..., 0:1]) / (
            in_bounds[..., 1:] - in_bounds[..., 0:1]
        )
        alpha = bounds[..., 0].unsqueeze(-1)
        beta = bounds[..., 1].unsqueeze(-1)
        coeffs = self.bern_coeffs.expand(bounds.shape[0], *self.bern_coeffs.shape)

        # Two applications of the subdivision property (DeepBern-Nets, Prop. 2).
        # Splitting at beta then alpha/beta divides by zero when beta == 0, and
        # splitting at alpha then (beta-alpha)/(1-alpha) when alpha == 1, so take
        # whichever order has the larger denominator (always >= 1/2 here).
        left_first = beta >= 1 - alpha
        one = torch.ones_like(alpha)
        beta_safe = torch.where(left_first, beta, one)
        one_minus_alpha_safe = torch.where(left_first, one, 1 - alpha)

        zero_to_beta, _ = _de_casteljau_split(coeffs, beta_safe)
        _, via_left = _de_casteljau_split(zero_to_beta, alpha / beta_safe)
        _, alpha_to_one = _de_casteljau_split(coeffs, alpha)
        via_right, _ = _de_casteljau_split(alpha_to_one, (beta - alpha) / one_minus_alpha_safe)

        new_coeffs_lb_ub = torch.where(left_first, via_left, via_right)
        lb = new_coeffs_lb_ub.min(axis=-1, keepdim=True)[0]
        ub = new_coeffs_lb_ub.max(axis=-1, keepdim=True)[0]
        return torch.concat((lb, ub), -1)

    def binom(self, n, k):
        # Integer inputs would be promoted to float32, whose lgamma is off by
        # ~1e-6 relative for degree ~40; in float64 nCk is exact to degree ~50
        n, k = n.double(), k.double()
        nCk = torch.lgamma(n + 1) - torch.lgamma(k + 1) - torch.lgamma(n - k + 1)
        nCk = torch.exp(nCk)
        nCk = torch.floor(nCk + 0.5)
        return nCk

    def bern_basis(self, x):
        y = x.unsqueeze(-1)
        basis = (
            self.nCk.to(y.dtype)
            * (y) ** self._basis_indices
            * (1 - y) ** (self._deg_tensor - self._basis_indices)
        )
        return basis
        # return basis / diff

    def forward(self, x):
        x = (x - self.input_bounds[..., 0]) / (
            self.input_bounds[..., 1] - self.input_bounds[..., 0]
        )
        basis = self.bern_basis(x)
        with torch.no_grad():
            basis_shape = torch.tensor(basis.shape)
            basis_sum_to_one = torch.isclose(
                torch.sum(basis, axis=-1).sum(), basis_shape[:-1].prod().to(basis.dtype)
            )
            if not basis_sum_to_one:
                raise Exception(
                    f"Basis doesn't sum to 1, {torch.sum(basis,axis = -1).sum()}"
                )
        out = basis * self.bern_coeffs
        out = out.sum(axis=-1)
        return out


def _de_casteljau_split(coeffs, t):
    """Split Bernstein coefficients (..., n+1) on [0,1] at t (..., 1) into the
    coefficients of the same polynomial on [0,t] and on [t,1]."""
    left, right = [coeffs[..., 0]], [coeffs[..., -1]]
    c = coeffs
    for _ in range(coeffs.shape[-1] - 1):
        c = (1 - t) * c[..., :-1] + t * c[..., 1:]
        left.append(c[..., 0])
        right.append(c[..., -1])
    return torch.stack(left, -1), torch.stack(right[::-1], -1)
