import torch
import torch.nn as nn

from layers.bernstein import BernsteinLayer


def _linear_interval(bounds, layer):
    """Interval arithmetic through an nn.Linear: bounds (..., in, 2) -> (..., out, 2)."""
    W_pos = layer.weight.clamp(min=0)
    W_neg = layer.weight.clamp(max=0)
    lb, ub = bounds[..., 0], bounds[..., 1]
    out_lb = lb @ W_pos.T + ub @ W_neg.T + layer.bias
    out_ub = ub @ W_pos.T + lb @ W_neg.T + layer.bias
    return torch.stack((out_lb, out_ub), -1)


class BernMLP(nn.Module):
    """Fully connected DeepBern-Net (Khedr & Shoukry, arXiv:2305.13508):
    Linear -> Bernstein -> ... -> Linear, defined on a fixed input box.

    Each Bernstein activation is a polynomial in Bernstein form on the interval
    `input_bounds` of its pre-activation. Those intervals come from propagating
    the input box through the network with interval arithmetic for the linear
    layers and the coefficient range for the Bernstein layers (Algorithm 1 /
    Prop. 1), so they must be refreshed whenever the weights change:
    `update_bounds` runs on every training-mode forward and on `eval()`.
    Queries outside `input_bounds` are outside the model's domain.
    """

    def __init__(self, in_dim, out_dim, hidden=(64, 64), degree=8, input_bounds=None):
        super().__init__()
        if input_bounds is None:
            input_bounds = torch.stack((torch.zeros(in_dim), torch.ones(in_dim)), -1)
        self.register_buffer("input_bounds", input_bounds.clone())

        sizes = [in_dim, *hidden]
        layers = []
        for a, b in zip(sizes[:-1], sizes[1:]):
            layers += [nn.Linear(a, b), BernsteinLayer([b], degree)]
        layers.append(nn.Linear(sizes[-1], out_dim))
        self.net = nn.Sequential(*layers)
        self.update_bounds()

    @torch.no_grad()
    def update_bounds(self):
        """Set each Bernstein layer's input interval from the current weights.
        Returns an enclosure of the output over the whole input box."""
        bounds = self.input_bounds
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                bounds = _linear_interval(bounds, layer)
            else:
                layer.input_bounds = bounds
                bounds = layer.bern_bounds
        return bounds

    def train(self, mode=True):
        super().train(mode)
        if not mode:
            self.update_bounds()
        return self

    def forward(self, x):
        if self.training:
            self.update_bounds()
        return self.net(x)

    @torch.no_grad()
    def output_bounds(self, box):
        """Bern-IBP: sound bounds on the output over each input sub-box.
        box: (B, in_dim, 2) inside `input_bounds` -> (B, out_dim, 2)."""
        if self.training:
            self.update_bounds()
        lo, hi = self.input_bounds[..., 0], self.input_bounds[..., 1]
        if (box[..., 0] < lo).any() or (box[..., 1] > hi).any() or (box[..., 0] > box[..., 1]).any():
            raise ValueError("box must be a valid interval inside the model's input_bounds")
        bounds = box
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                bounds = _linear_interval(bounds, layer)
            else:
                # Linear-layer intervals of a sub-box lie inside the layer's
                # stored input interval, up to floating-point rounding
                bounds = torch.maximum(torch.minimum(bounds, layer.input_bounds[..., 1:]),
                                       layer.input_bounds[..., :1])
                bounds = layer.subinterval_bounds(bounds)
        return bounds
