"""
Neural network building blocks for all three PINN models.
All use tanh activations for smooth second-order derivatives (required for HJB).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Fully-connected network with tanh activations and Xavier initialisation."""

    def __init__(self, in_dim: int, out_dim: int, hidden: int = 128, depth: int = 4):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers.append(nn.Linear(hidden, out_dim))
        self.net = nn.Sequential(*layers)
        self._init()

    def _init(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain("tanh"))
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ValueNet(MLP):
    """V(t, x) — raw output (can be positive or negative)."""
    pass


class PolicyNet(nn.Module):
    """
    π(t, x) ∈ (0, 1) — sigmoid output enforces admissibility constraint.

    Final-layer bias initialised to +1.5 so sigmoid(1.5) ≈ 0.82 at init,
    anchoring π* near the Merton solution from the start. This prevents
    convergence to the trivial π*→0 degenerate fixed point where large
    negative backbone values satisfy the HJB with low residual but no trading.
    """

    def __init__(self, in_dim: int, hidden: int = 128, depth: int = 4):
        super().__init__()
        self.backbone = MLP(in_dim, 1, hidden, depth)
        # Shift initial output away from sigmoid(0)=0.5 toward sigmoid(1.5)≈0.82
        with torch.no_grad():
            final = self.backbone.net[-1]  # last nn.Linear
            if isinstance(final, nn.Linear):
                final.bias.fill_(1.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.backbone(x))


class IneqMultiplierNet(nn.Module):
    """μ(t, x) ≥ 0 — softplus output guarantees dual feasibility."""

    def __init__(self, in_dim: int, n_constraints: int, hidden: int = 128, depth: int = 4):
        super().__init__()
        self.backbone = MLP(in_dim, n_constraints, hidden, depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.backbone(x))
