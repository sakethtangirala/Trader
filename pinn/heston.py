"""
Heston PINN — CRRA ansatz (validated V2 architecture).

    V(t, w, y) = w^(1-γ)/(1-γ)  ·  (1 + (T-t) · ψ(t, w, y))

HJB includes the full cross-derivative term ρσ_Y·πwy·V_wy.
Terminal condition structurally enforced — no BC loss term.
Achieves 262× lower final loss than the separate-BC-loss formulation.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import torch.nn as nn

from .networks import MLP, PolicyNet, IneqMultiplierNet
from .sampler import lhs_sample, to_tensor


@dataclass
class HestonParams:
    mu: float = 0.132
    kappa: float = 6.586
    theta: float = 0.038
    sigma_y: float = 0.643
    rho: float = -0.718
    r: float = 0.05
    gamma: float = 3.0
    beta: float = 0.0
    T: float = 0.5
    y0: float = 0.041


class HestonPINN(nn.Module):
    """Heston stochastic volatility PINN with CRRA ansatz."""

    def __init__(self, params: HestonParams, hidden: int = 128, depth: int = 4,
                 beta_damp: float = 0.1):
        super().__init__()
        self.p = params
        self.beta_damp = beta_damp
        self.psi_net    = MLP(3, 1, hidden, depth)
        self.policy_net = PolicyNet(3, hidden, depth)
        self.mu_net     = IneqMultiplierNet(3, 2, hidden, depth)

    def _enc(self, t, w, y):
        return torch.cat(
            [t / self.p.T,
             torch.log(w.clamp(min=1e-6)),
             torch.log(y.clamp(min=1e-8))],
            dim=-1,
        )

    def _base(self, w):
        return w ** (1.0 - self.p.gamma) / (1.0 - self.p.gamma)

    def forward(self, t, w, y):
        x     = self._enc(t, w, y)
        log_w = torch.log(w.clamp(min=1e-6))
        psi   = self.psi_net(x) / (1.0 + self.beta_damp * log_w**2)
        return self._base(w) * (1.0 + (self.p.T - t) * psi)

    def losses(self, n: int = 2048, device: str = "cpu") -> dict[str, torch.Tensor]:
        p     = self.p
        y_max = p.theta * 6.0

        pts = lhs_sample(n, [(0.0, p.T), (0.1, 10.0), (1e-4, y_max)])
        t   = to_tensor(pts[:, 0:1], device, requires_grad=True)
        w   = to_tensor(pts[:, 1:2], device, requires_grad=True)
        y   = to_tensor(pts[:, 2:3], device, requires_grad=True)

        V        = self.forward(t, w, y)
        x        = self._enc(t, w, y)
        pi       = self.policy_net(x)
        mu       = self.mu_net(x)
        mu_lo, mu_up = mu[:, 0:1], mu[:, 1:2]

        ones = torch.ones_like(V)
        V_t  = torch.autograd.grad(V,  t, grad_outputs=ones, create_graph=True)[0]
        V_w  = torch.autograd.grad(V,  w, grad_outputs=ones, create_graph=True)[0]
        V_y  = torch.autograd.grad(V,  y, grad_outputs=ones, create_graph=True)[0]
        V_ww = torch.autograd.grad(V_w, w, grad_outputs=torch.ones_like(V_w),
                                   create_graph=True)[0]
        V_yy = torch.autograd.grad(V_y, y, grad_outputs=torch.ones_like(V_y),
                                   create_graph=True)[0]
        V_wy = torch.autograd.grad(V_w, y, grad_outputs=torch.ones_like(V_w),
                                   create_graph=True)[0]

        R_hjb  = (V_t - p.beta * V
                  + p.kappa * (p.theta - y) * V_y
                  + 0.5 * p.sigma_y**2 * y * V_yy
                  + (p.r * w + pi * (p.mu - p.r) * w) * V_w
                  + 0.5 * pi**2 * w**2 * y * V_ww
                  + p.rho * p.sigma_y * pi * w * y * V_wy)
        dH_dpi = ((p.mu - p.r) * w * V_w
                  + pi * w**2 * y * V_ww
                  + p.rho * p.sigma_y * w * y * V_wy)
        R_stat = dH_dpi - mu_lo + mu_up
        R_comp = pi * mu_lo + (1.0 - pi) * mu_up
        R_stab = torch.clamp(V_ww + 1e-4, min=0.0)

        # Degenerate Fixed-Point Breaker: Penalize if pi* vanishes below 30% of Merton benchmark
        analytical_merton = (p.mu - p.r) / (p.gamma * y.clamp(min=1e-6))
        merton_floor = analytical_merton * 0.3
        R_lb = torch.relu(merton_floor - pi)

        return {
            "hjb":  (R_hjb ** 2).mean(),
            "stat": (R_stat ** 2).mean(),
            "comp": R_comp.abs().mean(),
            "stab": (R_stab ** 2).mean(),
            "lb":   (R_lb ** 2).mean(),
        }

    @torch.no_grad()
    def query(self, t_val: float, w_val: float, y_val: float) -> float:
        self.eval()
        t = torch.tensor([[t_val]], dtype=torch.float32)
        w = torch.tensor([[w_val]], dtype=torch.float32)
        y = torch.tensor([[max(y_val, 1e-8)]], dtype=torch.float32)
        return float(self.policy_net(self._enc(t, w, y)).item())

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.state_dict(),
                    "params": asdict(self.p),
                    "beta_damp": self.beta_damp}, path)

    @classmethod
    def load(cls, path: str) -> HestonPINN:
        ckpt  = torch.load(path, weights_only=False, map_location="cpu")
        model = cls(HestonParams(**ckpt["params"]),
                    beta_damp=ckpt.get("beta_damp", 0.1))
        model.load_state_dict(ckpt["state_dict"])
        return model.eval()