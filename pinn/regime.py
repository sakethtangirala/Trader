"""
Regime-Switching PINN — CRRA ansatz (validated V2 architecture).

    V^i(t, w) = w^(1-γ)/(1-γ)  ·  (1 + (T-t) · ψ^i(t, w))

Coupled HJB loss via generator Q̂. No BC loss term.
Achieves 384× lower final loss than the separate-BC-loss formulation.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .networks import MLP, PolicyNet, IneqMultiplierNet
from .sampler import lhs_sample, to_tensor


@dataclass
class RegimeParams:
    mu: tuple[float, float] = (0.185, 0.401)
    sigma: tuple[float, float] = (0.223, 0.362)
    q: tuple[float, float] = (4.45, 1.69)
    r: float = 0.05
    gamma: float = 3.0
    beta: float = 0.0
    T: float = 0.5


class RegimePINN(nn.Module):
    """Two coupled PINNs (one per regime) with CRRA ansatz, trained jointly."""

    def __init__(self, params: RegimeParams, hidden: int = 128, depth: int = 4,
                 beta_damp: float = 0.1):
        super().__init__()
        self.p = params
        self.beta_damp = beta_damp
        self.psi_nets    = nn.ModuleList([MLP(2, 1, hidden, depth) for _ in range(2)])
        self.policy_nets = nn.ModuleList([PolicyNet(2, hidden, depth) for _ in range(2)])
        self.mu_nets     = nn.ModuleList([IneqMultiplierNet(2, 2, hidden, depth)
                                          for _ in range(2)])

    def _enc(self, t, w):
        return torch.cat([t / self.p.T, torch.log(w.clamp(min=1e-6))], dim=-1)

    def _base(self, w):
        return w ** (1.0 - self.p.gamma) / (1.0 - self.p.gamma)

    def _V(self, t, w, i: int):
        x     = self._enc(t, w)
        log_w = torch.log(w.clamp(min=1e-6))
        psi   = self.psi_nets[i](x) / (1.0 + self.beta_damp * log_w**2)
        return self._base(w) * (1.0 + (self.p.T - t) * psi)

    def losses(self, n: int = 2048, device: str = "cpu") -> dict[str, torch.Tensor]:
        p   = self.p
        pts = lhs_sample(n, [(0.0, p.T), (0.1, 10.0)])
        t   = to_tensor(pts[:, 0:1], device, requires_grad=True)
        w   = to_tensor(pts[:, 1:2], device, requires_grad=True)

        x      = self._enc(t, w)
        V_both = [self._V(t, w, i) for i in range(2)]
        agg    = {k: torch.tensor(0.0, device=device)
                  for k in ["hjb", "stat", "comp", "stab", "lb"]}

        for i in range(2):
            j = 1 - i
            mu_i, sigma_i, q_ij = p.mu[i], p.sigma[i], p.q[i]

            V_i      = V_both[i]
            V_j      = V_both[j]
            pi_i     = self.policy_nets[i](x)
            kkt      = self.mu_nets[i](x)
            mu_lo, mu_up = kkt[:, 0:1], kkt[:, 1:2]

            ones = torch.ones_like(V_i)
            V_t  = torch.autograd.grad(V_i, t, grad_outputs=ones, create_graph=True)[0]
            V_w  = torch.autograd.grad(V_i, w, grad_outputs=ones, create_graph=True)[0]
            V_ww = torch.autograd.grad(V_w, w, grad_outputs=torch.ones_like(V_w),
                                       create_graph=True)[0]

            R_hjb  = (V_t - p.beta * V_i
                      + (p.r * w + pi_i * (mu_i - p.r) * w) * V_w
                      + 0.5 * sigma_i**2 * pi_i**2 * w**2 * V_ww
                      + q_ij * (V_j - V_i))
            dH_dpi = (mu_i - p.r) * w * V_w + sigma_i**2 * pi_i * w**2 * V_ww
            R_stat = dH_dpi - mu_lo + mu_up
            R_comp = pi_i * mu_lo + (1.0 - pi_i) * mu_up
            R_stab = torch.clamp(V_ww + 1e-4, min=0.0)

            # Lower-bound penalty per regime state
            merton_floor = (mu_i - p.r) / (p.gamma * sigma_i**2) * 0.3
            R_lb = torch.relu(torch.full_like(pi_i, float(merton_floor)) - pi_i)

            agg["hjb"]  = agg["hjb"]  + (R_hjb ** 2).mean()
            agg["stat"] = agg["stat"] + (R_stat ** 2).mean()
            agg["comp"] = agg["comp"] + R_comp.abs().mean()
            agg["stab"] = agg["stab"] + (R_stab ** 2).mean()
            agg["lb"]   = agg["lb"] + (R_lb ** 2).mean()

        return agg

    @torch.no_grad()
    def query(self, t_val: float, w_val: float, regime_idx: int) -> float:
        self.eval()
        t = torch.tensor([[t_val]], dtype=torch.float32)
        w = torch.tensor([[w_val]], dtype=torch.float32)
        return float(self.policy_nets[regime_idx](self._enc(t, w)).item())

    def merton_ratio(self, regime_idx: int) -> float:
        p = self.p
        return float(np.clip(
            (p.mu[regime_idx] - p.r) / (p.gamma * p.sigma[regime_idx]**2),
            0.0, 1.0,
        ))

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        d = asdict(self.p)
        torch.save({"state_dict": self.state_dict(),
                    "params": d, "beta_damp": self.beta_damp}, path)

    @classmethod
    def load(cls, path: str) -> RegimePINN:
        ckpt   = torch.load(path, weights_only=False, map_location="cpu")
        d      = ckpt["params"]
        params = RegimeParams(
            mu=tuple(d["mu"]), sigma=tuple(d["sigma"]), q=tuple(d["q"]),
            r=d["r"], gamma=d["gamma"], beta=d["beta"], T=d["T"],
        )
        model = cls(params, beta_damp=ckpt.get("beta_damp", 0.1))
        model.load_state_dict(ckpt["state_dict"])
        return model.eval()
