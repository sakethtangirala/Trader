"""
Train all three PINNs using the CRRA ansatz (structurally enforced terminal condition).

    V(t, w) = w^(1-γ)/(1-γ) · (1 + (T-t) · ψ(t, w))

No BC loss term — terminal condition is analytically guaranteed.
Saves to models/gbm_pinn.pt, heston_pinn.pt, regime_pinn.pt.

Run once before starting main.py:
    python train_pinns.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
from loguru import logger

from pinn.gbm    import GBMPINN,  GBMParams
from pinn.heston import HestonPINN, HestonParams
from pinn.regime import RegimePINN, RegimeParams
from settings import (
    CRRA_GAMMA, RISK_FREE_RATE, TRADING_HORIZON_YEARS,
    PINN_HIDDEN_UNITS, PINN_DEPTH, PINN_EPOCHS, PINN_BATCH_SIZE, PINN_LR,
    PINN_ALPHA,
    GBM_PINN_PATH, HESTON_PINN_PATH, REGIME_PINN_PATH,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _train(model: torch.nn.Module, tag: str, save_path: str) -> torch.nn.Module:
    model = model.to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=PINN_LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=PINN_EPOCHS)
    a     = PINN_ALPHA

    logger.info(f"[{tag}] Starting — {PINN_EPOCHS} epochs on {DEVICE} | CRRA ansatz")

    for epoch in range(PINN_EPOCHS + 1):
        opt.zero_grad()
        L     = model.losses(n=PINN_BATCH_SIZE, device=DEVICE)
        
        # Base loss allocation from shared configs
        total = (a["hjb"]   * L["hjb"]
               + a["stat"] * L["stat"]
               + a["comp"] * L["comp"]
               + a["stab"] * L["stab"])
        
        # Inject the lower-bound equilibrium breaker if provided by the model components
        if "lb" in L:
            total += a.get("lb", 0.5) * L["lb"]

        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        sched.step()

        if epoch % 1000 == 0:
            lb_str = f" lb={L['lb']:.3e}" if "lb" in L else ""
            logger.info(
                f"[{tag}] epoch {epoch:>5d} | "
                f"HJB={L['hjb']:.3e} stat={L['stat']:.3e} "
                f"comp={L['comp']:.3e} stab={L['stab']:.3e}"
                f"{lb_str} | total={total.item():.3e}"
            )

    model.save(save_path)
    logger.info(f"[{tag}] Saved → {save_path}")
    return model.cpu().eval()


def train_gbm() -> GBMPINN:
    # μ updated from paper's historical 0.128 → 0.155 to reflect modern S&P 500 drift
    # (realized 10-year CAGR 2014-2024 ≈ 13-16%; 0.155 is a conservative central estimate).
    # σ updated 0.173 → 0.178 (slightly higher post-2020 realized vol).
    params = GBMParams(mu=0.155, sigma=0.178, r=RISK_FREE_RATE,
                       gamma=CRRA_GAMMA, beta=0.0, T=TRADING_HORIZON_YEARS)
    model  = _train(GBMPINN(params, hidden=PINN_HIDDEN_UNITS, depth=PINN_DEPTH),
                    "GBM", GBM_PINN_PATH)
    logger.info(f"[GBM] Merton ratio = {model.merton_ratio():.4f}")
    return model


def train_heston() -> HestonPINN:
    # μ updated 0.132 → 0.160 (higher drift in stochastic vol model).
    # Vol surface params (κ, θ, σ_Y, ρ) kept from paper calibration — these
    # reflect the shape of variance dynamics, which is more stable than drift.
    params = HestonParams(
        mu=0.160, kappa=6.586, theta=0.038, sigma_y=0.643, rho=-0.718,
        r=RISK_FREE_RATE, gamma=CRRA_GAMMA, beta=0.0,
        T=TRADING_HORIZON_YEARS, y0=0.041,
    )
    return _train(HestonPINN(params, hidden=PINN_HIDDEN_UNITS, depth=PINN_DEPTH),
                  "Heston", HESTON_PINN_PATH)


def train_regime() -> RegimePINN:
    # Risk-on μ raised 0.185 → 0.210 (modern bull regime is stronger than historical avg).
    # Risk-off params kept conservative — crash/bear dynamics are not faster than history.
    params = RegimeParams(
        mu=(0.210, 0.401), sigma=(0.223, 0.362), q=(4.45, 1.69),
        r=RISK_FREE_RATE, gamma=CRRA_GAMMA, beta=0.0, T=TRADING_HORIZON_YEARS,
    )
    model  = _train(RegimePINN(params, hidden=PINN_HIDDEN_UNITS, depth=PINN_DEPTH),
                    "Regime", REGIME_PINN_PATH)
    for i, label in enumerate(["risk-on", "risk-off"]):
        logger.info(f"[Regime] Merton ratio ({label}) = {model.merton_ratio(i):.4f}")
    return model


if __name__ == "__main__":
    logger.info(f"Training PINNs (CRRA ansatz) on {DEVICE}")
    train_gbm()
    train_heston()
    train_regime()
    logger.info("All PINNs trained. Start the agent with: python main.py")