"""
Latin Hypercube Sampling for collocation points.
Matches the paper's Section 4: "2048 collocation points per iteration by Latin Hypercube Sampling."
"""
import numpy as np
import torch
from scipy.stats import qmc


def lhs_sample(n: int, bounds: list[tuple[float, float]]) -> np.ndarray:
    """
    Draw n points over a rectangular domain via Latin Hypercube Sampling.
    bounds: list of (lo, hi) per dimension.
    Returns array of shape (n, d).
    """
    d = len(bounds)
    sampler = qmc.LatinHypercube(d=d, seed=None)
    raw = sampler.random(n=n)
    lo = np.array([b[0] for b in bounds], dtype=np.float32)
    hi = np.array([b[1] for b in bounds], dtype=np.float32)
    return qmc.scale(raw, lo, hi).astype(np.float32)


def to_tensor(arr: np.ndarray, device: str = "cpu", requires_grad: bool = False) -> torch.Tensor:
    t = torch.tensor(arr, dtype=torch.float32, device=device)
    if requires_grad:
        t.requires_grad_(True)
    return t
