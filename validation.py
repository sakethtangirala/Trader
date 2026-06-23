"""
validation.py — Monte Carlo Permutation Test + Walk-Forward Validation Framework.

Answers the question: "Is the strategy's edge statistically real, or is it
memorising random patterns in the training data?"

Scientific protocol:
  1. Run the strategy on real price history → compute Profit Factor (PF_real).
  2. Shuffle the log-return series 500 times (destroys pattern memory, preserves
     return distribution) → run strategy on each shuffled path → PF_shuffled.
  3. p-value = fraction of shuffled paths where PF_shuffled ≥ PF_real.
  4. Hard gate: if p-value ≥ 0.01, the model is fitting noise. Do not deploy.

Walk-Forward Folds:
  Splits the backtest window into anchored training + out-of-sample test slices.
  Each fold: train on [0 … fold_end], test on [fold_end … fold_end+step].
  Reports per-fold Sharpe, PF, and permutation p-value.

Usage:
    python validation.py                         # test DEFAULT_UNIVERSE (20 tickers)
    python validation.py AAPL MSFT NVDA GOOGL    # test specific tickers
    python validation.py --n 1000               # 1000 permutations (slow but rigorous)
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger

warnings.filterwarnings("ignore")
logger.remove()
logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | {message}", level="INFO")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_UNIVERSE = [
    "AAPL", "MSFT", "JPM", "JNJ", "XOM", "WMT", "CVX", "PG",
    "CSCO", "HD", "MRK", "BAC", "VZ", "PFE", "ABT", "MMM",
    "IBM", "GE", "KO", "UNH",
]

EMA_FAST = 20
EMA_SLOW = 50
RF_DAILY = 0.05 / 252

# ─────────────────────────────────────────────────────────────────────────────
# Core metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_strategy_returns(close: pd.Series,
                              position_signal: pd.Series) -> pd.Series:
    """
    Per-bar strategy returns using close-to-close log returns.

    Returns are shifted forward by 1 bar (the signal at T predicts T+1).
    position_signal: 1 = long, 0 = cash, -1 = short.
    """
    log_ret     = np.log(close / close.shift(1))
    future_ret  = log_ret.shift(-1)                    # predict next bar
    strat_ret   = position_signal * future_ret
    return strat_ret.dropna()


def calculate_profit_factor(strategy_returns: pd.Series) -> float:
    """PF = sum(gains) / sum(|losses|). ∞ if no losses."""
    gains  = strategy_returns[strategy_returns > 0].sum()
    losses = strategy_returns[strategy_returns < 0].abs().sum()
    return float(gains / losses) if losses > 0 else float("inf")


def calculate_sharpe(strategy_returns: pd.Series) -> float:
    exc = strategy_returns - RF_DAILY
    return float(exc.mean() / exc.std() * np.sqrt(252)) if exc.std() > 0 else 0.0


def ema_crossover_signal(close: pd.Series) -> pd.Series:
    """
    Signal: 1 (long) when EMA20 > EMA50, else 0 (cash).
    Mirrors the live strategy's bull-regime entry condition.
    """
    fast = close.ewm(span=EMA_FAST, adjust=False).mean()
    slow = close.ewm(span=EMA_SLOW, adjust=False).mean()
    return (fast > slow).astype(float)


# ─────────────────────────────────────────────────────────────────────────────
# Permutation engine
# ─────────────────────────────────────────────────────────────────────────────

def generate_price_permutation(close: pd.Series, seed: int | None = None) -> pd.Series:
    """
    Reconstruct a synthetic close series from shuffled log returns.

    Preserves the marginal return distribution and overall price scale
    but destroys sequential pattern memory (autocorrelation, trends).
    The resulting path is statistically pure noise with the same volatility.
    """
    rng       = np.random.default_rng(seed)
    log_rets  = np.log(close / close.shift(1)).dropna().values.copy()
    rng.shuffle(log_rets)
    initial   = float(close.iloc[0])
    log_path  = np.concatenate([[0.0], np.cumsum(log_rets)])
    perm_close = pd.Series(initial * np.exp(log_path), index=close.index)
    return perm_close


def run_permutation_test(close: pd.Series,
                          n: int = 500,
                          signal_fn=ema_crossover_signal) -> dict:
    """
    Monte Carlo permutation test on a single price series.

    Returns:
        pvalue      — fraction of permuted paths scoring ≥ real (lower is better)
        pf_real     — profit factor on actual price series
        pf_null_p50 — median permuted profit factor (null distribution centre)
        sharpe_real — Sharpe on real series
        n_permutations — n passed in
    """
    # Real strategy performance
    signal_real  = signal_fn(close)
    ret_real     = compute_strategy_returns(close, signal_real)
    pf_real      = calculate_profit_factor(ret_real)
    sr_real      = calculate_sharpe(ret_real)

    # Null distribution
    pf_null: list[float] = []
    for i in range(n):
        perm_close  = generate_price_permutation(close, seed=i)
        signal_perm = signal_fn(perm_close)
        ret_perm    = compute_strategy_returns(perm_close, signal_perm)
        pf_null.append(calculate_profit_factor(ret_perm))

    null_arr = np.array(pf_null)
    pvalue   = float((null_arr >= pf_real).mean())

    return {
        "pvalue":         pvalue,
        "pf_real":        pf_real,
        "pf_null_p50":    float(np.median(null_arr)),
        "pf_null_p95":    float(np.percentile(null_arr, 95)),
        "sharpe_real":    sr_real,
        "n_permutations": n,
        "pass":           pvalue < 0.01,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Walk-Forward Folds
# ─────────────────────────────────────────────────────────────────────────────

def walk_forward_test(close: pd.Series,
                       n_folds: int = 5,
                       perm_n: int = 100,
                       signal_fn=ema_crossover_signal) -> pd.DataFrame:
    """
    Anchored walk-forward validation.

    Each fold:
      - Training window: [0 … fold_end]   (anchor expands each fold)
      - Test window:     [fold_end … fold_end + step]

    Returns a DataFrame with per-fold metrics and permutation p-values.
    """
    n       = len(close)
    step    = n // (n_folds + 1)
    rows: list[dict] = []

    for fold in range(n_folds):
        train_end  = step * (fold + 1)
        test_end   = min(train_end + step, n)
        if test_end - train_end < EMA_SLOW + 10:
            continue

        train_close = close.iloc[:train_end]
        test_close  = close.iloc[train_end:test_end]

        # Fit signal on TRAINING data (to get EMA state at fold boundary)
        # then evaluate on TEST data
        full_signal = signal_fn(close.iloc[:test_end])
        test_signal = full_signal.iloc[train_end:test_end]
        test_ret    = compute_strategy_returns(test_close, test_signal)

        # Permutation test on TRAINING data only (in-sample edge check)
        perm_result = run_permutation_test(train_close, n=perm_n, signal_fn=signal_fn)

        rows.append({
            "fold":          fold + 1,
            "train_bars":    train_end,
            "test_bars":     test_end - train_end,
            "test_pf":       calculate_profit_factor(test_ret),
            "test_sharpe":   calculate_sharpe(test_ret),
            "train_pvalue":  perm_result["pvalue"],
            "train_pf":      perm_result["pf_real"],
            "gate_pass":     perm_result["pass"],
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def validate_ticker(ticker: str, n: int = 500,
                    start: str = "2020-01-01") -> dict:
    """Full permutation test + walk-forward report for one ticker."""
    df = yf.download(ticker, start=start, progress=False, auto_adjust=True)
    if df.empty or len(df) < 100:
        logger.warning(f"{ticker}: insufficient data")
        return {}

    close  = df["Close"].squeeze().dropna()
    perm   = run_permutation_test(close, n=n)
    wf     = walk_forward_test(close, n_folds=5, perm_n=min(n, 100))

    status = "PASS ✓" if perm["pass"] else "FAIL ✗"
    logger.info(
        f"{ticker:<6} | {status} | p={perm['pvalue']:.4f} | "
        f"PF_real={perm['pf_real']:.2f} | PF_null_p95={perm['pf_null_p95']:.2f} | "
        f"Sharpe={perm['sharpe_real']:.2f}"
    )

    if not wf.empty:
        pass_rate = wf["gate_pass"].mean()
        avg_oos   = wf["test_pf"].mean()
        logger.info(
            f"       Walk-forward: {len(wf)} folds | "
            f"gate_pass_rate={pass_rate:.0%} | avg_OOS_PF={avg_oos:.2f}"
        )
        for _, row in wf.iterrows():
            flag = "✓" if row["gate_pass"] else "✗"
            logger.info(
                f"       Fold {int(row['fold'])}: {flag} "
                f"train_p={row['train_pvalue']:.3f} "
                f"OOS_PF={row['test_pf']:.2f} "
                f"OOS_Sharpe={row['test_sharpe']:.2f}"
            )

    return {"ticker": ticker, **perm, "walk_forward": wf}


# ─────────────────────────────────────────────────────────────────────────────
# Fast in-cycle version (used by engine.py for per-symbol gate)
# ─────────────────────────────────────────────────────────────────────────────

def quick_pvalue(close: pd.Series, n: int = 100) -> float:
    """
    Fast permutation p-value for use inside the live trading cycle.
    Uses n=100 permutations (≈ 2ms per call on CPU).
    Returns 1.0 on any failure (fail-safe: don't gate on error).
    """
    try:
        if len(close) < EMA_SLOW + 10:
            return 1.0
        result = run_permutation_test(close, n=n)
        return result["pvalue"]
    except Exception:
        return 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    n    = 500
    for a in sys.argv[1:]:
        if a.startswith("--n="):
            n = int(a.split("=")[1])
        elif a == "--n" and sys.argv.index(a) + 1 < len(sys.argv):
            n = int(sys.argv[sys.argv.index(a) + 1])

    tickers = args if args else DEFAULT_UNIVERSE

    logger.info(f"Monte Carlo Permutation Test | n={n} | {len(tickers)} tickers")
    logger.info("Gate: p-value < 0.01 → PASS (statistically significant edge)")
    logger.info("=" * 70)

    results = []
    for ticker in tickers:
        res = validate_ticker(ticker, n=n)
        if res:
            results.append(res)

    if results:
        passed = sum(1 for r in results if r.get("pass"))
        logger.info("=" * 70)
        logger.info(
            f"Summary: {passed}/{len(results)} tickers pass the 1% gate. "
            f"Only trade symbols with PASS status."
        )


if __name__ == "__main__":
    main()
