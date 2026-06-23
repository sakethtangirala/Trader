"""
backtest.py — Historical backtesting of the autonomous trading agent.

Replays the full strategy on three periods:
  2006-2010  → 2008 Financial Crisis
  2018-2022  → 2020 COVID Crash + Recovery
  2003-2024  → Full 20-year history

Benchmark: SPY buy-and-hold from the same start date.
Outputs:   8 PNG graphs → graphs/ folder.

Simplifications vs production (documented):
  - Daily bars only (no 1-min microstructure or GK volatility)
  - FinBERT sentiment omitted (no historical RSS archives)
  - No limit-order slippage model (daily close fill)
  - HMM refit every 21 trading days (not every cycle) for performance

Usage:
    python backtest.py
"""
from __future__ import annotations

import os
import sys
import warnings
from datetime import date as date_type

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
from loguru import logger
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config (mirrors settings.py)
# ---------------------------------------------------------------------------

GRAPHS_DIR      = "graphs"
os.makedirs(GRAPHS_DIR, exist_ok=True)

INITIAL_CAPITAL  = 100_000.0
MAX_POSITIONS    = 8
MAX_POS_WEIGHT   = 0.10
CASH_RESERVE     = 0.20
STOP_LOSS        = 0.03
TAKE_PROFIT      = 0.06
MIN_POS_WEIGHT   = 0.015
CRRA_GAMMA       = 3.0
RISK_FREE_RATE   = 0.05
T_HORIZON        = 0.5
HMM_LOOKBACK     = 504       # trading days (~2 years)
HMM_REFIT_EVERY  = 21        # refit HMM monthly for backtest performance
RSI_PERIOD       = 14
EMA_FAST         = 20
EMA_SLOW         = 50
CRASH_THRESHOLD  = -0.03     # MARKET_CRASH_SPY_THRESHOLD

# Liquid large-caps that traded continuously from 2003–2024
UNIVERSE = [
    "AAPL", "MSFT", "JPM",  "JNJ",  "XOM",
    "WMT",  "CVX",  "PG",   "CSCO", "HD",
    "MRK",  "T",    "BAC",  "VZ",   "PFE",
    "ABT",  "MMM",  "IBM",  "GE",   "KO",
]

PERIODS = {
    "2008_crisis":  ("2006-01-01", "2010-12-31"),
    "2020_crash":   ("2018-01-01", "2022-12-31"),
    "full_history": ("2003-01-01", "2024-12-31"),
}
WARMUP_YEARS = 2   # extra history before each period for HMM warm-up

# ---------------------------------------------------------------------------
# Plot styling
# ---------------------------------------------------------------------------

BG      = "#0F172A"
AX_BG   = "#1E293B"
GRID    = "#334155"
TEXT    = "#F1F5F9"
MUTED   = "#94A3B8"
BLUE    = "#60A5FA"
GRAY    = "#64748B"
PURPLE  = "#A78BFA"
AMBER   = "#F59E0B"

REGIME_BG = {"bull": "#064E3B", "bear": "#78350F", "crash": "#7F1D1D"}

plt.rcParams.update({
    "figure.facecolor": BG,    "axes.facecolor":  AX_BG,
    "axes.edgecolor":   GRID,  "axes.labelcolor": MUTED,
    "xtick.color":      MUTED, "ytick.color":     MUTED,
    "grid.color":       GRID,  "grid.alpha":      0.4,
    "text.color":       TEXT,  "legend.facecolor": AX_BG,
    "legend.edgecolor": GRID,  "font.family":     "monospace",
    "axes.titlesize":   13,    "axes.labelsize":  10,
})

# Key events for annotations
EVENTS_2008 = {
    "2007-07-31": "Bear Stearns",
    "2008-09-15": "Lehman",
    "2009-03-09": "SPY bottom",
}
EVENTS_2020 = {
    "2020-02-20": "COVID sell-off",
    "2020-03-23": "Market bottom",
    "2020-11-09": "Vaccine news",
    "2022-01-05": "Fed hike cycle",
}

# ---------------------------------------------------------------------------
# Signal utilities (mirrors engine.py, daily bars)
# ---------------------------------------------------------------------------

def _rsi(series: pd.Series) -> float:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(RSI_PERIOD).mean()
    loss  = (-delta.clip(upper=0)).rolling(RSI_PERIOD).mean()
    rs    = gain / loss
    val   = (100 - 100 / (1 + rs)).iloc[-1]
    return float(val) if np.isfinite(val) else 50.0


def _merton(mu: float, sigma: float) -> float:
    if sigma <= 0:
        return 0.0
    return float(np.clip((mu - RISK_FREE_RATE) / (CRRA_GAMMA * sigma ** 2), 0.0, 1.0))


def _time_to_horizon(d: date_type) -> float:
    year_end = date_type(d.year, 12, 31)
    return min(max((year_end - d).days, 1) / 252.0, T_HORIZON)


def _signal(close: pd.Series, regime: str, alloc: float) -> str:
    if len(close) < EMA_SLOW + 5:
        return "hold"
    rsi_v = _rsi(close)
    ef    = float(close.ewm(span=EMA_FAST, adjust=False).mean().iloc[-1])
    es    = float(close.ewm(span=EMA_SLOW, adjust=False).mean().iloc[-1])
    if regime == "crash":
        return "sell_all"
    if regime == "bull":
        if ef > es and rsi_v < 70 and alloc > 0.05:
            return "buy"
        if rsi_v > 75 or ef < es or alloc < 0.02:
            return "sell"
    else:  # bear
        if rsi_v < 30 and ef > es and alloc > 0.05:
            return "buy"
        if ef < es or alloc < 0.02:
            return "sell"
    return "hold"


# ---------------------------------------------------------------------------
# PINN loader (optional; graceful Merton fallback)
# ---------------------------------------------------------------------------

def _load_pinns() -> dict:
    pinns: dict = {}
    try:
        from pinn.gbm    import GBMPINN
        from pinn.heston import HestonPINN
        from pinn.regime import RegimePINN
        for name, cls, path in [
            ("gbm",    GBMPINN,    "models/gbm_pinn.pt"),
            ("heston", HestonPINN, "models/heston_pinn.pt"),
            ("regime", RegimePINN, "models/regime_pinn.pt"),
        ]:
            try:
                pinns[name] = cls.load(path)
            except Exception:
                pass
    except ImportError:
        pass
    src = "PINNs" if pinns else "Merton ratio (PINNs not found)"
    logger.info(f"Allocation source: {src}")
    return pinns


def _query_alloc(pinns: dict, regime: str, t: float,
                 w: float, vix_var: float) -> float:
    """
    Query PINN hierarchy for market-level equity allocation.
    VIX variance clipped to Heston training domain [1e-4, 0.228] to prevent
    extrapolation during extreme events (2008 VIX hit 80 → vix_var = 0.64).
    Falls back to 0.5 (neutral) if all queries fail; per-stock Merton ratio
    is applied on top of this in the signal filter.
    """
    if regime == "crash":
        return 0.0
    # Clip to Heston training range: y ∈ [1e-4, θ×6 = 0.038×6 = 0.228]
    vix_c = float(np.clip(vix_var, 1e-4, 0.228))
    w_c   = float(np.clip(w, 0.1, 10.0))
    try:
        if "heston" in pinns:
            return float(np.clip(pinns["heston"].query(t, w_c, vix_c), 0, 1))
        if "regime" in pinns:
            return float(np.clip(pinns["regime"].query(t, w_c, 1 if regime == "bear" else 0), 0, 1))
        if "gbm" in pinns:
            return float(np.clip(pinns["gbm"].query(t, w_c), 0, 1))
    except Exception:
        pass
    return 0.5   # neutral fallback; per-stock Merton filters actual entries


# ---------------------------------------------------------------------------
# Regime detection (mirrors engine.py detect_regime)
# ---------------------------------------------------------------------------

def _detect_regime(spy_close: pd.Series, vix_val: float) -> str:
    """
    VIX + moving average regime classification for backtesting.

    The production system uses an unsupervised HMM refit each cycle, which
    works well for live trading (short, recent window, stable calibration).
    Over 20-year backtests the HMM's state ranking creates a labeling
    artefact: even in a sustained bull market the lowest-return state gets
    labelled "crash", blocking all buys for extended periods.

    VIX is an objective market-implied volatility measure that maps cleanly
    to risk regimes across the full history without ambiguity:

      VIX < 20  + SPY MA50 > MA200  → bull
      VIX 20–30 or MA50 < MA200     → bear
      VIX > 30                       → crash

    The hard crash backstop (daily SPY return ≤ -3%) still fires separately
    and immediately.
    """
    if len(spy_close) < 50:
        return "bear"

    ma50  = float(spy_close.rolling(50).mean().iloc[-1])
    ma200 = float(spy_close.rolling(200).mean().iloc[-1]) if len(spy_close) >= 200 else ma50

    if vix_val > 30:
        return "crash"
    elif vix_val > 20 or ma50 < ma200:
        return "bear"
    else:
        return "bull"


# ---------------------------------------------------------------------------
# BacktestEngine
# ---------------------------------------------------------------------------

class BacktestEngine:

    def __init__(self, name: str, start: str, end: str,
                 universe: list[str], pinns: dict):
        self.name    = name
        self.start   = pd.Timestamp(start)
        self.end     = pd.Timestamp(end)
        self.fetch   = self.start - pd.DateOffset(years=WARMUP_YEARS)
        self.uni     = universe
        self.pinns   = pinns

        # Filled in by load_data
        self.close:  pd.DataFrame | None = None   # (date × symbol), ffilled
        self.volume: pd.DataFrame | None = None
        self.vix:    pd.Series    | None = None
        self.days:   pd.DatetimeIndex | None = None

        # Results
        self.strat_vals: list[float]          = []
        self.spy_vals:   list[float]          = []
        self.dates:      list[pd.Timestamp]   = []
        self.regimes:    list[str]            = []
        self.allocs:     list[float]          = []

    # ------------------------------------------------------------------
    def load_data(self) -> bool:
        logger.info(f"[{self.name}] Downloading {self.fetch.date()} – {self.end.date()}")
        tickers = self.uni + ["SPY"]
        try:
            raw = yf.download(
                tickers, start=self.fetch, end=self.end,
                progress=False, auto_adjust=True,
            )
        except Exception as e:
            logger.error(f"[{self.name}] Download failed: {e}")
            return False

        if not isinstance(raw.columns, pd.MultiIndex):
            logger.error(f"[{self.name}] Unexpected column format")
            return False

        self.close  = raw["Close"].ffill()
        self.volume = raw["Volume"].ffill().fillna(1e8)

        # VIX separately (yfinance MultiIndex handles ^ tickers inconsistently)
        try:
            vix_raw = yf.download("^VIX", start=self.fetch, end=self.end,
                                  progress=False, auto_adjust=True)
            vix_col = vix_raw["Close"]
            # yfinance sometimes returns DataFrame instead of Series for single tickers
            if isinstance(vix_col, pd.DataFrame):
                vix_col = vix_col.iloc[:, 0]
            self.vix = vix_col.ffill()
        except Exception:
            self.vix = pd.Series(20.0, index=self.close.index)

        self.days = self.close[self.close.index >= self.start].index
        logger.info(f"[{self.name}] {len(self.uni)} symbols, {len(self.days)} days")
        return True

    # ------------------------------------------------------------------
    def run(self) -> None:
        if not self.load_data() or self.close is None:
            return

        cash       = INITIAL_CAPITAL
        positions: dict[str, dict] = {}   # sym → {shares, avg_price}
        spy_shares: float | None   = None

        regime     = "bear"
        last_refit = -HMM_REFIT_EVERY

        for i, today in enumerate(self.days):

            # ── 1. Regime detection (every 5 days) ──────────────────
            if i - last_refit >= 5:
                spy_c  = self.close["SPY"].loc[:today].dropna()
                vix_s2 = self.vix.loc[:today]
                if isinstance(vix_s2, pd.DataFrame):
                    vix_s2 = vix_s2.iloc[:, 0]
                vix_now = float(vix_s2.dropna().iloc[-1]) if len(vix_s2.dropna()) else 20.0
                regime     = _detect_regime(spy_c, vix_now)
                last_refit = i

            # ── 2. Current prices ────────────────────────────────────
            def px(sym: str) -> float:
                col = self.close.get(sym)
                if col is None:
                    return 0.0
                s = col.loc[:today].dropna()
                return float(s.iloc[-1]) if len(s) else 0.0

            spy_px = px("SPY")
            if spy_shares is None and spy_px > 0:
                spy_shares = INITIAL_CAPITAL / spy_px

            # ── 3. Hard crash backstop ───────────────────────────────
            spy_hist = self.close["SPY"].loc[:today].dropna()
            if len(spy_hist) >= 2:
                spy_1d = (spy_hist.iloc[-1] - spy_hist.iloc[-2]) / spy_hist.iloc[-2]
                if spy_1d <= CRASH_THRESHOLD:
                    regime = "crash"

            # ── 4. VIX variance ──────────────────────────────────────
            vix_s   = self.vix.loc[:today]
            if isinstance(vix_s, pd.DataFrame):
                vix_s = vix_s.iloc[:, 0]
            vix_s   = vix_s.dropna()
            vix_val = float(vix_s.iloc[-1]) if len(vix_s) else 20.0
            vix_var = (vix_val / 100.0) ** 2

            # ── 5. Portfolio value & PINN allocation ─────────────────
            port = cash + sum(
                pos["shares"] * px(sym) for sym, pos in positions.items()
            )
            w    = max(port / INITIAL_CAPITAL, 0.1)
            t    = _time_to_horizon(today.date())

            alloc = _query_alloc(self.pinns, regime, t, w, vix_var)

            # ── 6. Exit positions ────────────────────────────────────
            to_exit = []
            for sym, pos in positions.items():
                p = px(sym)
                if p <= 0:
                    continue
                pnl = (p - pos["avg_price"]) / pos["avg_price"]
                if pnl <= -STOP_LOSS or pnl >= TAKE_PROFIT or regime == "crash":
                    cash += pos["shares"] * p
                    to_exit.append(sym)
            for sym in to_exit:
                positions.pop(sym, None)

            # ── 7. Find buy candidates ───────────────────────────────
            buy_cands: list[tuple[str, float, float]] = []   # (sym, price, merton)

            if regime != "crash" and len(positions) < MAX_POSITIONS:
                for sym in self.uni:
                    if sym in positions:
                        continue
                    col = self.close.get(sym)
                    if col is None:
                        continue
                    series = col.loc[:today].dropna().iloc[-90:]
                    if len(series) < EMA_SLOW + 5:
                        continue
                    p = float(series.iloc[-1])
                    if p <= 0:
                        continue

                    # Per-stock Merton ratio
                    ret  = series.pct_change().dropna()
                    mu   = float(ret.mean() * 252)
                    sig  = float(ret.std()  * np.sqrt(252))
                    m    = _merton(mu, sig)

                    # Use whichever is higher: PINN market alloc or per-stock Merton.
                    # This prevents PINN conservatism from blocking all buys when
                    # individual stocks have strong positive expected returns.
                    eff_alloc = max(alloc, m) if self.pinns else m

                    sig_label = _signal(series, regime, eff_alloc)
                    if sig_label == "buy":
                        buy_cands.append((sym, p, m))

                # Sort by Merton weight (mirrors production)
                buy_cands.sort(key=lambda x: x[2], reverse=True)

            # First-cycle diagnostic (Day 0 and first trade day)
            if i == 0 or (i < 30 and buy_cands and not positions):
                avg_m = np.mean([m for _, _, m in buy_cands]) if buy_cands else 0
                logger.info(
                    f"[{self.name}] Day {i}: regime={regime} "
                    f"pinn_alloc={alloc:.4f} avg_merton={avg_m:.4f} "
                    f"buy_cands={len(buy_cands)} cash={cash:,.0f}"
                )

            # ── 8. Enter positions ───────────────────────────────────
            slots        = MAX_POSITIONS - len(positions)
            to_buy       = buy_cands[:slots]
            total_merton = sum(m for _, _, m in to_buy) or 1.0

            if to_buy:
                # PINN sets a portfolio-level equity ceiling.
                # Fallback: if PINN alloc is suppressed (calibration range, warm-up),
                # use the average Merton ratio of buy candidates as the sizing floor.
                avg_merton   = sum(m for _, _, m in to_buy) / len(to_buy)
                portfolio_alloc = max(alloc, avg_merton * 0.5)

            for sym, p, m in to_buy:
                weight   = m / total_merton
                tgt_val  = port * portfolio_alloc * weight
                if tgt_val < port * MIN_POS_WEIGHT:
                    continue
                pos_val = min(tgt_val,
                              port * MAX_POS_WEIGHT,
                              cash * (1 - CASH_RESERVE))
                if pos_val <= 0 or p <= 0 or cash < pos_val:
                    continue
                shares = pos_val / p
                cash  -= pos_val
                positions[sym] = {"shares": shares, "avg_price": p}

            # ── 9. Record ────────────────────────────────────────────
            port = cash + sum(pos["shares"] * px(sym)
                              for sym, pos in positions.items())
            spy_v = (spy_shares or 0) * spy_px

            self.strat_vals.append(port)
            self.spy_vals.append(spy_v)
            self.dates.append(today)
            self.regimes.append(regime)
            self.allocs.append(alloc)

        if self.strat_vals:
            strat_ret = (self.strat_vals[-1] / INITIAL_CAPITAL - 1) * 100
            spy_ret   = (self.spy_vals[-1]   / INITIAL_CAPITAL - 1) * 100
            logger.info(
                f"[{self.name}] Done — "
                f"Strategy {strat_ret:+.1f}%  vs  SPY {spy_ret:+.1f}%"
            )

    # ------------------------------------------------------------------
    def results(self) -> pd.DataFrame:
        df = pd.DataFrame({
            "strategy": self.strat_vals,
            "spy":      self.spy_vals,
            "regime":   self.regimes,
            "alloc":    self.allocs,
        }, index=self.dates)
        df["s_ret"] = df["strategy"].pct_change()
        df["b_ret"] = df["spy"].pct_change()
        return df


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def metrics(vals: list[float], dates: list) -> dict:
    arr  = np.array(vals, dtype=float)
    rets = np.diff(arr) / arr[:-1]
    if len(rets) == 0 or len(dates) < 2:
        return {}
    n_yr = (dates[-1] - dates[0]).days / 365.25
    cagr = (arr[-1] / arr[0]) ** (1 / max(n_yr, 0.01)) - 1
    rf_d = RISK_FREE_RATE / 252
    exc  = rets - rf_d
    shrp = float(exc.mean() / exc.std() * np.sqrt(252)) if exc.std() > 0 else 0.0
    peak = np.maximum.accumulate(arr)
    mdd  = float(((arr - peak) / peak).min())
    return {
        "CAGR %":       cagr * 100,
        "Sharpe":       shrp,
        "Max DD %":     mdd * 100,
        "Win Rate %":   float(np.sum(rets > 0) / len(rets) * 100),
        "Ann. Vol %":   float(rets.std() * np.sqrt(252) * 100),
        "Calmar":       cagr / abs(mdd) if mdd else 0.0,
        "Total Ret %":  float(arr[-1] / arr[0] - 1) * 100,
    }


# ---------------------------------------------------------------------------
# Shared plot helpers
# ---------------------------------------------------------------------------

def _shade_regimes(ax, dates, regimes):
    if not dates:
        return
    cur, s = regimes[0], dates[0]
    for i in range(1, len(dates)):
        if regimes[i] != cur or i == len(dates) - 1:
            ax.axvspan(s, dates[i],
                       color=REGIME_BG.get(cur, AX_BG),
                       alpha=0.35, linewidth=0)
            cur, s = regimes[i], dates[i]


def _annotate_events(ax, events: dict):
    ylim = ax.get_ylim()
    for ds, lbl in events.items():
        try:
            dt = pd.Timestamp(ds)
            ax.axvline(dt, color=AMBER, alpha=0.6, linewidth=0.9, linestyle="--")
            ax.text(dt, ylim[1] * 0.98, lbl,
                    rotation=90, fontsize=6.5, color=AMBER,
                    va="top", ha="right", alpha=0.85)
        except Exception:
            pass


def _fmt_ax(ax):
    ax.set_facecolor(AX_BG)
    ax.tick_params(colors=MUTED)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.grid(True, alpha=0.3)


def _save(fig, name: str):
    path = os.path.join(GRAPHS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    logger.info(f"  → {path}")


# ---------------------------------------------------------------------------
# Graph 1 / 2 / 3 — Equity curves
# ---------------------------------------------------------------------------

def plot_equity(df: pd.DataFrame, title: str,
                filename: str, events: dict | None = None):
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(13, 7),
        gridspec_kw={"height_ratios": [3, 1]},
        facecolor=BG,
    )
    fig.subplots_adjust(hspace=0.06)
    dates = df.index.tolist()

    s = df["strategy"] / df["strategy"].iloc[0] * 100
    b = df["spy"]      / df["spy"].iloc[0]      * 100

    _shade_regimes(ax1, dates, df["regime"].tolist())
    ax1.plot(dates, s, color=BLUE,  linewidth=1.8, label="Strategy",  zorder=5)
    ax1.plot(dates, b, color=GRAY,  linewidth=1.2, label="SPY B&H",   zorder=4, linestyle="--")
    ax1.axhline(100, color="#475569", linewidth=0.5, linestyle=":")

    if events:
        _annotate_events(ax1, events)

    regime_patches = [
        mpatches.Patch(facecolor=REGIME_BG["bull"],  alpha=0.7, label="Bull"),
        mpatches.Patch(facecolor=REGIME_BG["bear"],  alpha=0.7, label="Bear"),
        mpatches.Patch(facecolor=REGIME_BG["crash"], alpha=0.7, label="Crash"),
    ]
    ax1.legend(handles=[ax1.lines[0], ax1.lines[1]] + regime_patches,
               fontsize=8, loc="upper left")
    ax1.set_title(title, pad=10)
    ax1.set_ylabel("Value (rebased 100)")
    ax1.tick_params(labelbottom=False)
    _fmt_ax(ax1)

    # Drawdown panel
    def dd(arr):
        peak = np.maximum.accumulate(arr)
        return (arr - peak) / peak * 100

    ax2.fill_between(dates, dd(s.values), 0, color=BLUE, alpha=0.45, label="Strategy")
    ax2.fill_between(dates, dd(b.values), 0, color=GRAY, alpha=0.25, label="SPY")
    ax2.set_ylabel("Drawdown %")
    ax2.legend(fontsize=7, loc="lower left")
    _fmt_ax(ax2)

    _save(fig, filename)


# ---------------------------------------------------------------------------
# Graph 4 — Drawdown comparison (full history)
# ---------------------------------------------------------------------------

def plot_drawdown(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(13, 4), facecolor=BG)
    dates   = df.index.tolist()

    def dd(col):
        arr  = df[col].values.astype(float)
        peak = np.maximum.accumulate(arr)
        return (arr - peak) / peak * 100

    ax.fill_between(dates, dd("strategy"), 0, color=BLUE, alpha=0.5, label="Strategy")
    ax.fill_between(dates, dd("spy"),      0, color=GRAY, alpha=0.3, label="SPY B&H")
    ax.plot(dates, dd("strategy"), color=BLUE, linewidth=0.7)
    ax.plot(dates, dd("spy"),      color=GRAY, linewidth=0.5, linestyle="--")
    ax.set_title("Maximum Drawdown — Full History 2003–2024", pad=10)
    ax.set_ylabel("Drawdown %")
    ax.legend(fontsize=9)
    _fmt_ax(ax)
    _save(fig, "drawdown_comparison.png")


# ---------------------------------------------------------------------------
# Graph 5 — Monthly returns heatmap
# ---------------------------------------------------------------------------

def plot_monthly_heatmap(df: pd.DataFrame, name: str):
    dr   = df["s_ret"].dropna()
    dr.index = pd.DatetimeIndex(dr.index)
    monthly = dr.resample("ME").apply(lambda x: (1 + x).prod() - 1) * 100

    pivot = monthly.groupby([monthly.index.year, monthly.index.month]).sum()
    pivot = pivot.unstack(level=1)
    pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                     "Jul","Aug","Sep","Oct","Nov","Dec"]

    fig, ax = plt.subplots(
        figsize=(14, max(4, len(pivot) * 0.45)), facecolor=BG
    )
    ax.set_facecolor(AX_BG)

    mat  = pivot.values
    vmax = np.nanpercentile(np.abs(mat[np.isfinite(mat)]), 95) if np.isfinite(mat).any() else 5
    im   = ax.imshow(mat, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(12))
    ax.set_xticklabels(pivot.columns, fontsize=8, color=MUTED)
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index, fontsize=8, color=MUTED)

    for i in range(len(pivot)):
        for j in range(12):
            v = mat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                        fontsize=6.5,
                        color="white" if abs(v) > vmax * 0.55 else TEXT)

    plt.colorbar(im, ax=ax, label="Monthly Return %", shrink=0.8)
    label = name.replace("_", " ").title()
    ax.set_title(f"Monthly Returns Heatmap — {label}", pad=10)
    _save(fig, f"monthly_heatmap_{name}.png")


# ---------------------------------------------------------------------------
# Graph 6 — Rolling Sharpe (full history)
# ---------------------------------------------------------------------------

def plot_rolling_sharpe(df: pd.DataFrame):
    rf_d = RISK_FREE_RATE / 252
    win  = 252

    def rsharpe(ret_col):
        r = df[ret_col].dropna()
        return r.rolling(win).apply(
            lambda x: (x - rf_d).mean() / (x - rf_d).std() * np.sqrt(252)
            if (x - rf_d).std() > 0 else 0.0,
            raw=True,
        )

    ss = rsharpe("s_ret")
    bs = rsharpe("b_ret")

    fig, ax = plt.subplots(figsize=(13, 4), facecolor=BG)
    ax.plot(ss.index, ss.values, color=BLUE, linewidth=1.4, label="Strategy")
    ax.plot(bs.index, bs.values, color=GRAY, linewidth=1.0, label="SPY B&H", linestyle="--")
    ax.fill_between(ss.index, ss.values, bs.values,
                    where=(ss.values > bs.values),
                    color=BLUE, alpha=0.15, label="Strategy outperforms")
    ax.axhline(0, color="#475569", linewidth=0.5)
    ax.axhline(1, color="#10B981", linewidth=0.5, linestyle=":", alpha=0.6)
    ax.set_title("Rolling 252-Day Sharpe Ratio — Full History 2003–2024", pad=10)
    ax.set_ylabel("Sharpe Ratio")
    ax.legend(fontsize=8)
    _fmt_ax(ax)
    _save(fig, "rolling_sharpe.png")


# ---------------------------------------------------------------------------
# Graph 7 — PINN allocation over time (full history)
# ---------------------------------------------------------------------------

def plot_allocation(df: pd.DataFrame):
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(13, 6),
        gridspec_kw={"height_ratios": [2, 1]},
        facecolor=BG,
    )
    fig.subplots_adjust(hspace=0.06)
    dates = df.index.tolist()

    _shade_regimes(ax1, dates, df["regime"].tolist())
    ax1.plot(dates, df["alloc"], color=PURPLE, linewidth=1.2, label="π* allocation")
    ax1.fill_between(dates, df["alloc"], 0, color=PURPLE, alpha=0.15)
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("π* (equity fraction)")
    ax1.set_title("PINN/Merton Optimal Allocation π*(t, w, y) — Full History", pad=10)
    ax1.tick_params(labelbottom=False)
    ax1.legend(fontsize=8)
    _fmt_ax(ax1)

    rmap  = {"bull": 2, "bear": 1, "crash": 0}
    rvals = [rmap.get(r, 1) for r in df["regime"]]
    ax2.fill_between(dates, rvals, 0, color=BLUE, alpha=0.5)
    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(["Crash", "Bear", "Bull"], fontsize=8)
    ax2.set_ylabel("Regime")
    _fmt_ax(ax2)
    _save(fig, "pinn_allocation.png")


# ---------------------------------------------------------------------------
# Graph 8 — Summary stats table
# ---------------------------------------------------------------------------

def plot_summary(all_results: dict[str, pd.DataFrame]):
    rows = []
    col_order = ["CAGR %", "Sharpe", "Max DD %", "Ann. Vol %", "Calmar", "Win Rate %", "Total Ret %"]

    for name, df in all_results.items():
        label = name.replace("_", " ").title()
        sm    = metrics(df["strategy"].tolist(), df.index.tolist())
        bm    = metrics(df["spy"].tolist(),      df.index.tolist())
        row   = {"Period": label}
        for k in col_order:
            sv = sm.get(k, 0)
            bv = bm.get(k, 0)
            fmt = ".1f" if "%" in k else ".2f"
            row[f"Strat {k}"] = f"{sv:{fmt}}"
            row[f"SPY {k}"]   = f"{bv:{fmt}}"
        rows.append(row)

    if not rows:
        return
    tdf     = pd.DataFrame(rows)
    n_cols  = len(tdf.columns)
    n_rows  = len(tdf)

    fig, ax = plt.subplots(
        figsize=(max(16, n_cols * 1.6), 2 + n_rows * 0.7), facecolor=BG
    )
    ax.set_facecolor(BG)
    ax.axis("off")

    tbl = ax.table(
        cellText  = tdf.values,
        colLabels = tdf.columns,
        cellLoc   = "center",
        loc       = "center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.8)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(GRID)
        if r == 0:
            cell.set_facecolor("#1E40AF")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 1:
            cell.set_facecolor("#1E293B")
            cell.set_text_props(color=TEXT)
        else:
            cell.set_facecolor("#0F172A")
            cell.set_text_props(color=TEXT)

    ax.set_title("Performance Summary vs S&P 500 Benchmark",
                 fontsize=14, color=TEXT, y=0.98)
    _save(fig, "summary_stats.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logger.remove()
    logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | {message}", level="INFO")

    logger.info("=" * 55)
    logger.info("Autonomous Trading Agent — Historical Backtest")
    logger.info("=" * 55)

    pinns       = _load_pinns()
    all_results: dict[str, pd.DataFrame] = {}

    for name, (start, end) in PERIODS.items():
        logger.info(f"\nPeriod: {name}  ({start} → {end})")
        eng = BacktestEngine(name, start, end, UNIVERSE, pinns)
        eng.run()
        if eng.strat_vals:
            all_results[name] = eng.results()

    if not all_results:
        logger.error("No results — check data connection.")
        return

    logger.info("\nGenerating graphs...")

    if "2008_crisis" in all_results:
        plot_equity(all_results["2008_crisis"],
                    "2008 Financial Crisis — Strategy vs S&P 500",
                    "equity_2008_crisis.png", EVENTS_2008)

    if "2020_crash" in all_results:
        plot_equity(all_results["2020_crash"],
                    "2020 COVID Crash & Recovery — Strategy vs S&P 500",
                    "equity_2020_crash.png", EVENTS_2020)

    if "full_history" in all_results:
        plot_equity(all_results["full_history"],
                    "Full History 2003–2024 — Strategy vs S&P 500",
                    "equity_full_history.png")
        plot_drawdown(all_results["full_history"])
        plot_rolling_sharpe(all_results["full_history"])
        plot_allocation(all_results["full_history"])

    for name, df in all_results.items():
        plot_monthly_heatmap(df, name)

    plot_summary(all_results)

    logger.info(f"\nDone. {len(os.listdir(GRAPHS_DIR))} files in graphs/")
    for f in sorted(os.listdir(GRAPHS_DIR)):
        if f.endswith(".png"):
            kb = os.path.getsize(os.path.join(GRAPHS_DIR, f)) // 1024
            logger.info(f"  {f:<45} {kb} KB")


if __name__ == "__main__":
    main()
