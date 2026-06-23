"""
backtest_engine.py — Event-driven, high-fidelity backtester for the PINN trading agent.

Architecture:
  MarketData        — time-gated OHLCV access; all data pre-loaded, no lookahead
  IntradaySimulator — Brownian-bridge 1-min bar synthesis for the fill engine
  EventClock        — dynamic tick schedule (2h bull / 15min bear, 5min exit sub-steps)
  RegimeDetector    — rolling 504-bar 3-state HMM, refitted at every major cycle
  PINNAllocator     — loads trained .pt models; falls back to Merton analytical π*
  SignalEngine      — EMA trend, 12-1 momentum, bear-recovery entry logic
  FillEngine        — 2-bps passive limit fill or 1.5-bps market-sweep penalty
  Portfolio         — position ledger, cash, peak-price tracker, equity history
  RiskManager       — γ-scaling, vol targeting, inv-vol weighting, sector cap, sizing
  BacktestEngine    — main chronological event loop
  ResultsAnalyzer   — CAGR, Sharpe, Sortino, MaxDD, turnover; plots vs SPY B&H

Anti-lookahead guarantees:
  · Every data accessor takes an `as_of: date` and returns only rows ≤ as_of.
  · HMM is refit on the trailing 504 calendar days prior to the event timestamp.
  · PINNs are statically calibrated from paper Tables 1-4 (no historical fitting).

Usage:
    python backtest_engine.py                      # default 2017-01-01 to 2025-06-30
    python backtest_engine.py 2003-01-01 2025-06-30
"""
from __future__ import annotations

import hashlib
import os
import sys
import warnings
from collections import deque
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
from loguru import logger
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Constants (mirrors settings.py — standalone so no import needed)
# ─────────────────────────────────────────────────────────────────────────────

INITIAL_CAPITAL         = 100_000.0
CRRA_GAMMA              = 3.0
CRRA_GAMMA_BULL         = 1.5          # regime-aware γ; scales π* × 2 in bull
RISK_FREE_RATE          = 0.05
TRADING_HORIZON_T       = 0.5

MAX_POSITIONS           = 14
MAX_POS_WEIGHT          = 0.10
MIN_POS_WEIGHT          = 0.015        # hysteresis buffer
CASH_RESERVE_PCT        = 0.10
SECTOR_MAX_POS          = 3

STOP_LOSS_PCT           = 0.03
TRAILING_STOP_PCT       = 0.10
TAKE_PROFIT_PCT         = 0.0          # disabled

HMM_LOOKBACK_DAYS       = 504
HMM_N_STATES            = 3
EMA_FAST                = 20
EMA_SLOW                = 50
RSI_PERIOD              = 14
MOM_LOOKBACK_MONTHS     = 12
MOM_SKIP_MONTHS         = 1

VOL_TARGET              = 0.15
VOL_LOOKBACK_DAYS       = 20
VOL_SCALE_MIN           = 0.5
VOL_SCALE_MAX           = 1.5

VIX_HIGH_THRESHOLD      = 25.0
CYCLE_BULL_MIN          = 120          # 2 h in calm bull
CYCLE_BEAR_MIN          = 15           # 15 min otherwise
STOP_SCAN_MIN           = 5            # exit-only sub-loop cadence

LIMIT_OFFSET_BPS        = 2
LIMIT_TIMEOUT_MIN       = 4
MARKET_SLIPPAGE_BPS     = 1.5

MARKET_OPEN             = time(9, 30)
MARKET_CLOSE            = time(16, 0)

GRAPHS_DIR = "graphs"
os.makedirs(GRAPHS_DIR, exist_ok=True)

# GICS sector map for universe tickers
SECTOR_MAP: dict[str, str] = {
    "AAPL": "Technology",   "MSFT": "Technology",   "CSCO": "Technology",
    "IBM":  "Technology",   "TXN":  "Technology",   "ORCL": "Technology",
    "ADP":  "Technology",   "NVDA": "Technology",   "AMD":  "Technology",
    "INTC": "Technology",   "QCOM": "Technology",   "META": "Technology",
    "GOOGL":"Technology",   "AMZN": "Consumer Cyclical",
    "JNJ":  "Healthcare",   "MRK":  "Healthcare",   "PFE":  "Healthcare",
    "ABT":  "Healthcare",   "UNH":  "Healthcare",   "AMGN": "Healthcare",
    "MDT":  "Healthcare",   "LLY":  "Healthcare",   "BMY":  "Healthcare",
    "WMT":  "Consumer Defensive", "PG": "Consumer Defensive",
    "KO":   "Consumer Defensive", "TGT":"Consumer Defensive",
    "CL":   "Consumer Defensive", "COST":"Consumer Defensive",
    "JPM":  "Financial Services", "BAC":"Financial Services",
    "WFC":  "Financial Services", "C":  "Financial Services",
    "USB":  "Financial Services", "GS": "Financial Services",
    "HD":   "Consumer Cyclical",  "MCD":"Consumer Cyclical",
    "NKE":  "Consumer Cyclical",  "LOW":"Consumer Cyclical",
    "TSLA": "Consumer Cyclical",
    "MMM":  "Industrials",  "GE":  "Industrials",   "HON": "Industrials",
    "CAT":  "Industrials",  "EMR": "Industrials",   "GD":  "Industrials",
    "XOM":  "Energy",       "CVX": "Energy",        "SLB": "Energy",
    "T":    "Communication Services", "VZ": "Communication Services",
    "DUK":  "Utilities",    "SO":  "Utilities",
    "SPY":  "ETF",          "QQQ": "ETF",
}

DEFAULT_UNIVERSE = [
    "AAPL", "MSFT", "JPM",  "JNJ",  "XOM",
    "WMT",  "CVX",  "PG",   "CSCO", "HD",
    "MRK",  "BAC",  "VZ",   "PFE",  "ABT",
    "MMM",  "IBM",  "GE",   "KO",   "UNH",
    "AMGN", "MDT",  "MCD",  "TGT",  "NKE",
    "TXN",  "ORCL", "WFC",  "HON",  "CAT",
]

# ─────────────────────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Position:
    symbol:     str
    shares:     float
    avg_price:  float
    peak_price: float
    entry_dt:   datetime

@dataclass
class Fill:
    symbol:    str
    side:      str           # 'buy' | 'sell'
    qty:       float
    price:     float
    dt:        datetime
    fill_type: str           # 'limit_passive' | 'market_sweep' | 'stop' | 'crash'
    cost_bps:  float

@dataclass
class PendingOrder:
    symbol:       str
    qty:          float
    limit_price:  float
    submitted_dt: datetime

# ─────────────────────────────────────────────────────────────────────────────
# MarketData — time-gated, anti-lookahead data layer
# ─────────────────────────────────────────────────────────────────────────────

class MarketData:
    """
    All OHLCV and VIX data pre-loaded once.  Every accessor is gated by `as_of`
    so callers can only observe data that existed at simulation time T.
    """

    def __init__(self, symbols: list[str], start: date, end: date) -> None:
        all_tickers = sorted(set(symbols + ["SPY"]))
        # Extra 3-year warmup so HMM has 504 days on day 1 of the backtest
        warmup = (datetime.combine(start, time()) - timedelta(days=900)).date()
        logger.info(f"Downloading {len(all_tickers)} tickers {warmup} → {end} …")
        raw = yf.download(
            all_tickers, start=str(warmup), end=str(end),
            progress=False, auto_adjust=True,
        )
        if isinstance(raw.columns, pd.MultiIndex):
            self._open   = raw["Open"].ffill()
            self._high   = raw["High"].ffill()
            self._low    = raw["Low"].ffill()
            self._close  = raw["Close"].ffill()
            self._volume = raw["Volume"].ffill().fillna(1e6)
        else:
            raise RuntimeError("Unexpected yfinance column format — need multi-ticker download.")

        vix_raw = yf.download("^VIX", start=str(warmup), end=str(end),
                              progress=False, auto_adjust=True)
        vc = vix_raw["Close"]
        if isinstance(vc, pd.DataFrame):
            vc = vc.iloc[:, 0]
        self._vix: pd.Series = vc.ffill()

        # Pre-index dates for O(1) day lookups — eliminates repeated O(N) date scans
        dates_arr = np.array([d.date() for d in self._close.index])
        self._date_to_iloc: dict[date, int] = {d: i for i, d in enumerate(dates_arr)}

        # Caches — populated lazily, live for the whole backtest run
        self._ohlcv_cache:  dict[tuple, pd.DataFrame] = {}
        self._sigma_cache:  dict[tuple, float]         = {}

        logger.info(f"Data ready: {len(self._close)} bars × {len(all_tickers)} tickers")

    # ── O(1) day-bar lookup (no DataFrame construction per call) ─────────────

    def day_bar(self, symbol: str, day: date) -> dict | None:
        """
        Returns {open, high, low, close} for the given calendar day.
        Returns None if the day has no data for that symbol.
        Anti-lookahead: only day==as_of is returned; callers must not pass future dates.
        """
        iloc = self._date_to_iloc.get(day)
        if iloc is None:
            return None
        col_c = self._close.get(symbol)
        if col_c is None or pd.isna(col_c.iloc[iloc]):
            return None
        return {
            "open":  float(self._open.iloc[iloc].get(symbol, col_c.iloc[iloc])),
            "high":  float(self._high.iloc[iloc].get(symbol, col_c.iloc[iloc])),
            "low":   float(self._low.iloc[iloc].get(symbol,  col_c.iloc[iloc])),
            "close": float(col_c.iloc[iloc]),
        }

    # ── Gated accessors (used for multi-bar lookups: HMM, signals) ───────────

    def close_series(self, symbol: str, as_of: date) -> pd.Series:
        col = self._close.get(symbol)
        if col is None:
            return pd.Series(dtype=float)
        iloc = self._date_to_iloc.get(as_of)
        if iloc is None:
            return pd.Series(dtype=float)
        return col.iloc[:iloc + 1].dropna()

    def ohlcv(self, symbol: str, as_of: date) -> pd.DataFrame:
        """Full OHLCV history up to as_of — cached per (symbol, date)."""
        key = (symbol, as_of)
        if key in self._ohlcv_cache:
            return self._ohlcv_cache[key]
        iloc = self._date_to_iloc.get(as_of)
        if iloc is None:
            return pd.DataFrame()
        slc = slice(None, iloc + 1)
        df  = pd.DataFrame({
            "open":   self._open.get(symbol,   pd.Series(dtype=float)).iloc[slc],
            "high":   self._high.get(symbol,   pd.Series(dtype=float)).iloc[slc],
            "low":    self._low.get(symbol,    pd.Series(dtype=float)).iloc[slc],
            "close":  self._close.get(symbol,  pd.Series(dtype=float)).iloc[slc],
            "volume": self._volume.get(symbol, pd.Series(dtype=float)).iloc[slc],
        }).dropna(subset=["close"])
        self._ohlcv_cache[key] = df
        return df

    def latest_close(self, symbol: str, as_of: date) -> float:
        iloc = self._date_to_iloc.get(as_of)
        if iloc is None:
            return 0.0
        col = self._close.get(symbol)
        if col is None:
            return 0.0
        # Walk back to find the last non-NaN value
        for i in range(iloc, max(iloc - 5, -1), -1):
            v = col.iloc[i]
            if not pd.isna(v):
                return float(v)
        return 0.0

    def vix(self, as_of: date) -> float:
        iloc = self._date_to_iloc.get(as_of)
        if iloc is None:
            return 20.0
        for i in range(iloc, max(iloc - 5, -1), -1):
            v = self._vix.iloc[i]
            if not pd.isna(v):
                return float(v)
        return 20.0

    def daily_sigma(self, symbol: str, as_of: date, window: int = 60) -> float:
        """Annualised daily σ from rolling `window` bars — cached."""
        key = (symbol, as_of, window)
        if key in self._sigma_cache:
            return self._sigma_cache[key]
        s = self.close_series(symbol, as_of).iloc[-window:]
        if len(s) < 5:
            return 0.20
        sigma = float(s.pct_change().dropna().std() * np.sqrt(252))
        self._sigma_cache[key] = sigma
        return sigma

    def trading_days(self, start: date, end: date) -> list[date]:
        return [d for d in self._date_to_iloc
                if start <= d <= end]

# ─────────────────────────────────────────────────────────────────────────────
# IntradaySimulator — Brownian-bridge 1-min bar synthesis
# ─────────────────────────────────────────────────────────────────────────────

class IntradaySimulator:
    """
    Constructs synthetic 1-minute OHLCV paths anchored to daily Open/Close and
    bounded by daily High/Low via a Brownian bridge.  Seeds are deterministic
    (hash of symbol + date) for exact reproducibility.

    Used by FillEngine to simulate whether a 2-bps limit is touched within 4 min.
    """

    BARS_PER_DAY = 390

    def __init__(self) -> None:
        self._cache: dict[tuple, np.ndarray] = {}

    def _bars(self, symbol: str, day: date,
              o: float, h: float, lp: float, c: float,
              sigma_daily: float) -> np.ndarray:
        key = (symbol, day)
        if key in self._cache:
            return self._cache[key]

        n   = self.BARS_PER_DAY
        rng = np.random.default_rng(
            int(hashlib.md5(f"{symbol}{day}".encode()).hexdigest(), 16) % (2 ** 31)
        )

        sigma_min = sigma_daily / np.sqrt(252 * n)
        log_drift = np.log(c / o) if o > 0 and c > 0 else 0.0

        # Brownian bridge: enforce path ends at close
        t     = np.arange(1, n + 1) / n
        z     = rng.standard_normal(n) * sigma_min
        noise = np.cumsum(z) - np.outer(t, np.cumsum(z)[-1]).flatten()
        path  = t * log_drift + noise

        prices = o * np.exp(path)
        prices = np.clip(prices, lp * 0.999, h * 1.001)

        bars = np.empty((n, 4))   # [open, high, low, close]
        prev = o
        for i in range(n):
            bar_c = float(prices[i])
            spread = abs(rng.standard_normal()) * sigma_min * prev * 0.4
            bar_h  = min(max(prev, bar_c) + spread, h)
            bar_l  = max(min(prev, bar_c) - spread, lp)
            bars[i] = [prev, bar_h, bar_l, bar_c]
            prev = bar_c

        self._cache[key] = bars
        return bars

    def get_price_at_minute(self, symbol: str, day: date, minute: int,
                            o: float, h: float, lp: float, c: float,
                            sigma: float) -> float:
        bars = self._bars(symbol, day, o, h, lp, c, sigma)
        idx  = max(0, min(minute, len(bars) - 1))
        return float(bars[idx, 3])

    def min_low_range(self, symbol: str, day: date,
                      min_from: int, min_to: int,
                      o: float, h: float, lp: float, c: float,
                      sigma: float) -> float:
        bars = self._bars(symbol, day, o, h, lp, c, sigma)
        a, b = max(0, min_from), min(min_to, len(bars) - 1)
        return float(bars[a:b + 1, 2].min()) if a <= b else float("inf")

# ─────────────────────────────────────────────────────────────────────────────
# RegimeDetector — rolling 504-bar HMM, strict no-lookahead
# ─────────────────────────────────────────────────────────────────────────────

class RegimeDetector:
    """
    3-state Gaussian HMM trained exclusively on the 504 daily SPY bars
    immediately preceding the simulation timestamp.

    State-to-label mapping is dynamic per fit (sorted by mean daily return):
      highest mean → 'bull', middle → 'bear', lowest → 'crash'.

    Includes a 2-reading confirmation smoother (_RegimeSmoother) identical
    to the one in main.py to prevent single-candle covariance spikes from
    triggering spurious regime flips.
    """

    def __init__(self, data: MarketData) -> None:
        self._data      = data
        self._confirmed = "bear"
        self._history: deque[str] = deque(maxlen=2)
        self._last_fit: Optional[date] = None
        self._cache: dict[date, str] = {}

    def _fit_and_predict(self, as_of: date) -> str:
        spy = self._data.ohlcv("SPY", as_of).tail(HMM_LOOKBACK_DAYS)
        if len(spy) < 60:
            return "bear"

        spy["ret"]  = spy["close"].pct_change()
        spy["vol"]  = spy["ret"].rolling(20).std()
        spy["vrat"] = spy["volume"] / spy["volume"].rolling(20).mean()
        spy.dropna(inplace=True)

        X      = spy[["ret", "vol", "vrat"]].values
        scaler = StandardScaler()
        X_sc   = scaler.fit_transform(X)

        model = GaussianHMM(n_components=HMM_N_STATES, covariance_type="full",
                            n_iter=200, random_state=42)
        model.fit(X_sc)

        # hmmlearn can produce degenerate transition rows (sum=0) when a state
        # is never visited during EM. Normalise any such rows to uniform before
        # calling predict(), which validates the matrix and would otherwise crash.
        row_sums = model.transmat_.sum(axis=1, keepdims=True)
        zero_rows = (row_sums.flatten() == 0)
        if zero_rows.any():
            model.transmat_[zero_rows] = 1.0 / HMM_N_STATES
            row_sums = model.transmat_.sum(axis=1, keepdims=True)
            model.transmat_ /= row_sums

        states = model.predict(X_sc)
        cur    = int(states[-1])

        mean_ret = {s: X[states == s, 0].mean()
                    for s in range(HMM_N_STATES) if (states == s).any()}
        ranked   = sorted(mean_ret, key=lambda s: mean_ret[s], reverse=True)
        labels   = ["bull", "bear", "crash"]
        regime_map = {ranked[i]: labels[i] for i in range(len(ranked))}
        return regime_map.get(cur, "bear")

    def detect(self, as_of: date) -> str:
        if as_of in self._cache:
            return self._cache[as_of]

        raw = self._fit_and_predict(as_of)

        # VIX hard backstop: force crash on ≥ 40
        vix = self._data.vix(as_of)
        if vix >= 40.0:
            raw = "crash"

        # 2-reading confirmation smoother
        self._history.append(raw)
        if len(self._history) == 2 and len(set(self._history)) == 1:
            self._confirmed = raw
        regime = self._confirmed

        self._cache[as_of] = regime
        return regime

# ─────────────────────────────────────────────────────────────────────────────
# PINNAllocator — real .pt models or Merton analytical fallback
# ─────────────────────────────────────────────────────────────────────────────

class PINNAllocator:
    """
    Tries to load the trained PINN .pt files from models/.
    If unavailable (ImportError / missing files), falls back to the analytical
    Merton ratio solution which the PINNs are approximating:

        π*(t, w) ≈ (μ - r) / (γ_eff · σ²)     [GBM/Merton approximation]

    The Heston PINN additionally incorporates VIX variance (y = (VIX/100)²).
    All outputs are clipped to [0, 1] and γ-scaled for regime.

    Paper-calibrated parameters used in mock (Tables 1-4):
        μ=0.128, σ=0.173 (GBM), κ=6.586, θ=0.038, σ_Y=0.643, ρ=−0.718 (Heston)
    """

    # Paper Table 1 / Table 2 calibrated values
    _MU_GBM    = 0.128
    _SIGMA_GBM = 0.173
    _THETA     = 0.038   # Heston long-run variance

    def __init__(self, model_dir: str = "models") -> None:
        self._pinns: dict = {}
        self._try_load(model_dir)
        if self._pinns:
            logger.info(f"PINNs loaded: {list(self._pinns.keys())}")
        else:
            logger.info("PINN models unavailable — using Merton analytical fallback")

    def _try_load(self, model_dir: str) -> None:
        try:
            from pinn.gbm    import GBMPINN
            from pinn.heston import HestonPINN
            from pinn.regime import RegimePINN
            for name, cls, fname in [
                ("gbm",    GBMPINN,    "gbm_pinn.pt"),
                ("heston", HestonPINN, "heston_pinn.pt"),
                ("regime", RegimePINN, "regime_pinn.pt"),
            ]:
                path = os.path.join(model_dir, fname)
                if os.path.exists(path):
                    self._pinns[name] = cls.load(path)
        except (ImportError, Exception) as e:
            logger.debug(f"PINN load skipped: {e}")

    def query(self, regime: str, t: float, w: float,
              vix_var: float, gamma_eff: float = CRRA_GAMMA) -> float:
        if regime == "crash":
            return 0.0

        alloc: float
        w_c   = float(np.clip(w,       0.1, 10.0))
        vix_c = float(np.clip(vix_var, 1e-4, 0.228))

        pinn_raw: float | None = None
        if self._pinns:
            try:
                if "heston" in self._pinns:
                    pinn_raw = float(self._pinns["heston"].query(t, w_c, vix_c))
                elif "regime" in self._pinns:
                    idx      = 1 if regime == "bear" else 0
                    pinn_raw = float(self._pinns["regime"].query(t, w_c, idx))
                else:
                    pinn_raw = float(self._pinns["gbm"].query(t, w_c))
            except Exception as exc:
                logger.debug(f"PINN query failed: {exc}")

        merton = self._merton_mock(vix_var, gamma_eff)

        if pinn_raw is not None and pinn_raw > 0.05:
            # PINN output is plausible — use it
            alloc = float(np.clip(pinn_raw, 0.0, 1.0))
            logger.debug(f"PINN alloc={alloc:.4f} (raw={pinn_raw:.4f})")
        else:
            # PINN output is near-zero or negative (calibration gap) → Merton fallback
            alloc = merton
            if pinn_raw is not None:
                logger.debug(f"PINN raw={pinn_raw:.4f} near-zero → Merton fallback {merton:.4f}")

        # Regime-aware γ scaling: mirrors engine.py compute_signals()
        if regime == "bull" and gamma_eff < CRRA_GAMMA:
            alloc = float(np.clip(alloc * (CRRA_GAMMA / gamma_eff), 0.0, 1.0))

        # Bear dampening: lower effective μ in uncertain regime
        if regime == "bear":
            alloc *= 0.6

        return float(np.clip(alloc, 0.0, 1.0))

    def _merton_mock(self, vix_var: float, gamma_eff: float) -> float:
        """Analytical Merton π* using paper-calibrated params + live VIX."""
        sigma_sq = max(float(vix_var), self._SIGMA_GBM ** 2)
        alloc    = (self._MU_GBM - RISK_FREE_RATE) / (gamma_eff * sigma_sq)
        return float(np.clip(alloc, 0.0, 1.0))

# ─────────────────────────────────────────────────────────────────────────────
# SignalEngine — per-stock entry/exit signals
# ─────────────────────────────────────────────────────────────────────────────

class SignalEngine:
    """
    Mirrors the signal logic from engine.py / research.py:
      · Bull: EMA20 > EMA50 and PINN π* > 0.05  → buy
      · Bull sell: EMA20 < EMA50 or PINN π* < 0.02
      · Bear primary:  RSI < 30 and EMA20 > EMA50 and PINN π* > 0.05  → buy
      · Bear recovery: RSI < 35 and price > EMA20 and PINN π* > 0.05  → buy (iter 6)
      · Crash: sell_all
      · Momentum score: 12-1 month return (ranking key)
    """

    @staticmethod
    def _rsi(series: pd.Series) -> float:
        delta = series.diff()
        gain  = delta.clip(lower=0).rolling(RSI_PERIOD).mean()
        loss  = (-delta.clip(upper=0)).rolling(RSI_PERIOD).mean()
        rs    = gain / loss
        val   = (100 - 100 / (1 + rs)).iloc[-1]
        return float(val) if np.isfinite(val) else 50.0

    @staticmethod
    def _momentum(full_series: pd.Series) -> float:
        lb = MOM_LOOKBACK_MONTHS * 21
        sk = MOM_SKIP_MONTHS     * 21
        if len(full_series) <= sk:
            return 0.0
        end_px   = float(full_series.iloc[-sk])
        start_px = float(full_series.iloc[-min(lb + sk, len(full_series))])
        return float(end_px / start_px - 1) if start_px > 0 else 0.0

    def evaluate(self, ohlcv: pd.DataFrame, regime: str,
                 pinn_alloc: float) -> dict:
        """Returns dict with signal, rsi, mom_score, sigma_est, merton."""
        if len(ohlcv) < EMA_SLOW + 5:
            return {"signal": "hold", "mom_score": 0.0, "sigma_est": 0.20, "merton": 0.0}

        close    = ohlcv["close"]
        ef       = float(close.ewm(span=EMA_FAST, adjust=False).mean().iloc[-1])
        es       = float(close.ewm(span=EMA_SLOW, adjust=False).mean().iloc[-1])
        rsi      = self._rsi(close)
        price    = float(close.iloc[-1])
        mom      = self._momentum(close)

        ret      = close.pct_change().dropna()
        sigma    = float(max(ret.std() * np.sqrt(252), 0.01))
        mu_ann   = float(ret.mean() * 252)
        merton   = float(np.clip((mu_ann - RISK_FREE_RATE) / (CRRA_GAMMA * sigma**2), 0, 1))

        if regime == "crash":
            signal = "sell_all"
        elif regime == "bull":
            if ef > es and pinn_alloc > 0.05:
                signal = "buy"
            elif ef < es or pinn_alloc < 0.02:
                signal = "sell"
            else:
                signal = "hold"
        else:  # bear
            if rsi < 30 and ef > es and pinn_alloc > 0.05:
                signal = "buy"
            elif rsi < 35 and price > ef and pinn_alloc > 0.05:
                signal = "buy"  # bear recovery (iter 6)
            elif ef < es or pinn_alloc < 0.02:
                signal = "sell"
            else:
                signal = "hold"

        return {
            "signal":    signal,
            "rsi":       rsi,
            "mom_score": mom,
            "sigma_est": sigma,
            "merton":    merton,
        }

# ─────────────────────────────────────────────────────────────────────────────
# FillEngine — 2-bps limit fill or 1.5-bps market sweep
# ─────────────────────────────────────────────────────────────────────────────

class FillEngine:
    """
    For each buy order submitted at simulation minute T:
      1. Compute limit_price = close(T) × (1 − 2bps).
      2. Inspect synthesised 1-min Lows for minutes T+1 … T+4.
      3. If any Low ≤ limit_price → passive fill at limit_price, 0 slippage.
      4. Otherwise → market fill at simulated price at T+5 + 1.5bps slippage.
    Sell orders and stop exits always execute at market (simulated price).
    """

    def __init__(self, data: MarketData, sim: IntradaySimulator) -> None:
        self._data = data
        self._sim  = sim

    def try_fill_order(self, order: PendingOrder, day: date,
                       minute_now: int) -> Fill:
        symbol = order.symbol
        bar    = self._data.day_bar(symbol, day)
        if bar is None:
            price = self._data.latest_close(symbol, day) * (1 + MARKET_SLIPPAGE_BPS / 10_000)
            return Fill(symbol, "buy", order.qty, price, order.submitted_dt,
                        "market_sweep", MARKET_SLIPPAGE_BPS)

        o, h, lp, c = bar["open"], bar["high"], bar["low"], bar["close"]
        sigma = self._data.daily_sigma(symbol, day)

        limit_px = order.limit_price

        # Inspect T+1 to T+4 for passive fill
        for m_offset in range(1, LIMIT_TIMEOUT_MIN + 1):
            m = minute_now + m_offset
            bar_low = self._sim.min_low_range(
                symbol, day, m, m, o, h, lp, c, sigma)
            if bar_low <= limit_px:
                return Fill(symbol, "buy", order.qty, limit_px,
                            order.submitted_dt, "limit_passive", 0.0)

        # Market sweep at T+5
        sweep_price = self._sim.get_price_at_minute(
            symbol, day, minute_now + LIMIT_TIMEOUT_MIN + 1, o, h, lp, c, sigma)
        sweep_price = max(sweep_price, 0.0001)
        sweep_price *= 1 + MARKET_SLIPPAGE_BPS / 10_000
        return Fill(symbol, "buy", order.qty, sweep_price,
                    order.submitted_dt, "market_sweep", MARKET_SLIPPAGE_BPS)

    def market_sell(self, symbol: str, qty: float, day: date,
                    minute: int, reason: str) -> Fill:
        bar = self._data.day_bar(symbol, day)
        if bar is not None:
            sigma = self._data.daily_sigma(symbol, day)
            price = self._sim.get_price_at_minute(
                symbol, day, minute, bar["open"], bar["high"], bar["low"], bar["close"], sigma)
        else:
            price = self._data.latest_close(symbol, day)

        price = max(price, 0.0001)
        dt    = datetime.combine(day, MARKET_OPEN) + timedelta(minutes=minute)
        return Fill(symbol, "sell", qty, price, dt, reason, 0.0)

# ─────────────────────────────────────────────────────────────────────────────
# Portfolio — positions, cash, equity tracking
# ─────────────────────────────────────────────────────────────────────────────

class Portfolio:
    def __init__(self, initial_cash: float) -> None:
        self.cash                 = initial_cash
        self.positions: dict[str, Position] = {}
        self._equity_history: deque[float]  = deque(maxlen=VOL_LOOKBACK_DAYS + 1)
        self._last_equity_date: Optional[date] = None
        self.fills: list[Fill]    = []
        self.equity_curve: list[tuple[date, float]] = []

    def apply_fill(self, fill: Fill) -> None:
        self.fills.append(fill)
        if fill.side == "buy":
            cost = fill.qty * fill.price
            if cost > self.cash:
                fill.qty  = self.cash / fill.price if fill.price > 0 else 0
                cost      = fill.qty * fill.price
            self.cash -= cost
            sym = fill.symbol
            if sym in self.positions:
                pos = self.positions[sym]
                total_shares = pos.shares + fill.qty
                pos.avg_price = (pos.avg_price * pos.shares + fill.price * fill.qty) / total_shares if total_shares > 0 else fill.price
                pos.shares    = total_shares
                pos.peak_price = max(pos.peak_price, fill.price)
            else:
                self.positions[sym] = Position(
                    symbol=sym, shares=fill.qty, avg_price=fill.price,
                    peak_price=fill.price, entry_dt=fill.dt,
                )
        else:  # sell
            sym = fill.symbol
            if sym in self.positions:
                pos = self.positions[sym]
                qty = min(fill.qty, pos.shares)
                self.cash += qty * fill.price
                pos.shares -= qty
                if pos.shares < 1e-6:
                    del self.positions[sym]

    def mark_to_market(self, prices: dict[str, float]) -> float:
        port = self.cash
        for sym, pos in self.positions.items():
            p = prices.get(sym, pos.avg_price)
            pos.peak_price = max(pos.peak_price, p)
            port += pos.shares * p
        return port

    def record_equity(self, equity: float, today: date) -> None:
        if self._last_equity_date != today:
            self._equity_history.append(equity)
            self._last_equity_date = today

    def vol_scale(self) -> float:
        if len(self._equity_history) < 5:
            return 1.0
        vals = np.array(self._equity_history, dtype=float)
        rets = np.diff(vals) / np.maximum(vals[:-1], 1e-6)
        vol  = float(np.std(rets) * np.sqrt(252))
        if vol < 1e-6:
            return float(VOL_SCALE_MAX)
        return float(np.clip(VOL_TARGET / vol, VOL_SCALE_MIN, VOL_SCALE_MAX))

# ─────────────────────────────────────────────────────────────────────────────
# RiskManager — sizing, stops, sector cap
# ─────────────────────────────────────────────────────────────────────────────

class RiskManager:
    """
    Implements the full institutional sizing stack:
      1. PINN π* (regime-aware γ already applied by PINNAllocator)
      2. Volatility targeting (20-day realized vol → scale factor)
      3. Inverse-volatility weighting across slots
      4. GICS sector cap (max 3 per sector)
      5. Position caps (10% equity, 90% cash)
      6. Hysteresis buffer (1.5% equity minimum)
    """

    def __init__(self) -> None:
        self._peak_equity = INITIAL_CAPITAL

    def update_peak(self, equity: float) -> None:
        self._peak_equity = max(self._peak_equity, equity)

    def reset_peak(self, equity: float) -> None:
        """
        Reset peak to current equity after a forced full liquidation.
        Prevents the drawdown circuit breaker from permanently blocking trading
        when all-cash equity is compared against a historical peak it can never
        recover to without trading. In production the bot restarts from current
        equity after a human intervention — this mirrors that behaviour.
        """
        self._peak_equity = equity

    def check_circuit_breaker(self, equity: float,
                               daily_start: float) -> bool:
        daily_loss = (equity - daily_start) / daily_start if daily_start > 0 else 0
        drawdown   = (equity - self._peak_equity) / self._peak_equity if self._peak_equity > 0 else 0
        return daily_loss < -0.05 or drawdown < -0.10

    def should_exit(self, pos: Position, current_price: float
                    ) -> tuple[bool, str]:
        if current_price <= 0:
            return False, ""
        pnl       = (current_price - pos.avg_price) / pos.avg_price
        from_peak = (current_price - pos.peak_price) / pos.peak_price
        if pnl <= -STOP_LOSS_PCT:
            return True, f"stop_loss({pnl:.2%})"
        if TRAILING_STOP_PCT > 0 and from_peak <= -TRAILING_STOP_PCT:
            return True, f"trailing_stop({from_peak:.2%} from peak)"
        return False, ""

    @staticmethod
    def sector_cap_filter(candidates: list[dict]) -> list[dict]:
        counts: dict[str, int] = {}
        filtered = []
        for c in candidates:
            sector = SECTOR_MAP.get(c["symbol"], f"Unknown:{c['symbol']}")
            if counts.get(sector, 0) < SECTOR_MAX_POS:
                filtered.append(c)
                counts[sector] = counts.get(sector, 0) + 1
        return filtered

    def size_order(self, symbol: str, price: float,
                   pinn_alloc: float, vol_scale: float,
                   inv_vol_weight: float, all_inv_vol_sum: float,
                   equity: float, cash: float) -> float:
        if price <= 0 or inv_vol_weight <= 0 or all_inv_vol_sum <= 0:
            return 0.0

        scaled_alloc     = float(np.clip(pinn_alloc * vol_scale, 0.0, 1.0))
        total_deployable = equity * scaled_alloc
        weight           = inv_vol_weight / all_inv_vol_sum
        target_value     = total_deployable * weight

        max_by_equity  = equity * MAX_POS_WEIGHT
        max_by_cash    = cash   * (1.0 - CASH_RESERVE_PCT)
        position_value = min(target_value, max_by_equity, max_by_cash)

        if position_value < equity * MIN_POS_WEIGHT:
            return 0.0

        return max(round(position_value / price, 4), 0.0)

# ─────────────────────────────────────────────────────────────────────────────
# BacktestEngine — main event-driven loop
# ─────────────────────────────────────────────────────────────────────────────

class BacktestEngine:
    """
    Chronological event loop stepping through market hours at 15-min (bear) or
    2-h (calm bull) intervals.  Between major cycles, a 5-min sub-loop scans
    only stop-loss and trailing-stop conditions.

    All components access data via `MarketData.ohlcv(symbol, as_of=T.date())`
    which is strictly gated to information available at simulation time T.
    """

    def __init__(self, universe: list[str], start: date, end: date,
                 model_dir: str = "models") -> None:
        self.universe  = universe
        self.start     = start
        self.end       = end

        logger.info("Initialising components …")
        self._data     = MarketData(universe, start, end)
        self._sim      = IntradaySimulator()
        self._regime   = RegimeDetector(self._data)
        self._pinn     = PINNAllocator(model_dir)
        self._signal   = SignalEngine()
        self._fill     = FillEngine(self._data, self._sim)
        self._risk     = RiskManager()
        self._port     = Portfolio(INITIAL_CAPITAL)

        self._daily_start_equity = INITIAL_CAPITAL
        self._circuit_open       = False

    # ── Intraday helpers ─────────────────────────────────────────────────────

    def _prices_now(self, day: date, minute: int,
                    pos_only: bool = False) -> dict[str, float]:
        """
        Intraday price snapshot at `minute` minutes after market open.
        pos_only=True: only prices for held positions (used in exit scans).
        pos_only=False: held positions + full universe (used in major cycles).
        """
        syms = list(self._port.positions.keys())
        if not pos_only:
            held = set(syms)
            syms += [s for s in self.universe if s not in held]
        prices = {}
        for sym in syms:
            bar = self._data.day_bar(sym, day)
            if bar is None:
                prices[sym] = self._data.latest_close(sym, day)
                continue
            sigma = self._data.daily_sigma(sym, day)
            prices[sym] = self._sim.get_price_at_minute(
                sym, day, minute,
                bar["open"], bar["high"], bar["low"], bar["close"], sigma)
        return prices

    def _time_to_horizon(self, today: date) -> float:
        year_end = date(today.year, 12, 31)
        days_rem = max((year_end - today).days, 1)
        return min(days_rem / 252.0, TRADING_HORIZON_T)

    # ── Exit scan (5-min sub-loop) ────────────────────────────────────────────

    def _run_exit_scan(self, day: date, minute: int) -> None:
        if not self._port.positions:
            return
        prices = self._prices_now(day, minute, pos_only=True)
        for sym in list(self._port.positions.keys()):
            pos = self._port.positions.get(sym)
            if pos is None:
                continue
            p = prices.get(sym, pos.avg_price)
            pos.peak_price = max(pos.peak_price, p)
            triggered, reason = self._risk.should_exit(pos, p)
            if triggered:
                fill = self._fill.market_sell(sym, pos.shares, day, minute, reason)
                self._port.apply_fill(fill)
                logger.debug(f"  EXIT {sym} [{reason}] @ {fill.price:.2f}")

    # ── Sweep pending limit orders ────────────────────────────────────────────

    def _sweep_pending_orders(self, day: date, minute: int) -> None:
        remaining = []
        for order in self._pending_orders:
            fill = self._fill.try_fill_order(order, day, minute)
            self._port.apply_fill(fill)
            logger.debug(
                f"  FILL {fill.symbol} {fill.fill_type} "
                f"qty={fill.qty:.3f} @ {fill.price:.2f} ({fill.cost_bps:.1f}bps)"
            )
        self._pending_orders = remaining   # all orders resolved in one sweep

    # ── Major allocation cycle ────────────────────────────────────────────────

    def _run_major_cycle(self, day: date, minute: int) -> str:
        """
        Full allocation logic: regime → PINN → signals → sector cap → sizing → orders.
        Returns confirmed regime label for interval scheduling.
        """
        equity = self._port.mark_to_market(self._prices_now(day, minute))
        self._port.record_equity(equity, day)
        # Peak is updated from EOD equity only (see run() below) — not intraday.
        # Intraday Brownian-bridge marks can briefly exceed daily closes, which
        # would set an artificially high peak and trigger the drawdown circuit
        # breaker permanently. EOD-only tracking matches production behaviour.

        # Circuit breaker
        if self._risk.check_circuit_breaker(equity, self._daily_start_equity):
            if not self._circuit_open:
                logger.warning(f"  CIRCUIT BREAKER @ {equity:,.0f}")
                self._circuit_open = True
                for sym in list(self._port.positions.keys()):
                    pos = self._port.positions.get(sym)
                    if pos:
                        fill = self._fill.market_sell(sym, pos.shares, day, minute, "circuit_breaker")
                        self._port.apply_fill(fill)
                # Reset peak to post-liquidation cash so the drawdown CB doesn't
                # permanently block trading. Mirrors production: bot restarts from
                # current equity after a human intervention, not from the all-time high.
                self._risk.reset_peak(float(self._port.cash))
            return "bear"

        self._circuit_open = False

        # Regime + VIX
        regime  = self._regime.detect(day)
        vix_val = self._data.vix(day)
        vix_var = (vix_val / 100.0) ** 2

        # SPY hard crash backstop
        spy_s = self._data.close_series("SPY", day)
        if len(spy_s) >= 2:
            spy_ret = (float(spy_s.iloc[-1]) - float(spy_s.iloc[-2])) / float(spy_s.iloc[-2])
            if spy_ret <= -0.03:
                regime = "crash"

        # Crash: close all positions at market
        if regime == "crash":
            for sym in list(self._port.positions.keys()):
                pos = self._port.positions.get(sym)
                if pos:
                    fill = self._fill.market_sell(sym, pos.shares, day, minute, "crash_exit")
                    self._port.apply_fill(fill)
            return "crash"

        # Exit scan for existing positions
        self._run_exit_scan(day, minute)

        # γ-aware PINN allocation
        gamma_eff = CRRA_GAMMA_BULL if regime == "bull" else CRRA_GAMMA
        t_hor     = self._time_to_horizon(day)
        w_norm    = max(equity / INITIAL_CAPITAL, 0.1)
        pinn_alloc = self._pinn.query(regime, t_hor, w_norm, vix_var, gamma_eff)

        # Skip buys if max positions reached
        n_open = len(self._port.positions)
        if n_open >= MAX_POSITIONS:
            return regime

        # Scan universe for buy signals
        candidates: list[dict] = []
        for sym in self.universe:
            if sym in self._port.positions:
                continue
            ohlcv = self._data.ohlcv(sym, day)
            if len(ohlcv) < EMA_SLOW + 5:
                continue
            sig = self._signal.evaluate(ohlcv, regime, pinn_alloc)
            if sig["signal"] == "buy":
                candidates.append({
                    "symbol":   sym,
                    "price":    self._data.latest_close(sym, day),
                    "mom":      sig["mom_score"],
                    "sigma":    sig["sigma_est"],
                    "merton":   sig["merton"],
                })

        # Sort by momentum, apply sector cap
        candidates.sort(key=lambda x: x["mom"], reverse=True)
        candidates = self._risk.sector_cap_filter(candidates)

        # Fill open slots
        slots   = MAX_POSITIONS - n_open
        to_buy  = candidates[:slots]
        if not to_buy:
            return regime

        vol_sc       = self._port.vol_scale()
        inv_vol_sum  = sum(1.0 / c["sigma"] for c in to_buy)
        if inv_vol_sum <= 0:
            return regime

        equity = self._port.mark_to_market(self._prices_now(day, minute))

        for c in to_buy:
            price = c["price"]
            if price <= 0:
                continue
            inv_vol_w = 1.0 / c["sigma"]
            qty = self._risk.size_order(
                c["symbol"], price, pinn_alloc, vol_sc,
                inv_vol_w, inv_vol_sum, equity, self._port.cash,
            )
            if qty <= 0:
                continue
            limit_px = round(price * (1 - LIMIT_OFFSET_BPS / 10_000), 4)
            submitted_dt = datetime.combine(day, MARKET_OPEN) + timedelta(minutes=minute)
            order = PendingOrder(c["symbol"], qty, limit_px, submitted_dt)
            # Attempt fill immediately (sweeps within this cycle's minute window)
            fill = self._fill.try_fill_order(order, day, minute)
            self._port.apply_fill(fill)
            logger.debug(
                f"  BUY {c['symbol']} qty={fill.qty:.3f} @ {fill.price:.2f} "
                f"[{fill.fill_type}] mom={c['mom']:+.2f}"
            )

        return regime

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self) -> "BacktestResults":
        logger.info(f"Backtest: {self.start} → {self.end}  |  universe={len(self.universe)}")

        trading_days = self._data.trading_days(self.start, self.end)
        if not trading_days:
            raise RuntimeError("No trading days in the specified range.")

        regime   = "bear"
        equity_curve: list[tuple[date, float]] = []
        last_log_day: Optional[date] = None

        for day in trading_days:
            # Daily reset
            self._daily_start_equity = self._port.mark_to_market(
                self._prices_now(day, 0))
            self._circuit_open = False

            # Dynamic interval: updated after every major cycle based on detected regime.
            # Starting at 0 (market open); next_major_minute advances by the detected interval.
            # This correctly handles mid-day regime flips (e.g. bear→bull cuts 15-min→2h).
            next_major_minute = 0

            for minute in range(0, 390, STOP_SCAN_MIN):
                if minute >= next_major_minute:
                    # ── Major allocation cycle ──────────────────────────────
                    regime   = self._run_major_cycle(day, minute)
                    vix_val  = self._data.vix(day)
                    interval = (CYCLE_BULL_MIN
                                if regime == "bull" and vix_val < VIX_HIGH_THRESHOLD
                                else CYCLE_BEAR_MIN)
                    # Always run a final major cycle near close regardless of interval
                    next_major_minute = min(minute + interval, 385) \
                                        if minute < 380 else 999
                else:
                    # ── 5-min exit sub-step ─────────────────────────────────
                    self._run_exit_scan(day, minute)

            # Record end-of-day equity and update peak from EOD close only.
            # Using EOD (not intraday) prevents Brownian-bridge noise from
            # pushing the peak above the real close and tripping the drawdown
            # circuit breaker permanently on a subsequent down day.
            eod_prices = self._prices_now(day, 385)
            equity     = self._port.mark_to_market(eod_prices)
            self._port.record_equity(equity, day)
            self._risk.update_peak(equity)      # EOD-only peak update
            equity_curve.append((day, equity))

            if last_log_day is None or (day - last_log_day).days >= 30:
                n_pos = len(self._port.positions)
                logger.info(
                    f"  {day}  equity=${equity:>12,.0f}  "
                    f"regime={regime:<5}  positions={n_pos:>2}  "
                    f"VIX={vix_val:.1f}"
                )
                last_log_day = day

        return BacktestResults(
            equity_curve  = equity_curve,
            fills         = self._port.fills,
            universe      = self.universe,
            start         = self.start,
            end           = self.end,
            data          = self._data,
        )

# ─────────────────────────────────────────────────────────────────────────────
# BacktestResults — metrics and visualisation
# ─────────────────────────────────────────────────────────────────────────────

class BacktestResults:
    def __init__(self, equity_curve: list[tuple[date, float]],
                 fills: list[Fill], universe: list[str],
                 start: date, end: date, data: MarketData) -> None:
        self.equity_curve = equity_curve
        self.fills        = fills
        self.universe     = universe
        self.start        = start
        self.end          = end
        self._data        = data

    # ── Metric helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _cagr(vals: np.ndarray, n_days: int) -> float:
        if vals[0] <= 0 or n_days < 1:
            return 0.0
        return float((vals[-1] / vals[0]) ** (252 / n_days) - 1)

    @staticmethod
    def _sharpe(rets: np.ndarray, rf: float = 0.05 / 252) -> float:
        exc = rets - rf
        return float(exc.mean() / exc.std() * np.sqrt(252)) if exc.std() > 0 else 0.0

    @staticmethod
    def _sortino(rets: np.ndarray, rf: float = 0.05 / 252) -> float:
        exc = rets - rf
        dd  = exc[exc < 0]
        return float(exc.mean() / dd.std() * np.sqrt(252)) if len(dd) > 0 and dd.std() > 0 else 0.0

    @staticmethod
    def _max_dd(vals: np.ndarray) -> float:
        peak = np.maximum.accumulate(vals)
        return float(((vals - peak) / np.maximum(peak, 1e-9)).min())

    def _turnover(self) -> float:
        if not self.fills or not self.equity_curve:
            return 0.0
        total_traded = sum(abs(f.qty * f.price) for f in self.fills)
        avg_equity   = float(np.mean([e for _, e in self.equity_curve]))
        n_years      = (self.end - self.start).days / 365.25
        return total_traded / avg_equity / n_years if avg_equity > 0 and n_years > 0 else 0.0

    def _spy_curve(self) -> np.ndarray:
        spy_vals = []
        spy_shares: Optional[float] = None
        for day, _ in self.equity_curve:
            p = self._data.latest_close("SPY", day)
            if spy_shares is None and p > 0:
                spy_shares = INITIAL_CAPITAL / p
            spy_vals.append((spy_shares or 0) * p)
        return np.array(spy_vals)

    # ── Public report ─────────────────────────────────────────────────────────

    def report(self) -> dict:
        vals   = np.array([v for _, v in self.equity_curve])
        rets   = np.diff(vals) / np.maximum(vals[:-1], 1e-9)
        n_days = len(vals) - 1

        spy      = self._spy_curve()
        spy_rets = np.diff(spy) / np.maximum(spy[:-1], 1e-9)

        # Count fills by type
        passive_fills = sum(1 for f in self.fills if f.fill_type == "limit_passive")
        sweep_fills   = sum(1 for f in self.fills if f.fill_type == "market_sweep")
        total_fills   = len(self.fills)
        avg_cost_bps  = float(np.mean([f.cost_bps for f in self.fills])) if self.fills else 0.0

        # Annual returns (strategy vs SPY)
        dates_idx  = pd.DatetimeIndex([d for d, _ in self.equity_curve])
        ann_strat: dict[int, float] = {}
        ann_spy:   dict[int, float] = {}
        for yr in sorted(set(dates_idx.year)):
            mask = dates_idx.year == yr
            sv, bv = vals[mask], spy[mask]
            if len(sv) > 1:
                ann_strat[yr] = float(sv[-1] / sv[0] - 1) * 100
                ann_spy[yr]   = float(bv[-1] / bv[0] - 1) * 100

        m = {
            "period":            f"{self.start} → {self.end}",
            "final_equity":      float(vals[-1]),
            "total_return_pct":  float(vals[-1] / vals[0] - 1) * 100,
            "CAGR_pct":          self._cagr(vals, n_days) * 100,
            "Sharpe":            self._sharpe(rets),
            "Sortino":           self._sortino(rets),
            "MaxDD_pct":         self._max_dd(vals) * 100,
            "Calmar":            self._cagr(vals, n_days) / abs(self._max_dd(vals)) if self._max_dd(vals) != 0 else 0.0,
            "annual_turnover":   self._turnover(),
            "total_fills":       total_fills,
            "passive_fills":     passive_fills,
            "sweep_fills":       sweep_fills,
            "avg_fill_cost_bps": avg_cost_bps,
            "spy_CAGR_pct":      self._cagr(spy, n_days) * 100,
            "spy_Sharpe":        self._sharpe(spy_rets),
            "spy_MaxDD_pct":     self._max_dd(spy) * 100,
            "annual_strat":      ann_strat,
            "annual_spy":        ann_spy,
        }
        return m

    def print_report(self) -> None:
        m = self.report()
        sep = "─" * 58
        print(f"\n{'═'*58}")
        print(f"  BACKTEST RESULTS  {m['period']}")
        print(f"{'═'*58}")
        print(f"  Final Equity      ${m['final_equity']:>14,.2f}")
        print(f"  Total Return      {m['total_return_pct']:>+.1f}%")
        print(f"  CAGR              {m['CAGR_pct']:>+.2f}%")
        print(f"  Sharpe            {m['Sharpe']:>7.3f}")
        print(f"  Sortino           {m['Sortino']:>7.3f}")
        print(f"  Max Drawdown      {m['MaxDD_pct']:>7.1f}%")
        print(f"  Calmar            {m['Calmar']:>7.3f}")
        print(f"  Annual Turnover   {m['annual_turnover']:>7.2f}×")
        print(sep)
        print(f"  SPY B&H CAGR      {m['spy_CAGR_pct']:>+.2f}%")
        print(f"  SPY Sharpe        {m['spy_Sharpe']:>7.3f}")
        print(f"  SPY Max Drawdown  {m['spy_MaxDD_pct']:>7.1f}%")
        print(sep)
        print(f"  Fills: {m['total_fills']} total  "
              f"({m['passive_fills']} passive, {m['sweep_fills']} sweep)  "
              f"avg {m['avg_fill_cost_bps']:.2f}bps")
        print(sep)
        print("  Annual Returns  (Strategy vs SPY):")
        for yr in sorted(m["annual_strat"].keys()):
            s = m["annual_strat"].get(yr, 0.0)
            b = m["annual_spy"].get(yr, 0.0)
            flag = "▲" if s >= b else "▼"
            print(f"    {yr}:  {s:>+6.1f}%  vs SPY {b:>+6.1f}%  {flag}")
        print(f"{'═'*58}\n")

    def plot(self, filename: str = "backtest_engine_results.png") -> str:
        dates = [d for d, _ in self.equity_curve]
        vals  = np.array([v for _, v in self.equity_curve])
        spy   = self._spy_curve()

        BG, AX = "#0F172A", "#1E293B"
        MUTED, TEXT = "#94A3B8", "#F1F5F9"
        BLUE, GRAY, GREEN, RED = "#60A5FA", "#64748B", "#34D399", "#F87171"

        fig, axes = plt.subplots(2, 2, figsize=(16, 10), facecolor=BG)
        fig.suptitle("Event-Driven PINN Backtest Results", fontsize=14,
                     color=TEXT, y=0.99)
        plt.rcParams.update({
            "text.color": TEXT, "axes.labelcolor": MUTED,
            "xtick.color": MUTED, "ytick.color": MUTED,
        })

        # 1. Equity curve
        ax = axes[0, 0]
        ax.set_facecolor(AX)
        norm_v   = vals / vals[0] * 100
        norm_spy = spy  / spy[0]  * 100
        ax.plot(dates, norm_v,   color=BLUE, lw=1.6, label="Strategy")
        ax.plot(dates, norm_spy, color=GRAY, lw=1.0, ls="--", label="SPY B&H")
        ax.axhline(100, color="#475569", lw=0.4, ls=":")
        ax.set_title("Equity Curve (Normalised to 100)", color=TEXT)
        ax.set_ylabel("Value", color=MUTED)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

        # 2. Drawdown
        ax = axes[0, 1]
        ax.set_facecolor(AX)
        peak = np.maximum.accumulate(vals)
        dd   = (vals - peak) / np.maximum(peak, 1e-9) * 100
        spy_peak = np.maximum.accumulate(spy)
        spy_dd   = (spy - spy_peak) / np.maximum(spy_peak, 1e-9) * 100
        ax.fill_between(dates, dd,     alpha=0.5, color=RED,  label="Strategy DD")
        ax.fill_between(dates, spy_dd, alpha=0.3, color=GRAY, label="SPY DD")
        ax.set_title("Drawdown %", color=TEXT)
        ax.set_ylabel("%", color=MUTED)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

        # 3. Annual returns
        ax = axes[1, 0]
        ax.set_facecolor(AX)
        idx = pd.DatetimeIndex(dates)
        ann_s, ann_b = {}, {}
        for yr in sorted(set(idx.year)):
            mask = idx.year == yr
            sv, bv = vals[mask], spy[mask]
            if len(sv) > 1:
                ann_s[yr] = float(sv[-1] / sv[0] - 1) * 100
                ann_b[yr] = float(bv[-1] / bv[0] - 1) * 100
        years = sorted(ann_s.keys())
        x     = np.arange(len(years))
        w     = 0.35
        bars_s = [ann_s.get(y, 0) for y in years]
        bars_b = [ann_b.get(y, 0) for y in years]
        ax.bar(x - w/2, bars_s, w, color=[GREEN if v >= 0 else RED for v in bars_s], alpha=0.8, label="Strategy")
        ax.bar(x + w/2, bars_b, w, color=GRAY, alpha=0.5, label="SPY")
        ax.set_xticks(x)
        ax.set_xticklabels([str(y) for y in years], rotation=45, fontsize=7)
        ax.axhline(0, color="#475569", lw=0.5)
        ax.set_title("Annual Returns %", color=TEXT)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

        # 4. Fill cost distribution
        ax = axes[1, 1]
        ax.set_facecolor(AX)
        if self.fills:
            passive_n = sum(1 for f in self.fills if f.fill_type == "limit_passive")
            sweep_n   = sum(1 for f in self.fills if f.fill_type == "market_sweep")
            ax.bar(["Passive\n(limit, 0bps)", "Sweep\n(+1.5bps)"],
                   [passive_n, sweep_n],
                   color=[GREEN, RED], alpha=0.8)
            ax.set_title("Fill Distribution", color=TEXT)
            ax.set_ylabel("Count", color=MUTED)
        else:
            ax.text(0.5, 0.5, "No fills", ha="center", va="center",
                    transform=ax.transAxes, color=MUTED)
        ax.grid(True, alpha=0.3, axis="y")

        fig.tight_layout()
        path = os.path.join(GRAPHS_DIR, filename)
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
        plt.close(fig)
        logger.info(f"Chart saved → {path}")
        return path

# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.remove()
    logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | {message}", level="INFO")

    args  = sys.argv[1:]
    start = date.fromisoformat(args[0]) if len(args) >= 1 else date(2017, 1, 1)
    end   = date.fromisoformat(args[1]) if len(args) >= 2 else date(2025, 6, 30)

    engine  = BacktestEngine(DEFAULT_UNIVERSE, start, end)
    results = engine.run()
    results.print_report()
    results.plot()


if __name__ == "__main__":
    main()
