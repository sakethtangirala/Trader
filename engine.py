"""
Market data (OpenBB) + HMM regime detection + PINN-based optimal allocation.

Architecture: PINN as risk multiplier on a momentum-first stock selection layer.
  · Stock selection  — 12-1 month momentum ranks buy candidates (cross-sectional)
  · Risk scaling     — PINN π*(t,w,y) scales total equity exposure (0→1 multiplier)
  · Position sizing  — inverse-vol weights (1/σ_i / Σ1/σ_j) distribute PINN exposure across ranked candidates
  · Exits            — trailing stop (-10% from peak) + EMA trend reversal

Allocation hierarchy (most → least sophisticated):
  1. Heston PINN   — queries π*(t, w, y) using live VIX as variance proxy
  2. Regime PINN   — queries π*(t, w) for the HMM-detected regime
  3. GBM PINN      — queries π*(t, w) for current wealth/horizon
  4. Merton ratio  — analytical π* = (μ-r)/(γσ²), per-stock rolling estimates
  5. EMA rules     — trend-confirmation filter alongside PINN
"""
from __future__ import annotations
import os
import pickle
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from openbb import obb
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from loguru import logger

from settings import (
    HMM_N_STATES, HMM_LOOKBACK_DAYS, HMM_MODEL_PATH,
    RSI_PERIOD, EMA_FAST, EMA_SLOW, VOLUME_MA_PERIOD,
    CRRA_GAMMA, CRRA_GAMMA_BULL, RISK_FREE_RATE, RISK_FREE_RATE_DYNAMIC,
    TRADING_HORIZON_YEARS,
    GBM_PINN_PATH, HESTON_PINN_PATH, REGIME_PINN_PATH,
    INTRADAY_MAX_STALE_MIN,
)



# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def fetch_ohlcv(symbol: str, lookback_days: int = HMM_LOOKBACK_DAYS) -> pd.DataFrame:
    end   = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    try:
        result = obb.equity.price.historical(
            symbol=symbol, start_date=start, end_date=end,
            interval="1d", provider="yfinance",
        )
        df = result.to_df()
        df.index = pd.to_datetime(df.index)
        df.columns = [c.lower() for c in df.columns]
        return df.sort_index()
    except Exception:
        try:
            result = obb.index.price.historical(
                symbol=symbol, start_date=start, end_date=end,
                interval="1d", provider="yfinance",
            )
            df = result.to_df()
            df.index = pd.to_datetime(df.index)
            df.columns = [c.lower() for c in df.columns]
            return df.sort_index()
        except Exception as e:
            logger.error(f"fetch_ohlcv({symbol}): {e}")
            return pd.DataFrame()


def fetch_vix() -> float:
    """Return current VIX² as Heston variance proxy y_t = (VIX/100)²."""
    try:
        df = fetch_ohlcv("^VIX", lookback_days=5)
        if df.empty:
            return 0.041          # paper's calibrated θ as fallback
        vix = float(df["close"].iloc[-1])
        return (vix / 100.0) ** 2
    except Exception:
        return 0.041


# ---------------------------------------------------------------------------
# Intraday microstructure — 1-minute data, Garman-Klass, lead-lag
# ---------------------------------------------------------------------------

def fetch_1min(symbol: str, bars: int = 30) -> pd.DataFrame:
    """
    1-minute OHLCV bars for today's session (free via yfinance, up to 7 days back).
    Returns the last `bars` bars with forward-filled gaps, or empty on failure.

    NOTE: yfinance 1-min data can be delayed several minutes — always pass through
    check_1min_freshness() before using for lead-lag front-running.
    """
    try:
        import yfinance as yf
        df = yf.Ticker(symbol).history(period="1d", interval="1m")
        if df.empty:
            return pd.DataFrame()
        df.columns = [c.lower() for c in df.columns]
        df = df.tail(bars)
        # Forward-fill missing bars to prevent dimension mismatches in GK/lead-lag
        df = df.ffill().dropna(subset=["open", "high", "low", "close"])
        return df
    except Exception as e:
        logger.warning(f"fetch_1min({symbol}): {e}")
        return pd.DataFrame()


def check_1min_freshness(df_1min: pd.DataFrame,
                          max_stale_min: int = INTRADAY_MAX_STALE_MIN) -> bool:
    """
    Returns True only if the latest 1-min bar is within max_stale_min of now.
    yfinance intraday can lag by several minutes; stale bars produce aliased
    lead-lag signals that act on yesterday's momentum, not current microstructure.
    """
    if df_1min.empty:
        return False
    try:
        last_bar = df_1min.index[-1]
        if last_bar.tzinfo is None:
            last_bar = last_bar.tz_localize("America/New_York")
        now_et    = pd.Timestamp.now(tz="America/New_York")
        stale_min = (now_et - last_bar).total_seconds() / 60.0
        if stale_min > max_stale_min:
            logger.warning(
                f"1-min data stale by {stale_min:.1f} min "
                f"(>{max_stale_min} min) — skipping lead-lag"
            )
            return False
        return True
    except Exception:
        return True  # can't determine staleness; allow through


def garman_klass_vol(df: pd.DataFrame) -> np.ndarray:
    """
    Garman-Klass extreme-value volatility estimator per bar:
      GK = 0.5·(ln H/L)² − (2ln2 − 1)·(ln C/O)²

    Returns a 1-D float array of per-bar GK values (variance units).
    """
    h       = df["high"].values.astype(float)
    low_arr = df["low"].values.astype(float)
    c       = df["close"].values.astype(float)
    o       = df["open"].values.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        hl = np.log(np.where((low_arr > 0) & (h >= low_arr), h / low_arr, 1.0))
        co = np.log(np.where(o > 0, c / o, 1.0))
    return 0.5 * hl ** 2 - (2.0 * np.log(2.0) - 1.0) * co ** 2


def check_vol_acceleration(df_1min: pd.DataFrame,
                            lookback: int = 15,
                            accel_threshold: float = 3.0) -> bool:
    """
    Realized Volatility Signature tracker.

    Returns True (micro-regime safety halt) when the first derivative of
    1-minute GK volatility is accelerating exponentially within the window:
      · Positive slope on Δ(GK) series  — direction of acceleration
      · Latest GK > accel_threshold × rolling mean  — magnitude of spike

    When True, overrides the macro HMM signal regardless of regime.
    """
    if df_1min.empty or len(df_1min) < max(lookback, 5):
        return False

    gk  = garman_klass_vol(df_1min.tail(lookback))
    dgk = np.diff(gk)

    if len(dgk) < 3 or not np.isfinite(gk).all():
        return False

    slope   = float(np.polyfit(np.arange(len(dgk)), dgk, 1)[0])
    gk_mean = float(gk.mean())
    ratio   = float(gk[-1]) / max(gk_mean, 1e-12)

    accelerating = (slope > 0) and (ratio > accel_threshold)
    if accelerating:
        logger.warning(
            f"Micro-regime HALT: GK slope={slope:.3e}  "
            f"last/mean={ratio:.2f}× (>{accel_threshold}×)"
        )
    return accelerating


def lead_lag_correlation(sym_1min: pd.DataFrame,
                          spy_1min: pd.DataFrame,
                          max_lag: int = 5) -> dict:
    """
    Rolling 1-minute lead-lag cross-correlation between SPY and a stock.

    Positive lag  → SPY leads the stock (momentum front-running signal).
    Negative lag  → stock leads SPY.

    Returns dict with keys: optimal_lag (int), peak_corr (float in [-1, 1]).
    """
    if sym_1min.empty or spy_1min.empty:
        return {"optimal_lag": 0, "peak_corr": 0.0}

    sym_ret = sym_1min["close"].pct_change().dropna()
    spy_ret = spy_1min["close"].pct_change().dropna()
    common  = sym_ret.index.intersection(spy_ret.index)

    if len(common) < max_lag + 5:
        return {"optimal_lag": 0, "peak_corr": 0.0}

    s = sym_ret.loc[common].values
    p = spy_ret.loc[common].values
    n = len(s)

    best_lag, best_corr = 0, 0.0
    for lag in range(-max_lag, max_lag + 1):
        try:
            a = p[:n - lag] if lag > 0 else (p[-lag:] if lag < 0 else p)
            b = s[lag:]     if lag > 0 else (s[:n + lag] if lag < 0 else s)
            if len(a) < 5:
                continue
            corr = float(np.corrcoef(a, b)[0, 1])
            if np.isfinite(corr) and abs(corr) > abs(best_corr):
                best_corr, best_lag = corr, lag
        except Exception:
            continue

    return {"optimal_lag": best_lag, "peak_corr": round(best_corr, 4)}


# ---------------------------------------------------------------------------
# PINN loading
# ---------------------------------------------------------------------------

def load_pinns() -> dict:
    """Load trained PINN models. Returns empty dict if not yet trained."""
    try:
        from pinn.gbm    import GBMPINN
        from pinn.heston import HestonPINN
        from pinn.regime import RegimePINN
    except ImportError:
        logger.warning("PINN package not importable — run: pip install torch scipy")
        return {}

    pinns: dict = {}
    for name, cls, path in [
        ("gbm",    GBMPINN,    GBM_PINN_PATH),
        ("heston", HestonPINN, HESTON_PINN_PATH),
        ("regime", RegimePINN, REGIME_PINN_PATH),
    ]:
        try:
            pinns[name] = cls.load(path)
            logger.info(f"PINN loaded: {name.upper()} ← {path}")
        except Exception as e:
            logger.warning(f"PINN not available ({name}): {e}")

    if not pinns:
        logger.warning("No PINNs loaded — using Merton ratio + RSI/EMA fallback")
    return pinns


# ---------------------------------------------------------------------------
# Sector classification (GICS, cached per session)
# ---------------------------------------------------------------------------

_sector_cache: dict[str, str] = {}


def get_sector(symbol: str) -> str:
    """
    GICS sector string for a ticker, cached for the process lifetime.

    Called only for stocks that pass the buy signal filter (<< full universe size),
    so the yfinance round-trip cost is bounded to a small subset per cycle.
    Returns 'Unknown' on any failure — Unknown stocks are treated as their own
    singleton sector and not blocked by the sector cap.
    """
    if symbol in _sector_cache:
        return _sector_cache[symbol]
    try:
        import yfinance as yf
        info   = yf.Ticker(symbol).info
        sector = str(info.get("sector") or "Unknown")
    except Exception:
        sector = "Unknown"
    _sector_cache[symbol] = sector
    return sector


# ---------------------------------------------------------------------------
# Parameter estimation (rolling, per-stock)
# ---------------------------------------------------------------------------

def estimate_params(df: pd.DataFrame) -> dict:
    """Annualised μ and σ from rolling 90-day returns."""
    ret = df["close"].pct_change().dropna()
    mu  = float(ret.mean() * 252)
    sig = float(ret.std()  * np.sqrt(252))
    return {"mu": mu, "sigma": max(sig, 0.01)}


# Daily cache for the risk-free rate — fetched once from ^IRX, reused all day.
_rf_rate_cache: dict[str, float] = {}


def fetch_risk_free_rate() -> float:
    """
    Current annualised risk-free rate from the 3-month T-bill (^IRX via yfinance).
    Cached per calendar day — one network round-trip per trading day.
    Falls back to the static RISK_FREE_RATE (5%) on any failure.

    Corrects Merton sizing for high-rate regimes: when r=5.3% (2023), the equity
    risk premium (μ − r) shrinks, and the model should be proportionally less aggressive.
    """
    if not RISK_FREE_RATE_DYNAMIC:
        return RISK_FREE_RATE
    from datetime import date as _date
    today = str(_date.today())
    if today in _rf_rate_cache:
        return _rf_rate_cache[today]
    try:
        import yfinance as yf
        hist = yf.Ticker("^IRX").history(period="5d")
        if not hist.empty:
            rate = float(hist["Close"].iloc[-1]) / 100.0
            rate = float(np.clip(rate, 0.0, 0.20))
            _rf_rate_cache[today] = rate
            logger.info(f"Risk-free rate (3-mo T-bill): {rate:.2%}")
            return rate
    except Exception as e:
        logger.debug(f"fetch_risk_free_rate failed: {e}")
    _rf_rate_cache[today] = RISK_FREE_RATE
    return RISK_FREE_RATE


def compute_merton_ratio(mu: float, sigma: float,
                         r: float | None = None,
                         gamma: float = CRRA_GAMMA) -> float:
    """
    Closed-form Merton ratio π* = (μ-r)/(γσ²), clipped to [0, 1].
    Uses the dynamic 3-month T-bill rate when r is not supplied.
    """
    if r is None:
        r = fetch_risk_free_rate()
    if sigma <= 0:
        return 0.0
    ratio = (mu - r) / (gamma * sigma ** 2)
    return float(np.clip(ratio, 0.0, 1.0))


def compute_momentum(df: pd.DataFrame,
                     lookback_months: int = 12,
                     skip_months: int = 1) -> float:
    """
    Cross-sectional 12-1 month momentum score (Jegadeesh-Titman canonical factor).

    Returns the compounded return from (lookback_months) ago to (skip_months) ago,
    skipping the most recent month to avoid short-term reversal bias.
    Gracefully degrades to whatever history is available when df is short.
    """
    close = df["close"].dropna()
    trading_days_per_month = 21
    lookback_days = lookback_months * trading_days_per_month
    skip_days     = skip_months     * trading_days_per_month

    if len(close) <= skip_days:
        return 0.0

    end_px   = float(close.iloc[-skip_days])
    start_px = float(close.iloc[-min(lookback_days + skip_days, len(close))])
    if start_px <= 0:
        return 0.0
    return float(end_px / start_px - 1)


def time_to_horizon() -> float:
    """Remaining fraction of the TRADING_HORIZON_YEARS window (in years)."""
    now      = datetime.now()
    year_end = datetime(now.year, 12, 31)
    days_rem = max((year_end - now).days, 1)
    return min(days_rem / 252.0, TRADING_HORIZON_YEARS)


# ---------------------------------------------------------------------------
# PINN allocation (market-level)
# ---------------------------------------------------------------------------

def get_pinn_allocation(
    regime: str,
    pinns: dict,
    w_norm: float = 1.0,
    vix_var: float = 0.041,
) -> float | None:
    """
    Query the best available PINN for the market-level optimal equity fraction.
    Returns None if no PINN is loaded (signals fallback to Merton/rules).

    Priority:
      Heston (variance-aware) > Regime (coupling-aware) > GBM (baseline)
    """
    if not pinns:
        return None

    t = time_to_horizon()
    w = max(w_norm, 0.1)

    if regime == "crash":
        return 0.0  # no PINN needed; rule is deterministic

    if "heston" in pinns:
        try:
            alloc = pinns["heston"].query(t, w, vix_var)
            logger.debug(f"Heston PINN allocation: {alloc:.4f} (y={vix_var:.4f})")
            return float(np.clip(alloc, 0.0, 1.0))
        except Exception as e:
            logger.warning(f"Heston PINN query failed: {e}")

    if "regime" in pinns:
        idx = 1 if regime == "bear" else 0   # 0=risk-on(bull), 1=risk-off(bear)
        try:
            alloc = pinns["regime"].query(t, w, idx)
            logger.debug(f"Regime PINN allocation: {alloc:.4f} (regime={regime})")
            return float(np.clip(alloc, 0.0, 1.0))
        except Exception as e:
            logger.warning(f"Regime PINN query failed: {e}")

    if "gbm" in pinns:
        try:
            alloc = pinns["gbm"].query(t, w)
            logger.debug(f"GBM PINN allocation: {alloc:.4f}")
            return float(np.clip(alloc, 0.0, 1.0))
        except Exception as e:
            logger.warning(f"GBM PINN query failed: {e}")

    return None


# ---------------------------------------------------------------------------
# HMM regime detection (unchanged from v1)
# ---------------------------------------------------------------------------

def _build_features(df: pd.DataFrame):
    df = df.copy()
    df["returns"]      = df["close"].pct_change()
    df["volatility"]   = df["returns"].rolling(20).std()
    df["volume_ratio"] = df["volume"] / df["volume"].rolling(VOLUME_MA_PERIOD).mean()
    df = df.dropna(subset=["returns", "volatility", "volume_ratio"])
    return df[["returns", "volatility", "volume_ratio"]].values, df


def detect_regime(df: pd.DataFrame) -> tuple[str, GaussianHMM | None]:
    X, _ = _build_features(df)
    if len(X) < 60:
        return "bear", None

    scaler  = StandardScaler()
    X_sc    = scaler.fit_transform(X)
    model   = GaussianHMM(n_components=HMM_N_STATES, covariance_type="full",
                          n_iter=200, random_state=42)
    model.fit(X_sc)

    states  = model.predict(X_sc)
    cur     = int(states[-1])

    # Map by mean return: highest=bull, middle=bear, lowest=crash
    mean_ret = {s: X[states == s, 0].mean() for s in range(HMM_N_STATES) if (states == s).any()}
    ranked   = sorted(mean_ret, key=lambda s: mean_ret[s], reverse=True)
    labels   = ["bull", "bear", "crash"]
    regime_map = {ranked[i]: labels[i] for i in range(len(ranked))}
    regime   = regime_map.get(cur, "bear")

    os.makedirs(Path(HMM_MODEL_PATH).parent, exist_ok=True)
    with open(HMM_MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "scaler": scaler, "regime_map": regime_map}, f)

    logger.info(f"HMM regime → {regime} (state {cur}) | map={regime_map}")
    return regime, model


# ---------------------------------------------------------------------------
# Technical indicators (used as signal filter alongside PINN)
# ---------------------------------------------------------------------------

def _rsi(series: pd.Series, period: int = RSI_PERIOD) -> float:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss
    return float((100 - 100 / (1 + rs)).iloc[-1])


# ---------------------------------------------------------------------------
# Unified signal + allocation
# ---------------------------------------------------------------------------

def compute_signals(df: pd.DataFrame, regime: str,
                    pinns: dict | None = None,
                    account_equity: float | None = None,
                    initial_equity: float = 100_000.0,
                    vix_var: float = 0.041,
                    ticker: str = "",
                    sentiment_engine=None,
                    df_1min: pd.DataFrame | None = None,
                    spy_1min: pd.DataFrame | None = None) -> dict:
    """
    Returns a dict with:
      signal           — 'buy' | 'sell' | 'sell_all' | 'hold'
      mom_score        — 12-1 month momentum score; primary candidate ranking key
      merton_ratio     — per-stock analytical π* from rolling params; used for sizing
      pinn_allocation  — market-level π* from best available PINN (or merton fallback)
      daily_volume     — latest daily volume (shares) for market impact sizing
      gk_vol           — Garman-Klass intraday daily volatility (0 if no 1-min data)
      micro_halt       — True if GK acceleration triggered a micro-regime safety halt
      lead_lag_lag     — SPY→stock lead lag in minutes (positive = SPY leads)
      lead_lag_corr    — peak cross-correlation at optimal lag
      mu_est, sigma_est, rsi, ema_fast, ema_slow, close, regime
    """
    if df.empty or len(df) < EMA_SLOW + 5:
        return {"signal": "hold", "regime": regime,
                "mom_score": 0.0, "merton_ratio": 0.0, "pinn_allocation": 0.0,
                "daily_volume": 0.0, "gk_vol": 0.0,
                "micro_halt": False, "lead_lag_lag": 0, "lead_lag_corr": 0.0,
                "mu_est": 0.0, "sigma_est": 0.01, "rsi": 50.0,
                "ema_fast": 0.0, "ema_slow": 0.0, "close": 0.0,
                "sentiment_score": 0.0, "sentiment_reason": ""}

    params      = estimate_params(df)
    mu, sigma   = params["mu"], params["sigma"]
    merton      = compute_merton_ratio(mu, sigma)
    mom_score   = compute_momentum(df)

    rsi         = _rsi(df["close"])
    ema_fast    = float(df["close"].ewm(span=EMA_FAST, adjust=False).mean().iloc[-1])
    ema_slow    = float(df["close"].ewm(span=EMA_SLOW, adjust=False).mean().iloc[-1])
    close       = float(df["close"].iloc[-1])
    daily_vol   = float(df["volume"].iloc[-1]) if "volume" in df.columns else 0.0

    # --- Intraday microstructure ---
    micro_halt  = False
    gk_vol      = 0.0
    lead_lag    = {"optimal_lag": 0, "peak_corr": 0.0}

    if df_1min is not None and not df_1min.empty:
        micro_halt = check_vol_acceleration(df_1min)
        gk         = garman_klass_vol(df_1min)
        # Daily volatility from 1-min GK: √(mean_GK × 390 bars/day)
        gk_vol     = float(np.sqrt(np.nanmean(gk) * 390))
        # Lead-lag only runs if 1-min data is fresh (not delayed/stale yfinance cache)
        data_fresh = check_1min_freshness(df_1min)
        if data_fresh and spy_1min is not None and not spy_1min.empty:
            lead_lag = lead_lag_correlation(df_1min, spy_1min)
        elif not data_fresh:
            lead_lag = {"optimal_lag": 0, "peak_corr": 0.0}  # suppressed — stale data

    if micro_halt and ticker:
        logger.warning(f"{ticker}: micro-regime halt overrides HMM — forcing hold")

    # PINN market-level allocation with regime-aware γ scaling.
    # In confirmed bull the PINN output is scaled by (γ_bear / γ_bull) to reflect
    # lower risk aversion — equivalent to retraining at γ=1.5 without the cost.
    w_norm      = (account_equity / initial_equity) if account_equity else 1.0
    pinn_alloc  = get_pinn_allocation(regime, pinns or {}, w_norm, vix_var)
    if pinn_alloc is not None and regime == "bull" and CRRA_GAMMA_BULL < CRRA_GAMMA:
        pinn_alloc = float(np.clip(pinn_alloc * (CRRA_GAMMA / CRRA_GAMMA_BULL), 0.0, 1.0))
        logger.debug(f"γ-scale bull: π*={pinn_alloc:.4f} (γ_eff={CRRA_GAMMA_BULL})")
    # Merton floor — attenuated by VIX level.
    # In low-vol environments (VIX ≈ 15) the main risk is PINN calibration gaps
    # (output ≈ 0 on unfamiliar inputs) so a full Merton floor prevents missing
    # good trades.  In high-VIX regimes the Heston PINN's stochastic-vol
    # awareness is authoritative: Merton assumes constant σ and drift, so
    # applying a full Merton floor at VIX=40 would systematically override the
    # very signal the PINN was trained to produce in those conditions.
    # Linear decay: floor_scale = 1.0 at VIX=15 → 0.25 at VIX=40+.
    if pinn_alloc is not None:
        vix_level    = (vix_var ** 0.5) * 100.0
        floor_scale  = float(np.clip(1.0 - 0.03 * (vix_level - 15.0), 0.25, 1.0))
        merton_floor = merton * floor_scale
        pinn_alloc   = max(float(pinn_alloc), merton_floor)
    else:
        pinn_alloc = merton

    # Signal logic: momentum-first entry, PINN scales exposure, EMA confirms trend.
    # RSI ceiling removed from bull entries — momentum stocks legitimately hold
    # RSI > 70 for months; capping entries there systematically exits the best holdings.
    # Trailing stop (risk_manager.py) handles reversal protection instead.
    # Micro-halt overrides all buy signals (not sells/crash exits).
    if regime == "crash":
        signal = "sell_all"
    elif micro_halt:
        signal = "hold"
    elif regime == "bull":
        if ema_fast > ema_slow and pinn_alloc > 0.05:
            signal = "buy"
        elif ema_fast < ema_slow or pinn_alloc < 0.02:
            signal = "sell"
        else:
            signal = "hold"
    else:  # bear
        if rsi < 30 and ema_fast > ema_slow and pinn_alloc > 0.05:
            # Primary bear entry: deeply oversold + EMA crossover + PINN exposure
            signal = "buy"
        elif rsi < 35 and close > ema_fast and pinn_alloc > 0.05:
            # Bear recovery entry (research iter 6, ACCEPTED): price above EMA20
            # catches early recovery before the slow EMA crossover fires (which lags
            # 50-100 days after a price bottom). RSI < 35 guards against falling knives.
            signal = "buy"
        elif ema_fast < ema_slow or pinn_alloc < 0.02:
            signal = "sell"
        else:
            signal = "hold"

    # --- Sentiment guardrails (FinBERT) ---
    sentiment_score  = 0.0
    sentiment_reason = ""
    if sentiment_engine is not None and ticker and signal in ("buy", "sell"):
        sentiment_score = sentiment_engine.get_ticker_sentiment(ticker)
        signal, sentiment_reason = sentiment_engine.apply_guardrails(
            ticker, signal, sentiment_score
        )

    return {
        "signal":           signal,
        "mom_score":        round(mom_score, 4),
        "merton_ratio":     round(merton, 4),
        "pinn_allocation":  round(pinn_alloc, 4),
        "mu_est":           round(mu, 4),
        "sigma_est":        round(sigma, 4),
        "rsi":              round(rsi, 2),
        "ema_fast":         round(ema_fast, 4),
        "ema_slow":         round(ema_slow, 4),
        "close":            round(close, 4),
        "regime":           regime,
        "daily_volume":     daily_vol,
        "gk_vol":           round(gk_vol, 6),
        "micro_halt":       micro_halt,
        "lead_lag_lag":     lead_lag["optimal_lag"],
        "lead_lag_corr":    lead_lag["peak_corr"],
        "sentiment_score":  round(sentiment_score, 4),
        "sentiment_reason": sentiment_reason,
    }
