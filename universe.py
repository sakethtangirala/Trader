"""
Daily universe builder.

Stage 1 — Primary (OpenBB finviz screener):
  Pulls most-active, top-momentum, and oversold large/mid-caps. Fast, curated.

Stage 2 — Fallback (Alpaca assets + yfinance filter):
  Fetches every fractionable US equity from Alpaca, batch-downloads 1 month of
  price/volume via yfinance, filters by price and liquidity, sorts by momentum.

Returns up to UNIVERSE_SIZE symbols, refreshed once per trading day.
"""
import pandas as pd
import yfinance as yf
from openbb import obb
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass, AssetStatus
from loguru import logger

from settings import (
    ALPACA_API_KEY,
    ALPACA_SECRET_KEY,
    FALLBACK_WATCHLIST,
    PAPER_TRADING,
    UNIVERSE_MIN_AVG_VOLUME,
    UNIVERSE_MIN_PRICE,
    UNIVERSE_SIZE,
)


# ---------------------------------------------------------------------------
# Primary: OpenBB screener (finviz)
# ---------------------------------------------------------------------------

def _screener_universe() -> list[str]:
    symbols: set[str] = set()

    screens = [
        dict(signal="most_active",  mktcap="large_over", limit=UNIVERSE_SIZE),
        dict(signal="top_gainers",  mktcap="large_over", limit=50),
        dict(signal="oversold",     mktcap="mid_over",   limit=50),
        dict(signal="unusual_volume", mktcap="mid_over", limit=50),
    ]

    for params in screens:
        try:
            df = obb.equity.screener(
                provider="finviz",
                price_min=UNIVERSE_MIN_PRICE,
                **params,
            ).to_df()

            col = next((c for c in ["symbol", "ticker", "Symbol"] if c in df.columns), None)
            if col:
                symbols.update(df[col].dropna().tolist())
            elif not df.empty:
                symbols.update(df.index.tolist())
        except Exception as e:
            logger.warning(f"Screener screen failed ({params.get('signal')}): {e}")

    # Keep clean alpha-only symbols
    clean = [s for s in symbols if isinstance(s, str) and s.isalpha() and len(s) <= 5]
    logger.info(f"Screener universe: {len(clean)} symbols")
    return clean[:UNIVERSE_SIZE]


# ---------------------------------------------------------------------------
# Fallback: Alpaca asset list + yfinance liquidity filter
# ---------------------------------------------------------------------------

def _alpaca_universe() -> list[str]:
    try:
        client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=PAPER_TRADING)
        assets = client.get_all_assets(
            GetAssetsRequest(asset_class=AssetClass.US_EQUITY, status=AssetStatus.ACTIVE)
        )
    except Exception as e:
        logger.error(f"Alpaca asset fetch failed: {e}")
        return []

    candidates = [
        a.symbol for a in assets
        if a.tradable
        and getattr(a, "fractionable", False)
        and a.symbol.isalpha()
        and len(a.symbol) <= 5
    ]
    logger.info(f"Alpaca fractionable equities: {len(candidates)}")

    qualified: list[tuple[str, float]] = []
    chunk_size = 200

    # Only scan the first 3000 to keep runtime reasonable
    for i in range(0, min(len(candidates), 3000), chunk_size):
        chunk = candidates[i : i + chunk_size]
        try:
            raw = yf.download(
                chunk,
                period="1mo",
                interval="1d",
                progress=False,
                auto_adjust=True,
                threads=True,
            )
            if raw.empty:
                continue

            # MultiIndex: (price_type, ticker)
            if isinstance(raw.columns, pd.MultiIndex):
                tickers_in = raw["Close"].columns if "Close" in raw else []
                for sym in tickers_in:
                    try:
                        close = raw["Close"][sym].dropna()
                        volume = raw["Volume"][sym].dropna()
                        if len(close) < 5:
                            continue
                        if close.iloc[-1] >= UNIVERSE_MIN_PRICE and volume.mean() >= UNIVERSE_MIN_AVG_VOLUME:
                            momentum = (close.iloc[-1] - close.iloc[0]) / close.iloc[0]
                            qualified.append((sym, float(momentum)))
                    except Exception:
                        continue
        except Exception as e:
            logger.warning(f"Chunk {i // chunk_size} yfinance error: {e}")

    qualified.sort(key=lambda x: x[1], reverse=True)
    result = [s for s, _ in qualified[:UNIVERSE_SIZE]]
    logger.info(f"Alpaca fallback universe: {len(result)} symbols")
    return result


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_universe() -> list[str]:
    """
    Build today's trading universe. Called once per trading day.
    Returns up to UNIVERSE_SIZE ticker symbols sorted by momentum/activity.
    """
    logger.info("Building trading universe...")

    universe = _screener_universe()

    if len(universe) < 20:
        logger.warning(f"Screener returned only {len(universe)} symbols — using Alpaca fallback")
        universe = _alpaca_universe()

    if not universe:
        logger.error("Both universe sources failed — falling back to default watchlist")
        return FALLBACK_WATCHLIST

    logger.info(f"Universe ready: {len(universe)} symbols")
    return universe
