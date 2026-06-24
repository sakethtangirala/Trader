"""
Daily universe builder.

Primary source is stock_info.csv validated against Alpaca's active tradable
asset list. OpenBB and Alpaca screener paths remain as fallbacks.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import AssetClass, AssetStatus
from alpaca.trading.requests import GetAssetsRequest
from loguru import logger
from openbb import obb

from settings import (
    ALPACA_API_KEY,
    ALPACA_SECRET_KEY,
    CSV_UNIVERSE_ENABLED,
    CSV_UNIVERSE_MAX_SYMBOLS,
    CSV_UNIVERSE_PATH,
    CSV_UNIVERSE_REQUIRE_FRACTIONABLE,
    CSV_UNIVERSE_TICKER_COLUMN,
    FALLBACK_WATCHLIST,
    PAPER_TRADING,
    UNIVERSE_MIN_AVG_VOLUME,
    UNIVERSE_MIN_PRICE,
    UNIVERSE_SIZE,
)


_alpaca_assets_cache: dict[str, object] | None = None


def _alpaca_assets_by_symbol() -> dict[str, object]:
    global _alpaca_assets_cache
    if _alpaca_assets_cache is not None:
        return _alpaca_assets_cache

    client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=PAPER_TRADING)
    assets = client.get_all_assets(
        GetAssetsRequest(asset_class=AssetClass.US_EQUITY, status=AssetStatus.ACTIVE)
    )
    _alpaca_assets_cache = {asset.symbol.upper(): asset for asset in assets}
    logger.info(f"Alpaca active US equity assets: {len(_alpaca_assets_cache):,}")
    return _alpaca_assets_cache


def _clean_ticker(raw: str) -> str:
    return raw.strip().upper()


def _candidate_symbols(ticker: str) -> list[str]:
    """
    Conservative symbol variants for historical CSV formats.

    Alpaca commonly represents share classes with dots, while older CSVs may use
    dashes. Preferred tickers also differ across vendors, so test common forms.
    """
    clean = _clean_ticker(ticker)
    candidates = [clean]

    if "-" in clean:
        candidates.append(clean.replace("-", "."))
        root, suffix = clean.split("-", 1)
        if suffix.startswith("P") and len(suffix) >= 2:
            candidates.append(f"{root}.PR{suffix[1:]}")
            candidates.append(f"{root}.P{suffix[1:]}")

    if "." in clean:
        candidates.append(clean.replace(".", "-"))

    seen: set[str] = set()
    unique: list[str] = []
    for symbol in candidates:
        if symbol and symbol not in seen:
            unique.append(symbol)
            seen.add(symbol)
    return unique


def _asset_is_eligible(asset) -> bool:
    if not getattr(asset, "tradable", False):
        return False
    if CSV_UNIVERSE_REQUIRE_FRACTIONABLE and not getattr(asset, "fractionable", False):
        return False
    return True


def _csv_universe() -> list[str]:
    """
    Primary universe source.  Reads stock_info.csv which carries pre-validated
    Alpaca columns (AlpacaStatus / AlpacaTradable / AlpacaFractionable).  When
    those columns are present the function trusts them and skips the live Alpaca
    API call entirely — making this path fast and network-failure-proof.

    Falls back to a live Alpaca cross-check only when the pre-validated columns
    are absent (e.g. a plain ticker-only CSV was provided instead).
    """
    if not CSV_UNIVERSE_ENABLED:
        return []

    csv_path = Path(CSV_UNIVERSE_PATH)
    if not csv_path.exists():
        logger.warning(f"CSV universe file not found: {csv_path}")
        return []

    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
    except Exception as e:
        logger.error(f"CSV universe read failed: {e}")
        return []

    total_rows = len(df)

    # ── Fast path: pre-validated columns present — no live API call needed ──
    if {"AlpacaStatus", "AlpacaTradable"}.issubset(df.columns):
        df = df[df["AlpacaStatus"].str.lower() == "active"]
        df = df[df["AlpacaTradable"].astype(str).str.lower() == "true"]
        if CSV_UNIVERSE_REQUIRE_FRACTIONABLE and "AlpacaFractionable" in df.columns:
            df = df[df["AlpacaFractionable"].astype(str).str.lower() == "true"]

        # Use the confirmed Alpaca symbol when available
        sym_col = "AlpacaSymbol" if "AlpacaSymbol" in df.columns else CSV_UNIVERSE_TICKER_COLUMN
        if sym_col not in df.columns:
            logger.error(f"CSV universe missing '{sym_col}' column: {csv_path}")
            return []

        symbols = (
            df[sym_col]
            .dropna()
            .str.strip()
            .str.upper()
            .drop_duplicates()
            .tolist()
        )

        if CSV_UNIVERSE_MAX_SYMBOLS > 0:
            symbols = symbols[:CSV_UNIVERSE_MAX_SYMBOLS]

        logger.info(
            f"CSV universe (pre-validated): {len(symbols):,} symbols "
            f"from {total_rows:,} rows ({csv_path})"
        )
        return symbols

    # ── Slow path: no pre-validated columns — cross-check against Alpaca live ──
    logger.info("CSV has no pre-validated columns; falling back to live Alpaca check")

    if CSV_UNIVERSE_TICKER_COLUMN not in df.columns:
        logger.error(f"CSV universe missing '{CSV_UNIVERSE_TICKER_COLUMN}' column: {csv_path}")
        return []

    try:
        assets_by_symbol = _alpaca_assets_by_symbol()
    except Exception as e:
        logger.error(f"Alpaca asset fetch failed for CSV universe: {e}")
        return []

    symbols: list[str] = []
    seen: set[str] = set()

    for raw_ticker in df[CSV_UNIVERSE_TICKER_COLUMN].dropna():
        for candidate in _candidate_symbols(str(raw_ticker)):
            asset = assets_by_symbol.get(candidate)
            if asset is not None and _asset_is_eligible(asset):
                if candidate not in seen:
                    symbols.append(candidate)
                    seen.add(candidate)
                break

    if CSV_UNIVERSE_MAX_SYMBOLS > 0:
        symbols = symbols[:CSV_UNIVERSE_MAX_SYMBOLS]

    logger.info(
        f"CSV universe (live-checked): {len(symbols):,} eligible Alpaca symbols "
        f"from {total_rows:,} rows ({csv_path})"
    )
    return symbols


def _screener_universe() -> list[str]:
    symbols: set[str] = set()

    screens = [
        dict(signal="most_active", mktcap="large_over", limit=UNIVERSE_SIZE),
        dict(signal="top_gainers", mktcap="large_over", limit=50),
        dict(signal="oversold", mktcap="mid_over", limit=50),
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

    clean = [s for s in symbols if isinstance(s, str) and s.isalpha() and len(s) <= 5]
    logger.info(f"Screener universe: {len(clean)} symbols")
    return clean[:UNIVERSE_SIZE]


def _alpaca_universe() -> list[str]:
    try:
        assets = list(_alpaca_assets_by_symbol().values())
    except Exception as e:
        logger.error(f"Alpaca asset fetch failed: {e}")
        return []

    candidates = [
        asset.symbol
        for asset in assets
        if asset.tradable
        and getattr(asset, "fractionable", False)
        and asset.symbol.isalpha()
        and len(asset.symbol) <= 5
    ]
    logger.info(f"Alpaca fractionable equities: {len(candidates)}")

    qualified: list[tuple[str, float]] = []
    chunk_size = 200

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

            if isinstance(raw.columns, pd.MultiIndex):
                tickers_in = raw["Close"].columns if "Close" in raw else []
                for sym in tickers_in:
                    try:
                        close = raw["Close"][sym].dropna()
                        volume = raw["Volume"][sym].dropna()
                        if len(close) < 5:
                            continue
                        liquid = volume.mean() >= UNIVERSE_MIN_AVG_VOLUME
                        if close.iloc[-1] >= UNIVERSE_MIN_PRICE and liquid:
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


def build_universe() -> list[str]:
    """
    Build today's trading universe. Called once per trading day.

    CSV-backed Alpaca validation is primary; dynamic screeners are fallback only.
    """
    logger.info("Building trading universe...")

    universe = _csv_universe()
    if universe:
        logger.info(f"Universe ready from CSV: {len(universe):,} symbols")
        return universe

    logger.warning("CSV universe unavailable/empty; falling back to OpenBB screener")
    universe = _screener_universe()

    if len(universe) < 20:
        logger.warning(f"Screener returned only {len(universe)} symbols; using Alpaca fallback")
        universe = _alpaca_universe()

    if not universe:
        logger.error("All universe sources failed; falling back to default watchlist")
        return FALLBACK_WATCHLIST

    logger.info(f"Universe ready: {len(universe)} symbols")
    return universe
