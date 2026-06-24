"""
Build or audit stock_info.csv using Alpaca as the source of truth.

Default usage:
    python audit_alpaca_listings.py

Default mode fetches every active Alpaca US equity asset, filters to practical
tradable symbols, checks yfinance price/liquidity, and overwrites stock_info.csv
with the resulting universe. A stock_info.csv.bak backup is written first.

Legacy audit mode:
    python audit_alpaca_listings.py --audit-existing stock_info.csv
"""
from __future__ import annotations

import argparse
import csv
import re
import shutil
from pathlib import Path

import pandas as pd
import yfinance as yf
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import AssetClass, AssetStatus
from alpaca.trading.requests import GetAssetsRequest
from loguru import logger

from settings import (
    ALPACA_API_KEY,
    ALPACA_SECRET_KEY,
    CSV_UNIVERSE_PATH,
    CSV_UNIVERSE_REQUIRE_FRACTIONABLE,
    PAPER_TRADING,
    UNIVERSE_MIN_AVG_VOLUME,
    UNIVERSE_MIN_PRICE,
)


INPUT_TICKER_COLUMN = "Ticker"
BACKUP_SUFFIX = ".bak"
YFINANCE_CHUNK_SIZE = 200
DEFAULT_MAX_SYMBOLS = 0
EXCLUDED_NAME_PATTERN = re.compile(
    r"\b(warrant|right|unit|preferred|preference|depositary|note|bond|debenture)\b",
    re.IGNORECASE,
)


def _require_credentials() -> None:
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        raise EnvironmentError(
            "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in .env"
        )


def _fetch_alpaca_assets() -> dict[str, object]:
    _require_credentials()
    client = TradingClient(
        api_key=ALPACA_API_KEY,
        secret_key=ALPACA_SECRET_KEY,
        paper=PAPER_TRADING,
    )
    assets = client.get_all_assets(
        GetAssetsRequest(asset_class=AssetClass.US_EQUITY, status=AssetStatus.ACTIVE)
    )
    by_symbol = {asset.symbol.upper(): asset for asset in assets}
    logger.info(f"Loaded {len(by_symbol):,} active Alpaca US equity assets")
    return by_symbol


def _clean_ticker(raw: str) -> str:
    return raw.strip().upper()


def _candidate_symbols(ticker: str) -> list[tuple[str, str]]:
    clean = _clean_ticker(ticker)
    candidates: list[tuple[str, str]] = [("exact", clean)]

    if "-" in clean:
        candidates.append(("dash_to_dot", clean.replace("-", ".")))
        root, suffix = clean.split("-", 1)
        if suffix.startswith("P") and len(suffix) >= 2:
            candidates.append(("preferred_dot_pr", f"{root}.PR{suffix[1:]}"))
            candidates.append(("preferred_dot_p", f"{root}.P{suffix[1:]}"))

    if "." in clean:
        candidates.append(("dot_to_dash", clean.replace(".", "-")))

    seen: set[str] = set()
    unique: list[tuple[str, str]] = []
    for match_type, symbol in candidates:
        if symbol and symbol not in seen:
            unique.append((match_type, symbol))
            seen.add(symbol)
    return unique


def _is_practical_asset(asset, require_fractionable: bool) -> bool:
    symbol = str(getattr(asset, "symbol", "")).upper()
    name = str(getattr(asset, "name", ""))

    if not getattr(asset, "tradable", False):
        return False
    if require_fractionable and not getattr(asset, "fractionable", False):
        return False
    if not symbol.isalpha() or len(symbol) > 5:
        return False
    return not EXCLUDED_NAME_PATTERN.search(name)


def _price_liquidity_rows(assets: list[object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for idx in range(0, len(assets), YFINANCE_CHUNK_SIZE):
        chunk = assets[idx : idx + YFINANCE_CHUNK_SIZE]
        tickers = [asset.symbol for asset in chunk]
        try:
            raw = yf.download(
                tickers,
                period="1mo",
                interval="1d",
                progress=False,
                auto_adjust=True,
                threads=True,
            )
        except Exception as e:
            logger.warning(f"yfinance chunk {idx // YFINANCE_CHUNK_SIZE} failed: {e}")
            continue

        if raw.empty:
            continue

        by_symbol = {asset.symbol: asset for asset in chunk}
        close_frame, volume_frame = _extract_close_volume(raw, tickers)

        for symbol in close_frame.columns:
            try:
                close = close_frame[symbol].dropna()
                volume = volume_frame[symbol].dropna()
                if len(close) < 5 or volume.empty:
                    continue

                last_price = float(close.iloc[-1])
                avg_volume = float(volume.mean())
                if last_price < UNIVERSE_MIN_PRICE:
                    continue
                if avg_volume < UNIVERSE_MIN_AVG_VOLUME:
                    continue

                asset = by_symbol[symbol]
                momentum = float(close.iloc[-1] / close.iloc[0] - 1)
                rows.append(_asset_row(asset, last_price, avg_volume, momentum))
            except Exception:
                continue

        logger.info(
            f"Liquidity scan {min(idx + YFINANCE_CHUNK_SIZE, len(assets)):,}/"
            f"{len(assets):,}: kept={len(rows):,}"
        )

    rows.sort(key=lambda row: float(row["Momentum1M"]), reverse=True)
    return rows


def _extract_close_volume(
    raw: pd.DataFrame,
    tickers: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if isinstance(raw.columns, pd.MultiIndex):
        close_frame = raw["Close"] if "Close" in raw else pd.DataFrame()
        volume_frame = raw["Volume"] if "Volume" in raw else pd.DataFrame()
        return close_frame, volume_frame

    if len(tickers) != 1:
        return pd.DataFrame(), pd.DataFrame()

    symbol = tickers[0]
    close_frame = pd.DataFrame({symbol: raw["Close"]}) if "Close" in raw else pd.DataFrame()
    volume_frame = pd.DataFrame({symbol: raw["Volume"]}) if "Volume" in raw else pd.DataFrame()
    return close_frame, volume_frame


def _asset_row(
    asset,
    last_price: float,
    avg_volume: float,
    momentum: float,
) -> dict[str, object]:
    return {
        "Ticker": asset.symbol,
        "Name": getattr(asset, "name", ""),
        "Exchange": getattr(asset, "exchange", ""),
        "AlpacaListed": "TRUE",
        "AlpacaSymbol": asset.symbol,
        "AlpacaMatchType": "alpaca_source",
        "AlpacaName": getattr(asset, "name", ""),
        "AlpacaExchange": getattr(asset, "exchange", ""),
        "AlpacaStatus": getattr(asset, "status", ""),
        "AlpacaTradable": getattr(asset, "tradable", ""),
        "AlpacaMarginable": getattr(asset, "marginable", ""),
        "AlpacaShortable": getattr(asset, "shortable", ""),
        "AlpacaFractionable": getattr(asset, "fractionable", ""),
        "LastPrice": round(last_price, 4),
        "AvgVolume1M": round(avg_volume, 2),
        "Momentum1M": round(momentum, 6),
    }


def _write_rows(path: Path, rows: list[dict[str, object]], backup: bool) -> None:
    if not rows:
        raise ValueError("Refusing to write empty stock universe")

    fieldnames = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")

    with temp_path.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if path.exists() and backup:
        backup_path = path.with_name(f"{path.name}{BACKUP_SUFFIX}")
        shutil.copy2(path, backup_path)
        logger.info(f"Backup written to {backup_path}")

    temp_path.replace(path)
    logger.info(f"Wrote {len(rows):,} filtered Alpaca universe rows to {path}")


def build_filtered_universe(
    output_path: Path,
    max_symbols: int = DEFAULT_MAX_SYMBOLS,
    backup: bool = True,
) -> None:
    assets_by_symbol = _fetch_alpaca_assets()
    practical_assets = [
        asset
        for asset in assets_by_symbol.values()
        if _is_practical_asset(asset, CSV_UNIVERSE_REQUIRE_FRACTIONABLE)
    ]
    practical_assets.sort(key=lambda asset: asset.symbol)
    logger.info(
        f"Practical Alpaca candidates before liquidity filter: "
        f"{len(practical_assets):,}"
    )

    rows = _price_liquidity_rows(practical_assets)
    if max_symbols > 0:
        rows = rows[:max_symbols]
    _write_rows(output_path, rows, backup)


def _audit_row(row: dict[str, str], assets_by_symbol: dict[str, object]) -> dict[str, str]:
    ticker = _clean_ticker(row.get(INPUT_TICKER_COLUMN, ""))
    match_type = ""
    alpaca_symbol = ""
    asset = None

    for candidate_type, candidate in _candidate_symbols(ticker):
        if candidate in assets_by_symbol:
            match_type = candidate_type
            alpaca_symbol = candidate
            asset = assets_by_symbol[candidate]
            break

    row["AlpacaListed"] = "TRUE" if asset is not None else "FALSE"
    row["AlpacaSymbol"] = alpaca_symbol
    row["AlpacaMatchType"] = match_type
    row["AlpacaName"] = str(getattr(asset, "name", "")) if asset is not None else ""
    row["AlpacaExchange"] = str(getattr(asset, "exchange", "")) if asset is not None else ""
    row["AlpacaStatus"] = str(getattr(asset, "status", "")) if asset is not None else ""
    row["AlpacaTradable"] = str(getattr(asset, "tradable", "")) if asset is not None else ""
    row["AlpacaMarginable"] = str(getattr(asset, "marginable", "")) if asset is not None else ""
    row["AlpacaShortable"] = str(getattr(asset, "shortable", "")) if asset is not None else ""
    row["AlpacaFractionable"] = str(getattr(asset, "fractionable", "")) if asset is not None else ""
    return row


def audit_existing_csv(input_path: Path, backup: bool = True) -> None:
    assets_by_symbol = _fetch_alpaca_assets()

    with input_path.open("r", encoding="utf-8-sig", newline="") as infile:
        reader = csv.DictReader(infile)
        if reader.fieldnames is None:
            raise ValueError(f"{input_path} has no header row")
        if INPUT_TICKER_COLUMN not in reader.fieldnames:
            raise ValueError(
                f"{input_path} must contain a '{INPUT_TICKER_COLUMN}' column"
            )

        extra_fields = [
            "AlpacaListed",
            "AlpacaSymbol",
            "AlpacaMatchType",
            "AlpacaName",
            "AlpacaExchange",
            "AlpacaStatus",
            "AlpacaTradable",
            "AlpacaMarginable",
            "AlpacaShortable",
            "AlpacaFractionable",
        ]
        fieldnames = list(reader.fieldnames) + [
            field for field in extra_fields if field not in reader.fieldnames
        ]
        rows = [_audit_row(row, assets_by_symbol) for row in reader]

    temp_path = input_path.with_name(f"{input_path.name}.tmp")
    with temp_path.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if backup:
        backup_path = input_path.with_name(f"{input_path.name}{BACKUP_SUFFIX}")
        shutil.copy2(input_path, backup_path)
        logger.info(f"Backup written to {backup_path}")

    temp_path.replace(input_path)
    listed = sum(1 for row in rows if row["AlpacaListed"] == "TRUE")
    logger.info(
        f"Audited {len(rows):,} rows in {input_path} | "
        f"listed={listed:,} missing={len(rows) - listed:,}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build stock_info.csv from Alpaca or audit an existing CSV."
    )
    parser.add_argument(
        "input_csv",
        nargs="?",
        type=Path,
        default=Path(CSV_UNIVERSE_PATH),
        help="CSV path. Defaults to settings.CSV_UNIVERSE_PATH.",
    )
    parser.add_argument(
        "--audit-existing",
        action="store_true",
        help="Audit existing CSV rows instead of rebuilding from Alpaca.",
    )
    parser.add_argument(
        "--max-symbols",
        type=int,
        default=DEFAULT_MAX_SYMBOLS,
        help="Limit rebuilt universe size after liquidity ranking. 0 means no cap.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not write <csv>.bak before replacing the CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = args.input_csv

    if args.audit_existing:
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
        audit_existing_csv(csv_path, backup=not args.no_backup)
        return

    build_filtered_universe(
        csv_path,
        max_symbols=args.max_symbols,
        backup=not args.no_backup,
    )


if __name__ == "__main__":
    main()
