"""
Autonomous Trading Agent — main orchestration loop.

Cycle (every 15 min during market hours):
  1. Rebuild universe daily (OpenBB screener + Alpaca assets)
  2. Check circuit breakers
  3. Detect SPY market regime via HMM
  4. Crash → close all and halt
  5. Stop-loss / take-profit on open positions
  6. Query Heston/Regime/GBM PINN for market-level allocation
  7. Scan universe for RSI+EMA buy signals
  8. Size via Merton-weighted PINN allocation, execute
  9. Log everything to logs/trade_journal.log

Run train_pinns.py once before starting for optimal PINN allocation.
Falls back to Merton ratio + RSI/EMA rules if no PINN is available.
"""
import os
import time
from collections import deque
from datetime import datetime

import pytz
from loguru import logger

from brokerage import Broker
from ui import TradingUI
from engine import (
    compute_signals, detect_regime, fetch_ohlcv, fetch_vix, load_pinns,
    fetch_1min, get_sector,
)
from sentiment import SentimentWorker
from validation import quick_pvalue
from risk_manager import RiskManager
from sentiment import FreeSentimentEngine
from universe import build_universe
from settings import (
    BULL_BYPASS_PINN,
    BULL_LEVERAGE_FACTOR,
    BULL_MAX_POSITION_SIZE,
    CYCLE_INTERVAL_SECONDS,
    HMM_REFIT_DAYS,
    MAX_POSITION_SIZE,
    PERMUTATION_N,
    PERMUTATION_PVALUE_GATE,
    UNIVERSE_BATCH_SIZE,
    WALK_FORWARD_GATE,
    CYCLE_INTERVAL_BULL_SECONDS,
    FALLBACK_WATCHLIST,
    INITIAL_EQUITY,
    LIMIT_ORDER_ENABLED,
    LIMIT_ORDER_TIMEOUT_MIN,
    LOG_FILE,
    LOG_RETENTION,
    LOG_ROTATION,
    MARKET_CRASH_SPY_THRESHOLD,
    MARKET_TIMEZONE,
    MAX_POSITIONS,
    PAPER_TRADING,
    PINN_DAILY_CACHE,
    SECTOR_MAX_POSITIONS,
    SIGNAL_LOOKBACK_DAYS,
    STOP_CHECK_INTERVAL_SECONDS,
    VIX_HIGH_THRESHOLD,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
os.makedirs("logs", exist_ok=True)
logger.add(
    LOG_FILE,
    rotation=LOG_ROTATION,
    retention=LOG_RETENTION,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
    level="DEBUG",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_market_open() -> bool:
    tz = pytz.timezone(MARKET_TIMEZONE)
    now = datetime.now(tz)
    if now.weekday() >= 5:
        return False
    open_t  = now.replace(hour=9,  minute=30, second=0, microsecond=0)
    close_t = now.replace(hour=16, minute=0,  second=0, microsecond=0)
    return open_t <= now <= close_t


def is_new_day(last_day: int) -> bool:
    return datetime.now().day != last_day


def _sweep_stale_orders(broker: Broker) -> None:
    """
    Cancel limit orders that have been open longer than LIMIT_ORDER_TIMEOUT_MIN
    and resubmit them as market orders to guarantee execution.

    Called at the top of each cycle so no order lingers across multiple bars.
    """
    if not LIMIT_ORDER_ENABLED:
        return
    open_orders = broker.get_open_orders()
    if not open_orders:
        return
    now = datetime.now(tz=pytz.UTC)
    for order in open_orders:
        try:
            age_min = (now - order.created_at).total_seconds() / 60.0
            if age_min >= LIMIT_ORDER_TIMEOUT_MIN:
                logger.info(
                    f"Stale limit order: {order.symbol} qty={order.qty} "
                    f"age={age_min:.1f} min — cancelling and market-filling"
                )
                broker.cancel_and_market_fill(
                    order.symbol, str(order.id), float(order.qty)
                )
        except Exception as e:
            logger.warning(f"Sweep order {getattr(order, 'id', '?')}: {e}")


def _next_cycle_interval(regime: str, vix_var: float) -> int:
    """
    Dynamic cycle frequency based on confirmed regime and VIX level.

    Bull + calm VIX  → 2 h  (CYCLE_INTERVAL_BULL_SECONDS)
      Minimal portfolio drift; high transaction overhead for tiny adjustments.
    Bear / high VIX  → 15 min  (CYCLE_INTERVAL_SECONDS)
      Active risk management required; every bar can matter.
    """
    vix = (vix_var ** 0.5) * 100.0   # back to VIX index level
    if regime == "bull" and vix < VIX_HIGH_THRESHOLD:
        logger.info(
            f"Bull + VIX={vix:.1f} < {VIX_HIGH_THRESHOLD} — "
            f"throttling to {CYCLE_INTERVAL_BULL_SECONDS // 3600}h cycle"
        )
        return CYCLE_INTERVAL_BULL_SECONDS
    return CYCLE_INTERVAL_SECONDS


# ---------------------------------------------------------------------------
# HMM regime smoother
# ---------------------------------------------------------------------------
class _RegimeSmoother:
    """
    Confirmation filter for HMM regime labels.
    Regime only changes downstream when the last `n` readings all agree.
    Conservative default: 'bear' until confirmed otherwise.
    """

    def __init__(self, n: int = 2):
        self._n        = n
        self._history: deque[str] = deque(maxlen=n)
        self._confirmed = "bear"

    def update(self, raw: str) -> str:
        """Append raw HMM label; return confirmed regime."""
        self._history.append(raw)
        if len(self._history) == self._n and len(set(self._history)) == 1:
            if self._confirmed != raw:
                logger.info(
                    f"Regime confirmed: {self._confirmed} → {raw} "
                    f"({self._n} consecutive readings)"
                )
            self._confirmed = raw
        return self._confirmed


_regime_smoother = _RegimeSmoother(n=2)
_cycle_counter:  int = 0

# Walk-forward HMM lock: only refit the HMM model every HMM_REFIT_DAYS
# trading days. Between refits, feed the cached raw label into the smoother.
_hmm_last_refit:  object     = None   # date of last HMM refit
_hmm_raw_cached:  str        = "bear" # last raw HMM output

# Daily per-symbol permutation p-value cache.
# Populated at first buy-candidate evaluation for a symbol each day.
# Cleared when is_new_day() fires. Values: {symbol: float}
_pvalue_cache: dict[str, float] = {}
_pvalue_cache_date: object      = None

# Global terminal UI instance (started in main(), used throughout run_cycle)
_ui: TradingUI | None = None


# ---------------------------------------------------------------------------
# Daily PINN allocation cache
# ---------------------------------------------------------------------------
class _PINNCache:
    """Thin wrapper that stores today's PINN allocation and the regime it was computed for."""
    def __init__(self):
        self._date:     object  = None
        self._regime:   str     = ""
        self._alloc:    float   = -1.0

    def get(self, today, regime: str) -> float | None:
        if not PINN_DAILY_CACHE:
            return None   # caching disabled → always recompute
        if self._date == today and self._regime == regime and self._alloc >= 0:
            return self._alloc
        return None

    def set(self, today, regime: str, alloc: float) -> None:
        self._date   = today
        self._regime = regime
        self._alloc  = alloc


_pinn_cache = _PINNCache()


# ---------------------------------------------------------------------------
# Core trading cycle
# ---------------------------------------------------------------------------

def run_cycle(broker: Broker, risk: RiskManager,
              universe: list[str], pinns: dict,
              sentiment: FreeSentimentEngine | None,
              sentiment_worker: "SentimentWorker | None" = None) -> int:
    """Returns the number of seconds to sleep before the next cycle."""
    global _cycle_counter
    _cycle_counter += 1

    # Batch rotation: evaluate UNIVERSE_BATCH_SIZE stocks per cycle.
    # The full universe rotates over N cycles (N = ceil(len/batch)).
    # Positions are always exit-scanned; only BUY candidates are batched.
    if UNIVERSE_BATCH_SIZE > 0 and len(universe) > UNIVERSE_BATCH_SIZE:
        start        = ((_cycle_counter - 1) * UNIVERSE_BATCH_SIZE) % len(universe)
        end          = start + UNIVERSE_BATCH_SIZE
        active_batch = (universe[start:end] if end <= len(universe)
                        else universe[start:] + universe[:end - len(universe)])
        logger.info(f"Batch {_cycle_counter}: symbols [{start}:{min(end, len(universe))}]"
                    f" of {len(universe)} (size={len(active_batch)})")
    else:
        active_batch = universe

    # Pre-warm sentiment scores for the NEXT batch in the background.
    if sentiment_worker is not None:
        next_start   = (_cycle_counter * UNIVERSE_BATCH_SIZE) % max(len(universe), 1)
        next_end     = next_start + UNIVERSE_BATCH_SIZE
        next_batch   = (universe[next_start:next_end] if next_end <= len(universe)
                        else universe[next_start:] + universe[:next_end - len(universe)])
        sentiment_worker.submit_batch(next_batch)

    # Initialise with safe defaults so early-return paths can use _next_cycle_interval
    regime  = "bear"
    vix_var = 0.041

    # Sweep stale limit orders from the previous cycle before any new logic
    _sweep_stale_orders(broker)

    account = broker.get_account()
    risk.refresh(account)
    equity  = float(account.equity)
    cash    = float(account.cash)
    initial = INITIAL_EQUITY
    risk.record_equity(equity)

    logger.info(
        f"Equity: ${equity:>12,.2f}  Cash: ${cash:>12,.2f}  "
        f"P&L today: ${equity - float(account.last_equity):>+,.2f}"
    )

    # 1. Circuit breakers
    if risk.check_circuit_breakers():
        logger.warning("Circuit breaker active — cancelling orders")
        broker.cancel_all_orders()
        return CYCLE_INTERVAL_SECONDS

    # 2. Fetch VIX (Heston variance proxy) + SPY 1-min baseline for lead-lag
    vix_var  = fetch_vix()
    spy_1min = fetch_1min("SPY", bars=30)
    logger.info(f"VIX variance proxy y = {vix_var:.5f} (VIX ≈ {(vix_var**0.5)*100:.1f})")

    # 3. Detect SPY market regime (HMM)
    spy_df = fetch_ohlcv("SPY")
    if spy_df.empty:
        logger.error("SPY data unavailable — skipping cycle")
        return CYCLE_INTERVAL_SECONDS

    # --- Compute SPY 1-Day Return Baseline ---
    spy_1d_ret = 0.0
    if len(spy_df) >= 2:
        spy_1d_ret = (
            (spy_df["close"].iloc[-1] - spy_df["close"].iloc[-2])
            / spy_df["close"].iloc[-2]
        )

    # --- Hard crash backstop (rule-based, fires before HMM) ---
    if len(spy_df) >= 2 and spy_1d_ret <= MARKET_CRASH_SPY_THRESHOLD:
        logger.warning(
            f"SPY HARD CRASH BACKSTOP: 1-day return {spy_1d_ret:.2%} ≤ "
            f"{MARKET_CRASH_SPY_THRESHOLD:.2%} — closing all positions"
        )
        broker.close_all()
        return CYCLE_INTERVAL_SECONDS

    # --- HMM walk-forward lock: refit at most every HMM_REFIT_DAYS trading days ---
    # Between refits the cached raw label feeds into the smoother as-is.
    # This prevents intraday HMM noise from flipping regime every 15 minutes.
    global _hmm_last_refit, _hmm_raw_cached
    today_date = datetime.now().date()
    days_since = (today_date - _hmm_last_refit).days if _hmm_last_refit else HMM_REFIT_DAYS
    if days_since >= HMM_REFIT_DAYS:
        _hmm_raw_cached, _ = detect_regime(spy_df)
        _hmm_last_refit    = today_date
        logger.info(f"HMM refitted (fold advance) → raw={_hmm_raw_cached}")
    raw_regime = _hmm_raw_cached
    regime     = _regime_smoother.update(raw_regime)
    if raw_regime != regime:
        logger.debug(
            f"HMM raw={raw_regime} | confirmed={regime} "
            f"(smoothed — waiting for {_regime_smoother._n}-cycle consensus)"
        )

    # Daily PINN allocation cache — re-query only when date or regime changes.
    today           = datetime.now().date()
    w_norm          = equity / initial
    cached_alloc    = _pinn_cache.get(today, regime)

    spy_sig = compute_signals(spy_df, regime, pinns, equity, initial, vix_var,
                               ticker="SPY", sentiment_engine=sentiment,
                               df_1min=spy_1min)

    if cached_alloc is None:
        market_alloc = spy_sig["pinn_allocation"]
        _pinn_cache.set(today, regime, market_alloc)
        logger.info(f"PINN π*={market_alloc:.4f} regime={regime} w={w_norm:.3f} — cached EOD")
    else:
        market_alloc = cached_alloc

    # Update terminal dashboard header
    pnl_today = equity - float(account.last_equity)
    from engine import fetch_risk_free_rate
    if _ui:
        _ui.update_header(
            equity    = equity,
            cash      = cash,
            pnl_today = pnl_today,
            regime    = regime,
            vix       = (vix_var ** 0.5) * 100.0,
            rf_rate   = fetch_risk_free_rate(),
            pinn_alloc= market_alloc,
            cycle_num = _cycle_counter,
            universe_n= len(universe),
        )

    # 4. Crash regime validation gate -> avoid false liquidations on rebalance noise
    if risk.is_crash_regime(spy_sig):
        vix_level = (vix_var ** 0.5) * 100.0
        # Hard validation validation rule: Require real market fear or downside pressure
        if vix_level > 24.0 or spy_1d_ret <= -0.02:
            logger.warning(
                f"CRASH REGIME VALIDATED: VIX={vix_level:.1f}, SPY 1D={spy_1d_ret:.2%}. "
                f"Executing emergency liquidation."
            )
            broker.close_all()
            for sym in broker.get_positions():
                risk.clear_peak(sym)
            return CYCLE_INTERVAL_SECONDS
        else:
            logger.info(
                f"HMM flagged CRASH but validation gate BLOCKED liquidation. "
                f"VIX={vix_level:.1f} and SPY 1D={spy_1d_ret:.2%} indicate noise. "
                f"Overriding to defensive 'bear' state."
            )
            regime = "bear"
            spy_sig["regime"] = "bear"

    # 5. Trailing-stop / stop-loss scan
    positions = broker.get_positions()
    for symbol, pos in list(positions.items()):
        exit_flag, reason = risk.should_exit(pos)
        if exit_flag:
            logger.info(f"Exit {symbol} [{reason}]")
            broker.sell(symbol)
            risk.clear_peak(symbol)
            if _ui:
                _ui.add_exit(symbol, reason)

    # 6. Find buy candidates across universe
    positions = broker.get_positions()
    n_open    = len(positions)

    if n_open >= MAX_POSITIONS:
        logger.info(f"Max positions ({MAX_POSITIONS}) reached — no new buys")
        return _next_cycle_interval(regime, vix_var)

    buy_candidates: list[dict] = []
    if _ui:
        _ui.start_scan(len([s for s in active_batch if s not in positions]))
    for symbol in active_batch:
        if symbol in positions:
            continue
        df = fetch_ohlcv(symbol, lookback_days=SIGNAL_LOOKBACK_DAYS)
        if df.empty:
            if _ui:
                _ui.tick_scan(symbol, "hold")
            continue
        df_1min = fetch_1min(symbol, bars=30)
        sig = compute_signals(df, regime, pinns, equity, initial, vix_var,
                              ticker=symbol, sentiment_engine=sentiment,
                              df_1min=df_1min, spy_1min=spy_1min)
        if _ui:
            _ui.tick_scan(symbol, sig["signal"])
        logger.debug(f"{symbol} {sig['signal']} π*={sig['pinn_allocation']:.3f} "
                     f"mom={sig['mom_score']:+.3f} rsi={sig['rsi']:.0f}")
        if sig["signal"] == "buy":
            # Walk-forward p-value gate: skip symbols whose EMA signal is
            # statistically indistinguishable from shuffled noise today.
            if WALK_FORWARD_GATE:
                global _pvalue_cache, _pvalue_cache_date
                today_d = datetime.now().date()
                if _pvalue_cache_date != today_d:
                    _pvalue_cache      = {}
                    _pvalue_cache_date = today_d
                if symbol not in _pvalue_cache:
                    _pvalue_cache[symbol] = quick_pvalue(
                        df["close"], n=PERMUTATION_N)
                pval = _pvalue_cache[symbol]
                if pval >= PERMUTATION_PVALUE_GATE:
                    logger.debug(
                        f"  {symbol}: p-value={pval:.3f} ≥ {PERMUTATION_PVALUE_GATE} "
                        f"— edge not significant, skipping"
                    )
                    continue

            buy_candidates.append({
                "symbol":       symbol,
                "price":        sig["close"],
                "momentum":     sig["mom_score"],
                "merton":       sig["merton_ratio"],
                "sigma":        max(sig["sigma_est"], 0.01),
                "pinn":         sig["pinn_allocation"],
                "daily_volume": sig["daily_volume"],
                "gk_vol":       sig["gk_vol"] if sig["gk_vol"] > 0 else sig["sigma_est"],
            })

    # Rank by 12-1 month momentum
    buy_candidates.sort(key=lambda x: x["momentum"], reverse=True)

    # Sector diversification cap
    if SECTOR_MAX_POSITIONS > 0:
        sector_counts: dict[str, int] = {}
        diversified: list[dict] = []
        for cand in buy_candidates:
            sector = get_sector(cand["symbol"])
            bucket = sector if sector != "Unknown" else f"Unknown:{cand['symbol']}"
            if sector_counts.get(bucket, 0) < SECTOR_MAX_POSITIONS:
                diversified.append(cand)
                sector_counts[bucket] = sector_counts.get(bucket, 0) + 1
        buy_candidates = diversified

    # Finalise scan and show candidates in dashboard
    if _ui:
        # Attach p-value and sector to each candidate for display
        for c in buy_candidates:
            c["pvalue"] = _pvalue_cache.get(c["symbol"], 1.0) if WALK_FORWARD_GATE else 1.0
            c["sector"] = get_sector(c["symbol"])
            c["signal"] = "buy"
        _ui.finish_scan(buy_candidates)
        _ui.update_positions(broker.get_positions())

    # 7. Execute buys — fill open slots
    slots   = MAX_POSITIONS - n_open
    to_buy  = buy_candidates[:slots]

    if not to_buy:
        logger.info("No buy signals this cycle")
        return _next_cycle_interval(regime, vix_var)

    vix_level       = (vix_var ** 0.5) * 100.0
    confirmed_bull  = (regime == "bull" and vix_level < VIX_HIGH_THRESHOLD)

    for c in to_buy:
        if confirmed_bull and BULL_BYPASS_PINN:
            weights   = [(1.0 / x["sigma"]) * (max(x["momentum"], 0.0) + 1e-6)
                         for x in ([c] + [x for x in to_buy if x != c])]
            eff_alloc  = 1.0                                 
            eff_equity = equity * BULL_LEVERAGE_FACTOR   
            max_ps     = BULL_MAX_POSITION_SIZE          
        else:
            weights    = [1.0 / c["sigma"]] + [1.0 / x["sigma"] for x in to_buy if x != c]
            eff_alloc  = market_alloc
            eff_equity = equity
            max_ps     = MAX_POSITION_SIZE

        qty = risk.position_size(
            price            = c["price"],
            pinn_allocation  = eff_alloc,
            stock_weights    = weights,
            n_buy_signals    = len(to_buy),
            daily_volume     = c.get("daily_volume", 0.0),
            intraday_vol     = c.get("gk_vol", 0.0),
            equity_override  = eff_equity,
            max_pos_size     = max_ps,
        )
        if qty > 0:
            broker.buy(c["symbol"], qty, limit_price=c["price"])

    logger.info("Cycle complete")
    return _next_cycle_interval(regime, vix_var)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("=" * 60)
    logger.info(f"Autonomous Trading Agent | paper={PAPER_TRADING}")

    broker  = Broker()
    account = broker.get_account()
    risk    = RiskManager(account)

    # ── Terminal UI — suppress all logger output to terminal, file-log only ──
    global _ui
    logger.remove()
    logger.add(LOG_FILE, rotation=LOG_ROTATION, retention=LOG_RETENTION,
               format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
               level="DEBUG")
    _ui = TradingUI()
    _ui.start()

    try:
        sentiment: FreeSentimentEngine | None = FreeSentimentEngine()
        sentiment_worker: SentimentWorker | None = SentimentWorker(sentiment)
    except Exception as e:
        logger.warning(f"Sentiment engine failed to load: {e}")
        sentiment        = None
        sentiment_worker = None

    pinns = load_pinns()

    last_day = -1
    universe: list[str] = FALLBACK_WATCHLIST

    while True:
        if is_new_day(last_day):
            risk.reset_daily_start()
            last_day = datetime.now().day
            try:
                universe = build_universe()
            except Exception as e:
                logger.error(f"Universe build failed: {e}")
            logger.info(
                f"Universe ({len(universe)} symbols): "
                f"{universe[:15]}{'...' if len(universe) > 15 else ''}"
            )

        if is_market_open():
            try:
                sleep_secs = run_cycle(broker, risk, universe, pinns, sentiment,
                                       sentiment_worker)
            except Exception as exc:
                logger.error(f"Cycle error: {exc}", exc_info=True)
                sleep_secs = CYCLE_INTERVAL_SECONDS

            # Start countdown display
            interval_label = ("2h (bull)" if sleep_secs >= 7000
                              else "15-min (bear)")
            if _ui:
                _ui.start_countdown(sleep_secs, interval_label)

            elapsed = 0
            while elapsed < sleep_secs and is_market_open():
                chunk = min(STOP_CHECK_INTERVAL_SECONDS, sleep_secs - elapsed)
                time.sleep(chunk)
                elapsed += chunk
                if _ui:
                    _ui.tick_countdown(elapsed)
                if elapsed < sleep_secs and is_market_open():
                    try:
                        account   = broker.get_account()
                        risk.refresh(account)
                        positions = broker.get_positions()
                        for sym, pos in list(positions.items()):
                            exit_flag, reason = risk.should_exit(pos)
                            if exit_flag:
                                logger.info(f"[exit-check] {sym} [{reason}]")
                                broker.sell(sym)
                                risk.clear_peak(sym)
                    except Exception as exc:
                        logger.warning(f"Exit-check error: {exc}")
        else:
            logger.debug("Market closed — sleeping 5 min")
            time.sleep(300)


if __name__ == "__main__":
    main()