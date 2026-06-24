"""
Risk management: circuit breakers, drawdown limits, trailing-stop exit logic,
PINN-informed position sizing, volatility targeting, and Square-Root Law market-impact.

Exit hierarchy (per position, checked each cycle):
  1. Hard stop-loss  (-3% from entry)        — protects against catastrophic loss
  2. Trailing stop   (-10% from peak price)  — lets winners compound; exits on reversal

Sizing hierarchy (institutional-grade):
  1. PINN π*(t,w,y)  — market-level equity allocation ceiling
  2. Regime γ-scale  — applied in engine.py before reaching here
  3. Vol targeting   — scale by (VOL_TARGET / realized_vol), clipped to [MIN, MAX]
  4. Inv-vol weights — 1/σ per stock distributes risk evenly across positions
  5. Position caps   — 15% equity (bear/neutral) or 25% (bull, via max_pos_size param), 10% cash reserve
"""
from __future__ import annotations
from collections import deque
from datetime import date as _date

import numpy as np
from loguru import logger

from settings import (
    CASH_RESERVE_PCT,
    DAILY_LOSS_CIRCUIT_BREAKER,
    MARKET_IMPACT_THRESHOLD,
    MAX_PORTFOLIO_DRAWDOWN,
    MAX_POSITION_SIZE,
    MIN_POSITION_WEIGHT,
    STOP_LOSS_PCT,
    TAKE_PROFIT_PCT,
    TRAILING_STOP_PCT,
    VOL_LOOKBACK_DAYS,
    VOL_SCALE_MAX,
    VOL_SCALE_MIN,
    VOL_TARGET,
)

_MARKET_IMPACT_Y: float = 0.5


class RiskManager:
    def __init__(self, account):
        self._peak_equity        = float(account.equity)
        self._daily_start_equity = float(account.equity)
        self._account            = account
        self.circuit_breaker_tripped = False
        self._peak_prices: dict[str, float] = {}
        # Volatility targeting — one equity reading per trading day
        self._equity_history: deque[float] = deque(maxlen=VOL_LOOKBACK_DAYS + 1)
        self._vol_last_date: _date | None  = None

    # ------------------------------------------------------------------
    # Account state
    # ------------------------------------------------------------------

    def refresh(self, account) -> None:
        self._account = account
        equity = float(account.equity)
        if equity > self._peak_equity:
            self._peak_equity = equity

    def record_equity(self, equity: float) -> None:
        """Store one equity snapshot per calendar day for realized-vol estimation."""
        today = _date.today()
        if self._vol_last_date != today:
            self._equity_history.append(equity)
            self._vol_last_date = today

    def vol_scale(self) -> float:
        """
        Volatility targeting scale factor = VOL_TARGET / realized_vol,
        clipped to [VOL_SCALE_MIN, VOL_SCALE_MAX].

        Returns 1.0 until at least 5 daily readings are available.
        Applied to total_deployable inside position_size() so every order
        is automatically sized smaller in high-vol periods and larger in calm ones.
        """
        if len(self._equity_history) < 5:
            return 1.0
        vals     = np.array(self._equity_history, dtype=float)
        rets     = np.diff(vals) / vals[:-1]
        real_vol = float(np.std(rets) * np.sqrt(252))
        if real_vol < 1e-6:
            return float(VOL_SCALE_MAX)
        scale = VOL_TARGET / real_vol
        return float(np.clip(scale, VOL_SCALE_MIN, VOL_SCALE_MAX))

    def reset_daily_start(self) -> None:
        self._daily_start_equity = float(self._account.equity)
        self.circuit_breaker_tripped = False
        logger.info(f"Daily start equity reset to ${self._daily_start_equity:,.2f}")

    # ------------------------------------------------------------------
    # Peak price management (trailing stop)
    # ------------------------------------------------------------------

    def update_peak(self, symbol: str, current_price: float) -> None:
        """Ratchet the peak price upward; never downward."""
        self._peak_prices[symbol] = max(
            self._peak_prices.get(symbol, current_price), current_price
        )

    def clear_peak(self, symbol: str) -> None:
        """Call when a position is fully closed so the peak resets on re-entry."""
        self._peak_prices.pop(symbol, None)

    # ------------------------------------------------------------------
    # Circuit breakers
    # ------------------------------------------------------------------

    def check_circuit_breakers(self) -> bool:
        equity = float(self._account.equity)

        daily_loss = (equity - self._daily_start_equity) / self._daily_start_equity
        if daily_loss < -DAILY_LOSS_CIRCUIT_BREAKER:
            logger.warning(
                f"CIRCUIT BREAKER [daily loss]: {daily_loss:.2%} "
                f"exceeds -{DAILY_LOSS_CIRCUIT_BREAKER:.0%}"
            )
            self.circuit_breaker_tripped = True
            return True

        drawdown = (equity - self._peak_equity) / self._peak_equity
        if drawdown < -MAX_PORTFOLIO_DRAWDOWN:
            logger.warning(
                f"CIRCUIT BREAKER [drawdown]: {drawdown:.2%} "
                f"exceeds -{MAX_PORTFOLIO_DRAWDOWN:.0%}"
            )
            self.circuit_breaker_tripped = True
            return True

        return False

    # ------------------------------------------------------------------
    # Per-position exit (trailing stop replaces fixed take-profit)
    # ------------------------------------------------------------------

    def should_exit(self, position) -> tuple[bool, str]:
        """
        Check exit conditions for an open position.

        Priority:
          1. Hard stop-loss  — exit if PnL from entry ≤ -STOP_LOSS_PCT
          2. Trailing stop   — exit if price fell ≥ TRAILING_STOP_PCT from peak
          3. Fixed take-profit — only if TAKE_PROFIT_PCT > 0 (currently disabled)
        """
        symbol  = position.symbol
        cost    = float(position.avg_entry_price)
        current = float(position.current_price)

        # Ratchet peak upward before checking
        self.update_peak(symbol, current)
        peak       = self._peak_prices[symbol]
        pnl        = (current - cost)   / cost
        from_peak  = (current - peak)   / peak

        if pnl <= -STOP_LOSS_PCT:
            return True, f"stop-loss {pnl:.2%} from entry"

        if TRAILING_STOP_PCT > 0 and from_peak <= -TRAILING_STOP_PCT:
            return True, f"trailing-stop {from_peak:.2%} from peak (peak={peak:.2f})"

        if TAKE_PROFIT_PCT > 0 and pnl >= TAKE_PROFIT_PCT:
            return True, f"take-profit {pnl:.2%}"

        return False, ""

    # ------------------------------------------------------------------
    # PINN-informed position sizing + market-impact penalty
    # ------------------------------------------------------------------

    @staticmethod
    def _market_impact(qty: float, daily_volume: float,
                       intraday_vol: float) -> float:
        """I = Y · σ · √(Q/V) — fraction of price."""
        if daily_volume <= 0 or intraday_vol <= 0 or qty <= 0:
            return 0.0
        return float(_MARKET_IMPACT_Y * intraday_vol * np.sqrt(qty / daily_volume))

    def position_size(
        self,
        price: float,
        pinn_allocation: float,
        stock_weights: list[float],
        n_buy_signals: int,
        daily_volume: float = 0.0,
        intraday_vol: float = 0.0,
        equity_override: float | None = None,
        max_pos_size: float = MAX_POSITION_SIZE,
    ) -> float:
        # equity_override: satellite_equity (or satellite × leverage in bull)
        # max_pos_size: overridden to BULL_MAX_POSITION_SIZE in confirmed bull
        equity = equity_override if equity_override is not None else float(self._account.equity)
        cash   = float(self._account.cash)

        # Vol targeting: scale γ-adjusted PINN allocation by realized-vol ratio
        vs               = self.vol_scale()
        scaled_alloc     = float(np.clip(pinn_allocation * vs, 0.0, 1.0))
        total_deployable = equity * scaled_alloc
        logger.debug(f"position_size: π*={pinn_allocation:.3f} × vol_scale={vs:.2f} → {scaled_alloc:.3f}")

        total_w = sum(stock_weights) if stock_weights else 1.0
        if total_w <= 0 or n_buy_signals == 0:
            this_weight = 1.0 / max(n_buy_signals, 1)
        else:
            this_weight = stock_weights[0] / total_w

        target_value   = total_deployable * this_weight
        max_by_equity  = equity * max_pos_size
        max_by_cash    = cash   * (1.0 - CASH_RESERVE_PCT)
        position_value = min(target_value, max_by_equity, max_by_cash)

        # Hysteresis: suppress micro-positions
        if position_value < equity * MIN_POSITION_WEIGHT:
            logger.debug(
                f"Hysteresis: ${position_value:.0f} < "
                f"${equity * MIN_POSITION_WEIGHT:.0f} floor — suppressed"
            )
            return 0.0

        qty = position_value / price if price > 0 else 0.0

        # Square-Root Law market-impact check (uses fixed threshold, not take-profit)
        if daily_volume > 0 and intraday_vol > 0 and qty > 0:
            impact = self._market_impact(qty, daily_volume, intraday_vol)
            if impact > MARKET_IMPACT_THRESHOLD:
                safe_qty = daily_volume * (
                    MARKET_IMPACT_THRESHOLD / max(_MARKET_IMPACT_Y * intraday_vol, 1e-10)
                ) ** 2
                logger.warning(
                    f"Market impact {impact:.4f} > {MARKET_IMPACT_THRESHOLD:.2f} — "
                    f"scaling qty {qty:.4f} → {safe_qty:.4f}"
                )
                qty = min(qty, safe_qty)
                residual = self._market_impact(qty, daily_volume, intraday_vol)
                if residual > 2.0 * MARKET_IMPACT_THRESHOLD:
                    logger.warning(
                        f"Residual impact {residual:.4f} > 2× threshold — suppressed"
                    )
                    return 0.0

        return max(round(qty, 4), 0.0)

    # ------------------------------------------------------------------
    # Market-wide crash signal
    # ------------------------------------------------------------------

    def is_crash_regime(self, spy_signal: dict) -> bool:
        return spy_signal.get("signal") == "sell_all"
