"""
Alpaca paper-trading wrapper. All order submission and account queries go through here.

Order execution strategy:
  - buy()  submits a DAY limit order pegged LIMIT_ORDER_OFFSET_BPS below current price.
           Passive fill reduces crossing the bid-ask spread ("maker" behaviour).
  - cancel_and_market_fill() is called by main.py on orders stale past LIMIT_ORDER_TIMEOUT_MIN
           to guarantee execution before the next cycle.
  - sell() and close_all() always use market orders (urgency takes priority over spread).
"""
from __future__ import annotations

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    GetOrdersRequest,
    LimitOrderRequest,
    MarketOrderRequest,
)
from alpaca.trading.enums import OrderSide, QueryOrderStatus, TimeInForce
from loguru import logger

from settings import (
    ALPACA_API_KEY,
    ALPACA_SECRET_KEY,
    LIMIT_ORDER_ENABLED,
    LIMIT_ORDER_OFFSET_BPS,
    PAPER_TRADING,
)


class Broker:
    def __init__(self):
        if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
            raise EnvironmentError(
                "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in .env"
            )
        self._client = TradingClient(
            api_key=ALPACA_API_KEY,
            secret_key=ALPACA_SECRET_KEY,
            paper=PAPER_TRADING,
        )
        mode = "PAPER" if PAPER_TRADING else "LIVE"
        logger.info(f"Broker initialised — {mode} trading")

    # ------------------------------------------------------------------
    # Account / portfolio queries
    # ------------------------------------------------------------------

    def get_account(self):
        return self._client.get_account()

    def get_positions(self) -> dict:
        """Returns {symbol: position_object} for all open positions."""
        return {p.symbol: p for p in self._client.get_all_positions()}

    def get_open_orders(self) -> list:
        """Return all open (pending/partially-filled) orders."""
        try:
            return list(
                self._client.get_orders(
                    filter=GetOrdersRequest(status=QueryOrderStatus.OPEN)
                )
            )
        except Exception as e:
            logger.warning(f"get_open_orders failed: {e}")
            return []

    # ------------------------------------------------------------------
    # Order execution
    # ------------------------------------------------------------------

    def buy(self, symbol: str, qty: float, limit_price: float | None = None) -> bool:
        """
        Submit a buy order.

        If LIMIT_ORDER_ENABLED and limit_price is provided, submits a DAY limit
        order LIMIT_ORDER_OFFSET_BPS below limit_price (passive fill).
        Otherwise falls back to a market order.
        """
        if qty <= 0:
            logger.warning(f"BUY skipped — qty={qty} for {symbol}")
            return False
        try:
            if LIMIT_ORDER_ENABLED and limit_price is not None:
                pegged = round(limit_price * (1 - LIMIT_ORDER_OFFSET_BPS / 10_000), 2)
                order  = self._client.submit_order(
                    LimitOrderRequest(
                        symbol=symbol,
                        qty=qty,
                        side=OrderSide.BUY,
                        time_in_force=TimeInForce.DAY,
                        limit_price=pegged,
                    )
                )
                logger.info(
                    f"LIMIT BUY  {qty:>8.4f} {symbol:<6} "
                    f"@ {pegged:.2f} (−{LIMIT_ORDER_OFFSET_BPS}bps) | order_id={order.id}"
                )
            else:
                order = self._client.submit_order(
                    MarketOrderRequest(
                        symbol=symbol,
                        qty=qty,
                        side=OrderSide.BUY,
                        time_in_force=TimeInForce.DAY,
                    )
                )
                logger.info(f"MARKET BUY {qty:>8.4f} {symbol:<6} | order_id={order.id}")
            return True
        except Exception as e:
            logger.error(f"BUY failed {symbol}: {e}")
            return False

    def cancel_and_market_fill(self, symbol: str, order_id: str, qty: float) -> bool:
        """
        Cancel a stale limit order then immediately submit a market order.
        Called by main.py when a limit order exceeds LIMIT_ORDER_TIMEOUT_MIN.
        """
        try:
            self._client.cancel_order_by_id(order_id)
            logger.info(f"Cancelled stale limit order {order_id} for {symbol}")
        except Exception as e:
            logger.warning(f"Cancel order {order_id} failed (may already be filled): {e}")
        return self.buy(symbol, qty, limit_price=None)   # market fallback

    def sell(self, symbol: str, qty: float | None = None) -> bool:
        """Sell `qty` shares via market order, or close the entire position."""
        try:
            if qty is not None and qty > 0:
                order = self._client.submit_order(
                    MarketOrderRequest(
                        symbol=symbol,
                        qty=qty,
                        side=OrderSide.SELL,
                        time_in_force=TimeInForce.DAY,
                    )
                )
                logger.info(f"SELL {qty:>8.4f} {symbol:<6} | order_id={order.id}")
            else:
                self._client.close_position(symbol)
                logger.info(f"SELL ALL {symbol}")
            return True
        except Exception as e:
            logger.error(f"SELL failed {symbol}: {e}")
            return False

    def close_all(self) -> None:
        logger.warning("CLOSE ALL POSITIONS + cancel open orders")
        try:
            self._client.close_all_positions(cancel_orders=True)
        except Exception as e:
            logger.error(f"close_all failed: {e}")

    def cancel_all_orders(self) -> None:
        try:
            self._client.cancel_orders()
            logger.info("All open orders cancelled")
        except Exception as e:
            logger.error(f"cancel_orders failed: {e}")
