import os
from dotenv import load_dotenv

load_dotenv()

# --- Alpaca credentials (paper trading) ---
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
PAPER_TRADING = True

# --- Universe (dynamic, rebuilt daily) ---
UNIVERSE_SIZE = 150             # max symbols for screener/Alpaca fallback paths only; CSV primary returns full ~1,900
UNIVERSE_MIN_PRICE = 5.0        # exclude penny stocks
UNIVERSE_MIN_AVG_VOLUME = 1_000_000  # minimum 1M avg daily shares
SIGNAL_LOOKBACK_DAYS = 260      # OHLCV lookback: extended to support 12-1 momentum score

# Primary universe source: local CSV, validated against Alpaca tradable assets.
# Set CSV_UNIVERSE_MAX_SYMBOLS = 0 to allow the full validated CSV universe.
CSV_UNIVERSE_ENABLED: bool = True
CSV_UNIVERSE_PATH: str = "stock_info.csv"
CSV_UNIVERSE_TICKER_COLUMN: str = "Ticker"
CSV_UNIVERSE_MAX_SYMBOLS: int = 0
CSV_UNIVERSE_REQUIRE_FRACTIONABLE: bool = True

# Emergency fallback if all universe sources fail
FALLBACK_WATCHLIST = ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA", "AMD", "SPY", "QQQ"]

# --- Universe batch rotation ---
# The screener can return 200-500+ symbols.  Evaluating all of them every 15-min
# cycle is slow (one OHLCV + 1-min fetch per symbol).  Instead, rotate through
# the universe in UNIVERSE_BATCH_SIZE chunks — each batch gets a full signal pass,
# while the rest are skipped until their turn comes around.
# Positions are always exit-scanned regardless of which batch is active.
# Set to 0 to disable batching (evaluate all symbols every cycle).
UNIVERSE_BATCH_SIZE: int = 50

# --- Walk-forward validation ---
# HMM is expensive to refit (full EM on 504 bars). Lock the fitted model for
# HMM_REFIT_DAYS trading days; only refit when the fold advances.
# The 2-cycle RegimeSmoother still runs every cycle for label stability.
HMM_REFIT_DAYS: int = 7     # refit HMM once per week of trading days

# Monte Carlo permutation test gate.
# Before adding a symbol to buy_candidates, test if its EMA signal has
# statistically significant edge vs. shuffled noise. Symbols failing the
# gate are silently skipped for the rest of that trading day.
PERMUTATION_N: int           = 100    # permutations per symbol (fast: ~2ms each)
PERMUTATION_PVALUE_GATE: float = 0.05  # exclude symbols with p-value >= this
WALK_FORWARD_GATE: bool      = True   # set False to disable gate entirely

# --- HMM regime detection ---
HMM_N_STATES = 3           # bull / bear / crash
HMM_LOOKBACK_DAYS = 504    # ~2 years of training data
HMM_MODEL_PATH = "models/hmm_spy.pkl"

# --- Technical indicators ---
RSI_PERIOD = 14
EMA_FAST = 20
EMA_SLOW = 50
VOLUME_MA_PERIOD = 20

# --- Bull-mode aggressive parameters ---
BULL_BYPASS_PINN: bool    = True   # ignore PINN ceiling in confirmed bull; deploy 100% of sleeve
BULL_LEVERAGE_FACTOR: float = 1.5  # notional leverage on satellite in bull (150% of sleeve)
BULL_MAX_POSITION_SIZE: float = 0.25  # 25% of (leveraged) satellite per position in bull

# --- Position sizing & risk ---
MAX_PORTFOLIO_DRAWDOWN = 0.10   # halt all trading if portfolio down 10%
MAX_POSITION_SIZE = 0.15        # 15% per position in bear/neutral
STOP_LOSS_PCT = 0.03            # hard stop: exit if position down 3% from entry
TRAILING_STOP_PCT = 0.10        # trailing stop: exit if down 10% from peak price
TAKE_PROFIT_PCT = 0.0           # DISABLED — trailing stop replaced fixed take-profit
MAX_POSITIONS = 14              # max concurrent satellite positions
CASH_RESERVE_PCT = 0.10         # keep 10% of satellite budget in cash

# Market impact threshold for position sizing (separate from take-profit)
MARKET_IMPACT_THRESHOLD = 0.06  # suppress if impact cost > 6% of position value

# --- Sector diversification cap ---
# Raised from 3 to 5 (personal model — allows concentrated participation in mega-trends
# like the 2021-2024 AI/semiconductor run without being capped at 3 positions).
SECTOR_MAX_POSITIONS: int = 5   # max open positions per GICS sector at any time

# --- Dynamic risk-free rate ---
# Fetches current 3-month T-bill yield (^IRX) once per day for Merton ratio.
# Corrects for high-rate regimes (2022-2023: r=5.3%) where equity risk premium
# shrinks and the model should be less aggressive.
RISK_FREE_RATE_DYNAMIC: bool = True

# --- Intraday exit check interval ---
# Full allocation cycles throttle to 2h in calm bull markets (CYCLE_INTERVAL_BULL_SECONDS).
# Trailing stops must still be monitored at higher frequency to avoid flash-crash slippage.
# Between full cycles, a lightweight exit-only check runs every N seconds.
STOP_CHECK_INTERVAL_SECONDS: int = 300  # 5-min intraday exit scan cadence

# --- Circuit breakers ---
DAILY_LOSS_CIRCUIT_BREAKER = 0.05   # halt if portfolio drops 5% in one day
MARKET_CRASH_SPY_THRESHOLD = -0.03  # SPY 1-day return below -3% = crash

# --- Regime-aware risk aversion (dynamic γ) ---
# PINNs are trained at CRRA_GAMMA = 3.0. In confirmed bull markets we apply a
# runtime scale of (CRRA_GAMMA / CRRA_GAMMA_BULL) to π*, raising effective
# deployment without retraining. The Merton ratio is linear in 1/γ so the
# approximation is exact for Merton and close for the full HJB solution.
CRRA_GAMMA_BULL: float = 1.5        # effective γ in confirmed bull — doubles π*

# --- Portfolio volatility targeting ---
# Scale total allocation up/down each cycle to maintain target annualized vol.
# This is standard at AQR, BlackRock (MVCS), Bridgewater (All Weather).
VOL_TARGET: float          = 0.15   # target annualized portfolio volatility
VOL_LOOKBACK_DAYS: int     = 20     # rolling window for realized vol estimate
VOL_SCALE_MIN: float       = 0.5    # never reduce allocation below 50% of π*
VOL_SCALE_MAX: float       = 1.5    # never increase above 150% of γ-scaled π*

# --- PINN / Optimal Control (from paper) ---
CRRA_GAMMA = 3.0                    # CRRA risk aversion γ (paper default, PINN training)
RISK_FREE_RATE = 0.05               # annual risk-free rate r
TRADING_HORIZON_YEARS = 0.5         # T in the HJB (6-month rolling horizon)
PINN_HIDDEN_UNITS = 128             # neurons per hidden layer
PINN_DEPTH = 4                      # hidden layers per network
PINN_EPOCHS = 10_000                # training epochs (per paper)
PINN_BATCH_SIZE = 2048              # LHS collocation points per step (per paper)
PINN_LR = 1e-3                      # Adam learning rate
PINN_ALPHA = {                      # composite loss weights — no BC term (CRRA ansatz)
    "hjb":  1.0,
    "stat": 1.0,
    "comp": 0.1,
    "stab": 0.01,
    "lb":   0.5,   # lower-bound penalty: prevents π*→0 degenerate fixed point
}
GBM_PINN_PATH    = "models/gbm_pinn.pt"
HESTON_PINN_PATH = "models/heston_pinn.pt"
REGIME_PINN_PATH = "models/regime_pinn.pt"

# --- Account baseline ---
# Normalized wealth fed to the PINN: w = equity / INITIAL_EQUITY
# PINN trained with w=1.0 representing this starting capital.
# Update if the paper account is reset or a different account is used.
INITIAL_EQUITY: float = 100_000.0

# --- Hysteresis buffer (Step 1) ---
# Suppress new positions whose target weight is below this fraction of equity.
# Prevents microscopic trades from being generated when Merton weights drift slightly.
MIN_POSITION_WEIGHT: float = 0.015   # 1.5% of equity minimum position size

# --- Limit order execution (Step 2) ---
LIMIT_ORDER_ENABLED: bool  = True    # submit limit orders; fall back to market on timeout
LIMIT_ORDER_OFFSET_BPS: int = 2     # peg limit this many bps below current price (passive fill)
LIMIT_ORDER_TIMEOUT_MIN: int = 4    # cancel & market-fill if unfilled after this many minutes

# --- Dynamic cycle frequency (Step 3) ---
CYCLE_INTERVAL_BULL_SECONDS: int = 7200   # 2 h in confirmed bull + low VIX
VIX_HIGH_THRESHOLD: float        = 25.0   # VIX above this → always use 15-min interval

# --- Time-scale alignment ---
# PINN π*(t,w,y) changes on the scale of days (t moves by ~1/252 per day).
# Re-querying every 15 min is redundant; treat it as a daily strategic ceiling.
# Re-query on date change OR macro regime change (bull→bear etc.).
PINN_DAILY_CACHE = True             # cache daily PINN allocation; refresh on date/regime change

# 1-minute yfinance data freshness gate.
# Lead-lag front-running is invalid on stale bars; skip if data is older than this.
INTRADAY_MAX_STALE_MIN = 5          # max minutes since last 1-min bar before skipping lead-lag

# Sentiment rolling-window cache.
# Peak absolute FinBERT score within the last N minutes is used, not a snapshot.
# Prevents big news at minute 1 being 50% decayed by minute 14 of the same cycle.
SENTIMENT_WINDOW_MIN = 15           # rolling window length in minutes

# --- Scheduling ---
CYCLE_INTERVAL_SECONDS = 900    # run every 15 minutes during market hours
MARKET_TIMEZONE = "America/New_York"

# --- Logging ---
LOG_FILE = "logs/trade_journal.log"
LOG_ROTATION = "1 day"
LOG_RETENTION = "90 days"
