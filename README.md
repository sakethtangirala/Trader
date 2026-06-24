# Autonomous Trading Agent

Paper trading agent implementing continuous-time portfolio optimization via Physics-Informed Neural Networks. Built by Saketh Tangirala (Georgia Tech CS) based on his co-authored paper submitted to *Finance Research Letters*.

Zero runtime API cost. Free Alpaca paper account.

---

## Research Foundation

> *"Portfolio Optimization in Continuous Time: A Physics-Informed Neural Network Approach"*  
> Kopeliovich, Malur, Pokojovy, Tangirala — *Finance Research Letters* (submitted)

The agent solves the Hamilton–Jacobi–Bellman equation for CRRA utility maximization under GBM, Heston stochastic volatility, and Markov regime-switching using a PINN with CRRA ansatz:

```
V(t, w) = w^(1-γ)/(1-γ) · (1 + (T-t) · ψ(t, w))
```

Terminal condition is structurally enforced — no BC loss term. Training loss:

```
L = 1.0·L_HJB + 1.0·L_stat + 0.1·L_comp + 0.01·L_stab + 0.5·L_lb
```

`L_lb = mean(relu(Merton×0.3 − π*)²)` — lower-bound penalty that prevents convergence to the degenerate π*=0 fixed point. Combined with `PolicyNet` final-layer bias initialisation at +1.5 (σ(1.5)≈0.82), this anchors the policy near the Merton solution from epoch 0.

---

## Project Structure

```
Trader/
├── main.py               # Orchestration loop, _PINNCache, market-hours scheduler
├── engine.py             # Data fetch, HMM, GK volatility, PINN allocation, momentum, signals
├── risk_manager.py       # Circuit breakers, vol targeting, position sizing, market impact
├── brokerage.py          # Alpaca TradingClient wrapper
├── universe.py           # Daily universe builder (stock_info.csv primary; screener + Alpaca fallback)
├── stock_info.csv        # ~1,900 Alpaca-validated tickers (pre-validated, no API call needed)
├── sentiment.py          # FinBERT + RSS + keyword filter + decay + peak cache
├── settings.py           # All constants (single source of truth)
├── train_pinns.py        # PINN training script (run once)
├── research.py           # Quantitative research loop — 11-iteration scientific protocol
├── validation.py         # Monte Carlo permutation test + walk-forward validation framework
├── backtest_engine.py    # Event-driven high-fidelity backtester (Brownian-bridge intraday sim)
├── backtest.py           # Simplified daily-bar historical backtester (3 periods)
├── ui.py                 # Rich terminal dashboard (live scan progress, positions, countdown)
├── pinn/
│   ├── networks.py  # MLP, PolicyNet, IneqMultiplierNet
│   ├── sampler.py   # Latin Hypercube Sampling
│   ├── gbm.py       # GBM PINN — GBMPINN, GBMParams
│   ├── heston.py    # Heston PINN — HestonPINN, HestonParams
│   └── regime.py    # Regime PINN — RegimePINN, RegimeParams
├── models/
│   ├── gbm_pinn.pt      # Trained (final loss 4.5e-7)
│   ├── heston_pinn.pt   # Trained (final loss 6.5e-7)
│   └── regime_pinn.pt   # Trained (final loss 2.2e-5)
├── ml_references/   # Prior HJB-PINN implementations (reference only)
├── graphs/          # Output charts from research.py, backtest.py, backtest_engine.py
├── logs/
│   └── trade_journal.log
├── .env             # ALPACA_API_KEY, ALPACA_SECRET_KEY (never commit)
└── requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env          # fill in Alpaca paper trading keys
python train_pinns.py         # ~30 min CPU — run once, produces models/*.pt
python main.py
```

FinBERT (~500 MB) downloads on first run and caches locally. If `models/*.pt` are missing, the agent falls back to Merton ratio + RSI/EMA rules without crashing.

---

## How a Cycle Works

Every 15 minutes in bear/high-VIX, every 2 hours in confirmed bull + VIX < 25 (09:30–16:00 ET, weekdays):

**1. Circuit breakers**
- Daily loss > 5% → cancel all orders, skip cycle
- Peak drawdown > 10% → cancel all orders, skip cycle

**2. VIX + SPY 1-min data**
- `fetch_vix()` returns `(VIX/100)²` as the Heston variance proxy `y`
- `fetch_1min("SPY", bars=30)` fetches the last 30 one-minute bars

**3. SPY hard crash backstop + HMM macro regime**

*Hard backstop (fires before HMM):*
- Compute SPY 1-day return from last two daily closes
- If return ≤ `MARKET_CRASH_SPY_THRESHOLD` (−3%) → `close_all()` immediately, return
- Rule-based; not subject to smoothing or ML classification

*HMM walk-forward lock + 2-cycle confirmation (`_RegimeSmoother`):*
- 3-state Gaussian HMM refitted at most every `HMM_REFIT_DAYS = 7` trading days (walk-forward fold advance). Between refits the cached raw label is fed into the smoother — prevents intraday HMM noise from flipping regime every 15 minutes.
- Features: daily return, 20-day rolling vol, volume ratio
- States ranked by mean return → **bull** / **bear** / **crash**
- Raw label still passed through `_regime_smoother.update()` each cycle — confirmed regime only changes after 2 consecutive identical readings, preventing single-candle covariance spikes
- HMM crash liquidation (→ `close_all()`) requires 2 consecutive crash readings

**4. Daily PINN allocation (`_PINNCache`)**
- `get_pinn_allocation()` queries the best available PINN for `π*(t, w, y)`
- Priority: Heston (VIX-aware) → Regime (transition-risk-aware) → GBM → Merton ratio
- `_PINNCache` caches the result by `(date, regime)` — re-queries only on date change or regime flip, since `t` advances ~1/252 per calendar day

**5. Exit scan**
- Per open position, checked in order: (1) hard stop-loss — exit if P&L ≤ −3% from entry; (2) trailing stop — exit if price ≤ peak × (1 − 10%)
- Fixed take-profit is disabled (research iter 1: replacing +6% take-profit with trailing stop tripled Sharpe across all periods)

**6. Universe scan (batch rotation)**
The universe is divided into `UNIVERSE_BATCH_SIZE = 50` chunks and rotated each cycle — only the active batch gets a full OHLCV + signal pass; positions are always exit-scanned regardless. Allows scaling to 200–500 symbols without ballooning cycle time. The background `SentimentWorker` thread pre-warms FinBERT scores for the *next* batch while the current cycle runs.

For each symbol in the active batch:
- Fetch 260 days daily OHLCV + 30 one-minute bars (260 days supports 12-1 month momentum score)
- Run `compute_signals()` which applies in order:
  - **GK micro-halt**: if 1-min GK acceleration detected → force `hold`
  - **EMA trend + PINN**: bull: EMA20 > EMA50 and PINN π* > 0.05 → buy (no RSI ceiling); bear has two entry paths: (1) RSI < 30 and EMA20 > EMA50 and PINN π* > 0.05 (primary oversold entry); (2) RSI < 35 and price > EMA20 and PINN π* > 0.05 (recovery entry — catches early recovery before slow EMA crossover fires; research iter 6, accepted)
  - **FinBERT guardrails**: buy + sentiment < −0.20 → hold; sell + sentiment < −0.60 → accelerate sell
- `compute_momentum()` produces a 12-1 month momentum score per stock (12-month return excluding last month to avoid reversal bias)
- **Monte Carlo p-value gate** (`WALK_FORWARD_GATE = True`): before a buy candidate is accepted, `quick_pvalue(close, n=100)` runs 100 return-shuffle permutations. If p-value ≥ `PERMUTATION_PVALUE_GATE (0.05)`, the EMA signal is statistically indistinguishable from noise and the symbol is skipped for the rest of the trading day. Results cached per symbol per day.

**7. Sizing + execution — dual-mode (bull vs bear)**

*Confirmed bull (`regime == "bull"` and `VIX < 25`):*
- `BULL_BYPASS_PINN = True` → `eff_alloc = 1.0` (PINN ceiling bypassed; full deployment)
- `eff_equity = equity × BULL_LEVERAGE_FACTOR` (1.5× notional leverage on full equity)
- Weights = `(1/σ_i) × (momentum_i + ε)` — momentum × inv-vol hybrid; fast-moving names with moderate vol receive the largest allocation
- `max_pos_size = BULL_MAX_POSITION_SIZE = 0.25` (25% of leveraged equity per position)

*Bear / neutral (all other regimes):*
- `eff_alloc = max(PINN π*, per-stock Merton ratio)` — Merton floor prevents PINN calibration gaps from silencing buys on fundamentally strong stocks (mirrors research.py Iter 10 exactly: `eff = max(alloc, m)`)
- `eff_equity = equity`; no leverage
- Pure inverse-vol weights: `1/σ_i / Σ(1/σ_j)`
- `max_pos_size = MAX_POSITION_SIZE = 0.15`

*Both modes:*
- **Sector cap** (`SECTOR_MAX_POSITIONS = 5`): at most 5 positions per GICS sector
- Fill up to 14 open positions
- Market impact check: `I = 0.5 · σ_GK · √(qty/volume)` — if I > 6% scale qty down; if scaled I > 12% suppress order entirely
- Hard cash cap: `position_value ≤ cash × 0.90` always enforced

**Between full cycles — intraday exit scan:**
- Full allocation cycles throttle to 2h in confirmed bull + VIX < 25
- Trailing stops checked every `STOP_CHECK_INTERVAL_SECONDS = 300` s (5 min) between cycles via a lightweight exit-only loop — no signal computation, no universe scan
- Prevents flash-crash slippage where a −10% intraday drop recovers before the next 2h cycle fires

---

## Strategy Layers

### Layer 1 — HMM Macro Regime
Three-state Gaussian HMM on 504 days of SPY daily data. State-to-label mapping is dynamic (sorted by mean return each fit). Persisted to `models/hmm_spy.pkl`.

**Walk-forward fold lock** (`HMM_REFIT_DAYS = 7`): the HMM is expensive to refit (full EM on 504 bars). Model parameters are locked for 7 trading days; only the fold boundary triggers a refit. Between refits, the cached raw label from the last fit is fed into the smoother each cycle. This eliminates intraday HMM churn where micro-volatility spikes would flip the regime label every 15 minutes.

**`_RegimeSmoother`** (2-cycle confirmation): raw HMM labels are not used directly even after a refit. Confirmed regime only changes when the last 2 consecutive readings agree. Prevents single volatile candles from triggering full reallocations or clearing the PINN cache.

**Hard crash backstop**: a rule-based SPY 1-day return check (`≤ −3%`) fires before the HMM each cycle. If triggered, all positions close immediately — no ML confirmation required.

### Layer 2 — PINN Optimal Allocation + Regime-Aware γ Scaling
Queries the best loaded PINN for market-level equity fraction `π*(t, w, y)`:

| Priority | PINN | State space | Key feature |
|---|---|---|---|
| 1 | Heston | (t, w, y) | VIX-aware; cross-derivative V_wy via ρ = −0.718 |
| 2 | Regime | (t, w) × 2 | Coupled V⁰/V¹ via generator Q̂ — transition risk |
| 3 | GBM | (t, w) | Baseline |
| 4 | Merton ratio | analytical | (μ−r)/(γσ²) — PINN allocation fallback only; not used for per-stock sizing |

Note: per-stock *sizing* uses inverse-volatility weights (`1/σ_i / Σ 1/σ_j`) — Merton is not used for sizing. Merton appears only as a VIX-attenuated floor on the total PINN allocation to guard against calibration gaps: `floor_scale = clip(1 − 0.03×(VIX−15), 0.25, 1.0)`. At VIX=15 the floor is full Merton; at VIX=40+ it decays to 25% of Merton, preventing the constant-σ Merton estimate from overriding the Heston PINN's stochastic-vol signal in high-fear regimes.

`t = min(days_to_year_end / 252, 0.5)` — CRRA paper horizon T = 0.5 years.  
`w = equity / INITIAL_EQUITY` — normalized wealth sourced from `settings.py`.

**Regime-aware γ scaling** (`engine.py`, applied in `compute_signals()`): PINNs are trained at CRRA γ=3.0. In confirmed bull markets the raw π* is scaled by `CRRA_GAMMA / CRRA_GAMMA_BULL = 3.0 / 1.5 = 2×`, raising effective deployment from ~42% to ~84% without retraining. Bear and crash regimes use γ=3.0 unchanged. The Merton ratio is exactly linear in 1/γ; the PINN approximation follows closely.

**PINN architecture fix** (`pinn/networks.py`, `pinn/gbm.py`, `pinn/heston.py`, `pinn/regime.py`): the `PolicyNet` previously converged to the degenerate fixed point π*=0 — a valid HJB solution (no trading) with low residual loss that made the agent trade nothing. Two fixes: (1) `PolicyNet` final-layer bias initialised to +1.5 so `sigmoid(1.5)≈0.82` at epoch 0, anchoring near the Merton solution from the start; (2) lower-bound penalty `L_lb = mean(relu(Merton×0.3 − π*)²)` added to all three PINNs, penalising solutions where π* falls below 30% of the analytical Merton ratio. The `compute_signals()` Merton floor (`max(PINN, Merton)`) handles any residual calibration gaps at inference time.

**Dynamic risk-free rate** (`engine.py: fetch_risk_free_rate()`): fetches the 3-month T-bill yield (`^IRX`) from yfinance once per calendar day and caches the result. Used in `compute_merton_ratio()` as the risk-free rate `r`, replacing the static 5%. This corrects the Merton fallback for high-rate regimes (e.g. r=5.3% in 2023 meaningfully shrinks the equity risk premium μ−r and should reduce allocation). PINN models have r=5% baked into training; dynamic r only affects the Merton fallback path.

Result cached in `_PINNCache` keyed by `(date, confirmed_regime, vix_bucket)`. VIX is bucketed to the nearest 2 points so an intraday spike (e.g. 18 → 28) invalidates the cache and triggers a fresh PINN query instead of serving the morning's calm-market allocation all day.

### Layer 3 — Intraday Microstructure

**Garman-Klass Volatility** (per 1-min bar):
```
GK = 0.5·(ln H/L)² − (2·ln2 − 1)·(ln C/O)²
```
Daily vol estimate: `σ_GK = √(mean(GK) × 390)`

**Micro-Regime Halt** — computed in `check_vol_acceleration()`:
- Fit linear slope to `Δ(GK)` over last 15 bars
- If slope > 0 **and** `GK[-1] / mean(GK) > 3.0` → return `True` → signal forced to `hold`
- Overrides HMM regime; does not override crash exits or stop-losses

**Lead-Lag Cross-Correlation** — computed in `lead_lag_correlation()`:
- 1-min returns of stock vs SPY at lags [−5, …, +5] minutes
- Returns `optimal_lag` (int) and `peak_corr` (float)
- **Gated by `check_1min_freshness()`**: if latest 1-min bar is > 5 min old, lead-lag is suppressed (returns zeros) to avoid acting on stale yfinance cache

### Layer 4 — FinBERT Sentiment

**Keyword pre-filter** (`_is_relevant()`):
```python
r"\b(halt|investigation|acquisition|restructure|bankruptcy|fda|ceo|ch11|fraud)\b"
```
Headline must match the regex OR contain the ticker symbol — otherwise dropped before FinBERT runs.

**Exponential decay**: `score × e^(−0.05 × age_minutes)` — half-life ≈ 14 min.

**Rolling peak cache** (`_peak_cache`): within a 15-min window, retains the highest-magnitude score seen. If breaking news arrives at minute 1 of a cycle, it remains fully weighted at minute 14.

**Guardrails** (`apply_guardrails()`):
- `buy` + sentiment < −0.20 → `hold`
- `sell` + sentiment < −0.60 → `sell` (pass-through, already selling)

**`SentimentWorker`** (`sentiment.py`): background daemon thread that pre-warms FinBERT scores for the *next* batch while the main cycle executes the current one. Tickers are submitted to a `queue.Queue`; the worker scores them continuously and writes to a shared dict. The main cycle reads from the dict instantly with no FinBERT blocking. Deduplication: re-submitting an in-flight ticker is a no-op.

### Layer 5 — Sizing, Volatility Targeting, Impact + Execution

**Position sizing — confirmed bull** (`regime == "bull"` and `VIX < 25`):
```
eff_alloc        = 1.0                              [PINN bypassed — full deployment]
eff_equity       = equity × 1.5                    [150% notional leverage on full equity]
vol_scale        = clip(VOL_TARGET / realized_20d_vol, 0.5, 1.5)
deployed         = eff_equity × clip(eff_alloc × vol_scale, 0, 1)
hybrid_weight_i  = (1/σ_i) × (momentum_i + ε) / Σ[(1/σ_j) × (momentum_j + ε)]
target_value_i   = deployed × hybrid_weight_i
qty_i            = target_value_i / price_i         [cap: 25% of eff_equity]
```

**Position sizing — bear / neutral:**
```
pinn_alloc       = max(π*(PINN), Merton_i)          [Merton floor per stock — Iter 10 match]
γ_scaled         = pinn_alloc × (CRRA_GAMMA / CRRA_GAMMA_BULL)  [bull γ-scale, if applicable]
vol_scale        = clip(VOL_TARGET / realized_20d_vol, 0.5, 1.5)
deployed         = equity × clip(γ_scaled × vol_scale, 0, 1)
inv_vol_weight_i = (1/σ_i) / Σ(1/σ_j)
target_value_i   = deployed × inv_vol_weight_i      [cap: 15% of equity]
qty_i            = target_value_i / price_i
```

**Volatility targeting** (`risk_manager.py: vol_scale()`): `RiskManager.record_equity()` stores one equity snapshot per *trading* day (guarded by `_vol_last_date != today`). `vol_scale()` computes 20-trading-day realized annualized vol (`std(daily_returns) × √252`) from those snapshots and returns `clip(15% / realized_vol, 0.5, 1.5)`. Applied to the γ-scaled PINN allocation before sizing — automatically de-levers in high-vol periods and increases deployment in calm bull markets. Standard technique at AQR, BlackRock MVCS, Bridgewater.

**Inverse-volatility weighting** (`main.py`, bear/neutral only): per-stock weight is `1/σ_i / Σ(1/σ_j)`. In confirmed bull the system switches to momentum × inv-vol hybrid weights `(1/σ_i × momentum_i)` to concentrate in fast movers. Merton-ratio sizing is retired from both modes — μ estimates are noise-dominated at 260-day horizons. Research iter 10: test Sharpe 0.48 → 0.53, CAGR 9.5% → 10.1%.

**Hysteresis buffer** (`MIN_POSITION_WEIGHT = 0.015`): if `target_value < equity × 1.5%`, the order is suppressed entirely. Prevents microscopic rebalancing trades whose spread and impact cost exceed any alpha.

**Square-Root Law market impact** (`_market_impact()`):
```
I = 0.5 · σ_GK · √(qty / daily_volume)
```
- If `I > 0.06` (`MARKET_IMPACT_THRESHOLD`): scale qty to `volume × (0.06 / (0.5 · σ_GK))²`
- If scaled `I > 0.12` (2× threshold): suppress order entirely

Hard caps always applied last:
- Bull: `position_value ≤ (equity × 1.5) × 0.25` ≈ 37.5% of equity per position (cash constraint is the binding limit in practice)
- Bear: `position_value ≤ equity × 0.15`
- Always: `position_value ≤ cash × 0.90` (10% cash reserve enforced regardless of mode)

**Limit order execution** (`brokerage.py`): buys are submitted as DAY limit orders pegged `LIMIT_ORDER_OFFSET_BPS = 2` bps below the current close price, inviting passive fill from natural intraday volatility. At the start of each cycle, `_sweep_stale_orders()` checks all open orders — any limit order older than `LIMIT_ORDER_TIMEOUT_MIN = 4` min is cancelled and resubmitted as a market order to guarantee execution before the next bar.

**Dynamic cycle frequency** (`_next_cycle_interval()`):
| Condition | Sleep interval |
|---|---|
| Confirmed bull + VIX < 25 | 2 hours (`CYCLE_INTERVAL_BULL_SECONDS`) |
| Bear / VIX ≥ 25 / crash | 15 minutes (`CYCLE_INTERVAL_SECONDS`) |

In calm bull markets the loop throttles to once every 2 hours, reducing transactional overhead when portfolio drift is minimal. VIX ≥ 25 always forces 15-minute cadence regardless of regime.

---

## Risk Controls

| Control | Value | Where enforced |
|---|---|---|
| SPY hard crash backstop | SPY 1-day return ≤ −3% | `main.py: run_cycle()` — fires before HMM |
| HMM crash confirmation | 2 consecutive crash readings | `main.py: _RegimeSmoother` |
| Daily loss circuit breaker | −5% | `risk_manager.py: check_circuit_breakers()` |
| Peak drawdown circuit breaker | −10% | `risk_manager.py: check_circuit_breakers()` |
| Stop-loss per position | −3% from entry | `risk_manager.py: should_exit()` |
| Trailing stop per position | −10% from peak price | `risk_manager.py: should_exit()` — replaces fixed take-profit |
| Micro-regime halt | GK last/mean > 3× and slope > 0 | `engine.py: check_vol_acceleration()` |
| Hysteresis buffer | target < 1.5% equity | `risk_manager.py: position_size()` |
| Market impact scale-down | I > 6% | `risk_manager.py: position_size()` |
| Market impact suppression | scaled I > 12% | `risk_manager.py: position_size()` |
| Stale limit order sweep | order age > 4 min → market-fill | `main.py: _sweep_stale_orders()` |
| Max open positions | 14 | `main.py: run_cycle()` |
| Max position size | 15% of equity (bear) / 25% of leveraged equity (bull) | `risk_manager.py: position_size()` via `max_pos_size` |
| Cash reserve | 10% always | `risk_manager.py: position_size()` |
| Sector diversification cap | 5 positions per GICS sector | `main.py: run_cycle()` via `get_sector()` |
| Intraday exit scan | every 5 min between full cycles | `main.py: main()` sleep loop |
| Paper trading lock | `PAPER_TRADING = True` | `settings.py`, `brokerage.py` |

---

## PINN Training Details

```bash
python train_pinns.py   # ~30 min CPU, ~5 min GPU
```

Parameters in `train_pinns.py` (updated for modern market dynamics):

| Model | Parameters | Note |
|---|---|---|
| GBM | μ=0.155, σ=0.178, r=0.05, γ=3, T=0.5 | μ raised from paper's 0.128 (modern S&P 500 realized drift ~15%) |
| Heston | μ=0.160, κ=6.586, θ=0.038, σ_Y=0.643, ρ=−0.718 | μ raised from 0.132; vol surface params unchanged |
| Regime (risk-on) | μ=0.210, σ=0.223 | μ raised from 0.185 (modern bull regime is structurally faster) |
| Regime (risk-off) | μ=0.401, σ=0.362 | unchanged — crash dynamics are not faster than historical |
| Regime Q̂ | q₀₁=4.45, q₁₀=1.69 | unchanged |

Training config: Adam lr=1e-3, CosineAnnealingLR, 10,000 epochs, 2048 LHS collocation points/step, grad clip 1.0.

Loss weights (`PINN_ALPHA` in `settings.py`): HJB=1.0, stat=1.0, comp=0.1, stab=0.01, **lb=0.5** (lower-bound penalty — new; prevents π*→0 degenerate fixed point).

`PolicyNet` architecture change (`pinn/networks.py`): final linear layer bias initialised to +1.5 (`sigmoid(1.5)≈0.82`) to anchor initial policy near the Merton solution and prevent collapse to zero allocation.

**Current models** were retrained with the above parameters on 2026-06-21. Run `python train_pinns.py` again (~30 min CPU) only if the parameters are changed.

---

## Universe Building

Rebuilt once per trading day at market open. `stock_info.csv` is the canonical source — ~1,900 tickers pre-validated against the Alpaca API (tradable, fractionable, active) with price, volume, and 1-month momentum columns.

**Source priority:**

1. **`stock_info.csv` (primary)** — read directly using the pre-validated columns (`AlpacaStatus`, `AlpacaTradable`, `AlpacaFractionable`). No live Alpaca API call. Returns all eligible rows (up to `CSV_UNIVERSE_MAX_SYMBOLS`; 0 = unlimited).
2. **OpenBB finviz screener (fallback)** — `most_active`, `top_gainers`, `oversold`, `unusual_volume`. Only reached if the CSV is missing or empty.
3. **Alpaca assets API (fallback)** — all fractionable US equities; filter price > $5, avg volume > 1M, sort by 30-day momentum. Only reached if the screener returns fewer than 20 symbols.
4. **Hardcoded 10-ticker watchlist** — last resort if all sources fail.

**Updating the CSV:** Re-run `audit_alpaca_listings.py` to regenerate `stock_info.csv` against the current Alpaca asset list. The CSV format must preserve the `AlpacaSymbol`, `AlpacaStatus`, `AlpacaTradable`, and `AlpacaFractionable` columns for the fast path to apply.

---

## Data Sources

| Data | Source | Notes |
|---|---|---|
| Daily OHLCV | `obb.equity.price.historical(provider="yfinance")` | 260-day lookback for signals + momentum score, 504-day for HMM |
| VIX | `fetch_ohlcv("^VIX")` — tries `obb.equity` then `obb.index` endpoint | Returns `(VIX/100)²`; falls back to `0.041` (Heston θ) on failure |
| Risk-free rate | `yfinance.Ticker("^IRX").history()` | 3-month T-bill yield; cached daily; used in Merton fallback |
| 1-min intraday | `yfinance.Ticker.history(period="1d", interval="1m")` | Gated by 5-min freshness check |
| Universe screener | `obb.equity.screener(provider="finviz")` | Scrapes finviz.com, no key needed |
| Asset list | `TradingClient.get_all_assets()` | Alpaca paper account |
| News headlines | `https://finance.yahoo.com/rss/headline?s={ticker}` | Public RSS, no rate limit |
| Sentiment | `ProsusAI/finbert` via HuggingFace transformers | Local inference, ~500 MB cached |
| Order execution | `TradingClient(paper=True)` | Alpaca paper account PA3OWGV0WPLO |

---

## Key Settings (`settings.py`)

| Constant | Value | Purpose |
|---|---|---|
| `PAPER_TRADING` | `True` | Never change without explicit confirmation |
| `CYCLE_INTERVAL_SECONDS` | 900 | 15-minute cycle |
| `UNIVERSE_SIZE` | 150 | Max symbols in daily universe |
| `SIGNAL_LOOKBACK_DAYS` | 260 | Daily OHLCV lookback — extended to support 12-1 month momentum score |
| `HMM_LOOKBACK_DAYS` | 504 | ~2 years for HMM training |
| `CRRA_GAMMA` | 3.0 | Risk aversion coefficient |
| `TRADING_HORIZON_YEARS` | 0.5 | T in HJB equation |
| `INITIAL_EQUITY` | 100_000.0 | `w = equity / INITIAL_EQUITY` — update if paper account is reset |
| `MARKET_CRASH_SPY_THRESHOLD` | −0.03 | Hard SPY 1-day return backstop (fires before HMM) |
| `PINN_DAILY_CACHE` | `True` | Cache PINN allocation by `(date, regime, vix_bucket)` — 2-pt VIX buckets so intraday spikes re-query |
| `MIN_POSITION_WEIGHT` | 0.015 | Suppress positions < 1.5% of equity (hysteresis) |
| `LIMIT_ORDER_ENABLED` | `True` | Submit limit orders instead of market orders |
| `LIMIT_ORDER_OFFSET_BPS` | 2 | Peg limit 2 bps below current price for passive fill |
| `LIMIT_ORDER_TIMEOUT_MIN` | 4 | Cancel and market-fill if unfilled after 4 min |
| `CYCLE_INTERVAL_BULL_SECONDS` | 7200 | 2-hour interval in confirmed bull + VIX < 25 |
| `VIX_HIGH_THRESHOLD` | 25.0 | VIX above this forces 15-min interval regardless of regime |
| `INTRADAY_MAX_STALE_MIN` | 5 | Max age of 1-min bar before skipping lead-lag |
| `SENTIMENT_WINDOW_MIN` | 15 | Rolling window for peak sentiment cache |
| `MAX_POSITION_SIZE` | 0.15 | Max 15% of equity per position in bear/neutral |
| `BULL_BYPASS_PINN` | `True` | In confirmed bull, skip PINN ceiling and deploy 100% of equity |
| `BULL_LEVERAGE_FACTOR` | 1.5 | Notional leverage on full equity in confirmed bull (150%) |
| `BULL_MAX_POSITION_SIZE` | 0.25 | Max 25% of leveraged equity per position in confirmed bull |
| `SECTOR_MAX_POSITIONS` | 5 | Max positions per GICS sector |
| `STOP_CHECK_INTERVAL_SECONDS` | 300 | Intraday trailing-stop check cadence between full cycles |
| `CRRA_GAMMA_BULL` | 1.5 | Effective γ in confirmed bull — scales π* by 3.0/1.5=2× without retraining |
| `VOL_TARGET` | 0.15 | Target annualized portfolio volatility for vol-targeting scale |
| `VOL_LOOKBACK_DAYS` | 20 | Rolling window (trading days) for realized vol estimate — one snapshot stored per trading day |
| `VOL_SCALE_MIN` | 0.5 | Floor on vol-targeting scale (never reduce below 50% of π*) |
| `VOL_SCALE_MAX` | 1.5 | Ceiling on vol-targeting scale (never increase above 150% of π*) |
| `RISK_FREE_RATE_DYNAMIC` | `True` | Fetch live 3-month T-bill rate (`^IRX`) daily; falls back to 5% on failure |
| `UNIVERSE_BATCH_SIZE` | 50 | Symbols evaluated per cycle; full universe rotates over N cycles |
| `HMM_REFIT_DAYS` | 7 | Walk-forward fold window — HMM refitted at most once per 7 trading days |
| `WALK_FORWARD_GATE` | `True` | Enable Monte Carlo p-value gate on buy candidates |
| `PERMUTATION_N` | 100 | Permutations per symbol for p-value test (≈2ms each) |
| `PERMUTATION_PVALUE_GATE` | 0.05 | Exclude symbols whose EMA signal p-value ≥ this threshold |

---

## Quantitative Research Loop

`research.py` implements a rigorous 11-iteration scientific protocol with strict train/val/test separation. A change is accepted only when all three periods improve (or hold) on Sharpe without violating the MaxDD constraint (< 15%).

**Periods:** Train 2003–2016 / Val 2017–2020 / Test 2021–2025

| Iter | Change | Decision | Test Sharpe | Test CAGR |
|---|---|---|---|---|
| 0 | Baseline | — | 0.01 | +5.0% |
| 1 | Trailing stop −10% replaces +6% take-profit | **ACCEPT** | 0.37 | +7.9% |
| 2 | Stop-loss −3% → −6% | REJECT | — | — |
| 3 | Cash reserve 20% → 10% | **ACCEPT** | 0.37 | +7.9% |
| 4 | Max positions 8 → 14 | **ACCEPT** | 0.34 | +8.0% |
| 5 | UNIVERSE_40 + momentum ranking | REJECT (MaxDD −17%) | — | — |
| 6 | Bear recovery entry | **ACCEPT** | 0.33 | +7.8% |
| 7 | UNIVERSE_40 + momentum + sector cap | REJECT (MaxDD −16%) | — | — |
| 8 | Dynamic γ: bull γ=1.5 doubles PINN π* | **ACCEPT** | 0.48 | +9.5% |
| 9 | Volatility targeting 15% ann. vol | **ACCEPT** | 0.48 | +9.5% |
| 10 | Inverse-vol weighting replaces Merton sizing | **ACCEPT** | **0.53** | **+10.1%** |

**Best config (Iter 10):** Test Sharpe 0.53, MaxDD −7.9%, Calmar 1.27. All settings reflected in `settings.py`.

**`research.py` restructured (2026-06-23):** Iters 1–3 are baked into `StrategyConfig` defaults. The loop now runs Iters 0–7 starting from the Iter 3 config (trailing stop + 10% cash reserve) as the baseline. Re-running `research.py` begins from this validated foundation and only tests ideas beyond it.

---

## Event-Driven Backtest Engine

`backtest_engine.py` is a high-fidelity standalone backtester designed to complement the simplified `backtest.py` daily-bar engine.

Key design features:
- **Dynamic event clock**: 15-min steps in bear/high-VIX, 2-h steps in calm bull — matches live agent exactly
- **5-min exit sub-loop**: trailing stop and stop-loss checked every 5 min between full allocation cycles; prevents flash-crash slippage during 2-h bull intervals
- **Brownian-bridge 1-min bar synthesis**: `IntradaySimulator` generates reproducible synthetic intraday paths anchored to daily OHLC via a Brownian bridge seeded on `hash(symbol + date)` — used by the fill engine
- **High-fidelity fill simulation**: 2-bps passive limit fill (inspects synthesised 1-min Lows for T+1 to T+4); if limit untouched → market fill at T+5 + 1.5-bps slippage penalty
- **Rolling 504-bar HMM refit**: refitted at each major cycle using only data strictly prior to simulation time T — no lookahead
- **Full risk stack**: dynamic γ, vol targeting, inverse-vol weighting, sector cap, hysteresis buffer — all identical to live agent

```bash
python backtest_engine.py 2021-01-01 2025-06-30
```

---

## Statistical Validation

`validation.py` tests whether the strategy's edge is statistically real or memorising noise.

```bash
python validation.py                    # test default 20-ticker universe
python validation.py AAPL MSFT NVDA    # test specific symbols
python validation.py --n 1000          # 1000 permutations (rigorous)
```

**Monte Carlo Permutation Test:**
1. Run EMA crossover signal on real price history → compute Profit Factor PF_real
2. Shuffle the log-return series N times (destroys autocorrelation, preserves vol distribution) → compute PF_shuffled for each
3. p-value = fraction of shuffled paths where PF_shuffled ≥ PF_real
4. Gate: p-value < 0.01 → PASS (statistically significant edge)

**Walk-Forward Folds:** 5 anchored folds; train on [0…fold_end], test OOS on [fold_end…fold_end+step]. Per-fold: Sharpe, Profit Factor, in-sample permutation p-value.

**Profit Factor formula:** `PF = Σmax(0, R) / Σmax(0, −R)` — ratio of gross gains to gross losses on per-bar strategy returns.

**In-cycle gate** (`main.py`): `quick_pvalue(close, n=100)` runs 100 permutations per symbol before any buy is accepted. Cached per-symbol per-day — runs once, reused all day.

---

## Terminal Dashboard

`python main.py` launches a live Rich terminal UI that replaces all debug log lines. All `logger.debug` output is redirected to `logs/trade_journal.log`; the terminal shows only the structured display.

**Panels (dynamically sized to terminal width):**

| Panel | Content |
|---|---|
| **PINN Trading Agent** | Equity, Cash, P&L Today, Regime, VIX, RF Rate, π*, Cycle # |
| **Universe Scan** | Live progress bar + rolling ticker of last 12 symbols with ✓/✗/· signals |
| **Buy Candidates** | Table: Symbol, Signal, π*, Momentum, σ, p-value, Sector |
| **Open Positions** | Table: Symbol, Shares, Entry, Current, P&L%, Peak, Trailing-stop Δ |
| **Cycle Timer** | Countdown bar: `Next cycle in Xm Ys  │  15-min (bear)` |

---

## Known Limitations

- **yfinance 1-min latency**: Free yfinance intraday data is cached and can lag several minutes. The freshness gate (`INTRADAY_MAX_STALE_MIN = 5`) suppresses lead-lag when stale, but GK and micro-halt still run on whatever bars are available.
- **PINN calibration requires manual retraining**: `train_pinns.py` parameters have been updated to reflect modern market dynamics (GBM μ=0.155, Heston μ=0.160, Regime risk-on μ=0.210) but existing `models/*.pt` files still use the original paper values until `python train_pinns.py` is re-run (~30 min CPU). The dynamic γ and vol-targeting runtime adjustments compensate partially, but the full effect only lands after retraining.
- **HMM walk-forward lock**: HMM now refits at most every 7 trading days. If market regime shifts intra-week (e.g. flash crash on day 3 of a fold), the locked model may lag by up to 4 trading days before the next refit. The hard crash backstop (`SPY -3% day`) fires instantly regardless of fold boundary.
- **PINN degenerate-solution risk**: The lower-bound `L_lb` penalty and bias initialisation fix prevent the π*→0 collapse in future training runs. The `max(PINN, Merton)` floor in `compute_signals()` provides runtime protection against any residual calibration gaps in already-trained models.
