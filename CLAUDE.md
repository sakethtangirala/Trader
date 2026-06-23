# CLAUDE.md — Autonomous Trading Agent

## Project Summary

This is a fully autonomous paper-trading agent built by Saketh Tangirala (Georgia Tech CS) in collaboration with Claude Sonnet 4.6 via Claude Code. It implements Saketh's co-authored Finance Research Letters paper ("Portfolio Optimization in Continuous Time: A Physics-Informed Neural Network Approach") as a live trading system: a Physics-Informed Neural Network (PINN) solves the Hamilton–Jacobi–Bellman equation under GBM, Heston, and regime-switching dynamics to produce an optimal portfolio allocation π*(t,w,y) each cycle. The agent runs on Alpaca paper trading (account PA3OWGV0WPLO, $100k starting equity) and operates at zero runtime API cost — all inference is local, all data is free (yfinance, Yahoo RSS, FinBERT).

---

## Architecture Map

| File | Responsibility |
|---|---|
| `main.py` | Orchestration loop — 15-min cycles during market hours, daily universe rebuild, startup of FinBERT and PINNs |
| `engine.py` | Data fetching (OpenBB/yfinance), VIX fetch, HMM regime detection on SPY, PINN allocation hierarchy, RSI/EMA signal generation, per-stock parameter estimation |
| `risk_manager.py` | Circuit breakers, daily reset, stop-loss/take-profit exit checks, PINN+Merton position sizing |
| `brokerage.py` | Alpaca paper trading wrapper — buy, sell, close all, cancel orders |
| `universe.py` | Daily trading universe — OpenBB finviz screener → Alpaca fractionable assets + yfinance filter → 10-ticker fallback |
| `sentiment.py` | FinBERT (ProsusAI/finbert, local) + Yahoo Finance RSS — sentiment guardrails |
| `settings.py` | Single source of truth for all config: risk thresholds, PINN hyperparams, model paths, scheduling |
| `train_pinns.py` | Train all three PINNs (CRRA ansatz). Run once before starting the agent. |
| `pinn/networks.py` | Shared NN building blocks: MLP, PolicyNet, IneqMultiplierNet (tanh, Xavier init) |
| `pinn/sampler.py` | Latin Hypercube Sampling — 2048 collocation points per step (per paper Section 4) |
| `pinn/gbm.py` | GBM PINN — 2D state (t, w), CRRA ansatz, no BC loss |
| `pinn/heston.py` | Heston PINN — 3D state (t, w, y), cross-derivative V_wy, ρ=−0.718, CRRA ansatz |
| `pinn/regime.py` | Regime PINN — coupled V^0/V^1 via generator Q̂, CRRA ansatz |
| `ml_references/` | Saketh's 4 prior HJB-PINN implementations (v1–v4) — read-only research reference |

---

## Non-Negotiable Invariants

**Never violate these without Saketh explicitly confirming the change:**

### 1. Paper trading lock
`PAPER_TRADING = True` must remain `True` in `settings.py`.
Only change to `False` if the user says exactly "go live" or "switch to live trading."
Do not accept "test live" or "try live" as confirmation.

### 2. No secrets in files
Never write to `.env`. Never include `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`, or any credential in any file that could be committed.

### 3. Risk thresholds are hard floors
These values in `settings.py` and enforced in `risk_manager.py` must not be relaxed without explicit user confirmation:
- `MAX_POSITION_SIZE = 0.10` — max 10% equity per position
- `CASH_RESERVE_PCT = 0.20` — always keep 20% cash
- `DAILY_LOSS_CIRCUIT_BREAKER = 0.05` — halt if down 5% in a day
- `MAX_PORTFOLIO_DRAWDOWN = 0.10` — halt if down 10% from peak
- `STOP_LOSS_PCT = 0.03` — exit at -3% per position
- `TAKE_PROFIT_PCT = 0.06` — exit at +6% per position

They may be tightened freely, but never loosened without asking.

### 4. CRRA ansatz is canonical — do not revert
The PINN architecture uses `V = base·(1+(T-t)·ψ)` with no BC loss term. This is Saketh's validated approach from `ml_references/version2/` and achieves 87–384× lower loss than the separate-BC-loss formulation. Do not introduce a BC loss term or revert to learning V directly.

### 5. Alpaca account is paper only
Never construct a `TradingClient` with `paper=False` unless the user has confirmed "go live."

---

## Coding Conventions

- **Match the file's existing style** — indentation, spacing, import order, docstring style.
- **`from __future__ import annotations`** in any file using `X | Y` union types or `list[X]`/`dict[X]` generics.
- **No new top-level dependencies** without flagging it first. Add to `requirements.txt` with a version pin.
- **Type annotations on all function signatures.**
- **No bare `except:`** — always `except Exception` or a specific type.
- **Loguru, not print** — `from loguru import logger` everywhere.
- **Ruff is present** — run `ruff check .` after edits; fix all issues before declaring done.
- **Settings are the single source of truth** — no magic numbers inline.

---

## Before Declaring a Task Finished

1. Run `ruff check .` and fix any errors.
2. Scan diff for secrets — no credentials in any file.
3. Update `README.md` if file structure, strategy logic, or setup steps changed.
4. Update this `CLAUDE.md` if invariants, conventions, or fragile areas changed.
5. Don't claim the agent "works" without noting it hasn't been live-tested during market hours yet.

---

## Known Fragile Areas

### FinBERT first-run download
`FreeSentimentEngine.__init__()` downloads ~500 MB from HuggingFace on first run. Blocks startup 1–3 min. Non-fatal (wrapped in try/except in `main.py`), but the process will appear to hang. Expected.

### OpenBB → finviz fallback chain
`universe.py` scrapes finviz.com via OpenBB. Can fail silently (rate limits, HTML changes). Alpaca assets fallback takes 2–5 min for 3000 symbols. 10-ticker last-resort watchlist always available. Don't remove any layer.

### OpenBB VIX fetch
`fetch_vix()` tries equity then index endpoint. Falls back to `0.041` (Heston θ) on failure — silently degrades Heston PINN quality without erroring.

### HMM state label ordering
`detect_regime()` maps states to bull/bear/crash by mean return each cycle. Labels can flip between cycles if the training window contains an unusual period. Known statistical property of unsupervised HMMs, not a bug.

### `MARKET_CRASH_SPY_THRESHOLD` — wired in
Used in `main.py` as the SPY hard crash backstop. If SPY 1-day return ≤ −0.03, `close_all()` fires before HMM runs. No longer unused.

### Models directory
`models/gbm_pinn.pt`, `models/heston_pinn.pt`, `models/regime_pinn.pt` are the canonical trained models (CRRA ansatz, final losses: GBM 4.5e-7, Heston 6.5e-7, Regime 2.2e-5). If deleted, re-run `python train_pinns.py` (~30 min CPU).

---

## Spec Workflow

Follow this procedure for all non-trivial features and bugs. Commands in `.claude/commands/`.

### New Feature: `/spec-create` → `/spec-execute`

```
/spec-create <feature-name> "Description"
```
Produces `.claude/specs/<feature-name>/requirements.md`, `design.md`, `tasks.md`.

```
/spec-execute <N> <feature-name>
```
Implements Task N. Load steering docs + full spec first. Mark task ✅ in tasks.md when done.

| Scenario | Approach |
|---|---|
| New feature, clear scope | `/spec-create` then `/spec-execute` per task |
| Quick isolated change | Inline — no spec needed |
| Bug | Bug workflow below |
| Touching risk thresholds or PAPER_TRADING | Always spec first, flag to Saketh |

### Bug Fixes: `/bug-create` → `/bug-analyze` → `/bug-fix` → `/bug-verify`

### Steering: `/spec-steering-setup`
Refreshes `.claude/steering/` when project has changed significantly.

### Context Loading Order
Load once, don't reload per sub-task:
1. `.claude/steering/product.md` + `tech.md` + `structure.md`
2. `.claude/specs/<feature>/requirements.md` + `design.md` + `tasks.md`
3. Source files for the current task only

---

## Drift Log

| Item | Status |
|---|---|
| `hmm_spy.pkl` | Auto-created at first runtime cycle, not present until then |
| Ruff linter | `.ruff_cache/` present — run `ruff check .` after edits |

---

## Research Log (quantitative research loop)

### Findings as of 2026-06-20

Research framework: `research.py`. Train 2003–2016 / Val 2017–2020 / Test 2021–2025.

| Iteration | Change | Decision | Reason |
|---|---|---|---|
| 0 | Baseline | — | CAGR 4.8/6.2/5.0%, Sharpe -0.01/0.17/0.01 |
| 1 | Trailing stop -10% replaces fixed +6% take-profit | **ACCEPT** | Sharpe tripled all periods; profit factor 1.3 → 2.3+ |
| 2 | Widen stop-loss -3% → -6% | **REJECT** | Test Sharpe regressed -0.15 |
| 3 | Cash reserve 20% → 10% | ACCEPT (no effect) | Reserve wasn't binding; position weight caps constrain sizing |
| 4 | Max positions 8 → 14 | **ACCEPT** | Val Sharpe +0.11, all periods pass |

**Best config so far (Iter 4):**
- `trailing_stop = 0.10`, `take_profit = 0.0`, `stop_loss = 0.03`
- `cash_reserve = 0.10`, `max_positions = 14`
- Train/Val/Test Sharpe: 0.21 / 0.50 / 0.34
- Train/Val/Test MaxDD: -13.1% / -13.1% / -7.9%

**Remaining gap to SPY:** ~2–5% CAGR.

**Architecture change implemented (2026-06-20):** PINN now used as a pure risk multiplier on a momentum-first stock selection layer. Changes in `engine.py` + `main.py`:
- `compute_momentum()` added — 12-1 month momentum score per stock
- RSI ceiling removed from bull entry (was cutting NVDA/MSFT at RSI > 70)
- RSI sell removed from bull (trailing stop + EMA handle exits)
- Buy candidates ranked by momentum score instead of Merton ratio
- `SIGNAL_LOOKBACK_DAYS` extended 90 → 260 to support full momentum window
- Merton ratio retained for position SIZING (risk_manager.py unchanged)

**Iter 5 (UNIVERSE_40 + momentum): REJECTED** — MaxDD exceeded 15% in ALL periods (-15.4%/-17.1%/-17.2%). Momentum ranking without sector cap causes severe clustering (7-9 tech/healthcare stocks during trend runs). Confirmed sector diversification cap is required before momentum expansion can be accepted.

**Iter 6 (bear recovery mode): ACCEPTED** — All periods pass, negligible change (+1 trade on train, Sharpe unchanged). Bear recovery mode is live in research.py.

**Iter 7 (UNIVERSE_40 + momentum + sector cap 3/sector): REJECTED** — MaxDD still exceeded 15% in all periods (train −15.2%, val −18.0%, test −16.1%). Sector cap of 3 was insufficient; the combination of bear_recovery_mode + momentum + larger universe compounds drawdown risk beyond what PINN parameters are calibrated for. Universe expansion is off the table until a tighter drawdown control mechanism is found.

**Final best config (Iter 4 + Iter 6):**
- stop_loss=0.03, take_profit=0.0, trailing_stop=0.10, cash_reserve=0.10, max_positions=14, bear_recovery_mode=True
- Sharpe (train/val/test): 0.21 / 0.49 / 0.33 — MaxDD: −13.1% / −13.5% / −7.9%

**Live system delta vs best backtest config:** momentum ranking + RSI ceiling removal + sector cap are live-only features not validated in the fixed-universe backtest. They are sound in principle for the dynamic screener universe (which self-selects for quality/momentum) but have no backtest confirmation.

**Do NOT re-investigate stop-loss widening or universe expansion until MaxDD < 15% in all periods is achievable.**
