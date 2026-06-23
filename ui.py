"""
ui.py — Rich terminal dashboard for the PINN Trading Agent.

Replaces scattered logger.debug lines with a clean, dynamically-sizing
terminal display that shows: portfolio state, live universe scan progress,
buy candidates, open positions, and a countdown to the next cycle.
"""
from __future__ import annotations

from typing import Optional

from rich import box
from rich.columns import Columns
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# ─────────────────────────────────────────────────────────────────────────────
# Colour scheme
# ─────────────────────────────────────────────────────────────────────────────
REGIME_COLOUR = {"bull": "bright_green", "bear": "yellow", "crash": "bright_red"}
SIGNAL_COLOUR = {"buy": "bright_green", "sell": "red", "hold": "dim", "sell_all": "bright_red"}
C_HEAD   = "bold cyan"
C_DIM    = "dim"
C_EQUITY = "bright_white"
C_GREEN  = "bright_green"
C_RED    = "red"
C_AMBER  = "yellow"


def _pnl_colour(val: float) -> str:
    return C_GREEN if val >= 0 else C_RED


def _pnl_str(val: float) -> str:
    return f"[{_pnl_colour(val)}]{val:+,.2f}[/]"


# ─────────────────────────────────────────────────────────────────────────────
# Main UI class
# ─────────────────────────────────────────────────────────────────────────────

class TradingUI:
    """
    Single instance of the terminal dashboard.  All state is updated via
    the public `update_*` / `start_*` / `tick_*` methods, which are safe
    to call from the main trading loop (single-threaded).
    """

    def __init__(self) -> None:
        self.console = Console()
        self._live:   Optional[Live] = None

        # ── Portfolio state ──────────────────────────────────────────────
        self.equity:     float = 0.0
        self.cash:       float = 0.0
        self.pnl_today:  float = 0.0
        self.regime:     str   = "bear"
        self.vix:        float = 20.0
        self.rf_rate:    float = 0.05
        self.pinn_alloc: float = 0.0
        self.cycle_num:  int   = 0
        self.universe_n: int   = 0

        # ── Scan state ───────────────────────────────────────────────────
        self._scan_total:   int  = 0
        self._scan_done:    int  = 0
        self._scan_current: str  = ""
        self._scan_signals: list[tuple[str, str]] = []   # (symbol, signal)
        self._scan_active:  bool = False

        # ── Results ──────────────────────────────────────────────────────
        self.buy_candidates: list[dict] = []
        self.positions:      dict       = {}
        self.exits_this_cycle: list[tuple[str, str]] = []  # (sym, reason)

        # ── Countdown state ──────────────────────────────────────────────
        self._cycle_total:   int = 900
        self._cycle_elapsed: int = 0
        self._cycle_label:   str = "15-min (bear)"

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def start(self) -> None:
        self._live = Live(
            self._render(),
            console   = self.console,
            refresh_per_second = 4,
            screen    = False,
            transient = False,
        )
        self._live.__enter__()

    def stop(self) -> None:
        if self._live:
            self._live.__exit__(None, None, None)
            self._live = None

    def _refresh(self) -> None:
        if self._live:
            self._live.update(self._render())

    # ── State update API ──────────────────────────────────────────────────

    def update_header(self, equity: float, cash: float, pnl_today: float,
                      regime: str, vix: float, rf_rate: float,
                      pinn_alloc: float, cycle_num: int,
                      universe_n: int) -> None:
        self.equity     = equity
        self.cash       = cash
        self.pnl_today  = pnl_today
        self.regime     = regime
        self.vix        = vix
        self.rf_rate    = rf_rate
        self.pinn_alloc = pinn_alloc
        self.cycle_num  = cycle_num
        self.universe_n = universe_n
        self._refresh()

    def update_positions(self, positions: dict) -> None:
        self.positions = positions
        self._refresh()

    def start_scan(self, total: int) -> None:
        self._scan_total   = total
        self._scan_done    = 0
        self._scan_current = ""
        self._scan_signals = []
        self._scan_active  = True
        self.buy_candidates      = []
        self.exits_this_cycle    = []
        self._refresh()

    def tick_scan(self, symbol: str, signal: str) -> None:
        self._scan_done    = min(self._scan_done + 1, self._scan_total)
        self._scan_current = symbol
        # Keep the last 12 scanned symbols visible
        self._scan_signals.append((symbol, signal))
        if len(self._scan_signals) > 12:
            self._scan_signals = self._scan_signals[-12:]
        self._refresh()

    def finish_scan(self, buy_candidates: list[dict]) -> None:
        self._scan_active   = False
        self.buy_candidates = buy_candidates
        self._refresh()

    def add_exit(self, symbol: str, reason: str) -> None:
        self.exits_this_cycle.append((symbol, reason))
        self._refresh()

    def start_countdown(self, total_seconds: int, label: str) -> None:
        self._cycle_total   = max(total_seconds, 1)
        self._cycle_elapsed = 0
        self._cycle_label   = label
        self._scan_active   = False
        self._refresh()

    def tick_countdown(self, elapsed: int) -> None:
        self._cycle_elapsed = elapsed
        self._refresh()

    # ── Render ────────────────────────────────────────────────────────────

    def _render(self) -> Group:
        return Group(
            self._render_header(),
            self._render_scan_or_results(),
            self._render_positions(),
            self._render_countdown(),
        )

    # ── Header panel ──────────────────────────────────────────────────────

    def _render_header(self) -> Panel:
        rc = REGIME_COLOUR.get(self.regime, "white")

        row1 = Columns([
            Text.assemble(("Equity  ", C_DIM), (f"${self.equity:>12,.2f}", C_EQUITY)),
            Text.assemble(("Cash  ",  C_DIM),  (f"${self.cash:>12,.2f}", C_DIM)),
            Text.assemble(("P&L Today  ", C_DIM),
                          (f"{self.pnl_today:+,.2f}",
                           _pnl_colour(self.pnl_today))),
        ], equal=True, expand=True)

        row2 = Columns([
            Text.assemble(("Regime  ", C_DIM),
                          (self.regime.upper(), f"bold {rc}")),
            Text.assemble(("VIX  ", C_DIM),
                          (f"{self.vix:.1f}",
                           C_RED if self.vix >= 25 else C_GREEN)),
            Text.assemble(("RF Rate  ", C_DIM),
                          (f"{self.rf_rate:.2%}", C_DIM)),
            Text.assemble(("π*  ", C_DIM),
                          (f"{self.pinn_alloc:.3f}",
                           C_GREEN if self.pinn_alloc > 0.5 else C_AMBER)),
            Text.assemble(("Cycle  ", C_DIM),
                          (f"#{self.cycle_num}", C_DIM)),
            Text.assemble(("Universe  ", C_DIM),
                          (f"{self.universe_n} symbols", C_DIM)),
        ], equal=True, expand=True)

        return Panel(Group(row1, row2),
                     title="[bold cyan]PINN Trading Agent[/]",
                     border_style="cyan", box=box.ROUNDED)

    # ── Scan progress + results ───────────────────────────────────────────

    def _render_scan_or_results(self) -> Panel:
        if self._scan_active or (self._scan_done > 0 and self._scan_done < self._scan_total):
            return self._render_scan_progress()
        return self._render_candidates()

    def _render_scan_progress(self) -> Panel:
        pct   = self._scan_done / max(self._scan_total, 1)
        width = max(self.console.width - 10, 20)
        filled = int(pct * width)
        bar   = (f"[bright_green]{'█' * filled}[/]"
                 f"[dim]{'░' * (width - filled)}[/]")

        status = Text.assemble(
            (f"  {self._scan_done}/{self._scan_total}", "bold"),
            ("  scanning  ", C_DIM),
            (self._scan_current, "bold yellow"),
            (f"  ({pct:.0%})", C_DIM),
        )

        # Recent signal ticker
        ticker_parts: list[str] = []
        for sym, sig in self._scan_signals[-15:]:
            col = SIGNAL_COLOUR.get(sig, "dim")
            icon = "✓" if sig == "buy" else ("✗" if sig == "sell" else "·")
            ticker_parts.append(f"[{col}]{icon}[dim]{sym}[/][/]")
        ticker = Text.from_markup("  " + "  ".join(ticker_parts))

        return Panel(Group(Text.from_markup(bar), status, ticker),
                     title="[bold]Universe Scan[/]",
                     border_style="bright_blue", box=box.ROUNDED)

    def _render_candidates(self) -> Panel:
        if not self.buy_candidates and not self.exits_this_cycle:
            body = Text("No buy signals this cycle.", style=C_DIM)
        else:
            body = Group(
                self._build_candidates_table(),
                self._build_exits_text(),
            )
        return Panel(body,
                     title=f"[bold green]Buy Candidates ({len(self.buy_candidates)})[/]",
                     border_style="green", box=box.ROUNDED)

    def _build_candidates_table(self) -> Table:
        t = Table(box=box.SIMPLE, show_header=True, header_style=C_HEAD,
                  expand=True, padding=(0, 1))
        t.add_column("Symbol",   style="bold")
        t.add_column("Signal",   justify="center")
        t.add_column("π*",       justify="right")
        t.add_column("Momentum", justify="right")
        t.add_column("σ",        justify="right")
        t.add_column("p-val",    justify="right")
        t.add_column("Sector",   style=C_DIM)

        for c in self.buy_candidates:
            sig    = c.get("signal", "buy")
            scol   = SIGNAL_COLOUR.get(sig, "white")
            pval   = c.get("pvalue", 1.0)
            pv_col = C_GREEN if pval < 0.05 else C_AMBER if pval < 0.10 else C_RED
            mom    = c.get("momentum", 0.0)
            t.add_row(
                c["symbol"],
                f"[{scol}]{sig.upper()}[/]",
                f"{c.get('pinn', 0.0):.3f}",
                f"[{C_GREEN if mom > 0 else C_RED}]{mom:+.3f}[/]",
                f"{c.get('sigma', 0.0):.3f}",
                f"[{pv_col}]{pval:.3f}[/]",
                c.get("sector", "—"),
            )
        return t

    def _build_exits_text(self) -> Text:
        if not self.exits_this_cycle:
            return Text("")
        t = Text()
        t.append("Exits this cycle: ", style=C_DIM)
        for sym, reason in self.exits_this_cycle:
            t.append(f"{sym} ", style="bold red")
            t.append(f"[{reason}]  ", style=C_DIM)
        return t

    # ── Positions ─────────────────────────────────────────────────────────

    def _render_positions(self) -> Panel:
        if not self.positions:
            body = Text("No open positions.", style=C_DIM)
        else:
            body = self._build_positions_table()
        n = len(self.positions)
        return Panel(body,
                     title=f"[bold]Open Positions ({n}/14)[/]",
                     border_style="bright_blue", box=box.ROUNDED)

    def _build_positions_table(self) -> Table:
        t = Table(box=box.SIMPLE, show_header=True, header_style=C_HEAD,
                  expand=True, padding=(0, 1))
        t.add_column("Symbol",  style="bold")
        t.add_column("Shares",  justify="right")
        t.add_column("Entry",   justify="right")
        t.add_column("Current", justify="right")
        t.add_column("P&L %",  justify="right")
        t.add_column("Peak",    justify="right", style=C_DIM)
        t.add_column("Trail Δ", justify="right")

        for sym, pos in self.positions.items():
            try:
                entry   = float(pos.avg_entry_price)
                current = float(pos.current_price)
                pnl_pct = (current - entry) / entry
                peak    = float(getattr(pos, "peak_price", current))
                trail   = (current - peak) / peak if peak > 0 else 0.0
                pnl_col = _pnl_colour(pnl_pct)
                tr_col  = C_RED if trail <= -0.07 else (C_AMBER if trail <= -0.04 else C_DIM)
                t.add_row(
                    sym,
                    f"{float(pos.qty):.3f}",
                    f"${entry:.2f}",
                    f"${current:.2f}",
                    f"[{pnl_col}]{pnl_pct:+.2%}[/]",
                    f"${peak:.2f}",
                    f"[{tr_col}]{trail:+.2%}[/]",
                )
            except Exception:
                t.add_row(sym, "—", "—", "—", "—", "—", "—")
        return t

    # ── Countdown ─────────────────────────────────────────────────────────

    def _render_countdown(self) -> Panel:
        remaining = max(self._cycle_total - self._cycle_elapsed, 0)
        pct       = self._cycle_elapsed / max(self._cycle_total, 1)
        width     = max(self.console.width - 10, 20)
        filled    = int(pct * width)

        mins, secs = divmod(remaining, 60)
        time_str   = f"{mins}m {secs:02d}s" if mins else f"{secs}s"

        bar = (f"[cyan]{'█' * filled}[/]"
               f"[dim]{'░' * (width - filled)}[/]")

        status = Text.assemble(
            ("  Next cycle in  ", C_DIM),
            (time_str, "bold cyan"),
            ("  │  ", C_DIM),
            (self._cycle_label, C_DIM),
        )

        return Panel(Group(Text.from_markup(bar), status),
                     title="[bold]Cycle Timer[/]",
                     border_style="dim", box=box.ROUNDED)
