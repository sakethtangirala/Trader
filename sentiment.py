"""
Free media sentiment engine using FinBERT + Yahoo Finance RSS.

FinBERT (ProsusAI/finbert) is a BERT model fine-tuned on financial news.
Unlike generic sentiment tools (VADER), it understands domain nuance:
  "crushed earnings" → positive
  "missed guidance"  → negative
  "flat performance" → neutral

Yahoo Finance RSS feeds are public and rate-limit-free — no API key needed.
The model (~500 MB) downloads once on first run and is cached locally.

Sentiment score per ticker: float in [-1.0, +1.0]
  +1.0 = unanimous bullish coverage
  -1.0 = unanimous bearish coverage
   0.0 = neutral / no news found
"""
from __future__ import annotations

import queue
import re
import threading
import time
import feedparser
import numpy as np
from transformers import pipeline
from loguru import logger

from settings import SENTIMENT_WINDOW_MIN


# Yahoo Finance RSS endpoint — public, no key, no rate limit
_RSS_URL = "https://finance.yahoo.com/rss/headline?s={ticker}"

# FinBERT label → numeric score
_LABEL_MAP = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}

# Regex pre-filter: only process headlines containing high-impact volatility keywords
# OR a direct ticker alias. Drops ~80% of generic headlines before hitting FinBERT.
_HIGH_IMPACT = re.compile(
    r"\b(halt|investigation|acquisition|restructure|bankruptcy|fda|ceo|ch11|fraud)\b",
    re.IGNORECASE,
)

# Exponential information decay constant λ — score × e^(−λ·Δt_minutes)
# λ=0.05 → half-life ≈ 14 minutes; λ=0.1 → half-life ≈ 7 minutes
_DECAY_LAMBDA: float = 0.05


class FreeSentimentEngine:
    """
    Singleton-style sentiment engine. Initialise once at startup
    (FinBERT load takes ~5s); reuse across all tickers and cycles.
    """

    def __init__(self):
        logger.info("Loading FinBERT (ProsusAI/finbert) — first run downloads ~500 MB...")
        self._nlp = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            truncation=True,
            max_length=512,
        )
        logger.info("FinBERT loaded and ready")

        # Rolling peak-sentiment cache: ticker → (peak_score, timestamp_unix)
        # Stores the highest-magnitude signal seen within the current 15-min window.
        # Prevents significant news at minute 1 from being 50% decayed by minute 14.
        self._peak_cache: dict[str, tuple[float, float]] = {}

    # ------------------------------------------------------------------

    @staticmethod
    def _age_minutes(entry) -> float:
        """Return headline age in minutes relative to now (UTC). 0.0 on parse failure."""
        try:
            pub = entry.get("published_parsed")
            if pub is None:
                return 0.0
            age_sec = time.time() - time.mktime(pub)
            return max(age_sec / 60.0, 0.0)
        except Exception:
            return 0.0

    @staticmethod
    def _is_relevant(headline: str, ticker: str) -> bool:
        """
        Keyword pre-filter: pass headline only if it contains a high-impact
        structural volatility keyword OR the ticker symbol itself.
        Drops ~80% of generic market-noise headlines before FinBERT inference.
        """
        return bool(_HIGH_IMPACT.search(headline)) or ticker.lower() in headline.lower()

    def _fetch_headlines(self, ticker: str,
                          max_articles: int) -> list[tuple[str, float]]:
        """Return list of (headline, age_minutes) tuples, pre-filtered for relevance."""
        url  = _RSS_URL.format(ticker=ticker)
        feed = feedparser.parse(url)
        result = []
        for entry in feed.entries[:max_articles * 3]:   # over-fetch to survive pruning
            if not hasattr(entry, "title"):
                continue
            headline = entry.title
            if not self._is_relevant(headline, ticker):
                continue
            result.append((headline, self._age_minutes(entry)))
            if len(result) >= max_articles:
                break
        return result

    def _score_headline(self, headline: str, age_minutes: float = 0.0) -> float:
        """Run FinBERT and apply exponential information decay e^(−λ·Δt)."""
        try:
            result = self._nlp(headline)[0]
            label  = result["label"].lower()
            conf   = float(result["score"])
            raw    = _LABEL_MAP.get(label, 0.0) * conf
            decay  = float(np.exp(-_DECAY_LAMBDA * age_minutes))
            return raw * decay
        except Exception:
            return 0.0

    # ------------------------------------------------------------------

    def get_ticker_sentiment(self, ticker: str, max_articles: int = 5) -> float:
        """
        Return decay-weighted average FinBERT sentiment score for `ticker`.

        Pipeline:
          1. Fetch RSS headlines (over-fetch to survive keyword pruning)
          2. Keyword pre-filter — drops generic noise before FinBERT
          3. FinBERT inference on each remaining headline
          4. Multiply raw score by e^(−λ·age_minutes) — older news counts less
          5. Return mean of decayed scores (0.0 if nothing survives the filter)
        """
        try:
            items = self._fetch_headlines(ticker, max_articles)
        except Exception as e:
            logger.warning(f"Sentiment RSS fetch failed ({ticker}): {e}")
            return 0.0

        if not items:
            logger.debug(f"No relevant headlines for {ticker} after pre-filter — neutral")
            snapshot = 0.0
        else:
            scores   = [self._score_headline(h, age) for h, age in items]
            snapshot = float(np.mean(scores))
            logger.debug(
                f"Sentiment {ticker}: snapshot={snapshot:+.3f} "
                f"({len(items)} headlines, λ-decay={_DECAY_LAMBDA}, "
                f"sample: '{items[0][0][:55]}…' age={items[0][1]:.1f}min)"
            )

        # --- Rolling peak-magnitude window ---
        # Within a 15-min cycle, big news at minute 1 must not be discarded by
        # the time the agent evaluates at minute 14.  Keep the highest-magnitude
        # score seen this window; if the cached value is older than the window,
        # reset it to the current snapshot.
        now    = time.time()
        window = SENTIMENT_WINDOW_MIN * 60.0
        cached = self._peak_cache.get(ticker)

        if cached is not None:
            peak_score, peak_time = cached
            if now - peak_time < window:
                # Still within window: keep whichever score has higher |magnitude|
                if abs(snapshot) > abs(peak_score):
                    self._peak_cache[ticker] = (snapshot, now)
                    final = snapshot
                else:
                    final = peak_score  # cached peak still dominates
            else:
                # Window expired: start fresh with current snapshot
                self._peak_cache[ticker] = (snapshot, now)
                final = snapshot
        else:
            self._peak_cache[ticker] = (snapshot, now)
            final = snapshot

        if final != snapshot:
            logger.debug(
                f"Sentiment {ticker}: using cached peak {final:+.3f} "
                f"(snapshot={snapshot:+.3f} — peak dominates within {SENTIMENT_WINDOW_MIN}-min window)"
            )
        return final

    # ------------------------------------------------------------------

    def apply_guardrails(self, ticker: str, signal: str,
                         sentiment_score: float) -> tuple[str, str]:
        """
        Apply BlackRock-style media guardrails to a technical signal.

        Returns (final_signal, reason_string).

        Rules:
          BUY  + sentiment < -0.20  → hold   (negative coverage blocks entry)
          SELL + sentiment < -0.60  → sell    (media panic accelerates exit)
          Any  + sentiment unchanged → pass through
        """
        if signal == "buy" and sentiment_score < -0.20:
            reason = (f"BUY blocked on {ticker}: "
                      f"technicals say buy but sentiment={sentiment_score:+.3f}")
            logger.warning(reason)
            return "hold", reason

        if signal == "sell" and sentiment_score < -0.60:
            reason = (f"Accelerated SELL on {ticker}: "
                      f"media panic sentiment={sentiment_score:+.3f}")
            logger.warning(reason)
            return "sell", reason

        return signal, ""


class SentimentWorker:
    """
    Background daemon thread that pre-warms FinBERT sentiment scores.

    Usage:
        worker = SentimentWorker(engine)
        worker.submit_batch(["NVDA", "AAPL", "MSFT"])   # non-blocking
        score = worker.get("NVDA")                        # instant cache read

    The worker processes the queue continuously.  Scores are written to a
    shared dict protected by a lock.  The main trading loop never blocks on
    FinBERT — it always reads the most recent cached value.

    submit_batch() deduplicates: re-submitting a ticker already in the queue
    is a no-op, preventing the queue from growing unboundedly across cycles.
    """

    def __init__(self, engine: FreeSentimentEngine) -> None:
        self._engine   = engine
        self._queue: queue.Queue[str] = queue.Queue()
        self._pending:  set[str]      = set()          # dedup guard
        self._results:  dict[str, float] = {}
        self._lock      = threading.Lock()
        self._thread    = threading.Thread(target=self._run, daemon=True, name="SentimentWorker")
        self._thread.start()
        logger.info("SentimentWorker started (background FinBERT thread)")

    def submit_batch(self, tickers: list[str]) -> None:
        """Queue a list of tickers for background scoring.  Thread-safe."""
        with self._lock:
            for t in tickers:
                if t not in self._pending:
                    self._pending.add(t)
                    self._queue.put(t)

    def get(self, ticker: str) -> float:
        """Return the most recent cached score (0.0 if not yet scored)."""
        with self._lock:
            return self._results.get(ticker, 0.0)

    def _run(self) -> None:
        while True:
            try:
                ticker = self._queue.get(timeout=2.0)
                score  = self._engine.get_ticker_sentiment(ticker)
                with self._lock:
                    self._results[ticker] = score
                    self._pending.discard(ticker)
            except queue.Empty:
                continue
            except Exception as exc:
                logger.debug(f"SentimentWorker error ({ticker}): {exc}")
