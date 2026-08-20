# Trader

An algorithmic trading bot that trades a single stock ticker based on the sentiment of its recent news headlines, using Lumibot for strategy execution/backtesting and FinBERT for financial sentiment classification.

## How it works

On each daily trading iteration, the bot:

1. Pulls the last 3 days of news headlines for the configured ticker via the Alpaca API.
2. Runs the headlines through FinBERT (finbert_utils.py) to get a sentiment label (positive / negative / neutral) and a confidence score.
3. Sizes a position based on a configurable fraction of available cash (cash_at_risk).
4. If sentiment is positive with >99.9% confidence, closes any short position and places a bracket buy order (take-profit at +20%, stop-loss at -5%).
5. If sentiment is negative with >99.9% confidence, closes any long position and places a bracket sell order (take-profit at -20%, stop-loss at +5%).

The strategy can be backtested against historical Yahoo Finance data or run live/paper through Alpaca.

## Stack

- Trading/backtesting: Lumibot, Yahoo Finance historical data
- Broker: Alpaca (paper trading by default)
- Sentiment model: FinBERT (ProsusAI/finbert) via Hugging Face transformers, with CUDA acceleration when available
- Language: Python

## Setup

1. Create an Alpaca account (paper trading is fine) and generate API credentials.
2. In tradingbot.py, set:
   API_KEY = "your-alpaca-key"
   API_SECRET = "your-alpaca-secret"
   BASE_URL = "https://paper-api.alpaca.markets"
3. Install dependencies:
   pip install lumibot alpaca-trade-api timedelta transformers torch
4. Set the ticker and risk fraction in the parameters dict (default: TSLA, 50% of cash at risk).
5. Run:
   python tradingbot.py
   This runs a backtest from 2020-01-01 to 2023-12-31 by default. To trade live/paper, swap the strategy.backtest(...) call for trader.add_strategy(strategy) / trader.run_all().
