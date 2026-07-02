# Stock Market Predictive Analysis (NSE)

A pipeline for predicting short-term price moves of NSE-listed (Indian) stocks by combining
technical-indicator signal mining with news-sentiment analysis. It covers ~2000 tickers with
daily OHLCV history from 2021 onward.

## Repository Layout

```
config/                  Ticker universe and helpers
  tickers.py              Company name -> NSE symbol map (~2000 stocks)
  valid_tickers_1000.py   Auto-generated subset with enough history (>1000 rows) to model safely

data/
  technical/               Per-symbol OHLCV + indicator CSVs: <SYMBOL>_data.csv
  news_sentiment/          Scraped "stocks to watch" news and labeled news CSVs

scripts/
  fetch_technical_data.py  Downloads OHLCV from yfinance, computes indicators + fundamentals
  update_tickers.py        Filters tickers by history length, writes valid_tickers_1000.py
  scraper.py               Crawls Moneycontrol sitemaps for "stocks-to-watch-today" article URLs
  scrapemint.py            Fetches those articles, extracts per-company blurbs into stocks_news.csv
  searcher.py               Standalone helper to pull "stocks to watch" text from one Moneycontrol article
  etnow.py                 Brute-force ID scan of livemint.com to discover "stocks-to-watch" articles
  build_data.py            Labels each news item Positive/Negative/Neutral from same-day price move
  combo.py                 Per-symbol candlestick-pattern + indicator-state co-occurrence analysis
  signal_scanner.py        Scans ~60 technical signals/combos per symbol, scores them by expected
                            value, profit factor and statistical significance; builds a cross-symbol
                            leaderboard

models/
  unimodal/run_models.py    Trains Ridge/Lasso/RandomForest/XGBoost/MLP per ticker to predict next
                             closing price from technical features only
  multimodal/sentiment.py   Scores news headlines with FinBERT
  multimodal/technical.py   Merges labeled news with technical data for a sentiment+technical model
                             (work in progress — fusion model not yet implemented)

reports/unimodal/          Prediction plots and RMSE/MAPE summary from run_models.py
results/combos/            Per-symbol output of combo.py
results/signals_v2/        Per-symbol signal scores + NSE_signal_leaderboard.csv from signal_scanner.py
```

## Pipeline

1. **Collect technical data** — `scripts/fetch_technical_data.py` pulls daily OHLCV per ticker
   from Yahoo Finance and computes indicators (SMA/EMA, RSI, MACD, Bollinger Bands, ATR,
   Stochastic, OBV, CCI, Williams %R, VWAP, CMF) plus basic fundamentals, saving one CSV per
   symbol under `data/technical/`.
2. **Curate the ticker universe** — `scripts/update_tickers.py` keeps only tickers with more
   than 1000 rows of history to reduce overfitting risk, writing `config/valid_tickers_1000.py`.
3. **Collect news** — `scripts/scraper.py` finds "Stocks to Watch" article URLs from
   Moneycontrol's sitemaps; `scripts/scrapemint.py` visits each URL and extracts per-company
   news blurbs into `data/news_sentiment/stocks_news.csv`. `scripts/etnow.py` is an alternate
   source that brute-force scans LiveMint article IDs.
4. **Label news by market reaction** — `scripts/build_data.py` joins each news item with same-day
   OHLC data and labels it Positive/Negative/Neutral based on the intraday move and the move from
   previous close, producing `data/news_sentiment/labeled_news.csv`.
5. **Mine technical signals** — `scripts/combo.py` and `scripts/signal_scanner.py` detect
   candlestick patterns and indicator states per symbol, evaluate their forward-looking
   performance (win rate, expected value, profit factor, Fisher-exact significance) and rank the
   best signals/combos, saving results under `results/`.
6. **Model** — `models/unimodal/run_models.py` trains several regressors per ticker on technical
   features to predict next closing price. `models/multimodal/` scores news sentiment with
   FinBERT and is intended to fuse it with technical data into a combined model; this fusion step
   is not yet complete.

## Requirements

Python 3.12+ with: `pandas`, `numpy`, `scipy`, `scikit-learn`, `xgboost`, `matplotlib`,
`yfinance`, `TA-Lib`, `requests`, `beautifulsoup4`, `htmldate`, `torch`, `transformers`.

TA-Lib requires its underlying C library to be installed separately before `pip install TA-Lib`
will succeed.

## Usage

```bash
# Refresh technical data for all tickers in config/tickers.py
python scripts/fetch_technical_data.py

# Rebuild the "sufficient history" ticker list
python scripts/update_tickers.py

# Collect and label news sentiment
python scripts/scraper.py
python scripts/scrapemint.py
python scripts/build_data.py

# Mine technical signals
python scripts/combo.py
python scripts/signal_scanner.py            # all symbols
python scripts/signal_scanner.py --symbol RELIANCE

# Train per-ticker price prediction models
python models/unimodal/run_models.py
```

## Notes

- Data is scraped from Moneycontrol and LiveMint; be mindful of request rates and each site's
  terms of service.
- The multimodal (sentiment + technical) model is under active development — sentiment scoring
  and news labeling exist, but the fused predictive model is not yet built.
