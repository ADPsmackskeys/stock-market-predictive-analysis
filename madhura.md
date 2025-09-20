# Project Progress Summary - by Madhura
*Please read this file to understand what I have done so far.  
This is not the complete summary, only my contributions.  
This file will be updated later to include contributions by others.*

---

## config/

| File                    | Description                                                                 |
|-------------------------|-----------------------------------------------------------------------------|
| `__init__.py`           | Empty file. Ensures that tickers can be treated as a Python module.         |
| `tickers.py`            | Contains `stocks` as a dictionary and `TICKERS` as a list.                  |
| `valid_tickers_1000.py` | Contains `TICKERS` as a list. Only includes tickers with more than 1000 days of data. |

---

## data/news_sentiment/

| File               | Description                                                                 |
|--------------------|-----------------------------------------------------------------------------|
| `stock_news.csv`   | Contains news in the format: **Company, News, Symbol, Date**.               |
| `labeled_news.csv` | Contains news with sentiment labels in the format: **Company, Date, News, Label**. |

---

## data/technical/

| File                  | Description                                                                 |
|-----------------------|-----------------------------------------------------------------------------|
| `stockname_data.csv`  | Daily stock data (from **Jan 01, 2021 → Sep 19, 2025**) with OHLC, indicators, and fundamentals for 2186 stocks. |

**Columns included:**

| Column Name   | Description                          |
|---------------|--------------------------------------|
| Date          | Trading date                         |
| Adj Close     | Adjusted closing price               |
| Close         | Closing price                        |
| High          | Highest price of the day             |
| Low           | Lowest price of the day              |
| Open          | Opening price                        |
| Volume        | Trading volume                       |
| SMA_20        | 20-day Simple Moving Average         |
| SMA_50        | 50-day Simple Moving Average         |
| EMA_20        | 20-day Exponential Moving Average    |
| EMA_50        | 50-day Exponential Moving Average    |
| RSI_14        | 14-day Relative Strength Index       |
| MACD          | MACD value                           |
| MACD_Signal   | MACD signal line                     |
| MACD_Hist     | MACD histogram                       |
| BB_upper      | Bollinger Bands upper band           |
| BB_middle     | Bollinger Bands middle band          |
| BB_lower      | Bollinger Bands lower band           |
| ATR_14        | 14-day Average True Range            |
| STOCH_K       | Stochastic %K                        |
| STOCH_D       | Stochastic %D                        |
| OBV           | On-Balance Volume                    |
| CCI_20        | 20-day Commodity Channel Index       |
| Williams_%R   | Williams %R indicator                |
| VWAP          | Volume Weighted Average Price        |
| CMF_20        | 20-day Chaikin Money Flow            |
| MarketCap     | Market capitalization                |
| PE            | Price-to-Earnings ratio              |
| EPS           | Earnings Per Share                   |
| PB            | Price-to-Book ratio                  |
| DividendYield | Dividend yield                       |

---

## models/unimodal/

| File             | Description                                                                 |
|------------------|-----------------------------------------------------------------------------|
| `run_models.py`  | Trains the models `Ridge Regression`, `Lasso Regression`, `Random Forest Regressor`, `XGBoost Regressor`, `MLP Regressor`. <br> • Trains data till **May 31st, 2025**, tests on data from **June 1st → Sep 30th, 2025**. <br> • Handles missing values (forward fill, inf/-inf → NaN). <br> • Scales data with `StandardScaler`. <br> • Picks the best model using least RMSE. <br> • Saves prediction plots to `reports/unimodal/stockname_prediction.png`. |

---

## models/multimodal/

| Status         | Description        |
|----------------|--------------------|
| In Progress    | Work still ongoing |

---

## reports/unimodal/

| File                          | Description                                                                 |
|-------------------------------|-----------------------------------------------------------------------------|
| `stockname_prediction.png`    | Prediction of the best model for June → September 2025 for each stock. Comparison of predicted vs actual values. |

---

## scripts/

| File                      | Description                                                                 |
|---------------------------|-----------------------------------------------------------------------------|
| `fetch_technical_data.py` | Collects **OHLC data** + **technical indicators** using **TA-Lib**, and also fetches fundamentals. <br> • Time period: **Jan 1, 2021 → Sep 19, 2025** <br> • Dataset: **2,136 stocks**. |
| `update_tickers.py`       | Generates `valid_tickers_1000.py` to filter stocks with enough history. <br> • Criteria: Only tickers with **> 1000 data entries**. <br> • Purpose: Prevent overfitting, improve stability. <br> • Result: Out of **2,136 stocks**, **1,562 stocks** passed threshold and were stored as a Python list. |

---
