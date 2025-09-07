# Fetch Technical Data along with important indicators for all stocks present in TICKERS
import yfinance as yf
import pandas as pd
import talib
import os, sys
from datetime import datetime, timedelta
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config.tickers import TICKERS

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'technical')
os.makedirs(DATA_DIR, exist_ok=True)

def add_indicators(df):
    close = df["Close"].astype(float).values
    high = df["High"].astype(float).values
    low = df["Low"].astype(float).values
    volume = df["Volume"].astype(float).values

    # --- TA-Lib Indicators ---
    df["SMA_20"] = talib.SMA(close, timeperiod=20)
    df["SMA_50"] = talib.SMA(close, timeperiod=50)
    df["EMA_20"] = talib.EMA(close, timeperiod=20)
    df["EMA_50"] = talib.EMA(close, timeperiod=50)
    df["RSI_14"] = talib.RSI(close, timeperiod=14)
    df["MACD"], df["MACD_Signal"], df["MACD_Hist"] = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    df["BB_upper"], df["BB_middle"], df["BB_lower"] = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
    df["ATR_14"] = talib.ATR(high, low, close, timeperiod=14)
    df["STOCH_K"], df["STOCH_D"] = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)
    df["OBV"] = talib.OBV(close, volume)
    df["CCI_20"] = talib.CCI(high, low, close, timeperiod=20)
    df["Williams_%R"] = talib.WILLR(high, low, close, timeperiod=14)

    # --- Manual Indicators ---
    df["VWAP"] = (df["Volume"] * (df["High"] + df["Low"] + df["Close"]) / 3).cumsum() / df["Volume"].cumsum()
    mfm = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / (df["High"] - df["Low"])
    mfm = mfm.replace([np.inf, -np.inf], np.nan).fillna(0)
    mfv = mfm * df["Volume"]
    df["CMF_20"] = mfv.rolling(20).sum() / df["Volume"].rolling(20).sum()

    df.dropna(inplace=True)
    return df

def clean_dataframe(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.reset_index(inplace=True)
    return df[["Date", "Open", "High", "Low", "Close", "Volume"]]

def fetch_historical_data(ticker, years=5):
    """Fetch historical data and compute indicators."""
    file_path = os.path.join(DATA_DIR, f"{ticker}_data.csv")
    end = datetime.today()
    start = end - timedelta(days=365 * years)

    try:
        df = yf.download(f"{ticker}.NS", start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'))
        if df.empty:
            print(f"No data for {ticker}")
            return

        df = clean_dataframe(df)
        df = add_indicators(df)
        df.to_csv(file_path, index=False)
        print(f"Saved {ticker} data ({len(df)} rows)")
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")

if __name__ == "__main__":
    for ticker in TICKERS:
        fetch_historical_data(ticker)

    with open(os.path.join(DATA_DIR, ".init_done"), "w") as f:
        f.write("Historical fetch complete with indicators")
