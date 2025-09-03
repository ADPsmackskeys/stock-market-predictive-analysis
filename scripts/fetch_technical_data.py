# Fetch Technical Data along with important indicators for all stocks present in TICKERS
import yfinance as yf
import pandas as pd
import talib
import os, sys
from datetime import datetime, timedelta
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config.tickers import TICKERS

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'technical')
os.makedirs(DATA_DIR, exist_ok=True)

import talib
import numpy as np

def add_indicators(df):
    # TA-Lib needs float64 arrays
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

    df["MACD"], df["MACD_Signal"], df["MACD_Hist"] = talib.MACD(
        close, fastperiod=12, slowperiod=26, signalperiod=9
    )

    df["BB_upper"], df["BB_middle"], df["BB_lower"] = talib.BBANDS(
        close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
    )

    df["ATR_14"] = talib.ATR(high, low, close, timeperiod=14)

    df["STOCH_K"], df["STOCH_D"] = talib.STOCH(
        high, low, close,
        fastk_period=14,
        slowk_period=3, slowk_matype=0,
        slowd_period=3, slowd_matype=0
    )

    df["OBV"] = talib.OBV(close, volume)

    df["CCI_20"] = talib.CCI(high, low, close, timeperiod=20)

    df["Williams_%R"] = talib.WILLR(high, low, close, timeperiod=14)

    # --- Manual Indicators (use Pandas Series, not numpy) ---
    # VWAP
    df["VWAP"] = (df["Volume"] * (df["High"] + df["Low"] + df["Close"]) / 3).cumsum() / df["Volume"].cumsum()

    # Chaikin Money Flow (CMF) – safe divide by zero
    mfm = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / (df["High"] - df["Low"])
    mfm = mfm.replace([np.inf, -np.inf], np.nan).fillna(0)  # handle div/0
    mfv = mfm * df["Volume"]
    df["CMF_20"] = mfv.rolling(20).sum() / df["Volume"].rolling(20).sum()

    # Drop NaNs created by rolling/indicators
    df.dropna(inplace=True)

    return df


def clean_dataframe(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)  # drop ticker level
    df.reset_index(inplace=True)
    cols_to_keep = ["Date", "Open", "High", "Low", "Close", "Volume"]
    return df[cols_to_keep]


def fetch_or_update_data(ticker, years=5):
    file_path = os.path.join(DATA_DIR, f"{ticker}_data.csv")
    end = datetime.today()

    if os.path.exists(file_path):
        df_existing = pd.read_csv(file_path, parse_dates=["Date"])
        last_date = df_existing["Date"].max()
        start = pd.to_datetime(last_date) + timedelta(days=1)

        if start <= end:
            print(f"Updating {ticker} from {start.date()} to {end.date()}")
            df_new = yf.download(ticker, start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'))
            if not df_new.empty:
                df_new = clean_dataframe(df_new)
                df_updated = pd.concat([df_existing, df_new], ignore_index=True)
                df_updated.drop_duplicates(subset="Date", inplace=True)
                df_updated = add_indicators(df_updated)
                df_updated.to_csv(file_path, index=False)
                print(f"Updated {ticker}: added {len(df_new)} rows (total {len(df_updated)})")
            else:
                print(f"No new data for {ticker}")
        else:
            print(f"{ticker} already up-to-date")
    else:
        start = end - timedelta(days=365 * years)
        print(f"Fetching full history for {ticker}")
        df = yf.download(ticker, start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'))
        if not df.empty:
            df = clean_dataframe(df)
            df = add_indicators(df)
            df.to_csv(file_path, index=False)
            print(f"Saved {ticker} data ({len(df)} rows)")
        else:
            print(f"No data for {ticker}")


if __name__ == "__main__":
    for ticker in TICKERS:
        fetch_or_update_data(ticker)

    with open(os.path.join(DATA_DIR, ".init_done"), "w") as f:
        f.write("Historical fetch/update complete with indicators")
