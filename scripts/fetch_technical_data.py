import os
import pandas as pd
import yfinance as yf
import talib as ta
from datetime import datetime
from config.tickers import TICKERS  # adjust your import if needed

DATA_DIR = "data/technical"
os.makedirs(DATA_DIR, exist_ok=True)

failed_tickers = []

# Start and end dates
START_DATE = "2021-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")

def add_indicators(df):
    close = df["Close"].astype(float).values
    high = df["High"].astype(float).values
    low = df["Low"].astype(float).values
    volume = df["Volume"].astype(float).values

    # Moving Averages
    df["SMA_20"] = ta.SMA(close, timeperiod=20)
    df["SMA_50"] = ta.SMA(close, timeperiod=50)
    df["EMA_20"] = ta.EMA(close, timeperiod=20)
    df["EMA_50"] = ta.EMA(close, timeperiod=50)

    # RSI
    df["RSI_14"] = ta.RSI(close, timeperiod=14)

    # MACD
    macd, macdsignal, macdhist = ta.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    df["MACD"] = macd
    df["MACD_Signal"] = macdsignal
    df["MACD_Hist"] = macdhist

    # Bollinger Bands
    upper, middle, lower = ta.BBANDS(close, timeperiod=20)
    df["BB_upper"] = upper
    df["BB_middle"] = middle
    df["BB_lower"] = lower

    # ATR
    df["ATR_14"] = ta.ATR(high, low, close, timeperiod=14)

    # Stochastic Oscillator
    slowk, slowd = ta.STOCH(high, low, close)
    df["STOCH_K"] = slowk
    df["STOCH_D"] = slowd

    # OBV
    df["OBV"] = ta.OBV(close, volume)

    # CCI
    df["CCI_20"] = ta.CCI(high, low, close, timeperiod=20)

    # Williams %R
    df["Williams_%R"] = ta.WILLR(high, low, close, timeperiod=14)

    # VWAP
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    df["VWAP"] = (typical_price * df["Volume"]).cumsum() / df["Volume"].cumsum()

    # Chaikin Money Flow
    mf_multiplier = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / (df["High"] - df["Low"])
    mf_volume = mf_multiplier * df["Volume"]
    df["CMF_20"] = mf_volume.rolling(20).sum() / df["Volume"].rolling(20).sum()

    return df

def add_fundamentals(ticker, df):
    try:
        info = yf.Ticker(ticker).info
        fundamentals = {
            "MarketCap": info.get("marketCap", None),
            "PE": info.get("trailingPE", None),
            "EPS": info.get("trailingEps", None),
            "PB": info.get("priceToBook", None),
            "DividendYield": info.get("dividendYield", None)
        }
        for key, value in fundamentals.items():
            df[key] = value
        return df
    except Exception as e:
        print(f"Failed to fetch fundamentals for {ticker}: {e}")
        return df

for ticker in TICKERS:
    file_path = os.path.join(DATA_DIR, f"{ticker}_data.csv")
    print(f"Processing {ticker}")

    try:
        yf_ticker = f"{ticker}.NS" if not ticker.endswith((".NS", ".BO")) else ticker
        new_df = yf.download(
            yf_ticker,
            start=START_DATE,
            end=END_DATE,
            interval="1d",
            auto_adjust=False,
            progress=False
        )

        if new_df.empty:
            print(f"No data for {ticker}")
            failed_tickers.append(ticker)
            continue

        if isinstance(new_df.columns, pd.MultiIndex):
            new_df.columns = [c[0] for c in new_df.columns]

        for col in ["Open", "High", "Low", "Close", "Volume"]:
            new_df[col] = pd.to_numeric(new_df[col], errors="coerce")

        new_df = new_df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
        if new_df.empty:
            print(f"No valid rows for {ticker}")
            failed_tickers.append(ticker)
            continue

        new_df.reset_index(inplace=True)
        new_df = add_indicators(new_df)
        new_df = add_fundamentals(yf_ticker, new_df)

        if os.path.exists(file_path):
            old_df = pd.read_csv(file_path, parse_dates=["Date"])
            combined = pd.concat([old_df, new_df]).drop_duplicates(subset=["Date"]).sort_values("Date")
            combined.to_csv(file_path, index=False)
        else:
            new_df.to_csv(file_path, index=False)

    except Exception as e:
        print(f"Error for {ticker}: {e}")
        failed_tickers.append(ticker)

print("\nFailed Tickers:", failed_tickers)
