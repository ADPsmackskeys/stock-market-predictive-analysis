import os
import sys
import pandas as pd
import yfinance as yf
import talib as ta
from datetime import datetime, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.tickers import TICKERS  

DATA_DIR = "data/technical"
os.makedirs(DATA_DIR, exist_ok=True)

# A gap this long between two consecutive *real* (post zero-volume-strip)
# trading days means the stock was effectively halted/dormant for an extended
# stretch (long suspension, corporate action, illiquidity). Treating the rows
# on either side as adjacent would blend two unrelated price regimes into one
# rolling-window indicator value and would also produce a fake giant "return"
# across the gap in downstream forward-window calculations. Segmenting resets
# indicator lookback (and lets signal_scanner.py mask forward windows) at
# each such gap, so every indicator/return is computed only within a single
# unbroken run of real trading days.
GAP_THRESHOLD_DAYS = 45

failed_tickers = []

# Start and end dates
START_DATE = "2021-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")
# yf.download's `end` is exclusive, so requesting end=END_DATE would never
# actually fetch today's session -- push the request window one day further.
FETCH_END = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")

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
    df["EMA_200"] = ta.EMA(close, timeperiod=200)

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


def assign_segments(df):
    """Number each row by which unbroken run of real trading days it belongs
    to -- increments every time the gap since the previous real trade exceeds
    GAP_THRESHOLD_DAYS. Must run after clean_raw() so gaps reflect real
    trading days only, not zero-volume stale-quote rows."""
    df = df.copy()
    if df.empty:
        df["Segment"] = pd.Series(dtype=int)
        return df
    gap_days = df["Date"].diff().dt.days.fillna(0)
    df["Segment"] = (gap_days > GAP_THRESHOLD_DAYS).cumsum()
    return df


def add_indicators_by_segment(df):
    """Apply add_indicators() separately within each segment, so a rolling
    window (e.g. EMA_200, ATR_14) never blends prices from before and after a
    long trading gap. Segments too short for a given indicator's lookback
    simply get NaN there, same as at the start of any fresh history."""
    pieces = [add_indicators(seg.reset_index(drop=True)) for _, seg in df.groupby("Segment", sort=True)]
    return pd.concat(pieces, ignore_index=True)


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

RAW_COLS = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]


def clean_raw(df):
    """Drop rows with zero (or missing) volume -- these are not real trading
    days. When a stock isn't actually trading, Yahoo Finance carries the last
    real price forward as a stale quote instead of omitting the row, so the
    next genuine trade after such a gap looks like an extreme single-day
    price move that never happened (e.g. a stock frozen at a stale price for
    months, then "jumping" hundreds of percent the day real trading resumes).
    Dropping these rows means every remaining row is a genuine trading day,
    so returns/indicators are computed only across real price history."""
    if df.empty:
        return df
    return df[df["Volume"] > 0].reset_index(drop=True)


def update_ticker(ticker, force_clean=False):
    """Fetch and merge only the missing days for one ticker, then recompute
    indicators over the full price history. Returns a short status string.

    force_clean=True also rebuilds tickers with no new data to fetch -- this
    covers both zero-volume rows needing stripping and segment/indicator
    recomputation after a change to that logic, since "already up to date"
    would otherwise skip a ticker entirely and never pick up the fix."""
    file_path = os.path.join(DATA_DIR, f"{ticker}_data.csv")

    old_df = None
    removed = 0
    fetch_start = START_DATE
    if os.path.exists(file_path):
        old_df = pd.read_csv(file_path, parse_dates=["Date"])
        if not old_df.empty:
            orig_len = len(old_df)
            old_df = clean_raw(old_df)
            removed = orig_len - len(old_df)
            if not old_df.empty:
                last_date = old_df["Date"].max()
                fetch_start = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    has_new_data = fetch_start <= END_DATE
    if not has_new_data and not force_clean:
        return "already up to date"

    yf_ticker = f"{ticker}.NS" if not ticker.endswith((".NS", ".BO")) else ticker
    new_df = pd.DataFrame(columns=RAW_COLS)

    if has_new_data:
        print(f"Processing {ticker} from {fetch_start}")
        fetched = yf.download(
            yf_ticker,
            start=fetch_start,
            end=FETCH_END,
            interval="1d",
            auto_adjust=False,
            progress=False
        )
        if not fetched.empty:
            if isinstance(fetched.columns, pd.MultiIndex):
                fetched.columns = [c[0] for c in fetched.columns]
            for col in ["Open", "High", "Low", "Close", "Volume"]:
                fetched[col] = pd.to_numeric(fetched[col], errors="coerce")
            fetched = fetched.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
            if not fetched.empty:
                fetched.reset_index(inplace=True)
                new_df = clean_raw(fetched[RAW_COLS])

    # Merge with the existing raw price history (not the old indicator columns)
    # before recomputing indicators, since things like EMA_200 need months of
    # trailing lookback that the freshly-downloaded rows alone don't have.
    if old_df is not None and not old_df.empty:
        old_raw = old_df[RAW_COLS]
        combined_raw = (
            pd.concat([old_raw, new_df], ignore_index=True)
            .drop_duplicates(subset=["Date"])
            .sort_values("Date")
            .reset_index(drop=True)
        )
    elif not new_df.empty:
        combined_raw = new_df.sort_values("Date").reset_index(drop=True)
    else:
        return "no valid data"

    combined_raw = assign_segments(combined_raw)
    combined = add_indicators_by_segment(combined_raw)

    fundamental_cols = ["MarketCap", "PE", "EPS", "PB", "DividendYield"]
    if has_new_data:
        combined = add_fundamentals(yf_ticker, combined)
    elif old_df is not None and all(c in old_df.columns for c in fundamental_cols):
        # No new rows fetched, so nothing about the company changed -- reuse
        # the already-fetched fundamentals instead of hitting the network
        # again for every ticker on a recompute-only (e.g. --clean) pass.
        for col in fundamental_cols:
            combined[col] = old_df[col].iloc[-1]
    else:
        combined = add_fundamentals(yf_ticker, combined)
    combined.to_csv(file_path, index=False)

    n_segments = combined["Segment"].nunique()

    parts = []
    if removed:
        parts.append(f"removed {removed} zero-volume rows")
    if len(new_df):
        parts.append(f"{len(new_df)} new rows")
    if n_segments > 1:
        parts.append(f"{n_segments} segments (gap>{GAP_THRESHOLD_DAYS}d)")
    parts.append(f"{len(combined)} total")
    return ", ".join(parts)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fetch/update NSE technical data")
    parser.add_argument("--clean", action="store_true",
                         help="Force-rebuild every ticker's cleaning/segmenting/indicators, even ones with no new data to fetch")
    args = parser.parse_args()

    for ticker in TICKERS:
        try:
            status = update_ticker(ticker, force_clean=args.clean)
            print(f"  {ticker}: {status}")
        except Exception as e:
            print(f"Error for {ticker}: {e}")
            failed_tickers.append(ticker)

    print("\nFailed Tickers:", failed_tickers)
