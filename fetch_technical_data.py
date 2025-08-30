import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta

DATA_DIR = os.path.join(os.path.dirname(__file__), '.', 'data', 'technical')
os.makedirs(DATA_DIR, exist_ok=True)

TICKERS = [
    "RELIANCE.NS", "BHARTIARTL.NS", "TCS.NS", "INFY.NS", "HCLTECH.NS",
    "ICICIBANK.NS", "HDFCBANK.NS", "SBIN.NS", "LTI.NS", "TECHM.NS"
]

def fetch_historical_data(ticker, years=5):
    try:
        end = datetime.today()
        start = end - timedelta(days=365 * years)
        df = yf.download(ticker, start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'))
        if not df.empty:
            df.reset_index(inplace=True)
            file_path = os.path.join(DATA_DIR, f"{ticker}_data.csv")
            df.to_csv(file_path, index=False)
            print(f"Saved {ticker} historical data")
        else:
            print(f"No data for {ticker}")
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")

if __name__ == "__main__":
    for ticker in TICKERS:
        fetch_historical_data(ticker)
    
    with open(os.path.join(DATA_DIR, ".init_done"), "w") as f:
        f.write("Historical fetch complete")
