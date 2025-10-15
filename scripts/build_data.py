import pandas as pd
import os

# ----------------- CONFIG -----------------
NEWS_FILE = "data/news_sentiment/stocks_news_new.csv"      # News CSV with 'Symbol', 'Date', 'News'
OHLC_FOLDER = "data/technical"                         # Folder containing OHLC CSVs for each stock
OUTPUT_FILE = "data/news_sentiment/labeled_news.csv"   # Output labeled CSV
NEUTRAL_THRESHOLD = 0.002                              # ±0.2% threshold for neutral
# ------------------------------------------

# Load news dataset
news_df = pd.read_csv(NEWS_FILE, parse_dates=["Date"])

# Helper: find same-day or next available trading date in OHLC data
def get_valid_trading_date(news_date, ohlc_df):
    if news_date in ohlc_df.index:
        return news_date
    future_dates = ohlc_df.index[ohlc_df.index > news_date]
    return future_dates[0] if len(future_dates) > 0 else None

# Helper: label sentiment based on intraday move (same-day open → close)
def label_sentiment(news_date, ohlc_df):
    valid_date = get_valid_trading_date(news_date, ohlc_df)
    if valid_date is None:
        return None

    try:
        day_open = ohlc_df.loc[valid_date]["Open"]
        day_close = ohlc_df.loc[valid_date]["Close"]
    except Exception:
        return None

    pct_change = (day_close - day_open) / day_open

    if abs(pct_change) <= NEUTRAL_THRESHOLD:
        return "Neutral"
    elif pct_change > 0:
        return "Positive"
    else:
        return "Negative"

# Process all news
final_data = []

for idx, row in news_df.iterrows():
    company = row["Symbol"].strip()
    news_date = pd.to_datetime(row["Date"])
    news_text = row["News"]

    ohlc_path = os.path.join(OHLC_FOLDER, f"{company}_data.csv")
    if not os.path.exists(ohlc_path):
        print(f"Missing OHLC for {company}, skipping...")
        continue

    ohlc_df = pd.read_csv(ohlc_path, parse_dates=["Date"]).set_index("Date").sort_index()
    sentiment = label_sentiment(news_date, ohlc_df)

    if sentiment is None:
        continue

    final_data.append({
        "Company": company,
        "Date": news_date,
        "News": news_text,
        "Label": sentiment
    })

# Create DataFrame and save
labeled_df = pd.DataFrame(final_data)
labeled_df.to_csv(OUTPUT_FILE, index=False)

print(f"Done! Labeled dataset saved to {OUTPUT_FILE}, total rows: {len(labeled_df)}")
