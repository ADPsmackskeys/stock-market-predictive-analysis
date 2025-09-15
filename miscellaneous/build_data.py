import pandas as pd
import os

# ----------------- CONFIG -----------------
NEWS_FILE = "data/news_sentiment/stocks_news.csv"      # News CSV with 'Symbol', 'Date', 'News'
OHLC_FOLDER = "data/technical"                         # Folder containing OHLC CSVs for each stock
OUTPUT_FILE = "data/news_sentiment/labeled_news.csv"   # Output labeled CSV
NEUTRAL_THRESHOLD = 0.002                              # ±0.2% threshold for neutral
# ------------------------------------------

# Load news dataset
news_df = pd.read_csv(NEWS_FILE, parse_dates=["Date"])

# Helper: find next trading date in OHLC data
def get_next_trading_date(news_date, ohlc_df):
    future_dates = ohlc_df.index[ohlc_df.index > news_date]
    return future_dates[0] if len(future_dates) > 0 else None

# Helper: label sentiment based on intraday move (next close vs next open)
def label_sentiment(news_date, ohlc_df):
    next_date = get_next_trading_date(news_date, ohlc_df)
    if next_date is None:
        return None

    try:
        next_open = ohlc_df.loc[next_date]["Open"]   # open on next trading day
        next_close = ohlc_df.loc[next_date]["Close"] # close on next trading day
    except Exception:
        return None

    pct_change = (next_close - next_open) / next_open

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
