import pandas as pd
import os

# ----------------- CONFIG -----------------
NEWS_FILE = "data/news_sentiment/stocks_news_new.csv"  # News CSV with 'Symbol', 'Date', 'News'
OHLC_FOLDER = "data/technical"                         # Folder containing OHLC CSVs for each stock
OUTPUT_FILE = "data/news_sentiment/labeled_news.csv"   # Output labeled CSV
NEUTRAL_THRESHOLD = 0.002                              # ±0.2% threshold for neutral
# ------------------------------------------

# Load news dataset
news_df = pd.read_csv(NEWS_FILE, parse_dates=["Date"])

# ----------------- HELPERS -----------------
def get_valid_trading_date(news_date, ohlc_df):
    """Return same-day or next valid trading date."""
    if news_date in ohlc_df.index:
        return news_date
    future_dates = ohlc_df.index[ohlc_df.index > news_date]
    return future_dates[0] if len(future_dates) > 0 else None


def label_sentiment(news_date, ohlc_df):
    """Label based on intraday open → close move."""
    valid_date = get_valid_trading_date(news_date, ohlc_df)
    if valid_date is None:
        return None, None, None

    try:
        day_open = ohlc_df.loc[valid_date, "Open"]
        day_close = ohlc_df.loc[valid_date, "Close"]
    except Exception:
        return None, None, None

    # Calculate same-day % change
    pct_change_intraday = (day_close - day_open) / day_open

    # Label same-day sentiment
    if abs(pct_change_intraday) <= NEUTRAL_THRESHOLD:
        intraday_label = "Neutral"
    elif pct_change_intraday > 0:
        intraday_label = "Positive"
    else:
        intraday_label = "Negative"

    # ----- NEW: Compare previous close vs current close -----
    prev_dates = ohlc_df.index[ohlc_df.index < valid_date]
    if len(prev_dates) == 0:
        return intraday_label, None, None  # No previous day to compare

    prev_date = prev_dates[-1]
    prev_close = ohlc_df.loc[prev_date, "Close"]
    pct_change_prevclose = (day_close - prev_close) / prev_close

    # Label based on previous close comparison
    if abs(pct_change_prevclose) <= NEUTRAL_THRESHOLD:
        prevclose_label = "Neutral"
    elif pct_change_prevclose > 0:
        prevclose_label = "Positive"
    else:
        prevclose_label = "Negative"

    return intraday_label, pct_change_intraday, prevclose_label


# ----------------- MAIN PROCESS -----------------
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

    intraday_label, pct_intraday, prevclose_label = label_sentiment(news_date, ohlc_df)
    if intraday_label is None:
        continue

    final_data.append({
        "Company": company,
        "Date": news_date,
        "News": news_text,
        "Intraday_Label": intraday_label,
        "PrevClose_Label": prevclose_label,
        "Intraday_Change(%)": round(pct_intraday * 100, 3) if pct_intraday is not None else None
    })

# Save labeled data
labeled_df = pd.DataFrame(final_data)
labeled_df.to_csv(OUTPUT_FILE, index=False)

print(f"Done! Labeled dataset saved to {OUTPUT_FILE}")
print(f"Total rows labeled: {len(labeled_df)}")
