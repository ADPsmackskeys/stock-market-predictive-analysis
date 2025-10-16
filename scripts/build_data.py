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

# Helper: find same-day or next available trading date in OHLC data
def get_valid_trading_date(news_date, ohlc_df):
    if news_date in ohlc_df.index:
        return news_date
    future_dates = ohlc_df.index[ohlc_df.index > news_date]
    return future_dates[0] if len(future_dates) > 0 else None

# Helper: create a single label from a percentage change
def create_label(pct_change):
    if abs(pct_change) <= NEUTRAL_THRESHOLD:
        return "Neutral"
    elif pct_change > 0:
        return "Positive"
    else:
        return "Negative"

# --- NEW FUNCTION ---
# Get labels for multiple time horizons (T+1, T+2, T+5 trading sessions)
def get_labels_for_horizons(news_date, ohlc_df):
    start_date = get_valid_trading_date(news_date, ohlc_df)
    if start_date is None:
        return None

    try:
        # Find the integer index location of our starting date
        start_loc = ohlc_df.index.get_loc(start_date)
        start_open = ohlc_df.iloc[start_loc]["Open"]
    except KeyError:
        return None # Should not happen due to get_valid_trading_date, but good practice

    labels = {}

    # Define horizons in terms of trading sessions from the start date
    # T+1: Open of start_date -> Close of start_date (index offset 0)
    # T+2: Open of start_date -> Close of next day (index offset 1)
    # T+3: Open of start_date -> Close of day after that (index offset 2)
    horizons = {"T1": 0, "T2": 1, "T3": 2}

    for name, offset in horizons.items():
        try:
            # Find the closing price for the target future date
            future_close = ohlc_df.iloc[start_loc + offset]["Close"]
            pct_change = (future_close - start_open) / start_open
            labels[f"Label_{name}"] = create_label(pct_change)
        except IndexError:
            # This happens if the news is too close to the end of the OHLC data
            labels[f"Label_{name}"] = None

    return labels

# Process all news
final_data = []

# --- MODIFIED MAIN LOOP ---
for idx, row in news_df.iterrows():
    company = row["Symbol"].strip()
    news_date = pd.to_datetime(row["Date"])
    news_text = row["News"]

    ohlc_path = os.path.join(OHLC_FOLDER, f"{company}_data.csv")
    if not os.path.exists(ohlc_path):
        print(f"Missing OHLC for {company}, skipping...")
        continue

    ohlc_df = pd.read_csv(ohlc_path, parse_dates=["Date"]).set_index("Date").sort_index()
    
    # Get the dictionary of labels for all horizons
    all_labels = get_labels_for_horizons(news_date, ohlc_df)

    if all_labels is None:
        continue

    # Prepare the dictionary for the final DataFrame row
    data_row = {
        "Company": company,
        "Date": news_date,
        "News": news_text,
    }
    # Add all the labels (Label_T1, Label_T2, etc.) to the row
    data_row.update(all_labels)
    final_data.append(data_row)

# Create DataFrame and save
labeled_df = pd.DataFrame(final_data)
labeled_df.to_csv(OUTPUT_FILE, index=False)

print(f"Done! Labeled dataset saved to {OUTPUT_FILE}, total rows: {len(labeled_df)}")