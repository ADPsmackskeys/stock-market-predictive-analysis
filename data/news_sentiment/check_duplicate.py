import pandas as pd
import sys

sys.path.insert(1, './config')
from tickers import stocks

# --- Load CSV ---
csv_path = "data/news_sentiment/stocks_news_new.csv"
df = pd.read_csv(csv_path)

unique_col1 = df['Date'].unique()
print("Unique values in 'col1':")
print(unique_col1)

def get_symbol(company_name):
    if pd.isna(company_name):
        return None
    name_lc = company_name.lower().strip()
    for key, symbol in stocks.items():
        if name_lc in key.lower():   # your logic here
            return symbol
    return None

# --- Add Symbol column ---
df["Symbol"] = df["Company"].apply(get_symbol)
print (df[df.isna ().any (axis = 1)])

# --- Save back to same CSV ---

df.to_csv(csv_path, index=False)

# --- Report ---
unmatched = df[df["Symbol"].isna()]["Company"].unique()
print(f"Updated file saved: {csv_path}")
print(f"Unmatched companies: {len(unmatched)}")
if len(unmatched) > 0:
    print("Some unmatched names:", unmatched)
