import pandas as pd
import os

NEWS_FILE = "stocks_news.csv"               
OHLC_FOLDER = "data/technical"            
OUTPUT_FILE = "output/labeled_news.csv"     
NEUTRAL_THRESHOLD = 0.002            


news_df = pd.read_csv (NEWS_FILE, parse_dates=["Date"])

def get_next_trading_date (news_date, ohlc_df):
    
    future_dates = ohlc_df.index[ohlc_df.index > news_date]
    return future_dates[0] if len(future_dates) > 0 else None

def label_sentiment (news_date, ohlc_df):
    
    next_date = get_next_trading_date (news_date, ohlc_df)
    if next_date is None:
        return None

    try:
        prev_close = ohlc_df.loc[:news_date].iloc[-1]["Close"]
        next_close = ohlc_df.loc[next_date]["Close"]
    except IndexError:
        return None

    percentage_change = (next_close - prev_close) / prev_close

    if abs (percentage_change) <= NEUTRAL_THRESHOLD:
        return "Neutral"
    elif percentage_change > 0:
        return "Positive"
    else:
        return "Negative"

final_data = []

for idx, row in news_df.iterrows ():
    company = row["Symbol"].strip ()
    news_date = row["Date"]

    
    ohlc_path = os.path.join (OHLC_FOLDER, f"{company}.NS_data.csv")
    if not os.path.exists (ohlc_path):
        continue

    ohlc_df = pd.read_csv (ohlc_path, parse_dates=["Date"]).set_index ("Date").sort_index ()

    sentiment = label_sentiment (news_date, ohlc_df)

    if sentiment:
        final_data.append ({
            "Company": company,
            "Date": news_date,
            "News": row["News"],
            "Label": sentiment
        })

labeled_df = pd.DataFrame(final_data)
labeled_df.to_csv(OUTPUT_FILE, index=False)

print ("Dataset is ready.")
