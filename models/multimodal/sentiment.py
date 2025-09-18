import pandas as pd
import os
DATA_DIR = os.path.join ("data", "news_sentiment")
news_file = os.path.join(DATA_DIR, "stock_news.csv")
news_df = pd.read_csv(news_file, parse_dates=["Date"])  
print (news_df.head())