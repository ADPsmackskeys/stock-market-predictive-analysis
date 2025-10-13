# Multimodal Analysis - Train LSTM with sentiments
import os
import pandas as pd
from config.valid_tickers_1000 import TICKERS

DATA_TECH = "data/technical"
NEWS_FILE = "data/news_sentiment/labeled_news.csv"
OUTPUT_DIR = "data/merged"

os.makedirs (OUTPUT_DIR, exist_ok=True)

news_df = pd.read_csv (NEWS_FILE, parse_dates = ["Date"])
label_map = {"Negative": -1, "Neutral": 0, "Positive": 1}
news_df["Sentiment"] = news_df["Label"].map (label_map)
