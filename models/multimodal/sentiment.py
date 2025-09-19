import os
import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline

DATA_DIR = os.path.join("data", "news_sentiment")
news_file = os.path.join(DATA_DIR, "stocks_news.csv") 
news_df = pd.read_csv(news_file, parse_dates=["Date"])

# Using Finbert
finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone',num_labels=3)
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
finbert.eval()

nlp = pipeline("sentiment-analysis", model=finbert, tokenizer=tokenizer)

def finbert_score(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = finbert(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1).numpy()[0]
        score = probs[1]*1 + probs[0]*0 + probs[2]*(-1)
    return score

news_df['FinBERT_Score'] = news_df['News'].apply(finbert_score)
print (news_df)

