import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import sys
import os
sys.path.insert(1, './config')
from tickers import stocks, keys
from htmldate import find_date

def normalize(name):
    return re.sub(r'[^A-Za-z0-9 ]+', '', name).strip().lower()

def find_matching_value(data_dict, word):
    for key, value in data_dict.items():
        if word.lower() in key.lower():
            return (key, value)

with open('./scrapelist.txt', 'r') as f:
    searchlist = [line.strip() for line in f]

cnt = 0
data = []
newslist = []
file_path = "data/news_sentiment/stocks_news.csv"

for k in searchlist:
    cnt += 1
    print("Link ", cnt)
    response = requests.get(k)
    print(response)
    soup = BeautifulSoup(response.content, 'html.parser')

    headlines = soup.find_all('p')
    sentences = []

    j = 0

    for i in range(len(headlines)):
        if headlines[i].text == 'Stocks to Watch':
            flag = True
            j = i + 1

    try:
        while "Disclosure:" not in headlines[j].text:
            if ("News18" in headlines[j].text or "Disclaimer:" in headlines[j].text or "Discover the latest Business News" in headlines[j].text):
                break

            p_tag = headlines[j]
            for br in p_tag.find_all('br'):
                br.insert_after('\n')
                br.unwrap()

            text = p_tag.get_text(separator="\n").strip()
            sentences.append(text)
            j += 1

        lower_to_original = {k.lower(): k for k in stocks.keys()}
        keys = list(lower_to_original.keys())

        for i in range(len(sentences) - 1):
            company_name = sentences[i].strip()
            news_text = sentences[i + 1].strip()

            if len(news_text.split()) < 5:
                continue

            newslist.append(news_text)

            if any(company_name.lower() in k for k in keys) and news_text in newslist:
                print(company_name.lower())
                cnt += 1
                data.append({
                    "Company": find_matching_value(stocks, company_name)[0],
                    "News": news_text,
                    "Symbol": find_matching_value(stocks, company_name)[1],
                    "Date": find_date(k)
                })

    except IndexError:
        continue

# Write once at the end, outside the loop
if data:
    df_new = pd.DataFrame(data)

    if os.path.exists(file_path):
        try:
            df_existing = pd.read_csv(file_path, quoting=1, on_bad_lines='skip')
            df = pd.concat([df_existing, df_new], ignore_index=True)
            df = df.drop_duplicates()
            df.to_csv(file_path, index=False, encoding="utf-8-sig", quoting=1)
            print("Saved to stocks_news.csv")
        
        except pd.errors.ParserError:
            print("Warning: CSV was corrupted, rebuilding from current session data only.")
            df = df_new
    else:
        df = df_new
print("Saved to stocks_news.csv")