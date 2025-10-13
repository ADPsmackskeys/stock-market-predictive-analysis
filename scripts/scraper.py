import requests
from searcher import searchlist
from bs4 import BeautifulSoup
import pandas as pd
import re
from  config.tickers import stocks #To those wondering about this. Go check config.tickers.py
from htmldate import find_date

def normalize (name):
    return re.sub(r'[^A-Za-z0-9 ]+', '', name).strip ().lower ()

cnt = 0
data = []
newslist = []

for k in searchlist:
    cnt += 1
    print ("Link ", cnt)
    response = requests.get (k)
    print (response)
    soup = BeautifulSoup (response.content,'html.parser')
    
    headlines = soup.find_all ('p')
    sentences = []

    j = 0

    for i in range (len (headlines)):
        if (headlines[i].text == 'Stocks to Watch'):
            flag = True
            j = i + 1

    try:
        while ("Disclosure:" not in headlines[j].text):
            if ("News18" in headlines[j].text or "Disclaimer:" in headlines[j].text or "Discover the latest Business News" in headlines[j].text or "Stocks Trade Ex-Dividend" in headlines[j].text or "Block" in headlines[j].text or "Bulk" in headlines[j].text):
                break

            p_tag = headlines[j]
            for br in p_tag.find_all('br'):
                br.insert_after('\n')  
                br.unwrap()  

            text = p_tag.get_text(separator="\n").strip()

            print(text, "\n")
            sentences.append (headlines[j].text)
            j += 1

        for i in range(len(sentences) - 1):
            company_name = sentences[i].strip()
            news_text = sentences[i + 1].strip()
            newslist.append (news_text)

        for key in stocks.keys():
            if (company_name.lower ()) in (key.lower ()) and "Q1 (Consolidated" not in news_text[-1]:
                data.append({
                    "Company": key,
                    "News": news_text,
                    "Symbol": stocks[key],
                    "Date": find_date (k)
                })
            
    except IndexError:
        continue


df = pd.DataFrame(data)
df.to_csv("data/news_sentiment/stocks_news.csv", index = False, encoding = "utf-8-sig")
print("Saved to stocks_news.csv")