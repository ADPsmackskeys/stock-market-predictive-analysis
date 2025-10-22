import requests
# from searcher import searchlist
from bs4 import BeautifulSoup
import pandas as pd
import re
import sys
sys.path.insert(1, './config')
from tickers import stocks, keys # To those wondering about this. Go check config/tickers.py. 
from htmldate import find_date # Noted. Thanks for doing it. Fixed it a bit (We can't treat folders as modules. And there was no space after the # <3)

def normalize (name):
    return re.sub(r'[^A-Za-z0-9 ]+', '', name).strip ().lower ()

def find_matching_value(data_dict, word):
    for key, value in data_dict.items():
        if word.lower() in key.lower():
            return (key, value)

with open('./scrapelist.txt', 'r') as f:
    # This loops through each line in the file
    # .strip() removes leading/trailing whitespace, including '\n'
    searchlist = [line.strip() for line in f]

# print(my_lines_stripped)
# Output: ['Line 1', 'Line 2', 'Line 3']

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
            if ("News18" in headlines[j].text or "Disclaimer:" in headlines[j].text or "Discover the latest Business News" in headlines[j].text):
                break

            p_tag = headlines[j]
            for br in p_tag.find_all('br'):
                br.insert_after('\n')  
                br.unwrap()  

            text = p_tag.get_text(separator="\n").strip()

            print(text, "\n")
            sentences.append (text)
            j += 1

        cnt = 0
        for i in range(len(sentences) - 1):
            company_name = sentences[i].strip()
            news_text = sentences[i + 1].strip()
            newslist.append (news_text)

            if (company_name.lower ()) in (keys):
                print (cnt)
                cnt += 1 
                data.append({
                    "Company": find_matching_value (stocks, company_name)[0],
                    "News": news_text,
                    "Symbol": find_matching_value (stocks, company_name)[1],
                    "Date": find_date (k)
                })
            
    except IndexError:
        continue


df = pd.DataFrame(data)
df.to_csv("data/news_sentiment/stocks_news.csv", index = False, encoding = "utf-8-sig")
print("Saved to stocks_news.csv")