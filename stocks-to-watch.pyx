import requests
import re
from bs4 import BeautifulSoup

regex = "https://www.moneycontrol.com/*stocks-to-watch*"
url = 'https://www.moneycontrol.com/'
reqs = requests.get(url)
soup = BeautifulSoup(reqs.text, 'html.parser')
print ("Still on")

urls = []
for link in soup.find_all('a'):
    print(link.get('href'))