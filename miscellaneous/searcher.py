import random

import time
try:
	from googlesearch import search
except ImportError: 
	print("No module named 'google' found")

user_agents = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Version/15.1 Safari/605.1.15",
    "Mozilla/5.0 (Linux; Android 10) AppleWebKit/537.36 Chrome/119.0.0.0 Mobile Safari/537.36"
]

headers = {
    "User-Agent": random.choice(user_agents)
}


# to search
query = "site:www.moneycontrol.com inurl:stocks-to-watch"
searchlist = []

for j in search (query, num_results = 1000, region = "in", lang = "en", headers = headers):
	searchlist.append (j)
	time.sleep (0.2)


