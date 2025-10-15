import requests
from bs4 import BeautifulSoup
import sys
sys.path.insert(1, './config')
from tickers import stocks

base_sitemap = "https://www.moneycontrol.com/news/sitemap/sitemap-post-{}-{}.xml"
keyword = "stocks-to-watch-today"
all_urls = []

for year in range(2021, 2026):
    for month in range(1, 13):
        if year == 2025 and month > 9:
            break

        sitemap_url = base_sitemap.format(year, str(month).zfill(2))
        # print(f"Processing {sitemap_url} ...")

        try:
            response = requests.get(sitemap_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
            if response.status_code != 200:
                # print(f"Failed to fetch {sitemap_url}, status: {response.status_code}")
                continue

            soup = BeautifulSoup(response.content, "xml")  # parse as XML

            for loc_tag in soup.find_all("loc"):
                loc = loc_tag.text.strip()
                if keyword in loc:
                    all_urls.append(loc)

        except Exception as e:
            print(f"Error fetching {sitemap_url}: {e}")

# Deduplicate
searchlist = list(set(all_urls))
with open("scrapelist.txt", 'w') as file:
    for item in searchlist:
    
        file.write(f"{item}\n")
print(f"List successfully saved")

print(f"\nTotal 'stocks-to-watch-today' articles found: {len(searchlist)}")

