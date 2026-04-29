import requests
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---- CONFIG ----
BASE_URL = "https://www.livemint.com/market/stock-market-news/"
PREFIX = "stocks-to-watch"
HEADERS = {"User-Agent": "Mozilla/5.0"}

SEED_URLS = [
    "https://www.livemint.com/market/stock-market-news/stocks-to-watch-tcs-bosch-ntpc-nhpc-among-10-shares-in-focus-today-11775698272963.html",
    "https://www.livemint.com/market/stock-market-news/stocks-to-watch-hdfc-bank-wipro-rvnl-among-10-shares-in-focus-today-11775438785511.html"
]

RANGE = 20000          # keep small for testing
MAX_WORKERS = 20     # threads


# ---- STEP 1: extract IDs ----
def extract_id(url):
    match = re.search(r"-(\d+)\.html", url)
    return int(match.group(1)) if match else None


seed_ids = [extract_id(u) for u in SEED_URLS if extract_id(u)]
print("Seed IDs:", seed_ids)

# ---- STEP 2: generate candidate IDs ----
candidate_ids = set()

for sid in seed_ids:
    for i in range(-RANGE, RANGE):
        candidate_ids.add(sid + i)

candidate_ids = sorted(candidate_ids)
print("Total candidates:", len(candidate_ids))


# ---- STEP 3: check URLs ----
def check_url(cid):
    url = f"{BASE_URL}{PREFIX}-{cid}.html"

    try:
        r = requests.head(url, headers=HEADERS, timeout=5, allow_redirects=True)

        if r.status_code == 200:
            print(f"FOUND: {r.url}")
            return r.url

    except Exception as e:
        print(f"Error for {cid}: {e}")

    return None


valid_urls = set()

print("Starting scan...")

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {executor.submit(check_url, cid): cid for cid in candidate_ids}

    for i, future in enumerate(as_completed(futures)):
        if i % 50 == 0:
            print(f"Progress: {i}/{len(candidate_ids)}")

        result = future.result()
        if result:
            valid_urls.add(result)


# ---- STEP 4: save results ----
with open("mint_stocks_to_watch.txt", "w") as f:
    for u in sorted(valid_urls):
        f.write(u + "\n")

print("\nDone.")
print("Total valid articles found:", len(valid_urls))