"""
Scrape LiveMint "Stocks to Watch" articles discovered via the Wayback Machine.

Why the Wayback Machine instead of a search engine: LiveMint only publishes
sitemaps for the last two days, and Google/DuckDuckGo-based discovery either
rate-limits or truncates results well below the number of articles that
actually exist. The Internet Archive's CDX API supports literal URL-prefix
search against its full historical index, with no rate limiting and no
per-query cap, so it finds far more of these articles.

Each article's numeric id is recorded in a "seen ids" file after it's scraped, so
re-running the script only fetches articles published since the last run instead
of re-scraping the full history every time.

Usage:
    python scripts/scrape_livemint_archive.py
    python scripts/scrape_livemint_archive.py --limit 20   # test on a few articles
    python scripts/scrape_livemint_archive.py --rescan     # ignore the seen-ids cache
"""

import os
import re
import sys
import time
import argparse
import requests
import pandas as pd
from bs4 import BeautifulSoup
from htmldate import find_date

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.tickers import stocks

# ----------------- CONFIG -----------------
CDX_API = "https://web.archive.org/cdx/search/cdx"

# LiveMint has used a few different path shapes for these articles over the years.
URL_PREFIXES = [
    "https://www.livemint.com/market/stock-market-news/stocks-to-watch",
    "https://www.livemint.com/market/stock-market-news/stock-to-watch",
    "https://www.livemint.com/market/stocks-to-watch",
]

ARTICLE_ID_RE = re.compile(r"-(\d+)\.html")
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}
OUTPUT_FILE = "data/news_sentiment/stocks_news_new.csv"
SEEN_IDS_FILE = "data/news_sentiment/livemint_scraped_ids.txt"
REQUEST_DELAY = 0.5  
MIN_NEWS_WORDS = 5
# ------------------------------------------


def load_seen_ids():
    if not os.path.exists(SEEN_IDS_FILE):
        return set()
    with open(SEEN_IDS_FILE) as f:
        return set(line.strip() for line in f if line.strip())


def save_seen_ids(ids):
    if not ids:
        return
    os.makedirs(os.path.dirname(SEEN_IDS_FILE), exist_ok=True)
    with open(SEEN_IDS_FILE, "a") as f:
        for article_id in sorted(ids):
            f.write(article_id + "\n")


def fetch_archived_urls(prefix):
    """Query the Wayback Machine CDX API for every archived URL starting with `prefix`."""
    params = {
        "url": prefix,
        "matchType": "prefix",
        "output": "text",
        "fl": "original",
        "collapse": "urlkey",
    }
    resp = requests.get(CDX_API, params=params, timeout=60)
    resp.raise_for_status()
    return [line.strip() for line in resp.text.splitlines() if line.strip()]


def dedupe_urls(urls):
    """
    Collapse every AMP / tracking-param / re-crawl variant of the same article down
    to one canonical URL, keyed on the numeric article id LiveMint embeds in the
    filename. URLs without that id are junk (mangled links, section pages) and
    are dropped. Returns {article_id: canonical_url}.
    """
    best = {}
    for url in urls:
        match = ARTICLE_ID_RE.search(url)
        if not match:
            continue
        article_id = match.group(1)
        is_amp = "/amp-" in url
        has_query = "?" in url
        score = (not is_amp, not has_query)  # prefer canonical non-amp, no-query form
        if article_id not in best or score > best[article_id][0]:
            best[article_id] = (score, url)
    return {article_id: url for article_id, (score, url) in best.items()}


def find_matching_company(name):
    """Match an extracted headline name against config.tickers.stocks (exact, then substring)."""
    name_l = name.strip().lower()
    for key, symbol in stocks.items():
        if name_l == key.lower():
            return key, symbol
    for key, symbol in stocks.items():
        key_l = key.lower()
        if name_l in key_l or key_l in name_l:
            return key, symbol
    return None, None


def scrape_article(url):
    """
    Return a list of {Company, News, Symbol, Date} dicts for one article, or None
    if the request itself failed (so the caller can retry it on the next run instead
    of permanently marking it as seen).
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"  [FAIL] {url}: {e}")
        return None

    soup = BeautifulSoup(resp.content, "html.parser")
    paragraphs = soup.find_all("div", class_="storyParagraph")
    if not paragraphs:
        return []

    try:
        pub_date = find_date(resp.text)
    except Exception:
        pub_date = None

    rows = []
    for p in paragraphs:
        text = p.get_text(separator=" ", strip=True)
        if ":" not in text:
            continue
        name, news = text.split(":", 1)
        name, news = name.strip(), news.strip()
        if len(news.split()) < MIN_NEWS_WORDS:
            continue
        key, symbol = find_matching_company(name)
        if not symbol:
            continue
        rows.append({"Company": key, "News": news, "Symbol": symbol, "Date": pub_date})
    return rows


def main():
    parser = argparse.ArgumentParser(description="Scrape LiveMint Stocks to Watch via Wayback Machine")
    parser.add_argument("--limit", type=int, default=None, help="Only scrape the first N new articles (for testing)")
    parser.add_argument("--rescan", action="store_true", help="Ignore the seen-ids cache and rescrape everything")
    args = parser.parse_args()

    print("Querying Wayback Machine CDX API...")
    all_urls = []
    for prefix in URL_PREFIXES:
        found = fetch_archived_urls(prefix)
        print(f"  {prefix}: {len(found)} raw entries")
        all_urls.extend(found)

    id_to_url = dedupe_urls(all_urls)
    print(f"Deduped {len(all_urls)} raw entries to {len(id_to_url)} unique articles")

    seen_ids = set() if args.rescan else load_seen_ids()
    new_items = [(aid, url) for aid, url in id_to_url.items() if aid not in seen_ids]
    print(f"{len(seen_ids)} articles already scraped previously, {len(new_items)} new to fetch")

    if args.limit:
        new_items = new_items[:args.limit]

    all_rows = []
    newly_seen = set()
    for i, (article_id, url) in enumerate(new_items, 1):
        print(f"[{i}/{len(new_items)}] {url}")
        rows = scrape_article(url)
        if rows is None:
            time.sleep(REQUEST_DELAY)
            continue
        newly_seen.add(article_id)
        print(f"    -> {len(rows)} stocks")
        all_rows.extend(rows)
        time.sleep(REQUEST_DELAY)

    save_seen_ids(newly_seen)

    if not all_rows:
        print("No new rows scraped.")
        return

    df_new = pd.DataFrame(all_rows, columns=["Company", "News", "Symbol", "Date"])
    before = len(df_new)
    df_new = df_new.drop_duplicates(subset=["Symbol", "News", "Date"])
    if len(df_new) < before:
        print(f"Dropped {before - len(df_new)} duplicate rows within this run")

    if os.path.exists(OUTPUT_FILE):
        df_existing = pd.read_csv(OUTPUT_FILE)
        existing_count = len(df_existing)
        combined = pd.concat([df_existing, df_new], ignore_index=True)
        df = combined.drop_duplicates(subset=["Symbol", "News", "Date"])
        skipped_existing = len(combined) - len(df)
        if skipped_existing:
            print(f"Skipped {skipped_existing} rows already present in {OUTPUT_FILE}")
    else:
        existing_count = 0
        df = df_new

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nAdded {len(df) - existing_count} rows, {len(df)} total -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()