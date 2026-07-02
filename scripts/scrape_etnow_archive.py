"""
Scrape ET Now "Stocks to Watch Today" articles.

Unlike moneycontrol.com, etnownews.com serves plain HTTP requests fine and
publishes its own monthly sitemap archive (back to November 2022), so this
talks to the site directly instead of going through the Wayback Machine.

Each stock mention is rendered as a run of <div class="...article-paragraph...">
elements, but the exact shape has drifted across the site's history: newer
articles use a Name / short headline / full-text triplet (plus a redundant
summary table earlier in the article that repeats the same names), older ones
use a plain Name / full-text pair, and some paragraphs glue the company name
directly onto its own text with no separator at all. Rather than assume one
fixed layout, extraction anchors on paragraphs that are short and match a real
ticker in config.tickers.stocks (or, for the glued case, long paragraphs that
*start with* a real ticker's short name) and treats everything up to the next
such anchor as that company's news -- this is robust to the layout drift and
to the redundant summary table, which mentions the same names again but
doesn't parse as a valid stand-alone entry.

Each scraped article id is recorded in a "seen ids" file, so re-running the
script only fetches articles published since the last run.

Usage:
    python scripts/scrape_etnow_archive.py
    python scripts/scrape_etnow_archive.py --limit 20   # test on a few articles
    python scripts/scrape_etnow_archive.py --rescan     # ignore the seen-ids cache
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
SITEMAP_INDEX = "https://www.etnownews.com/staticsitemap/etnow/sitemap-index.xml"
KEYWORD = "stocks-to-watch-today"

ARTICLE_ID_RE = re.compile(r"-article-(\d+)$")
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
OUTPUT_FILE = "data/news_sentiment/stocks_news_new.csv"
SEEN_IDS_FILE = "data/news_sentiment/etnow_scraped_ids.txt"
REQUEST_DELAY = 0.5  # seconds between article fetches
MIN_NEWS_WORDS = 5
NAME_MAX_WORDS = 6  # paragraphs this short are company-name/headline lines, not news text
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


def fetch_all_article_urls():
    """Crawl every monthly ET Now sitemap and collect 'stocks to watch today' article URLs."""
    resp = requests.get(SITEMAP_INDEX, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    index_soup = BeautifulSoup(resp.content, "xml")
    month_urls = [loc.get_text(strip=True) for loc in index_soup.find_all("loc")]

    articles = {}
    for month_url in month_urls:
        try:
            resp = requests.get(month_url, headers=HEADERS, timeout=30)
            if resp.status_code != 200:
                continue
            soup = BeautifulSoup(resp.content, "xml")
            for loc in soup.find_all("loc"):
                url = loc.get_text(strip=True)
                if KEYWORD not in url.lower():
                    continue
                match = ARTICLE_ID_RE.search(url)
                if not match:
                    continue
                articles[match.group(1)] = url
        except requests.exceptions.RequestException as e:
            print(f"  [FAIL] {month_url}: {e}")
    return articles


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


# Articles refer to companies by informal short names ("HDFC Bank") while
# config.tickers.stocks keys are formal legal names ("HDFC Bank Limited"), so
# glued-paragraph matching needs the formal name's corporate suffix stripped first.
_CORP_SUFFIXES = {"limited", "ltd", "ltd.", "inc", "incorporated", "corporation", "corp", "plc", "co"}


def _strip_corp_suffix(name):
    """Strip trailing corporate-suffix words and parenthetical qualifiers, e.g.
    'NBCC (India) Limited' -> 'NBCC'. Loops since either can trail the other."""
    prev = None
    while prev != name:
        prev = name
        name = re.sub(r"\s*\([^()]*\)\s*$", "", name).strip()
        words = name.split()
        while words and words[-1].lower().strip(".") in _CORP_SUFFIXES:
            words = words[:-1]
        name = " ".join(words)
    return name


_COMPANY_SHORT_KEYS = sorted(
    ((_strip_corp_suffix(key), key) for key in stocks.keys()),
    key=lambda pair: len(pair[0]), reverse=True,
)


def find_leading_company(text):
    """
    Detect 'CompanyName' glued directly onto its own news text with no separator.
    The remainder only needs to be non-empty (not a full MIN_NEWS_WORDS check): this
    also matches short 'NameHeadline' fragments where the real news text is a
    separate paragraph that follows and gets joined on afterward.
    """
    text_l = text.lower()
    for short_name, full_key in _COMPANY_SHORT_KEYS:
        if short_name and text_l.startswith(short_name.lower()):
            rest = text[len(short_name):].strip()
            if rest:
                return full_key, rest
    return None


def extract_stocks(paragraphs):
    """
    Find every paragraph that anchors a real company (a short paragraph matching a
    ticker outright, or a long paragraph starting with one glued to its text) and
    collect everything up to the next such anchor as that company's news. Adjacent
    anchors for the same company (the short name-only paragraph immediately followed
    by its own glued full-text paragraph) are merged into a single entry.
    """
    raw_anchors = []  # (paragraph_index, company_key, symbol, glued_remainder_or_None)
    for i, text in enumerate(paragraphs):
        text = text.strip()
        if not text:
            continue
        words = text.split()
        if len(words) <= NAME_MAX_WORDS:
            key, symbol = find_matching_company(text)
            if symbol:
                raw_anchors.append((i, key, symbol, None))
                continue
        # Try the glued-prefix match regardless of overall length: a short
        # "NameHeadline" fragment (no separate news text yet) needs this too,
        # not just long paragraphs where name and full text are glued together.
        combined = find_leading_company(text)
        if combined:
            key, symbol = find_matching_company(combined[0])
            if symbol:
                raw_anchors.append((i, key, symbol, combined[1]))

    if not raw_anchors:
        return []

    merged = []
    for anchor in raw_anchors:
        if merged and merged[-1][2] == anchor[2]:
            continue  # same company as the previous anchor -- still the same entry
        merged.append(anchor)

    results = []
    for idx, (i, key, symbol, glued_rest) in enumerate(merged):
        next_i = merged[idx + 1][0] if idx + 1 < len(merged) else len(paragraphs)
        parts = [glued_rest] if glued_rest else []
        for j in range(i + 1, next_i):
            t = paragraphs[j].strip()
            if t:
                parts.append(t)
        news = " ".join(parts).strip()
        if len(news.split()) >= MIN_NEWS_WORDS:
            results.append((key, symbol, news))
    return results


def scrape_article(url):
    """
    Return a list of {Company, News, Symbol, Date} dicts for one article, or None
    if the request itself failed (so the caller can retry it on the next run).
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"  [FAIL] {url}: {e}")
        return None

    soup = BeautifulSoup(resp.content, "html.parser")
    paragraphs = [p.get_text(strip=True) for p in soup.find_all("div", class_=lambda c: c and "article-paragraph" in c)]

    try:
        pub_date = find_date(resp.text)
    except Exception:
        pub_date = None

    rows = []
    for key, symbol, news in extract_stocks(paragraphs):
        rows.append({"Company": key, "News": news, "Symbol": symbol, "Date": pub_date})
    return rows


def main():
    parser = argparse.ArgumentParser(description="Scrape ET Now Stocks to Watch Today articles")
    parser.add_argument("--limit", type=int, default=None, help="Only scrape the first N new articles (for testing)")
    parser.add_argument("--rescan", action="store_true", help="Ignore the seen-ids cache and rescrape everything")
    args = parser.parse_args()

    print("Crawling ET Now sitemaps...")
    articles = fetch_all_article_urls()
    print(f"Found {len(articles)} unique 'stocks to watch today' article URLs")

    seen_ids = set() if args.rescan else load_seen_ids()
    new_items = [(aid, url) for aid, url in articles.items() if aid not in seen_ids]
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
            continue  # request failed, retry next run
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
