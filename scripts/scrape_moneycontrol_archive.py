"""
Scrape Moneycontrol "Stocks to Watch" articles via the Wayback Machine.

moneycontrol.com now blocks plain HTTP requests site-wide (Akamai WAF, 403 on the
sitemap, the homepage, and article pages alike), so this goes through the Internet
Archive instead of moneycontrol.com directly: the Wayback Machine's CDX API finds
every archived article URL under the "stocks to watch" path (no rate limiting, full
historical coverage), and each article's content is then read from its cached
snapshot rather than from the live (blocked) page.

Within an article, the "Stocks to Watch" section is a marker paragraph followed by
alternating short paragraphs (company name) and long paragraphs (that company's
news, sometimes split across more than one paragraph) — this replaces the older
<p><strong>Name</strong> text</p> assumption that no longer matches the site's
current markup. The marker is searched for across every <p> on the page rather
than inside a specific container id, because that id is reused for an unrelated
stock-quote widget on some page variants and would otherwise silently match the
wrong element.

Some Wayback captures of an article are incomplete (the page was snapshotted
before its JS-rendered body finished loading), so each article keeps a few of its
most recent captures as fallback candidates and tries them in order until one
actually contains the "Stocks to Watch" section.

Each scraped article id is recorded in a "seen ids" file, so re-running the script
only fetches articles published since the last run.

Usage:
    python scripts/scrape_moneycontrol_archive.py
    python scripts/scrape_moneycontrol_archive.py --limit 20   # test on a few articles
    python scripts/scrape_moneycontrol_archive.py --rescan     # ignore the seen-ids cache
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
URL_PREFIX = "https://www.moneycontrol.com/news/business/markets/stocks-to-watch-today"

ARTICLE_ID_RE = re.compile(r"-(\d+)\.html")
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
OUTPUT_FILE = "data/news_sentiment/stocks_news.csv"
SEEN_IDS_FILE = "data/news_sentiment/moneycontrol_scraped_ids.txt"
REQUEST_DELAY = 0.5  # seconds between article fetches
MIN_NEWS_WORDS = 5
NAME_MAX_WORDS = 6  # paragraphs this short (within the section) are company-name headers, not news
MAX_SNAPSHOT_CANDIDATES = 3  # fallback captures to try per article if the newest one is incomplete

SECTION_MARKER = "stocks to watch"
# Other section headers moneycontrol groups on the same page; hitting one of these
# after the Stocks to Watch marker means that section has ended.
STOP_SECTION_TITLES = {
    "results today", "quarterly earnings", "quarterly business update",
    "bulk deals", "block deals", "ipo corner", "mainboard listing", "sme listing",
    "stocks in fn&o ban", "stocks in f&o ban",
}
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


JUNK_URL_MARKERS = ("%0a", "%0d", "%7c", "download")


def is_valid_candidate_url(url):
    """Reject captures of mangled URLs: embedded newlines, or two URLs concatenated together."""
    low = url.lower()
    if any(marker in low for marker in JUNK_URL_MARKERS):
        return False
    if low.count("http") > 1:  # a second scheme means another URL got appended
        return False
    return True


def candidate_quality(url):
    """Higher is better: prefer the canonical desktop URL with no query string."""
    is_amp = "/amp" in url.lower()
    has_query = "?" in url
    return (not is_amp, not has_query)


def fetch_archived_articles():
    """
    Query the Wayback Machine CDX API for every archived "stocks to watch" article,
    keyed by the numeric article id moneycontrol embeds in the filename. Returns
    {article_id: [(timestamp, original_url), ...]}, best/cleanest capture first,
    capped at MAX_SNAPSHOT_CANDIDATES per article so an incomplete capture has
    fallbacks to try. Junk captures (mangled/concatenated URLs) tend to have been
    crawled *after* the real article, so sorting by recency alone would keep
    picking them over the clean original -- candidates are ranked by URL quality
    first and recency second instead.
    """
    params = {
        "url": URL_PREFIX,
        "matchType": "prefix",
        "output": "text",
        "fl": "timestamp,original",
    }
    resp = requests.get(CDX_API, params=params, timeout=60)
    resp.raise_for_status()

    grouped = {}
    for line in resp.text.splitlines():
        line = line.strip()
        if not line:
            continue
        timestamp, original = line.split(" ", 1)
        if not is_valid_candidate_url(original):
            continue
        match = ARTICLE_ID_RE.search(original)
        if not match:
            continue  # no article id: junk / non-article URL
        article_id = match.group(1)
        grouped.setdefault(article_id, []).append((timestamp, original))

    for article_id, candidates in grouped.items():
        candidates.sort(key=lambda c: (candidate_quality(c[1]), c[0]), reverse=True)
        grouped[article_id] = candidates[:MAX_SNAPSHOT_CANDIDATES]
    return grouped


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


# Articles refer to companies by their informal short name ("Pfizer", "Adani
# Enterprises") while config.tickers.stocks keys are formal legal names ("Pfizer
# Limited"), so matching needs the formal name's corporate suffix stripped first.
_CORP_SUFFIXES = {"limited", "ltd", "ltd.", "inc", "incorporated", "corporation", "corp", "plc", "co"}


def _strip_corp_suffix(name):
    words = name.split()
    while words and words[-1].lower().strip(".") in _CORP_SUFFIXES:
        words = words[:-1]
    return " ".join(words)


# Longest short-name first, so "Adani Enterprises" is tried before a shorter unrelated prefix match.
_COMPANY_SHORT_KEYS = sorted(
    ((_strip_corp_suffix(key), key) for key in stocks.keys()),
    key=lambda pair: len(pair[0]), reverse=True,
)


def find_leading_company(text):
    """
    Detect 'CompanyName' followed immediately by news text within a single paragraph,
    with or without a separating colon (moneycontrol uses both styles depending on
    the article's template). Matches against companies' informal short names, since
    that's what these blurbs use rather than the full legal name.
    """
    text_l = text.lower()
    for short_name, full_key in _COMPANY_SHORT_KEYS:
        if short_name and text_l.startswith(short_name.lower()):
            rest = text[len(short_name):].lstrip(": ").strip()
            if len(rest.split()) >= MIN_NEWS_WORDS:
                return full_key, rest
    return None


def extract_stocks_to_watch(soup):
    """
    Find the "Stocks to Watch" marker (it isn't always wrapped in a <p>, so this
    searches raw text nodes rather than <p> tags) and walk every <p> that follows it
    in document order, stopping at the next section or the end of the document.

    Two paragraph layouts are both in use across moneycontrol's articles: a company
    name in its own short paragraph followed by a separate (possibly multi-paragraph)
    news paragraph, or a company name and its news combined into one paragraph. Both
    are handled here.
    """
    # Match "Stocks to Watch" and short variants like "Stocks to Watch Today", but not
    # the article's title/description text which repeats the same phrase in a full
    # sentence -- those are always much longer than the standalone section divider.
    def is_marker(s):
        text = s.strip().lower()
        return text.startswith(SECTION_MARKER) and len(text.split()) <= 6

    marker = soup.find(string=is_marker)
    if marker is None:
        return []
    paragraphs = [p.get_text(strip=True) for p in marker.find_all_next("p")]

    results = []
    i = 0
    n = len(paragraphs)
    while i < n:
        text = paragraphs[i].strip()
        if not text:
            i += 1
            continue
        if text.lower() in STOP_SECTION_TITLES or text.lower().startswith(("disclosure", "disclaimer")):
            break

        if len(text.split()) > NAME_MAX_WORDS:
            combined = find_leading_company(text)
            if combined:
                results.append(combined)
            i += 1
            continue

        name = text
        i += 1
        news_parts = []
        while i < n and paragraphs[i].strip() and len(paragraphs[i].strip().split()) > NAME_MAX_WORDS:
            news_parts.append(paragraphs[i].strip())
            i += 1
        news = " ".join(news_parts).strip()
        if news:
            results.append((name, news))
    return results


def scrape_article(candidates):
    """
    Try each (timestamp, original_url) snapshot candidate for one article, newest
    first, until one actually contains the "Stocks to Watch" section (some captures
    are incomplete because the page's JS-rendered body hadn't loaded when crawled).

    Returns a list of {Company, News, Symbol, Date} dicts (possibly empty, if every
    candidate loaded fine but genuinely had no matching section/tickers), or None if
    every candidate failed at the request level (so the caller can retry next run).
    """
    any_success = False
    for timestamp, original_url in candidates:
        snapshot_url = f"https://web.archive.org/web/{timestamp}id_/{original_url}"
        try:
            resp = requests.get(snapshot_url, headers=HEADERS, timeout=20)
            resp.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"  [FAIL] {original_url} @ {timestamp}: {e}")
            continue
        any_success = True

        soup = BeautifulSoup(resp.content, "html.parser")
        pairs = extract_stocks_to_watch(soup)
        if not pairs:
            continue  # incomplete/irrelevant capture, try the next candidate

        try:
            pub_date = find_date(resp.text)
        except Exception:
            pub_date = None

        rows = []
        for name, news in pairs:
            if len(news.split()) < MIN_NEWS_WORDS:
                continue
            key, symbol = find_matching_company(name)
            if not symbol:
                continue
            rows.append({"Company": key, "News": news, "Symbol": symbol, "Date": pub_date})
        return rows

    return [] if any_success else None


def main():
    parser = argparse.ArgumentParser(description="Scrape Moneycontrol Stocks to Watch via Wayback Machine")
    parser.add_argument("--limit", type=int, default=None, help="Only scrape the first N new articles (for testing)")
    parser.add_argument("--rescan", action="store_true", help="Ignore the seen-ids cache and rescrape everything")
    args = parser.parse_args()

    print("Querying Wayback Machine CDX API...")
    articles = fetch_archived_articles()
    print(f"Found {len(articles)} unique archived articles")

    seen_ids = set() if args.rescan else load_seen_ids()
    new_items = [(aid, candidates) for aid, candidates in articles.items() if aid not in seen_ids]
    print(f"{len(seen_ids)} articles already scraped previously, {len(new_items)} new to fetch")

    if args.limit:
        new_items = new_items[:args.limit]

    all_rows = []
    newly_seen = set()
    for i, (article_id, candidates) in enumerate(new_items, 1):
        print(f"[{i}/{len(new_items)}] {candidates[0][1]}")
        rows = scrape_article(candidates)
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
        df_existing = pd.read_csv(OUTPUT_FILE, quoting=1, on_bad_lines="skip")
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
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig", quoting=1)
    print(f"\nAdded {len(df) - existing_count} rows, {len(df)} total -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
