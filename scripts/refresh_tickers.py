"""Rebuild config/tickers.py from NSE's own published equity list.

config/tickers.py is the universe every other stage of the pipeline walks
(fetch_technical_data.py fetches one CSV per symbol, scraper.py maps news
headlines back to symbols). It was hand-maintained, so it drifts in three
different ways at once and each one fails silently:

  * new listings never enter the universe at all, so they're never fetched;
  * delisted/merged companies stay in it forever, and every run wastes a
    Yahoo request on them before dumping them into failed_tickers.txt;
  * a *renamed* symbol looks like a delisting plus a brand-new listing, which
    is the expensive case -- the cached data/technical/<OLD>_data.csv is
    orphaned and the same company starts again from an empty history under
    its new symbol.

So this script doesn't just overwrite the list: it works out which of the
three happened for every difference, resolves renames from NSE's own symbol
change log (and, from the second run on, from ISIN identity), and can carry
the cached price history across a rename instead of throwing it away.

Sources (all NSE's own archives, no scraping):
  EQUITY_L.csv      -- the current universe: symbol, company name, series, ISIN
  symbolchange.csv  -- every symbol rename NSE has ever published (no header;
                       columns are company name, old symbol, new symbol, date)

Usage:
    python3 scripts/refresh_tickers.py                     # report + rewrite config
    python3 scripts/refresh_tickers.py --dry-run           # report only
    python3 scripts/refresh_tickers.py --rename-data-files # also carry cached CSVs across renames
"""

import os
import re
import csv
import sys
import time
import argparse
import datetime
import subprocess
import urllib.error
import urllib.request
import http.cookiejar
from io import StringIO

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TICKERS_FILE = os.path.join(REPO_ROOT, "config", "tickers.py")
# Full copy of the row NSE served for every symbol we kept. It exists purely so
# the *next* run can diff on ISIN: an ISIN is the company's permanent identity
# and survives both symbol and name changes, which makes it the only fully
# reliable rename signal. symbolchange.csv covers the first run (and anything
# older than this snapshot), but it lags -- NSE publishes the file on its own
# schedule, and a rename that isn't in it yet would otherwise be misread as a
# delisting-plus-new-listing and cost us that ticker's cached history.
SNAPSHOT_FILE = os.path.join(REPO_ROOT, "config", "nse_equity_snapshot.csv")
DATA_DIR = os.path.join(REPO_ROOT, "data", "technical")

EQUITY_LIST_URL = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
SYMBOL_CHANGE_URL = "https://nsearchives.nseindia.com/content/equities/symbolchange.csv"
# nseindia.com serves the archive files to browsers only; a bare urllib request
# gets a 403. Sending a browser UA is enough today, but the site has previously
# also required a session cookie, so fetch() falls back to warming one up on the
# main domain before retrying.
NSE_HOME = "https://www.nseindia.com"
BROWSER_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"),
    "Accept": "text/csv,application/csv,text/plain,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}

# Series NSE assigns to equity symbols: EQ is the normal rolling segment, BE is
# trade-for-trade (compulsory delivery, usually surveillance-flagged), BZ is the
# same but for companies in default on exchange compliance. The existing hand
# built universe contains all three (~1864 EQ / 197 BE / 20 BZ survive into the
# current list), so keeping all three is what preserves the current behaviour --
# --series is there for anyone who'd rather train only on the liquid EQ segment.
DEFAULT_SERIES = ("EQ", "BE", "BZ")

# Refuse to rewrite the universe from a fetch that came back implausibly small
# or that would delete an implausible share of it. A truncated/error-page
# response and a genuine mass-delisting are indistinguishable at the parse
# level, and the destructive one is far more likely, so both guards fail closed
# and ask for --force rather than quietly shrinking every downstream run's
# universe. The live list is ~2570 symbols; the largest single-day real drop is
# nowhere near 10%.
MIN_PLAUSIBLE_SYMBOLS = 1500
MAX_REMOVAL_FRACTION = 0.10


# --------------------------------------------------------------------------
# Fetching
# --------------------------------------------------------------------------

def fetch(url, attempts=3):
    """GET a URL as text, retrying with a warmed-up NSE session on 403."""
    opener = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
    last_error = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers=BROWSER_HEADERS)
            with opener.open(request, timeout=60) as response:
                return response.read().decode("utf-8-sig", errors="replace")
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code in (401, 403):
                # Pick up whatever cookie the landing page sets, then retry the
                # archive URL through the same opener.
                try:
                    opener.open(urllib.request.Request(NSE_HOME, headers=BROWSER_HEADERS),
                                timeout=60).read()
                except Exception:
                    pass
        except Exception as exc:
            last_error = exc
        if attempt < attempts - 1:
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed to fetch {url}: {last_error}")


def fetch_equity_list(series):
    """Return {symbol: row dict} for the requested series, from EQUITY_L.csv."""
    reader = csv.DictReader(StringIO(fetch(EQUITY_LIST_URL)))
    universe = {}
    for raw_row in reader:
        # NSE's header and several columns carry leading spaces ("
        # SERIES", " ISIN NUMBER"), so normalise both sides of every cell.
        row = {(key or "").strip(): (value or "").strip()
               for key, value in raw_row.items() if key}
        symbol, name = row.get("SYMBOL"), row.get("NAME OF COMPANY")
        if not symbol or not name:
            continue
        if row.get("SERIES") not in series:
            continue
        universe[symbol] = row
    return universe


def fetch_symbol_changes():
    """Return {old_symbol: [(date, new_symbol), ...]} from NSE's rename log.

    The file has no header row and is not deduplicated: a symbol that has been
    renamed more than once appears once per rename, so callers have to walk the
    chain (see resolve_rename) rather than treat it as a flat mapping.
    """
    changes = {}
    for row in csv.reader(StringIO(fetch(SYMBOL_CHANGE_URL))):
        if len(row) < 4:
            continue
        _company, old_symbol, new_symbol, date_text = (cell.strip() for cell in row[:4])
        if not old_symbol or not new_symbol:
            continue
        try:
            date = datetime.datetime.strptime(date_text, "%d-%b-%Y")
        except ValueError:
            continue
        changes.setdefault(old_symbol, []).append((date, new_symbol))
    return changes


# --------------------------------------------------------------------------
# Reading the current universe
# --------------------------------------------------------------------------

def load_current_universe():
    """Return {symbol: company name} as config/tickers.py has it today."""
    if not os.path.exists(TICKERS_FILE):
        return {}
    from config.tickers import stocks
    # tickers.py is keyed by company name; every consumer works symbol-first,
    # and the name side is exactly what NSE keeps editing, so invert it.
    return {symbol: name for name, symbol in stocks.items()}


def load_snapshot():
    """Return {isin: symbol} from the previous run's snapshot, if there is one."""
    if not os.path.exists(SNAPSHOT_FILE):
        return {}
    with open(SNAPSHOT_FILE, newline="", encoding="utf-8") as handle:
        return {row["ISIN NUMBER"]: row["SYMBOL"]
                for row in csv.DictReader(handle)
                if row.get("ISIN NUMBER") and row.get("SYMBOL")}


# --------------------------------------------------------------------------
# Classifying the difference
# --------------------------------------------------------------------------

def normalise_name(name):
    """Collapse a company name to a comparable core.

    NSE rewrites names cosmetically all the time ("and" -> "&", case flips,
    punctuation, suffix spelling), so raw name equality would report hundreds
    of fake changes and would drown the real renames. Dropping punctuation,
    the "and"/"&" alternation and the legal suffix leaves only the words that
    actually identify the company.
    """
    name = re.sub(r"[^a-z0-9]+", " ", name.lower())
    name = re.sub(r"\b(and|limited|ltd|the)\b", " ", name)
    return re.sub(r"\s+", " ", name).strip()


def resolve_rename(symbol, changes, live_symbols, max_hops=10):
    """Follow symbolchange.csv from a vanished symbol to a live one, if it leads there.

    Renames chain (A -> B -> C), and a symbol can appear as the source of more
    than one entry, so at each hop take the most recent rename and stop as soon
    as the chain lands on something in the current universe. Returning None
    means the chain ran out without reaching a live symbol -- i.e. the company
    really is gone (delisted, merged away), not renamed.
    """
    seen = {symbol}
    for _ in range(max_hops):
        if symbol not in changes:
            return None
        _date, symbol = max(changes[symbol], key=lambda entry: entry[0])
        if symbol in seen:
            return None
        seen.add(symbol)
        if symbol in live_symbols:
            return symbol
    return None


def diff_universes(current, live, changes, prior_isin_to_symbol):
    """Classify every difference between the current and live universes.

    Returns a dict of lists: renamed_symbols, renamed_companies, added,
    delisted, and possible_renames (name-only matches, reported for review but
    never applied -- two unrelated companies can share a normalised name).
    """
    live_symbols = set(live)
    vanished = sorted(set(current) - live_symbols)
    appeared = set(live_symbols) - set(current)

    # Pass 1: ISIN identity, from the previous run's snapshot. Exact when we
    # have it, because the ISIN doesn't change when the symbol does.
    renamed_symbols = {}
    for symbol, row in live.items():
        previous_symbol = prior_isin_to_symbol.get(row.get("ISIN NUMBER", ""))
        if previous_symbol and previous_symbol != symbol and previous_symbol in current:
            renamed_symbols[previous_symbol] = symbol

    # Pass 2: NSE's published rename log, for anything the snapshot couldn't
    # cover (notably the very first run, when there is no snapshot at all).
    for symbol in vanished:
        if symbol in renamed_symbols:
            continue
        new_symbol = resolve_rename(symbol, changes, live_symbols)
        if new_symbol:
            renamed_symbols[symbol] = new_symbol

    # Pass 3: name-only matches. Suggested, never applied -- a shared
    # normalised name is evidence, not proof, of the same company.
    live_by_name = {}
    for symbol, row in live.items():
        live_by_name.setdefault(normalise_name(row["NAME OF COMPANY"]), []).append(symbol)
    possible_renames = []
    for symbol in vanished:
        if symbol in renamed_symbols:
            continue
        candidates = [candidate for candidate in live_by_name.get(normalise_name(current[symbol]), [])
                      if candidate in appeared]
        if len(candidates) == 1:
            possible_renames.append((symbol, current[symbol], candidates[0]))

    renamed_targets = set(renamed_symbols.values())
    return {
        "renamed_symbols": sorted(renamed_symbols.items()),
        # A symbol that stayed put while its company name changed. Cosmetic
        # rewrites are filtered out by normalise_name; what's left is real
        # (rebrands, mergers that kept the listing).
        "renamed_companies": sorted(
            (symbol, current[symbol], live[symbol]["NAME OF COMPANY"])
            for symbol in set(current) & live_symbols
            if normalise_name(current[symbol]) != normalise_name(live[symbol]["NAME OF COMPANY"])),
        "added": sorted(symbol for symbol in appeared if symbol not in renamed_targets),
        "delisted": sorted(symbol for symbol in vanished if symbol not in renamed_symbols),
        "possible_renames": possible_renames,
    }


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------

def build_stocks_mapping(live):
    """Return [(company name, symbol)] for tickers.py, ordered by symbol.

    tickers.py is keyed by company *name*, but a handful of companies list two
    share classes under one name (the DVR pairs: FEL/FELDVR, GATECH/GATECHDVR,
    JISLJALEQS/JISLDVREQS). Writing those straight into a dict would silently
    drop one symbol from the universe, so collided names get their symbol
    appended -- scraper.py matches names by substring, so the suffix doesn't
    break headline lookups.
    """
    name_counts = {}
    for row in live.values():
        name_counts[row["NAME OF COMPANY"]] = name_counts.get(row["NAME OF COMPANY"], 0) + 1
    pairs = []
    for symbol in sorted(live):
        name = live[symbol]["NAME OF COMPANY"]
        pairs.append((f"{name} ({symbol})" if name_counts[name] > 1 else name, symbol))
    return pairs


def write_tickers_file(pairs, path=TICKERS_FILE):
    """Rewrite config/tickers.py, keeping the exact shape consumers import."""
    lines = ["stocks = {\n"]
    for name, symbol in pairs:
        # Company names are NSE-controlled free text and do contain quotes, so
        # escape rather than assume they're clean.
        escaped = name.replace("\\", "\\\\").replace('"', '\\"')
        lines.append(f'  "{escaped}": "{symbol}",\n')
    lines.append("}\n\n\n")
    lines.append("TICKERS = list(stocks.values())\n")
    lines.append("keys = list (stocks.keys ())\n")
    with open(path, "w", encoding="utf-8") as handle:
        handle.writelines(lines)


def write_snapshot(live, path=SNAPSHOT_FILE):
    columns = ["SYMBOL", "NAME OF COMPANY", "SERIES", "DATE OF LISTING", "ISIN NUMBER"]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for symbol in sorted(live):
            writer.writerow({column: live[symbol].get(column, "") for column in columns})


def rename_data_files(renames, dry_run):
    """Carry data/technical/<OLD>_data.csv across a confirmed symbol rename.

    Without this the rename costs the ticker its entire cached history: the old
    file is orphaned and fetch_technical_data.py starts the new symbol from an
    empty cache. Uses `git mv` where the file is tracked so the rename lands as
    a rename in history rather than a delete plus an add.
    """
    moved, skipped = [], []
    for old_symbol, new_symbol in renames:
        old_path = os.path.join(DATA_DIR, f"{old_symbol}_data.csv")
        new_path = os.path.join(DATA_DIR, f"{new_symbol}_data.csv")
        if not os.path.exists(old_path):
            continue
        if os.path.exists(new_path):
            # The new symbol already has its own cache; merging two price
            # histories is a judgement call, so leave both for a human.
            skipped.append((old_symbol, new_symbol, "target file already exists"))
            continue
        if dry_run:
            moved.append((old_symbol, new_symbol))
            continue
        try:
            subprocess.run(["git", "mv", old_path, new_path],
                           cwd=REPO_ROOT, check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            os.rename(old_path, new_path)  # untracked file, or no git available
        moved.append((old_symbol, new_symbol))
    return moved, skipped


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def build_report(diff, current, live, moved, skipped, orphans):
    out = []
    add = out.append
    add(f"Current universe : {len(current)} symbols")
    add(f"Live NSE list    : {len(live)} symbols")
    add("")
    add(f"Renamed symbols  : {len(diff['renamed_symbols'])}")
    for old_symbol, new_symbol in diff["renamed_symbols"]:
        add(f"    {old_symbol:14s} -> {new_symbol:14s} {live[new_symbol]['NAME OF COMPANY']}")
    add(f"New listings     : {len(diff['added'])}")
    for symbol in diff["added"]:
        add(f"    {symbol:14s} {live[symbol]['NAME OF COMPANY']} "
            f"(listed {live[symbol].get('DATE OF LISTING', '?')})")
    add(f"Delisted/merged  : {len(diff['delisted'])}  (removed from the universe; "
        f"their cached CSVs are left on disk)")
    for symbol in diff["delisted"]:
        add(f"    {symbol:14s} {current[symbol]}")
    add(f"Company renames  : {len(diff['renamed_companies'])}  (symbol unchanged)")
    for symbol, old_name, new_name in diff["renamed_companies"]:
        add(f"    {symbol:14s} {old_name}  ->  {new_name}")
    if diff["possible_renames"]:
        add(f"Possible renames : {len(diff['possible_renames'])}  (name match only -- "
            f"NOT applied, confirm before carrying data across)")
        for old_symbol, name, candidate in diff["possible_renames"]:
            add(f"    {old_symbol:14s} -> {candidate:14s} {name}")
    if moved:
        add(f"Data files moved : {len(moved)}")
        for old_symbol, new_symbol in moved:
            add(f"    {old_symbol}_data.csv -> {new_symbol}_data.csv")
    if skipped:
        add(f"Data files skipped: {len(skipped)}")
        for old_symbol, new_symbol, reason in skipped:
            add(f"    {old_symbol} -> {new_symbol}: {reason}")
    if orphans:
        add(f"Orphaned data files: {len(orphans)}  (symbol no longer in the universe; "
            f"nothing was deleted)")
        add("    " + ", ".join(orphans))
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(
        description="Rebuild config/tickers.py from NSE's published equity list.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would change without writing config/tickers.py, "
                             "the snapshot, or moving any data file.")
    parser.add_argument("--rename-data-files", action="store_true",
                        help="Also rename data/technical/<OLD>_data.csv to <NEW>_data.csv for "
                             "every confirmed rename, so the ticker keeps its cached history "
                             "instead of re-fetching from scratch under the new symbol.")
    parser.add_argument("--series", default=",".join(DEFAULT_SERIES),
                        help=f"Comma-separated NSE series to include "
                             f"(default: {','.join(DEFAULT_SERIES)}; use EQ for the liquid "
                             f"rolling segment only).")
    parser.add_argument("--force", action="store_true",
                        help="Write the new universe even if it trips the sanity guards "
                             f"(fewer than {MIN_PLAUSIBLE_SYMBOLS} symbols, or more than "
                             f"{MAX_REMOVAL_FRACTION:.0%} of the current universe removed) -- "
                             "those guards exist to stop a truncated download from gutting "
                             "the ticker list.")
    parser.add_argument("--report-file", default=None,
                        help="Also write the change report to this path.")
    args = parser.parse_args()

    series = {part.strip().upper() for part in args.series.split(",") if part.strip()}
    current = load_current_universe()

    print("Fetching NSE equity list ...")
    live = fetch_equity_list(series)
    print(f"  {len(live)} symbols in series {sorted(series)}")
    print("Fetching NSE symbol change log ...")
    try:
        changes = fetch_symbol_changes()
        print(f"  {len(changes)} renamed symbols on record")
    except RuntimeError as exc:
        # Renames then fall back to ISIN-only detection: degraded, not fatal.
        print(f"  WARNING: {exc}\n  Continuing without the rename log; renames may be "
              f"reported as delisting + new listing.")
        changes = {}

    diff = diff_universes(current, live, changes, load_snapshot())

    if current:
        removal_fraction = len(diff["delisted"]) / len(current)
        problems = []
        if len(live) < MIN_PLAUSIBLE_SYMBOLS:
            problems.append(f"only {len(live)} symbols fetched "
                            f"(expected at least {MIN_PLAUSIBLE_SYMBOLS})")
        if removal_fraction > MAX_REMOVAL_FRACTION:
            problems.append(f"{removal_fraction:.1%} of the current universe would be "
                            f"removed (limit {MAX_REMOVAL_FRACTION:.0%})")
        if problems and not args.force:
            print("\nRefusing to rewrite the ticker list:", file=sys.stderr)
            for problem in problems:
                print(f"  - {problem}", file=sys.stderr)
            print("  Re-run with --force if this is genuinely correct.", file=sys.stderr)
            return 1

    moved, skipped = [], []
    if args.rename_data_files and os.path.isdir(DATA_DIR):
        moved, skipped = rename_data_files(diff["renamed_symbols"], args.dry_run)

    orphans = []
    if os.path.isdir(DATA_DIR):
        cached = {name[:-len("_data.csv")] for name in os.listdir(DATA_DIR)
                  if name.endswith("_data.csv")}
        # NIFTY50 is fetch_technical_data.py's benchmark index, not an equity,
        # so it is never in EQUITY_L.csv and is not an orphan.
        orphans = sorted(cached - set(live) - {"NIFTY50"} - {old for old, _ in moved})

    report = build_report(diff, current, live, moved, skipped, orphans)
    print("\n" + report)

    if args.report_file:
        with open(args.report_file, "w", encoding="utf-8") as handle:
            handle.write(report + "\n")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return 0

    write_tickers_file(build_stocks_mapping(live))
    write_snapshot(live)
    print(f"\nWrote {os.path.relpath(TICKERS_FILE, REPO_ROOT)} ({len(live)} symbols) "
          f"and {os.path.relpath(SNAPSHOT_FILE, REPO_ROOT)}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
