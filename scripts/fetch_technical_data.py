import os
import sys
import argparse
import numpy as np
import pandas as pd
import yfinance as yf
import talib as ta
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.tickers import TICKERS  

DATA_DIR = "data/technical"
os.makedirs(DATA_DIR, exist_ok=True)

# A gap this long between two consecutive *real* (post zero-volume-strip)
# trading days means the stock was effectively halted/dormant for an extended
# stretch (long suspension, corporate action, illiquidity). Treating the rows
# on either side as adjacent would blend two unrelated price regimes into one
# rolling-window indicator value and would also produce a fake giant "return"
# across the gap in downstream forward-window calculations. Segmenting resets
# indicator lookback (and lets signal_scanner.py mask forward windows) at
# each such gap, so every indicator/return is computed only within a single
# unbroken run of real trading days.
GAP_THRESHOLD_DAYS = 45

# How many percentage points of mismatch between an old cached Close and a
# freshly re-fetched Close (for the *same* historical date) counts as
# evidence that a split/bonus/rights issue has happened since the ticker was
# last updated. Splits and bonuses are applied *retroactively* by Yahoo --
# every already-cached row silently goes stale the moment a new corporate
# action occurs, even though nothing about those rows "looks" wrong on their
# own. Ordinary cash dividends also nudge Adj-Close-based series by a small,
# continuous amount (typically well under 1% per dividend); the threshold is
# set high enough to ignore that routine drift and only fire on the kind of
# large, discontinuous jump a split/bonus/rights issue produces (Best
# Agrolife's Jan-2026 10:1 split + 1:2 bonus was ~15x; Cupid's 1:10 split +
# 1:1 bonus was ~20x) -- not the ~0.1-1% nudges of a normal dividend.
ADJUSTMENT_TOLERANCE = 0.03
# Trading days at the tail of the existing cache to re-fetch and compare
# against on every run, purely to check whether they're still on the same
# adjustment basis they were saved under.
OVERLAP_CHECK_DAYS = 10

# Start and end dates
START_DATE = "2021-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")
# yf.download's `end` is exclusive, so requesting end=END_DATE would never
# actually fetch today's session -- push the request window one day further.
FETCH_END = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")

def add_indicators(df):
    close = df["Close"].astype(float).values
    high = df["High"].astype(float).values
    low = df["Low"].astype(float).values
    volume = df["Volume"].astype(float).values

    # Moving Averages
    df["SMA_20"] = ta.SMA(close, timeperiod=20)
    df["SMA_50"] = ta.SMA(close, timeperiod=50)
    df["EMA_20"] = ta.EMA(close, timeperiod=20)
    df["EMA_50"] = ta.EMA(close, timeperiod=50)
    df["EMA_200"] = ta.EMA(close, timeperiod=200)

    # RSI
    df["RSI_14"] = ta.RSI(close, timeperiod=14)

    # MACD
    macd, macdsignal, macdhist = ta.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    df["MACD"] = macd
    df["MACD_Signal"] = macdsignal
    df["MACD_Hist"] = macdhist

    # Bollinger Bands
    upper, middle, lower = ta.BBANDS(close, timeperiod=20)
    df["BB_upper"] = upper
    df["BB_middle"] = middle
    df["BB_lower"] = lower

    # ATR
    df["ATR_14"] = ta.ATR(high, low, close, timeperiod=14)

    # Stochastic Oscillator
    slowk, slowd = ta.STOCH(high, low, close)
    df["STOCH_K"] = slowk
    df["STOCH_D"] = slowd

    # OBV
    df["OBV"] = ta.OBV(close, volume)

    # CCI
    df["CCI_20"] = ta.CCI(high, low, close, timeperiod=20)

    # Williams %R
    df["Williams_%R"] = ta.WILLR(high, low, close, timeperiod=14)

    # VWAP
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    df["VWAP"] = (typical_price * df["Volume"]).cumsum() / df["Volume"].cumsum()

    # Chaikin Money Flow
    mf_multiplier = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / (df["High"] - df["Low"])
    mf_volume = mf_multiplier * df["Volume"]
    df["CMF_20"] = mf_volume.rolling(20).sum() / df["Volume"].rolling(20).sum()

    return df


def assign_segments(df):
    """Number each row by which unbroken run of real trading days it belongs
    to -- increments every time the gap since the previous real trade exceeds
    GAP_THRESHOLD_DAYS. Must run after clean_raw() so gaps reflect real
    trading days only, not zero-volume stale-quote rows."""
    df = df.copy()
    if df.empty:
        df["Segment"] = pd.Series(dtype=int)
        return df
    gap_days = df["Date"].diff().dt.days.fillna(0)
    df["Segment"] = (gap_days > GAP_THRESHOLD_DAYS).cumsum()
    return df


def add_indicators_by_segment(df):
    """Apply add_indicators() separately within each segment, so a rolling
    window (e.g. EMA_200, ATR_14) never blends prices from before and after a
    long trading gap. Segments too short for a given indicator's lookback
    simply get NaN there, same as at the start of any fresh history."""
    pieces = [add_indicators(seg.reset_index(drop=True)) for _, seg in df.groupby("Segment", sort=True)]
    return pd.concat(pieces, ignore_index=True)


def add_fundamentals(ticker, df):
    try:
        info = yf.Ticker(ticker).info
        fundamentals = {
            "MarketCap": info.get("marketCap", None),
            "PE": info.get("trailingPE", None),
            "EPS": info.get("trailingEps", None),
            "PB": info.get("priceToBook", None),
            "DividendYield": info.get("dividendYield", None)
        }
        for key, value in fundamentals.items():
            df[key] = value
        return df
    except Exception as e:
        print(f"Failed to fetch fundamentals for {ticker}: {e}")
        return df

# NOTE: no "Adj Close" column anymore. Downloads now use auto_adjust=True (see
# _download() below), so the "Close" column returned by yfinance already *is*
# the split/dividend-adjusted price -- a separate Adj Close column would just
# be a duplicate of Close, not a different, more-correct series to fall back
# on. See ADJUSTMENT_TOLERANCE / _detect_retroactive_adjustment for how stale
# (pre-existing corporate-action) cached rows are handled.
RAW_COLS = ["Date", "Open", "High", "Low", "Close", "Volume"]


def clean_raw(df):
    """Drop rows with zero (or missing) volume -- these are not real trading
    days. When a stock isn't actually trading, Yahoo Finance carries the last
    real price forward as a stale quote instead of omitting the row, so the
    next genuine trade after such a gap looks like an extreme single-day
    price move that never happened (e.g. a stock frozen at a stale price for
    months, then "jumping" hundreds of percent the day real trading resumes).
    Dropping these rows means every remaining row is a genuine trading day,
    so returns/indicators are computed only across real price history."""
    if df.empty:
        return df
    return df[df["Volume"] > 0].reset_index(drop=True)


def _download(yf_ticker, start, end):
    """Thin wrapper around yf.download that always requests split/dividend-
    adjusted prices and normalizes the result to RAW_COLS.

    auto_adjust=True (rather than the previous auto_adjust=False +
    a separate "Adj Close" column) means Open/High/Low/Close all come back
    already rescaled for every split and bonus issue Yahoo knows about, as of
    the moment of the request -- not just Close, and not left for downstream
    code to reconcile itself. This is what actually fixes the corporate-action
    bug: previously Close carried the raw, un-rescaled price, so a stock like
    BESTAGRO (10:1 split + 1:2 bonus, effectively ~15x, Jan 2026) or CUPID
    (1:10 split + 1:1 bonus, ~20x, Apr 2024) would show a fake multi-hundred-
    percent single-day "return" on its record date, corrupting every
    indicator and forward-return window that touched it.
    """
    fetched = yf.download(
        yf_ticker,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    if fetched.empty:
        return pd.DataFrame(columns=RAW_COLS)
    if isinstance(fetched.columns, pd.MultiIndex):
        fetched.columns = [c[0] for c in fetched.columns]
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        fetched[col] = pd.to_numeric(fetched[col], errors="coerce")
    fetched = fetched.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    if fetched.empty:
        return pd.DataFrame(columns=RAW_COLS)
    fetched = fetched.reset_index()
    return clean_raw(fetched[RAW_COLS])


def _detect_retroactive_adjustment(yf_ticker, old_df):
    """Re-fetch the most recent OVERLAP_CHECK_DAYS of already-cached trading
    days and compare their (adjusted) Close against what's on disk.

    Splits/bonuses/rights issues are applied retroactively: an adjusted Close
    from six months ago is a different number today than it was the day
    before a new split, even though nothing about that row's Date/Open/High/
    Low/Close individually looks wrong. A plain incremental "fetch only what's
    missing and append" scheme (the previous behaviour) has no way to notice
    this -- it will happily keep appending correctly-adjusted new rows onto a
    stale-basis history forever. Returns True if a large, split/bonus-sized
    discrepancy is found (see ADJUSTMENT_TOLERANCE for what counts as
    "large" vs. routine dividend drift), meaning the cached history for this
    ticker can no longer be trusted and needs a full re-fetch rather than an
    incremental merge.
    """
    if old_df is None or old_df.empty:
        return False
    recent = old_df.sort_values("Date").tail(OVERLAP_CHECK_DAYS)
    if recent.empty:
        return False
    start = recent["Date"].min().strftime("%Y-%m-%d")
    end = (recent["Date"].max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    check = _download(yf_ticker, start, end)
    if check.empty:
        return False
    merged = recent.merge(check[["Date", "Close"]], on="Date", suffixes=("_old", "_new"))
    if merged.empty:
        return False
    ratio = (merged["Close_new"] / merged["Close_old"]).replace([np.inf, -np.inf], np.nan).dropna()
    if ratio.empty:
        return False
    return bool((ratio - 1.0).abs().max() > ADJUSTMENT_TOLERANCE)


def update_ticker(ticker, force_clean=False):
    """Fetch and merge only the missing days for one ticker, then recompute
    indicators over the full price history. Returns (ticker, status_string,
    error_or_None) -- returned as a tuple (rather than printed directly)
    because this function runs inside a worker process under
    ProcessPoolExecutor, and worker stdout doesn't reliably interleave with
    the parent process's prints.

    force_clean=True now forces a full re-fetch from START_DATE (not just a
    local recompute over the existing cached raw prices). This is stronger
    than it used to be, and deliberately so: _detect_retroactive_adjustment
    (below) only compares the most recent OVERLAP_CHECK_DAYS of cached data
    against a fresh fetch, which catches a split/bonus that happens *after*
    a ticker was last updated -- but it can't see one that's already sitting,
    undetected, in the *middle* of a file that was built incrementally under
    a version of this script that didn't adjust for corporate actions at all
    (which is exactly how BESTAGRO's and CUPID's caches were likely built).
    The only way to guarantee those already-baked-in discontinuities get
    caught and corrected is a genuine full re-fetch, which is what --clean
    now does. Run it once across the whole universe after adopting this fix;
    the lightweight tail-check below is what keeps things correct on an
    ongoing, incremental basis after that.
    """
    file_path = os.path.join(DATA_DIR, f"{ticker}_data.csv")

    old_df = None
    removed = 0
    fetch_start = START_DATE
    if os.path.exists(file_path):
        old_df = pd.read_csv(file_path, parse_dates=["Date"])
        if not old_df.empty:
            orig_len = len(old_df)
            old_df = clean_raw(old_df)
            removed = orig_len - len(old_df)

    yf_ticker = f"{ticker}.NS" if not ticker.endswith((".NS", ".BO")) else ticker

    try:
        rebuilt_for_adjustment = False
        if force_clean:
            # --clean means "don't trust the cache at all" -- skip the
            # tail-only detection check (it would just cost a network call to
            # confirm what we're about to do unconditionally anyway) and go
            # straight to a full rebuild.
            if old_df is not None and not old_df.empty:
                rebuilt_for_adjustment = True
                old_df = None
                removed = 0
        elif old_df is not None and not old_df.empty:
            if _detect_retroactive_adjustment(yf_ticker, old_df):
                rebuilt_for_adjustment = True
                old_df = None
                removed = 0

        if old_df is not None and not old_df.empty:
            last_date = old_df["Date"].max()
            fetch_start = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        has_new_data = fetch_start <= END_DATE
        if not has_new_data and not force_clean and not rebuilt_for_adjustment:
            return ticker, "already up to date", None

        new_df = pd.DataFrame(columns=RAW_COLS)
        if has_new_data or rebuilt_for_adjustment:
            new_df = _download(yf_ticker, fetch_start, FETCH_END)

        # Merge with the existing raw price history (not the old indicator
        # columns) before recomputing indicators, since things like EMA_200
        # need months of trailing lookback that the freshly-downloaded rows
        # alone don't have.
        if old_df is not None and not old_df.empty:
            old_raw = old_df[RAW_COLS]
            combined_raw = (
                pd.concat([old_raw, new_df], ignore_index=True)
                .drop_duplicates(subset=["Date"])
                .sort_values("Date")
                .reset_index(drop=True)
            )
        elif not new_df.empty:
            combined_raw = new_df.sort_values("Date").reset_index(drop=True)
        else:
            return ticker, "no valid data", None

        combined_raw = assign_segments(combined_raw)
        combined = add_indicators_by_segment(combined_raw)

        fundamental_cols = ["MarketCap", "PE", "EPS", "PB", "DividendYield"]
        if has_new_data or rebuilt_for_adjustment:
            combined = add_fundamentals(yf_ticker, combined)
        elif old_df is not None and all(c in old_df.columns for c in fundamental_cols):
            # No new rows fetched, so nothing about the company changed --
            # reuse the already-fetched fundamentals instead of hitting the
            # network again for every ticker on a recompute-only pass.
            for col in fundamental_cols:
                combined[col] = old_df[col].iloc[-1]
        else:
            combined = add_fundamentals(yf_ticker, combined)
        combined.to_csv(file_path, index=False)

        n_segments = combined["Segment"].nunique()

        parts = []
        if rebuilt_for_adjustment:
            parts.append("rebuilt: full re-fetch (--clean or detected split/bonus)")
        if removed:
            parts.append(f"removed {removed} zero-volume rows")
        if len(new_df):
            parts.append(f"{len(new_df)} new rows")
        if n_segments > 1:
            parts.append(f"{n_segments} segments (gap>{GAP_THRESHOLD_DAYS}d)")
        parts.append(f"{len(combined)} total")
        return ticker, ", ".join(parts), None

    except Exception as e:
        return ticker, None, str(e)


def main():
    parser = argparse.ArgumentParser(description="Fetch/update NSE technical data")
    parser.add_argument("--clean", action="store_true",
                         help="Force a full re-fetch (not just a local recompute) for every ticker, "
                              "even ones with no new data -- run this once after adopting the "
                              "corporate-action fix to catch splits/bonuses already baked into old caches")
    parser.add_argument("--workers", type=int, default=None,
                         help="Parallel worker processes (default: min(CPU count, 8), since this is a "
                              "network-bound task and too many workers just hits Yahoo's rate limits)")
    parser.add_argument("--ticker", type=str, default=None,
                         help="Update a single ticker only (skips parallelization)")
    args = parser.parse_args()

    failed_tickers = []

    if args.ticker:
        ticker, status, err = update_ticker(args.ticker, force_clean=args.clean)
        if err:
            print(f"Error for {ticker}: {err}")
        else:
            print(f"  {ticker}: {status}")
        return

    n_workers = args.workers or min(os.cpu_count() or 1, 8)
    print(f"Updating {len(TICKERS)} tickers across {n_workers} worker processes...\n")

    done = 0
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(update_ticker, t, args.clean): t for t in TICKERS}
        for future in as_completed(futures):
            ticker = futures[future]
            done += 1
            try:
                t, status, err = future.result()
            except Exception as e:
                # Crash inside the worker process itself (not caught by the
                # try/except in update_ticker) -- still record it and move on
                # rather than letting one bad ticker kill the whole run.
                print(f"[{done}/{len(TICKERS)}] {ticker}: WORKER CRASH: {e}")
                failed_tickers.append(ticker)
                continue

            if err:
                print(f"[{done}/{len(TICKERS)}] {t}: ERROR: {err}")
                failed_tickers.append(t)
            else:
                print(f"[{done}/{len(TICKERS)}] {t}: {status}")

    print("\nFailed Tickers:", failed_tickers)


if __name__ == "__main__":
    main()