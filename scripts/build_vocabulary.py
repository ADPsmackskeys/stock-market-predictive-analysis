"""
Distil a point-in-time signal_scanner.py scan into the FIXED SIGNAL VOCABULARY
for the walk-forward backtest.

WHY THIS EXISTS
---------------
The walk-forward study refits signal statistics at every rebalance date, but it
must NOT re-run combo discovery at every one of them -- the exhaustive pair
search (itertools.combinations over ~97 atomic signals, per direction, per
symbol) is the single most expensive stage in signal_scanner.py, and paying it
33 times is what makes the naive approach a multi-day job.

The way out is to separate the two questions:

  1. WHICH signals are worth watching for this symbol?  <- answered ONCE, here,
     from data strictly before the backtest starts.
  2. HOW GOOD is each of them, as of date D?            <- refit per window,
     cheaply, over the fixed set this file produces.

Fixing the vocabulary on pre-backtest data keeps (1) honest: a combo that only
became discoverable because of 2024-2026 price action never enters the set, so
the backtest cannot select a signal it could not have known to watch. Refitting
(2) per window keeps the statistics honest. Both halves are needed -- a fixed
vocabulary with full-history statistics is still a leaky backtest, and so is a
rediscovered vocabulary with as-of statistics.

WHAT IS AND IS NOT CARRIED FORWARD
----------------------------------
Only signal IDENTITY survives into the vocabulary: Symbol, SignalName,
Components, Type, Direction, HorizonClass. Every fitted number --
CompositeScore, BestHorizon, AvgEV, the Target*_BestHorizon quantile ladder,
all the per-horizon columns -- is deliberately DROPPED, because each one has to
be recomputed at each rebalance date over that date's admissible occurrences.

The two columns that do come along are renamed as a tripwire:

  Count          -> DiscoveryCount
  ScoredThrough  -> DiscoveredThrough

They are provenance, not inputs. The rename exists so that downstream code
written against signal_scanner.py's schema (which reads row["Count"]) raises a
KeyError instead of silently scoring a 2026 trade against a 2023 occurrence
count. If you find yourself reaching for DiscoveryCount in a decision, that is
the bug this rename is meant to catch.

HorizonClass is kept because it is not a fitted value -- it selects which
horizons a signal is scored over at all (see HORIZON_GROUPS in
signal_scanner.py), so the refit needs it to know which windows to evaluate.

Usage:
    # 1. the discovery scan (hours -- run it on a real CPU, not a laptop)
    python scripts/signal_scanner.py --as-of 2023-12-31 \
        --output-folder results/signals_asof_2023-12-31 --skip-bearish

    # 2. distil it (seconds)
    python scripts/build_vocabulary.py \
        --scan-folder results/signals_asof_2023-12-31 \
        --expect-as-of 2023-12-31 --bullish-only
"""

import os
import sys
import glob
import argparse

import numpy as np
import pandas as pd

# The live folder signal_scanner.py writes by default and live_signal_scan.py
# reads. Distilling THIS folder would produce a vocabulary discovered over the
# full 2021-2026 history -- which is exactly the leak the vocabulary is meant
# to close, and it would be invisible in the output file. Refuse by default.
LIVE_SCAN_FOLDER = "results/signals_v2"

SCAN_SUFFIX = "_signals.csv"

# Signal identity -- everything needed to recompute the signal's boolean series
# and know which horizons to score it over. Nothing fitted.
IDENTITY_COLS = ["Symbol", "SignalName", "Components", "Type", "Direction", "HorizonClass"]

# Provenance, renamed so downstream code cannot mistake them for live values.
PROVENANCE_RENAME = {"Count": "DiscoveryCount", "ScoredThrough": "DiscoveredThrough"}


def read_scan_folder(folder: str) -> pd.DataFrame:
    """Concatenate every per-symbol signals CSV in `folder` into one frame.

    Symbol is taken from the file's own column when present and derived from
    the filename when it isn't, rather than assumed: signal_scanner.py sets it
    from df["Symbol"] when the price file carries that column and falls back to
    the path stem otherwise, so either can legitimately reach disk.
    """
    paths = sorted(glob.glob(os.path.join(folder, f"*{SCAN_SUFFIX}")))
    if not paths:
        sys.exit(f"[ERROR] No {SCAN_SUFFIX} files in {folder}\n"
                 f"        Run signal_scanner.py --as-of ... --output-folder {folder} first.")

    frames, unreadable = [], []
    for p in paths:
        try:
            d = pd.read_csv(p)
        except Exception as e:
            unreadable.append((os.path.basename(p), str(e)))
            continue
        if d.empty:
            continue
        if "Symbol" not in d.columns:
            d["Symbol"] = os.path.basename(p)[: -len(SCAN_SUFFIX)]
        frames.append(d)

    if unreadable:
        print(f"[WARNING] {len(unreadable)} file(s) could not be read, e.g. {unreadable[0]}")
    if not frames:
        sys.exit(f"[ERROR] Every file in {folder} was empty or unreadable.")

    print(f"Read {len(frames)} non-empty symbol file(s) of {len(paths)} found in {folder}")
    return pd.concat(frames, ignore_index=True)


def check_as_of(df: pd.DataFrame, expect_as_of: str | None) -> None:
    """Verify the scan really is point-in-time, and fail loudly if it isn't.

    ScoredThrough is signal_scanner.py's record of the last date it actually
    saw for that symbol (df["Date"].max() after the --as-of truncation), so a
    correct as-of scan produces values that are all <= the cutoff but NOT all
    identical -- a stock that stopped trading in November 2023 legitimately
    reports November, and one whose last 2023 session was the 28th reports the
    28th. Spread below the cutoff is normal; anything above it is a leak.

    This is the check that catches the expensive mistake: pointing this script
    at a full-history scan and getting a vocabulary that silently contains
    combos only discoverable with hindsight.
    """
    if "ScoredThrough" not in df.columns:
        print("[WARNING] No ScoredThrough column -- cannot verify this scan was point-in-time. "
              "Scans written before that column existed have none; re-run the scan if in doubt.")
        return

    st = pd.to_datetime(df["ScoredThrough"], errors="coerce")
    n_bad = int(st.isna().sum())
    if n_bad:
        print(f"[WARNING] {n_bad} row(s) have an unparseable ScoredThrough")

    lo, hi = st.min(), st.max()
    print(f"ScoredThrough spans {lo.date()} .. {hi.date()} across {st.nunique()} distinct value(s)")

    if expect_as_of is None:
        print("[WARNING] No --expect-as-of given, so the cutoff was not verified. "
              "Pass it to turn this into a hard check.")
        return

    cutoff = pd.Timestamp(expect_as_of)
    past = st > cutoff
    if past.any():
        rows = df.loc[past, ["Symbol", "ScoredThrough"]].drop_duplicates().head(5)
        sys.exit(
            f"\n[ERROR] {int(past.sum())} row(s) across {rows.Symbol.nunique()}+ symbol(s) were "
            f"scored past the declared cutoff {cutoff.date()}.\n"
            f"        This scan is NOT point-in-time and its vocabulary would leak.\n"
            f"        Worst offenders:\n{rows.to_string(index=False)}\n"
            f"        Re-run: signal_scanner.py --as-of {cutoff.date()} --output-folder <dir>"
        )
    print(f"  OK: every row scored through {cutoff.date()} or earlier")


def report(vocab: pd.DataFrame, scan_folder: str) -> None:
    """Print the coverage report.

    The question this is really answering is not "did the distillation work"
    but "is there enough pre-cutoff history to backtest on at all". A symbol
    with no vocabulary entries cannot be traded in any window, and a signal
    needs MIN_OCCURRENCES (10) occurrences before the cutoff to have been
    scored at all -- so a short discovery window silently shrinks the tradeable
    universe, and you want that number in front of you before you interpret a
    single backtest return.
    """
    print("\n" + "=" * 72)
    print("VOCABULARY COVERAGE")
    print("=" * 72)

    n_sym = vocab["Symbol"].nunique()
    print(f"\nEntries: {len(vocab):,}  across {n_sym:,} symbol(s)")

    if "Direction" in vocab.columns:
        print("\nBy direction:")
        for d, n in vocab["Direction"].value_counts().items():
            print(f"  {d:<9} {n:>8,}  ({vocab[vocab.Direction == d].Symbol.nunique():,} symbols)")

    # How many symbols exist in the price universe but got no vocabulary at
    # all -- these are untradeable for the whole backtest, not just early on.
    price_files = glob.glob(os.path.join("data/technical", "*_data.csv"))
    if price_files:
        universe = {os.path.basename(f).replace("_data.csv", "") for f in price_files}
        covered = {str(s).replace("_data", "") for s in vocab["Symbol"].unique()}
        missing = universe - covered
        print(f"\nPrice universe: {len(universe):,} symbols")
        print(f"  with vocabulary:    {len(universe & covered):,} "
              f"({len(universe & covered) / len(universe) * 100:.1f}%)")
        print(f"  with NONE:          {len(missing):,} "
              f"-- untradeable for the entire backtest")

    per_sym = vocab.groupby("Symbol").size()
    print(f"\nSignals per symbol: "
          f"p10 {int(np.percentile(per_sym, 10))}, "
          f"median {int(per_sym.median())}, "
          f"p90 {int(np.percentile(per_sym, 90))}, "
          f"max {int(per_sym.max())}")

    if "Components" in vocab.columns:
        print(f"\nDistinct Components sets (global vocabulary): {vocab['Components'].nunique():,}")
    for col in ("Type", "HorizonClass"):
        if col in vocab.columns:
            print(f"\nBy {col}:")
            for k, n in vocab[col].value_counts().items():
                print(f"  {str(k):<12} {n:>8,}")

    # Quantifies how much of the live signal set is only discoverable WITH
    # hindsight. A large gap is not an error -- it is the reason the walk-
    # forward exists, and it is the number to quote when someone asks why the
    # backtest trades fewer names than the live scan shows.
    live = glob.glob(os.path.join(LIVE_SCAN_FOLDER, f"*{SCAN_SUFFIX}"))
    if live and os.path.abspath(scan_folder) != os.path.abspath(LIVE_SCAN_FOLDER):
        print(f"\nVs the live full-history scan in {LIVE_SCAN_FOLDER}:")
        print(f"  live scan covers      {len(live):,} symbols")
        print(f"  this vocabulary       {n_sym:,} symbols "
              f"({n_sym / len(live) * 100:.0f}% of it)")
        print(f"  -> the difference is signal coverage that exists only because of "
              f"post-cutoff data.")


def main():
    ap = argparse.ArgumentParser(
        description="Distil a point-in-time signal_scanner.py scan into the fixed "
                    "signal vocabulary for the walk-forward backtest.")
    ap.add_argument("--scan-folder", required=True, metavar="DIR",
                    help="Folder of *_signals.csv written by a signal_scanner.py --as-of run.")
    ap.add_argument("--expect-as-of", default=None, metavar="YYYY-MM-DD",
                    help="Assert every row was scored through this date or earlier. Strongly "
                         "recommended -- without it a full-history scan passes through silently.")
    ap.add_argument("-o", "--output", default=None, metavar="CSV",
                    help="Output path (default: results/vocabulary_<expect-as-of>.csv).")
    ap.add_argument("--bullish-only", action="store_true",
                    help="Keep Bullish entries only. Matches a long-only backtest and avoids "
                         "carrying a bearish half that the Target*_Price sign bug in "
                         "live_signal_scan.py would misprice anyway.")
    ap.add_argument("--allow-live-folder", action="store_true",
                    help=f"Permit --scan-folder {LIVE_SCAN_FOLDER}. Refused by default because "
                         f"that scan is fitted over the full history and its vocabulary leaks.")
    args = ap.parse_args()

    if (os.path.abspath(args.scan_folder) == os.path.abspath(LIVE_SCAN_FOLDER)
            and not args.allow_live_folder):
        sys.exit(f"[ERROR] {args.scan_folder} is the live full-history scan.\n"
                 f"        Its signals were discovered over 2021-today, so a vocabulary built\n"
                 f"        from it cannot support a backtest inside that period.\n"
                 f"        Produce a point-in-time scan first:\n"
                 f"          python scripts/signal_scanner.py --as-of YYYY-MM-DD \\\n"
                 f"              --output-folder results/signals_asof_YYYY-MM-DD --skip-bearish\n"
                 f"        Or pass --allow-live-folder if you really mean it.")

    df = read_scan_folder(args.scan_folder)
    check_as_of(df, args.expect_as_of)

    if args.bullish_only:
        if "Direction" not in df.columns:
            sys.exit("[ERROR] --bullish-only given but the scan has no Direction column.")
        n_before = len(df)
        df = df[df["Direction"] == "Bullish"]
        print(f"\nBullish-only: kept {len(df):,} of {n_before:,} rows")
        if df.empty:
            sys.exit("[ERROR] No Bullish rows in this scan.")

    missing = [c for c in IDENTITY_COLS if c not in df.columns]
    if missing:
        sys.exit(f"[ERROR] Scan is missing required identity column(s): {missing}\n"
                 f"        Found: {sorted(df.columns)[:20]} ...")

    keep = IDENTITY_COLS + [c for c in PROVENANCE_RENAME if c in df.columns]
    vocab = df[keep].rename(columns=PROVENANCE_RENAME).copy()

    # A symbol/signal pair must appear once. Duplicates would multiply that
    # signal's weight in every downstream ranking without ever looking wrong.
    n_before = len(vocab)
    vocab = vocab.drop_duplicates(subset=["Symbol", "Components", "Direction"])
    if len(vocab) != n_before:
        print(f"\nDropped {n_before - len(vocab):,} duplicate (Symbol, Components, Direction) row(s)")

    vocab = vocab.sort_values(["Symbol", "Direction", "SignalName"]).reset_index(drop=True)

    out = args.output or os.path.join(
        "results", f"vocabulary_{args.expect_as_of or 'asof'}.csv")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    vocab.to_csv(out, index=False)

    report(vocab, args.scan_folder)
    print(f"\nWrote {len(vocab):,} vocabulary entries -> {out}")
    print(f"  columns: {list(vocab.columns)}")
    print("\nEvery fitted statistic was dropped on purpose -- CompositeScore, BestHorizon,")
    print("AvgEV and the Target* ladder must be refit per rebalance date, over occurrences")
    print("admissible as of that date. This file answers only 'which signals to watch'.")


if __name__ == "__main__":
    main()
