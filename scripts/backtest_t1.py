"""
Backtest: buy every fresh bullish signal, exit at the first target still ahead
of the entry price.

WHAT THIS ANSWERS
-----------------
"If I had bought the top 30 stocks whose signals fired each session since
January 2024, entering at the next open and selling at T1, what would I have
made?"

WHY IT IS NOT A LOOP OVER live_signal_scan.py
---------------------------------------------
The obvious implementation is to re-run live_signal_scan.py once per trading
session with the data truncated to that date. That is ~690 runs at 1-2 min
each: 11-23 hours. It is also entirely redundant.

scan_symbol() calls compute_all_signals(df) over a symbol's whole history and
then reads only the LAST row (`np.asarray(series)[-1]`). But every signal here
is a backward-looking function of indicators, so the boolean at row t does not
change when rows after t arrive. Running the scan 690 times recomputes the same
arrays 690 times and throws away all but one row each time.

The equivalence that removes the loop: running the scan as-of session t and
keeping the `Status == Fresh` rows yields exactly the episodes whose
`since_idx == t`. Taking the union over every t in the range therefore yields
every episode START in the range -- and those are computable in a single
vectorised pass per signal:

    starts = combined & ~shift(combined, 1)

live_signal_scan.get_since_index() walks backwards from one row; the same
answer for ALL rows at once is

    last_false = np.maximum.accumulate(np.where(~combined, idx, -1))
    since_idx  = last_false + 1          # valid wherever combined[t]

which was verified against get_since_index() across 1320 (signal, session)
pairs with zero mismatches. One pass instead of 690, same trades.

ENTRY, TARGET AND EXIT
----------------------
Entry is the OPEN of the session after the firing bar. Not the firing bar's own
close: that close is what tells you the signal fired, so you cannot also trade
on it. The overnight gap between the two is recorded as GapPct, because it is a
real cost -- if signals cluster after strong closes, that gap eats the edge and
it should be visible rather than buried in the return.

The target ladder stays anchored to SignalClose (the firing bar's close), since
that is the reference signal_scanner.py's MFE_Target_P10/P20/P30 quantiles are
measured from. Re-anchoring to the entry price would change what the quantile
means.

Those two facts together create the case this script handles explicitly: the
open can gap ABOVE a target that is anchored to the previous close. Aiming at a
level already behind you is not a trade, so the target cascades to the first
rung still above the entry -- T1, else T2, else T3. If the open has cleared all
three, there is nothing left to aim at and the trade is recorded as
`no_target_above_entry` rather than given a synthetic target. TargetUsed says
which rung each trade actually aimed at, and returns are NOT comparable across
rungs, so always group by it.

A target is a resting limit order: it fills the moment the session trades at
the level, whether or not price closed beyond it (see get_target_hit_date in
live_signal_scan.py for the worked SKYGOLD example). So a hit is High >= level
within the holding window. On a session that gapped straight through, the fill
is that session's Open, which is better than the level -- not the level itself.

Holding window is rows s+1 .. s+BestHorizon, matching get_mfe_so_far(): the
quantiles are measured over the h rows AFTER the firing bar, and entering at
s+1's open means every one of those rows' highs is capturable.

Usage:
    python scripts/backtest_t1.py
    python scripts/backtest_t1.py --top-n 10 --start 2024-01-01
    python scripts/backtest_t1.py --signals-folder results/signals_asof_2023-12-31
"""

import os
import sys
import glob
import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import signal_scanner as ss

CANDIDATES_FILE = "results/backtest_candidates.parquet"
# One file per portfolio size, so a sweep does not overwrite its own output.
TRADES_FILE_TMPL = "results/backtest_t1_trades_top{n}.csv"

# ── EDIT THIS LINE to change which portfolio sizes get compared ───────────
# Every size is selected from the SAME scan, so adding or removing values
# costs seconds, not another pass over the universe. --top-n overrides it for
# a one-off run. Note the sizes are nested: top 5 is always a strict subset of
# top 10, which is a subset of top 20, and so on -- the comparison is "does
# concentrating into the higher-ranked names help", not five unrelated
# portfolios.
TOP_N_VALUES = [5, 10, 20, 30, 50]

# Outcome labels. Kept as constants so the summary and any downstream filter
# agree on spelling instead of matching string literals in two places.
OUT_TARGET_HIT   = "target_hit"
OUT_HORIZON      = "horizon_close"      # window elapsed, target never reached
OUT_NO_TARGET    = "no_target_above_entry"
OUT_STILL_OPEN   = "still_open"         # horizon extends past the data we have
OUT_NO_ENTRY_BAR = "no_entry_bar"       # fired on the final row; no next open


def since_index_all(combined: np.ndarray) -> np.ndarray:
    """get_since_index() for every row at once.

    For each t where combined[t] is True, the episode containing t started at
    (index of the last False at or before t) + 1. A running maximum over the
    False positions gives that for all t in one pass. Values at rows where
    combined is False are meaningless and must be masked by the caller.
    """
    idx = np.arange(len(combined))
    return np.maximum.accumulate(np.where(~combined, idx, -1)) + 1


def resolve_trade(opens, highs, closes, segments, n,
                  s: int, best_h: int, target_pcts, signal_close: float) -> dict:
    """Simulate one position: enter at row s+1's open, aim at the first ladder
    rung still above that entry, exit on the first touch or at the window's
    final close.

    `s` is the firing bar. Returns a dict of the resolved trade, always
    including an Outcome -- an unresolvable trade is labelled, never dropped
    silently, so the ledger's row count always matches the candidate count.
    """
    e = s + 1
    if e > n - 1:
        return {"Outcome": OUT_NO_ENTRY_BAR}

    entry = opens[e]
    if not np.isfinite(entry) or entry <= 0 or not np.isfinite(signal_close) or signal_close <= 0:
        return {"Outcome": OUT_NO_ENTRY_BAR}

    # Ladder in price space, anchored to the FIRING bar's close (not the entry
    # price -- see module docstring). Sorted rather than assumed ascending:
    # P10 <= P20 <= P30 holds by construction in signal_scanner.py, but a
    # degenerate distribution can make two rungs equal and a hand-edited
    # signals file could make them out of order.
    rungs = []
    for i, pct in enumerate(target_pcts, start=1):
        if pd.isna(pct):
            continue
        rungs.append((i, float(pct), float(signal_close) * (1.0 + float(pct))))
    rungs.sort(key=lambda r: r[2])

    # The cascade: first rung strictly above the entry. `>` not `>=` because a
    # level exactly at the entry price is already satisfied and would book a
    # guaranteed zero-return "win".
    ahead = [r for r in rungs if r[2] > entry]
    gap_pct = entry / float(signal_close) - 1.0
    if not ahead:
        return {"Outcome": OUT_NO_TARGET, "Entry": entry, "GapPct": gap_pct,
                "TargetUsed": np.nan, "TargetLevel": np.nan}

    tgt_num, tgt_pct, level = ahead[0]

    # Window end: the horizon measured from the firing bar, clipped to the data
    # we actually have. `truncated` distinguishes "the target was never reached
    # in 30 sessions" from "we have not seen 30 sessions yet", which are
    # completely different facts and must not share an outcome label.
    w_end_ideal = s + int(best_h)
    w_end = min(n - 1, w_end_ideal)
    truncated = w_end_ideal > n - 1

    seg_crossed = bool(segments is not None and segments[e] != segments[w_end])

    base = {
        "Entry": entry, "GapPct": gap_pct,
        "TargetUsed": tgt_num, "TargetPct": tgt_pct, "TargetLevel": level,
        "WindowEndRow": w_end, "SegmentCrossed": seg_crossed,
    }

    hits = np.flatnonzero(highs[e:w_end + 1] >= level)
    if hits.size:
        r = e + int(hits[0])
        # Gapped clean through the level -> the resting order fills at the open,
        # which is strictly better than the level. Filling at `level` here would
        # understate the return on exactly the trades that went best.
        fill = opens[r] if (np.isfinite(opens[r]) and opens[r] > level) else level
        return {**base, "Outcome": OUT_TARGET_HIT, "ExitRow": r, "ExitPrice": float(fill),
                "SessionsHeld": r - e, "Return": float(fill) / entry - 1.0}

    if truncated:
        # Unresolved, not a loss: mark to market so it is visible, but the
        # summary must exclude these or every recent signal reads as a failure.
        mtm = closes[w_end]
        return {**base, "Outcome": OUT_STILL_OPEN, "ExitRow": w_end,
                "ExitPrice": float(mtm), "SessionsHeld": w_end - e,
                "Return": float(mtm) / entry - 1.0}

    exit_px = closes[w_end]
    return {**base, "Outcome": OUT_HORIZON, "ExitRow": w_end, "ExitPrice": float(exit_px),
            "SessionsHeld": w_end - e, "Return": float(exit_px) / entry - 1.0}


def scan_symbol_episodes(job: tuple) -> list:
    """Every fresh bullish episode for one symbol, already resolved into a
    trade.

    Resolving here rather than in the parent is what keeps the whole run to a
    single pass: the parent would otherwise have to re-open every price file to
    simulate the trades it selected. It also reduces one row per (symbol,
    session) -- the best-scoring signal -- instead of shipping back every
    signal-level episode, which across the universe is ~12M rows against
    ~780k. Since the parent's top-N selection is per symbol anyway (one
    position per stock per day), the discarded rows could never have been
    picked.
    """
    tech_path, sig_path, start_date, filters = job
    try:
        df = pd.read_csv(tech_path, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
        scored = pd.read_csv(sig_path)
    except Exception:
        return []
    if len(df) < 60 or scored.empty:
        return []

    scored = scored[scored["Direction"] == "Bullish"]
    if scored.empty:
        return []

    # Mirror live_signal_scan.py's own eligibility filters so this backtests
    # the list that tool would actually have shown, not a looser one.
    gap = filters.get("max_ev_median_gap")
    if gap is not None and gap >= 0 and "AvgMedianReturn" in scored.columns:
        scored = scored[(scored["AvgEV"] - scored["AvgMedianReturn"]).abs() <= gap]
    if filters.get("min_count") is not None:
        scored = scored[scored["Count"] >= filters["min_count"]]
    if filters.get("min_composite") is not None:
        scored = scored[scored["CompositeScore"] >= filters["min_composite"]]
    if filters.get("exclude_concentrated") and "Concentration_Flag" in scored.columns:
        scored = scored[scored["Concentration_Flag"].fillna(0) != 1]
    scored = scored[scored["BestHorizon"].notna()]
    if scored.empty:
        return []

    signal_map, _ = ss.compute_all_signals(df)
    n = len(df)
    opens   = df["Open"].to_numpy(dtype=float)
    highs   = df["High"].to_numpy(dtype=float)
    closes  = df["Close"].to_numpy(dtype=float)
    dates   = df["Date"].to_numpy()
    segments = df["Segment"].to_numpy() if "Segment" in df.columns else None

    # Only episodes firing at or after the backtest start are tradeable, but
    # `combined` still has to be built over the FULL history: an episode that
    # began in 2023 and was still true in 2024 did not fire in 2024, and
    # truncating the array first would mislabel its first in-range row as a
    # fresh start.
    in_range = df["Date"].to_numpy() >= np.datetime64(start_date)

    best_per_row = {}   # row -> (composite, resolved trade dict)
    for _, row in scored.iterrows():
        comps = [c.strip() for c in str(row["Components"]).split("+")]
        combined = np.ones(n, dtype=bool)
        ok = True
        for c in comps:
            arr = signal_map.get(c)
            if arr is None:
                ok = False
                break
            combined &= np.asarray(arr, dtype=bool)
        if not ok:
            continue

        starts = combined & ~np.r_[False, combined[:-1]]
        starts &= in_range
        if not starts.any():
            continue

        composite = row["CompositeScore"]
        best_h = int(row["BestHorizon"])
        tpcts = (row.get("Target1_Conservative_BestHorizon", np.nan),
                 row.get("Target2_Median_BestHorizon", np.nan),
                 row.get("Target3_Stretch_BestHorizon", np.nan))
        if all(pd.isna(t) for t in tpcts):
            continue

        for s in np.flatnonzero(starts):
            s = int(s)
            prev = best_per_row.get(s)
            if prev is not None and prev[0] >= composite:
                continue
            trade = resolve_trade(opens, highs, closes, segments, n,
                                  s, best_h, tpcts, closes[s])
            trade.update({
                "Symbol": row["Symbol"], "SignalName": row["SignalName"],
                "SignalDate": dates[s], "SignalClose": float(closes[s]),
                "BestHorizon": best_h, "CompositeScore": float(composite),
                "Count": row["Count"],
                "EntryDate": dates[s + 1] if s + 1 < n else np.datetime64("NaT"),
                "ExitDate": (dates[trade["ExitRow"]] if "ExitRow" in trade
                             else np.datetime64("NaT")),
            })
            best_per_row[s] = (composite, trade)

    return [t for _, t in best_per_row.values()]


def headline_stats(sel: pd.DataFrame) -> dict:
    """The numbers the comparison table shows, for one portfolio size.

    Only target_hit and horizon_close carry a realised return:
    no_target_above_entry was never entered, still_open has not finished, and
    no_entry_bar never started. Averaging those in would read unfinished
    trades as losses.
    """
    res = sel[sel["Outcome"].isin([OUT_TARGET_HIT, OUT_HORIZON])]
    if res.empty:
        return {"trades": len(sel), "resolved": 0}
    return {
        "trades":   len(sel),
        "resolved": len(res),
        "reached":  float((res["Outcome"] == OUT_TARGET_HIT).mean()),
        "mean":     float(res["Return"].mean()),
        "median":   float(res["Return"].median()),
        "std":      float(res["Return"].std()),
        "total":    float(res["Return"].sum()),
        "sessions": float(res["SessionsHeld"].mean()),
        "symbols":  int(sel["Symbol"].nunique()),
    }


def comparison_table(rows: list) -> None:
    """One line per portfolio size. This is the headline of a sweep -- the
    per-size detail below it is the same data broken out."""
    print("\n" + "=" * 84)
    print("COMPARISON ACROSS PORTFOLIO SIZES")
    print("=" * 84)
    print(f"\n{'N':>4} {'trades':>9} {'resolved':>9} {'reached':>8} "
          f"{'mean':>9} {'median':>9} {'std':>8} {'total':>11} {'sess':>6} {'syms':>6}")
    print("-" * 84)
    for n, st in rows:
        if not st.get("resolved"):
            print(f"{n:>4} {st['trades']:>9,}        0   (no resolved trades)")
            continue
        print(f"{n:>4} {st['trades']:>9,} {st['resolved']:>9,} "
              f"{st['reached']*100:>7.1f}% {st['mean']*100:>+8.3f}% "
              f"{st['median']*100:>+8.3f}% {st['std']*100:>7.2f}% "
              f"{st['total']*100:>+10.0f}% {st['sessions']:>6.1f} {st['symbols']:>6,}")
    print("\nmean/median/std/total are per-trade returns at equal notional, gross of")
    print("costs. 'total' is the sum of returns at 1 unit per trade -- it scales with")
    print("trade count, so compare 'mean' across sizes, not 'total'.")


def summarise(sel: pd.DataFrame, top_n: int) -> None:
    """Print the headline numbers, split by the facts that make returns
    non-comparable: which rung was aimed at, and whether the trade resolved."""
    print("\n" + "=" * 72)
    print(f"RESULTS — top {top_n} stocks per session, exit at first target above entry")
    print("=" * 72)

    print(f"\nTrades: {len(sel):,}  "
          f"over {sel['SignalDate'].dt.date.nunique():,} sessions  "
          f"({sel['Symbol'].nunique():,} distinct symbols)")
    print(f"Window: {sel['SignalDate'].min().date()} .. {sel['SignalDate'].max().date()}")

    print("\nBy outcome:")
    for k, v in sel["Outcome"].value_counts().items():
        print(f"  {k:<24} {v:>7,}  ({v/len(sel)*100:5.1f}%)")

    # still_open and no_entry_bar have no realised return; no_target_above_entry
    # was never entered at all. Only resolved trades belong in a P&L number.
    res = sel[sel["Outcome"].isin([OUT_TARGET_HIT, OUT_HORIZON])].copy()
    if res.empty:
        print("\nNo resolved trades in range.")
        return

    print(f"\nResolved trades: {len(res):,}")
    hit = res["Outcome"] == OUT_TARGET_HIT
    print(f"  target reached      : {hit.sum():,} ({hit.mean()*100:.1f}%)")
    print(f"  horizon expired     : {(~hit).sum():,} ({(~hit).mean()*100:.1f}%)")

    print("\nReturn per trade (equal notional, gross of costs):")
    r = res["Return"]
    print(f"  mean   : {r.mean()*100:+.3f}%")
    print(f"  median : {r.median()*100:+.3f}%")
    print(f"  std    : {r.std()*100:.3f}%")
    print(f"  p5/p95 : {r.quantile(.05)*100:+.2f}% / {r.quantile(.95)*100:+.2f}%")
    print(f"  worst  : {r.min()*100:+.2f}%   best: {r.max()*100:+.2f}%")
    print(f"  sum of returns (1 unit each): {r.sum()*100:+.1f}% "
          f"over {len(res):,} trades")

    print("\nBy target rung aimed at (returns are NOT comparable across rungs):")
    for t, g in res.groupby("TargetUsed"):
        h = (g["Outcome"] == OUT_TARGET_HIT).mean()
        print(f"  T{int(t)}: {len(g):>7,} trades | reached {h*100:5.1f}% | "
              f"mean {g['Return'].mean()*100:+.3f}% | median {g['Return'].median()*100:+.3f}%")

    print("\nMean overnight gap (entry open vs firing close):")
    print(f"  {res['GapPct'].mean()*100:+.3f}%  "
          f"(median {res['GapPct'].median()*100:+.3f}%)")
    print("  This is paid on every trade before the target is measured.")

    print(f"\nSessions held: mean {res['SessionsHeld'].mean():.1f}, "
          f"median {int(res['SessionsHeld'].median())}, max {int(res['SessionsHeld'].max())}")

    if res["SegmentCrossed"].any():
        n_seg = int(res["SegmentCrossed"].sum())
        print(f"\n[NOTE] {n_seg:,} resolved trade(s) span a Segment boundary (a long "
              f"trading gap).\n       signal_scanner.py nulls forward windows across "
              f"these; filter on SegmentCrossed to match.")

    print("\nBy year:")
    for y, g in res.groupby(res["SignalDate"].dt.year):
        h = (g["Outcome"] == OUT_TARGET_HIT).mean()
        print(f"  {y}: {len(g):>7,} trades | reached {h*100:5.1f}% | "
              f"mean {g['Return'].mean()*100:+.3f}% | total {g['Return'].sum()*100:+.1f}%")


def main():
    ap = argparse.ArgumentParser(
        description="Backtest buying fresh bullish signals and exiting at the first "
                    "target above the entry price.")
    ap.add_argument("--start", default="2024-01-01", help="First tradeable signal date.")
    ap.add_argument("--top-n", type=int, nargs="+", default=None, metavar="N",
                    help=f"Stocks per session, ranked by CompositeScore. One position per "
                         f"stock per session. Accepts several values to sweep, e.g. "
                         f"--top-n 5 10 20. Defaults to TOP_N_VALUES at the top of this "
                         f"file (currently {TOP_N_VALUES}).")
    ap.add_argument("--detail", action="store_true",
                    help="Print the full per-size breakdown as well as the comparison "
                         "table. Implied when only one size is being run.")
    ap.add_argument("--signals-folder", default=None,
                    help=f"Scored signals folder (default {ss.OUTPUT_FOLDER}). Point this "
                         f"at a signal_scanner.py --as-of output to remove the hindsight "
                         f"in the ranking.")
    ap.add_argument("--workers", type=int, default=None, help="Default: os.cpu_count().")
    ap.add_argument("--max-ev-median-gap", type=float, default=0.1,
                    help="Mirrors live_signal_scan.py's filter of the same name. "
                         "Negative disables.")
    ap.add_argument("--min-count", type=int, default=None)
    ap.add_argument("--min-composite", type=float, default=None)
    ap.add_argument("--exclude-concentrated", action="store_true")
    ap.add_argument("--reuse-candidates", action="store_true",
                    help=f"Skip the scan and re-select from {CANDIDATES_FILE}. The scan is "
                         f"the only expensive part, so this makes sweeping --top-n free.")
    args = ap.parse_args()

    folder = args.signals_folder or ss.OUTPUT_FOLDER

    if args.reuse_candidates:
        if not os.path.exists(CANDIDATES_FILE):
            sys.exit(f"[ERROR] {CANDIDATES_FILE} not found -- run once without "
                     f"--reuse-candidates first.")
        cand = pd.read_parquet(CANDIDATES_FILE)
        print(f"Reusing {len(cand):,} candidates from {CANDIDATES_FILE}")
    else:
        filters = {
            "max_ev_median_gap": args.max_ev_median_gap,
            "min_count": args.min_count,
            "min_composite": args.min_composite,
            "exclude_concentrated": args.exclude_concentrated,
        }
        jobs = []
        for sig_path in sorted(glob.glob(os.path.join(folder, "*_data_signals.csv"))):
            symbol = os.path.basename(sig_path).replace("_data_signals.csv", "")
            tech_path = os.path.join(ss.INPUT_FOLDER, f"{symbol}{ss.DATA_SUFFIX}")
            if os.path.exists(tech_path):
                jobs.append((tech_path, sig_path, args.start, filters))
        if not jobs:
            sys.exit(f"[ERROR] No scored signals found in {folder}")

        n_workers = max(1, args.workers or os.cpu_count() or 1)
        print(f"Scanning {len(jobs):,} symbols from {folder} "
              f"across {n_workers} worker(s), signals from {args.start}...")

        rows = []
        chunksize = max(1, min(16, len(jobs) // (n_workers * 8)))
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            for i, out in enumerate(ex.map(scan_symbol_episodes, jobs, chunksize=chunksize), 1):
                rows.extend(out)
                if i % 300 == 0:
                    print(f"  [{i}/{len(jobs)}] {len(rows):,} candidate episodes")

        if not rows:
            sys.exit("[ERROR] No episodes found in range.")
        cand = pd.DataFrame(rows)
        cand["SignalDate"] = pd.to_datetime(cand["SignalDate"])
        os.makedirs("results", exist_ok=True)
        cand.to_parquet(CANDIDATES_FILE, index=False)
        print(f"\n{len(cand):,} candidate episodes -> {CANDIDATES_FILE}")

    # Selection. The worker already reduced to one row per (symbol, session),
    # so this is purely "the top N stocks that session".
    #
    # Ranked once with a single groupby rather than once per size: every size
    # is a prefix of the same ranking, so RankInSession <= n is all a size
    # needs, and it lands in the output CSV as a useful column besides.
    cand = cand.sort_values(["SignalDate", "CompositeScore"], ascending=[True, False])
    cand["RankInSession"] = cand.groupby("SignalDate").cumcount() + 1

    top_ns = sorted(set(args.top_n or TOP_N_VALUES))
    if any(n < 1 for n in top_ns):
        sys.exit(f"[ERROR] --top-n values must be >= 1, got {top_ns}")

    rows = []
    for n in top_ns:
        sel = cand[cand["RankInSession"] <= n].copy()
        path = TRADES_FILE_TMPL.format(n=n)
        sel.to_csv(path, index=False)
        print(f"  top {n:>3}/session: {len(sel):>7,} trades -> {path}")
        rows.append((n, headline_stats(sel)))

    comparison_table(rows)

    if args.detail or len(top_ns) == 1:
        for n in top_ns:
            summarise(cand[cand["RankInSession"] <= n].copy(), n)

    print(f"\nRe-run with --reuse-candidates to re-select without rescanning "
          f"({len(cand):,} candidates cached).")


if __name__ == "__main__":
    main()
