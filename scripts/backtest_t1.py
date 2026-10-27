"""
Backtest fresh bullish signals with independent fixed T1/T2/T3 baselines and a
simple dynamic position experiment. Dynamic positions buy ~10,000 per entry; a
fresh signal with a strictly higher CompositeScore adds another ~10,000, updates
the target, and may extend the horizon. All accumulated shares sell together.

WHAT THIS ANSWERS
-----------------
"If I had bought the top N bullish stocks whose signals fired each session,
entering at the next open and independently targeting T1, T2, or T3, what would
I have made?"

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
open can gap ABOVE a target that is anchored to the previous close. Each target
strategy is evaluated independently. T1 aims only at T1, T2 only at T2, and T3
only at T3. If a strategy's target is already at or below the entry, that
strategy records `no_target_above_entry` rather than inventing a trade. The
three strategies share the same signals, entry prices, and holding horizons.

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


def resolve_trade_target(opens, highs, closes, segments, n,
                         s: int, best_h: int, target_pct,
                         target_num: int, signal_close: float) -> dict:
    """Simulate one target-specific strategy for one signal.

    All target strategies enter at the open of row s+1 and use the same
    BestHorizon. The only difference between T1, T2, and T3 is the target rung.
    A target at or below the entry price is unavailable for that strategy.
    """
    e = s + 1
    if e > n - 1:
        return {"Outcome": OUT_NO_ENTRY_BAR}

    entry = opens[e]
    if (not np.isfinite(entry) or entry <= 0
            or not np.isfinite(signal_close) or signal_close <= 0):
        return {"Outcome": OUT_NO_ENTRY_BAR}

    gap_pct = entry / float(signal_close) - 1.0

    # The target ladder is anchored to the signal bar's close, not the entry.
    if pd.isna(target_pct):
        return {
            "Outcome": OUT_NO_TARGET,
            "Entry": float(entry),
            "GapPct": gap_pct,
            "TargetUsed": target_num,
            "TargetPct": np.nan,
            "TargetLevel": np.nan,
        }

    target_pct = float(target_pct)
    level = float(signal_close) * (1.0 + target_pct)

    # A target already at or below entry is not a valid target for this strategy.
    if not np.isfinite(level) or level <= entry:
        return {
            "Outcome": OUT_NO_TARGET,
            "Entry": float(entry),
            "GapPct": gap_pct,
            "TargetUsed": target_num,
            "TargetPct": target_pct,
            "TargetLevel": level,
        }

    # Holding window is rows s+1 .. s+BestHorizon, clipped to available data.
    w_end_ideal = s + int(best_h)
    w_end = min(n - 1, w_end_ideal)
    truncated = w_end_ideal > n - 1

    seg_crossed = bool(
        segments is not None and segments[e] != segments[w_end]
    )

    base = {
        "Entry": float(entry),
        "GapPct": gap_pct,
        "TargetUsed": target_num,
        "TargetPct": target_pct,
        "TargetLevel": level,
        "WindowEndRow": w_end,
        "SegmentCrossed": seg_crossed,
    }

    hits = np.flatnonzero(highs[e:w_end + 1] >= level)
    if hits.size:
        r = e + int(hits[0])
        # If the market opens above the target, model the fill at that open.
        fill = opens[r] if (np.isfinite(opens[r]) and opens[r] > level) else level
        return {
            **base,
            "Outcome": OUT_TARGET_HIT,
            "ExitRow": r,
            "ExitPrice": float(fill),
            "SessionsHeld": r - e,
            "Return": float(fill) / entry - 1.0,
        }

    if truncated:
        # The target was not reached, and the full horizon is not yet available.
        mtm = closes[w_end]
        return {
            **base,
            "Outcome": OUT_STILL_OPEN,
            "ExitRow": w_end,
            "ExitPrice": float(mtm),
            "SessionsHeld": w_end - e,
            "Return": float(mtm) / entry - 1.0,
        }

    exit_px = closes[w_end]
    return {
        **base,
        "Outcome": OUT_HORIZON,
        "ExitRow": w_end,
        "ExitPrice": float(exit_px),
        "SessionsHeld": w_end - e,
        "Return": float(exit_px) / entry - 1.0,
    }


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

    # Normalize case/whitespace so only explicitly bullish rows are eligible.
    scored = scored[
        scored["Direction"].astype(str).str.strip().str.casefold() == "bullish"
    ]
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
            # Resolve three independent target strategies. All use the same
            # signal, entry open, and BestHorizon; only the target rung differs.
            strategy_results = {}
            for target_num, target_pct in enumerate(tpcts, start=1):
                result = resolve_trade_target(
                    opens, highs, closes, segments, n,
                    s, best_h, target_pct, target_num, closes[s]
                )
                result["EntryDate"] = (
                    dates[s + 1] if s + 1 < n else np.datetime64("NaT")
                )
                result["ExitDate"] = (
                    dates[result["ExitRow"]]
                    if "ExitRow" in result else np.datetime64("NaT")
                )
                strategy_results[target_num] = result

            # Keep the existing generic trade columns mapped to T1, so the
            # existing CSV/statistics/log workflow now represents the T1 strategy.
            trade = dict(strategy_results[1])
            trade.update({
                "Symbol": row["Symbol"],
                # Canonical ticker comes from the actual technical-data filename.
                # SignalRow lets the position simulator access the exact price bar.
                "TechSymbol": os.path.basename(tech_path).replace(ss.DATA_SUFFIX, ""),
                "SignalRow": int(s),
                "SignalName": row["SignalName"],
                "Direction": str(row.get("Direction", "")),
                "SignalDate": dates[s],
                "SignalClose": float(closes[s]),
                "BestHorizon": best_h,
                "CompositeScore": float(composite),
                "Count": row["Count"],
            })

            # Persist all strategy-specific outcomes in the candidate cache so
            # T2/T3 can be selected without rescanning the price history.
            for target_num, result in strategy_results.items():
                for key, value in result.items():
                    trade[f"T{target_num}_{key}"] = value

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


def comparison_table(rows: list, target_label: str = "T1") -> None:
    """One line per portfolio size. This is the headline of a sweep -- the
    per-size detail below it is the same data broken out."""
    print("\n" + "=" * 84)
    print(f"COMPARISON ACROSS PORTFOLIO SIZES — {target_label} STRATEGY")
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


def summarise(sel: pd.DataFrame, top_n: int, target_label: str = "T1") -> None:
    """Print headline results for one target strategy and portfolio size."""
    print("\n" + "=" * 72)
    print(f"RESULTS — top {top_n} stocks per session, dedicated {target_label} target strategy")
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

    # This summary is already for one dedicated target strategy, so all
    # resolved trades in this section target the same rung.

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



def export_trader_log(sel: pd.DataFrame, top_n: int, target_label: str = "T1") -> None:
    """Export a trader-friendly ENTRY/EXIT log to TXT and Excel.

    ENTRY rows are created for trades that were actually entered.
    EXIT rows are created only for resolved trades. Unresolved trades
    are shown as OPEN entries without a synthetic exit.
    """
    os.makedirs("results", exist_ok=True)

    resolved_outcomes = [OUT_TARGET_HIT, OUT_HORIZON]
    entered_outcomes = resolved_outcomes + [OUT_STILL_OPEN]

    entered = sel[sel["Outcome"].isin(entered_outcomes)].copy()
    events = []

    for _, trade in entered.iterrows():
        if pd.isna(trade.get("Entry")) or pd.isna(trade.get("EntryDate")):
            continue

        symbol = trade["Symbol"]
        signal_date = pd.to_datetime(trade["SignalDate"])
        entry_date = pd.to_datetime(trade["EntryDate"])
        outcome = trade["Outcome"]

        # Match the ENTRY and EXIT events for the same backtest trade.
        trade_id = f"{symbol}_{signal_date.strftime('%Y%m%d')}"

        # ENTRY EVENT
        events.append({
            "Date": entry_date,
            "Symbol": symbol,
            "Trade Type": "ENTRY",
            "LTP/CMP": float(trade["Entry"]),
            "Gain/Loss %": np.nan,
            "Days Taken": np.nan,
            "Status": "OPEN" if outcome == OUT_STILL_OPEN else "CLOSED",
            "Trade ID": trade_id,
        })

        # EXIT EVENT: only resolved trades get an exit row.
        if outcome in resolved_outcomes:
            exit_date = trade.get("ExitDate")
            exit_price = trade.get("ExitPrice")
            trade_return = trade.get("Return")
            sessions_held = trade.get("SessionsHeld")

            if (
                pd.isna(exit_date)
                or pd.isna(exit_price)
                or pd.isna(trade_return)
                or pd.isna(sessions_held)
            ):
                continue

            events.append({
                "Date": pd.to_datetime(exit_date),
                "Symbol": symbol,
                "Trade Type": "EXIT",
                "LTP/CMP": float(exit_price),
                # Return is stored as a decimal: 0.05 means +5%.
                "Gain/Loss %": float(trade_return),
                # SessionsHeld counts trading sessions in this backtest.
                "Days Taken": int(sessions_held),
                "Status": outcome,
                "Trade ID": trade_id,
            })

    columns = [
        "Date",
        "Symbol",
        "Trade Type",
        "LTP/CMP",
        "Gain/Loss %",
        "Days Taken",
        "Status",
        "Trade ID",
    ]
    log = pd.DataFrame(events, columns=columns)

    if not log.empty:
        # Entries appear before exits when dates are identical.
        log["_event_order"] = log["Trade Type"].map({"ENTRY": 0, "EXIT": 1})
        log = (
            log.sort_values(
                ["Date", "Symbol", "_event_order", "Trade ID"],
                kind="stable",
            )
            .drop(columns="_event_order")
            .reset_index(drop=True)
        )
        log["Date"] = pd.to_datetime(log["Date"])

    # OUTPUT 1: TEXT FILE
    suffix = "" if target_label == "T1" else f"_{target_label.lower()}"
    txt_path = f"results/trader_trades{suffix}_top{top_n}.txt"
    text_log = log.copy()
    if not text_log.empty:
        text_log["Date"] = text_log["Date"].dt.strftime("%Y-%m-%d")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"TRADER TRADE LOG - {target_label} STRATEGY - TOP {top_n} STOCKS PER SESSION\n")
        f.write("=" * 105 + "\n\n")
        f.write("Gain/Loss % is populated only on EXIT rows.\n")
        f.write("Days Taken counts trading sessions, not calendar days.\n")
        f.write("OPEN means the trade has no resolved exit yet.\n\n")

        if text_log.empty:
            f.write("No entered trades found.\n")
        else:
            f.write(
                text_log.to_string(
                    index=False,
                    formatters={
                        "LTP/CMP": lambda x: "" if pd.isna(x) else f"{x:,.2f}",
                        "Gain/Loss %": lambda x: (
                            "" if pd.isna(x) else f"{x * 100:+.2f}%"
                        ),
                        "Days Taken": lambda x: (
                            "" if pd.isna(x) else str(int(x))
                        ),
                    },
                )
            )
            f.write("\n")

    # OUTPUT 2: EXCEL FILE
    xlsx_path = f"results/trader_trades{suffix}_top{top_n}.xlsx"
    with pd.ExcelWriter(
        xlsx_path,
        engine="openpyxl",
        date_format="yyyy-mm-dd",
        datetime_format="yyyy-mm-dd",
    ) as writer:
        log.to_excel(writer, sheet_name="Trade Log", index=False)
        ws = writer.sheets["Trade Log"]
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions

        widths = {
            "A": 14,  # Date
            "B": 18,  # Symbol
            "C": 14,  # Trade Type
            "D": 16,  # LTP/CMP
            "E": 16,  # Gain/Loss %
            "F": 14,  # Days Taken
            "G": 20,  # Status
            "H": 28,  # Trade ID
        }
        for col, width in widths.items():
            ws.column_dimensions[col].width = width

        for row in range(2, ws.max_row + 1):
            ws.cell(row=row, column=1).number_format = "yyyy-mm-dd"
            ws.cell(row=row, column=4).number_format = "#,##0.00"
            ws.cell(row=row, column=5).number_format = "+0.00%;-0.00%;0.00%"
            ws.cell(row=row, column=6).number_format = "0"

    print(f"  Trader log -> {txt_path}")
    print(f"  Trader log -> {xlsx_path}")


def strategy_view(sel: pd.DataFrame, target_num: int) -> pd.DataFrame:
    """Return a view of candidates with generic trade fields for T1, T2, or T3."""
    out = sel.copy()
    prefix = f"T{target_num}_"
    fields = [
        "Outcome", "Entry", "GapPct", "TargetUsed", "TargetPct",
        "TargetLevel", "WindowEndRow", "SegmentCrossed", "ExitRow",
        "ExitPrice", "SessionsHeld", "Return", "EntryDate", "ExitDate",
    ]
    missing = [field for field in fields if f"{prefix}{field}" not in out.columns]
    if missing:
        raise ValueError(
            f"Candidate cache is missing {prefix} strategy fields: {missing}. "
            "Rerun without --reuse-candidates to rebuild the cache."
        )
    for field in fields:
        out[field] = out[f"{prefix}{field}"]
    return out



DYNAMIC_BUDGET_PER_BUY = 10_000


def _shares_for_budget(price: float) -> int:
    """Buy approximately ₹10,000 worth, rounded to whole shares (minimum one)."""
    if not np.isfinite(price) or price <= 0:
        return 0
    return max(1, int(np.floor(DYNAMIC_BUDGET_PER_BUY / price + 0.5)))


def _date_at(prices: pd.DataFrame, row_index: int):
    if 0 <= int(row_index) < len(prices):
        return pd.Timestamp(prices.iloc[int(row_index)]["Date"])
    return pd.NaT


def _close_dynamic_position(position: dict, symbol: str, date, row_index: int,
                            price: float, reason: str, trades: list,
                            transactions: list):
    """Sell every share accumulated in a position and record the round trip."""
    shares = int(position["Shares"])
    proceeds = shares * float(price)
    pnl = proceeds - float(position["TotalCost"])
    invested = float(position["TotalCost"])
    ret = pnl / invested if invested else np.nan

    trades.append({
        "Symbol": symbol,
        "Status": reason,
        "InitialSignalDate": position["InitialSignalDate"],
        "EntryDate": position["EntryDate"],
        "ExitDate": pd.Timestamp(date),
        "InitialCompositeScore": position["InitialScore"],
        "FinalCompositeScore": position["GoverningScore"],
        "BuyCount": position["BuyCount"],
        "SharesBought": shares,
        "AverageBuyPrice": invested / shares if shares else np.nan,
        "TotalInvested": invested,
        "ExitPrice": float(price),
        "TotalProceeds": proceeds,
        "PnL": pnl,
        "Return": ret,
        "InitialTarget": position["InitialTarget"],
        "FinalTarget": position["TargetPrice"],
        "InitialHorizonEndDate": position["InitialHorizonEndDate"],
        "FinalHorizonEndDate": position["HorizonEndDate"],
        "TargetUpdates": position["TargetUpdates"],
        "FinalSignalDate": position["GoverningSignalDate"],
    })
    transactions.append({
        "Date": pd.Timestamp(date),
        "Symbol": symbol,
        "Action": "SELL_ALL",
        "SignalDate": position["GoverningSignalDate"],
        "CompositeScore": position["GoverningScore"],
        "Price": float(price),
        "Shares": shares,
        "BuyAmount": np.nan,
        "CumulativeShares": 0,
        "CumulativeInvested": invested,
        "TargetBefore": position["TargetPrice"],
        "TargetAfter": position["TargetPrice"],
        "HorizonBefore": position["HorizonEndDate"],
        "HorizonAfter": position["HorizonEndDate"],
        "Reason": reason,
        "PnL": pnl,
        "Return": ret,
    })


def simulate_dynamic_symbol(symbol: str, prices: pd.DataFrame,
                            signals: pd.DataFrame, target_num: int,
                            top_n: int) -> tuple[list, list]:
    """Simulate one stock's position, adding ~₹10,000 only on stronger signals.

    Signals are known after their signal-day close. New buys/adds execute at the
    next session's open. An accepted stronger signal replaces the target and can
    extend the horizon to the newer signal's horizon, but does not shorten it.
    Existing shares and added shares are sold together on target/horizon exit.
    """
    trades: list = []
    transactions: list = []
    if prices.empty or signals.empty:
        return trades, transactions

    prices = prices.sort_values("Date").reset_index(drop=True)
    prices["Date"] = pd.to_datetime(prices["Date"])
    n_rows = len(prices)
    opens = pd.to_numeric(prices["Open"], errors="coerce").to_numpy(dtype=float)
    highs = pd.to_numeric(prices["High"], errors="coerce").to_numpy(dtype=float)
    closes = pd.to_numeric(prices["Close"], errors="coerce").to_numpy(dtype=float)

    target_col = f"T{target_num}_TargetPct"
    if target_col not in signals.columns:
        return trades, transactions

    by_row = {}
    for _, signal in signals.sort_values("SignalRow").iterrows():
        try:
            r = int(signal["SignalRow"])
        except (TypeError, ValueError):
            continue
        if 0 <= r < n_rows:
            # Candidate scan already keeps only the highest-scoring signal per
            # stock/session, so there can be only one signal event per row here.
            by_row[r] = signal

    if not by_row:
        return trades, transactions

    position = None
    pending_order = None
    first_signal_row = min(by_row)

    for r in range(first_signal_row, n_rows):
        date = pd.Timestamp(prices.iloc[r]["Date"])
        open_price = opens[r]
        high_price = highs[r]
        close_price = closes[r]

        # 1) Execute the previous close's buy/add at today's open.
        if pending_order is not None and np.isfinite(open_price) and open_price > 0:
            order = pending_order
            qty = _shares_for_budget(float(open_price))
            amount = qty * float(open_price)
            if order["Action"] == "BUY":
                pos = {
                    "EntryDate": date,
                    "InitialSignalDate": order["SignalDate"],
                    "InitialScore": order["CompositeScore"],
                    "GoverningScore": order["CompositeScore"],
                    "GoverningSignalDate": order["SignalDate"],
                    "Shares": qty,
                    "TotalCost": amount,
                    "BuyCount": 1,
                    "TargetPrice": order["TargetPrice"],
                    "InitialTarget": order["TargetPrice"],
                    "HorizonEndRow": order["HorizonEndRow"],
                    "InitialHorizonEndDate": _date_at(prices, order["HorizonEndRow"]),
                    "HorizonEndDate": _date_at(prices, order["HorizonEndRow"]),
                    "TargetUpdates": 0,
                }
                position = pos
                transactions.append({
                    "Date": date, "Symbol": symbol, "Action": "BUY",
                    "SignalDate": order["SignalDate"],
                    "CompositeScore": order["CompositeScore"],
                    "Price": float(open_price), "Shares": qty,
                    "BuyAmount": amount, "CumulativeShares": qty,
                    "CumulativeInvested": amount,
                    "TargetBefore": np.nan, "TargetAfter": pos["TargetPrice"],
                    "HorizonBefore": pd.NaT, "HorizonAfter": pos["HorizonEndDate"],
                    "Reason": "top_n_signal", "PnL": np.nan, "Return": np.nan,
                })
            elif position is not None:
                old_target = position["TargetPrice"]
                old_horizon = position["HorizonEndDate"]
                position["Shares"] += qty
                position["TotalCost"] += amount
                position["BuyCount"] += 1
                position["TargetUpdates"] += 1
                transactions.append({
                    "Date": date, "Symbol": symbol,
                    "Action": "ADD_BUY_TARGET_UPDATE",
                    "SignalDate": order["SignalDate"],
                    "CompositeScore": order["CompositeScore"],
                    "Price": float(open_price), "Shares": qty,
                    "BuyAmount": amount,
                    "CumulativeShares": position["Shares"],
                    "CumulativeInvested": position["TotalCost"],
                    "TargetBefore": old_target, "TargetAfter": position["TargetPrice"],
                    "HorizonBefore": old_horizon, "HorizonAfter": position["HorizonEndDate"],
                    "Reason": "higher_composite_score", "PnL": np.nan, "Return": np.nan,
                })
        pending_order = None

        # 2) Today's high can trigger the active target. Sell ALL shares bought.
        # If the session gaps above the target, use the open as the fill.
        if position is not None and np.isfinite(high_price) and high_price >= position["TargetPrice"]:
            fill = float(open_price) if np.isfinite(open_price) and open_price > position["TargetPrice"] else float(position["TargetPrice"])
            _close_dynamic_position(position, symbol, date, r, fill, "target_hit", trades, transactions)
            position = None
        # 3) If target was not hit, liquidate all shares at the horizon-end close.
        elif position is not None and r >= position["HorizonEndRow"] and np.isfinite(close_price) and close_price > 0:
            _close_dynamic_position(position, symbol, date, r, float(close_price), "horizon_close", trades, transactions)
            position = None

        # 4) Signal arrives at today's close. It can only affect the next session.
        signal = by_row.get(r)
        if signal is None or r + 1 >= n_rows:
            continue
        try:
            score = float(signal["CompositeScore"])
            best_h = int(signal["BestHorizon"])
            pct = float(signal[target_col])
            signal_close = float(closes[r])
        except (TypeError, ValueError):
            continue
        if not np.isfinite(score) or not np.isfinite(pct) or not np.isfinite(signal_close) or signal_close <= 0 or best_h < 1:
            continue
        proposed_target = signal_close * (1.0 + pct)
        if not np.isfinite(proposed_target) or proposed_target <= 0:
            continue
        proposed_horizon_row = r + best_h

        if position is not None:
            # Strictly higher than the currently governing signal; after each
            # accepted update, the governing score becomes this new score.
            if score > position["GoverningScore"]:
                previous_target = position["TargetPrice"]
                previous_horizon_row = position["HorizonEndRow"]
                position["TargetPrice"] = proposed_target
                position["GoverningScore"] = score
                position["GoverningSignalDate"] = date
                # The requested experiment extends the expiry, never shortens it.
                position["HorizonEndRow"] = max(previous_horizon_row, proposed_horizon_row)
                position["HorizonEndDate"] = _date_at(prices, position["HorizonEndRow"])
                pending_order = {
                    "Action": "ADD",
                    "SignalDate": date,
                    "CompositeScore": score,
                    "TargetPrice": proposed_target,
                    "HorizonEndRow": position["HorizonEndRow"],
                }
                # Adjustment is visible immediately in the transaction log even
                # though the additional purchase executes at the next open.
                transactions.append({
                    "Date": date, "Symbol": symbol,
                    "Action": "TARGET_UPDATE_PENDING_ADD",
                    "SignalDate": date, "CompositeScore": score,
                    "Price": np.nan, "Shares": 0, "BuyAmount": np.nan,
                    "CumulativeShares": position["Shares"],
                    "CumulativeInvested": position["TotalCost"],
                    "TargetBefore": previous_target, "TargetAfter": proposed_target,
                    "HorizonBefore": _date_at(prices, previous_horizon_row),
                    "HorizonAfter": position["HorizonEndDate"],
                    "Reason": "higher_composite_score; add at next open",
                    "PnL": np.nan, "Return": np.nan,
                })
            # Lower/equal-scoring signals are simply ignored: no extra buy.
        else:
            # Top-N constrains only brand-new positions, not additions to a
            # position that is already open.
            try:
                rank = int(signal["RankInSession"])
            except (KeyError, TypeError, ValueError):
                rank = top_n + 1
            if rank <= top_n:
                pending_order = {
                    "Action": "BUY",
                    "SignalDate": date,
                    "CompositeScore": score,
                    "TargetPrice": proposed_target,
                    "HorizonEndRow": proposed_horizon_row,
                }

    # End of available data: report any still-open inventory at the latest close.
    if position is not None and n_rows:
        mark_date = pd.Timestamp(prices.iloc[-1]["Date"])
        mark_price = closes[-1]
        shares = int(position["Shares"])
        invested = float(position["TotalCost"])
        market_value = shares * float(mark_price) if np.isfinite(mark_price) else np.nan
        pnl = market_value - invested if np.isfinite(market_value) else np.nan
        ret = pnl / invested if invested and np.isfinite(pnl) else np.nan
        trades.append({
            "Symbol": symbol, "Status": "still_open", "InitialSignalDate": position["InitialSignalDate"],
            "EntryDate": position["EntryDate"], "ExitDate": pd.NaT,
            "InitialCompositeScore": position["InitialScore"],
            "FinalCompositeScore": position["GoverningScore"],
            "BuyCount": position["BuyCount"], "SharesBought": shares,
            "AverageBuyPrice": invested / shares if shares else np.nan,
            "TotalInvested": invested, "ExitPrice": np.nan,
            "TotalProceeds": np.nan, "PnL": pnl, "Return": ret,
            "MarkDate": mark_date, "MarkPrice": mark_price,
            "MarketValue": market_value,
            "InitialTarget": position["InitialTarget"], "FinalTarget": position["TargetPrice"],
            "InitialHorizonEndDate": position["InitialHorizonEndDate"],
            "FinalHorizonEndDate": position["HorizonEndDate"],
            "TargetUpdates": position["TargetUpdates"],
            "FinalSignalDate": position["GoverningSignalDate"],
        })
    return trades, transactions


DYNAMIC_TRADE_COLUMNS = [
    "Symbol", "Status", "InitialSignalDate", "EntryDate", "ExitDate",
    "InitialCompositeScore", "FinalCompositeScore", "BuyCount", "SharesBought",
    "AverageBuyPrice", "TotalInvested", "ExitPrice", "TotalProceeds", "PnL",
    "Return", "MarkDate", "MarkPrice", "MarketValue", "InitialTarget", "FinalTarget",
    "InitialHorizonEndDate", "FinalHorizonEndDate", "TargetUpdates", "FinalSignalDate",
]
DYNAMIC_TRANSACTION_COLUMNS = [
    "Date", "Symbol", "Action", "SignalDate", "CompositeScore", "Price", "Shares",
    "BuyAmount", "CumulativeShares", "CumulativeInvested", "TargetBefore", "TargetAfter",
    "HorizonBefore", "HorizonAfter", "Reason", "PnL", "Return",
]


def simulate_dynamic_strategy(candidates: pd.DataFrame, top_n: int, target_num: int,
                              price_cache: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the dynamic position rules for every ticker in one portfolio-size test."""
    all_trades = []
    all_transactions = []
    if candidates.empty:
        return pd.DataFrame(columns=DYNAMIC_TRADE_COLUMNS), pd.DataFrame(columns=DYNAMIC_TRANSACTION_COLUMNS)

    for symbol, group in candidates.groupby("TechSymbol", sort=False):
        symbol = str(symbol)
        if symbol not in price_cache:
            tech_path = os.path.join(ss.INPUT_FOLDER, f"{symbol}{ss.DATA_SUFFIX}")
            if not os.path.exists(tech_path):
                print(f"  [WARN] Dynamic simulation skipped {symbol}: price file not found: {tech_path}")
                price_cache[symbol] = None
            else:
                try:
                    price_cache[symbol] = pd.read_csv(
                        tech_path, usecols=["Date", "Open", "High", "Close"], parse_dates=["Date"]
                    ).sort_values("Date").reset_index(drop=True)
                except Exception as exc:
                    print(f"  [WARN] Dynamic simulation skipped {symbol}: cannot read OHLC data ({exc})")
                    price_cache[symbol] = None
        prices = price_cache.get(symbol)
        if prices is None or prices.empty:
            continue
        t, e = simulate_dynamic_symbol(symbol, prices, group, target_num, top_n)
        all_trades.extend(t)
        all_transactions.extend(e)

    trades_df = pd.DataFrame(all_trades, columns=DYNAMIC_TRADE_COLUMNS)
    transactions_df = pd.DataFrame(all_transactions, columns=DYNAMIC_TRANSACTION_COLUMNS)
    if not trades_df.empty:
        trades_df = trades_df.sort_values(["InitialSignalDate", "Symbol"], kind="stable")
    if not transactions_df.empty:
        transactions_df = transactions_df.sort_values(["Date", "Symbol"], kind="stable")
    return trades_df, transactions_df


def print_dynamic_summary(trades: pd.DataFrame, target_label: str, top_n: int):
    """Small sanity summary so empty or zero-update runs are obvious."""
    if trades.empty:
        print(f"  DYNAMIC {target_label} top {top_n}: 0 positions; check candidate cache and ticker price files.")
        return
    closed = trades[trades["Status"].isin(["target_hit", "horizon_close"])]
    open_count = int((trades["Status"] == "still_open").sum())
    updates = int(pd.to_numeric(trades.get("TargetUpdates", 0), errors="coerce").fillna(0).sum()) if "TargetUpdates" in trades else 0
    total_pnl = pd.to_numeric(closed.get("PnL", pd.Series(dtype=float)), errors="coerce").sum()
    print(f"  DYNAMIC {target_label} top {top_n}: {len(trades):,} positions, "
          f"{len(closed):,} closed, {open_count:,} still open, "
          f"{updates:,} accepted target updates, realised PnL {total_pnl:+,.2f}")

def main():
    ap = argparse.ArgumentParser(
        description="Backtest fresh bullish signals with independent T1, T2, and T3 targets.")
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
        required_cache_columns = {
            "Direction", "T1_Outcome", "T2_Outcome", "T3_Outcome",
            "TechSymbol", "SignalRow", "BestHorizon", "CompositeScore",
            "T1_TargetPct", "T2_TargetPct", "T3_TargetPct",
        }
        missing_cache_columns = required_cache_columns.difference(cand.columns)
        if missing_cache_columns:
            sys.exit(
                f"[ERROR] {CANDIDATES_FILE} was created by an older script and "
                f"is missing {sorted(missing_cache_columns)}. Run once without "
                "--reuse-candidates to rebuild it."
            )
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
    # Ranking is shared by all target strategies.
    if "Direction" not in cand.columns:
        sys.exit(
            "[ERROR] Candidate data has no Direction column, so bullish-only "
            "trading cannot be guaranteed. Run without --reuse-candidates."
        )

    cand = cand[
        cand["Direction"].astype(str).str.strip().str.casefold() == "bullish"
    ].copy()
    if cand.empty:
        sys.exit("[ERROR] No bullish candidate episodes found in range.")

    cand = cand.sort_values(["SignalDate", "CompositeScore"], ascending=[True, False])
    cand["RankInSession"] = cand.groupby("SignalDate").cumcount() + 1

    top_ns = sorted(set(args.top_n or TOP_N_VALUES))
    if any(n < 1 for n in top_ns):
        sys.exit(f"[ERROR] --top-n values must be >= 1, got {top_ns}")

    os.makedirs("results", exist_ok=True)
    # Only Date/Open/High/Close are cached for the dynamic position simulation.
    # Shared across portfolio sizes and T1/T2/T3 so OHLC files are read once.
    price_cache = {}

    # Baseline and dynamic strategy are run together for each target and top-N.
    for target_num in (1, 2, 3):
        target_label = f"T{target_num}"
        rows = []

        for n in top_ns:
            selected = cand[cand["RankInSession"] <= n].copy()
            sel = strategy_view(selected, target_num)

            # Keep the original filenames for T1. T2 and T3 get separate files.
            if target_num == 1:
                csv_path = TRADES_FILE_TMPL.format(n=n)
            else:
                csv_path = f"results/backtest_t{target_num}_trades_top{n}.csv"

            sel.to_csv(csv_path, index=False)
            print(
                f"  {target_label} top {n:>3}/session: {len(sel):>7,} "
                f"trades -> {csv_path}"
            )

            export_trader_log(sel, n, target_label=target_label)
            rows.append((n, headline_stats(sel)))

            # Dynamic strategy: only new positions are restricted by top-N.
            # Fresh signals for already-held stocks remain eligible for upgrades.
            dynamic_trades, dynamic_transactions = simulate_dynamic_strategy(
                cand, n, target_num, price_cache
            )
            dynamic_trades_path = f"results/backtest_dynamic_t{target_num}_trades_top{n}.csv"
            dynamic_transactions_path = f"results/backtest_dynamic_t{target_num}_transactions_top{n}.csv"
            dynamic_trades.to_csv(dynamic_trades_path, index=False)
            dynamic_transactions.to_csv(dynamic_transactions_path, index=False)
            print(f"  Dynamic {target_label} top {n:>3}: {len(dynamic_trades):>7,} positions "
                  f"-> {dynamic_trades_path}")
            print(f"    Buy/add/sell event log -> {dynamic_transactions_path}")
            print_dynamic_summary(dynamic_trades, target_label, n)

        comparison_table(rows, target_label=target_label)

        if args.detail or len(top_ns) == 1:
            for n in top_ns:
                selected = cand[cand["RankInSession"] <= n].copy()
                summarise(strategy_view(selected, target_num), n, target_label=target_label)

    print(f"\nRe-run with --reuse-candidates to re-select without rescanning "
          f"({len(cand):,} candidates cached).")


if __name__ == "__main__":
    main()