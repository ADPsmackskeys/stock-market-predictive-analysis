"""
Live Signal Scan — which already-validated signals/strategies are firing
right now, across the whole universe.

signal_scanner.py answers "historically, how good is signal X for stock Y" and
writes each stock's top 25 bullish + top 25 bearish signals to
results/signals_v2/<SYMBOL>_data_signals.csv. This script answers the other
half: "of those already-validated signals, which ones are true *today*?" It
recomputes each atomic signal's boolean value on the most recent row of
data/technical/<SYMBOL>_data.csv (no backtesting, no scoring -- just today's
state), then checks whether every component listed in a scored signal's
"Components" field is true right now. A match is a signal that both (a) has
a known historical track record for that stock and (b) is actionable today.

Columns added beyond what signal_scanner.py stores per signal:
  - ATR_Adjusted_EV: AvgEV expressed in units of *today's* ATR% for that
    stock (AvgEV / (ATR_14/Close) as of the most recent row), not the ATR%
    from whenever each historical occurrence happened. AvgEV alone isn't
    comparable across stocks -- a 5% AvgEV means something very different
    for a low-vol large-cap than for a stock whose ATR% is routinely 5%+ on
    its own. Expressing it as "how many of this stock's *current* typical
    daily ranges does the historical edge amount to" makes it comparable,
    and reflects the volatility regime the stock is actually in right now.
    NOTE: this is a MULTIPLE of today's ATR%, not a percentage itself --
    to turn it back into a return, multiply it by today's ATR% again.
  - AvgWinRate_HitEV: pass-through of signal_scanner.py's separate win-rate
    definition -- fraction of historical occurrences whose actual return
    cleared the signal's own AvgEV (as opposed to AvgWinRate, which is
    MFE-threshold-based). A low AvgWinRate_HitEV next to a high AvgWinRate
    is a tell that AvgEV is being carried by a few big occurrences rather
    than being a typical outcome.
  - Target1_Price / Target2_Price / Target3_Price: signal_scanner.py's
    Target1_Conservative_BestHorizon / Target2_Median_BestHorizon /
    Target3_Stretch_BestHorizon are historical MFE-quantile *returns*
    (~80% / ~50% / ~20% historical clear rates respectively, at the
    horizon this signal scores best at) -- this script converts them into
    actual price levels using *today's* Close, so they're directly usable
    as a scale-out ladder: Target1 near/high-confidence, Target2 mid,
    Target3 far/stretch. Unlike AvgEV-based targets, these are quantiles,
    not means, so a single outlier historical occurrence can't drag them
    around the way it can drag AvgEV.

Usage:
    python scripts/live_signal_scan.py
    python scripts/live_signal_scan.py --min-composite 5
    python scripts/live_signal_scan.py --min-count 20
    python scripts/live_signal_scan.py --max-ev-median-gap 0.1
"""

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import signal_scanner as ss

OUTPUT_FILE = "results/current_signals.csv"


def get_current_signal_state(df: pd.DataFrame) -> dict:
    """Boolean value of every atomic/candle signal on the most recent row.

    Values in signal_map are a mix of pandas Series (atomic signals, via the
    _safe() wrapper in signal_scanner.py) and raw numpy arrays (candlestick
    patterns from compute_candle_signals, since TA-Lib's functions return
    ndarrays and boolean comparisons on them stay ndarrays, not Series).
    np.asarray(...)[-1] handles both uniformly instead of assuming .iloc
    exists."""
    signal_map, _ = ss.compute_all_signals(df)
    return {name: bool(np.asarray(series)[-1]) for name, series in signal_map.items()}


def get_current_atr_pct(df: pd.DataFrame) -> float:
    """Today's ATR as a fraction of today's Close (e.g. 0.03 = 3%), for
    normalizing historical AvgEV to the stock's *current* volatility regime.
    NaN if ATR_14 isn't available or Close is zero/missing -- callers must
    treat that as "can't compute," not as "zero volatility."""
    if "ATR_14" not in df.columns or df.empty:
        return float("nan")
    last_close = df["Close"].iloc[-1]
    last_atr = df["ATR_14"].iloc[-1]
    if pd.isna(last_close) or pd.isna(last_atr) or last_close <= 0:
        return float("nan")
    return float(last_atr) / float(last_close)


def scan_symbol(tech_path: str, signals_path: str) -> list:
    try:
        df = pd.read_csv(tech_path, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    except Exception:
        return []
    if len(df) < 60:
        return []
    try:
        scored = pd.read_csv(signals_path)
    except Exception:
        return []
    if scored.empty:
        return []

    current_state = get_current_signal_state(df)
    last_date, last_close = df["Date"].iloc[-1], df["Close"].iloc[-1]
    atr_pct_now = get_current_atr_pct(df)

    hits = []
    for _, row in scored.iterrows():
        components = [c.strip() for c in str(row["Components"]).split("+")]
        if not components or not all(current_state.get(c, False) for c in components):
            continue

        avg_ev = row["AvgEV"]
        if not pd.isna(atr_pct_now) and atr_pct_now > 1e-9 and not pd.isna(avg_ev):
            atr_adjusted_ev = round(float(avg_ev) / atr_pct_now, 3)
        else:
            atr_adjusted_ev = np.nan

        # Convert the historical MFE-quantile target *returns* (from the
        # horizon this signal scores best at) into actual price levels using
        # today's Close. NaN-safe: a signal can legitimately have no target
        # columns if it predates this fix's per-symbol re-run.
        def _target_price(col):
            val = row.get(col, np.nan)
            if pd.isna(val) or pd.isna(last_close):
                return np.nan
            return round(float(last_close) * (1.0 + float(val)), 2)

        target1_price = _target_price("Target1_Conservative_BestHorizon")
        target2_price = _target_price("Target2_Median_BestHorizon")
        target3_price = _target_price("Target3_Stretch_BestHorizon")

        hits.append({
            "Symbol":            row["Symbol"],
            "Date":              last_date,
            "Close":             last_close,
            "SignalName":        row["SignalName"],
            "Direction":         row["Direction"],
            "HorizonClass":      row["HorizonClass"],
            "BestHorizon":       row.get("BestHorizon"),
            "Count":             row["Count"],
            "CompositeScore":    row["CompositeScore"],
            "AvgEV":             avg_ev,
            "ATR_Adjusted_EV":   atr_adjusted_ev,
            "AvgMedianReturn":   row.get("AvgMedianReturn", np.nan),
            "AvgWinRate":        row["AvgWinRate"],
            "AvgWinRate_HitEV":  row.get("AvgWinRate_HitEV", np.nan),
            "MaxTop1TradeShare": row.get("MaxTop1TradeShare", np.nan),
            "Concentration_Flag": row.get("Concentration_Flag", np.nan),
            "Target1_Pct":       row.get("Target1_Conservative_BestHorizon", np.nan),
            "Target2_Pct":       row.get("Target2_Median_BestHorizon", np.nan),
            "Target3_Pct":       row.get("Target3_Stretch_BestHorizon", np.nan),
            "Target1_Price":     target1_price,
            "Target2_Price":     target2_price,
            "Target3_Price":     target3_price,
        })
    return hits


def main():
    parser = argparse.ArgumentParser(description="Scan for validated signals currently active across the universe")
    parser.add_argument("--min-composite", type=float, default=None, help="Only keep hits with CompositeScore >= this")
    parser.add_argument("--min-count",     type=int,   default=None, help="Only keep hits with historical Count >= this")
    parser.add_argument("--max-ev-median-gap", type=float, default=0.1,
                         help="Only keep hits where |AvgEV - AvgMedianReturn| <= this (default 0.1). "
                              "A wide gap means AvgEV is being pulled around by a handful of outlier "
                              "occurrences rather than reflecting a typical one -- pass a negative number "
                              "(e.g. -1) to disable this filter entirely.")
    parser.add_argument("--exclude-concentrated", action="store_true",
                         help="Also drop hits whose Concentration_Flag is set (edge riding on a single trade)")
    parser.add_argument("--top",           type=int,   default=30,   help="Rows to print per direction")
    args = parser.parse_args()

    # signal_scanner.py names outputs "{symbol}_data_signals.csv" (Path(...).stem
    # keeps the "_data" from the input filename, then "_signals.csv" is appended).
    signal_files = sorted(glob.glob(os.path.join(ss.OUTPUT_FOLDER, "*_data_signals.csv")))
    print(f"Scanning {len(signal_files)} symbols for currently-active signals...")

    all_hits = []
    for i, sig_path in enumerate(signal_files, 1):
        symbol = os.path.basename(sig_path).replace("_data_signals.csv", "")
        tech_path = os.path.join(ss.INPUT_FOLDER, f"{symbol}{ss.DATA_SUFFIX}")
        if not os.path.exists(tech_path):
            continue
        all_hits.extend(scan_symbol(tech_path, sig_path))
        if i % 300 == 0:
            print(f"  [{i}/{len(signal_files)}] scanned, {len(all_hits)} hits so far")

    if not all_hits:
        print("No currently-active validated signals found.")
        return

    hits_df = pd.DataFrame(all_hits)
    if args.min_composite is not None:
        hits_df = hits_df[hits_df["CompositeScore"] >= args.min_composite]
    if args.min_count is not None:
        hits_df = hits_df[hits_df["Count"] >= args.min_count]

    if args.exclude_concentrated:
        n_before = len(hits_df)
        hits_df = hits_df[hits_df["Concentration_Flag"].fillna(0) != 1]
        n_dropped = n_before - len(hits_df)
        if n_dropped:
            print(f"  (dropped {n_dropped} hit(s) with Concentration_Flag set)")

    if args.max_ev_median_gap is not None and args.max_ev_median_gap >= 0:
        # AvgMedianReturn can be NaN for a signal that had no horizon clear
        # MIN_OCCURRENCES cleanly (see score_signal in signal_scanner.py) --
        # treat "can't compute the gap" as failing the filter rather than
        # silently passing it.
        ev_median_gap = (hits_df["AvgEV"] - hits_df["AvgMedianReturn"]).abs()
        n_before = len(hits_df)
        hits_df = hits_df[ev_median_gap <= args.max_ev_median_gap]
        n_dropped = n_before - len(hits_df)
        if n_dropped:
            print(f"  (dropped {n_dropped} hit(s) with |AvgEV - AvgMedianReturn| > "
                  f"{args.max_ev_median_gap} or unavailable AvgMedianReturn)")

    if hits_df.empty:
        print("No currently-active validated signals remain after filtering.")
        return

    hits_df = hits_df.sort_values("CompositeScore", ascending=False)
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    hits_df.to_csv(OUTPUT_FILE, index=False)
    as_of = hits_df["Date"].max()
    print(f"\n{len(hits_df)} currently-active validated signals (as of {as_of.date()}) -> {OUTPUT_FILE}")

    cols = ["Symbol", "SignalName", "HorizonClass", "Count", "CompositeScore",
            "AvgEV", "ATR_Adjusted_EV", "AvgWinRate", "AvgWinRate_HitEV",
            "Close", "Target1_Price", "Target2_Price", "Target3_Price", "BestHorizon"]
    for direction in ["Bullish", "Bearish"]:
        sub = hits_df[hits_df["Direction"] == direction].head(args.top)
        print(f"\n── Top {len(sub)} Active {direction} Signals ──")
        print(sub[cols].to_string(index=False))


if __name__ == "__main__":
    main()