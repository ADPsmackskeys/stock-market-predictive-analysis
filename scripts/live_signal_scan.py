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

Usage:
    python scripts/live_signal_scan.py
    python scripts/live_signal_scan.py --min-composite 5
    python scripts/live_signal_scan.py --min-count 20
"""

import os
import sys
import glob
import argparse
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import signal_scanner as ss

OUTPUT_FILE = "results/current_signals.csv"


def get_current_signal_state(df: pd.DataFrame) -> dict:
    """Boolean value of every atomic/candle signal on the most recent row."""
    signal_map, _ = ss.compute_all_signals(df)
    return {name: bool(series.iloc[-1]) for name, series in signal_map.items()}


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

    hits = []
    for _, row in scored.iterrows():
        components = [c.strip() for c in str(row["Components"]).split("+")]
        if not components or not all(current_state.get(c, False) for c in components):
            continue
        hits.append({
            "Symbol":         row["Symbol"],
            "Date":           last_date,
            "Close":          last_close,
            "SignalName":     row["SignalName"],
            "Direction":      row["Direction"],
            "HorizonClass":   row["HorizonClass"],
            "BestHorizon":    row.get("BestHorizon"),
            "Count":          row["Count"],
            "CompositeScore": row["CompositeScore"],
            "AvgEV":          row["AvgEV"],
            "AvgWinRate":     row["AvgWinRate"],
        })
    return hits


def main():
    parser = argparse.ArgumentParser(description="Scan for validated signals currently active across the universe")
    parser.add_argument("--min-composite", type=float, default=None, help="Only keep hits with CompositeScore >= this")
    parser.add_argument("--min-count",     type=int,   default=None, help="Only keep hits with historical Count >= this")
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

    hits_df = hits_df.sort_values("CompositeScore", ascending=False)
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    hits_df.to_csv(OUTPUT_FILE, index=False)
    as_of = hits_df["Date"].max()
    print(f"\n{len(hits_df)} currently-active validated signals (as of {as_of.date()}) -> {OUTPUT_FILE}")

    cols = ["Symbol", "SignalName", "HorizonClass", "Count", "CompositeScore", "AvgEV", "AvgWinRate", "BestHorizon"]
    for direction in ["Bullish", "Bearish"]:
        sub = hits_df[hits_df["Direction"] == direction].head(args.top)
        print(f"\n── Top {len(sub)} Active {direction} Signals ──")
        print(sub[cols].to_string(index=False))


if __name__ == "__main__":
    main()
