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
  - SignalSinceDate / SignalClose: the date this signal's episode actually
    started (see get_since_index) and that day's Close -- not the date it
    was scanned. In the console report SignalSinceDate is printed as the
    "Date" column, so it reads as "this has been active since X", not "as
    of today".
  - Target1_Price / Target2_Price / Target3_Price: signal_scanner.py's
    Target1_Conservative_BestHorizon / Target2_Median_BestHorizon /
    Target3_Stretch_BestHorizon are historical MFE-quantile *returns*
    (~90% / ~80% / ~70% historical clear rates respectively -- they are
    the P10/P20/P30 of the MFE distribution, see signal_scanner.py's
    MFE_Target_P*_{h}D -- at the horizon this signal scores best at) --
    this script converts them into actual price levels, so they're directly
    usable as a scale-out ladder: Target1 near/high-confidence, Target2 mid,
    Target3 far/stretch. Unlike
    AvgEV-based targets, these are quantiles, not means, so a single
    outlier historical occurrence can't drag them around the way it can
    drag AvgEV.

    They are anchored to SignalClose -- the Close on the day the signal
    first became true -- NOT to today's Close. The underlying quantiles are
    measured from the close of the bar the signal fired on, so the entry
    they describe is that bar's close; anchoring them to today's price
    would silently re-base the whole ladder every day a persistent signal
    stayed active, drifting the targets up with the price the signal was
    supposed to be predicting and making an already-half-captured move look
    like it still has the full historical run ahead of it. With the anchor
    fixed at the firing bar, comparing Close against Target1_Price is a
    real progress check: a signal several days old may already be past its
    own first target.
  - SignalAgeDays / DaysRemaining / MFE_SoFar / TargetsHit / Status: where
    this hit sits between "just fired" and "horizon spent". A signal being
    active today says nothing about whether its target ladder is still
    ahead of it: state-type signals stay true for weeks, so by the time
    they surface here the move their quantiles describe may be partly or
    entirely behind them, or the BestHorizon window they were measured over
    may have run out altogether. These columns make that visible instead of
    leaving every row looking equally fresh:
      SignalAgeDays  -- trading rows since the firing bar (0 = fired today)
      DaysRemaining  -- BestHorizon - SignalAgeDays; goes NEGATIVE once the
                        horizon is spent, which is informative, not an error
      MFE_SoFar      -- favourable excursion actually realized so far, in
                        the same units as Target*_Pct (see get_mfe_so_far)
      TargetsHit     -- how many ladder rungs MFE_SoFar has cleared (0-3)
      Status         -- Fresh / Running / Complete / Expired (see the
                        STATUS_* constants and resolve_status)
    Nothing is filtered on these -- they annotate, and the caller decides.
  - Target1_HitDate: the first session on which the Target1 level was
    actually traded through, or blank if it never was inside BestHorizon.
    This uses a STRICTER definition of "hit" than TargetsHit/MFE_SoFar do,
    and the two can legitimately disagree -- see get_target_hit_date. In
    short: MFE_SoFar counts a momentary wick, this counts only levels that
    fell inside the day's Open-Close body or inside the overnight gap
    between the previous Close and that day's Open. A row can therefore
    show TargetsHit >= 1 with a blank Target1_HitDate, meaning the level was
    brushed by a spike but never traded through in a way you could
    plausibly have been filled at.

Usage:
    python scripts/live_signal_scan.py
    python scripts/live_signal_scan.py --min-composite 5
    python scripts/live_signal_scan.py --min-count 20
    python scripts/live_signal_scan.py --max-ev-median-gap 0.1
    python scripts/live_signal_scan.py --skip-bearish
    python scripts/live_signal_scan.py --min-print-composite 1
    python scripts/live_signal_scan.py --no-excel
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
# Same content as OUTPUT_FILE, split one sheet per direction and set up as a
# working surface (frozen header, autofilter) rather than a flat dump. The CSV
# stays the primary/canonical output -- the workbook is written from it and is
# skipped without failing the run if openpyxl isn't installed.
OUTPUT_XLSX = "results/current_signals.xlsx"


def get_since_index(components: list, signal_map: dict, n_rows: int):
    """Row position at which the currently-active episode of this signal (AND
    of all its components) actually started -- i.e. the first row after the
    most recent row on which at least one component was False, walking back
    from today. Not "today", which is when live_signal_scan.py happened to
    notice it, but when it first became true.

    Returned as a row position rather than a date because callers need two
    things off that row: the date (for reporting) and the Close (which is
    what the target ladder is anchored to -- see scan_symbol).

    None if a component is missing from signal_map or the combined signal
    isn't actually True on the last row -- callers only invoke this after
    the current-state check has already passed, so that's a defensive
    fallback, not the expected path."""
    combined = np.ones(n_rows, dtype=bool)
    for c in components:
        arr = signal_map.get(c)
        if arr is None:
            return None
        combined &= np.asarray(arr, dtype=bool)
    if n_rows == 0 or not combined[-1]:
        return None
    false_positions = np.flatnonzero(~combined)
    return int(false_positions[-1]) + 1 if false_positions.size else 0


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


# Status values for a live hit, in the precedence order resolve_status()
# applies them. A row can genuinely qualify for more than one at once (an
# expired signal that also cleared every target), so the order is pinned here
# rather than left to whatever an if-chain happens to test first.
STATUS_FRESH    = "Fresh"     # fired on the latest bar; no forward rows yet
STATUS_COMPLETE = "Complete"  # the whole ladder has already been delivered
STATUS_EXPIRED  = "Expired"   # BestHorizon spent without completing it
STATUS_RUNNING  = "Running"   # still inside BestHorizon, ladder part-done
STATUS_UNKNOWN  = "Unknown"   # BestHorizon missing -- can't place it at all


def get_mfe_so_far(df: pd.DataFrame, since_idx, direction: str, best_h) -> float:
    """Maximum favourable excursion actually realized since this signal fired,
    as a fraction of the firing bar's Close -- the same quantity (and the same
    reference price) signal_scanner.py's MFE_Target_P* quantiles are measured
    in, so the two are directly comparable without further conversion.

    Deliberately mirrors add_forward_windows() in signal_scanner.py rather
    than approximating it:
      - the window is rows since_idx+1 .. since_idx+best_h, EXCLUDING the
        firing bar itself, because MFE_Bull_{h}D is
        High.rolling(h).max().shift(-h) over the *next* h rows divided by
        that day's Close;
      - it is capped at best_h rows even when the signal has stayed true for
        far longer, so an aged signal cannot keep accumulating excursion past
        the horizon its targets were actually measured over -- without the
        cap, a signal true for 500 sessions would compare a 500-session high
        against a 30-session quantile;
      - it reads High for Bullish and Low for Bearish, since MFE is built
        from intraday extremes; checking Close instead would miss every
        target that was touched and given back within the session.

    NaN when the signal fired on the latest bar (no forward rows exist yet).
    That is a real "not observed", not a zero -- callers must not treat it as
    "no move happened".
    """
    if since_idx is None or pd.isna(best_h):
        return float("nan")
    start = since_idx + 1
    end = min(len(df) - 1, since_idx + int(best_h))
    if start > end:
        return float("nan")
    anchor = df["Close"].iloc[since_idx]
    if pd.isna(anchor) or anchor <= 0:
        return float("nan")
    window = df.iloc[start:end + 1]
    if direction == "Bearish":
        return float(1.0 - window["Low"].min() / anchor)
    return float(window["High"].max() / anchor - 1.0)


def count_targets_hit(mfe_so_far: float, target_pcts) -> tuple:
    """(rungs cleared, rungs available) for this hit's ladder.

    Compared in *return* space against the raw Target*_Pct quantiles, not in
    price space against Target*_Price. That is deliberate: the quantiles are
    unsigned "how far did it move in the favourable direction" numbers for
    both directions, and get_mfe_so_far returns the matching unsigned
    quantity, so a single >= comparison is correct for Bullish and Bearish
    alike. Routing this through the price columns instead would inherit
    whatever sign convention they use.

    Rungs available is returned alongside because a signals file written
    before the target columns existed has none, and "0 of 0 hit" must not be
    allowed to read as a completed ladder.
    """
    available = [float(t) for t in target_pcts if not pd.isna(t)]
    if pd.isna(mfe_so_far) or not available:
        return 0, len(available)
    return sum(1 for t in available if mfe_so_far >= t), len(available)


def resolve_status(age: int, best_h, targets_hit: int, targets_available: int) -> str:
    """Place a hit in its lifecycle bucket, precedence per the STATUS_*
    constants: Fresh, then Complete, then Expired, then Running.

    Complete outranks Expired on purpose -- for a signal that delivered its
    whole ladder and then aged out, "it paid out" is the more useful of the
    two facts, and Expired is left to mean the thing you actually want to
    know: the horizon ran out with the move unfinished.
    """
    if pd.isna(best_h):
        return STATUS_UNKNOWN
    if age == 0:
        return STATUS_FRESH
    if targets_available and targets_hit >= targets_available:
        return STATUS_COMPLETE
    if age > int(best_h):
        return STATUS_EXPIRED
    return STATUS_RUNNING


def get_target_level(direction: str, anchor: float, target_pct) -> float:
    """The actual price a target *return* corresponds to, signed for
    direction: above the anchor for Bullish, below it for Bearish.

    Deliberately does NOT route through the Target*_Price columns. Those
    apply (1 + pct) regardless of direction, which places a Bearish signal's
    targets above its entry -- the level it is supposed to fall to, rendered
    as a level to rise to. Hit detection needs the real level to mean
    anything, so it computes its own here. The consequence is that for
    Bearish rows Target1_HitDate and Target1_Price refer to different
    prices, and will until those columns are corrected.
    """
    if pd.isna(target_pct) or pd.isna(anchor):
        return float("nan")
    if direction == "Bearish":
        return float(anchor) * (1.0 - float(target_pct))
    return float(anchor) * (1.0 + float(target_pct))


def get_target_hit_date(opens, closes, dates, since_idx, best_h, level):
    """First session within BestHorizon on which `level` was actually traded
    through, or NaT if it never was.

    "Traded through" is deliberately a stricter test than the one MFE_SoFar
    and TargetsHit use, and the two can disagree on the same row. MFE_SoFar
    asks only whether the day's High (or Low) reached the level, which counts
    a momentary wick that price immediately left. This asks whether the level
    fell inside one of two ranges price genuinely traversed:

      1. the day's body -- between that session's Open and Close; or
      2. the overnight gap -- between the previous session's Close and this
         session's Open. Without this second case a target that the market
         jumped clean over between sessions would never register as hit,
         because price never printed inside either day's body at that level,
         even though it unambiguously passed through it.

    Wick-only touches are excluded on purpose, so this column can be blank on
    a row whose TargetsHit is already 1. Scanning forward from the firing bar
    and returning on the first match makes this the EARLIEST such session,
    not the most recent.

    Takes preconverted numpy arrays rather than the DataFrame because it is
    called once per scored row per symbol; re-running .to_numpy() on the full
    price history inside that loop was pure waste.
    """
    if since_idx is None or pd.isna(best_h) or pd.isna(level):
        return pd.NaT
    start = since_idx + 1
    end = min(len(closes) - 1, since_idx + int(best_h))
    for d in range(start, end + 1):
        o, c, prev_c = opens[d], closes[d], closes[d - 1]
        if np.isnan(o) or np.isnan(c):
            continue
        if min(o, c) <= level <= max(o, c):
            return dates.iloc[d]
        if not np.isnan(prev_c) and min(prev_c, o) <= level <= max(prev_c, o):
            return dates.iloc[d]
    return pd.NaT


def scan_symbol(tech_path: str, signals_path: str, skip_bearish: bool = False) -> list:
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
    if skip_bearish:
        # Filtered here, before the per-row loop, rather than at the end of
        # main(): get_since_index walks the full price history once per
        # surviving row, so this is the earliest point the work can be
        # dropped. Do not expect a large speedup though -- measured across
        # the full universe it saves ~12% (694s -> 609s) even though it
        # removes ~43% of the hits, because compute_all_signals() below
        # recomputes every signal over the stock's whole history once per
        # symbol and is charged whether or not any bearish row survives.
        # That call, not the per-row walk, is what dominates this script.
        scored = scored[scored["Direction"] != "Bearish"]
        if scored.empty:
            return []

    # Values in signal_map are a mix of pandas Series (atomic signals, via
    # the _safe() wrapper in signal_scanner.py) and raw numpy arrays
    # (candlestick patterns from compute_candle_signals, since TA-Lib's
    # functions return ndarrays and boolean comparisons on them stay
    # ndarrays, not Series). np.asarray(...) handles both uniformly instead
    # of assuming .iloc exists. Kept as full arrays (not just the last row)
    # so get_since_index can walk back through history below.
    signal_map, _ = ss.compute_all_signals(df)
    current_state = {name: bool(np.asarray(series)[-1]) for name, series in signal_map.items()}
    last_date, last_close = df["Date"].iloc[-1], df["Close"].iloc[-1]
    atr_pct_now = get_current_atr_pct(df)
    # Converted once per symbol, not once per scored row -- get_target_hit_date
    # walks these for every hit and would otherwise rebuild them each time.
    opens_arr  = df["Open"].to_numpy(dtype=float)
    closes_arr = df["Close"].to_numpy(dtype=float)

    hits = []
    for _, row in scored.iterrows():
        components = [c.strip() for c in str(row["Components"]).split("+")]
        if not components or not all(current_state.get(c, False) for c in components):
            continue

        # The bar this signal actually fired on -- both the date to report it
        # under and, more importantly, the Close the target ladder is anchored
        # to (see module docstring).
        since_idx = get_since_index(components, signal_map, len(df))
        if since_idx is None:
            # Defensive only: the current-state check above has already passed,
            # so the combined signal is true on the last row by construction.
            # Fall back to today's bar rather than dropping an otherwise-valid
            # hit, and leave the date empty so the fallback is visible.
            since_date, signal_close = pd.NaT, last_close
        else:
            since_date = df["Date"].iloc[since_idx]
            signal_close = df["Close"].iloc[since_idx]

        avg_ev = row["AvgEV"]
        if not pd.isna(atr_pct_now) and atr_pct_now > 1e-9 and not pd.isna(avg_ev):
            atr_adjusted_ev = round(float(avg_ev) / atr_pct_now, 3)
        else:
            atr_adjusted_ev = np.nan

        # Convert the historical MFE-quantile target *returns* (from the
        # horizon this signal scores best at) into actual price levels off
        # signal_close -- the Close of the bar the signal fired on, which is
        # the same reference point those quantiles were measured from in
        # signal_scanner.py (MFE_Bull/Bear_{h}D are both ratios against that
        # day's Close). Using today's Close instead would re-anchor the ladder
        # every day a persistent signal stayed true. NaN-safe: a signal can
        # legitimately have no target columns if it predates this fix's
        # per-symbol re-run.
        def _target_price(col):
            val = row.get(col, np.nan)
            if pd.isna(val) or pd.isna(signal_close):
                return np.nan
            return round(float(signal_close) * (1.0 + float(val)), 2)

        target1_price = _target_price("Target1_Conservative_BestHorizon")
        target2_price = _target_price("Target2_Median_BestHorizon")
        target3_price = _target_price("Target3_Stretch_BestHorizon")

        # Lifecycle annotation -- see the module docstring. Age is counted in
        # TRADING ROWS, not calendar days, to stay commensurable with
        # BestHorizon: signal_scanner.py builds its forward windows with
        # .shift(-h) over rows, so a horizon of 30 means 30 sessions, and
        # differencing the dates instead would make every signal look older
        # than it is by roughly the weekends and holidays it spans.
        best_h = row.get("BestHorizon", np.nan)
        age = (len(df) - 1 - since_idx) if since_idx is not None else 0
        days_remaining = (int(best_h) - age) if not pd.isna(best_h) else np.nan
        mfe_so_far = get_mfe_so_far(df, since_idx, row["Direction"], best_h)
        targets_hit, targets_available = count_targets_hit(
            mfe_so_far,
            (row.get("Target1_Conservative_BestHorizon", np.nan),
             row.get("Target2_Median_BestHorizon", np.nan),
             row.get("Target3_Stretch_BestHorizon", np.nan)),
        )
        status = resolve_status(age, best_h, targets_hit, targets_available)

        # When T1 was actually traded through (body or overnight gap), as
        # opposed to merely brushed by a wick -- see get_target_hit_date.
        target1_level = get_target_level(
            row["Direction"], signal_close,
            row.get("Target1_Conservative_BestHorizon", np.nan))
        target1_hit_date = get_target_hit_date(
            opens_arr, closes_arr, df["Date"], since_idx, best_h, target1_level)

        hits.append({
            "Symbol":            row["Symbol"],
            "Date":              last_date,
            "SignalSinceDate":   since_date,
            "Close":             last_close,
            "SignalClose":       signal_close,
            "SignalAgeDays":     age,
            "DaysLeft":     days_remaining,
            "MFE_SoFar":         round(mfe_so_far, 5) if not pd.isna(mfe_so_far) else np.nan,
            "TargetsHit":        targets_hit,
            "Status":            status,
            "Target1_HitDate":   target1_hit_date,
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


def write_excel(hits_df: pd.DataFrame, path: str) -> bool:
    """Write the scan to a workbook, one sheet per direction, with the header
    row frozen and an autofilter across it so the sheet can be sorted and
    filtered in place -- the whole point of having it in Excel rather than
    just the CSV.

    Returns False after a warning, rather than raising, when openpyxl isn't
    installed: the CSV is the canonical output and has already been written by
    the time this runs, so a missing optional dependency must not throw away a
    completed scan.
    """
    try:
        from openpyxl.utils import get_column_letter
    except ImportError:
        print(f"  [WARNING] openpyxl not installed -- skipped {path} "
              f"(the CSV above is unaffected). Install with: pip install openpyxl")
        return False

    # Excel has no pandas Timestamp; writing date-only values keeps these as
    # plain dates rather than every cell reading as midnight-stamped datetimes.
    out = hits_df.copy()
    for col in ["Date", "SignalSinceDate", "Target1_HitDate"]:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col]).dt.date

    sheets = [(d, out[out["Direction"] == d]) for d in ["Bullish", "Bearish"]]
    sheets = [(name, frame) for name, frame in sheets if not frame.empty]
    if not sheets:
        # Defensive: Direction should only ever hold these two values, but an
        # empty workbook is a file Excel refuses to open, so fall back to one
        # sheet holding everything rather than writing something unopenable.
        sheets = [("Signals", out)]

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, frame in sheets:
            frame.to_excel(writer, sheet_name=name, index=False)
            ws = writer.sheets[name]
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            for i, col in enumerate(frame.columns, start=1):
                # Sized off the header plus a 200-row sample: measuring every
                # cell of an 11k-row frame to pick a column width costs more
                # than the slightly better fit is worth.
                sample = frame[col].head(200).astype(str).map(len).max()
                width = max(len(str(col)), int(sample) if pd.notna(sample) else 0) + 2
                ws.column_dimensions[get_column_letter(i)].width = min(width, 40)
    return True


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
    parser.add_argument("--min-print-composite", type=float, default=3.0,
                         help="Console report prints every hit with CompositeScore > this "
                              "(default 3.0) instead of a fixed top-N. Affects the printed "
                              "report only -- the CSV always gets everything that survived "
                              "the filters above. Pass a negative number to print all of it.")
    parser.add_argument("--no-excel", action="store_true",
                         help=f"Skip writing {OUTPUT_XLSX}. The workbook holds the same rows as "
                              f"the CSV, one sheet per direction; writing it adds a few seconds "
                              f"to an otherwise ~10-minute scan.")
    parser.add_argument("--skip-bearish", action="store_true",
                         help="Skip Bearish signals entirely. Saves only ~12%% of wall-clock "
                              "time (measured 694s -> 609s over 2136 symbols) despite removing "
                              "~43%% of the hits, because the per-symbol compute_all_signals() "
                              "call dominates this script and is paid either way -- use it to "
                              "cut the output down to bullish-only, not as a speed fix.")
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
        all_hits.extend(scan_symbol(tech_path, sig_path, skip_bearish=args.skip_bearish))
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
    # SignalSinceDate/SignalClose are now saved too, not console-only as
    # SignalSinceDate used to be: the Target*_Price columns are anchored to
    # SignalClose rather than to Close, so without them the CSV would carry a
    # target ladder that can't be reconciled against any price in the file.
    hits_df.to_csv(OUTPUT_FILE, index=False)
    as_of = hits_df["Date"].max()
    print(f"\n{len(hits_df)} currently-active validated signals (as of {as_of.date()}) -> {OUTPUT_FILE}")
    # Lifecycle breakdown up front: "active today" and "still has its move
    # ahead of it" are very different populations, and the split between them
    # is the first thing worth knowing about a scan.
    status_counts = hits_df["Status"].value_counts()
    print("  by status: " + ", ".join(f"{k} {v}" for k, v in status_counts.items()))

    if not args.no_excel and write_excel(hits_df, OUTPUT_XLSX):
        print(f"{len(hits_df)} rows also written to {OUTPUT_XLSX} "
              f"(one sheet per direction, filterable)")

    # SignalClose sits next to Close so the gap between "price when it fired"
    # and "price now" is readable at a glance -- that gap is how much of the
    # target ladder a still-active signal has already travelled.
    # SignalAgeDays / TargetsHit / MFE_SoFar are computed and saved to the CSV
    # but deliberately not printed -- Status and Target1_HitDate carry the same
    # story in less width.
    cols = ["Symbol", "Date", "SignalName", "Status", "DaysLeft", "Target1_HitDate",
            "Count", "CompositeScore", "SignalClose", "Close",
            "Target1_Price", "Target2_Price", "Target3_Price", "BestHorizon"]
    directions = ["Bullish"] if args.skip_bearish else ["Bullish", "Bearish"]
    for direction in directions:
        sub = hits_df[(hits_df["Direction"] == direction) &
                      (hits_df["CompositeScore"] > args.min_print_composite)].copy()
        print(f"\n── {len(sub)} Active {direction} Signals "
              f"(CompositeScore > {args.min_print_composite}) ──")
        if sub.empty:
            print("  (none)")
            continue
        # Display-only: show when the signal's episode actually started,
        # not today's scan date (which is the same for every row).
        sub["Date"] = sub["SignalSinceDate"].dt.date
        # to_datetime first: an all-NaT column can land as object dtype, on
        # which .dt would raise rather than just print blanks.
        sub["Target1_HitDate"] = pd.to_datetime(sub["Target1_HitDate"]).dt.date
        print(sub[cols].to_string(index=False))


if __name__ == "__main__":
    main()