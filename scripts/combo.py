"""
Per-symbol pipeline:
- Read raw per-symbol data from data/technical/<SYMBOL>_data.csv
- Compute classified candlestick pattern detections and indicator states
- Produce per-symbol SIGNAL CSV and performance CSV
- For each symbol only: analyze combos where bullish patterns pair with bullish indicator states
  and bearish patterns with bearish indicator states
- Save per-symbol combo CSVs per-indicator and a per-symbol combined CSV
"""

import os
import glob
import pandas as pd
import numpy as np
from scipy.stats import fisher_exact
from pathlib import Path

# ---------------------------
# CONFIG - tweak these
# ---------------------------
INPUT_DATA_FOLDER = "data/technical"   # folder containing SYMBOL_data.csv files
DATA_SUFFIX = "_data.csv"
OUTPUT_FOLDER = "results"              # where *_signals.csv and performance_*.csv go
COMBO_OUTPUT_FOLDER = os.path.join(OUTPUT_FOLDER, "combos")  # per-symbol combo files
Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)
Path(COMBO_OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

# Candlestick pattern classification: user-specified grouping (Bullish vs Bearish)
import talib
CANDLE_CLASSIFIED = {
    # Bullish patterns (we will still check sign from TA-Lib: val>0 => bullish occurrence)
    "CDL_BULLISH_ENGULFING": talib.CDLENGULFING,
    "CDL_PIERCING": talib.CDLPIERCING,
    "CDL_HARAMI": talib.CDLHARAMI,
    "CDL_MORNING_STAR": talib.CDLMORNINGSTAR,
    "CDL_THREE_WHITE_SOLDIERS": talib.CDL3WHITESOLDIERS,
    "CDL_THREE_INSIDE_UP": talib.CDL3INSIDE,       # TA-Lib function reused; naming kept per your mapping
    "CDL_THREE_OUTSIDE_UP": talib.CDL3OUTSIDE,
    "CDL_HAMMER": talib.CDLHAMMER,
    "CDL_INVERTED_HAMMER": talib.CDLINVERTEDHAMMER,

    # Bearish patterns
    "CDL_DARKCLOUDCOVER": talib.CDLDARKCLOUDCOVER,
    "CDL_EVENING_STAR": talib.CDLEVENINGSTAR,
    "CDL_THREE_BLACK_CROWS": talib.CDL3BLACKCROWS,
    "CDL_SHOOTING_STAR": talib.CDLSHOOTINGSTAR,
    "CDL_HARAMI_CROSS": talib.CDLHARAMICROSS,
    "CDL_ADVANCE_BLOCK": talib.CDLADVANCEBLOCK,
}

# Indicators present in per-symbol raw data
INDICATOR_COLS = ["EMA_Trend", "MACD_Trend", "RSI_State", "CMF_State"]

# Horizon(s) to consider (use columns Move_1D / Direction_1D etc.)
HORIZONS = [1, 2, 3]

# Which indicator states we treat as "bullish" vs "bearish"
BULLISH_STATES = {
    "EMA_Trend": ["Bullish"],
    "MACD_Trend": ["Bullish"],
    # RSI: treat Oversold as bullish (mean-reversion). Change if you want momentum interpretation.
    "RSI_State": ["Oversold"],
    "CMF_State": ["Positive"],
}
BEARISH_STATES = {
    "EMA_Trend": ["Bearish"],
    "MACD_Trend": ["Bearish"],
    "RSI_State": ["Overbought"],
    "CMF_State": ["Negative"],
}

# Combo analysis options
MIN_COUNT = 10                # minimum total occurrences (sum across horizons) to consider a combo
MIN_COUNT_PER_HORIZON = False # require min_count for every horizon if True
TOP_K_PER_INDICATOR = 3       # keep top K combos per indicator for a symbol (for fisher optionally)
COMPUTE_FISHER = False        # whether to compute Fisher exact p-values (slow if True)
VERBOSE = True

# ---------------------------
# Utility helpers
# ---------------------------

def safe_read_csv(path):
    try:
        return pd.read_csv(path, parse_dates=["Date"])
    except Exception as e:
        if VERBOSE:
            print(f"Failed to read {path}: {e}")
        return None

def save_df_csv(df, path):
    try:
        df.to_csv(path, index=False)
    except Exception as e:
        print(f"Failed to save {path}: {e}")

# ---------------------------
# Stage 1: build signals + performance for one symbol
# ---------------------------

def build_symbol_signals_and_performance(filepath, candlestick_map, indicator_cols, horizons):
    """
    Loads raw symbol CSV, computes candlestick indicators, indicator states,
    Move_{n}D and Direction_{n}D, then produces:
      - signals_df: one row per detected signal (candlestick OR indicator)
      - performance_df: aggregated Pattern/SignalType per horizon
    Returns (symbol, signals_df, performance_df)
    """
    df = safe_read_csv(filepath)
    if df is None or df.empty:
        return None, None, None

    df = df.sort_values("Date").reset_index(drop=True)
    base = os.path.basename(filepath)
    symbol_from_file = base.replace(DATA_SUFFIX, "")
    symbol = df["Symbol"].iloc[0] if "Symbol" in df.columns and not df["Symbol"].isnull().all() else symbol_from_file
    company = df["Company"].iloc[0] if "Company" in df.columns and not df["Company"].isnull().all() else ""

    if len(df) < 30:
        if VERBOSE:
            print(f"Skipping {symbol}: insufficient rows ({len(df)})")
        return symbol, None, None

    # compute TA-Lib candlestick function outputs
    for pname, func in candlestick_map.items():
        try:
            df[pname] = func(df["Open"], df["High"], df["Low"], df["Close"])
        except Exception:
            # if TA-Lib fails (unexpected), fallback to zeros
            df[pname] = 0

    # compute indicator states (matching your previous logic)
    if "EMA_20" in df.columns and "EMA_50" in df.columns:
        df["EMA_Trend"] = np.where(df["EMA_20"] > df["EMA_50"], "Bullish", "Bearish")
    else:
        df["EMA_Trend"] = np.nan

    if "MACD" in df.columns and "MACD_Signal" in df.columns:
        df["MACD_Trend"] = np.where(df["MACD"] > df["MACD_Signal"], "Bullish", "Bearish")
    else:
        df["MACD_Trend"] = np.nan

    if "RSI_14" in df.columns:
        df["RSI_State"] = np.where(df["RSI_14"] > 70, "Overbought",
                                   np.where(df["RSI_14"] < 30, "Oversold", "Neutral"))
    else:
        df["RSI_State"] = np.nan

    if "CMF_20" in df.columns:
        df["CMF_State"] = np.where(df["CMF_20"] > 0, "Positive", "Negative")
    else:
        df["CMF_State"] = np.nan

    # compute Move_nD and Direction_nD
    max_h = max(horizons)
    for n in horizons:
        df[f"Move_{n}D"] = df["Close"].shift(-n) / df["Close"] - 1
        df[f"Direction_{n}D"] = df[f"Move_{n}D"].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

    # build signals list: candlestick detections and (optionally) indicator-only rows
    pattern_cols = list(candlestick_map.keys())
    signals_list = []

    # skip last max_h rows because of lookahead NaN
    for idx, row in df.iloc[:-max_h].iterrows():
        date = row["Date"]
        # candlestick detections
        for pattern in pattern_cols:
            val = row.get(pattern, 0)
            if pd.isna(val):
                continue
            if val != 0:
                # val>0 generally indicates bullish occurrence; val<0 bearish
                signal_type = "Bullish" if val > 0 else "Bearish"
                signals_list.append({
                    "Date": date,
                    "SignalSource": "Candlestick",
                    "Pattern": pattern,
                    "SignalType": signal_type,
                    "EMA_Trend": row.get("EMA_Trend", np.nan),
                    "MACD_Trend": row.get("MACD_Trend", np.nan),
                    "RSI_State": row.get("RSI_State", np.nan),
                    "CMF_State": row.get("CMF_State", np.nan),
                    **{f"Move_{h}D": row.get(f"Move_{h}D", np.nan) for h in horizons},
                    **{f"Direction_{h}D": row.get(f"Direction_{h}D", np.nan) for h in horizons},
                })

        # optionally also record indicator-only signals (kept for completeness)
        for ind in indicator_cols:
            val = row.get(ind, None)
            if pd.isna(val):
                continue
            if val in ["Bullish", "Bearish", "Oversold", "Overbought", "Positive", "Negative"]:
                signals_list.append({
                    "Date": date,
                    "SignalSource": "Indicator",
                    "Pattern": ind,
                    "SignalType": val,
                    "EMA_Trend": row.get("EMA_Trend", np.nan),
                    "MACD_Trend": row.get("MACD_Trend", np.nan),
                    "RSI_State": row.get("RSI_State", np.nan),
                    "CMF_State": row.get("CMF_State", np.nan),
                    **{f"Move_{h}D": row.get(f"Move_{h}D", np.nan) for h in horizons},
                    **{f"Direction_{h}D": row.get(f"Direction_{h}D", np.nan) for h in horizons},
                })

    if not signals_list:
        if VERBOSE:
            print(f"No signals for symbol {symbol}")
        return symbol, None, None

    signals_df = pd.DataFrame(signals_list)
    signals_df["Symbol"] = symbol
    signals_df["Company"] = company

    # compute per-row SuccessFlag_{h}D (vectorized)
    for h in horizons:
        dir_col = f"Direction_{h}D"
        success_col = f"SuccessFlag_{h}D"
        if dir_col in signals_df.columns:
            signals_df[success_col] = np.where(
                (signals_df["SignalType"] == "Bullish") & (signals_df[dir_col] == 1), 1,
                np.where((signals_df["SignalType"] == "Bearish") & (signals_df[dir_col] == -1), 1, 0)
            )
        else:
            signals_df[success_col] = np.nan

    # Compute per-stock performance (pattern-level) for each horizon
    perf_frames = []
    for h in horizons:
        move_col = f"Move_{h}D"
        success_col = f"SuccessFlag_{h}D"
        grp = signals_df.groupby(["SignalSource", "Pattern", "SignalType"]).agg(
            SuccessRate=(success_col, "mean"),
            AvgReturn=(move_col, "mean"),
            Count=(move_col, "count")
        ).reset_index()
        grp["Horizon"] = f"{h}D"
        grp["Weight"] = grp["SuccessRate"] * grp["AvgReturn"]
        perf_frames.append(grp)

    performance_df = pd.concat(perf_frames, axis=0).sort_values("Weight", ascending=False)

    # save per-symbol outputs
    out_signals = os.path.join(OUTPUT_FOLDER, f"{symbol}_signals.csv")
    out_perf = os.path.join(OUTPUT_FOLDER, f"performance_{symbol}.csv")
    save_df_csv(signals_df, out_signals)
    save_df_csv(performance_df, out_perf)
    if VERBOSE:
        print(f"Saved {out_signals} and {out_perf}")

    return symbol, signals_df, performance_df

# ---------------------------
# Stage 2: per-symbol combo analysis (no global aggregation)
# ---------------------------

def compute_fisher_for_row(all_pat_typ_df, combo_rows, typ, horizon):
    """
    Helper to compute fisher p-value for a single row/horizon.
    all_pat_typ_df: all rows with same Pattern & SignalType (candles only)
    combo_rows: rows where indicator state matches this combo
    typ: 'Bullish' or 'Bearish'
    """
    dir_col = f"Direction_{horizon}D"
    if dir_col not in all_pat_typ_df.columns:
        return np.nan

    if typ == "Bullish":
        total_succ = int((all_pat_typ_df[dir_col] == 1).sum())
    else:
        total_succ = int((all_pat_typ_df[dir_col] == -1).sum())
    total_fail = int(len(all_pat_typ_df) - total_succ)

    if combo_rows.empty:
        return np.nan

    if typ == "Bullish":
        combo_succ = int((combo_rows[dir_col] == 1).sum())
    else:
        combo_succ = int((combo_rows[dir_col] == -1).sum())
    combo_fail = int(len(combo_rows) - combo_succ)

    rest_succ = max(total_succ - combo_succ, 0)
    rest_fail = max(total_fail - combo_fail, 0)
    table = np.array([[combo_succ, combo_fail], [rest_succ, rest_fail]])
    try:
        if table.sum() == 0:
            return np.nan
        _, p = fisher_exact(table)
        return p
    except Exception:
        return np.nan

def analyze_combos_for_symbol(signals_df, symbol, indicators, horizons, candlestick_classified,
                              bullish_states, bearish_states, min_count, min_count_per_horizon,
                              top_k, compute_fisher):
    """
    For one symbol: for each indicator, keep only candlestick occurrences where the indicator
    state aligns with the candlestick direction (bullish pattern + bullish indicator value, or bearish+bearish).
    Aggregate per (Pattern,SignalType,IndicatorState) across horizons, compute metrics and save per-indicator CSV.
    Also save a combined per-symbol combo CSV.
    """
    if signals_df is None or signals_df.empty:
        if VERBOSE:
            print(f"No signals to analyze for {symbol}")
        return None

    # Work only on candlestick-origin rows
    df_cand = signals_df[signals_df["SignalSource"] == "Candlestick"].copy()
    if df_cand.empty:
        if VERBOSE:
            print(f"No candlestick signals for {symbol}")
        return None

    # For each indicator, find matched combos and save per-symbol+indicator CSV
    all_symbol_frames = []
    for ind in indicators:
        if ind not in signals_df.columns:
            if VERBOSE:
                print(f"{symbol}: indicator {ind} missing, skipping")
            continue

        bull_states = bullish_states.get(ind, [])
        bear_states = bearish_states.get(ind, [])

        # Keep only candlestick rows where indicator state matches the pattern's directional expectation
        # indicator value at that row:
        # For each row: if pattern occurrence is Bullish -> require indicator value in bull_states
        #               if pattern occurrence is Bearish -> require indicator value in bear_states
        def indicator_matches_row(r):
            st = r.get(ind, None)
            if pd.isna(st):
                return False
            if r["SignalType"] == "Bullish":
                return st in bull_states
            elif r["SignalType"] == "Bearish":
                return st in bear_states
            return False

        df_cand[f"IndicatorMatch_{ind}"] = df_cand.apply(indicator_matches_row, axis=1)
        df_matched = df_cand[df_cand[f"IndicatorMatch_{ind}"]].copy()
        if df_matched.empty:
            if VERBOSE:
                print(f"{symbol} - {ind}: no matched candlestick+indicator occurrences")
            continue

        # Group by Pattern, SignalType, IndicatorState (indicator's value)
        key_cols = ["Pattern", "SignalType", ind]

        # Build aggregation dict for horizons
        agg_dict = {}
        for h in horizons:
            move_col = f"Move_{h}D"
            success_col = f"SuccessFlag_{h}D"
            agg_dict[f"Count_{h}D"] = (move_col, "count")
            agg_dict[f"SuccessRate_{h}D"] = (success_col, "mean")
            agg_dict[f"AvgReturn_{h}D"] = (move_col, "mean")

        grouped = df_matched.groupby(key_cols).agg(**agg_dict).reset_index()
        if grouped.empty:
            continue

        # Derived metrics
        grouped["Count_sum"] = grouped[[f"Count_{h}D" for h in horizons]].sum(axis=1, skipna=True)
        grouped["Count_min"] = grouped[[f"Count_{h}D" for h in horizons]].min(axis=1, skipna=True)

        sr_cols = [f"SuccessRate_{h}D" for h in horizons]
        grouped["SuccessRange"] = grouped[sr_cols].max(axis=1, skipna=True) - grouped[sr_cols].min(axis=1, skipna=True)
        grouped["ConsistencyScore"] = 1 - grouped["SuccessRange"]

        # ExpectedReturn per horizon and overall
        er_cols = []
        for h in horizons:
            grouped[f"ExpectedReturn_{h}D"] = grouped[f"SuccessRate_{h}D"] * grouped[f"AvgReturn_{h}D"]
            er_cols.append(f"ExpectedReturn_{h}D")
        grouped["ExpectedReturn_sum"] = grouped[er_cols].sum(axis=1, skipna=True)

        # Filter by counts
        if min_count_per_horizon:
            mask = grouped[[f"Count_{h}D" for h in horizons]].ge(min_count).all(axis=1)
        else:
            mask = grouped["Count_sum"].fillna(0) >= min_count
        grouped = grouped[mask].copy()
        if grouped.empty:
            continue

        # Rank and keep top_k
        long_h = max(horizons)
        grouped = grouped.sort_values(["ConsistencyScore", f"SuccessRate_{long_h}D", "Count_sum"], ascending=[False, False, False])
        top_df = grouped.head(top_k).copy()

        # Optionally compute fisher p-values for these top rows
        if compute_fisher:
            # all_pat_typ: all candlestick rows with same Pattern+SignalType (for baseline)
            for idx, row in top_df.iterrows():
                pat = row["Pattern"]
                typ = row["SignalType"]
                ind_state = row[ind]
                all_pat_typ = df_cand[(df_cand["Pattern"] == pat) & (df_cand["SignalType"] == typ)]
                combo_rows = all_pat_typ[all_pat_typ[ind] == ind_state]
                for h in horizons:
                    p = compute_fisher_for_row(all_pat_typ, combo_rows, typ, h)
                    top_df.at[idx, f"fisher_p_{h}D"] = p

        # annotate metadata and save per-symbol+indicator CSV
        top_df["Symbol"] = symbol
        top_df["Indicator"] = ind
        top_df["Company"] = signals_df["Company"].iloc[0] if "Company" in signals_df.columns else ""
        out_path = os.path.join(COMBO_OUTPUT_FOLDER, f"{symbol}_best_combos_{ind}.csv")
        save_df_csv(top_df, out_path)
        if VERBOSE:
            print(f"Saved {out_path} ({len(top_df)} rows)")
        all_symbol_frames.append(top_df)

    # Save combined per-symbol combos (concatenate per-indicator top results)
    if all_symbol_frames:
        combined = pd.concat(all_symbol_frames, ignore_index=True, sort=False)
        out_combined = os.path.join(COMBO_OUTPUT_FOLDER, f"{symbol}_best_combos_all.csv")
        save_df_csv(combined, out_combined)
        if VERBOSE:
            print(f"Saved combined per-symbol combos: {out_combined}")
        return combined
    else:
        if VERBOSE:
            print(f"No combos saved for {symbol}")
        return None

# ---------------------------
# Orchestrator (per-symbol)
# ---------------------------

def run_all_symbols(input_folder, data_suffix):
    files = sorted(glob.glob(os.path.join(input_folder, f"*{data_suffix}")))
    if not files:
        print("No input files found in", input_folder)
        return

    for i, fp in enumerate(files, 1):
        if VERBOSE:
            print(f"[{i}/{len(files)}] Processing file: {os.path.basename(fp)}")
        sym, signals_df, perf_df = build_symbol_signals_and_performance(fp, CANDLE_CLASSIFIED, INDICATOR_COLS, HORIZONS)
        if sym is None:
            continue
        if signals_df is None:
            continue
        # per-symbol combo analysis
        analyze_combos_for_symbol(signals_df, sym, INDICATOR_COLS, HORIZONS, CANDLE_CLASSIFIED,
                                  BULLISH_STATES, BEARISH_STATES, MIN_COUNT, MIN_COUNT_PER_HORIZON,
                                  TOP_K_PER_INDICATOR, COMPUTE_FISHER)

if __name__ == "__main__":
    run_all_symbols(INPUT_DATA_FOLDER, DATA_SUFFIX)
