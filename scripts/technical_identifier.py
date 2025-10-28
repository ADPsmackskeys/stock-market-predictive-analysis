import os
import sys
import pandas as pd
import numpy as np
import talib

# Add config folder to path
sys.path.insert(1, './config')
from tickers import stocks  # {"Company Name": "SYMBOL"}

# Paths
DATA_FOLDER = "data/technical"
FILE_SUFFIX = "_data.csv"
OUTPUT_FOLDER = "results"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Candlestick patterns to check using TA-Lib
candlestick_patterns = {
    "CDL_BULLISH_ENGULFING": talib.CDLENGULFING,
    "CDL_PIERCING": talib.CDLPIERCING,
    "CDL_HARAMI": talib.CDLHARAMI,
    "CDL_MORNING_STAR": talib.CDLMORNINGSTAR,
    "CDL_THREE_WHITE_SOLDIERS": talib.CDL3WHITESOLDIERS,
    "CDL_THREE_INSIDE": talib.CDL3INSIDE,
    "CDL_THREE_OUTSIDE": talib.CDL3OUTSIDE,
    "CDL_HAMMER": talib.CDLHAMMER,
    "CDL_INVERTED_HAMMER": talib.CDLINVERTEDHAMMER,
    "CDL_DARKCLOUDCOVER": talib.CDLDARKCLOUDCOVER,
    "CDL_EVENING_STAR": talib.CDLEVENINGSTAR,
    "CDL_THREE_BLACK_CROWS": talib.CDL3BLACKCROWS,
    "CDL_SHOOTING_STAR": talib.CDLSHOOTINGSTAR,
    "CDL_HARAMI_CROSS": talib.CDLHARAMICROSS,
    "CDL_ADVANCE_BLOCK": talib.CDLADVANCEBLOCK,
}

# Indicator columns to record as signals
indicator_cols = ["EMA_Trend", "MACD_Trend", "RSI_State", "CMF_State"]

all_signals = []

# ------------------------
# Iterate through all tickers
# ------------------------
for idx, (company, symbol) in enumerate(stocks.items(), 1):
    print(f"[{idx}/{len(stocks)}] Processing {symbol}...")
    path = os.path.join(DATA_FOLDER, f"{symbol}{FILE_SUFFIX}")
    if not os.path.exists(path):
        print(f"Skipping {symbol}: file not found.")
        continue

    try:
        df = pd.read_csv(path, parse_dates=["Date"]).sort_values("Date")
        df = df.set_index("Date")
    except Exception as e:
        print(f"Skipping {symbol}: {e}")
        continue

    if len(df) < 30:
        continue

    # ------------------------
    # Compute candlestick patterns
    # ------------------------
    for name, func in candlestick_patterns.items():
        try:
            df[name] = func(df["Open"], df["High"], df["Low"], df["Close"])
        except Exception:
            df[name] = 0

    # ------------------------
    # Indicator-only signals
    # ------------------------
    df["EMA_Trend"] = np.where(df["EMA_20"] > df["EMA_50"], "Bullish", "Bearish")
    df["MACD_Trend"] = np.where(df["MACD"] > df["MACD_Signal"], "Bullish", "Bearish")
    df["RSI_State"] = np.where(df["RSI_14"] > 70, "Overbought",
                               np.where(df["RSI_14"] < 30, "Oversold", "Neutral"))
    df["CMF_State"] = np.where(df["CMF_20"] > 0, "Positive", "Negative")

    # ------------------------
    # Compute 1D, 2D, 3D next moves
    # ------------------------
    for n in [1, 2, 3]:
        df[f"Move_{n}D"] = df["Close"].shift(-n) / df["Close"] - 1
        df[f"Direction_{n}D"] = df[f"Move_{n}D"].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

    # ------------------------
    # Record signals for this ticker
    # ------------------------
    pattern_cols = list(candlestick_patterns.keys())

    signals_list = []

    for date, row in df.iloc[:-3].iterrows():  # skip last 3 rows
        # Candlestick signals
        for pattern in pattern_cols:
            val = row[pattern]
            if val != 0:
                signals_list.append({
                    "Date": date,
                    "SignalSource": "Candlestick",
                    "Pattern": pattern,
                    "SignalType": "Bullish" if val > 0 else "Bearish",
                    "EMA_Trend": row["EMA_Trend"],
                    "MACD_Trend": row["MACD_Trend"],
                    "RSI_State": row["RSI_State"],
                    "CMF_State": row["CMF_State"],
                    "Move_1D": row["Move_1D"],
                    "Direction_1D": row["Direction_1D"],
                    "Move_2D": row["Move_2D"],
                    "Direction_2D": row["Direction_2D"],
                    "Move_3D": row["Move_3D"],
                    "Direction_3D": row["Direction_3D"],
                })

        # Indicator-only signals
        for ind in indicator_cols:
            val = row[ind]
            if val in ["Bullish", "Bearish", "Oversold", "Overbought", "Positive", "Negative"]:
                signals_list.append({
                    "Date": date,
                    "SignalSource": "Indicator",
                    "Pattern": ind,
                    "SignalType": val,
                    "EMA_Trend": row["EMA_Trend"],
                    "MACD_Trend": row["MACD_Trend"],
                    "RSI_State": row["RSI_State"],
                    "CMF_State": row["CMF_State"],
                    "Move_1D": row["Move_1D"],
                    "Direction_1D": row["Direction_1D"],
                    "Move_2D": row["Move_2D"],
                    "Direction_2D": row["Direction_2D"],
                    "Move_3D": row["Move_3D"],
                    "Direction_3D": row["Direction_3D"],
                })

    if not signals_list:
        continue

    df_signals = pd.DataFrame(signals_list)
    df_signals["Symbol"] = symbol
    df_signals["Company"] = company

    # Append to master all_signals
    all_signals.append(df_signals)

    # ------------------------
    # Compute per-stock performance/weights
    # ------------------------
    performance_list = []
    for n in [1, 2, 3]:
        grp = df_signals.groupby(["SignalSource", "Pattern", "SignalType"]).agg(
            SuccessRate=(f"Direction_{n}D", lambda x: (x == 1).mean() if "Bullish" in x.name else (x == -1).mean()),
            AvgReturn=(f"Move_{n}D", "mean"),
            Count=(f"Move_{n}D", "count")
        ).reset_index()
        grp["Horizon"] = f"{n}D"
        grp["Weight"] = grp["SuccessRate"] * grp["AvgReturn"]
        performance_list.append(grp)

    df_perf = pd.concat(performance_list, axis=0).sort_values("Weight", ascending=False)
    df_perf.to_csv(os.path.join(OUTPUT_FOLDER, f"performance_{symbol}.csv"), index=False)

# ------------------------
# Save master CSV
# ------------------------
if all_signals:
    # Combine all signals into a single DataFrame (optional, for Parquet)
    signals_df = pd.concat(all_signals, axis=0)
    
    # --- 1. Save per-ticker CSVs (compressed) ---
    for symbol, df_stock in signals_df.groupby("Symbol"):
        out_path = os.path.join(OUTPUT_FOLDER, f"{symbol}_signals.csv")
        df_stock.to_csv(out_path, index=False)

    print(f"Processed {len(signals_df)} signals from {len(stocks)} stocks.")
    print(f"Saved per-ticker CSVs in {OUTPUT_FOLDER}")

else:
    print("No signals detected.")
