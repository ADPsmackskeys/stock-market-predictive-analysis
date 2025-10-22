import pandas as pd
import talib
import os

DATA_FOLDER = "data/technical"
FILE_SUFFIX = "_data.csv"

# Strong candlestick patterns with base scores
candlestick_patterns = {
    # Bullish
    "CDL_BULLISH_ENGULFING": (talib.CDLENGULFING, 9),
    "CDL_PIERCING": (talib.CDLPIERCING, 8),
    "CDL_HARAMI": (talib.CDLHARAMI, 6),
    "CDL_MORNING_STAR": (talib.CDLMORNINGSTAR, 9),
    "CDL_THREE_WHITE_SOLDIERS": (talib.CDL3WHITESOLDIERS, 10),
    "CDL_THREE_INSIDE_UP": (talib.CDL3INSIDE, 7),
    "CDL_THREE_OUTSIDE_UP": (talib.CDL3OUTSIDE, 8),
    "CDL_HAMMER": (talib.CDLHAMMER, 7),
    "CDL_INVERTED_HAMMER": (talib.CDLINVERTEDHAMMER, 6),

    # Bearish
    "CDL_DARKCLOUDCOVER": (talib.CDLDARKCLOUDCOVER, 8),
    "CDL_EVENING_STAR": (talib.CDLEVENINGSTAR, 9),
    "CDL_THREE_BLACK_CROWS": (talib.CDL3BLACKCROWS, 10),
    "CDL_SHOOTING_STAR": (talib.CDLSHOOTINGSTAR, 7),
    "CDL_HARAMI_CROSS": (talib.CDLHARAMICROSS, 6),
    "CDL_ADVANCE_BLOCK": (talib.CDLADVANCEBLOCK, 7)
}

results = []

# Process all stock files
for file in os.listdir(DATA_FOLDER):
    if not file.endswith(FILE_SUFFIX):
        continue

    symbol = file.replace(FILE_SUFFIX, "")
    path = os.path.join(DATA_FOLDER, file)

    try:
        df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")
        if len(df) < 2:  # need at least 2 days
            continue
    except Exception as e:
        print(f"Skipping {symbol}, error: {e}")
        continue

    last_two = df.index[-2:]        
    second_last_date = last_two[0]  
    last_date = last_two[1]         


    signal_row = df.loc[second_last_date]
    impact_row = df.loc[last_date]

    score = 0
    reasons = []

    # Check candlestick patterns on signal day 
    for name, (func, base_score) in candlestick_patterns.items():
        try:
            values = func(df["Open"], df["High"], df["Low"], df["Close"])
            if values.loc[second_last_date] != 0:
                if values.loc[second_last_date] > 0:
                    score += base_score
                    reasons.append(f"{name} (Bullish +{base_score})")
                else:
                    score -= base_score
                    reasons.append(f"{name} (Bearish -{base_score})")
        except Exception:
            continue

    # Trend indicators on signal day 
    if signal_row["EMA_20"] > signal_row["EMA_50"]:
        score += 5
        reasons.append("EMA20 > EMA50 (Bullish +5)")
    elif signal_row["EMA_20"] < signal_row["EMA_50"]:
        score -= 5
        reasons.append("EMA20 < EMA50 (Bearish -5)")

    if signal_row["MACD"] > signal_row["MACD_Signal"]:
        score += 4
        reasons.append("MACD > Signal (Bullish +4)")
    elif signal_row["MACD"] < signal_row["MACD_Signal"]:
        score -= 4
        reasons.append("MACD < Signal (Bearish -4)")

    if signal_row["RSI_14"] > 70:
        score -= 3
        reasons.append("RSI > 70 (Overbought -3)")
    elif signal_row["RSI_14"] < 30:
        score += 3
        reasons.append("RSI < 30 (Oversold +3)")

    if signal_row["CMF_20"] > 0:
        score += 2
        reasons.append("CMF Positive (Bullish +2)")
    elif signal_row["CMF_20"] < 0:
        score -= 2
        reasons.append("CMF Negative (Bearish -2)")

    # Next-day price impact 
    price_change = (impact_row["Close"] - signal_row["Close"]) / signal_row["Close"]

    results.append({
        "Symbol": symbol,
        "Signal_Date": second_last_date.date(),
        "Impact_Date": last_date.date(),
        "Score": score,
        "Reasons": "; ".join(reasons),
        "Next_Day_Move": f"{price_change:.2%}"
    })

# Convert to DataFrame
signals_df = pd.DataFrame(results)

# Separate bullish and bearish
bullish_df = signals_df[signals_df["Score"] > 0].sort_values(
    by="Score", ascending=False
).head(20)

bearish_df = signals_df[signals_df["Score"] < 0].sort_values(
    by="Score", ascending=True
).head(20)

# Determine predicted direction based on Score
def predicted_direction(score):
    return "Bullish" if score > 0 else "Bearish"

# Print results 
print("Top 20 Strongest Bullish Signals:")
for _, row in bullish_df.iterrows():
    pred_dir = predicted_direction(row["Score"])
    actual_dir = "Bullish" if float(row["Next_Day_Move"].strip("%")) > 0 else "Bearish"
    match = "Correct" if pred_dir == actual_dir else "Incorrect"
    print(f"{row['Symbol']} → {row['Signal_Date']} → Score {row['Score']} → "
          f"Next day move {row['Next_Day_Move']} → Predicted vs Actual: {match}")

print("\nTop 20 Strongest Bearish Signals:")
for _, row in bearish_df.iterrows():
    pred_dir = predicted_direction(row["Score"])
    actual_dir = "Bullish" if float(row["Next_Day_Move"].strip("%")) > 0 else "Bearish"
    match = "Correct" if pred_dir == actual_dir else "Incorrect"
    print(f"{row['Symbol']} → {row['Signal_Date']} → Score {row['Score']} → "
          f"Next day move {row['Next_Day_Move']} → Predicted vs Actual: {match}")