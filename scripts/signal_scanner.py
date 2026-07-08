"""
Signal Scanner v2 — horizon-aware, MFE-based, profit-factor scored
===================================================================
Key changes from v1:
  - Signal-class-aware horizon groups (mean reversion / momentum / trend / candle)
  - Max Favourable Excursion (MFE) within window, not just close-at-N
  - Minimum move threshold to filter noise from "success"
  - Profit Factor and Expected Value as primary scoring metrics
  - Success rate alone is deprecated as primary metric
  - Cross-symbol leaderboard unchanged

Usage:
    python signal_scanner.py
    python signal_scanner.py --symbol RELIANCE
    python signal_scanner.py --triples
    python signal_scanner.py --no-fisher
"""

import os
import glob
import argparse
import itertools
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

warnings.filterwarnings("ignore")

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("[WARN] TA-Lib not found — candlestick patterns will be skipped.")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

INPUT_FOLDER    = "data/technical"
DATA_SUFFIX     = "_data.csv"
OUTPUT_FOLDER   = "results/signals_v2"
Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

# ── Horizon groups
# Signals are assigned to one class. Each class has its own horizon list.
# MFE is computed over the full window (not just close-at-N).
HORIZON_GROUPS = {
    "candle":   {"horizons": [2, 3, 5]},
    "mean_rev": {"horizons": [3, 5, 7]},
    "momentum": {"horizons": [5, 10, 15]},
    "trend":    {"horizons": [10, 20, 30]},
    "volume":   {"horizons": [3, 7, 10]},
}

# A "win" threshold that scales with each stock's *own* volatility (that day's
# ATR as a % of price) and with the horizon length (sqrt(h), since a random
# walk's expected drift grows with sqrt(time)). This replaces a flat per-class
# min_move_pct, which was the same bar for a stock with a 2.5% ATR% and one
# with a 10%+ ATR% -- systematically under-scoring low-vol names and inflating
# scores for noisy ones. K is a single tunable constant instead of five.
ATR_THRESHOLD_K = 0.15

# Signal name → horizon class
SIGNAL_HORIZON_CLASS = {
    # candle
    "CDL_Hammer": "candle", "CDL_InvertedHammer": "candle", "CDL_MorningStar": "candle",
    "CDL_ThreeWhiteSoldiers": "candle", "CDL_Piercing": "candle",
    "CDL_Bullish3Inside": "candle", "CDL_Bullish3Outside": "candle",
    "CDL_DragonFlyDoji": "candle", "CDL_BullishKicker": "candle",
    "CDL_Marubozu_Bull": "candle", "CDL_ShootingStar": "candle",
    "CDL_EveningStar": "candle", "CDL_ThreeBlackCrows": "candle",
    "CDL_DarkCloudCover": "candle", "CDL_HangingMan": "candle",
    "CDL_AdvanceBlock": "candle", "CDL_GravestoneDoji": "candle",
    "CDL_BearishKicker": "candle", "CDL_Engulfing_Bull": "candle",
    "CDL_Engulfing_Bear": "candle", "CDL_Harami_Bull": "candle",
    "CDL_Harami_Bear": "candle", "CDL_HaramiCross_Bull": "candle",
    "CDL_HaramiCross_Bear": "candle", "CDL_Doji": "candle",
    # mean reversion
    "RSI_Oversold": "mean_rev", "RSI_Overbought": "mean_rev",
    "RSI_Exit_Oversold": "mean_rev", "RSI_Exit_Overbought": "mean_rev",
    "BB_Lower_Touch": "mean_rev", "BB_Upper_Touch": "mean_rev",
    "BB_Lower_Bounce": "mean_rev", "BB_Upper_Reject": "mean_rev",
    "Stoch_Oversold": "mean_rev", "Stoch_Overbought": "mean_rev",
    "Stoch_Cross_Up": "mean_rev", "Stoch_Cross_Down": "mean_rev",
    "CCI_Oversold": "mean_rev", "CCI_Overbought": "mean_rev",
    "CCI_Exit_Oversold": "mean_rev", "CCI_Exit_Overbought": "mean_rev",
    "Williams_Oversold": "mean_rev", "Williams_Overbought": "mean_rev",
    # momentum
    "MACD_Cross_Up_Event": "momentum", "MACD_Cross_Down_Event": "momentum",
    "MACD_Zero_Cross_Up": "momentum", "MACD_Zero_Cross_Down": "momentum",
    "MACD_Hist_Rising": "momentum", "MACD_Hist_Falling": "momentum",
    "MACD_Bullish_State": "momentum", "MACD_Bearish_State": "momentum",
    "EMA_Cross_Up_Event": "momentum", "EMA_Cross_Down_Event": "momentum",
    "EMA_Bullish_State": "momentum", "EMA_Bearish_State": "momentum",
    "RSI_Cross50_Up": "momentum", "RSI_Cross50_Down": "momentum",
    "RSI_Bull_Momentum": "momentum", "RSI_Bear_Momentum": "momentum",
    "MACD_Hist_Positive": "momentum", "MACD_Hist_Negative": "momentum",
    "CMF_Positive": "momentum", "CMF_Negative": "momentum",
    "CMF_Strong_Bull": "momentum", "CMF_Strong_Bear": "momentum",
    "OBV_Rising": "momentum", "OBV_Falling": "momentum",
    "OBV_Trend_Bull": "momentum", "OBV_Trend_Bear": "momentum",
    "Stoch_Bull_Zone": "momentum", "Stoch_Bear_Zone": "momentum",
    "VWAP_Bull": "momentum", "VWAP_Bear": "momentum",
    # trend / structural
    "Golden_Cross_Event": "trend", "Death_Cross_Event": "trend",
    "Golden_Cross_State": "trend", "Death_Cross_State": "trend",
    "Price_Above_EMA200": "trend", "Price_Below_EMA200": "trend",
    "Price_Above_SMA50": "trend", "Price_Below_SMA50": "trend",
    # volume
    "Volume_Spike": "volume", "Volume_Dry_Up": "volume",
}

STRATEGY_HORIZON_OVERRIDE = {
    "RSI_Golden_Cross_Bull": "trend", "RSI_Golden_Cross_Event_Bull": "trend",
    "Death_Cross_RSI_Bear": "trend", "Death_Cross_Event_Bear": "trend",
    "Golden_Cross_MACD_RSI": "trend", "Triple_Bear_Confirm": "trend",
    "RSI_MACD_EMA_Bull": "momentum", "RSI_MACD_EMA_Bear": "momentum",
    "RSI_BB_MACD_Bull": "mean_rev", "BB_RSI_Oversold_Bounce": "mean_rev",
    "BB_RSI_Overbought_Reject": "mean_rev", "BB_Stoch_Bull": "mean_rev",
    "BB_Stoch_Bear": "mean_rev", "Volume_MACD_Bull": "volume",
    "Volume_EMA_Cross_Bull": "volume", "Volume_Golden_Cross": "trend",
    "Hammer_EMA_Bull": "candle", "MorningStar_MACD_Bull": "candle",
    "Engulfing_Bull_RSI": "candle", "Engulfing_Bear_RSI": "candle",
    "ShootingStar_EMA_Bear": "candle", "EveningStar_MACD_Bear": "candle",
}

MIN_OCCURRENCES = 10
TOP_N_PER_DIR   = 25
COMPUTE_FISHER  = True
FISHER_ALPHA    = 0.10
TEST_TRIPLES    = False
VERBOSE         = True

# ─────────────────────────────────────────────────────────────────────────────
# ATOMIC SIGNAL DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

def _safe(fn: Callable) -> Callable:
    def wrapper(df):
        try:
            return fn(df).fillna(False).astype(bool)
        except (KeyError, TypeError):
            return pd.Series(False, index=df.index)
    return wrapper

# RSI
def _rsi_oversold(df):         return df["RSI_14"] < 30
def _rsi_overbought(df):       return df["RSI_14"] > 70
def _rsi_bull_momentum(df):    return (df["RSI_14"] > 50) & (df["RSI_14"] < 70)
def _rsi_bear_momentum(df):    return (df["RSI_14"] < 50) & (df["RSI_14"] > 30)
def _rsi_cross_50_up(df):      return (df["RSI_14"] > 50) & (df["RSI_14"].shift(1) <= 50)
def _rsi_cross_50_down(df):    return (df["RSI_14"] < 50) & (df["RSI_14"].shift(1) >= 50)
def _rsi_exit_oversold(df):    return (df["RSI_14"] > 30) & (df["RSI_14"].shift(1) <= 30)
def _rsi_exit_overbought(df):  return (df["RSI_14"] < 70) & (df["RSI_14"].shift(1) >= 70)
# MAs
def _ema_bullish_state(df):    return df["EMA_20"] > df["EMA_50"]
def _ema_bearish_state(df):    return df["EMA_20"] < df["EMA_50"]
def _golden_cross_state(df):   return df["SMA_50"] > df["EMA_200"]
def _death_cross_state(df):    return df["SMA_50"] < df["EMA_200"]
def _golden_cross_event(df):   return (df["SMA_50"] > df["EMA_200"]) & (df["SMA_50"].shift(1) <= df["EMA_200"].shift(1))
def _death_cross_event(df):    return (df["SMA_50"] < df["EMA_200"]) & (df["SMA_50"].shift(1) >= df["EMA_200"].shift(1))
def _ema_cross_up_event(df):   return (df["EMA_20"] > df["EMA_50"]) & (df["EMA_20"].shift(1) <= df["EMA_50"].shift(1))
def _ema_cross_down_event(df): return (df["EMA_20"] < df["EMA_50"]) & (df["EMA_20"].shift(1) >= df["EMA_50"].shift(1))
def _price_above_sma20(df):    return df["Close"] > df["SMA_20"]
def _price_below_sma20(df):    return df["Close"] < df["SMA_20"]
def _price_above_sma50(df):    return df["Close"] > df["SMA_50"]
def _price_below_sma50(df):    return df["Close"] < df["SMA_50"]
def _price_above_ema200(df):   return df["Close"] > df["EMA_200"]
def _price_below_ema200(df):   return df["Close"] < df["EMA_200"]
# MACD
def _macd_bullish_state(df):   return df["MACD"] > df["MACD_Signal"]
def _macd_bearish_state(df):   return df["MACD"] < df["MACD_Signal"]
def _macd_cross_up_event(df):  return (df["MACD"] > df["MACD_Signal"]) & (df["MACD"].shift(1) <= df["MACD_Signal"].shift(1))
def _macd_cross_dn_event(df):  return (df["MACD"] < df["MACD_Signal"]) & (df["MACD"].shift(1) >= df["MACD_Signal"].shift(1))
def _macd_hist_positive(df):   return df["MACD_Hist"] > 0
def _macd_hist_negative(df):   return df["MACD_Hist"] < 0
def _macd_hist_rising(df):     return df["MACD_Hist"] > df["MACD_Hist"].shift(1)
def _macd_hist_falling(df):    return df["MACD_Hist"] < df["MACD_Hist"].shift(1)
def _macd_zero_cross_up(df):   return (df["MACD"] > 0) & (df["MACD"].shift(1) <= 0)
def _macd_zero_cross_dn(df):   return (df["MACD"] < 0) & (df["MACD"].shift(1) >= 0)
# BB
def _bb_lower_touch(df):       return df["Close"] <= df["BB_lower"]
def _bb_upper_touch(df):       return df["Close"] >= df["BB_upper"]
def _bb_lower_bounce(df):      return (df["Close"] > df["BB_lower"]) & (df["Close"].shift(1) <= df["BB_lower"].shift(1))
def _bb_upper_reject(df):      return (df["Close"] < df["BB_upper"]) & (df["Close"].shift(1) >= df["BB_upper"].shift(1))
def _bb_squeeze(df):           return (df["BB_upper"] - df["BB_lower"]) < (df["BB_upper"] - df["BB_lower"]).rolling(20).mean() * 0.7
def _bb_expansion_bull(df):    return (df["Close"] > df["BB_middle"]) & ((df["BB_upper"] - df["BB_lower"]) > (df["BB_upper"] - df["BB_lower"]).rolling(20).mean() * 1.3)
def _bb_expansion_bear(df):    return (df["Close"] < df["BB_middle"]) & ((df["BB_upper"] - df["BB_lower"]) > (df["BB_upper"] - df["BB_lower"]).rolling(20).mean() * 1.3)
# Stochastic
def _stoch_oversold(df):       return df["STOCH_K"] < 20
def _stoch_overbought(df):     return df["STOCH_K"] > 80
def _stoch_cross_up(df):       return (df["STOCH_K"] > df["STOCH_D"]) & (df["STOCH_K"].shift(1) <= df["STOCH_D"].shift(1))
def _stoch_cross_dn(df):       return (df["STOCH_K"] < df["STOCH_D"]) & (df["STOCH_K"].shift(1) >= df["STOCH_D"].shift(1))
def _stoch_bull_zone(df):      return (df["STOCH_K"] > 20) & (df["STOCH_K"] < 80) & (df["STOCH_K"] > df["STOCH_D"])
def _stoch_bear_zone(df):      return (df["STOCH_K"] > 20) & (df["STOCH_K"] < 80) & (df["STOCH_K"] < df["STOCH_D"])
# CMF / OBV
def _cmf_positive(df):         return df["CMF_20"] > 0
def _cmf_negative(df):         return df["CMF_20"] < 0
def _cmf_strong_bull(df):      return df["CMF_20"] > 0.1
def _cmf_strong_bear(df):      return df["CMF_20"] < -0.1
def _obv_rising(df):           return df["OBV"] > df["OBV"].shift(1)
def _obv_falling(df):          return df["OBV"] < df["OBV"].shift(1)
def _obv_trend_bull(df):       return df["OBV"] > df["OBV"].rolling(20).mean()
def _obv_trend_bear(df):       return df["OBV"] < df["OBV"].rolling(20).mean()
# Volume
def _volume_spike(df):         return df["Volume"] > df["Volume"].rolling(20).mean() * 1.5
def _volume_dry_up(df):        return df["Volume"] < df["Volume"].rolling(20).mean() * 0.5
# CCI / Williams / ATR / VWAP
def _cci_oversold(df):         return df["CCI_20"] < -100
def _cci_overbought(df):       return df["CCI_20"] > 100
def _cci_exit_oversold(df):    return (df["CCI_20"] > -100) & (df["CCI_20"].shift(1) <= -100)
def _cci_exit_overbought(df):  return (df["CCI_20"] < 100) & (df["CCI_20"].shift(1) >= 100)
def _williams_oversold(df):    return df["Williams_%R"] < -80
def _williams_overbought(df):  return df["Williams_%R"] > -20
def _atr_high_vol(df):         return df["ATR_14"] > df["ATR_14"].rolling(20).mean() * 1.3
def _atr_low_vol(df):          return df["ATR_14"] < df["ATR_14"].rolling(20).mean() * 0.7
def _vwap_bull(df):            return df["Close"] > df["VWAP"]
def _vwap_bear(df):            return df["Close"] < df["VWAP"]


ATOMIC_SIGNALS: Dict[str, dict] = {
    "RSI_Oversold":         {"fn": _safe(_rsi_oversold),        "direction": "Bullish"},
    "RSI_Overbought":       {"fn": _safe(_rsi_overbought),      "direction": "Bearish"},
    "RSI_Bull_Momentum":    {"fn": _safe(_rsi_bull_momentum),   "direction": "Bullish"},
    "RSI_Bear_Momentum":    {"fn": _safe(_rsi_bear_momentum),   "direction": "Bearish"},
    "RSI_Cross50_Up":       {"fn": _safe(_rsi_cross_50_up),     "direction": "Bullish"},
    "RSI_Cross50_Down":     {"fn": _safe(_rsi_cross_50_down),   "direction": "Bearish"},
    "RSI_Exit_Oversold":    {"fn": _safe(_rsi_exit_oversold),   "direction": "Bullish"},
    "RSI_Exit_Overbought":  {"fn": _safe(_rsi_exit_overbought), "direction": "Bearish"},
    "EMA_Bullish_State":    {"fn": _safe(_ema_bullish_state),   "direction": "Bullish"},
    "EMA_Bearish_State":    {"fn": _safe(_ema_bearish_state),   "direction": "Bearish"},
    "Golden_Cross_State":   {"fn": _safe(_golden_cross_state),  "direction": "Bullish"},
    "Death_Cross_State":    {"fn": _safe(_death_cross_state),   "direction": "Bearish"},
    "Golden_Cross_Event":   {"fn": _safe(_golden_cross_event),  "direction": "Bullish"},
    "Death_Cross_Event":    {"fn": _safe(_death_cross_event),   "direction": "Bearish"},
    "EMA_Cross_Up_Event":   {"fn": _safe(_ema_cross_up_event),  "direction": "Bullish"},
    "EMA_Cross_Down_Event": {"fn": _safe(_ema_cross_down_event),"direction": "Bearish"},
    "Price_Above_SMA20":    {"fn": _safe(_price_above_sma20),   "direction": "Bullish"},
    "Price_Below_SMA20":    {"fn": _safe(_price_below_sma20),   "direction": "Bearish"},
    "Price_Above_SMA50":    {"fn": _safe(_price_above_sma50),   "direction": "Bullish"},
    "Price_Below_SMA50":    {"fn": _safe(_price_below_sma50),   "direction": "Bearish"},
    "Price_Above_EMA200":   {"fn": _safe(_price_above_ema200),  "direction": "Bullish"},
    "Price_Below_EMA200":   {"fn": _safe(_price_below_ema200),  "direction": "Bearish"},
    "MACD_Bullish_State":   {"fn": _safe(_macd_bullish_state),  "direction": "Bullish"},
    "MACD_Bearish_State":   {"fn": _safe(_macd_bearish_state),  "direction": "Bearish"},
    "MACD_Cross_Up_Event":  {"fn": _safe(_macd_cross_up_event), "direction": "Bullish"},
    "MACD_Cross_Down_Event":{"fn": _safe(_macd_cross_dn_event), "direction": "Bearish"},
    "MACD_Hist_Positive":   {"fn": _safe(_macd_hist_positive),  "direction": "Bullish"},
    "MACD_Hist_Negative":   {"fn": _safe(_macd_hist_negative),  "direction": "Bearish"},
    "MACD_Hist_Rising":     {"fn": _safe(_macd_hist_rising),    "direction": "Bullish"},
    "MACD_Hist_Falling":    {"fn": _safe(_macd_hist_falling),   "direction": "Bearish"},
    "MACD_Zero_Cross_Up":   {"fn": _safe(_macd_zero_cross_up),  "direction": "Bullish"},
    "MACD_Zero_Cross_Down": {"fn": _safe(_macd_zero_cross_dn),  "direction": "Bearish"},
    "BB_Lower_Touch":       {"fn": _safe(_bb_lower_touch),      "direction": "Bullish"},
    "BB_Upper_Touch":       {"fn": _safe(_bb_upper_touch),      "direction": "Bearish"},
    "BB_Lower_Bounce":      {"fn": _safe(_bb_lower_bounce),     "direction": "Bullish"},
    "BB_Upper_Reject":      {"fn": _safe(_bb_upper_reject),     "direction": "Bearish"},
    "BB_Squeeze":           {"fn": _safe(_bb_squeeze),          "direction": "Both"},
    "BB_Expansion_Bull":    {"fn": _safe(_bb_expansion_bull),   "direction": "Bullish"},
    "BB_Expansion_Bear":    {"fn": _safe(_bb_expansion_bear),   "direction": "Bearish"},
    "Stoch_Oversold":       {"fn": _safe(_stoch_oversold),      "direction": "Bullish"},
    "Stoch_Overbought":     {"fn": _safe(_stoch_overbought),    "direction": "Bearish"},
    "Stoch_Cross_Up":       {"fn": _safe(_stoch_cross_up),      "direction": "Bullish"},
    "Stoch_Cross_Down":     {"fn": _safe(_stoch_cross_dn),      "direction": "Bearish"},
    "Stoch_Bull_Zone":      {"fn": _safe(_stoch_bull_zone),     "direction": "Bullish"},
    "Stoch_Bear_Zone":      {"fn": _safe(_stoch_bear_zone),     "direction": "Bearish"},
    "CMF_Positive":         {"fn": _safe(_cmf_positive),        "direction": "Bullish"},
    "CMF_Negative":         {"fn": _safe(_cmf_negative),        "direction": "Bearish"},
    "CMF_Strong_Bull":      {"fn": _safe(_cmf_strong_bull),     "direction": "Bullish"},
    "CMF_Strong_Bear":      {"fn": _safe(_cmf_strong_bear),     "direction": "Bearish"},
    "OBV_Rising":           {"fn": _safe(_obv_rising),          "direction": "Bullish"},
    "OBV_Falling":          {"fn": _safe(_obv_falling),         "direction": "Bearish"},
    "OBV_Trend_Bull":       {"fn": _safe(_obv_trend_bull),      "direction": "Bullish"},
    "OBV_Trend_Bear":       {"fn": _safe(_obv_trend_bear),      "direction": "Bearish"},
    "Volume_Spike":         {"fn": _safe(_volume_spike),        "direction": "Both"},
    "Volume_Dry_Up":        {"fn": _safe(_volume_dry_up),       "direction": "Both"},
    "CCI_Oversold":         {"fn": _safe(_cci_oversold),        "direction": "Bullish"},
    "CCI_Overbought":       {"fn": _safe(_cci_overbought),      "direction": "Bearish"},
    "CCI_Exit_Oversold":    {"fn": _safe(_cci_exit_oversold),   "direction": "Bullish"},
    "CCI_Exit_Overbought":  {"fn": _safe(_cci_exit_overbought), "direction": "Bearish"},
    "Williams_Oversold":    {"fn": _safe(_williams_oversold),   "direction": "Bullish"},
    "Williams_Overbought":  {"fn": _safe(_williams_overbought), "direction": "Bearish"},
    "ATR_High_Volatility":  {"fn": _safe(_atr_high_vol),        "direction": "Both"},
    "ATR_Low_Volatility":   {"fn": _safe(_atr_low_vol),         "direction": "Both"},
    "VWAP_Bull":            {"fn": _safe(_vwap_bull),           "direction": "Bullish"},
    "VWAP_Bear":            {"fn": _safe(_vwap_bear),           "direction": "Bearish"},
}

# ─────────────────────────────────────────────────────────────────────────────
# CANDLESTICK PATTERNS
# ─────────────────────────────────────────────────────────────────────────────

CANDLE_PATTERNS = {}
if TALIB_AVAILABLE:
    CANDLE_PATTERNS = {
        "CDL_Hammer":             (talib.CDLHAMMER,         "Bullish"),
        "CDL_InvertedHammer":     (talib.CDLINVERTEDHAMMER, "Bullish"),
        "CDL_MorningStar":        (talib.CDLMORNINGSTAR,    "Bullish"),
        "CDL_ThreeWhiteSoldiers": (talib.CDL3WHITESOLDIERS, "Bullish"),
        "CDL_Piercing":           (talib.CDLPIERCING,       "Bullish"),
        "CDL_Bullish3Inside":     (talib.CDL3INSIDE,        "Bullish"),
        "CDL_Bullish3Outside":    (talib.CDL3OUTSIDE,       "Bullish"),
        "CDL_DragonFlyDoji":      (talib.CDLDRAGONFLYDOJI,  "Bullish"),
        "CDL_BullishKicker":      (talib.CDLKICKING,        "Bullish"),
        "CDL_Marubozu_Bull":      (talib.CDLMARUBOZU,       "Bullish"),
        "CDL_ShootingStar":       (talib.CDLSHOOTINGSTAR,   "Bearish"),
        "CDL_EveningStar":        (talib.CDLEVENINGSTAR,    "Bearish"),
        "CDL_ThreeBlackCrows":    (talib.CDL3BLACKCROWS,    "Bearish"),
        "CDL_DarkCloudCover":     (talib.CDLDARKCLOUDCOVER, "Bearish"),
        "CDL_HangingMan":         (talib.CDLHANGINGMAN,     "Bearish"),
        "CDL_AdvanceBlock":       (talib.CDLADVANCEBLOCK,   "Bearish"),
        "CDL_GravestoneDoji":     (talib.CDLGRAVESTONEDOJI, "Bearish"),
        "CDL_BearishKicker":      (talib.CDLKICKING,        "Bearish"),
        "CDL_Engulfing_Bull":     (talib.CDLENGULFING,      "Bullish"),
        "CDL_Engulfing_Bear":     (talib.CDLENGULFING,      "Bearish"),
        "CDL_Harami_Bull":        (talib.CDLHARAMI,         "Bullish"),
        "CDL_Harami_Bear":        (talib.CDLHARAMI,         "Bearish"),
        "CDL_HaramiCross_Bull":   (talib.CDLHARAMICROSS,    "Bullish"),
        "CDL_HaramiCross_Bear":   (talib.CDLHARAMICROSS,    "Bearish"),
        "CDL_Doji":               (talib.CDLDOJI,           "Both"),
    }

# ─────────────────────────────────────────────────────────────────────────────
# PREDEFINED NAMED STRATEGIES
# ─────────────────────────────────────────────────────────────────────────────

PREDEFINED_STRATEGIES = {
    "RSI_Golden_Cross_Bull":       {"signals": ["RSI_Exit_Oversold",    "Golden_Cross_State"],     "direction": "Bullish"},
    "RSI_Golden_Cross_Event_Bull": {"signals": ["RSI_Oversold",         "Golden_Cross_Event"],     "direction": "Bullish"},
    "RSI_EMA_Bull":                {"signals": ["RSI_Oversold",         "EMA_Bullish_State"],      "direction": "Bullish"},
    "RSI_MACD_Bull":               {"signals": ["RSI_Exit_Oversold",    "MACD_Cross_Up_Event"],    "direction": "Bullish"},
    "RSI_MACD_Bear":               {"signals": ["RSI_Exit_Overbought",  "MACD_Cross_Down_Event"],  "direction": "Bearish"},
    "Death_Cross_RSI_Bear":        {"signals": ["RSI_Overbought",       "Death_Cross_State"],      "direction": "Bearish"},
    "Death_Cross_Event_Bear":      {"signals": ["RSI_Bear_Momentum",    "Death_Cross_Event"],      "direction": "Bearish"},
    "BB_RSI_Oversold_Bounce":      {"signals": ["BB_Lower_Touch",       "RSI_Oversold"],           "direction": "Bullish"},
    "BB_RSI_Overbought_Reject":    {"signals": ["BB_Upper_Touch",       "RSI_Overbought"],         "direction": "Bearish"},
    "BB_Stoch_Bull":               {"signals": ["BB_Lower_Bounce",      "Stoch_Oversold"],         "direction": "Bullish"},
    "BB_Stoch_Bear":               {"signals": ["BB_Upper_Reject",      "Stoch_Overbought"],       "direction": "Bearish"},
    "BB_MACD_Bounce":              {"signals": ["BB_Lower_Touch",       "MACD_Cross_Up_Event"],    "direction": "Bullish"},
    "Volume_MACD_Bull":            {"signals": ["Volume_Spike",         "MACD_Cross_Up_Event"],    "direction": "Bullish"},
    "Volume_EMA_Cross_Bull":       {"signals": ["Volume_Spike",         "EMA_Cross_Up_Event"],     "direction": "Bullish"},
    "Volume_Golden_Cross":         {"signals": ["Volume_Spike",         "Golden_Cross_Event"],     "direction": "Bullish"},
    "Volume_CMF_Bull":             {"signals": ["Volume_Spike",         "CMF_Strong_Bull"],        "direction": "Bullish"},
    "Volume_OBV_Bull":             {"signals": ["Volume_Spike",         "OBV_Trend_Bull"],         "direction": "Bullish"},
    "RSI_MACD_EMA_Bull":           {"signals": ["RSI_Oversold", "MACD_Bullish_State",  "EMA_Bullish_State"],   "direction": "Bullish"},
    "RSI_MACD_EMA_Bear":           {"signals": ["RSI_Overbought","MACD_Bearish_State", "EMA_Bearish_State"],   "direction": "Bearish"},
    "RSI_BB_MACD_Bull":            {"signals": ["RSI_Exit_Oversold","BB_Lower_Bounce", "MACD_Cross_Up_Event"], "direction": "Bullish"},
    "Golden_Cross_MACD_RSI":       {"signals": ["Golden_Cross_Event","MACD_Cross_Up_Event","RSI_Cross50_Up"],  "direction": "Bullish"},
    "Triple_Bear_Confirm":         {"signals": ["Death_Cross_Event","MACD_Cross_Down_Event","RSI_Cross50_Down"],"direction":"Bearish"},
    "Stoch_RSI_VWAP_Bull":         {"signals": ["Stoch_Oversold","RSI_Oversold",  "VWAP_Bull"],              "direction": "Bullish"},
    "Hammer_EMA_Bull":             {"signals": ["CDL_Hammer",    "EMA_Bullish_State"],                        "direction": "Bullish"},
    "MorningStar_MACD_Bull":       {"signals": ["CDL_MorningStar","MACD_Cross_Up_Event"],                     "direction": "Bullish"},
    "Engulfing_Bull_RSI":          {"signals": ["CDL_Engulfing_Bull","RSI_Oversold"],                         "direction": "Bullish"},
    "Engulfing_Bear_RSI":          {"signals": ["CDL_Engulfing_Bear","RSI_Overbought"],                       "direction": "Bearish"},
    "ShootingStar_EMA_Bear":       {"signals": ["CDL_ShootingStar","EMA_Bearish_State"],                      "direction": "Bearish"},
    "EveningStar_MACD_Bear":       {"signals": ["CDL_EveningStar","MACD_Cross_Down_Event"],                   "direction": "Bearish"},
    "ThreeWhiteSoldiers_OBV_Bull": {"signals": ["CDL_ThreeWhiteSoldiers","OBV_Trend_Bull"],                   "direction": "Bullish"},
    "ThreeBlackCrows_CMF_Bear":    {"signals": ["CDL_ThreeBlackCrows",  "CMF_Negative"],                      "direction": "Bearish"},
}

# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_candle_signals(df):
    signals = {}
    if not TALIB_AVAILABLE:
        return signals
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    for name, (fn, direction) in CANDLE_PATTERNS.items():
        try:
            raw = fn(o, h, l, c)
            signals[name] = (raw > 0) if "Bull" in name else (raw < 0) if "Bear" in name else (raw != 0)
        except Exception:
            signals[name] = pd.Series(False, index=df.index)
    return signals


def compute_all_signals(df):
    signal_map    = {n: s["fn"](df) for n, s in ATOMIC_SIGNALS.items()}
    direction_map = {n: s["direction"] for n, s in ATOMIC_SIGNALS.items()}
    for name, series in compute_candle_signals(df).items():
        signal_map[name]    = series
        direction_map[name] = CANDLE_PATTERNS[name][1]
    return signal_map, direction_map

# ─────────────────────────────────────────────────────────────────────────────
# FORWARD WINDOWS — close-to-close AND max favourable excursion (MFE)
# ─────────────────────────────────────────────────────────────────────────────

def add_forward_windows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Precompute for all horizons across all groups:
      Fwd_Close_{h}D  — close-to-close % return at exactly day h
      MFE_Bull_{h}D   — best HIGH achieved within next h sessions (bull MFE)
      MFE_Bear_{h}D   — worst LOW (as positive %) within next h sessions (bear MFE)

    MFE is the most important metric: it tells you whether the signal gave you
    the opportunity to take profit at any point in the window, regardless of
    where price ended up at day N.

    If the data has a "Segment" column (see fetch_technical_data.py's gap
    handling), any forward window that would span two segments -- i.e. cross
    a long real trading gap -- is nulled out. Segment ids are monotonically
    non-decreasing, so a window starting at row i and ending at row i+h spans
    a gap iff Segment[i] != Segment[i+h]; checking the endpoints is enough.
    Without this, a stock that went dormant for months and then resumed
    trading would still get a fake multi-hundred-percent "h-day return"
    computed across the gap.
    """
    df = df.copy()
    all_horizons = sorted(set(h for g in HORIZON_GROUPS.values() for h in g["horizons"]))
    has_segments = "Segment" in df.columns

    for h in all_horizons:
        df[f"Fwd_Close_{h}D"] = df["Close"].shift(-h) / df["Close"] - 1

        # Rolling max HIGH over the next h days
        # shift(-h) aligns the end of the window with the signal date
        df[f"MFE_Bull_{h}D"] = (
            df["High"].rolling(h).max().shift(-h) / df["Close"] - 1
        )
        # Rolling min LOW over the next h days → how far price dropped (positive = good for bears)
        df[f"MFE_Bear_{h}D"] = (
            1 - df["Low"].rolling(h).min().shift(-h) / df["Close"]
        )

        if has_segments:
            same_segment = df["Segment"] == df["Segment"].shift(-h)
            df[f"Fwd_Close_{h}D"] = df[f"Fwd_Close_{h}D"].where(same_segment)
            df[f"MFE_Bull_{h}D"] = df[f"MFE_Bull_{h}D"].where(same_segment)
            df[f"MFE_Bear_{h}D"] = df[f"MFE_Bear_{h}D"].where(same_segment)
    return df

# ─────────────────────────────────────────────────────────────────────────────
# HORIZON CLASS LOOKUP
# ─────────────────────────────────────────────────────────────────────────────

def get_horizon_class(signal_name: str) -> str:
    clean = signal_name.replace("STRAT_", "")
    if clean in STRATEGY_HORIZON_OVERRIDE:
        return STRATEGY_HORIZON_OVERRIDE[clean]
    # for auto-generated pair/triple names, use slowest component
    parts = [p.strip() for p in clean.split("+")]
    if len(parts) > 1:
        CLASS_RANK = {"candle": 0, "mean_rev": 1, "volume": 2, "momentum": 3, "trend": 4}
        classes = [SIGNAL_HORIZON_CLASS.get(p, "momentum") for p in parts]
        return max(classes, key=lambda c: CLASS_RANK.get(c, 3))
    return SIGNAL_HORIZON_CLASS.get(clean, "momentum")

# ─────────────────────────────────────────────────────────────────────────────
# SCORING — Profit Factor + Expected Value
# ─────────────────────────────────────────────────────────────────────────────

ALL_HORIZONS = sorted(set(h for g in HORIZON_GROUPS.values() for h in g["horizons"]))


def precompute_horizon_arrays(df: pd.DataFrame) -> Dict[int, Dict[str, np.ndarray]]:
    """Numpy arrays for every horizon, computed once per symbol instead of once
    per (combo x horizon) evaluation -- this is the main cost driver at scale,
    since the same MFE/close/threshold arrays were previously re-sliced out of
    the DataFrame (with pandas .loc/dropna overhead) for every single combo."""
    atr_pct = (df["ATR_14"] / df["Close"]).to_numpy()
    arrays = {}
    for h in ALL_HORIZONS:
        arrays[h] = {
            "mfe_bull":  df[f"MFE_Bull_{h}D"].to_numpy(),
            "mfe_bear":  df[f"MFE_Bear_{h}D"].to_numpy(),
            "fwd_close": df[f"Fwd_Close_{h}D"].to_numpy(),
            "min_move":  ATR_THRESHOLD_K * atr_pct * np.sqrt(h),
        }
    return arrays


def precompute_class_valid_masks(n: int) -> Dict[str, np.ndarray]:
    """Per-class boolean mask excluding the tail rows with no forward window
    (same restriction the old code applied via df.index[:-max_h])."""
    masks = {}
    for hclass, hgroup in HORIZON_GROUPS.items():
        max_h = max(hgroup["horizons"])
        valid = np.zeros(n, dtype=bool)
        valid[: n - max_h if n > max_h else n] = True
        masks[hclass] = valid
    return masks


def score_signal(mask_arr: np.ndarray, direction: str, signal_name: str,
                  horizon_arrays: Dict[int, Dict[str, np.ndarray]],
                  class_valid_masks: Dict[str, np.ndarray]) -> Optional[dict]:
    """
    Primary metrics (per horizon):
      WinRate_MFE  — % of occurrences where MFE exceeded the ATR-normalized threshold
      AvgWin       — average close-at-N return on winning occurrences
      AvgLoss      — average close-at-N return on losing occurrences
      EV           — Expected Value = win_rate × avg_win + loss_rate × avg_loss
      ProfitFactor — gross profit / gross loss
      CompositeScore_{h}D — per-horizon quality score (EV × max(PF,1) × log1p(count))

    Aggregate metrics (across all horizons in the signal's class):
      Consistency    — 1 - range of win rates across horizons
      CompositeScore — EV × max(PF,1) × consistency × log1p(count)
      BestHorizon    — horizon with the highest CompositeScore_{h}D

    CompositeScore (blended) is the robustness gate used for ranking/leaderboards
    — it penalizes a signal that only "worked" at one cherry-picked horizon.
    BestHorizon / CompositeScore_{h}D exist so a live scan can say *when* to
    expect the move, once a signal has already cleared that gate.
    """
    hclass   = get_horizon_class(signal_name)
    hgroup   = HORIZON_GROUPS.get(hclass, HORIZON_GROUPS["momentum"])
    horizons = hgroup["horizons"]
    valid_class = class_valid_masks.get(hclass, class_valid_masks["momentum"])

    combined_mask = mask_arr & valid_class
    count = int(combined_mask.sum())

    if count < MIN_OCCURRENCES:
        return None

    result = {"Count": count, "HorizonClass": hclass, "Direction": direction}
    all_evs, all_srs, all_pfs, horizon_composites = [], [], [], []
    mfe_key    = "mfe_bull" if direction == "Bullish" else "mfe_bear"
    multiplier = 1.0 if direction == "Bullish" else -1.0

    for h in horizons:
        harr = horizon_arrays.get(h)
        if harr is None:
            continue

        mfe_full      = harr[mfe_key]
        close_full    = harr["fwd_close"]
        min_move_full = harr["min_move"]

        valid_h = combined_mask & ~np.isnan(mfe_full) & ~np.isnan(close_full) & ~np.isnan(min_move_full)
        if int(valid_h.sum()) < MIN_OCCURRENCES:
            continue

        mfe_vals      = mfe_full[valid_h]
        close_vals    = close_full[valid_h]
        min_move_vals = min_move_full[valid_h]

        # Win = MFE exceeded this stock's own ATR- and horizon-scaled threshold
        win_mask  = mfe_vals >= min_move_vals
        win_rate  = float(win_mask.mean())
        loss_rate = 1.0 - win_rate

        # Realised returns (close-at-N) — signed for direction
        rv = close_vals * multiplier
        wins   = rv[win_mask]
        losses = rv[~win_mask]

        avg_win  = float(wins.mean())  if wins.size   > 0 else 0.0
        avg_loss = float(losses.mean()) if losses.size > 0 else 0.0

        ev = win_rate * avg_win + loss_rate * avg_loss

        neg_losses   = losses[losses < 0]
        gross_profit = float(wins[wins > 0].sum()) if wins.size > 0 else 0.0
        gross_loss   = float(np.abs(neg_losses).sum()) if neg_losses.size > 0 else 1e-9
        pf = gross_profit / gross_loss if gross_loss > 1e-9 else np.nan

        composite_h = round(ev * max(pf if not np.isnan(pf) else 0.0, 1.0) * np.log1p(count), 6)

        result[f"WinRate_MFE_{h}D"]     = round(win_rate, 4)
        result[f"AvgWin_{h}D"]          = round(avg_win, 5)
        result[f"AvgLoss_{h}D"]         = round(avg_loss, 5)
        result[f"EV_{h}D"]              = round(ev, 6)
        result[f"ProfitFactor_{h}D"]    = round(pf, 3) if not np.isnan(pf) else np.nan
        result[f"AvgReturn_Close_{h}D"] = round(float(close_vals.mean()), 5)
        result[f"AvgMinMovePct_{h}D"]   = round(float(min_move_vals.mean()) * 100, 3)
        result[f"CompositeScore_{h}D"]  = composite_h

        all_evs.append(ev)
        all_srs.append(win_rate)
        all_pfs.append(pf if not np.isnan(pf) else 0.0)
        horizon_composites.append((h, composite_h))

    if not all_evs:
        return None

    avg_ev   = float(np.mean(all_evs))
    avg_sr   = float(np.mean(all_srs))
    avg_pf   = float(np.mean([p for p in all_pfs if p > 0])) if any(p > 0 for p in all_pfs) else 0.0
    sr_range = max(all_srs) - min(all_srs) if len(all_srs) > 1 else 0.0
    consistency = max(1.0 - sr_range, 0.0)

    composite = round(avg_ev * max(avg_pf, 1.0) * consistency * np.log1p(count), 6)
    best_h, best_h_composite = max(horizon_composites, key=lambda t: t[1])

    result.update({
        "AvgEV":                round(avg_ev, 6),
        "AvgWinRate":           round(avg_sr, 4),
        "AvgProfitFactor":      round(avg_pf, 3),
        "Consistency":          round(consistency, 4),
        "CompositeScore":       composite,
        "BestHorizon":          best_h,
        "BestHorizonComposite": best_h_composite,
    })
    return result


def fisher_p_value(mask_arr: np.ndarray, direction: str,
                    horizon_arrays: Dict[int, Dict[str, np.ndarray]],
                    horizon: int, valid_class: np.ndarray) -> float:
    harr = horizon_arrays.get(horizon)
    if harr is None:
        return np.nan
    mfe_key       = "mfe_bull" if direction == "Bullish" else "mfe_bear"
    mfe_full      = harr[mfe_key]
    min_move_full = harr["min_move"]

    valid     = valid_class & ~np.isnan(mfe_full) & ~np.isnan(min_move_full)
    sig_valid = valid & mask_arr

    all_mfe, all_min_move = mfe_full[valid], min_move_full[valid]
    sig_mfe, sig_min_move = mfe_full[sig_valid], min_move_full[sig_valid]

    a = int((sig_mfe >= sig_min_move).sum())
    b = int((sig_mfe < sig_min_move).sum())
    c = max(int((all_mfe >= all_min_move).sum()) - a, 0)
    d = max(int((all_mfe < all_min_move).sum()) - b, 0)

    table = np.array([[a, b], [c, d]])
    if table.sum() == 0 or a + b == 0:
        return np.nan
    try:
        _, p = fisher_exact(table, alternative="greater")
        return round(p, 5)
    except Exception:
        return np.nan

# ─────────────────────────────────────────────────────────────────────────────
# COMBO BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_combo_mask(signal_names, signal_map_np: Dict[str, np.ndarray]) -> np.ndarray:
    n = len(next(iter(signal_map_np.values())))
    combined = np.ones(n, dtype=bool)
    for name in signal_names:
        if name in signal_map_np:
            combined = combined & signal_map_np[name]
        else:
            return np.zeros(n, dtype=bool)
    return combined

# ─────────────────────────────────────────────────────────────────────────────
# PER-SYMBOL PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_symbol(filepath: str, test_triples: bool = False) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(filepath, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    except Exception as e:
        if VERBOSE: print(f"[SKIP] {filepath}: {e}")
        return None

    if len(df) < 100:
        if VERBOSE: print(f"[SKIP] Too few rows ({len(df)}) in {filepath}")
        return None

    symbol  = df["Symbol"].iloc[0]  if "Symbol"  in df.columns else Path(filepath).stem
    company = df["Company"].iloc[0] if "Company" in df.columns else ""
    if VERBOSE: print(f"  → {symbol} ({len(df)} rows)")

    df = add_forward_windows(df)
    signal_map, direction_map = compute_all_signals(df)

    # Convert once per symbol: every combo evaluation below reuses these numpy
    # arrays directly instead of re-slicing the DataFrame (with pandas .loc /
    # dropna overhead) on every single call -- this was the actual bottleneck.
    signal_map_np     = {name: series.to_numpy() for name, series in signal_map.items()}
    horizon_arrays     = precompute_horizon_arrays(df)
    class_valid_masks  = precompute_class_valid_masks(len(df))

    bull_signals = [n for n, d in direction_map.items() if d == "Bullish"]
    bear_signals = [n for n, d in direction_map.items() if d == "Bearish"]
    rows = []

    def evaluate(name: str, signal_names: list, mask_arr: np.ndarray, direction: str):
        metrics = score_signal(mask_arr, direction, name, horizon_arrays, class_valid_masks)
        if metrics is None:
            return
        row = {
            "SignalName": name,
            "Components": " + ".join(signal_names),
            "Type": "single" if len(signal_names) == 1 else f"combo_{len(signal_names)}",
        }
        row.update(metrics)
        if COMPUTE_FISHER:
            hclass      = get_horizon_class(name)
            hgroup      = HORIZON_GROUPS.get(hclass, HORIZON_GROUPS["momentum"])
            valid_class = class_valid_masks.get(hclass, class_valid_masks["momentum"])
            ps = [fisher_p_value(mask_arr, direction, horizon_arrays, h, valid_class) for h in hgroup["horizons"]]
            valid_ps = [p for p in ps if not np.isnan(p)]
            row["Fisher_p_min"]    = round(min(valid_ps), 5) if valid_ps else np.nan
            row["Fisher_sig_flag"] = int(bool(valid_ps) and min(valid_ps) < FISHER_ALPHA)
        rows.append(row)

    # Singles
    for name in list(ATOMIC_SIGNALS.keys()) + list(CANDLE_PATTERNS.keys()):
        if name not in signal_map_np:
            continue
        direction = direction_map.get(name, "Both")
        if direction == "Both":
            continue
        evaluate(name, [name], signal_map_np[name], direction)

    # Predefined strategies
    for strat_name, spec in PREDEFINED_STRATEGIES.items():
        sig_names = spec["signals"]
        direction = spec["direction"]
        if not TALIB_AVAILABLE and any(s.startswith("CDL_") for s in sig_names):
            continue
        try:
            mask_arr = build_combo_mask(sig_names, signal_map_np)
            evaluate(f"STRAT_{strat_name}", sig_names, mask_arr, direction)
        except Exception:
            continue

    # Exhaustive pairs
    for direction, sig_list in [("Bullish", bull_signals), ("Bearish", bear_signals)]:
        active = [n for n in sig_list if n in signal_map_np and signal_map_np[n].sum() >= MIN_OCCURRENCES // 2]
        for s1, s2 in itertools.combinations(active, 2):
            mask_arr = signal_map_np[s1] & signal_map_np[s2]
            evaluate(f"{s1} + {s2}", [s1, s2], mask_arr, direction)

    # Exhaustive triples
    if test_triples:
        for direction, sig_list in [("Bullish", bull_signals), ("Bearish", bear_signals)]:
            active = sorted(
                [n for n in sig_list if n in signal_map_np and signal_map_np[n].sum() >= MIN_OCCURRENCES],
                key=lambda n: -signal_map_np[n].sum()
            )[:25]
            for s1, s2, s3 in itertools.combinations(active, 3):
                mask_arr = signal_map_np[s1] & signal_map_np[s2] & signal_map_np[s3]
                evaluate(f"{s1} + {s2} + {s3}", [s1, s2, s3], mask_arr, direction)

    if not rows:
        if VERBOSE: print(f"  [SKIP] No valid signals for {symbol}")
        return None

    result_df = pd.DataFrame(rows)
    result_df["Symbol"]  = symbol
    result_df["Company"] = company
    result_df = result_df.sort_values("CompositeScore", ascending=False)

    bull_top = result_df[result_df["Direction"] == "Bullish"].head(TOP_N_PER_DIR)
    bear_top = result_df[result_df["Direction"] == "Bearish"].head(TOP_N_PER_DIR)
    top_df   = pd.concat([bull_top, bear_top], ignore_index=True)

    out_path = os.path.join(OUTPUT_FOLDER, f"{symbol}_signals.csv")
    top_df.to_csv(out_path, index=False)
    if VERBOSE: print(f"  ✓ Saved {out_path}  ({len(top_df)} signals)")
    return top_df

# ─────────────────────────────────────────────────────────────────────────────
# CROSS-SYMBOL LEADERBOARD
# ─────────────────────────────────────────────────────────────────────────────

def build_leaderboard(all_results: list) -> pd.DataFrame:
    if not all_results:
        return pd.DataFrame()
    combined      = pd.concat(all_results, ignore_index=True)
    total_symbols = combined["Symbol"].nunique()
    agg = combined.groupby(["SignalName", "Direction", "HorizonClass"]).agg(
        Symbol_Count      = ("Symbol",         "nunique"),
        Avg_Composite     = ("CompositeScore",  "mean"),
        Avg_EV            = ("AvgEV",           "mean"),
        Avg_WinRate       = ("AvgWinRate",      "mean"),
        Avg_ProfitFactor  = ("AvgProfitFactor", "mean"),
        Avg_Consistency   = ("Consistency",     "mean"),
        Avg_Count         = ("Count",           "mean"),
    ).reset_index()
    agg["Prevalence_%"] = (agg["Symbol_Count"] / total_symbols * 100).round(1)
    agg = agg.sort_values("Avg_Composite", ascending=False)

    out_path = os.path.join(OUTPUT_FOLDER, "NSE_signal_leaderboard.csv")
    agg.to_csv(out_path, index=False)
    if VERBOSE: print(f"\n✓ Leaderboard → {out_path}")
    return agg

# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NSE Signal Scanner v2")
    parser.add_argument("--symbol",    type=str, default=None, help="Single symbol only")
    parser.add_argument("--triples",   action="store_true",    help="Test triple combos (slow)")
    parser.add_argument("--no-fisher", action="store_true",    help="Skip Fisher p-values")
    parser.add_argument("--workers",   type=int, default=None, help="Parallel worker processes (default: CPU count)")
    args = parser.parse_args()

    global COMPUTE_FISHER
    if args.no_fisher:
        COMPUTE_FISHER = False

    if args.symbol:
        fp = os.path.join(INPUT_FOLDER, f"{args.symbol}{DATA_SUFFIX}")
        if not os.path.exists(fp):
            print(f"File not found: {fp}")
            return
        run_symbol(fp, test_triples=args.triples)
        return

    files = sorted(glob.glob(os.path.join(INPUT_FOLDER, f"*{DATA_SUFFIX}")))
    if not files:
        print(f"No files found in {INPUT_FOLDER}")
        return

    n_workers = args.workers or os.cpu_count() or 1
    print(f"Processing {len(files)} symbols across {n_workers} worker processes...\n")

    all_results = []
    done = 0
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(run_symbol, fp, args.triples): fp for fp in files}
        for future in as_completed(futures):
            fp = futures[future]
            done += 1
            try:
                result = future.result()
            except Exception as e:
                print(f"[{done}/{len(files)}] [ERROR] {os.path.basename(fp)}: {e}")
                continue
            if result is not None:
                all_results.append(result)
            if done % 25 == 0 or done == len(files):
                print(f"[{done}/{len(files)}] processed")

    print(f"\nBuilding leaderboard from {len(all_results)} symbols...")
    lb = build_leaderboard(all_results)
    if not lb.empty:
        print("\n── Top 15 Universal Bull Signals ──")
        print(lb[lb["Direction"] == "Bullish"].head(15).to_string(index=False))
        print("\n── Top 15 Universal Bear Signals ──")
        print(lb[lb["Direction"] == "Bearish"].head(15).to_string(index=False))


if __name__ == "__main__":
    main()