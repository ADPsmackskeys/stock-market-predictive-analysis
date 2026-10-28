"""
Backtest fresh bullish signals with independent T1, T2, and T3 target strategies.
Each strategy uses the same signal, next-session-open entry, and holding horizon.

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
three strategies share the same signals and first entries; each then manages its
own position, target and holding horizon (see SCALING IN).

A target is a resting limit order: it fills the moment the session trades at
the level, whether or not price closed beyond it (see get_target_hit_date in
live_signal_scan.py for the worked SKYGOLD example). So a hit is High >= level
within the holding window. On a session that gapped straight through, the fill
is that session's Open, which is better than the level -- not the level itself.

Holding window is rows s+1 .. s+BestHorizon, matching get_mfe_so_far(): the
quantiles are measured over the h rows AFTER the firing bar, and entering at
s+1's open means every one of those rows' highs is capturable.

TRAILING STOP
-------------
Every trade also carries a trailing stop-loss TRAILING_STOP_PCT below the
highest high since entry (initialised at the entry price). The holding window
is walked day by day. On each session the stop is checked BEFORE the target
and BEFORE that session's high is added to the watermark:

    tsl_level = highest_high * (1 - TRAILING_STOP_PCT)
    Open  < tsl_level  -> exit at the Open (gap-down, worst-case fill)
    Low  <= tsl_level  -> exit at tsl_level
    High >= target     -> exit at target (or at the Open if it gapped through)
    then highest_high = max(highest_high, High)

If one session's range touches both the stop and the target, the stop wins
(pessimistic). The stop only ever ratchets up; it never moves down.

SCALING IN
----------
A stock has at most one open position per strategy (T1, T2, T3 and every
portfolio size each keep their own). A fresh bullish signal then means:

  * no open position -> ordinary entry at the next open, if it ranks in the
    session's top N;
  * open position    -> it buys MORE only if its CompositeScore is strictly
    higher than the score of the signal the position last acted on (the opening
    signal at first) AND its target is a higher price than the active target.
    If both hold:
        - another ~CAPITAL_PER_ENTRY is bought at the next session's open
          (same whole-share rule as an entry), regardless of top-N rank;
        - the target moves to that signal's target (same ladder rung: T1 uses
          Target1, T2 Target2, T3 Target3; never switched), anchored to ITS
          signal-bar close;
        - the horizon end moves to that signal's row + its BestHorizon (never
          earlier than the current end);
        - its score becomes the one later signals are compared against.
    The exit (target, trailing stop or horizon) sells EVERY lot at one price.

Timing: a signal on row s is only known at that bar's close, so it affects
sessions s+1 onward. The position is first walked through row s with the usual
rules; if it already left the market on or before row s it is never touched
retroactively, and the signal is an ordinary new-entry candidate. Every signal
seen while holding is logged, with its outcome and reason, in
results/target_adjustments_t{k}_top{n}.csv.

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

# ── Position sizing ───────────────────────────────────────────────────────
# Fixed capital per entry: every trade targets this much capital, rounded to
# a whole number of shares. See position_size() for the exact rule.
CAPITAL_PER_ENTRY = 10_000

# Trailing stop-loss distance below the highest high since entry (10.5%).
TRAILING_STOP_PCT = 1

# Outcome labels. Kept as constants so the summary and any downstream filter
# agree on spelling instead of matching string literals in two places.
OUT_TARGET_HIT   = "target_hit"
OUT_HORIZON      = "horizon_close"      # window elapsed, target never reached
OUT_TSL_HIT      = "trailing_stop_hit"  # trailing stop-loss triggered first
OUT_NO_TARGET    = "no_target_above_entry"
OUT_STILL_OPEN   = "still_open"         # horizon extends past the data we have
OUT_NO_ENTRY_BAR = "no_entry_bar"       # fired on the final row; no next open

# Outcomes that carry a realised return (the trade was entered and exited).
RESOLVED_OUTCOMES = [OUT_TARGET_HIT, OUT_TSL_HIT, OUT_HORIZON]


def since_index_all(combined: np.ndarray) -> np.ndarray:
    """get_since_index() for every row at once.

    For each t where combined[t] is True, the episode containing t started at
    (index of the last False at or before t) + 1. A running maximum over the
    False positions gives that for all t in one pass. Values at rows where
    combined is False are meaningless and must be masked by the caller.
    """
    idx = np.arange(len(combined))
    return np.maximum.accumulate(np.where(~combined, idx, -1)) + 1


# Columns the simulation pass needs from each price file.
_PRICE_COLUMNS = {"Date", "Open", "High", "Low", "Close", "Segment"}

# RejectionReason values for a signal that arrives while a position is open.
REJ_SCORE       = "score_not_higher"        # CompositeScore <= governing score
REJ_NO_ESTIMATE = "no_target_estimate"      # signal has no target for this strategy
REJ_LOWER       = "target_not_higher"       # new target <= the active target
REJ_NO_BAR      = "no_next_session"         # signal on the last row: nothing to buy at
REJ_STOP        = "stopped_at_next_open"    # trailing stop triggers at that same open
REJ_TARGET      = "target_not_above_entry"  # new target <= the add-on's buy price
REJ_CLOSED      = "position_closed_"        # + outcome that closed it on the signal bar


def position_size(price: float) -> float:
    """Whole shares for ONE purchase of ~CAPITAL_PER_ENTRY.

    round(CAPITAL_PER_ENTRY / price), half rounded up, never below 1; exactly
    1 share when a single share already costs more than CAPITAL_PER_ENTRY.
    """
    if not np.isfinite(price) or price <= 0:
        return np.nan
    if price > CAPITAL_PER_ENTRY:
        return 1.0
    return float(max(1.0, np.floor(CAPITAL_PER_ENTRY / price + 0.5)))


def _row_date(dates, row: int, n: int):
    """Date of a row index, or NaT when the row lies beyond the data we have."""
    return dates[row] if 0 <= row < n else np.datetime64("NaT")


def _trade_id(sig: dict, dates, s: int) -> str:
    return f"{sig['Symbol']}_{pd.Timestamp(dates[s]).strftime('%Y%m%d')}"


def _signal_fields(sig: dict, s: int, dates, closes) -> dict:
    """Metadata of the signal that opened a trade (shared by every trade row)."""
    return {
        "Symbol": sig["Symbol"],
        "SignalName": sig["SignalName"],
        "Direction": str(sig.get("Direction", "")),
        "SignalDate": dates[s],
        "SignalClose": float(closes[s]),
        "BestHorizon": int(sig["BestHorizon"]),
        "CompositeScore": float(sig["CompositeScore"]),
        "Count": sig["Count"],
        "RankInSession": int(sig["RankInSession"]),
        "TradeID": _trade_id(sig, dates, s),
    }


def open_position(sig: dict, s: int, opens, closes, n: int, target_num: int):
    """Try to open a position for `sig`, entering at the open of row s+1.

    Returns (position, None) when a position is opened, otherwise
    (None, terminal_fields) saying why there is no position. Entry, target and
    validity rules are unchanged: the target ladder is anchored to the signal
    bar's close, and a target at or below the entry price is unusable.
    """
    signal_close = float(closes[s])
    e = s + 1
    if e > n - 1:
        return None, {"Outcome": OUT_NO_ENTRY_BAR}

    entry = opens[e]
    if (not np.isfinite(entry) or entry <= 0
            or not np.isfinite(signal_close) or signal_close <= 0):
        return None, {"Outcome": OUT_NO_ENTRY_BAR}

    gap_pct = float(entry) / signal_close - 1.0
    target_pct = sig[f"Target{target_num}Pct"]
    if pd.isna(target_pct):
        return None, {
            "Outcome": OUT_NO_TARGET, "Entry": float(entry), "GapPct": gap_pct,
            "TargetUsed": target_num, "TargetPct": np.nan, "TargetLevel": np.nan,
        }

    target_pct = float(target_pct)
    level = signal_close * (1.0 + target_pct)
    if not np.isfinite(level) or level <= entry:
        return None, {
            "Outcome": OUT_NO_TARGET, "Entry": float(entry), "GapPct": gap_pct,
            "TargetUsed": target_num, "TargetPct": target_pct, "TargetLevel": level,
        }

    return {
        "s": s,
        "e": e,
        "entry": float(entry),               # first lot's price
        "gap_pct": gap_pct,
        "target_num": target_num,
        "level": level,                      # ACTIVE absolute target price
        "target_pct": target_pct,
        "initial_level": level,
        "initial_pct": target_pct,
        "gov_score": float(sig["CompositeScore"]),   # score of the last signal bought on
        "gov_s": s,                          # row of that signal
        "w_end_ideal": s + int(sig["BestHorizon"]),  # may lie beyond the data
        "next_row": e,                       # next session still to evaluate
        "highest_high": float(entry),        # trailing-stop watermark
        # Every purchase of this position. The exit sells all of them.
        "lots": [{"row": e, "price": float(entry),
                  "shares": position_size(float(entry)), "sig_s": s}],
        "adjustments": 0,                    # accepted add-ons
        "proposals": 0,                      # signals seen while open
        "sig": sig,
    }, None


def advance_position(pos: dict, upto: int, opens, highs, lows, closes, n: int,
                     final: bool = False):
    """Walk an open position forward, session by session, through row `upto`.

    Returns an exit dict as soon as the position leaves the market, else None
    (it is still open after row `upto`). Rules per session, in this order:

        1. trailing stop (gap-down fills at the Open, else at the stop level)
        2. target (fills at the level, or at the Open if it gapped through)
        3. only then does the session's High ratchet the stop watermark up

    The stop is checked first, so a session touching both stop and target
    counts as a stop. Reaching the end of the holding window without an exit
    closes the trade at that session's close (horizon_close); if the window
    runs past the data, the trade is still_open and is only reported as such
    when `final` is True (the end of the data).

    The target level and window end are read from `pos` on every call, so an
    accepted add-on takes effect from the session AFTER the signal bar. The
    exit price applies to every lot, so no per-lot return is computed here.
    """
    w_end = min(n - 1, pos["w_end_ideal"])
    truncated = pos["w_end_ideal"] > n - 1
    last = min(upto, w_end)
    level, e = pos["level"], pos["e"]
    hh = pos["highest_high"]
    r = pos["next_row"]

    def leave(outcome, row, price):
        return {"Outcome": outcome, "ExitRow": row, "ExitPrice": float(price),
                "SessionsHeld": row - e}

    while r <= last:
        o, h, l = opens[r], highs[r], lows[r]
        # Stop level from PRIOR sessions' highs only; the watermark is updated
        # at the bottom, after the stop and target have been checked.
        tsl_level = hh * (1.0 - TRAILING_STOP_PCT)

        if np.isfinite(o) and o < tsl_level:
            return leave(OUT_TSL_HIT, r, o)
        if np.isfinite(l) and l <= tsl_level:
            return leave(OUT_TSL_HIT, r, tsl_level)
        if np.isfinite(h) and h >= level:
            fill = o if (np.isfinite(o) and o > level) else level
            return leave(OUT_TARGET_HIT, r, fill)
        if np.isfinite(h) and h > hh:
            hh = float(h)
        r += 1

    pos["next_row"] = r
    pos["highest_high"] = hh
    if r <= w_end:
        return None                          # window not finished yet
    if truncated:
        return leave(OUT_STILL_OPEN, w_end, closes[w_end]) if final else None
    return leave(OUT_HORIZON, w_end, closes[w_end])


def _closed_trade(pos: dict, ex: dict, n: int, segments, dates, closes) -> dict:
    """Trade row for a position that has left the market (or is still open).

    Everything bought is sold at the single exit price, so the position's P&L is
    sum(shares) * exit - sum(invested) and its return is that over what was
    invested. With one lot this equals exit / entry - 1.
    """
    w_end = min(n - 1, pos["w_end_ideal"])
    shares = float(sum(l["shares"] for l in pos["lots"]))
    invested = float(sum(l["shares"] * l["price"] for l in pos["lots"]))
    abs_pnl = shares * ex["ExitPrice"] - invested
    return {
        **_signal_fields(pos["sig"], pos["s"], dates, closes),
        **ex,
        "Return": abs_pnl / invested,
        "Entry": pos["entry"],                       # first lot
        "AvgEntry": invested / shares,
        "Shares": shares,
        "InvestedAmount": invested,
        "AbsPnL": abs_pnl,
        "Lots": len(pos["lots"]),
        "GapPct": pos["gap_pct"],
        "TargetUsed": pos["target_num"],
        # Active (final) target, plus what the opening signal asked for.
        "TargetPct": pos["target_pct"],
        "TargetLevel": pos["level"],
        "InitialTargetPct": pos["initial_pct"],
        "InitialTargetLevel": pos["initial_level"],
        "FinalGoverningScore": pos["gov_score"],
        "GoverningSignalDate": dates[pos["gov_s"]],
        "AddOns": pos["adjustments"],
        "ProposedAddOns": pos["proposals"],
        "WindowEndRow": w_end,
        "SegmentCrossed": bool(
            segments is not None and segments[pos["e"]] != segments[w_end]
        ),
        "EntryDate": dates[pos["e"]],
        "ExitDate": dates[ex["ExitRow"]],
    }


def _lot_records(pos: dict, dates) -> list:
    """One row per purchase, for the ENTRY rows of the trader log."""
    tid = _trade_id(pos["sig"], dates, pos["s"])
    return [{
        "TradeID": tid,
        "Symbol": pos["sig"]["Symbol"],
        "LotNo": i,
        "TriggerSignalDate": dates[l["sig_s"]],
        "EntryDate": dates[l["row"]],
        "EntryPrice": l["price"],
        "Shares": l["shares"],
        "Invested": l["shares"] * l["price"],
    } for i, l in enumerate(pos["lots"], start=1)]


def _adj_record(sig, s, target_num, top_n, pos, dates, n, proposed, accepted,
                reason, new_w_ideal, lot=None) -> dict:
    """One row of the add-on log (every signal seen while a position was open)."""
    return {
        "Strategy": f"T{target_num}",
        "TopN": top_n,
        "Symbol": sig["Symbol"],
        "TradeID": _trade_id(pos["sig"], dates, pos["s"]),
        "SignalDate": dates[s],
        "SignalRank": int(sig["RankInSession"]),
        "InTopN": bool(int(sig["RankInSession"]) <= top_n),
        "NewCompositeScore": float(sig["CompositeScore"]),
        "GoverningScore": pos["gov_score"],
        "PreviousTarget": pos["level"],
        "ProposedTarget": proposed,
        "Accepted": bool(accepted),
        "RejectionReason": reason,
        "PreviousHorizonEnd": _row_date(dates, pos["w_end_ideal"], n),
        "NewHorizonEnd": _row_date(dates, new_w_ideal, n),
        "AddOnDate": dates[lot["row"]] if lot else np.datetime64("NaT"),
        "AddOnPrice": lot["price"] if lot else np.nan,
        "AddOnShares": lot["shares"] if lot else np.nan,
        "AddOnAmount": lot["shares"] * lot["price"] if lot else np.nan,
    }


def simulate_symbol_strategy(opens, highs, lows, closes, segments, dates,
                             sigs: list, target_num: int, top_n: int):
    """One symbol, one target strategy, one portfolio size.

    `sigs` is that symbol's fresh bullish signals (best one per session) in
    chronological order. At most one position is open at a time:

      * no open position -> an ordinary new entry at the next open, only when
        the signal ranks inside the session's top-N;
      * open position    -> the signal buys MORE only if its CompositeScore is
        strictly higher than the score of the signal the position last acted
        on AND its target is a higher price than the active target. Then, whatever its rank (top-N limits new positions only): another
        ~CAPITAL_PER_ENTRY is bought at the next open, and target, horizon end
        and governing score move to the new signal. The exit sells every lot.

    A signal on row s is only knowable at that bar's close, so it affects
    sessions s+1 onward. The position is therefore walked through row s first
    (including any exit on row s itself); a position that already left the
    market is never touched retroactively.

    Returns (trades, adjustments, lots), three lists of dicts.
    """
    n = len(opens)
    tkey = f"Target{target_num}Pct"
    trades, adjs, lots = [], [], []
    pos = None

    def finish(pos, ex):
        trades.append(_closed_trade(pos, ex, n, segments, dates, closes))
        lots.extend(_lot_records(pos, dates))

    for sig in sigs:
        s = sig["s"]

        if pos is not None:
            ex = advance_position(pos, s, opens, highs, lows, closes, n)
            if ex is not None:
                finish(pos, ex)
                if ex["ExitRow"] == s:
                    # Left the market on the signal's own bar: too late to
                    # influence it. The signal is then a normal entry candidate.
                    pct = sig[tkey]
                    adjs.append(_adj_record(
                        sig, s, target_num, top_n, pos, dates, n,
                        proposed=(closes[s] * (1.0 + pct)) if pd.notna(pct) else np.nan,
                        accepted=False, reason=REJ_CLOSED + ex["Outcome"],
                        new_w_ideal=pos["w_end_ideal"]))
                pos = None

        if pos is not None:
            # ── A signal while holding: add to the position, or not ───────
            pct = sig[tkey]
            score = float(sig["CompositeScore"])
            proposed = float(closes[s]) * (1.0 + float(pct)) if pd.notna(pct) else np.nan
            new_w_ideal = max(pos["w_end_ideal"], s + int(sig["BestHorizon"]))
            pos["proposals"] += 1

            buy_row = s + 1
            buy_px = float(opens[buy_row]) if buy_row <= n - 1 else np.nan
            tsl_level = pos["highest_high"] * (1.0 - TRAILING_STOP_PCT)

            if not score > pos["gov_score"]:
                reason = REJ_SCORE
            elif not np.isfinite(proposed):
                reason = REJ_NO_ESTIMATE
            elif not proposed > pos["level"]:
                reason = REJ_LOWER           # never add to lower the target
            elif buy_row > n - 1 or not np.isfinite(buy_px) or buy_px <= 0:
                reason = REJ_NO_BAR
            elif buy_px < tsl_level:
                reason = REJ_STOP            # position is stopped out at that open
            elif not proposed > buy_px:
                reason = REJ_TARGET          # same rule as a new entry
            else:
                reason = ""

            lot = None
            if not reason:
                lot = {"row": buy_row, "price": buy_px,
                       "shares": position_size(buy_px), "sig_s": s}
            adjs.append(_adj_record(
                sig, s, target_num, top_n, pos, dates, n,
                proposed=proposed, accepted=not reason, reason=reason,
                new_w_ideal=new_w_ideal if not reason else pos["w_end_ideal"],
                lot=lot))

            if not reason:
                # Entry of the first lot is untouched; this lot is extra.
                pos["lots"].append(lot)
                pos["level"] = proposed
                pos["target_pct"] = float(pct)
                pos["gov_score"] = score
                pos["gov_s"] = s
                pos["w_end_ideal"] = new_w_ideal
                pos["adjustments"] += 1
            continue                         # never a second position

        # ── No open position: ordinary new-entry logic ────────────────────
        if int(sig["RankInSession"]) > top_n:
            continue
        pos, terminal = open_position(sig, s, opens, closes, n, target_num)
        if terminal is not None:
            trades.append({
                **_signal_fields(sig, s, dates, closes),
                **terminal,
                "AddOns": 0,
                "ProposedAddOns": 0,
                "EntryDate": _row_date(dates, s + 1, n),
                "ExitDate": np.datetime64("NaT"),
            })

    if pos is not None:
        ex = advance_position(pos, n - 1, opens, highs, lows, closes, n, final=True)
        finish(pos, ex)

    return trades, adjs, lots


def simulate_symbol(job: tuple) -> dict:
    """Pass 2 worker: simulate every (strategy, portfolio size) for one symbol.

    Returns {(target_num, top_n): (trades, adjustments, lots)}.
    """
    tech_path, sig_records, top_ns = job
    try:
        df = (pd.read_csv(tech_path, parse_dates=["Date"],
                          usecols=lambda c: c in _PRICE_COLUMNS)
              .sort_values("Date").reset_index(drop=True))
    except Exception:
        return {}
    n = len(df)
    if n == 0:
        return {}

    opens   = df["Open"].to_numpy(dtype=float)
    highs   = df["High"].to_numpy(dtype=float)
    lows    = df["Low"].to_numpy(dtype=float)
    closes  = df["Close"].to_numpy(dtype=float)
    dates   = df["Date"].to_numpy()
    segments = df["Segment"].to_numpy() if "Segment" in df.columns else None

    sigs = []
    for rec in sig_records:
        d = pd.Timestamp(rec["SignalDate"]).to_datetime64()
        s = int(np.searchsorted(dates, d))
        if s < n and dates[s] == d:
            sigs.append({**rec, "s": s})
    sigs.sort(key=lambda x: x["s"])

    out = {}
    for target_num in (1, 2, 3):
        for top_n in top_ns:
            if not any(sg["RankInSession"] <= top_n for sg in sigs):
                out[(target_num, top_n)] = ([], [], [])   # never buys
                continue
            out[(target_num, top_n)] = simulate_symbol_strategy(
                opens, highs, lows, closes, segments, dates,
                sigs, target_num, top_n)
    return out


def scan_symbol_episodes(job: tuple) -> list:
    """Pass 1 worker: every fresh bullish signal for one symbol.

    Returns one row per (symbol, session) -- the best-scoring signal that fired
    -- with its three target estimates. Nothing is resolved here any more: whether
    a signal opens a trade or only proposes a target adjustment depends on which
    positions are open, which depends on the strategy, the portfolio size and
    the other symbols' ranks. That is decided in pass 2 (simulate_symbol).
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
    closes  = df["Close"].to_numpy(dtype=float)
    dates   = df["Date"].to_numpy()

    # Only episodes firing at or after the backtest start are tradeable, but
    # `combined` still has to be built over the FULL history: an episode that
    # began in 2023 and was still true in 2024 did not fire in 2024, and
    # truncating the array first would mislabel its first in-range row as a
    # fresh start.
    in_range = df["Date"].to_numpy() >= np.datetime64(start_date)

    best_per_row = {}   # row -> best-scoring signal that fired on that row
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
        tpcts = (row.get("Target1_Conservative_BestHorizon", np.nan),
                 row.get("Target2_Median_BestHorizon", np.nan),
                 row.get("Target3_Stretch_BestHorizon", np.nan))
        if all(pd.isna(t) for t in tpcts):
            continue

        for s in np.flatnonzero(starts):
            s = int(s)
            prev = best_per_row.get(s)
            if prev is not None and prev["CompositeScore"] >= composite:
                continue
            best_per_row[s] = {
                "Symbol": row["Symbol"],
                "SignalName": row["SignalName"],
                "Direction": str(row.get("Direction", "")),
                "SignalDate": dates[s],
                "SignalClose": float(closes[s]),
                "BestHorizon": int(row["BestHorizon"]),
                "CompositeScore": float(composite),
                "Count": row["Count"],
                "Target1Pct": tpcts[0],
                "Target2Pct": tpcts[1],
                "Target3Pct": tpcts[2],
            }

    return list(best_per_row.values())


def add_position_sizing(df: pd.DataFrame) -> pd.DataFrame:
    """Add Shares, InvestedAmount and AbsPnL columns (Fixed Capital Per Entry).

    Shares         = round(CAPITAL_PER_ENTRY / Entry), rounded half-up to the
                     nearest whole share; exactly 1 share when
                     Entry > CAPITAL_PER_ENTRY (the one share already costs more
                     than the budget).
    InvestedAmount = Shares * Entry
    AbsPnL         = InvestedAmount * Return  (== Shares * (ExitPrice - Entry))

    Rows without a usable Entry get NaN. AbsPnL is NaN wherever Return is NaN
    (never entered / no realised return). Idempotent: safe to call twice.
    """
    out = df.copy()
    if {"Shares", "InvestedAmount", "AbsPnL"}.issubset(out.columns):
        return out          # the simulation sized every lot already
    entry = pd.to_numeric(out["Entry"], errors="coerce")
    valid = np.isfinite(entry) & (entry > 0)
    safe_entry = entry.where(valid)

    shares = np.floor(CAPITAL_PER_ENTRY / safe_entry + 0.5)
    shares = shares.where(~(entry > CAPITAL_PER_ENTRY), 1.0)
    shares = shares.clip(lower=1.0)          # never a zero-share position
    shares = shares.where(valid)

    out["Shares"] = shares
    out["InvestedAmount"] = shares * entry.where(valid)
    out["AbsPnL"] = out["InvestedAmount"] * pd.to_numeric(out["Return"], errors="coerce")
    return out


def headline_stats(sel: pd.DataFrame) -> dict:
    """The numbers the comparison table shows, for one portfolio size.

    Only target_hit and horizon_close carry a realised return:
    no_target_above_entry was never entered, still_open has not finished, and
    no_entry_bar never started. Averaging those in would read unfinished
    trades as losses.
    """
    res = add_position_sizing(
        sel[sel["Outcome"].isin(RESOLVED_OUTCOMES)]
    )
    if res.empty:
        return {"trades": len(sel), "resolved": 0}
    return {
        "trades":   len(sel),
        "resolved": len(res),
        "reached":  float((res["Outcome"] == OUT_TARGET_HIT).mean()),
        "stopped":  float((res["Outcome"] == OUT_TSL_HIT).mean()),
        "mean":     float(res["Return"].mean()),
        "median":   float(res["Return"].median()),
        "std":      float(res["Return"].std()),
        "mean_pnl":   float(res["AbsPnL"].mean()),
        "median_pnl": float(res["AbsPnL"].median()),
        "total_pnl":  float(res["AbsPnL"].sum()),
        "sessions": float(res["SessionsHeld"].mean()),
        "symbols":  int(sel["Symbol"].nunique()),
        "added":    float((res["AddOns"] > 0).mean()),
    }


def comparison_table(rows: list, target_label: str = "T1") -> None:
    """One line per portfolio size. This is the headline of a sweep -- the
    per-size detail below it is the same data broken out."""
    width = 130
    print("\n" + "=" * width)
    print(f"COMPARISON ACROSS PORTFOLIO SIZES — {target_label} STRATEGY")
    print("=" * width)
    print(f"\n{'N':>4} {'trades':>8} {'resolved':>8} {'reached':>8} {'stopped':>8} {'added':>7} "
          f"{'mean %':>8} {'mean amt':>10} {'median %':>9} {'median amt':>11} "
          f"{'std':>7} {'total PnL':>14} {'sess':>5} {'syms':>6}")
    print("-" * width)
    for n, st in rows:
        if not st.get("resolved"):
            print(f"{n:>4} {st['trades']:>8,}        0   (no resolved trades)")
            continue
        print(f"{n:>4} {st['trades']:>8,} {st['resolved']:>8,} "
              f"{st['reached']*100:>7.1f}% {st['stopped']*100:>7.1f}% "
              f"{st['added']*100:>6.1f}% {st['mean']*100:>+7.3f}% "
              f"{st['mean_pnl']:>+10,.2f} {st['median']*100:>+8.3f}% "
              f"{st['median_pnl']:>+11,.2f} {st['std']*100:>6.2f}% "
              f"{st['total_pnl']:>+14,.2f} {st['sessions']:>5.1f} {st['symbols']:>6,}")
    print(f"\n'reached' = target hit, 'stopped' = trailing stop "
          f"({TRAILING_STOP_PCT*100:.1f}%) hit; the remainder expired at the horizon.")
    print("'added' = share of resolved positions that were added to at least once by a "
          "later, higher-scoring signal.")
    print("Returns and amounts are per POSITION: all lots together, sold at one exit.")
    print(f"Sizing: fixed capital per entry = {CAPITAL_PER_ENTRY:,} (whole shares), "
          f"gross of costs.")
    print("mean/median % are per-trade returns; 'mean amt'/'median amt' are the same")
    print("per-trade results in currency. 'total PnL' is the sum of absolute profit and")
    print("loss over resolved trades -- it scales with trade count, so compare 'mean'")
    print("across sizes, not 'total PnL'.")


def print_add_on_summary(sel: pd.DataFrame, adj) -> None:
    """Console summary of the signals seen while holding (adds and rejections)."""
    print("\nSignals while holding (add-on candidates):")
    if adj is None or adj.empty:
        print("  none")
        return
    acc = adj["Accepted"].astype(bool)
    in_top = adj["InTopN"].astype(bool)
    print(f"  seen {len(adj):,} | bought more {acc.sum():,} | skipped {(~acc).sum():,}")
    reasons = adj.loc[~acc, "RejectionReason"].astype(str)
    reasons = reasons.where(~reasons.str.startswith(REJ_CLOSED), "position_already_closed")
    for reason, cnt in reasons.value_counts().items():
        print(f"    skipped - {reason:<24} {cnt:>7,}")
    print(f"  from stocks outside that session's top N: {int((~in_top).sum()):,} "
          f"(bought more on {int((acc & ~in_top).sum()):,})")
    if acc.any():
        print(f"  total added: {adj.loc[acc, 'AddOnAmount'].sum():,.2f} "
              f"(avg {adj.loc[acc, 'AddOnAmount'].mean():,.2f} per add-on)")
    entered = sel[sel["Outcome"].isin(RESOLVED_OUTCOMES + [OUT_STILL_OPEN])]
    if len(entered):
        k = int((entered["AddOns"] > 0).sum())
        print(f"  positions added to at least once: {k:,} of {len(entered):,} "
              f"({k/len(entered)*100:.1f}%); avg lots per position "
              f"{entered['Lots'].mean():.2f}, max {int(entered['Lots'].max())}")


def summarise(sel: pd.DataFrame, top_n: int, target_label: str = "T1",
              adj=None) -> None:
    """Print headline results for one target strategy and portfolio size."""
    print("\n" + "=" * 72)
    print(f"RESULTS — top {top_n} stocks per session, dedicated {target_label} target strategy")
    print("=" * 72)

    print(f"\nTrades: {len(sel):,}  "
          f"over {sel['SignalDate'].dt.date.nunique():,} sessions  "
          f"({sel['Symbol'].nunique():,} distinct symbols)")
    print(f"Window: {sel['SignalDate'].min().date()} .. {sel['SignalDate'].max().date()}")

    print("\nBy outcome (one row per position or rejected entry attempt):")
    for k, v in sel["Outcome"].value_counts().items():
        print(f"  {k:<24} {v:>7,}  ({v/len(sel)*100:5.1f}%)")

    # still_open and no_entry_bar have no realised return; no_target_above_entry
    # was never entered at all. Only resolved trades belong in a P&L number.
    res = add_position_sizing(
        sel[sel["Outcome"].isin(RESOLVED_OUTCOMES)]
    )
    if res.empty:
        print("\nNo resolved trades in range.")
        return

    print(f"\nResolved trades: {len(res):,}")
    hit = res["Outcome"] == OUT_TARGET_HIT
    stopped = res["Outcome"] == OUT_TSL_HIT
    expired = res["Outcome"] == OUT_HORIZON
    print(f"  target reached      : {hit.sum():,} ({hit.mean()*100:.1f}%)")
    print(f"  trailing stop hit   : {stopped.sum():,} ({stopped.mean()*100:.1f}%)  "
          f"[{TRAILING_STOP_PCT*100:.1f}% below highest high]")
    print(f"  horizon expired     : {expired.sum():,} ({expired.mean()*100:.1f}%)")
    if stopped.any():
        print(f"  avg return | stopped: {res.loc[stopped, 'Return'].mean()*100:+.3f}% "
              f"({res.loc[stopped, 'AbsPnL'].mean():+,.2f}); "
              f"{(res.loc[stopped, 'Return'] > 0).mean()*100:.1f}% of stops locked in a gain")

    print(f"\nReturn per trade (fixed capital per entry = {CAPITAL_PER_ENTRY:,}, "
          f"gross of costs):")
    r = res["Return"]
    pnl = res["AbsPnL"]
    print(f"  mean   : {r.mean()*100:+.3f}%  ({pnl.mean():+,.2f})")
    print(f"  median : {r.median()*100:+.3f}%  ({pnl.median():+,.2f})")
    print(f"  std    : {r.std()*100:.3f}%")
    print(f"  p5/p95 : {r.quantile(.05)*100:+.2f}% / {r.quantile(.95)*100:+.2f}%")
    print(f"  worst  : {r.min()*100:+.2f}%   best: {r.max()*100:+.2f}%")
    print(f"  avg capital deployed per trade: {res['InvestedAmount'].mean():,.2f}")
    print(f"  total absolute PnL: {pnl.sum():+,.2f} "
          f"over {len(res):,} trades "
          f"({res['InvestedAmount'].sum():,.2f} total capital deployed)")

    # This summary is already for one dedicated target strategy, so all
    # resolved trades in this section target the same rung.

    print("\nMean overnight gap (entry open vs firing close):")
    print(f"  {res['GapPct'].mean()*100:+.3f}%  "
          f"(median {res['GapPct'].median()*100:+.3f}%)")
    print("  This is paid on every trade before the target is measured.")

    print(f"\nSessions held: mean {res['SessionsHeld'].mean():.1f}, "
          f"median {int(res['SessionsHeld'].median())}, max {int(res['SessionsHeld'].max())}")

    print_add_on_summary(sel, adj)

    if res["SegmentCrossed"].astype(bool).any():
        n_seg = int(res["SegmentCrossed"].astype(bool).sum())
        print(f"\n[NOTE] {n_seg:,} resolved trade(s) span a Segment boundary (a long "
              f"trading gap).\n       signal_scanner.py nulls forward windows across "
              f"these; filter on SegmentCrossed to match.")

    print("\nBy year:")
    for y, g in res.groupby(res["SignalDate"].dt.year):
        h = (g["Outcome"] == OUT_TARGET_HIT).mean()
        print(f"  {y}: {len(g):>7,} trades | reached {h*100:5.1f}% | "
              f"stopped {(g['Outcome'] == OUT_TSL_HIT).mean()*100:5.1f}% | "
              f"mean {g['Return'].mean()*100:+.3f}% ({g['AbsPnL'].mean():+,.2f}) | "
              f"total PnL {g['AbsPnL'].sum():+,.2f}")



def export_trader_log(sel: pd.DataFrame, top_n: int, target_label: str = "T1",
                      lots=None) -> None:
    """Export a trader-friendly ENTRY/EXIT log to TXT and Excel.

    ENTRY rows are created for every purchase of a position (the first entry
    plus any add-ons, taken from `lots`). One EXIT row sells all of them.
    EXIT rows are created only for resolved trades. Unresolved trades
    are shown as OPEN entries without a synthetic exit.
    """
    os.makedirs("results", exist_ok=True)

    resolved_outcomes = RESOLVED_OUTCOMES
    entered_outcomes = resolved_outcomes + [OUT_STILL_OPEN]

    entered = add_position_sizing(sel[sel["Outcome"].isin(entered_outcomes)])
    events = []
    lots_by_trade = (
        {tid: g.to_dict("records") for tid, g in lots.groupby("TradeID")}
        if lots is not None and not lots.empty else {}
    )

    for _, trade in entered.iterrows():
        if pd.isna(trade.get("Entry")) or pd.isna(trade.get("EntryDate")):
            continue

        symbol = trade["Symbol"]
        signal_date = pd.to_datetime(trade["SignalDate"])
        entry_date = pd.to_datetime(trade["EntryDate"])
        outcome = trade["Outcome"]

        # Match the ENTRY and EXIT events for the same backtest trade.
        trade_id = f"{symbol}_{signal_date.strftime('%Y%m%d')}"

        shares = float(trade["Shares"])             # all lots together
        invested = float(trade["InvestedAmount"])   # what all lots cost

        # ENTRY EVENTS: one per purchase (first entry, then any add-ons).
        trade_lots = lots_by_trade.get(trade_id) or [{
            "EntryDate": entry_date,
            "EntryPrice": float(trade["Entry"]),
            "Invested": invested,
        }]
        for lot in trade_lots:
            events.append({
                "Date": pd.to_datetime(lot["EntryDate"]),
                "Symbol": symbol,
                "Trade Type": "ENTRY",
                "LTP/CMP": float(lot["EntryPrice"]),
                "Money Invested/Exited": float(lot["Invested"]),
                "Gain/Loss %": np.nan,
                "Gain/Loss (Absolute)": np.nan,
                "Days Taken": np.nan,
                "Status": "OPEN" if outcome == OUT_STILL_OPEN else "CLOSED",
                "Trade ID": trade_id,
                "_invested": float(lot["Invested"]),
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
                "Money Invested/Exited": shares * float(exit_price),
                # Return is stored as a decimal: 0.05 means +5%.
                "Gain/Loss %": float(trade_return),
                "Gain/Loss (Absolute)": float(trade["AbsPnL"]),
                # SessionsHeld counts trading sessions in this backtest.
                "Days Taken": int(sessions_held),
                "Status": outcome,
                "Trade ID": trade_id,
                # Cost of ALL lots of this trade, released from exposure on exit.
                "_invested": invested,
            })

    columns = [
        "Date",
        "Symbol",
        "Trade Type",
        "LTP/CMP",
        "Money Invested/Exited",
        "Gain/Loss %",
        "Gain/Loss (Absolute)",
        "Total Money Invested",
        "Total Profit/Loss",
        "Days Taken",
        "Status",
        "Trade ID",
    ]
    log = pd.DataFrame(events, columns=columns + ["_invested"])

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

        # Running totals are computed AFTER the chronological sort.
        # Exposure: +invested on ENTRY, -that trade's initial invested on EXIT.
        is_entry = log["Trade Type"] == "ENTRY"
        exposure_delta = np.where(is_entry, log["_invested"], -log["_invested"])
        log["Total Money Invested"] = np.round(np.cumsum(exposure_delta), 2)
        # Realised P&L: only EXIT rows contribute; ENTRY rows carry forward.
        log["Total Profit/Loss"] = np.round(
            log["Gain/Loss (Absolute)"].fillna(0.0).cumsum(), 2
        )
    log = log.drop(columns="_invested")[columns]

    # OUTPUT 1: TEXT FILE
    suffix = "" if target_label == "T1" else f"_{target_label.lower()}"
    txt_path = f"results/trader_trades{suffix}_top{top_n}.txt"
    text_log = log.copy()
    if not text_log.empty:
        text_log["Date"] = text_log["Date"].dt.strftime("%Y-%m-%d")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"TRADER TRADE LOG - {target_label} STRATEGY - TOP {top_n} STOCKS PER SESSION\n")
        f.write("=" * 105 + "\n\n")
        f.write(f"Sizing: fixed capital per entry = {CAPITAL_PER_ENTRY:,}; "
                f"shares = nearest whole number (1 share if price > capital).\n")
        f.write("Money Invested/Exited: Shares*EntryPrice on ENTRY, Shares*ExitPrice on EXIT.\n")
        f.write("Total Money Invested: running active exposure (entries add, exits release cost).\n")
        f.write("Total Profit/Loss: running realised P&L; updates on EXIT rows only.\n")
        f.write("Gain/Loss % and Gain/Loss (Absolute) are populated only on EXIT rows.\n")
        f.write("Days Taken counts trading sessions, not calendar days.\n")
        f.write("OPEN means the trade has no resolved exit yet.\n")
        f.write(f"A trade with several ENTRY rows (same Trade ID) was added to by later, "
                f"higher-scoring signals (~{CAPITAL_PER_ENTRY:,} each); its single EXIT sells all of it.\n")
        f.write("Gain/Loss % on an EXIT row is over the total invested in that trade; "
                "Days Taken counts from its first entry.\n")
        f.write(f"Status trailing_stop_hit = exited by the {TRAILING_STOP_PCT*100:.1f}% trailing stop.\n\n")

        if text_log.empty:
            f.write("No entered trades found.\n")
        else:
            f.write(
                text_log.to_string(
                    index=False,
                    formatters={
                        "LTP/CMP": lambda x: "" if pd.isna(x) else f"{x:,.2f}",
                        "Money Invested/Exited": lambda x: (
                            "" if pd.isna(x) else f"{x:,.2f}"
                        ),
                        "Gain/Loss (Absolute)": lambda x: (
                            "" if pd.isna(x) else f"{x:+,.2f}"
                        ),
                        "Total Money Invested": lambda x: (
                            "" if pd.isna(x) else f"{x:,.2f}"
                        ),
                        "Total Profit/Loss": lambda x: (
                            "" if pd.isna(x) else f"{x:+,.2f}"
                        ),
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
            "Date": 14,
            "Symbol": 18,
            "Trade Type": 14,
            "LTP/CMP": 16,
            "Money Invested/Exited": 24,
            "Gain/Loss %": 16,
            "Gain/Loss (Absolute)": 22,
            "Total Money Invested": 22,
            "Total Profit/Loss": 20,
            "Days Taken": 14,
            "Status": 20,
            "Trade ID": 28,
        }
        formats = {
            "Date": "yyyy-mm-dd",
            "LTP/CMP": "#,##0.00",
            "Money Invested/Exited": "#,##0.00",
            "Gain/Loss %": "+0.00%;-0.00%;0.00%",
            "Gain/Loss (Absolute)": "+#,##0.00;-#,##0.00;0.00",
            "Total Money Invested": "#,##0.00",
            "Total Profit/Loss": "+#,##0.00;-#,##0.00;0.00",
            "Days Taken": "0",
        }
        for col_idx, name in enumerate(log.columns, start=1):
            letter = ws.cell(row=1, column=col_idx).column_letter
            ws.column_dimensions[letter].width = widths.get(name, 14)
            if name in formats:
                for row in range(2, ws.max_row + 1):
                    ws.cell(row=row, column=col_idx).number_format = formats[name]

    print(f"  Trader log -> {txt_path}")
    print(f"  Trader log -> {xlsx_path}")


def run_simulations(cand: pd.DataFrame, top_ns: list, n_workers: int) -> dict:
    """Pass 2: simulate every symbol for all strategies and portfolio sizes.

    Returns {(target_num, top_n): (trades_df, adjustments_df, lots_df)}.
    """
    jobs = []
    for symbol, g in cand.groupby("Symbol", sort=True):
        tech_path = os.path.join(ss.INPUT_FOLDER, f"{symbol}.csv")
        # print (tech_path)
        if not os.path.exists(tech_path):
            continue
        jobs.append((tech_path, g.sort_values("SignalDate").to_dict("records"), top_ns))
    if not jobs:
        sys.exit("[ERROR] No price files found for the candidate symbols.")

    print(f"\nSimulating positions for {len(jobs):,} symbols "
          f"x 3 strategies x {len(top_ns)} portfolio size(s)...")
    keys = [(t, n) for t in (1, 2, 3) for n in top_ns]
    trades = {k: [] for k in keys}
    adjs = {k: [] for k in keys}
    lots = {k: [] for k in keys}
    chunksize = max(1, min(16, len(jobs) // (n_workers * 8)))
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for i, out in enumerate(ex.map(simulate_symbol, jobs, chunksize=chunksize), 1):
            for key, (tr, ad, lt) in out.items():
                trades[key].extend(tr)
                adjs[key].extend(ad)
                lots[key].extend(lt)
            if i % 300 == 0:
                print(f"  [{i}/{len(jobs)}] symbols simulated")

    def frame(rows, date_cols, sort_cols):
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        for c in date_cols:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c])
        return df.sort_values(sort_cols, kind="stable").reset_index(drop=True)

    return {
        k: (
            frame(trades[k], ["SignalDate", "EntryDate", "ExitDate", "GoverningSignalDate"],
                  ["SignalDate", "Symbol"]),
            frame(adjs[k], ["SignalDate", "PreviousHorizonEnd", "NewHorizonEnd", "AddOnDate"],
                  ["SignalDate", "Symbol"]),
            frame(lots[k], ["TriggerSignalDate", "EntryDate"], ["EntryDate", "Symbol"]),
        )
        for k in keys
    }


def main():
    ap = argparse.ArgumentParser(
        description="Backtest fresh bullish signals with independent T1, T2, and T3 targets.")
    ap.add_argument("--start", default="2024-01-01", help="First tradeable signal date.")
    ap.add_argument("--top-n", type=int, nargs="+", default=None, metavar="N",
                    help=f"Stocks per session, ranked by CompositeScore. Limits NEW "
                         f"positions only; stocks already held can still be added to. "
                         f"Accepts several values to sweep, e.g. --top-n 5 10 20. "
                         f"Defaults to TOP_N_VALUES at the top of this file "
                         f"(currently {TOP_N_VALUES}).")
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
                    help=f"Skip the signal scan and re-select from {CANDIDATES_FILE}. The "
                         f"scan is the expensive part; the position simulation that follows "
                         f"re-reads each symbol's price file but is cheap.")
    args = ap.parse_args()

    folder = args.signals_folder or ss.OUTPUT_FOLDER
    n_workers = max(1, args.workers or os.cpu_count() or 1)

    if args.reuse_candidates:
        if not os.path.exists(CANDIDATES_FILE):
            sys.exit(f"[ERROR] {CANDIDATES_FILE} not found -- run once without "
                     f"--reuse-candidates first.")
        cand = pd.read_parquet(CANDIDATES_FILE)
        print(f"Reusing {len(cand):,} candidates from {CANDIDATES_FILE}")
        required_cache_columns = {
            "Direction", "SignalName", "SignalDate", "BestHorizon",
            "CompositeScore", "Count", "Target1Pct", "Target2Pct", "Target3Pct",
        }
        missing_cache_columns = required_cache_columns.difference(cand.columns)
        if missing_cache_columns:
            sys.exit(
                f"[ERROR] {CANDIDATES_FILE} was created by an older script and "
                f"is missing {sorted(missing_cache_columns)}. Run once without "
                "--reuse-candidates to rebuild it."
            )
        cand["SignalDate"] = pd.to_datetime(cand["SignalDate"])
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
            tech_path = os.path.join(ss.INPUT_FOLDER, f"{symbol}_data.csv")
            # print (tech_path)
            if os.path.exists(tech_path):
                jobs.append((tech_path, sig_path, args.start, filters))
        if not jobs:
            sys.exit(f"[ERROR] No scored signals found in {folder}")

        print(f"Scanning {len(jobs):,} symbols from {folder} "
              f"across {n_workers} worker(s), signals from {args.start}...")

        rows = []
        chunksize = max(1, min(16, len(jobs) // (n_workers * 8)))
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            for i, out in enumerate(ex.map(scan_symbol_episodes, jobs, chunksize=chunksize), 1):
                rows.extend(out)
                if i % 300 == 0:
                    print(f"  [{i}/{len(jobs)}] {len(rows):,} candidate signals")

        if not rows:
            sys.exit("[ERROR] No episodes found in range.")
        cand = pd.DataFrame(rows)
        cand["SignalDate"] = pd.to_datetime(cand["SignalDate"])
        os.makedirs("results", exist_ok=True)
        cand.to_parquet(CANDIDATES_FILE, index=False)
        print(f"\n{len(cand):,} candidate signals -> {CANDIDATES_FILE}")

    # Ranking is shared by every strategy and limits NEW positions only: a
    # session's signals are ranked by CompositeScore across all stocks, held or
    # not, exactly as the scanner list would show them.
    cand = cand[
        cand["Direction"].astype(str).str.strip().str.casefold() == "bullish"
    ].copy()
    if cand.empty:
        sys.exit("[ERROR] No bullish candidate signals found in range.")

    cand = cand.sort_values(["SignalDate", "CompositeScore"], ascending=[True, False])
    cand["RankInSession"] = cand.groupby("SignalDate").cumcount() + 1

    top_ns = sorted(set(args.top_n or TOP_N_VALUES))
    if any(n < 1 for n in top_ns):
        sys.exit(f"[ERROR] --top-n values must be >= 1, got {top_ns}")

    sim = run_simulations(cand, top_ns, n_workers)
    os.makedirs("results", exist_ok=True)

    # One comparison table per dedicated target strategy.
    for target_num in (1, 2, 3):
        target_label = f"T{target_num}"
        rows = []
        detail = []

        for n in top_ns:
            sel, adj, lots = sim[(target_num, n)]
            if sel.empty:
                print(f"  {target_label} top {n:>3}/session: no trades")
                rows.append((n, {"trades": 0, "resolved": 0}))
                continue

            # Keep the original filenames for T1. T2 and T3 get separate files.
            if target_num == 1:
                csv_path = TRADES_FILE_TMPL.format(n=n)
            else:
                csv_path = f"results/backtest_t{target_num}_trades_top{n}.csv"
            adj_path = f"results/target_adjustments_t{target_num}_top{n}.csv"

            sel.to_csv(csv_path, index=False)
            adj.to_csv(adj_path, index=False)
            print(
                f"  {target_label} top {n:>3}/session: {len(sel):>7,} positions -> {csv_path}; "
                f"{len(adj):>7,} signals while holding -> {adj_path}"
            )

            export_trader_log(sel, n, target_label=target_label, lots=lots)
            rows.append((n, headline_stats(sel)))
            detail.append((n, sel, adj))

        comparison_table(rows, target_label=target_label)

        if args.detail or len(top_ns) == 1:
            for n, sel, adj in detail:
                summarise(sel, n, target_label=target_label, adj=adj)

    print(f"\nRe-run with --reuse-candidates to skip the signal scan "
          f"({len(cand):,} candidate signals cached).")


if __name__ == "__main__":
    main()