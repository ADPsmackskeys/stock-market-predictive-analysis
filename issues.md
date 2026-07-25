This is actually a fairly sophisticated research engine. Whoever wrote it clearly understands many of the pitfalls of naive technical analysis. They already fixed several common mistakes (episode counting, MFE, ATR normalization, class-specific horizons).

That said, I think there are still **quite a few statistical, logical, and implementation problems**. Some are minor. Some are severe enough that I'd hesitate to trust the leaderboard without further validation.

I'll go from the biggest problems to the smallest.

---

# 1. Massive data snooping / multiple testing (The biggest issue)

This is the elephant in the room.

The script tests

* ~80 atomic signals
* dozens of predefined strategies
* thousands of pairs
* thousands of triples

Even without triples:

```
80 singles
+
3160 pairs
=
3240 tests
```

With triples:

```
80 choose 3

≈82,000 combinations
```

Across hundreds of stocks.

Eventually **something will appear amazing purely by chance.**

This is classic **multiple hypothesis testing**.

---

## Why Fisher doesn't solve this

The code computes

```
p < 0.10
```

for every signal.

Suppose you test

```
10,000 hypotheses
```

Expected false positives:

```
10,000 × 0.10

=1000
```

Nothing in the code adjusts for this.

It needs something like

* Bonferroni
* Benjamini-Hochberg FDR
* White's Reality Check
* Hansen SPA test

Otherwise the leaderboard will inevitably contain lucky signals.

---

# 2. In-sample optimization

Everything is learned on

```
same data

↓

same ranking
```

No train/test split.

No validation.

No walk-forward.

Example:

```
2010-2025

↓

discover best signal

↓

declare winner
```

That's not prediction.

That's explanation.

The leaderboard may completely fail on 2026.

---

# 3. No out-of-sample evaluation

The script never asks

> "Does the discovered edge survive on unseen data?"

It should be

```
2010-2020

↓

discover

↓

2021-2023 validate

↓

2024-2025 test
```

Instead

```
2010-2025

↓

discover

↓

done
```

---

# 4. MFE creates optimistic bias

This one is subtle.

The script defines success as

```
Did price EVER move enough?
```

Imagine

```
Entry

100
```

Then

```
103

99

95

90
```

MFE says

```
success
```

Reality:

Unless you magically sold exactly at 103,

you lost money.

MFE measures

> opportunity

not

> realizable profit.

---

A real strategy has

* exit rules
* stop loss
* trailing stop
* execution latency

MFE ignores all of those.

---

# 5. Win criterion differs from return metric

Notice what happens.

Success:

```
MFE exceeded threshold
```

Profit:

```
close return
```

Example

```
Entry

100
```

Then

```
108

96
```

This counts as

```
WIN
```

because

```
MFE =8%
```

But

```
close return

=-4%
```

So

```
winner

negative return
```

This mixes two incompatible concepts.

---

# 6. Expected value becomes hard to interpret

EV is computed as

```
win_rate × avg_win

+

loss_rate × avg_loss
```

But

```
win

≠

positive return
```

because win uses

```
MFE
```

while return uses

```
close
```

That means

```
wins

can contain negative returns
```

and

```
losses

can contain positive returns
```

This makes EV mathematically awkward.

---

# 7. Profit Factor is distorted

Profit factor uses

```
wins

↓

positive close returns
```

Suppose

```
MFE

triggered

↓

close negative
```

That observation is classified as a win

yet contributes

```
0

gross profit
```

Very strange.

---

# 8. No transaction costs

Every occurrence assumes

```
free trading
```

Reality

```
spread

slippage

brokerage

STT

tax

impact
```

Some signals average

```
0.3%
```

edge.

Those disappear immediately after costs.

---

# 9. No liquidity filter

Suppose

```
tiny illiquid stock
```

Signal says

```
buy
```

Historical high

```
+9%
```

But volume

```
₹20,000/day
```

Impossible to execute.

---

# 10. Uses High and Low

MFE uses

```
future HIGH
future LOW
```

Those are unknowable intraday.

Suppose

```
High

110
```

for

2 seconds.

You never capture it.

The script assumes every high was potentially tradable.

That's optimistic.

---

# 11. Horizon selection is manually hardcoded

Example

```
Momentum

5

10

15
```

Why?

Why not

```
4

8

13
```

or

```
7

14

21
```

These were chosen by intuition.

That itself is another optimization.

---

# 12. Composite score has arbitrary scaling

```
EV

×

PF

×

Consistency

×

log(count)
```

Why multiply?

Why

```
log(count)
```

instead of

```
sqrt(count)
```

?

Why not

```
Sharpe
```

?

There is no theoretical justification.

It's handcrafted.

---

# 13. Consistency metric is weak

Consistency is

```
1

-

range(win_rates)
```

Suppose

```
0.6

0.7

0.8
```

Range

```
0.2
```

Consistency

```
0.8
```

Suppose

```
0.8

0.8

0.6
```

Same score.

But the second may be much better.

Range ignores

* variance
* trend
* ordering
* uncertainty

---

# 14. Ignores confidence intervals

Signal A

```
20 trades

70%
```

Signal B

```
2000 trades

68%
```

The second is much more reliable.

The code doesn't model uncertainty.

---

# 15. Episode collapsing isn't perfect

Episode start:

```
True

True

True
```

↓

```
only first
```

Good.

But suppose

```
True

False

True
```

That's

two episodes.

Yet

```
False
```

could simply be indicator noise.

Episodes may fragment.

---

# 16. Overlapping forward windows remain

Even after episode collapsing

```
Day1 signal

↓

10-day window

Day5 signal

↓

10-day window
```

Huge overlap.

Returns aren't independent.

Statistical assumptions weaken.

---

# 17. Fisher assumptions violated

Fisher assumes

independent observations.

Forward windows overlap heavily.

Market regimes cluster.

Observations are autocorrelated.

Therefore

```
p-values

too optimistic.
```

---

# 18. Bull/Bear asymmetry

Bear success

```
future LOW
```

Bull success

```
future HIGH
```

But equities have

```
upward drift
```

Bear signals naturally behave differently.

No adjustment exists.

---

# 19. Same thresholds across industries

ATR normalization helps,

but

```
bank

FMCG

pharma

IT
```

have different behaviors.

Sector effects aren't modeled.

---

# 20. Ignores market regime

Signal performance changes dramatically.

Example

RSI Oversold

works well

```
bull markets
```

fails badly

```
crashes
```

Script mixes

```
2008

2020

2021

2022

2024
```

into one statistic.

---

# 21. Doesn't condition on volatility regime

Likewise

```
ATR high

ATR low
```

change signal quality.

Everything is pooled.

---

# 22. Persistent signals dominate combinations

Example

```
EMA Bull

True

200 days
```

Combined with

```
MACD Cross
```

almost every MACD signal occurs inside EMA.

So combinations aren't independent discoveries.

---

# 23. Pair explosion

Many combinations are logically redundant.

Example

```
RSI >50

+

RSI Bull Momentum
```

The second almost implies the first.

Thousands of combinations differ only trivially.

---

# 24. Correlated signals inflate search

MACD

EMA

Golden Cross

Price > EMA200

are highly correlated.

Testing them separately exaggerates effective search size.

---

# 25. Equal weighting of symbols

Leaderboard averages

```
small illiquid stock

=

Reliance
```

Maybe undesirable.

---

# 26. Survivorship bias

Depends entirely on

```
data/technical
```

If only current NSE stocks exist,

dead companies disappeared.

Historical performance becomes biased upward.

---

# 27. Missing corporate action checks

If price series aren't perfectly adjusted

```
splits

bonuses

mergers
```

forward returns become garbage.

---

# 28. Missing lookahead validation

It assumes indicator columns were computed correctly.

If preprocessing accidentally leaked future data,

the scanner happily ranks leaked signals.

---

# 29. No position sizing

Every signal is

```
1 occurrence
```

No notion of

* volatility targeting
* Kelly
* risk parity

---

# 30. No portfolio interaction

Signals evaluated individually.

Reality

```
20 signals

↓

portfolio

↓

correlation matters.
```

---

# 31. Hard minimum occurrence

```
MIN_OCCURRENCES =10
```

Very arbitrary.

A rare but powerful signal

```
9 occurrences

↓

discarded
```

while

```
10 mediocre occurrences

↓

accepted
```

---

# 32. Direction labeling is fixed

Example

```
RSI Oversold

↓

always bullish
```

But in strong downtrends

oversold often continues lower.

Context isn't incorporated.

---

# 33. ProcessPool overhead

For small datasets,

creating processes may take longer than computation.

Parallelization is beneficial mainly when there are many symbols and substantial per-symbol work.

---

# 34. Memory duplication

Each worker loads

* pandas
* NumPy
* TA-Lib
* all arrays

With hundreds of workers,

RAM usage can explode.

---

# 35. Ranking by average across horizons

Suppose

```
5D

excellent

10D

bad

15D

bad
```

Average may dilute a genuinely useful short-term signal.

Conversely,

one excellent horizon can be hidden by mediocre longer ones. The choice to average is a design decision, not an objective truth.

# Overall assessment

From a quantitative research perspective, I'd rate the script roughly as follows:

| Category                 |     Rating | Comments                                                                                                                                         |
| ------------------------ | ---------: | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| Software engineering     |   **9/10** | Well structured, modular, good use of vectorization and parallelism.                                                                             |
| Indicator implementation |   **9/10** | Clean definitions and sensible signal taxonomy.                                                                                                  |
| Computational efficiency | **8.5/10** | Good optimizations, though exhaustive combinations can still become expensive.                                                                   |
| Statistical methodology  | **5.5/10** | Better than many retail backtests, but vulnerable to multiple testing, dependence, and optimistic performance estimates.                         |
| Predictive reliability   | **4.5/10** | Without rigorous out-of-sample validation and corrections for data snooping, leaderboard rankings are likely to overstate true predictive power. |

The two issues that concern me most are **data snooping** (testing thousands of hypotheses without correcting for multiple comparisons) and the **absence of any out-of-sample or walk-forward validation**. Those alone are sufficient for a signal that looks outstanding in the historical data to fail completely when applied prospectively. The MFE-based scoring, while useful for measuring *opportunity*, also introduces an optimistic lens that should not be interpreted as evidence of an executable trading edge unless paired with explicit entry, exit, and risk management rules.

