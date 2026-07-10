# V2_PLAN.md — Clean research engine after the methodology review

A detailed methodology review (2026-07) found leakage and portfolio-design
problems in v1 and argued the tactical stop layer got ahead of the evidence.
The review is substantially correct. This document is the go-forward plan and
records what we've already confirmed.

## Reframed target (v1 was chasing a mirage)

The "politicians making 35–50%" figure is **ex-post winner selection** — best
outcomes chosen after the year ended, often one concentrated name, leveraged
ETFs, or options. Only ~32% of Congress reportedly beat SPY in 2025. The honest
investable benchmark is NANC: ~23.6% annualized since inception vs ~21.1% for
S&P 500 TR through 2026-06-30 — a **~2.5pt/yr edge**. That is the realistic
stretch target. Regular 40%+ requires concentration/leverage, not a signal.

## What we've already CONFIRMED this session

1. **Politician skill does not persist.** `research/skill_persistence.py`
   (next-open entry, year t → t+1): pooled **Spearman = +0.015, p=0.83, n=206**.
   Top-quartile members average +0.47% next year vs −0.65% bottom — swamped by
   year-to-year instability (individual pairs swing −0.39 … +0.58).
   **⇒ "top politician" / `member_skill` is descriptive, not tradeable.** Rank
   on trade-level features instead of who the trader is.

2. **The proposed trend gate HURTS.** `scripts/tactical_sweep.py`: removing the
   50d-MA gate (E4) lifted the tactical strategy from 7.1% → 11.0% CAGR
   (Sharpe 0.45 → 0.66). Confirms the insignificant timing study — no evidence
   for an MA gate. Every stop configuration still trailed SPY (16.5%).

3. **Confirmed code defects** (v1):
   - Position cap: clip-then-renormalize re-breached the cap
     → **FIXED** in `construct.py::_cap_weights` (water-fill + allows cash).
   - Entry price = filing-day close (`returns.py`) → must be next-session open.
   - Leaked AUC: `train.py` walk-forward loads full-history `skill_scores`
     via `features.build` → the reported **AUC 0.558 is contaminated**.
     (Note: `tactical.py`/`backtest.py` recompute point-in-time skill per fold,
     so those paths are clean — the leak is specifically the headline AUC.)
   - Future metadata: `market_cap_log` uses *today's* Yahoo market cap for all
     history (a real leak); sector/committee are current-only too.
   - Survivorship: no-price tickers dropped and weights renormalized.

## Known leaks / bad definitions still to fix

| Problem | v1 behavior | Fix in v2 |
|---|---|---|
| Impossible entry | enter at filing-day close | enter next-session **open** after disclosure_date; next-close as conservative sensitivity |
| `first_seen_at` | not stored | store it going forward; **cannot** reconstruct for history → use disclosure_date+next-open and FLAG the availability lag as an optimistic assumption |
| Overlapping labels | 63d labels leak across CV folds | **purge + embargo** 63 trading days between train/validation |
| Leaked skill | full-history leaderboard as a feature | recompute skill **inside each fold**; time-varying Bayesian estimate |
| Future metadata | current mktcap/sector/committee on old trades | drop until point-in-time versions exist; historical committee membership |
| Survivorship | drop delisted, renormalize | keep at **actual weight**; delisting = realized return (last price / −100%) |
| Forced full investment | weights renormalized to 100% | allow **cash**; only hold names clearing a hurdle |
| Wrong alpha | stock − SPY, beta=1 | sector-, QQQ- and factor-adjusted residual return |
| Weak stats | normal t-stat, trades assumed independent | **block bootstrap** clustered by member, ticker, filing report |
| Wrong target | binary "beats SPY @63d" | regress **net factor-adjusted forward return** (magnitude, after costs) |

## Two-model split

**A. Politician-skill model** — did a member make a good decision from the
*transaction* date? Not directly tradeable (we already showed it doesn't
persist), useful only if a time-varying Bayesian version shows any signal.

**B. Follower-alpha model** — expected residual return **from the moment we can
transact** (next open after disclosure). Multi-horizon labels: 5d disclosure
reaction, 21d remaining drift, 63d slower drift. This yields an **alpha-decay**
estimate so a signal doesn't stay "buy" for 90 days — the principled fix for the
MRVL problem (remaining expected alpha goes negative, no arbitrary stop needed).

Ranking hurdle: `score = (E[remaining excess return] − costs) / forecast_vol`.
Buy only while positive; MRVL drops out when remaining alpha turns negative.

## Feature hypotheses worth testing (trade-level, not member-level)

New position vs add-on · trade value relative to the member's *estimated
portfolio* (not raw $ bucket) · owner (self/spouse/joint/adviser) · filing speed
· repeat purchases same member+ticker · purchase→purchase vs rapid round-trips ·
large + repeat conviction · historical committee-company relevance · lobbying /
gov-contract / campaign-finance links · post-disclosure gap, volume reaction,
residual momentum · earnings/events between txn and disclosure · **properly
parsed options** (strike, expiry, call/put, approx delta).

## Model ladder (simple first)

1. Transparent rules: large, fast-filed, repeat purchase.
2. Elastic-net on forward factor-adjusted return.
3. Hierarchical Bayesian politician-skill (time-varying).
4. CatBoost/LightGBM for nonlinear interactions.
5. Daily "remaining alpha" model for held positions.

## Event-driven portfolio

One disclosure → one signal with a lifecycle:
`new filing → score → enter once or reject → update remaining alpha daily → exit`.
Never let an old filing become a new buy each rebalance. Start: 5–10 positions,
5% max/equity (less for options/leveraged ETFs), sector & factor limits, vol
targeting, cash allowed. Deploy as **70–80% SPY core + 20–30% congressional
sleeve**. Benchmark vs SPY, QQQ, sector-matched, naive copy, and NANC/GOP from
actual inception.

## Immediate build order

1. `research/skill_persistence.py` — **DONE** (result: no persistence).
2. `construct.py` cap fix — **DONE**.
3. `portfolio/backtest_v2.py` — next-open execution, purged+embargoed
   walk-forward, per-fold skill, cash, delisting handling, fixed caps.
4. `research/event_study_v2.py` — factor/sector-adjusted returns + clustered
   block-bootstrap CIs. Re-test the $250k–1M slice (n≈90) under clustering.
5. Four **locked** experiments: (a) large trades, (b) fast filings, (c) same-
   member repeat purchases, (d) large + repeat + post-disclosure momentum.
6. Lock **2024–2026 as an untouched holdout**; no tuning against it.
7. Only if a strategy beats the simple large-trade rule AND risk-matched
   benchmarks → broker integration (paper), then ≥3 months paper before capital.

## Shelved (pending evidence)

Tactical stop machinery (`tactical.py` X1–X6, E4 trend gate). The kill switch,
trailing/ATR stops, and MA gate did not beat SPY and the MA gate actively hurt.
Keep only the *idea* that staleness matters — implemented properly as the
remaining-alpha decay model in Follower-alpha, not as hard stops.

## Legal note

If ever commercialized: NANC's prospectus discloses uncertainty around the
statutory restriction on using PTRs for commercial purposes. Personal research
≠ a commercial product — get legal advice before selling anything.
