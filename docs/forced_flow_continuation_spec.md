# Forced-Flow Continuation Strategy — Frozen Before Outcomes

**Rules:** `goldset/forced_flow/continuation_strategy_rules.json`
(`forced-flow-continuation-v1`). Frozen **before** any return is computed.

This is the **last credible forced-flow hypothesis**: the announcement→effective
"index effect." It is *independent* of the two settled results — the reversal is
rejected and the effective-day study is null. The ≈0 effective-day residual is
**consistent with** an announcement-time move, but is **not evidence** of one; it
is equally consistent with decay, anticipation, or no exploitable effect. Track B
answers this cleanly, not by assuming it works.

## The strategy (exact, frozen)

- **Signal:** a verified S&P 500 addition announcement (Tier 1/2/3, exact time).
- **Entry:** the next regular-session **open after `announcement_public_time`** —
  even for intraday announcements. Conservative and executable with daily bars.
- **Primary exit:** effective-date **close**.
- **Secondary exits (reported separately):** D−1 close, effective-date open.
- **No entry** when the announcement time is date-only or the batch is unresolved.
- **Hedge:** long the added name, short `beta × SPY` using the frozen pre-event
  beta (same 60-day pre-event daily-return beta as Track A, clipped [0,3]).
- **Costs:** paid on **both** legs at **entry and exit** (15 bps/side baseline).

## Inference (frozen)

- Cluster by **announcement batch**.
- Report **quarterly** and **ad-hoc** separately.
- Stress: **delayed entry** (next+1 open) and **doubled cost** (30 bps/side).
- **Matched controls** (momentum + dollar-ADV, uncontaminated) and **placebo**
  announcements.
- Raw event-study residuals reported **separately** from after-cost portfolio.

## Predeclared sample classification

| verified, price-covered batches | claim |
|---|---|
| ≥ 75 | full test |
| 50–74 | underpowered candidate test |
| < 50 | descriptive pilot only |

Falling short downgrades the **claim**, never relaxes the kill gates or the
Tier/timestamp rules.

## Selection-bias guard (before any outcome)

Compare **resolved vs unresolved** batches by year, event type (quarterly/ad-hoc)
and number of additions, so source availability does not silently create a
selected sample. This comparison is reported before the continuation backtest.

## Sourcing discipline (return-blind)

No returns, price charts, or rule changes based on coverage until the fixed
denominator (all 113 batches) is resolved-or-unresolved and the announcement
manifest is frozen. Batches are resolved in deterministic (chronological) order,
never by apparent importance; unresolved records are preserved.
