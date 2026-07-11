# Strategy Scoreboard

The single table that matters. A candidate becomes *the* strategy only by
filling this row with real, post-cost, walk-forward numbers and clearing the
kill gates. Everything else is lab-building. Updated as verdicts land.

## The table

| Strategy | Indep. validation trades | Net return | Net Sharpe | Delayed-entry | 2× costs | Decision |
|---|---|---|---|---|---|---|
| **Forced flow** (index effect) | — | — | — | — | — | **BUILD NEXT (flagship)** — data-viable |
| **MGRM** (guidance underreaction) | — | — | — | — | — | **TEST ONCE** — blocked on extraction cert |
| **Actor B3** (Fed speaker deviation) | — | — | — | — | — | **SIDE EXPERIMENT (≤10%)** — census v1 frozen |

## Kill gates (a candidate advances only if ALL hold)

- ≥ 100 genuinely independent, executable events
- Positive net return after realistic costs (spread, impact, borrow, fees)
- Validation Sharpe > 1
- Positive delayed-entry result and positive doubled-cost result
- Bootstrap and HAC lower bounds above zero
- No dominant single ticker, event, or actor
- Positive paper-forward performance

If a candidate fails, **close the hypothesis** — no rescue via new feature slices.

## Effort allocation (this stage)

- **50%** forced-flow strategy + its historical data
- **30%** finish the one decisive MGRM experiment
- **10%** historical replay + eventual paper execution
- **10%** bounded Fed actor experiment (B0–B4), no graph expansion

## Grounded status (2026-07-11)

### Forced flow — flagship, data-viable
- **Mechanism:** index add/delete/reweight forces passive funds to trade
  irrespective of price; we trade the *unabsorbed* portion.
- **Data reality (checked):** daily `prices` = **2,547 tickers, 2012→2026**.
  Index additions hit mid-caps we DON'T have minute bars for, but DO have daily
  bars for — and the trade is a multi-day announce→effective move, so daily
  resolution is correct. This is the one hypothesis whose event universe our
  data can actually cover at the ≥100-event scale.
- **Gap:** `forced_flow_events` table exists but is empty; no ingest module. The
  events (index-change announcements w/ announcement_time + effective_date) must
  be sourced. Zero-vendor path exists (S&P change lists + official releases).
- **Four sub-tests to run separately:** pre-effective continuation, closing-
  auction pressure, post-effective reversal, related-company diffusion.

### MGRM — test once, then keep or kill
- **Blocker (checked):** `mgrm_extractor_certification.json` = `GOLDSET_TOO_SMALL`,
  provider `none:none`. `mgrm_report.json` = `MGRM_BLOCKED_SAMPLE_POWER`
  (3 usable events; needs ≥200 train / ≥100 val; extraction 0% certified).
- **To reach a verdict:** configure an extractor model + reference labeler,
  label the corpus, certify, then run G0/G1/G2 **once**. Pass → lock + paper.
  Fail → retire. No further feature archaeology.

### Actor B3 — bounded side experiment (demoted from "strategy")
- Now a ≤10% side test answering ONE question: after controlling for what was
  said, does *who said it and how they deviated* add out-of-sample lift (B1 vs
  B2/B3)? If yes, actor features become *modifiers* (credibility, expected vol,
  follow-through, asset mapping) inside the event engine — never a primary
  trigger. If no, stop.
- **State:** outcome writer merged (PR #6); census v1 rules frozen, BOARD_ONLY,
  PR #7 open. **Stop expanding** beyond the frozen Fed pilot. No universal
  actor graph until incremental value is shown on this cohort.

## Standing discipline

- A buy/sell display is a UI demo until a frozen walk-forward model produces the
  numbers above. No real execution because the plumbing works.
- Trade frequency is not performance. Watching all day ≠ trading all day.
- Personality is never the primary trigger; it modifies an event signal or it is
  dropped.
