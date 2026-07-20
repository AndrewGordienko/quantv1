# Forced-Flow Announcement→Effective Continuation — PREREGISTERED Outcome Test

**Status: PREREGISTERED, RETURN-BLIND.** Written and committed *before* any return
or price outcome is joined to the announcement corpus. Run **exactly once**
against the frozen sample:

- Frozen corpus: `goldset/forced_flow/census_freeze_v1.json`
  (`forced-flow-announcement-corpus-frozen-v1`).
- Manifest: `goldset/forced_flow/announcement_manifest_v1.jsonl`,
  **sha256 `152622d88239f213…`**, **75 verified genuine-addition batches** (20
  quarterly-rebalance, 55 ad-hoc), 2019-01→2025-11.
- Rename/merger exclusions: `announcement_renames_excluded_v1.jsonl` (33 excluded;
  the true addition denominator, not 113).

## Hypothesis (falsifiable)

**Mechanism:** a verified S&P 500 addition forces passive funds to buy the added
name between announcement and effective date; if that demand is not fully
absorbed at announcement, the name drifts up over the announcement→effective
window. **Prediction:** the hedged announcement→effective residual return is
**> 0 after realistic costs**, and survives delayed entry and doubled costs.
**Null (expected):** post-2010 the index-inclusion effect has decayed to at/below
detectable size; at n≈75 the minimum detectable effect is ~1.2–1.5%, so a null is
the prior.

## Unit of analysis

The **announcement batch** (one `event_batch_id`) is the independent cluster.
Multi-name batches (e.g. CARR+OTIS, CRH+CVNA+FIX) are one cluster: equal-weight the
added names within the batch, then treat the batch as the observation. Standard
errors cluster on batch. **Never** treat individual tickers as independent.

## Frozen trade rules (locked)

- **Signal:** the batch's verified `announcement_public_time` (Tier 1/2, exact
  minute) for its covered added tickers.
- **Entry:** the **next regular-session open after `announcement_public_time`**
  (executable with daily bars; intraday/after-hours both → next open).
- **Primary exit:** the batch **effective-date close** (census `effective_date`).
- **Secondary exits (reported separately, never as the headline):** D−1 close;
  effective-date open.
- **Hedge:** long the added name, short `beta × SPY`, `beta` = the frozen 60
  trading-day pre-announcement daily-return beta, clipped to [0, 3].
- **Costs:** 15 bps/side on **both legs** at **entry and exit** (baseline).
- **Weighting:** equal-weight added names within a batch; batches equal-weighted.

## Stresses (all preregistered, all reported)

1. **Delayed entry:** next+1 session open (must stay positive).
2. **Doubled cost:** 30 bps/side both legs (must stay positive).
3. **Stratification:** report **ad-hoc (55)** and **quarterly-rebalance (20)**
   **separately** — never pool into one number. (Quarterly windows are ~2–3 weeks;
   ad-hoc ~5–8 days. Different mechanisms, reported apart.)
4. **Controls:** matched control (nearest momentum + dollar-ADV, uncontaminated by
   an index event) and **placebo** (shuffled announcement dates on the same names).
5. **Concentration:** verdict must not depend on any single ticker, batch, or year
   (leave-one-batch-out; report worst-case drop).

## Inference (locked)

- Primary estimand: batch-clustered **mean net hedged residual** (announcement-open
  → effective-close), after 15 bps/side both legs.
- Confidence intervals: **batch block-bootstrap** (resample batches with
  replacement) + report the HAC/clustered lower bound.
- Report gross event-study residual **separately** from the after-cost portfolio.
- Data: the daily `prices` panel. **Caveat:** it is survivorship-biased by
  construction (`docs/point_in_time_panel_spec.md`), but S&P 500 *additions* are
  ~89% price-covered (they are current/surviving names), so the addition-side test
  is far less exposed than a full-universe study. Names with missing price windows
  are reported as coverage gaps, never silently dropped.

## Decision rule (no rescue filters)

- **Promotion is impossible at n=75.** The scoreboard requires **≥100 independent
  executable events**; this is a **full CANDIDATE test — reject-only**.
- **REJECT the announcement-continuation leg** if the primary net residual is not
  significantly > 0, OR fails delayed entry, OR fails doubled cost, OR the
  bootstrap lower bound ≤ 0, OR the effect is concentration-driven. On rejection,
  **close the leg** — no new feature slices, no threshold tuning, no rescue.
- **ADVANCE to a ≥100-event validation** only if the primary is significantly > 0
  net of costs AND survives delayed entry AND doubled cost AND is not
  concentration-driven — and even then it is a *candidate*, not promoted.

## Run protocol

Implement once as `scripts/forced_flow_continuation_test.py` →
`data/forced_flow_continuation_test.json`, keyed to `manifest_sha256`
`152622d88239f213…`. If the manifest hash changes, the test is invalid and must
not be re-run to chase a different result. Write the outcome up **power-honestly**
(state the MDE and that a null is uninformative about tiny effects, not proof of
zero). Register the single run in `data/experiment_registry.jsonl`.
