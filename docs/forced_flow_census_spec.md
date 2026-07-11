# Forced-Flow S&P 500 Census — Frozen Specification (task #5)

**Flagship hypothesis.** Index add/delete forces passive funds to trade
irrespective of price; we trade the *unabsorbed* portion. This is the
best-supported hypothesis in the repo by both sample availability and economic
mechanism. The milestone here is a **frozen, auditable event corpus** — NOT a
claim that the edge exists.

## Provenance (discovery source, not point-in-time evidence)

- Source: `fja05680/sp500`, file `sp500_changes_since_2019.csv`
- Pinned commit: `c403a121c2e766840f34837738cdd4725eeda818`
- Raw sha256: `7766d1603ceacea9715feed55e0383b95f44903d2452e33bf73cca4d4381c036`
- Frozen copy: `goldset/forced_flow/sp500_changes_since_2019.csv`
- The ingester re-verifies the sha256 every run and raises `CensusIntegrityError`
  on drift. The file is a **discovery source**; it is not authoritative
  point-in-time evidence, and its date column is the **effective date**, not an
  announcement.

## Frozen corpus (census v1)

- Census version: `forced-flow-sp500-census-v1`
- Census sha256: `927685472fb0f554dbaf46cb114716b62214bcc277ee9ab70f95129f2795fd9f`
- Window: 2019-01-18 → 2026-06-02

| | additions | deletions |
|---|---|---|
| legs | 161 | 163 |
| unique tickers | 161 | — |
| **effective-date batches (clustering unit)** | **113** | — |
| ad-hoc legs | 102 | — |
| quarterly-rebalance legs | 59 | — |
| coverage: COVERED | 142 | 72 |
| coverage: MARKET_DATA_UNAVAILABLE (kept) | 19 | 91 |

**Power is counted in effective-date batches, not legs.** 161 addition legs are
only 113 independent date-batches, and quarterly rebalances are systematic.
Inference clusters by effective date.

## Timestamp discipline (load-bearing)

- `effective_date` — market effective date (exact, from the source).
- `knowledge_time` — kept NULL. Never fabricated or estimated.
- `announcement_time` — kept NULL. Recent additions are announced before the
  effective date and can move immediately on the announcement, so announcement
  time is load-bearing, not optional. We do not invent it.
- `timestamp_status` = `EFFECTIVE_DATE_ONLY` for the whole corpus.

**Restriction:** without a verified public announcement time, pre-effective
returns are **not** a leak-free trading strategy. Label the horizons accurately
by the data that supports them:

| test | data needed | status |
|---|---|---|
| Effective-day return (open→close, close→close) | daily bars | **testable now** |
| D+1 → D+5 reversal | daily bars | **testable now** |
| Closing-auction / last-hour pressure | minute or auction/quote data | **blocked** — daily OHLC cannot see it |
| Announcement → effective continuation | verified announcement time | **blocked** until the announcement corpus is frozen |

Daily OHLC does **not** support a closing-auction claim; calling an
effective-day daily-bar result "closing-auction pressure" would overstate it.

## Schema (columns added to `forced_flow_events`)

`knowledge_time`, `event_batch_id`, `batch_size`, `change_type`
(`QUARTERLY_REBALANCE`|`AD_HOC`, inferred from the effective date — clearly
marked inferred), `change_reason` (`UNVERIFIED` until a reason source is
reconciled), `timestamp_status`, `coverage_status`
(`COVERED`|`MARKET_DATA_UNAVAILABLE`), `historical_ticker`, `source_commit`,
`source_sha256`.

## Inclusion / exclusion rules

- Include every add/delete leg in the frozen source over the window.
- Explode co-dated rows into per-ticker legs; assign `event_batch_id` per
  effective date.
- Keep uncovered legs in the census, marked `MARKET_DATA_UNAVAILABLE`. Never
  silently drop them (that would bias coverage upward).
- `change_type` is an inferred quarterly-vs-ad-hoc split (3rd-Friday-of-quarter
  proximity); `change_reason` (merger/spin-off/etc.) is `UNVERIFIED` pending a
  reason source (e.g. Wikipedia changes table) — a documented later enrichment.
- Permanent identifiers (CUSIP/PERMNO) are unavailable; `historical_ticker`
  preserves the as-recorded symbol.

## Test plan (after the freeze)

Each test: market/sector residuals, matched non-added controls (sector, size,
momentum, liquidity), placebo effective dates, **event-date-clustered** CIs,
delayed entry + doubled costs, quarterly-vs-ad-hoc split, concentration checks.

1. Announcement → effective continuation — **verified announcements only** (blocked
   until the announcement corpus is frozen; see
   `docs/forced_flow_announcement_spec.md`).
2. Effective-day return (open→close, close→close) — daily bars.
3. D+1 → D+5 reversal — daily bars.
4. Dose response: estimated index demand / ADV versus residual return.
5. Closing-auction / last-hour pressure — **blocked** unless intraday coverage
   exists for the added ticker.

Results are reported **separately per horizon**, never combined across
announcement / effective-day / multi-day windows.

A candidate advances only through the scoreboard kill gates
(`docs/strategy_scoreboard.md`).

## Known extensions (not done, documented)

- Pre-2019 additions via diffing the fja05680 1996→present membership file
  (extends to the 2012 start of our daily `prices`).
- Verified announcement timestamps from S&P press releases → unblocks test #1.
- Verified `change_reason` from a reconciled reason source.
