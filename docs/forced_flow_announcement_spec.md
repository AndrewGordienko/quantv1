# Forced-Flow Announcement Corpus — Frozen Sourcing Spec

**Rules artifact:** `goldset/forced_flow/announcement_rules.json`
(`forced-flow-announcement-rules-v1`)
**Parent census:** `forced-flow-sp500-census-v1` (sha256 `927685…`)

Purpose: source and freeze **verified public announcement timestamps** for the
S&P 500 addition census, to unblock the announcement→effective continuation
test. This is the load-bearing timestamp: recent additions are announced before
their effective date and can move immediately on the announcement.

## Blindness protocol (frozen before searching)

1. **Enumerate all 113 addition batches**, never a convenient subset. The
   worklist (`announcement_worklist_v1.jsonl`, all `UNRESOLVED`) fixes the
   denominator before any search.
2. **No returns while sourcing.** The module never imports prices or computes an
   outcome — enforced by a test that scans for price/return code tokens.
3. **Freeze before join.** The accepted manifest and rejection ledger are frozen
   before anything is joined to market outcomes.

## Source tiers (anything below Tier 3 is rejected)

- **Tier 1** — timestamped S&P Dow Jones Indices release.
- **Tier 2** — timestamped release via an official newswire (PR Newswire /
  Business Wire / GlobeNewswire) carrying the S&P DJI release.
- **Tier 3** — reliable contemporaneous reporting quoting the announcement with a
  defensible publication time.
- **Reject** — inferred/estimated times, article *update* times, and any date
  without a defensible publication time (`DATE_ONLY`).

## Record (per `announcement_rules.json`)

`event_batch_id, announcement_public_time, announcement_timezone, effective_date,
source_tier, source_url, source_sha256, original_or_correction, affected_tickers,
timestamp_precision, first_executable_time, verification_status`.

`timestamp_precision` must be `exact_minute`/`exact_hour`; coarser is `DATE_ONLY`
and rejected. `source_sha256` pins the fetched source for audit.

## Entry-timing rule

Tradable entry is the **first executable observation after publication**. Added
names are mid-caps with **daily bars only**, so entry is a daily **session
open**, never an intraday fill:

- Intraday announcement → next session open.
- After-hours announcement → next session open.
- Date-only announcement → **BLOCKED** from the continuation test (kept in the
  census, excluded from the tradable subset).

## Gate

Report how many of the 113 batches reach a `VERIFIED` Tier 1/2/3 timestamp. That
verified-**and**-covered subset — not 113 — is the tradable continuation sample.
If it is small, the continuation test is a **descriptive pilot**; do not relax
the tiers. Results are reported separately per horizon
(`docs/strategy_scoreboard.md`), never combined across announcement /
effective-day / multi-day windows.

## Honest framing

Announcement drift is the **economically central historical hypothesis** for the
index effect. It is *not* asserted to be "the strong effect" here — whether it
survives recently, after costs, on this verified sample is exactly what the
continuation test must determine.
