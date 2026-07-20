# SEC Event Atlas — bounded primary lane

The Event Atlas is the new discovery substrate. MGRM, government contracts,
insider clusters and actor context become event families or modifiers inside
one source-anchored corpus; they are not separate strategy oracles.

## Frozen stages

The Atlas uses exact public SEC acceptance timestamps, permanent CIKs and
accession numbers, immutable source URLs/hashes, and an extraction version.
The initial taxonomy has 15 auditable families and 51 explicit event types:

guidance, leadership, auditor, restructuring, capital return,
financing/dilution, M&A, commercial contracts, government contracts,
litigation/regulatory, activist ownership, insider, going concern,
cybersecurity, and restatement/internal controls.

The canonical unsigned manifest is ingested with:

```bash
uv run python scripts/sec_event_atlas.py ingest PATH/events.jsonl
```

Each line must be source-anchored and must not contain direction, returns,
labels, trade sides, or realized outcomes. Invalid records reject the complete
manifest; no partial corpus is inserted.

### Stage 1 — unsigned importance

```bash
uv run python scripts/sec_event_atlas.py unsigned
```

This reports family support and absolute one/five-session residual movement or
volatility diagnostics. It does not choose a trade direction. Discovery is
2022–2024, validation is 2025, and 2026 is prospective. The target universe is
1,000–3,000 US issuers, but the Atlas does not claim that coverage until the
manifest proves it.

## Phase-A pilot snapshot

The first real pilot froze 500 unique CIKs and 8,906 qualifying 8-K/8-K/A
candidates, then selected 500 filings by accession hash without inspecting
prices. Extraction produced 2,386 tagged events across 494 accessions, 221
issuers and all three discovery years, plus six observed no-material controls.
The strengthened stratified human-label queue contains 80 unlabeled records:
four per family, 20 controls (including 14 candidate controls), and
issuer-disjoint development/sealed partitions. Multi-event and exhibit-driven
filings are explicitly marked. `scripts/sec_goldset.py` scores reviewer-supplied
detection, type, evidence-grounding, and magnitude fields without imputing
unlabeled values.

The measured Stage-1 artifact is `data/sec_event_atlas_unsigned.json` and is
also served at `/api/research/event-atlas`. It currently reports 953/2,386
(39.9%) event tags with a local price window. This is a **current-ticker
diagnostic only**: point-in-time ticker mapping and delisting coverage are not
available, so no directional conclusion is permitted. Only 14 of the 221 event
tickers have a current `ticker_sectors` market-cap proxy, which confirms the
price/security-master coverage bottleneck rather than hiding it. Extraction
precision and recall remain `null` until the human gold set is labeled.

Tags sharing an accession/date are one market observation, not independent
events. Unsigned effects and controls cluster by accession/date and issuer;
portfolio replay must permit at most one position per accession/ticker.

The real slim event catalog and unlabeled gold queue are published as
`goldset/sec_event_atlas_phaseA_pilot.jsonl` (2,386 records) and
`goldset/sec_event_atlas_goldset_skeleton.jsonl` (80 records). The measured
summary is frozen in `goldset/sec_event_atlas_unsigned_summary.json`.

### Stage 2 — directional base rates

Only after Stage 1 clears support, year, ticker and sector breadth gates, freeze
no more than five event families from discovery. Estimate directional one/five-
day residuals with hierarchical/empirical-Bayes shrinkage, event-date clustered
uncertainty, and magnitude monotonicity. No family is selected on validation or
eventual returns.

### Stage 3–5

Then test reaction mismatch, peer diffusion and execution costs; run next-open
or next-executable replay with spreads, impact, borrow, sector hedges, delayed
entry and doubled costs; finally freeze a prospective shadow record. LLMs and
embeddings extract/cluster documents only. They do not issue trades.

## Portfolio lanes

- Lane A — SEC Event Atlas: 60% effort, primary discovery engine.
- Lane B — S&P announcement continuation: 25%, finish the 113-batch timestamp
  census and test once.
- Lane C — Opening Flow execution: 15%, operational paper canary only.

Actor expansion, additional bar-only rules, crypto expansion, and separate MGRM
infrastructure are paused. If the Atlas produces no post-cost directional edge
on untouched validation, the free-public-data equity hypothesis is closed rather
than rescued with a larger model.
