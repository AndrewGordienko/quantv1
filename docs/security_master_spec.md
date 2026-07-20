# Point-in-time security master gate

The SEC Event Atlas does not use ticker as a permanent identity. `security_id`
is keyed by CIK and instrument class; ticker, exchange, and effective intervals
are observations sourced from the filing available at its public timestamp.

`src/quantv1/ingest/security_master.py` extracts inline-XBRL cover fields first,
then cover text, and records lower-confidence fallback mappings separately.
Successive observations create intervals. Same-time conflicting mappings are
written to a conflict ledger and are not promotable. Delisting is counted only
when a source explicitly supplies a delisted date; an interval ending because a
later filing arrived is not evidence of delisting.

The coverage audit reports mapping and price-window coverage by family, year,
and issuer, plus linked-versus-unlinked missingness. Promotion requires the
predeclared 80% mapping/window rates, a 60% family floor, and explicit
delisting coverage. Until those gates pass, the unsigned Atlas report remains
diagnostic and no directional model may be fit.

```bash
uv run python scripts/security_master.py pilot --output goldset/sec_event_atlas_security_master_pilot.json
uv run python scripts/security_master.py build filings.jsonl mappings.json
uv run python scripts/security_master.py coverage coverage_input.json coverage.json
```

The pilot currently extracts 376 mappings from 500 cached filing documents
(before the v2 parser); the frozen v2 parser extracts 472/500 (94.4%), with
multiple listed classes retained in metadata. On the actual event corpus this
is 466/494 accessions (94.3%) and 2,279/2,386 tags (95.5%), with 28 accessions
remaining unmapped. The complete denominator comparison is
in `goldset/sec_event_atlas_coverage_audit.json`.

The linkage failure decomposition is in
`goldset/sec_event_atlas_linkage_failure_decomposition.json`. It finds 28
unmapped accessions but only 107 tags—not 547—so the apparent discrepancy was
caused by comparing the earlier pre-v2 tag count with the current accession
mapping. Failures are broken down by category, family, and year.

Trading rule is frozen as `PRIMARY_LIQUID_COMMON_CLASS_ISSUER_CAP_V1`: select
one eligible common class per issuer using trailing dollar ADV when available,
otherwise a deterministic security-id order; cap exposure once at issuer level.
Other classes remain in the identity master but are not silently added as
separate positions.

This is still not a passed promotion gate: the pilot has no delisting evidence
or survivorship-safe price windows yet. Mapping success alone cannot authorize
Stage 1.

The 20-security feasibility audit is generated with:

```bash
uv run python scripts/price_feasibility.py
```

It currently fails closed: only 10/20 symbols have local price rows, and the
price table has no adjusted bars, corporate-action factors, listing intervals,
delisting evidence, or terminal status fields.
