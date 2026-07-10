# Point-in-time data contract

M1/M2 require a licensed archived estimates feed. The local market-data key does
not include that entitlement, and an earnings calendar containing a latest
estimate is not point-in-time history.

Suitable products must expose immutable observation timestamps. Examples to
evaluate commercially are S&P Capital IQ Estimates Snapshot, FactSet Estimates
Point-in-Time Consensus, and LSEG I/B/E/S Point-in-Time. No vendor is accepted
until a sample passes the audit command.

## Required fields

Every record requires a vendor/source ID, source provenance, feature version, and
`known_at` timestamp.

- EPS and revenue consensus, actual, currency, basis, analyst count, and
  dispersion.
- Estimate additions/removals/revisions sufficient to calculate revision breadth.
- Previous and new guidance, plus pre-release consensus for the guided period.
- Earliest verified release timestamp and fiscal-period mapping.
- Point-in-time company-size classification for coverage auditing.
- Historical stock-loan availability and annualized fee, with an immutable
  `borrow_known_at` timestamp. Missing borrow makes short results non-deployable.
- Preferably a pre-release options-implied move.

The intake rejects snapshots at or after release, current values backfilled into
history, and final revised series.

## Coverage gate

Run:

```bash
uv run python scripts/earnings_sprint.py audit
```

The gate requires at least 80% joint EPS+revenue coverage in both training and
validation. Every eligible year, sector, and company-size group (20 or more
events) must also reach 80%. Coverage reports event counts, rates, missing
dimensions, and the underlying table counts.

## Manifest shape

Consensus records are JSON or JSONL:

```json
{
  "earnings_event_id": "event-id",
  "metric": "diluted_eps",
  "estimate_value": 1.43,
  "currency": "USD",
  "analyst_count": 31,
  "forecast_dispersion": 0.08,
  "revision_breadth": 0.42,
  "estimate_as_of": "2025-04-30T19:55:00Z",
  "vendor": "licensed-vendor",
  "vendor_record_id": "immutable-record-id",
  "is_point_in_time": true,
  "is_final_revised": false
}
```

Actual and guidance manifests additionally require their public timestamp,
immutable source record ID, source URL, currency/basis, and explicit guidance
role (`previous` or `new`). Loaders fail closed on invalid timestamps.

Company-size coverage uses a separate pre-sample manifest:

```json
{
  "ticker": "AAPL",
  "market_cap": 2100000000000,
  "known_at": "2021-06-29T20:00:00Z",
  "source": "licensed-vendor",
  "source_record_id": "immutable-size-record"
}
```

Load with `earnings_sprint.py universe-metadata FILE`. Observations after the
frozen universe date are rejected.

Positioning manifests may include `borrow_available`,
`borrow_fee_bps_annual`, and `borrow_known_at`. Available borrow requires a
nonnegative fee and a timestamp known before the release; assumed generic borrow
fees are not accepted for the primary long/short test.
