# Earnings data manifests

Generated/current consensus must never be backfilled into historical events.
The loaders reject unsafe records rather than substituting them.

## Reviewed release

```json
{
  "ticker": "AAPL",
  "cik": "0000320193",
  "fiscal_period_end": "2025-06-28",
  "fiscal_quarter": "Q3",
  "public_time": "2025-07-31T16:30:00-04:00",
  "source_type": "company_ir",
  "source_url": "https://investor.example.com/exact-release",
  "reviewed_earliest": true,
  "source_sha256": "optional-content-digest"
}
```

`public_time` must include an offset. Without `earnings_event_id`, the loader
links the direct release to the closest unverified SEC Item 2.02 candidate within
three days.

## SEC candidate classification

```json
{
  "earnings_event_id": "candidate-id",
  "event_classification": "VERIFIED_EARNINGS_RELEASE",
  "fiscal_period_end": "2025-06-28",
  "fiscal_quarter": "Q3"
}
```

The other valid classification is `NOT_EARNINGS`. A verified SEC record retains
`CONSERVATIVE_SEC_ONLY` status because an earlier IR/wire release may exist.

## Historical consensus snapshot

```json
{
  "earnings_event_id": "event-id",
  "metric": "diluted_eps",
  "estimate_value": 1.43,
  "currency": "USD",
  "analyst_count": 31,
  "estimate_as_of": "2025-07-31T15:55:00-04:00",
  "vendor": "licensed-vendor",
  "vendor_record_id": "immutable-record-id",
  "is_point_in_time": true,
  "is_final_revised": false
}
```

The snapshot must predate the release. Supported primary metrics are
`diluted_eps`, `revenue`, `gross_margin`, `free_cash_flow`, `bookings`,
`guidance_eps` and `guidance_revenue`.

## Actual

```json
{
  "earnings_event_id": "event-id",
  "metric": "diluted_eps",
  "actual_value": 1.57,
  "currency": "USD",
  "public_time": "2025-07-31T16:30:00-04:00",
  "source": "company_ir",
  "source_url": "https://investor.example.com/exact-release"
}
```

Guidance manifests carry metric, guidance period, lower/upper values and their
public time. Options manifests carry a pre-release straddle mid, underlying mid,
expiry and immutable source record ID; positioning manifests carry their own
pre-release observation time. Both are rejected at or after release time.

Call manifests require an exact call timestamp and `prepared`, `question` or
`answer` segments. They store raw speaker/section evidence only—behavioral labels
are deliberately deferred until the financial/reaction model establishes signal.
