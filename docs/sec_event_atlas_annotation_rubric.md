# SEC Event Atlas annotation rubric (frozen v1)

This rubric is frozen before review. The development partition (63 issuers)
may be used to correct extraction. The sealed partition (17 issuers) is scored
once, after the extractor version is frozen; it is not used for corrections or
family selection.

For each record, the reviewer supplies:

- `human_label`: `material_event`, `routine`, or `uncertain`.
- `document_detection`: whether the filing contains a consequential event.
- `event_type_label`: one taxonomy type, `MULTI_EVENT`, `ROUTINE`, or `UNKNOWN`.
- `evidence_span` and `evidence_grounding`: an exact source excerpt/location
  supporting the label.
- `magnitude_label`: structured magnitude when the event exposes one, otherwise
  `not_applicable` or `not_disclosed`.

Acceptance thresholds are fixed before certification:

- Document detection precision and recall: at least 0.85 on development and
  at least 0.80 on sealed aggregate certification.
- Event-type accuracy: at least 0.80 on labeled material events.
- Evidence grounding: at least 0.90 of labeled event records must cite the
  primary filing or an identified exhibit span.
- Routine controls: at least 0.90 confirmed as routine/no-material.
- No family is certified if its labeled support is fewer than two records; the
  17-record sealed set is aggregate evidence, not per-family proof.

Any `uncertain` record remains in the denominator and is reported separately;
it is not silently converted into a negative. A failed threshold freezes the
extractor as rejected and requires a new versioned development cycle.
