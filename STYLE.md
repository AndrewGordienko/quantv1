# Style

Keep the repository small, explicit, and difficult to misuse.

## Code

- Prefer a plain function or dataclass over a framework.
- Keep modules narrow; split accounting, ingestion, and modeling boundaries.
- Use one canonical definition for constants and trading decisions.
- Make time, units, price basis, and data provenance visible in names.
- Fail closed on missing point-in-time fields. Never silently impute provenance.
- Keep research stages nested and deterministic. Seed every randomized control.
- Do not add model complexity to rescue a failed hypothesis.

## Tests

- Test invariants and leakage boundaries, not implementation trivia.
- Every accounting change needs a cash/NAV path test.
- Every new point-in-time feature needs a future-data rejection test.
- Run `uv run python -m unittest discover -s tests` before publishing.

## Repository

- One active strategy document and one data contract.
- Generated data, caches, reports, models, and credentials stay untracked.
- Delete superseded plans instead of appending another versioned memo.
- Commit one coherent change with a terse, descriptive subject.
