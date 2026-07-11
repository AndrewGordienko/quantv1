# quantv1

Research infrastructure for testing whether public information can occasionally
disagree with the market's first liquid reaction to an earnings event.

The original hypothesis (EERM) is **economically data-blocked**: its M1/M2
mismatch models need point-in-time analyst consensus, which is only sold by
vendors or WRDS. That protocol is frozen and preserved as
`BLOCKED_DATA_ECONOMICALLY_INACCESSIBLE`; current consensus, revised history, and
final-period outcomes are not substitutes.

The **active** hypothesis is MGRM — Management Guidance Revision-Reaction
Mismatch — which reuses the same frozen reaction engine using only automatically
retrievable public SEC 8-K Item 2.02/7.01 data, with no analyst consensus.

> **No historical alpha test has been run.** MGRM extraction is currently
> **uncertified**: the guidance extractor has not passed a frozen gold-set
> accuracy audit, so no G1/G2 features are fit, no model is locked, and the
> holdout stays sealed. If MGRM ultimately fails, **forced flows** remain the
> next independent hypothesis.

## Strategy

**EERM (blocked).** For every verified earnings release, wait 30 minutes into the
first liquid regular session and measure `mismatch = fundamental surprise −
standardized residual reaction`.

**MGRM (active, zero-vendor).** When management revises its *own* forward
guidance, measure whether the initial reaction fully incorporated that change:
`mismatch = normalized guidance revision − standardized residual reaction`. G0
(reaction-only) is a diagnostic; G1 (structured guidance) and G2 (mismatch) are
fit **only** after the extractor is certified against the gold set and the data
gate passes. An elastic net predicts the five-day frozen-beta-hedged return;
trades require the prediction to exceed twice the estimated round-trip cost, are
beta hedged, and close on the fifth common session. No CatBoost until G2 shows
incremental signal.

See [docs/strategy.md](docs/strategy.md) for the frozen protocol and
[docs/data.md](docs/data.md) for the point-in-time data contract. Prior tested
hypotheses and negative results are recorded in
[docs/research_ledger.md](docs/research_ledger.md).

## Status

- EERM M0: price and reaction baseline available.
- EERM M1/M2: `BLOCKED_DATA_ECONOMICALLY_INACCESSIBLE` (paid consensus). Protocol
  frozen, not reused.
- MGRM extraction: **uncertified** — awaiting a human-labelled gold set (≥30 real
  filings, ≥5 sectors, ≥2 formats) and a configured extraction backend so the
  reconciled AGREED output can be certified.
- MGRM G1/G2 fitting, model locking, and holdout opening: **fail closed** without
  a valid certification.
- Retrospective 2025-07 holdout: sealed for both tracks.
- CatBoost, behavior embeddings, world models: prohibited until G2 shows signal.

## Commands

```bash
uv sync
uv run python -m unittest discover -s tests

# EERM (blocked) driver
uv run python scripts/earnings_sprint.py audit

# MGRM (active) driver
uv run python scripts/mgrm_sprint.py discover --tickers AAPL,MSFT
uv run python scripts/mgrm_sprint.py extract
uv run python scripts/mgrm_sprint.py link
uv run python scripts/mgrm_sprint.py goldset   # gold-set audit + certification
uv run python scripts/mgrm_sprint.py audit     # data gate (incl. goldset_certified)
uv run python scripts/mgrm_sprint.py run
```

Set the extraction backend with `MGRM_LLM_PROVIDER` (`openai` or `ollama`); with
no backend the extractor fails closed and cannot certify. Full command lists:

```bash
uv run python scripts/earnings_sprint.py --help
uv run python scripts/mgrm_sprint.py --help
```

## Layout

```text
src/quantv1/ingest/       point-in-time event and expectations intake
src/quantv1/research/     feature construction and validation contest
src/quantv1/portfolio/    constraints and daily mark-to-market ledger
scripts/                  narrow command-line entry points
tests/                    leakage, accounting, and protocol invariants
docs/                     active strategy and data contracts
```

Generated data lives under `data/` and is never committed. Engineering
conventions are in [STYLE.md](STYLE.md).
