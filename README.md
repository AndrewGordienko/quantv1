# quantv1

Research infrastructure for one primary hypothesis: earnings information can
occasionally disagree with the market's first liquid reaction.

The project is deliberately blocked until it has licensed point-in-time
expectations data. Current consensus, revised history, and final-period outcomes
are not substitutes.

## Strategy

For every verified earnings release, wait 30 minutes into the first liquid
regular session and measure:

```text
mismatch = fundamental surprise - standardized residual reaction
```

An elastic net predicts the five-day sector-adjusted return. A trade is allowed
only when the absolute prediction exceeds twice the estimated stock-plus-hedge
round-trip cost. Positions are beta hedged and close on the fifth common trading
day.

See [docs/strategy.md](docs/strategy.md) for the frozen protocol and
[docs/data.md](docs/data.md) for the point-in-time data contract. Prior tested
hypotheses and negative results are recorded in
[docs/research_ledger.md](docs/research_ledger.md).

## Status

- M0: price and reaction baseline available.
- M1/M2: blocked until representative EPS+revenue coverage reaches 80% in both
  training and validation.
- Final period: sealed. The code refuses to lock or open it unless M2 clears all
  validation gates.
- CatBoost: prohibited until M2 elastic net demonstrates signal.

## Commands

```bash
uv sync
uv run python -m unittest discover -s tests

uv run python scripts/earnings_sprint.py audit
uv run python scripts/earnings_sprint.py features
uv run python scripts/earnings_sprint.py run
```

Manifest ingestion and market-window acquisition commands are listed by:

```bash
uv run python scripts/earnings_sprint.py --help
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
