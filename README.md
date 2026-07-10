# quantv1 — Congressional Trading Alpha Engine

Ingests US politicians' stock-trade disclosures, scores which members are
actually skilled, models which disclosed purchases are likely to beat the
market, and outputs a daily recommended portfolio — served on a localhost
dashboard. See [PLAN.md](PLAN.md) for the full methodology and rationale.

## Why this is non-trivial (the traps it avoids)

- **Filing-date discipline.** The STOCK Act gives members 30–45 days to
  disclose, so you only ever act on stale news. Every return, label, and
  backtest entry is measured from the **filing date**, never the transaction
  date. Using the transaction date is look-ahead bias and fakes alpha.
- **Small-sample skill.** Most members have few trades, so ranking by raw
  return surfaces luck. Skill is estimated with **empirical-Bayes shrinkage**
  and shown with 95% credible intervals.
- **Honest backtest.** The model is **refit at each rebalance** on
  only-already-realized outcomes, member-skill is recomputed point-in-time, and
  it is benchmarked against SPY *and* a naive copy-everything book.

## Architecture

```
Stock Watcher feeds (House: fresh + disclosure dates; Senate: historical)
        │  ingest/
        ▼
     DuckDB  ──►  research/ (event study, empirical-Bayes skill)
        │         model/    (features → LightGBM, walk-forward CV, native SHAP)
        │         portfolio/(construction + point-in-time backtest)
        ▼
   FastAPI (read-only JSON)  ◄──  React + Vite dashboard (6 pages)
```

## Setup

```bash
uv sync                       # Python deps
cd frontend && npm install    # frontend deps
```

## Run

```bash
# 1. Build/refresh all data + models (ingest → skill → model → portfolio → backtest)
uv run python scripts/daily_update.py

# 2. Start the API (terminal 1)
uv run uvicorn quantv1.api.app:app --port 8000

# 3. Start the dashboard (terminal 2)
cd frontend && npm run dev        # http://localhost:5173
```

Individual stages can be run alone, e.g. `uv run python -m quantv1.research.skill`
or `uv run python -m quantv1.research.event_study`.

## Data sources

- **House** — `TattooedHead/house-stock-watcher-data` (fresh through 2026, has
  disclosure dates — the point-in-time-clean primary source).
- **Senate** — `timothycarambat/senate-stock-watcher-data` (historical through
  ~2020, no disclosure date → filing date estimated and flagged).
- **Prices** — yfinance (split/dividend-adjusted daily bars).
- **Committees** — `unitedstates/congress-legislators` (public domain).

## Notes / honest limitations

- The Senate feed is stale and lacks disclosure dates; its rows carry
  `filing_estimated = TRUE` and are excluded from the rigorous backtest.
- Committee data covers **current** members only, so the committee-match feature
  is weaker for members who have since left Congress.
- Delisted/renamed tickers simply have no price and are treated as
  non-investable — no survivorship fabrication.
- yfinance sector lookups are slow; they fill in incrementally across daily runs.
