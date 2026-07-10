# quantv1 — Congressional Trading Alpha Engine

A system that ingests US politicians' stock-trade disclosures, figures out *who* is actually good and *why* they might be trading, models expected excess returns, and outputs a daily recommended portfolio — all served on a localhost dashboard with a clean Google/DeepMind aesthetic.

---

## 0. Reality check (read this first)

Things people usually don't realize going in — these shape the whole design:

1. **You are always trading stale information.** The STOCK Act gives members 30–45 days to disclose. You never see the trade when it happens; you see it when it's *filed*. **Every backtest must use the disclosure/filing date, not the transaction date.** Using the transaction date is look-ahead bias and will make any strategy look brilliant. This is the #1 mistake in every amateur congress-tracker backtest.
2. **You never know the trade size.** Disclosures report ranges: $1,001–$15,000, $15,001–$50,000, … up to $50M+. Use the range midpoint (log-scaled), and treat "large relative to that member's own history" as more informative than absolute size.
3. **The academic evidence is mixed.** Pre-STOCK Act (Ziobrowski et al. 2004/2011): senators beat the market by ~85bps/month. Post-2012 studies (e.g., Belmont, Firebaugh, Sacerdote 2022) find the *average* member does **not** beat the market. But the average isn't the strategy — the whole edge, if it exists, is in **selection**: a small tail of members, specific trade types (purchases > sales), committee-relevant trades, and cluster events. That's exactly what a model can find.
4. **Sales are weak signals.** Members sell for liquidity, diversification, ethics-compliance, divorce. Purchases are the deliberate act. Most of the signal literature focuses on purchases only.
5. **Many trades aren't even theirs** — spouse accounts, dependent children, blind-ish managed accounts. The `owner` field matters as a feature.
6. **This is 100% legal.** These are mandated public disclosures. You're doing "copy-trading on public filings," not insider trading. Two real ETFs already do a naive version of this: **NANC** (tracks Democratic members, notably Pelosi-heavy — beat SPY in 2023–24) and **KRUZ** (Republican members). These are your *benchmarks to beat* — if your model can't outperform naive copy-everything ETFs, the model isn't adding value.

---

## 1. Data layer

### 1.1 Trade disclosures (the core dataset)

| Source | Cost | Notes |
|---|---|---|
| [Senate Stock Watcher data](https://github.com/timothycarambat/senate-stock-watcher-data) | Free | Daily-updated JSON on GitHub/S3, parsed from efdsearch.senate.gov. **Start here.** |
| House Stock Watcher (same author) | Free | Same format for House PTRs. |
| [Capitol Trades](https://www.capitoltrades.com) | Free (site) | Cleanest normalization; scrape-able for gap-filling / cross-validation. |
| [Quiver Quantitative](https://www.quiverquant.com/congresstrading/) | Paid API | Nicest structured API if we later want reliability. |
| [Finnhub congressional endpoint](https://finnhub.io/docs/api/congressional-trading) / [FMP](https://site.financialmodelingprep.com/datasets/ownership-senate-insider) | Freemium | Backup options. |
| Official sources: disclosures-clerk.house.gov, efdsearch.senate.gov | Free | Ground truth (PDFs/XML). Only parse directly if the aggregators fail us. |

**Plan:** ingest the two free Stock Watcher datasets first, cross-check a sample against Capitol Trades, and design the ingest module behind an interface so a paid API can be swapped in later.

### 1.2 Market data
- **yfinance** for daily OHLCV — free, fine for daily-frequency work.
- SPY (or an investable benchmark) + sector ETFs (XLK, XLE, XLV…) for benchmarking and factor exposure.
- Fama-French daily factors (free from Ken French's data library) for proper alpha attribution.

### 1.3 Context/enrichment data (this is the "why are they trading this" layer)
- **Committee assignments** — unitedstates/congress-legislators GitHub repo (free YAML/JSON): member ↔ committee ↔ jurisdiction. The key feature: *is this stock in a sector this member's committee oversees?* (Armed Services member buying Lockheed ≠ random trade.)
- **Ticker → sector/industry mapping** (yfinance metadata or a static GICS table).
- **Bill/legislation calendar** (optional, later): GovTrack/ProPublica Congress API — trades placed shortly before relevant votes are the spicy ones.
- **Member metadata**: party, chamber, state, tenure, leadership role.

### 1.4 Storage
- **DuckDB** — single file, zero ops, fast analytics SQL, plays perfectly with pandas/polars. Tables:
  - `trades` (raw normalized disclosures: member, ticker, tx_date, **filing_date**, type, amount_lo/hi, owner, asset_type)
  - `prices` (daily bars)
  - `members` (metadata + committees)
  - `features`, `scores`, `signals`, `portfolios`, `backtests` (derived, regenerable)
- Everything **point-in-time**: derived tables keyed by "as of" date so backtests only see what was knowable then.

---

## 2. Quant methodology

Four layers, each independently useful and inspectable on the dashboard.

### 2.1 Event studies — "does congressional buying predict returns at all?"
For every disclosed purchase, compute **abnormal returns** after the *filing date*:
- Horizons: 5, 21, 63, 126 trading days.
- Abnormal = stock return − beta-adjusted market return (CAPM residual; upgrade to Fama-French 3-factor later).
- Aggregate: average CAR (cumulative abnormal return) across all trades, then sliced by chamber, party, purchase-vs-sale, amount bucket, committee-match, member.

This is the honest foundation. If the all-trades CAR is ~0 (likely), the slices tell you where signal concentrates — and those slices become model features.

### 2.2 Politician skill scoring — "who is actually good?"
The leaderboard, and the hardest statistical problem here: most members have few trades, so naive average-return rankings are dominated by noise (someone with 3 lucky trades tops the list).

- **Empirical-Bayes shrinkage**: model each member's per-trade abnormal return as drawn from member skill μᵢ ~ Normal(μ₀, τ²); estimate the hyperparameters from all members, then shrink each member's raw mean toward the population mean proportional to how few trades they have. (Same math as baseball batting-average shrinkage.) A member with 200 trades keeps their estimate; a member with 4 trades gets pulled to ~average.
- Report **credible intervals**, not point estimates — the dashboard should show uncertainty bars on the leaderboard. That's the DeepMind touch: honest uncertainty.
- Secondary metrics per member: hit rate vs SPY, purchase-only CAR, information ratio, trade frequency, average disclosure lag (fast filers may be more informative).

### 2.3 Signal model — "given a new disclosure, how excited should we be?"
Supervised model scoring each new disclosed purchase.

- **Label:** did the stock beat SPY over the next 63 trading days after filing? (binary to start; regression on excess return later).
- **Features:**
  - Member skill score (from 2.2) — the big one
  - Committee–sector match (boolean + committee-specific historical CAR)
  - Amount-range midpoint, and size relative to member's own trade history
  - Purchase vs sale, owner (self/spouse/child), asset type (stock vs options — options trades are rarer and far more deliberate)
  - **Cluster buying**: # of distinct members buying the same ticker within 30 days (consensus is historically one of the strongest congress-signals)
  - Disclosure lag (tx_date → filing_date)
  - Stock context: sector, market cap bucket, 1/3/12-month momentum, volatility
  - Party × sector interactions (e.g., energy trades have historically differed by party)
- **Model:** logistic regression first (interpretable baseline), then **LightGBM** with **walk-forward time-series CV** (train on 2013–2019, validate 2020, test 2021 → roll forward). Never random K-fold — that leaks future information.
- **Explainability:** SHAP values per prediction → the dashboard shows *why* each stock is recommended ("Pelosi-tier trader + Armed Services match + 3-member cluster"). This is the "why are they trading it" answer, quantified.

### 2.4 Portfolio construction — "what do we hold today?"
- Universe each day = disclosures filed in the last N days (e.g., 90), scored by the model.
- Hold **top-K (~15–25)** names above a score threshold; score-weighted or equal-weighted with a per-position cap (e.g., 8%); optional sector cap (30%) so it doesn't become an all-tech fund.
- Exit: fixed horizon (~63 trading days), score decay, or a disclosed sale by the originating member.
- Daily job output: **target portfolio + explicit buy/sell delta vs yesterday**, with per-position rationale.

### 2.5 Backtesting — where these projects live or die
- Event-driven daily loop over point-in-time data; positions entered at **next day's open after filing** (you can't trade on a filing before you've seen it).
- Costs: ~10bps per side + slippage; skip/flag illiquid names (< $5M median daily dollar volume).
- Metrics: CAGR, Sharpe, information ratio vs SPY, max drawdown, turnover, exposure over time.
- **Benchmarks to beat, in order:** SPY buy-and-hold → naive copy-all-purchases equal-weight → NANC ETF. Each layer of modeling has to justify itself against the dumber version.
- Pitfall checklist: filing-date-only ✅, survivorship (delisted tickers — flag and exclude honestly) ✅, no random CV ✅, ticker changes/splits (yfinance auto-adjusts) ✅, don't tune on the test period ✅.

---

## 3. The dashboard (localhost, DeepMind vibe)

**Stack:** FastAPI backend (serves JSON from DuckDB) + **React + Vite + Tailwind** frontend, single `docker-free` local setup: `uvicorn` on :8000, Vite dev on :5173 proxying `/api`.

**Design language:** near-black background (#0a0a0f), one restrained accent (electric blue or mint), Inter/Söhne-style typography, generous whitespace, thin-line charts, uncertainty bands drawn explicitly, subtle motion. No bootstrap-dashboard clutter.

**Pages:**
1. **Today** — the money page. Recommended portfolio, buy/sell deltas since yesterday, each position expandable to its SHAP rationale.
2. **Live feed** — latest disclosures as they land, scored in real time, "days stale" indicator.
3. **Leaderboard** — shrunken skill scores with credible-interval bars, sortable; click into a member.
4. **Member profile** — trade history timeline, cumulative alpha curve, committee assignments, sector heatmap of their trades.
5. **Ticker view** — which members hold/traded it, price chart with disclosure markers.
6. **Research** — event-study CAR curves, backtest equity curve vs SPY/NANC, model metrics per walk-forward window.

---

## 4. Repo layout

```
quantv1/
├── PLAN.md
├── pyproject.toml            # uv-managed
├── data/                     # duckdb file + raw cache (gitignored)
├── src/quantv1/
│   ├── ingest/               # stockwatcher.py, prices.py, committees.py
│   ├── db.py                 # duckdb schema + point-in-time helpers
│   ├── research/             # event_study.py, skill.py (empirical Bayes)
│   ├── model/                # features.py, train.py, predict.py
│   ├── portfolio/            # construct.py, backtest.py
│   └── api/                  # fastapi app
├── frontend/                 # react + vite + tailwind
├── scripts/                  # daily_update.py (cron-able)
└── notebooks/                # exploration
```

**Python stack:** uv, pandas + polars, duckdb, yfinance, scikit-learn, lightgbm, shap, fastapi, uvicorn, matplotlib (research plots).

---

## 5. Build phases

**Phase 1 — Data foundation.** Ingest Senate + House Stock Watcher JSON → normalized DuckDB `trades`; pull prices for all disclosed tickers + SPY; committee data. Sanity dashboard-in-a-notebook: trades/month, top tickers, disclosure-lag distribution. *Milestone: `SELECT * FROM trades` looks trustworthy.*

**Phase 2 — Event study + leaderboard.** Abnormal-return engine, CAR curves by slice, empirical-Bayes skill scores. *Milestone: we know whether/where signal exists — this decides how fancy Phase 3 gets.*

**Phase 3 — Signal model.** Feature pipeline, logistic baseline, LightGBM + walk-forward CV, SHAP. *Milestone: out-of-sample AUC/IC meaningfully above chance on filing-date data.*

**Phase 4 — Portfolio + backtest.** Construction rules, cost-aware backtest, benchmark comparisons. *Milestone: equity curve vs SPY and naive-copy, honest verdict.*

**Phase 5 — Dashboard.** FastAPI endpoints + React frontend, all six pages. *Milestone: `make dev` → localhost shows today's portfolio.*

**Phase 6 — Daily operation.** `scripts/daily_update.py` (ingest → score → rebalance → persist), cron/launchd, feed freshness indicators.

**Later / stretch:** options-trade decoding, corporate-insider Form 4 overlay (same architecture, different filers), bill-calendar proximity features, paper-trading via Alpaca API, regime awareness (does the signal decay as more people copy it?).

---

## 6. Honest expectations

The realistic best case is a portfolio with a modest information ratio over SPY, concentrated in a handful of high-conviction cluster/committee trades per month — not a money printer. The disclosure lag caps how much edge can survive. But as a *quant learning project* this is close to ideal: real messy alt-data, a real statistical trap (small-sample skill estimation), a real bias trap (look-ahead via transaction dates), proper backtesting discipline, interpretable ML, and a product-shaped output. Even a null result — "congress-copying has no edge post-2020, here's the evidence" — is a genuinely good piece of research, and the dashboard makes it fun either way.
