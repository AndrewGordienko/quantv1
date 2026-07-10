# V3_PLAN.md — Legal public political-intelligence engine

v1 copied congressional trades. v2 proved (leak-free) that copying alone has no
robust factor-adjusted alpha. v3 reframes the whole system: **connect public
information faster and better than a normal investor** across many sources — the
only lawful edge. No MNPI, tips, leaks, hacked data, or breaches of a duty of
confidence, ever.

## The model

For each company i at time t, a decayed, novelty-weighted sum over layers:

```
Alpha[i,t] = P + G + F + M + E
  P  politician trades & ownership
  G  government policy / regulation / contracts / lobbying
  F  corporate fundamentals / earnings / insider (Form 4) filings
  M  market price / volume / volatility / options confirmation
  E  event novelty & estimated surprise (weights everything else)

rank[i,t] = (E[remaining factor-adjusted return] − costs) / forecast_risk
```

Trade only when the **conservative lower confidence bound** stays positive.
Exit when *remaining* expected alpha falls below costs — never on an arbitrary
stop (the correct answer to the MRVL problem).

## Flagship thesis (the immediate high-value experiment)

> large or repeated congressional purchase **+** direct political/company
> connection **+** a subsequent public policy/contract event **+** limited prior
> price reaction.

Much stronger than "a politician bought stock." This is what the graph + event
store are being built to detect.

## Temporal knowledge graph

`politician ↔ committee ↔ bill/regulator ↔ company ↔ lobbyist ↔ gov-contract ↔
SEC filing ↔ market event`, every edge timestamped with its public time.

## Data streams

| Layer | Sources (all public/free unless noted) | Extract |
|---|---|---|
| P politician | House/Senate PTRs, annual holdings | new vs add-on, self/spouse, size-vs-wealth, options, repeat |
| G government | Congress.gov, Federal Register, Regulations.gov, USAspending, SAM.gov, Senate LDA, OpenFEC | bills/votes/hearings, rules, contracts, lobbying topics, donors |
| F fundamentals | SEC EDGAR (8-K, 10-Q, **Form 4 insider**, 13D/G) | insider buys, ownership, financials, guidance |
| E earnings | 8-K earnings exhibits (paid consensus later) | EPS/rev surprise, guidance, estimate revisions, tone |
| M market | Alpaca/Polygon/IBKR | price reaction, rel-volume, vol, spreads, options confirm |

## Algorithms (in order of leverage)

1. **Entity resolution** — foundational. Politician name variants; company legal
   names / tickers / historical tickers; lobbying clients & subsidiaries;
   contractors → public parents; bills/rules → affected industries; committees →
   historical members. Deterministic IDs first, then embeddings + verified maps.
2. **LLM event extraction** — turn public text into structured events
   `{event_type, companies, direction, magnitude, novelty, effective_date,
   confidence, source_time}`. The LLM classifies; it never decides to trade.
   Every claim keeps its source + public timestamp.
3. **Event novelty/surprise** — matters only vs expectation: text-embedding
   nearest-neighbor to historical events, FinBERT sentiment, change-point
   detection on lobbying/contracts/spending, consensus surprise, abnormal
   volume/price right after publication.
4. **Hierarchical Bayesian source reliability** — per event-type / committee /
   agency / politician / insider / source. Unlike the (failed) permanent skill
   leaderboard, these **decay over time and shrink hard toward zero**.
5. **Multi-horizon alpha ensemble** — separate 1–5d / 21d / 63d models;
   Elastic-Net → CatBoost → LightGBM quantile → ensemble → conformal intervals.
   Predict **net residual return and quantiles**, never AUC.
6. **Remaining-alpha / hazard model** — daily: days-since-event, move-since-event,
   rel-volume, confirming/contradicting events, earnings proximity, current vol,
   thesis-still-active. Survival or Bayesian state-space. Drives exits.
7. **Portfolio optimization** — constrained mean-variance / CVaR; sector, beta,
   size, momentum, vol limits; cash allowed; size ∝ alpha/vol; penalize turnover
   & uncertainty. Benchmarks: SPY, QQQ, NANC/GOP, sector-matched. **No RL yet**
   (too few events, slow feedback → it would overfit the simulator).

## Build order

1. ~~Finish backtest_v2 & event_study_v2~~ — **DONE (v2).**
2. ~~Retire trend gate / permanent skill score / stop stack~~ — **DONE (shelved).**
3. **Entity graph + point-in-time event store** — foundation **STARTED**:
   `sec_entities` (ticker↔CIK, 1858/3439 traded resolved), generic `events`
   store (P/G/F/M/E, `source_time` gate), P layer populated (23k congress events).
4. Add G/F ingestion: **SEC EDGAR Form 4 insider** first (deterministic via CIK,
   pairs naturally with congress buys), then Congress.gov, Federal Register,
   USAspending, LDA, FEC.
5. Simple event-study baseline per new source (reuse `event_study_v2` machinery:
   next-open, factor-adjusted, cluster-robust, holdout).
6. Elastic-Net/CatBoost/LightGBM expected-return ensemble on combined features.
7. Daily remaining-alpha model.
8. Locked 2024–2026 OOS test.
9. Paper trade ≥ 3 months (Alpaca paper, sim $10k–$100k for power).
10. **$100 real-money canary** — operational only, not a performance test:
    2–3 fractional positions, ≤ $25–30 each, ≥ 25–40% cash, ≤ $50/day turnover,
    liquid US equities only, no options/leverage/shorting. Monitor continuously,
    **trade rarely** (event-driven, not high-frequency). Mind T+1 cash settlement.

## What exists now (this session)

- `db.py`: `sec_entities`, `events` (indexed), `factors` tables.
- `ingest/sec_entities.py`: SEC ticker↔CIK entity anchor.
- `events/store.py`: `upsert_events`, `populate_congress` (P layer), `layer_counts`.
- v2 research engine: `event_study_v2`, `backtest_v2`, `skill_persistence`,
  factor ingest — all leak-free, with a locked 2024+ holdout.

## Next concrete step

`ingest/edgar_form4.py` — for each CIK in `sec_entities` that a member traded,
pull recent Form 4 insider transactions from EDGAR, write `layer='F'` insider_buy
events, then run the flagship experiment: **congress purchase confirmed by an
insider buy within N days**, factor-adjusted, cluster-robust, holdout-tested.

## Legal

Lawful edge = speed/synthesis of PUBLIC info only. Never MNPI. If commercialized,
get legal advice — NANC's prospectus flags uncertainty on using PTRs commercially,
and personal research ≠ a product. This is a research system.
