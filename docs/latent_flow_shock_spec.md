# Latent Flow Shock — F1 bar-only research specification

## Decision

This is the third research engine alongside forced flow and MGRM. Its question
is whether an idiosyncratic intraday move reflects persistent flow or a temporary
liquidity disturbance. It must not be confused with the rejected generic
mean-reversion strategy: continuation is considered only after a mechanism
fingerprint is present; the reversal fingerprint is diagnostic until quotes and
trades support it.

## F1: formally rejected

The local database has minute OHLCV/VWAP data for 18 stocks plus factor ETFs;
the currently mapped factor-complete F1 subset is 13 names. It has no historical
NBBO, trade direction, quote size, or depth. F1 is therefore a screening and
data-contract exercise, not a deployable strategy.

At each regular-session minute, F1 calculates a rolling, pre-bar-frozen
two-factor residual:

`r_res = r_stock - beta_market * r_SPY - beta_sector * (r_sector - r_SPY)`.

It then combines:

- residual-return z-score;
- time-of-day relative volume (same minute in prior sessions);
- VWAP distance, retained as a diagnostic rather than a fitted input;
- four-minute signed-volume *proxy* (`sign(residual) × bar volume`), clearly
  not buyer/seller initiated order-flow imbalance;
- peer confirmation from other local names in the same sector;
- a direction-aware CUSUM change-point score; and
- an impact proxy, `abs(residual) / relative_volume`.

The frozen continuation gate requires a large residual, elevated volume,
persistent signed proxy, aligned score, and CUSUM state. A high-impact,
low-volume, unconfirmed jump is recorded as a reversal diagnostic but is never
traded. Entries are next-minute opens; all displayed results are factor-adjusted
and deduct a conservative 16 bps estimated round-trip cost.

Run it with:

```bash
uv run python scripts/latent_flow_sprint.py
```

The machine-readable output is `data/latent_flow_f1.json`.

## Result — do not rescue

On the current 2024-07-15 to 2026-07-09 narrow universe, the frozen F1 screen
emitted 157 de-duplicated episodes, of which 127 were executable without
crossing a session at the 30-minute horizon. Its 30-minute holdout mean net
outcome was **−18.2 bps** after the stated cost (38 episodes). The one-minute delayed and
doubled-cost checks were also negative. This is neither an alpha claim nor a
rejection of the full latent-flow hypothesis: F1 lacks the data that distinguishes
informed flow from liquidity withdrawal. It is a negative price/volume-only
screen, and it must not be rescued by threshold tuning. F1 is formally rejected.

## F2 data gate — the only permitted next step

Before any F2 feature or model code, obtain one small immutable sample: 10–20
liquid stocks, 20–40 sessions, every trade and NBBO update, and separate halt
and corporate-action files. Quotes must carry exchange timestamp, sequence,
bid, ask, bid size, ask size, and condition codes. Trades must carry exchange
timestamp, sequence, price, size, condition codes, and correction/cancellation
codes.

The selection is already frozen in
`goldset/microstructure/f2_pilot_selection_v1.json`: 11 liquid sector
representatives, SPY, their mapped sector ETFs, and 30 contiguous full XNYS
sessions. Its SHA-256 is pinned in the sample-manifest skeleton before any
observations are downloaded. The audit requires all selected stock, SPY, and
sector-ETF feeds on the exact selected sessions; it rejects omitted, substituted,
or extra symbols. The selection states explicitly that no returns, moves, news,
or volatility were used to choose symbols or days.

Use the skeleton at
`goldset/microstructure/f2_sample_manifest.skeleton.json`, fill it with vendor
documentation and SHA-256 hashes for the immutable exports, then run:

```bash
uv run python scripts/latent_flow_f2_audit.py PATH/manifest.json
```

The gate rejects a source without documented historical availability, exchange
timestamp + sequence ordering in the vendor's documented domain, a complete-NBBO attestation,
condition-code/correction provenance, hashes, 10–20 symbols, or 20–40 sessions.
It also rejects duplicate order keys, non-monotonic raw ordering, crossed quotes,
invalid prices/sizes, and missing required columns. Completeness remains a
vendor claim checked against its frozen documentation; an export alone cannot
independently prove that an upstream feed update was never lost.

The sequence domain is declared as a list such as `ticker, session` or
`ticker, venue, session`; feeds can additionally declare `channel`. The audit
groups ordering checks by that domain, so it does not falsely require one global
counter when a venue or channel maintains a separate sequence. Equal exchange
timestamps are valid only when distinct, increasing sequence values resolve the
tie.

No F2 implementation exists until this gate returns
`ACCEPTED_FOR_F2_FEATURE_RESEARCH`.

## Required ladder

| Rung | Adds | Status |
|---|---|---|
| F0 | Price and raw volume | superseded by F1 screen |
| F1 | Factor residual, CUSUM, time-of-day volume, peers, impact proxy | rejected; do not tune |
| F2 | Trades + NBBO: true OFI, effective spread, microprice, queue/cancel imbalance | audit gate built; sample absent |
| F3 | Broad universe plus pre-screened lead/lag and sector/peer propagation | data blocked |
| F4 | Options and auction imbalance | data blocked |
| F5 | Public events/actor context as modifier only | later, only after F2/F3 lift |

F2 requires historical trade and NBBO quote data with immutable timestamps.
Several levels of depth, auction imbalance and options positioning are optional
increments, not substitutes for NBBO. Hawkes, lead-lag, HMM and supervised
continuation models are prohibited until F2 has a positive incremental result
over F1 on identical candidate timestamps. F2's future label is cost-free
mid-to-mid factor residual return; costs enter only after feature comparison, in
the portfolio simulation. Execution must use the first quote after the completed
decision interval, never a quote that formed the feature.

## Promotion gate

Use purged walk-forward splits, next-bar execution, historical spread/slippage,
date/ticker clustering, delayed-entry and doubled-cost controls, shuffled-time
and shuffled-ticker controls, and leave-one-sector-out validation. Promotion
requires at least 100 independent days and 500 executable episodes, positive
post-cost return, validation Sharpe above 1, positive delayed-entry and
doubled-cost checks, and no concentration in a ticker, sector, or volatile week.
F1 does not meet any of those scale or data-quality conditions.
