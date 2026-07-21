# Crypto Perp Track — Frozen Spec (BTC/ETH only)

**Why this track exists:** it removes the gate that blocked the equity intraday
engine. BTC/ETH perp OHLCV + funding + order books are **free** (Binance USD-M
public API), 24/7, no PDT, symmetric shorts. Restricting to **BTC/ETH majors**
sidesteps crypto's worst backtest traps (survivorship from thousands of dead
tokens; wash-traded volume on thin pairs). Same discipline as every other lane —
the asset class does not lower the bar.

**Honest prior:** the first experiment (crypto TSMOM) is a *replication* of a
well-documented effect, not novel alpha; and most short-horizon crypto ideas die
on costs, just like the equity ones. This is a disciplined test, not a promise.

## Data (ingested, real)

`src/quantv1/ingest/crypto_perp.py` → `data/crypto/` (gitignored, regenerable):

| Symbol | Daily klines | Funding records | Range |
|---|---|---|---|
| BTCUSDT | 2,509 | 7,518 | 2019-09 → 2026-07 |
| ETHUSDT | 2,429 | 7,284 | 2019-11 → 2026-07 |

**Funding is the load-bearing cost term.** Measured average ≈ **3.2 bps/day (BTC),
3.8 bps/day (ETH)** — persistently *positive*, so a long perp pays ~12–14%/yr in
funding. Any backtest that ignores funding is fiction. Modeled explicitly.

## First experiment — TSMOM port (the one survivor)

Port the equity TSMOM overlay (the only thing that cleared honest costs) to
BTC/ETH perps. Frozen rules mirror `scripts/tsmom_etf_diag.py`:
- Signal: mean sign of cumulative return over {but crypto-appropriate} lookbacks.
- Long/short, **vol-targeted** (inverse trailing realized vol, capped).
- Costs: taker fees (~4–5 bps/side) **+ funding paid/received over the hold** +
  slippage. All three, or the result is meaningless.
- Report gross vs net separately.

**Mandatory sub-period decay analysis** (crypto has ~3 near-distinct regimes, a
much lower-information sample than the 2012–2026 ETF work): 2019–21 (bull),
2022 (bear), 2023–24, 2025–26. A single-period Sharpe is not admissible; the
verdict must hold across regimes or it is rejected.

## Later (only if TSMOM clears)

- **Funding-rate carry** — the most structural crypto edge (paid to be
  short-perp/long-spot when funding is persistently positive). Closer to a risk
  premium than alpha; unwinds violently in tails.
- **Liquidation-cascade fade** — forced deleveraging is non-informational selling
  with mechanical reversion; the crypto cousin of the (rejected) forced-flow
  thesis, but the flow is genuinely non-discretionary.
- **Order-book imbalance / OFI** — the literal intraday engine, on free L2 data,
  via the existing `fill_sim` (which must **walk the book**, not assume midpoint).

## Hazards (crypto-specific, worse than equities)

- **Survivorship** — mitigated by BTC/ETH only; NEVER build a cross-sectional alt
  result from a current-universe snapshot.
- **Wash trading** — reported volume is partly fabricated; trust volume only on
  majors/major venues.
- **Regime-fractured sample** — effective independent regimes ≈ 3; power is low →
  wide CIs, honest MDEs.
- **Thin books** — depth outside the top pairs is far less than the top quote; the
  fill simulator must walk the book.
- **Counterparty / custody / regulatory** — real return terms; **paper only** here.
- **24/7 ops tax** and **tax accounting** (many disposals) — operational, not alpha.

## Pre-registered gate (write it down before results)

A crypto signal advances only if, **net of taker fees + funding + slippage**:
net Sharpe > 1, **positive in every sub-period regime** (no single-year/regime
dominance), bootstrap lower bound > 0, and it clears **Deflated Sharpe** against
the global trial ledger (this is trial #N, not #1). Paper-only until it survives
a genuine forward record. Fail → archive, no rescue filters.

## Build order (same discipline)

data ingest (done) → book-walking fill/cost model incl. funding → TSMOM port
backtest (walk-forward + sub-period) → verdict → only then funding-carry /
liquidation-fade / OFI. Do not add ideas before one reaches a verdict.
