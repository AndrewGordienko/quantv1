# Strategy Scoreboard

> **Consolidated state & resourcing fork:** `docs/research_state_2026-07-20.md`
> (one real signal = sub-gate TSMOM overlay; all else null/blocked; decision fork
> a/b/c/d).

The single table that matters. A candidate becomes *the* strategy only by
filling this row with real, post-cost, walk-forward numbers and clearing the
kill gates. Everything else is lab-building. Updated as verdicts land.

## The table

| Strategy | Indep. validation trades | Net return | Net Sharpe | Delayed-entry | 2× costs | Decision |
|---|---|---|---|---|---|---|
| **Forced flow** (announcement→effective continuation) | 61 executable batches (candidate test) | gross +126 bps / **net +59 bps (CI incl. 0)** | — | +64 bps (CI incl. 0) | **−7 bps (fails)** | **REJECTED — leg closed.** Gross index effect exists but net not sig after hedged costs; fails doubled-cost; placebo +30 bps (survivorship). No rescue. |
| **MGRM** (guidance underreaction) | — | — | — | — | — | **TEST ONCE** — blocked on extraction cert |
| **Actor B3** (Fed speaker deviation) | — | — | — | — | — | **SIDE EXPERIMENT (≤10%)** — census v1 frozen |
| **Latent flow shock F1** (bars only) | 38 holdout episodes | −18.2 bps / 30m | — | Negative | Negative | **REJECTED** — F2 trades/NBBO separately gated |
| **SEC Event Atlas** (unsigned stage) | 2,386 tags / 494 accessions | diagnostic only | — | — | — | **PRIMARY DISCOVERY LANE** — 80-label queue and PIT security-master/price coverage gates pending |
| **Opening Flow P3** (prospective canary) | — | — | — | — | — | **SHADOW / PAPER ONLY** — live evidence not yet accumulated |
| **Crypto TSMOM** (BTC/ETH perps, daily) | 2359 days (backtest) | +22%/yr @21% vol | **1.07** net ✓ | **1.01** @2d ✓ | **1.00** @2× ✓ | **CANDIDATE — first to clear the full battery; cross-regime avg 0.95, replication not novel, daily trend NOT day-trade; needs Deflated Sharpe + forward paper. Paper-only** |
| **Diversified TSMOM** (ETF proxies) | diagnostic + **PAPER-FORWARD ARMED** | Sharpe-scaled | **0.66** @2bps | **0.59** @6d ✓ | **0.63** @5bps ✓ | **ROBUST MODEST DIVERSIFIER — below >1 gate; low SPY corr; overlay not standalone. Frozen paper-forward armed 2026-07-20, live from 2026-07-21** (`scripts/tsmom_paper_forward.py`, `goldset/tsmom_paper/`) |

## Kill gates (a candidate advances only if ALL hold)

- ≥ 100 genuinely independent, executable events
- Positive net return after realistic costs (spread, impact, borrow, fees)
- Validation Sharpe > 1
- Positive delayed-entry result and positive doubled-cost result
- Bootstrap and HAC lower bounds above zero
- No dominant single ticker, event, or actor
- Positive paper-forward performance

If a candidate fails, **close the hypothesis** — no rescue via new feature slices.

## Effort allocation (this stage)

- **60%** SEC Event Atlas extraction and unsigned validation
- **25%** forced-flow announcement timestamps and continuation test
- **15%** Opening Flow operational paper canary

## Grounded status (2026-07-11)

### Forced flow — flagship, data-viable
- **Mechanism:** index add/delete/reweight forces passive funds to trade
  irrespective of price; we trade the *unabsorbed* portion.
- **Data reality (checked):** daily `prices` = **2,547 tickers, 2012→2026**.
  Index additions hit mid-caps we DON'T have minute bars for, but DO have daily
  bars for — and the trade is a multi-day announce→effective move, so daily
  resolution is correct. This is the one hypothesis whose event universe our
  data can actually cover at the ≥100-event scale.
- **Gap:** `forced_flow_events` table exists but is empty; no ingest module. The
  events (index-change announcements w/ announcement_time + effective_date) must
  be sourced. Zero-vendor path exists (S&P change lists + official releases).
- **Four sub-tests to run separately:** pre-effective continuation, closing-
  auction pressure, post-effective reversal, related-company diffusion.
- **Announcement census progress (2026-07-20):** a reusable, return-blind,
  Tier-1/2-compliant resolver now exists (`scripts/forced_flow_resolve.py`):
  raw-fetch the S&P DJI / PR Newswire release → parse machine-readable
  `datePublished` (exact minute) → hash source bytes → validate exchange-qualified
  ticker + "S&P 500". **CENSUS FROZEN at 75/113 verified genuine-addition batches**
  (`goldset/forced_flow/census_freeze_v1.json`, manifest sha256 `152622d8…`; 20
  quarterly-rebalance, 55 ad-hoc; 33 renames/mergers excluded). Claim: **`full
  candidate test` — reject-only; promotion still needs ≥100 executable events**.
  The announcement→effective continuation test is **PREREGISTERED, return-blind**
  in `docs/forced_flow_continuation_test_spec.md`; it runs **exactly once** and
  can only kill or shortlist the leg, never promote it.
  (Tier 2, after-hours ~18:00 ET.) Window differs by type: **ad-hoc
  adds ~5–8 days** pre-effective vs **quarterly-rebalance adds ~2–3 weeks** (report
  separately per spec); outliers NOW intraday, TSLA 35-day window →
  `goldset/forced_flow/announcement_manifest_v1.jsonl` +
  `announcement_coverage_v1.json`. Slow grind (45 remain, ~7 real adds short of 75);
  **claim `underpowered candidate test`** (≥50); reaches **`full candidate test`** at ≥75
  (reject-only; ≥100 executable events required for promotion). Plan: stop the
  census at ≥75, freeze the manifest/rejection-ledger/rename-exclusions, then
  preregister and run the announcement→effective continuation test **once**
  (batch = independent unit; ad-hoc vs quarterly separated; costs, delayed entry,
  doubled costs, clustered CIs, concentration checks); kill/advance with no rescue
  filters; write the result up power-honestly (n≈75 → MDE ~1.2–1.5%, likely null).
- **Census-quality finding:** **31 of the 99 batches processed so far are renames
  / mergers, not fresh additions** (~31%) — e.g. LHX (Harris→L3Harris), GL
  (Torchmark→Globe Life), BKR (BHGE→Baker Hughes), NLOK, PEAK, J (JEC ticker
  change), AMCR (inherited Bemis's spot), TT (Ingersoll-Rand→Trane), HWM
  (Arconic→Howmet), LUMN (CenturyLink→Lumen), plus merger tickers VIAC/TFC/RTX in
  mixed batches — with no "set to join" release and **no announcement drift**.
  Logged in `announcement_renames_excluded_v1.jsonl`. Extrapolating ~61% real
  adds, the true tradable-addition denominator is likely **~65–70**, i.e. an
  **underpowered-candidate** sample (50–74), not a full test. Prior (disappearing
  index effect) already says the eventual test is likely null — low expected
  value, but now unblocked and mechanically closeable.

### MGRM — test once, then keep or kill
- **Blocker (checked):** `mgrm_extractor_certification.json` = `GOLDSET_TOO_SMALL`,
  provider `none:none`. `mgrm_report.json` = `MGRM_BLOCKED_SAMPLE_POWER`
  (3 usable events; needs ≥200 train / ≥100 val; extraction 0% certified).
- **To reach a verdict:** configure an extractor model + reference labeler,
  label the corpus, certify, then run G0/G1/G2 **once**. Pass → lock + paper.
  Fail → retire. No further feature archaeology.

### Actor B3 — bounded side experiment (demoted from "strategy")
- Now a ≤10% side test answering ONE question: after controlling for what was
  said, does *who said it and how they deviated* add out-of-sample lift (B1 vs
  B2/B3)? If yes, actor features become *modifiers* (credibility, expected vol,
  follow-through, asset mapping) inside the event engine — never a primary
  trigger. If no, stop.
- **State:** outcome writer merged (PR #6); census v1 rules frozen, BOARD_ONLY,
  PR #7 open. **Stop expanding** beyond the frozen Fed pilot. No universal
  actor graph until incremental value is shown on this cohort.

## Data infrastructure gate (2026-07-20)

The daily `prices` panel is **survivorship-biased by construction** and
**corporate-action-incomplete**. Audit: `scripts/pit_panel_audit.py` →
`data/pit_panel_audit.json`; contract + remediation in
`docs/point_in_time_panel_spec.md`. Hard numbers: of 163 S&P 500 deletions since
2019, **84 (51.5%) are absent entirely** and **0 present ones record a terminal
stop**, vs 11.2% of additions missing — a non-sign-neutral survivorship
signature. `close` is fully back-adjusted (yfinance `auto_adjust=True`) with no
raw price stored, so as-of-date price levels are unrecoverable.

**Consequence for the backlog:** daily cross-sectional residual momentum (#2) and
PCA/ETF residual stat-arb (#3) are **DATA-GATED**. They may run only as
survivorship-labelled diagnostics on current constituents — **no promotion, no
scoreboard row, no Deflated-Sharpe claim** — until the panel reports
`PANEL_CONTRACT_OK`. Remediation is the top infrastructure task: PIT identity
backbone (free, partly built in `ingest/security_master.py`), raw+factors for
survivors (free), delisted-inclusive history via the already-integrated Polygon
key (paid tier), delisting returns (CRSP, paid).

**Diagnostic gate run (2026-07-20, backlog #2 simple form).**
`scripts/daily_xs_reversal_diag.py` ran the cheap gross-signal gate *before*
committing panel budget. On the survivorship-optimistic panel (1,080 names avg):
residual momentum has **no gross signal**; short-term reversal has a real gross
edge (+12%/yr, Sharpe 0.57) but turnover ~1.0/day buries it — **all 24 net cells
negative** (10 & 20 bps/side). Verdict `NO_GROSS_SIGNAL_KILL_CANDIDATE`.
**Implication:** do **not** build the paid panel on behalf of #2's naive daily
L/S form; the eventual clean panel is justified (if at all) by the lower-turnover
**event** lane, not by cross-sectional reversal. Priority shifts further toward
the SEC Atlas / event lane.

**SEC Atlas signed-drift diagnostic (2026-07-20, backlog #1).**
`scripts/atlas_signed_drift_diag.py` froze a return-blind structural-sign map and
measured signed post-open drift on the 490 signed, current-linked discovery
events. **No family has significant tradeable drift** (all 5-day CIs cross zero),
and the immediate signed gap is **wrong-signed** for the largest positive
auto-tag families (`guidance_raised` −65 bps, `major_customer_win` −54 bps,
`activist_13d` −84 bps). **Two consequences:** (a) the directive's crux is
confirmed — the Atlas is unsigned in practice; (b) **human directional annotation
is a HARD gate**, not a formality — auto-tags cannot carry the trade sign. The
35% Atlas allocation should be read as *annotation-first*, and the signed test is
worth running (once) only on the correctly-signed **negative-news** families
(restatement, secondary_offering) after annotation + PIT prices land.

## Intraday engine track (2026-07-21) — SCOPED, DATA-GATED

Separate event-driven day-trade engine (NOT TSMOM sped up). Frozen spec:
`docs/strategy_intraday.md` (event-absorption, 5–60 min holds, ≤3 positions, flat
overnight, NO_TRADE by default, trade only if net edge > 2× cost). Load-bearing
gap = **NBBO quotes + tick trades**: contract skeleton in `ingest/nbbo.py`
(`quotes_nbbo`/`trades_tick`, `known_at` discipline, quote-rule aggressor
classifier + tests). **No intraday result may be computed until a paid Polygon
quotes/trades tier + a calibrated fill simulator exist** — bars are not order
flow. Build order: NBBO ingest → replay==live → fill simulator (before any
signal) → elastic-net baseline → validation gates (≥100 event clusters,
day-clustered SEs, Sharpe>1, 2× cost, +latency, beat matched non-event shocks,
Deflated Sharpe) → paper-forward.

## Crypto perp track (2026-07-21) — venue pivot, data UNBLOCKED

Crypto removes the paid-data gate: BTC/ETH perp OHLCV + funding + books are FREE
(Binance USD-M), 24/7, no PDT, symmetric shorts; BTC/ETH-only sidesteps
survivorship + wash-trading. Frozen spec: `docs/strategy_crypto.md`. Real data
ingested (`ingest/crypto_perp.py` → `data/crypto/`): BTC 2,509 daily bars +
7,518 funding recs, ETH 2,429 + 7,284, 2019→2026. **Funding measured ≈ 3.2–3.8
bps/day (a long perp pays ~12–14%/yr)** — must be modeled. First experiment =
**TSMOM port** (the one survivor) with taker fees + funding + slippage,
walk-forward + **mandatory sub-period decay** (≈3 regimes). Pre-registered gate:
net Sharpe>1 in EVERY regime, bootstrap LB>0, Deflated Sharpe, paper-only.
Replication, not novel alpha; honest prior is most variants die on costs.

## Crypto day-trade (OFI flow) — first real experiment REJECTED (2026-07-21)

First genuine intraday day-trade test on real data (89 days BTC+ETH 5-min signed
OFI from free Binance aggTrades). `scripts/crypto_ofi_experiment.py`: elastic-net,
NO_TRADE default, walk-forward, day-clustered Sharpe. **Result: no edge.** OFI→
forward-return corr ≈ 0; elastic-net zeroes OFI; predicted edges <3 bps vs 32 bps
gate → 0 executions; always-on gross day-Sharpe NEGATIVE. The accessible
(trade-level, 5-min) order-flow signal is dead on BTC/ETH perps. The remaining
microstructure avenue (L2 depth/microprice/queue at seconds resolution) needs a
live order-book collector (weeks) and is a separate, harder build — not a rescue
of this. Trial `crypto_ofi_flow_v1` (registry 11). No rescue filters.

## Multiple-testing check (2026-07-21) — Probabilistic Sharpe

Against the 11-trial global search, neither TSMOM candidate is confidently a >1
Sharpe strategy; both are real-but-modest:
- **Crypto TSMOM** (daily net, n=2359): PSR(>0)=0.998, **PSR(>0.5)=0.934**, PSR(>1)=0.572.
- **Equity TSMOM** (daily net, n=3337): PSR(>0)=0.990, PSR(>0.5)=0.705, **PSR(>1)=0.105**.
Crypto TSMOM is the stronger candidate (confidently >0.5); the point-estimate
Sharpes (1.07 / 0.66) overstate the confident lower bound. Both remain paper-only
candidates pending a real forward record — the "1.07" is not a promotable number.

## Standing discipline

- A buy/sell display is a UI demo until a frozen walk-forward model produces the
  numbers above. No real execution because the plumbing works.
- Trade frequency is not performance. Watching all day ≠ trading all day.
- Personality is never the primary trigger; it modifies an event signal or it is
  dropped.
