# Earnings expectation-reaction mismatch

Frozen 2026-07-10. Do not tune this specification against the sealed holdout.

> **Status:** EERM M1/M2 are `BLOCKED_DATA_ECONOMICALLY_INACCESSIBLE` — they need
> paid point-in-time analyst consensus. This protocol is preserved verbatim and
> not reused. The active zero-vendor reuse of the reaction engine is MGRM (see
> the appendix below); its extraction is currently **uncertified** and no
> historical alpha test has been run on either track.

## Event and decision

1. Establish the earliest verified public earnings-release time from company IR
   or a primary wire. SEC acceptance is a conservative fallback, not proof of
   earliest publication.
2. Find the first liquid regular session after publication.
3. Wait 30 minutes. Use only information whose `known_at` is no later than the
   decision timestamp.
4. Enter at the next executable quote. The bar screen is a coarse research stage
   using 15 basis points per side; it cannot promote without 95% quote coverage.
5. Hedge market/sector beta and exit at the fifth subsequent common trading-day
   close.

The 60-minute entry and 30-basis-point-per-side cases are mandatory robustness
checks.

## Features and target

Fundamental information includes point-in-time EPS and revenue surprise, new
guidance versus prior guidance and pre-release consensus, analyst dispersion,
revision breadth, and analyst counts.

The observed reaction is the 30-minute stock return less the beta-adjusted sector
return, standardized by pre-event volatility. Abnormal 30-minute volume and
liquidity are retained separately.

```text
fundamental surprise = mean(directional standardized surprises, revision breadth)
mismatch              = fundamental surprise - residual reaction score
diagnostic residual   = five-day stock return - sector ETF return
```

The canonical target is the executable mid-to-mid beta hedge:

```text
target = stock return(t -> t+5d) - frozen beta(t-) * sector ETF return(t -> t+5d)
```

Beta is estimated solely from pre-event daily returns, shrunk toward one,
clipped to `[0, 2]`, frozen at the decision, and stored in the feature artifact
with its observation count and estimation end. The same sector ETF and beta are
used in the label, prediction metrics, hedge, and portfolio accounting. Costs
remain outside the label.

Training-only 1%/99% winsorization, imputation, and standardization are mandatory.

## Model ladder

- M0: price, reaction, volume, volatility, liquidity, session, and sector.
- M1: M0 plus structured earnings information.
- M2: M1 plus beta-residual reaction, fundamental composite, and mismatch.

The ladder is strictly nested. M1/M2 remain unavailable unless point-in-time EPS
and revenue coverage is at least 80% in training and validation and representative
across eligible years, sectors, and company-size buckets. Elastic net is the only
permitted model family until M2 passes.

## Trading and risk

A prediction becomes a trade only when:

```text
abs(expected residual return) > 2 * estimated all-in round-trip cost
```

Long positive tails and short negative tails; otherwise no trade.

- At most five positions.
- 5% NAV per stock.
- 25% maximum gross.
- 15% maximum sector gross.
- 15% maximum net.
- No leverage and no averaging down.

Admission checks use the current marked book, not original weights. Before each
candidate entry, expired positions settle and surviving stock/hedge quantities
are marked with the latest close strictly available before that decision. The
new position is sized from marked NAV and rejected if projected gross, net, or
sector exposure breaches a limit.

Portfolio statistics come from signed stock/hedge quantities in a daily ledger.
NAV, returns, drawdown, gross/net/sector exposure, and stock/hedge turnover are
marked every common session rather than booked only on exit dates.

The hurdle is event-specific when eligible NBBO data exists: observed stock and
hedge spreads, participation relative to ADV, liquidity impact, an adverse-
selection buffer, and historical borrow enter the estimate. A short is
non-deployable if historical borrow availability or fee is unknown. Bar-only
development results retain the conservative 15-basis-point-per-side assumption.

## Validation and holdouts

Validation compares M0 versus M1 and M1 versus M2 using RMSE, MAE, Spearman IC,
net portfolio return, delayed entry, doubled costs, year/sector stability, and
company/event/quarter concentration. Deterministic block-feature and timestamp
permutation controls are recorded for every fitted model. Null permutations keep
ticker/event blocks atomic and move them only within the same year and sector.

The complete XNYS calendar, including zero-return cash sessions, feeds portfolio
statistics. Five-day overlap uses HAC/Newey-West inference with lag five.
Uncertainty also uses a two-way announcement-session/ticker cluster bootstrap
for mean net trade return, total return, annualized alpha, Sharpe, and M2-minus-M1
loss lift. The required trade, ticker, announcement-date, per-year, and effective
sample sizes are computed from the frozen 1% economic effect size and observed
training-target volatility before M2 is run. At least 50% of validation rows must
actually change in permutation controls.

Feature artifacts are explicit: `--coarse` is the deterministic 25% development
sample and can never promote; `--full` is the complete frozen sample required for
promotion.

Three dates have distinct meanings:

- retrospective holdout starts 2025-07-01 and may be opened once after validation;
- the research protocol was frozen 2026-07-10;
- the genuine prospective record begins on the first XNYS session after the
  final M2 feature set, coefficients, hyperparameters, cost model, and trading
  rules are locked.

The sealed retrospective holdout may be opened once, only after M2 has all of:

- net Sharpe above 1;
- deflated-Sharpe probability above 0.95;
- positive doubled-cost and delayed-entry returns;
- stable years and sectors;
- no positive-P&L concentration above 25%; and
- executable quote coverage of at least 95%.

It is a retrospective check, never described as prospective. The model-spec lock
records the immutable specification and prospective start date.

If properly measured M2 fails, retire it. The next independent strategy is forced
flow: index reconstitutions, ETF demand, corporate actions, and required shares
divided by ADV. Behavior is an M3 incremental overlay; political context is a
slow prior; world-model teachers remain shadow research only.

## Appendix: MGRM (Management Guidance Revision-Reaction Mismatch)

MGRM reuses this reaction engine — the same earliest-verified public time,
next-executable NBBO entry, pre-event frozen beta, five-day beta-hedged target,
cost model, XNYS ledger, cluster bootstrap, HAC, deflated Sharpe, and sealed
holdout — but replaces analyst consensus with management's own forward guidance
extracted from public SEC 8-K Item 2.02/7.01 exhibits.

- **Signal.** `mismatch = normalized guidance revision − standardized residual
  reaction`. The revision is the numeric midpoint change relative to management's
  own prior guidance for the same fiscal period; it is left missing when no
  numeric prior exists. Non-numeric raised/lowered/reaffirmed/withdrawn language
  is never converted to a fabricated magnitude — it survives as separate action
  counts and a withdrawal indicator. Normalization uses a frozen hierarchical
  backoff (company → sector → metric, ≥5 prior observations) or stays missing.
- **Ladder.** G0 reaction-only (diagnostic), G1 structured guidance, G2 mismatch.
- **Extraction gate.** Deterministic table+prose parsing must AGREE with a
  provider-agnostic LLM (OpenAI-compatible or local Ollama; provider/model
  recorded). Agreement is only a precision filter; the scientific gate is a
  frozen gold-set audit measuring field-level accuracy on the **reconciled AGREED
  output** that enters the model. Certification records the gold-set SHA-256, code
  hash, extractor version, provider/model, thresholds, and result.
- **Fail-closed.** Without a valid certification (absent, stale, wrong
  provider/model, or below threshold), G1/G2 feature promotion, fitting, model
  locking, and holdout opening all refuse to run. G0 stays available.
- **Linkage.** Previous-guidance links reject UNKNOWN periods, require a strictly
  earlier public timestamp and a different earnings event and accession, and keep
  one canonical record per event/ticker/metric/period so two documents from the
  same filing can never form a false revision.
- **Power.** `evaluate_power` on the actual eligible feature rows: executable
  events, unique tickers and announcement dates, events per year, effective
  sample size (executable trades / design effect), ≥95% executable quote
  coverage, and long/short deployability.

No CatBoost, behavior embeddings, world models, or unofficial transcripts. The
full-universe acquisition is not run until the extractor is certified against an
expanded, human-labelled gold set.
