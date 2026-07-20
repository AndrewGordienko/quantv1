# Point-in-Time Daily Panel — Data Contract and Gap Audit

**Frozen data contract for every cross-sectional / event daily strategy.**
Audit: `scripts/pit_panel_audit.py` → `data/pit_panel_audit.json` (read-only,
re-runnable, return-blind). Verdict as of **2026-07-20**:
`PANEL_CONTRACT_INCOMPLETE`.

This is the highest-ranked infrastructure blocker. Backlog items **#2 daily
cross-sectional residual momentum** and **#3 PCA/ETF residual stat-arb** are
*data-gated behind this contract*: a full-universe result on the current panel
could be a survivorship or corporate-action artifact, not alpha. Diagnostics are
permitted; **promotion is not**.

## The contract (all required before any promotion)

| Requirement | Meaning | Present today |
|---|---|---|
| `raw_ohlcv` | unadjusted as-of-date price levels | ❌ only back-adjusted `close` |
| `adjusted_ohlcv` | split/div-adjusted, stored *alongside* raw | ❌ (adjusted only, no raw) |
| `split_dividend_factors` | per-date cumulative factor to reconstruct either | ❌ absent |
| `listing_intervals` | first/last tradable date per security | ❌ absent |
| `delisting_evidence` | terminal date + reason (merger/bankruptcy/…) | ❌ absent |
| `delisting_return` | final-period return incl. terminal consideration | ❌ absent |
| `terminal_status` | ACTIVE / DELISTED / RENAMED | ⚠️ partial (`security_master`) |
| `filing_era_symbol` | point-in-time ticker↔CIK↔security_id | ⚠️ partial (`security_master`) |
| `pit_sectors` / `pit_market_cap` | as-of GICS + market cap, time-valid | ❌ single current snapshot |

## Measured gaps (real numbers, 2026-07-20)

- **`prices`**: 2,547 tickers, 2012-01-02 → 2026-07-09, 8,050,248 rows. A single
  `close` that `ingest/prices.py` builds with yfinance `auto_adjust=True` —
  **fully back-adjusted, no raw stored**. Back-adjustment is retroactive: a 2015
  row's close reflects *all later* splits/dividends, so it is **not** the level
  observable at that date. As-of price filters, lot sizing, and split detection
  are therefore impossible from this panel.
- **Survivorship is by construction, not by staleness.** Only 34/2,547 tickers
  (1.3%) go stale >1y — a *low* number that is misleading. The source comment is
  explicit: *"Delisted or renamed tickers simply return nothing from yfinance."*
  So delisted names are **absent**, not present-and-stale.
- **S&P 500 change cross-check** (`goldset/forced_flow/sp500_changes_since_2019.csv`),
  the cleanest available survivorship probe:

  | Population | Distinct | Missing entirely | % missing | Present that terminate early |
  |---|---|---|---|---|
  | **Deletions** (2019+) | 163 | **84** | **51.5%** | **0** |
  | **Additions** (2019+) | 161 | 18 | 11.2% | 0 |

  Half of all S&P 500 **deletions** — large-caps guaranteed to have 2012–2026
  history (ATVI, CELG, AGN, ALXN, ABC, ANTM, CERN, BHGE, CTXS…) — are gone. **Zero**
  present deletions record a terminal stop. The 51.5% vs 11.2% add/delete
  asymmetry means the bias is **not sign-neutral**: the panel disproportionately
  drops names that left the index (many via forced exit / distress), inflating
  any long-side momentum or reversal read.
- **`ticker_sectors`**: keyed by ticker only, one `market_cap`, no `as_of` /
  `valid_from` — a current snapshot, **not** point-in-time.

## Remediation path (return-blind, honest free vs. paid)

Ordered by leverage. Nothing here reads returns; it fixes the denominator first.

1. **PIT identity backbone — FREE, partially built.** `ingest/security_master.py`
   already emits `valid_from`/`valid_to` security intervals with explicit
   `delisted_at` from SEC filings (376/500 cached filings mapped). Continue:
   persist a `security_master` table (ticker↔CIK↔security_id, listing intervals,
   `formerNames` from the SEC submissions API) so historical symbols (BHGE→BKR,
   ANTM→ELV, ABC→COR, SYMC→NLOK→GEN) stop silently breaking identity.
2. **Raw + factors for survivors — FREE, do now.** Store `raw` OHLCV alongside
   `adjusted`, plus per-date split/dividend factors (Polygon `v3/reference/splits`
   + `v3/reference/dividends`, or yfinance `.actions`). Fixes point-in-time price
   levels for names that still trade.
3. **Delisted-inclusive history + delisting evidence — PAID (recommended vendor
   already integrated).** The true blocker. **Polygon** (`POLYGON_API_KEY`,
   already wired in `ingest/polygon_data.py`, Canada-friendly) retains delisted
   tickers: unadjusted daily aggs (`adjusted=false`), `v3/reference/tickers?active=false`
   with `delisted_utc`, and corporate actions. Requires a **paid tier** (free tier
   = 2y history / 5 req/min; insufficient for a 2012–2026 delisted panel).
   Rebuild the daily panel from Polygon rather than yfinance. **Delisting
   *returns*** (terminal cash/merger consideration) remain best-sourced from CRSP
   (WRDS, paid) — Polygon gives the delist date and last price but not always a
   clean terminal-consideration return.
4. **PIT sectors / market cap — MIXED.** GICS history is paid; market-cap history
   is reconstructable free but laborious (SEC XBRL `dei:EntityCommonStockShares
   Outstanding` × adjusted price). Lowest priority until 1–3 land.

## Gate rule

Until requirements 1–3 are met and `pit_panel_audit.json` reports
`PANEL_CONTRACT_OK`:

- Daily cross-sectional momentum / reversal / stat-arb may be run **only** as
  labelled diagnostics on the *current-constituent* subset, with the survivorship
  caveat attached to every number. No promotion, no scoreboard row, no Deflated
  Sharpe claim.
- Every diagnostic must state which tickers were dropped for missing history and
  must not silently restrict to survivors.
