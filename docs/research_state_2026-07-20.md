# Research State & Decision Memo — 2026-07-20

**Bottom line:** after a systematic sweep of every *unblocked* research lane, the
repo contains exactly **one** signal that survives honest scrutiny — a **robust
but modest, sub-gate diversifying TSMOM overlay**. Everything else is null,
cost-trapped, or blocked behind data you must buy or human labels you must fund.
This confirms the standing "no deployable alpha yet." The productive
free/unblocked research is now largely exhausted; the next real progress requires
a **resourcing decision** (below), not another algorithm.

This memo consolidates the 2026-07-20 iteration sweep. Every claim is backed by a
re-runnable, return-blind artifact.

## Lane-by-lane verdict

| Lane | Verdict | Evidence / artifact |
|---|---|---|
| **PIT daily panel** (infra, 40%) | **BLOCKED — survivorship-biased by construction** | 51.5% of S&P 500 deletions absent, 0 delisting terminations, back-adjusted-only close. `scripts/pit_panel_audit.py` → `data/pit_panel_audit.json`; contract in `docs/point_in_time_panel_spec.md` |
| **Daily cross-sectional #2** | **KILL-CANDIDATE (simple form)** | Residual momentum no gross signal; short-term reversal +12%/yr gross but turnover buries it — all 24 net cells negative. `scripts/daily_xs_reversal_diag.py`, trial `daily_xs_resid_momrev_diag_v1` |
| **SEC Atlas signed #1** (35%) | **DISCOVERY NULL + tags directionally unreliable** | No significant post-open drift in any family; `guidance_raised` gaps −65 bps, `major_customer_win` −54 bps, `activist_13d` −84 bps (auto-tags gap the wrong way). Human directional annotation is a HARD gate. `scripts/atlas_signed_drift_diag.py`, trial `atlas_signed_drift_diag_v1` |
| **PCA/ETF stat-arb #3** | Not run — ETF-only form is breadth-limited (11 sectors < ≥100-name gate); stock form gated on the panel | — |
| **Order-flow imbalance #4** | Blocked — needs trades + full NBBO (not owned); bar volume already shown inadequate (Latent Flow F1 rejected, −18.2 bps) | prior `research_ledger.md` |
| **Diversified TSMOM #5** | **✅ ONE REAL SIGNAL — robust modest diversifier, sub-gate** | Net Sharpe 0.49–0.66, cost-robust (0.63 @5 bps), corr to SPY 0.04–0.26; survives delayed entry (0.59 @6d), stable 3 sub-periods, positive 11/14 yrs, not 2022-concentrated, PSR(>0)=0.99 vs deflated-null 0.10. But PSR(>0.5)=0.70 and **<1 standalone gate**. `scripts/tsmom_etf_diag.py`, `scripts/tsmom_etf_stress.py`, trials `tsmom_etf_diag_v1`/`tsmom_etf_stress_v1` |
| **Forced-flow continuation #6** (10%) | Unblocked but slow grind; likely-null prior | Return-blind Tier-1/2 resolver built; **4/113 batches VERIFIED**; 109 remain at ~1 web-search/batch. `scripts/forced_flow_resolve.py` → `goldset/forced_flow/announcement_manifest_v1.jsonl` + `announcement_coverage_v1.json` |
| Politician / news / intraday mean-rev | Dead (prior work) | `docs/research_ledger.md` |

Global trial ledger: `data/experiment_registry.jsonl` (8 rows). Deflated-Sharpe
discipline maintained — TSMOM's 0.66 is judged against all configs tried.

## The one real signal — and what it is NOT

**Diversified TSMOM on 13 liquid ETF proxies** (equity/rates/credit/commodity/RE),
monthly rebalance, vol-targeted, frozen spec. It is:
- **Real:** survives cost-doubling, execution delay, sub-period splits, 2022
  removal, and multiple-testing (PSR(true Sharpe>0)=0.99).
- **Modest & sub-gate:** net Sharpe ~0.5–0.66 < the >1 standalone kill-gate;
  PSR(>0.5)=0.70 so we can't even claim it clears 0.5.
- **Survivorship-immune** (ETF proxies) and **low equity correlation** (0.04–0.26).

**Use:** a *diversifying overlay / sleeve*, not a standalone strategy, and not an
alpha discovery — it is a faithful replication of TSMOM's well-documented
post-2011 decay. Do **not** lever it up and call it alpha; do **not** re-run
variants to push the Sharpe (that only inflates the DSR haircut).

## Why everything else is stuck — three binding constraints

1. **No survivorship-safe / corporate-action-correct price panel.** Blocks daily
   cross-sectional #2/#3 *and* the Atlas event lane's price coverage (39.9%).
   Fix = **paid** (Polygon paid tier for delisted-inclusive daily + factors;
   CRSP for delisting returns). PIT identity backbone is free and ~94% built
   (`security_master.py`).
2. **No human directional annotation for SEC events.** The auto-tags gap the
   wrong way, so signed event tests are invalid. Fix = **human labor** (annotate
   direction+magnitude, starting with the correctly-signed negative-news families:
   restatement, secondary_offering).
3. **No trades + full NBBO.** Blocks genuine order-flow (#4). Fix = **paid**.

## The fork (pick one or more)

| Option | Cost | Expected value | Recommendation |
|---|---|---|---|
| **(a) Grind forced-flow census** to completion | ~36 loop iters of web-sourcing | Low — disappearing-index-effect prior says the test is likely null | Only worth finishing *because* the resolver exists and it closes an open question cleanly; do not prioritize |
| **(b) Fund the survivorship-safe panel** (Polygon paid + CRSP delisting) | \$\$ + build | High — unblocks #2/#3 honest retest and Atlas price coverage; the single highest-leverage spend | **Do this if you intend serious cross-sectional/event research** |
| **(c) Fund human annotation** of Atlas event direction | human hours | Medium — required before any signed event test; but discovery drift was null even biased, so temper expectations | Pair with (b); annotate the negative-news families first |
| **(d) Paper-forward the TSMOM overlay** from 2026-07-21 | ~free | Medium — the only thing ready to record honestly; builds a genuine prospective track | **Do this now** — it's the one lane past discovery |

## Loop status

Further autonomous loop iterations now hit **diminishing returns**: the remaining
unblocked work is the slow, likely-null forced-flow grind (option a). Meaningful
next steps (b)/(c) need a spend/staffing decision; (d) needs the 2026-07-21
prospective-start date. Recommend the loop either (i) continue chipping the
forced-flow census a few batches per iteration, or (ii) pause pending your call
on (b)/(c)/(d).
