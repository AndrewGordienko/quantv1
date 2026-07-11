# Research ledger

Append-only summary of material hypotheses. Superseded plans remain available in
Git history; this file preserves their decisions so rejected work is not revived
under a new name.

| Hypothesis | Status | Evidence and decision |
|---|---|---|
| Copy politicians with the best historical returns | Rejected | Year-to-next-year member skill had pooled Spearman 0.015 (`p=0.83`, `n=206`). Permanent `member_skill` is descriptive, not tradeable. |
| Congressional follower model and headline AUC | Rejected as demonstrated alpha | V1 used impossible filing-close entry, leaked full-history skill/current metadata, and survivorship-prone renormalization. The reported AUC 0.558 is contaminated. Later honest work did not establish net alpha. |
| Moving-average entry gate and tactical stops | Rejected | Removing the 50-day-MA gate improved the tactical sleeve; every tested stop configuration still trailed SPY. Do not restore hard stops as an alpha source. |
| Insider confirmation and generic political context | Rejected as standalone signals | Completed V5 review found no positive signal. Political information is limited to a slow contextual prior. |
| Generic public-news continuation/fade | Archived negative | The full 26-ticker validation rejected the fade. Apparent positive performance was concentrated in AAPL/MSFT and did not survive the broader harness (`6509079`, `69e13c0`). |
| Intraday mean reversion | Rejected | Completed V5 evidence found no positive net signal. Do not reslice the same public-news sample. |
| Powerful-person headline mentions | Invalid proxy | A mention is not a verified actor action. Primary-source CEO/CFO behavior remains untested and may enter only as M3 incremental lift over M2. |
| Earnings price/reaction baseline (M0) | Rejected on current validation | The elastic net produces a constant prediction and zero trades after the cost hurdle. It does not demonstrate a profitable strategy. |
| Structured earnings mismatch (EERM M1/M2) | Blocked — data economically inaccessible | Point-in-time analyst consensus/actuals/guidance-consensus are only sold by vendors or WRDS and cannot be honestly reconstructed for free. Preserved as `BLOCKED_DATA_ECONOMICALLY_INACCESSIBLE`; the frozen protocol is retained and not reused. No substitute or backfilled expectations are admitted. |
| Management guidance revision-reaction mismatch (MGRM) | Active zero-vendor hypothesis; extraction uncertified | Reuses the frozen reaction engine on public SEC 8-K Item 2.02/7.01 guidance only, no analyst consensus. The extractor has not passed a frozen gold-set accuracy audit (measured on the reconciled AGREED output that enters the model), so G1/G2 feature generation, fitting, model locking, and holdout opening all fail closed. G0 remains a reaction-only diagnostic. **No historical alpha test has occurred.** |
| EERM protocol correction | Engineering complete; alpha still untested | Canonical target and hedge now share a pre-event frozen beta; XNYS cash days, marked trade attribution, HAC statistics, announcement-date/ticker bootstrap, powered sample gates (executable trades / design effect, with ticker/date minimums), NBBO-exit ledger reconciliation, full-artifact enforcement, liquidity-aware costs, and fail-closed shorts are mandatory. This work does not constitute strategy evidence. |
| CatBoost, neural nets, world-model teacher | Prohibited | Complexity cannot advance until M2 elastic net demonstrates stable net signal. World-model work remains a later shadow optimizer only. |
| Forced flows from index/ETF/corporate actions | Next independent hypothesis | Not yet tested. Advance here if MGRM fails validation; use required shares divided by ADV as the core pressure measure. |

## Update rule

Add a row when a material hypothesis is promoted, rejected, invalidated, or
blocked. Include the untouched sample, costs, executable-trade count, and code or
experiment identifier. Never change a rejection to “untested” by renaming its
features.
