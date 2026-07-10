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
| Structured earnings mismatch (M1/M2) | Blocked, not rejected | Point-in-time EPS/revenue coverage is 0%; dispersion, revisions, actuals, guidance, size metadata, and executable quotes are incomplete. The final holdout remains sealed. |
| CatBoost, neural nets, world-model teacher | Prohibited | Complexity cannot advance until M2 elastic net demonstrates stable net signal. World-model work remains a later shadow optimizer only. |
| Forced flows from index/ETF/corporate actions | Next independent hypothesis | Not yet tested. Advance here if properly measured M2 fails validation; use required shares divided by ADV as the core pressure measure. |

## Update rule

Add a row when a material hypothesis is promoted, rejected, invalidated, or
blocked. Include the untouched sample, costs, executable-trade count, and code or
experiment identifier. Never change a rejection to “untested” by renaming its
features.
