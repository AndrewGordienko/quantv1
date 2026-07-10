"""G-layer experiments: government contracts, alone and with prior political trades.

Two questions, both leak-free (next-open entry, Carhart factor-adjusted CAR,
cluster-robust bootstrap by ticker, locked 2024+ holdout):

  1. STANDALONE — do large federal contract awards predict abnormal returns for
     the recipient after the award date?
  2. INTERACTION — do congressional purchases that are FOLLOWED by a contract
     award to the same company (within WINDOW days) outperform congress buys that
     are not? i.e. did the political trade anticipate government money?

Reuses the factor-model CAR machinery from event_study_v2 so the methodology is
identical across layers.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..config import DATA_DIR
from ..db import connect
from .event_study_v2 import (_load, _fit_betas, _car_ff, _cluster_bootstrap,
                             EST_WINDOW, EST_GAP, FACTOR_COLS, HOLDOUT_START)

HORIZONS = [21, 63]
WINDOW = 90     # days after a congress buy to look for a contract award


def _car_for(op, cl, ret, fac, ticker, when, horizons) -> dict:
    """Factor-adjusted CAR entering the next open after `when`, per horizon."""
    out = {f"car_{h}": np.nan for h in horizons}
    if ticker not in cl.columns:
        return out
    cal = cl.index
    i0 = cal.searchsorted(pd.Timestamp(when), side="right")   # next session
    col = cl.columns.get_loc(ticker)
    op_col, cl_col, ret_col = op.iloc[:, col].to_numpy(), cl.iloc[:, col].to_numpy(), ret.iloc[:, col].to_numpy()
    e = next((j for j in range(i0, len(cal)) if j < len(op_col) and np.isfinite(op_col[j])), None)
    if e is None:
        return out
    lo, hi = max(0, e - EST_GAP - EST_WINDOW), max(0, e - EST_GAP)
    rf = fac["rf"].to_numpy()
    fac_arr = fac[FACTOR_COLS].to_numpy()
    betas = _fit_betas(ret_col[lo:hi] - rf[lo:hi], fac_arr[lo:hi])
    for h in horizons:
        out[f"car_{h}"] = _car_ff(op_col, cl_col, ret_col, rf, fac_arr, betas, e, h)
    return out


def run(verbose: bool = True) -> dict:
    con = connect(read_only=True)
    op, cl, ret, fac = _load(con)
    # source_time = first_seen_at (conservative public availability), the honest
    # entry point — NOT the raw contract action_date.
    gov = con.execute("""
        SELECT ticker, source_time AS effective_date, magnitude, payload FROM events
        WHERE layer='G' AND event_type='gov_contract'
    """).df()
    tables = [t[0] for t in con.execute("SHOW TABLES").fetchall()]
    ev = (con.execute("SELECT trade_id, member_key, ticker, filing_date, amount_mid, "
                      "car_ff_63 AS car FROM event_v2 WHERE car_ff_63 IS NOT NULL").df()
          if "event_v2" in tables else pd.DataFrame())
    con.close()

    out = {"n_gov_events": int(len(gov)), "horizons": HORIZONS, "window_days": WINDOW}

    # --- 1. standalone contract event study --------------------------------
    if len(gov):
        gov["effective_date"] = pd.to_datetime(gov["effective_date"])
        recs = []
        for r in gov.itertuples(index=False):
            c = _car_for(op, cl, ret, fac, r.ticker, r.effective_date, HORIZONS)
            recs.append({"ticker": r.ticker, "date": r.effective_date, **c})
        gdf = pd.DataFrame(recs)
        gdf["year"] = gdf["date"].dt.year
        train, hold = gdf[gdf["date"] < HOLDOUT_START], gdf[gdf["date"] >= HOLDOUT_START]
        out["standalone"] = {
            f"car_{h}": {"train": _cluster_bootstrap(train, f"car_{h}", "ticker"),
                         "holdout": _cluster_bootstrap(hold, f"car_{h}", "ticker")}
            for h in HORIZONS}

    # --- 2. interaction: congress buy -> subsequent contract ----------------
    if len(ev) and len(gov):
        gov_by_tkr: dict[str, list] = {}
        for r in gov.itertuples(index=False):
            gov_by_tkr.setdefault(r.ticker, []).append(pd.Timestamp(r.effective_date))
        for tk in gov_by_tkr:
            gov_by_tkr[tk].sort()
        import bisect

        ev["filing_date"] = pd.to_datetime(ev["filing_date"])

        def has_contract_after(tk, when):
            lst = gov_by_tkr.get(tk)
            if not lst:
                return False
            i = bisect.bisect_right(lst, when)
            return i < len(lst) and lst[i] <= when + pd.Timedelta(days=WINDOW)

        ev["followed_by_contract"] = [
            has_contract_after(r.ticker, r.filing_date) for r in ev.itertuples(index=False)]
        ev["is_large"] = ev["amount_mid"] >= 50_000
        train = ev[ev["filing_date"] < HOLDOUT_START]

        def cmp(df, label):
            f = df[df["followed_by_contract"]]
            n = df[~df["followed_by_contract"]]
            return {"label": label,
                    "followed": _cluster_bootstrap(f, "car", "ticker"),
                    "not_followed": _cluster_bootstrap(n, "car", "ticker")}

        out["interaction"] = {
            "all": cmp(train, "all congress buys (train)"),
            "large": cmp(train[train["is_large"]], "large congress buys (train)"),
            "n_followed": int(ev["followed_by_contract"].sum()),
            "n_total": int(len(ev))}

    with open(DATA_DIR / "gov_experiment.json", "w") as f:
        json.dump(out, f, indent=2)
    if verbose:
        _print(out)
    return out


def _print(out):
    print(f"=== G-layer: {out['n_gov_events']} contract events ===")
    sa = out.get("standalone", {})
    print("\nStandalone contract event study (factor-adj CAR, cluster-robust):")
    for h in HORIZONS:
        t = sa.get(f"car_{h}", {}).get("train", {})
        ho = sa.get(f"car_{h}", {}).get("holdout", {})
        if t.get("mean") is not None:
            sig = "*" if (t["ci_low"] > 0 or t["ci_high"] < 0) else " "
            print(f"  {h}d train={t['mean']*100:+.2f}%{sig} "
                  f"[{t['ci_low']*100:+.2f},{t['ci_high']*100:+.2f}] n={t['n']}  "
                  f"holdout={ho.get('mean',0)*100:+.2f}% n={ho.get('n',0)}")
    it = out.get("interaction")
    if it:
        print(f"\nInteraction: congress buy followed by contract within {out['window_days']}d "
              f"({it['n_followed']}/{it['n_total']}):")
        for key in ("all", "large"):
            r = it[key]
            f, n = r["followed"], r["not_followed"]
            if f.get("mean") is not None and n.get("mean") is not None:
                print(f"  {r['label']:30s} followed={f['mean']*100:+.2f}% n={f['n']}  "
                      f"vs not={n['mean']*100:+.2f}% n={n['n']}  "
                      f"lift={(f['mean']-n['mean'])*100:+.2f}pp")


if __name__ == "__main__":
    run()
