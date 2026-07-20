"""Forced-flow announcement->effective continuation test — RUN EXACTLY ONCE.

Implements docs/forced_flow_continuation_test_spec.md verbatim against the FROZEN
corpus (goldset/forced_flow/census_freeze_v1.json, manifest sha256 152622d8...).
Aborts if the manifest hash changed. No rules added beyond the preregistration.

Reject-only at n=75 (promotion needs >=100 executable events). Gross event-study
residual is reported SEPARATELY from the after-cost portfolio. Output:
data/forced_flow_continuation_test.json keyed to manifest_sha256.
"""

from __future__ import annotations

import hashlib
import json

import duckdb
import numpy as np
import pandas as pd

from quantv1.config import DB_PATH, DATA_DIR, ROOT

FF = ROOT / "goldset" / "forced_flow"
MANIFEST = FF / "announcement_manifest_v1.jsonl"
FROZEN_SHA = "152622d88239f213"          # prefix pinned at freeze
OUT = DATA_DIR / "forced_flow_continuation_test.json"
COST_BPS = 15.0                          # per side, both legs, entry+exit
BETA_WIN = 60
BETA_MIN = 40                            # min pre-announcement obs to estimate beta
RNG = np.random.default_rng(20260720)


def is_quarterly(eff: str) -> bool:
    m, d = int(eff[5:7]), int(eff[8:10])
    return m in (3, 6, 9, 12) and 15 <= d <= 23


def load_manifest():
    raw = MANIFEST.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if not sha.startswith(FROZEN_SHA):
        raise SystemExit(f"MANIFEST HASH CHANGED ({sha[:16]} != {FROZEN_SHA}); refusing to run.")
    recs = [json.loads(l) for l in raw.decode().splitlines() if l.strip()]
    return recs, sha


def load_prices(tickers):
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = con.execute(
        "SELECT ticker,date,open,close FROM prices WHERE ticker IN (SELECT UNNEST(?)) ORDER BY date",
        [sorted(set(tickers) | {"SPY"})]).df()
    con.close()
    df["date"] = pd.to_datetime(df["date"])
    return {t: g.set_index("date").sort_index() for t, g in df.groupby("ticker")}


def one_event(px, spy, ticker, ann_date, eff_date, entry_offset, cost_bps):
    """Return dict with gross residual + net (after cost) for one added name, or None."""
    stock = px.get(ticker)
    if stock is None:
        return None
    ann, eff = pd.Timestamp(ann_date), pd.Timestamp(eff_date)
    after = stock.index[stock.index > ann]
    if len(after) <= entry_offset:
        return None
    entry = after[entry_offset]
    exits = stock.index[stock.index <= eff]
    if len(exits) == 0:
        return None
    exit_ = exits[-1]
    if exit_ <= entry:
        return None
    pre = stock.index[stock.index <= ann]
    if len(pre) < BETA_MIN:                      # spinoffs / recent listings: non-executable
        return None
    # beta on trailing daily returns ending at announcement
    r_s = stock["close"].pct_change().reindex(pre).dropna().iloc[-BETA_WIN:]
    r_m = spy["close"].pct_change().reindex(pre).dropna().reindex(r_s.index)
    j = pd.concat([r_s, r_m], axis=1).dropna()
    if len(j) < BETA_MIN or j.iloc[:, 1].var() == 0:
        return None
    beta = float(np.clip(j.cov().iloc[0, 1] / j.iloc[:, 1].var(), 0, 3))
    if entry not in spy.index or exit_ not in spy.index:
        return None
    long_ret = float(stock.loc[exit_, "close"] / stock.loc[entry, "open"] - 1)
    mkt_ret = float(spy.loc[exit_, "close"] / spy.loc[entry, "open"] - 1)
    resid = long_ret - beta * mkt_ret
    cost = 2 * cost_bps * (1 + beta) / 1e4       # long+short, entry+exit
    return {"resid": resid, "net": resid - cost, "beta": beta,
            "hold_days": int((exit_ - entry).days)}


def batch_values(recs, px, spy, entry_offset=0, cost_bps=COST_BPS):
    """Batch-level (gross resid, net) equal-weighting executable member names."""
    rows = []
    n_names = n_exec = 0
    for r in recs:
        vals = []
        for t in r["affected_tickers"]:
            n_names += 1
            o = one_event(px, spy, t, r["announcement_public_time"][:10],
                          r["effective_date"], entry_offset, cost_bps)
            if o:
                n_exec += 1
                vals.append(o)
        if vals:
            rows.append({"batch": r["event_batch_id"], "eff": r["effective_date"],
                         "quarterly": is_quarterly(r["effective_date"]),
                         "resid": float(np.mean([v["resid"] for v in vals])),
                         "net": float(np.mean([v["net"] for v in vals]))})
    return rows, n_names, n_exec


def stats(vals):
    v = np.array(vals, float)
    if len(v) < 3:
        return None
    boot = np.array([RNG.choice(v, len(v), replace=True).mean() for _ in range(5000)])
    return {"n": int(len(v)), "mean_bps": round(float(v.mean()) * 1e4, 1),
            "median_bps": round(float(np.median(v)) * 1e4, 1),
            "ci95_bps": [round(float(np.percentile(boot, 2.5)) * 1e4, 1),
                         round(float(np.percentile(boot, 97.5)) * 1e4, 1)],
            "p_gt_0": round(float((boot > 0).mean()), 3)}


def placebo(recs, px, spy):
    """Same names, random earlier window of equal length: the null."""
    rows = []
    for r in recs:
        vals = []
        for t in r["affected_tickers"]:
            stock = px.get(t)
            if stock is None:
                continue
            ann = pd.Timestamp(r["announcement_public_time"][:10])
            eff = pd.Timestamp(r["effective_date"])
            hist = stock.index[stock.index < ann - pd.Timedelta(days=90)]
            if len(hist) < BETA_MIN + 30:
                continue
            hold = max(1, len(stock.index[(stock.index > ann) & (stock.index <= eff)]))
            i = int(RNG.integers(BETA_MIN, len(hist) - hold - 1))
            fake_ann = hist[i]
            fake_eff = hist[min(i + hold, len(hist) - 1)]
            o = one_event(px, spy, t, fake_ann.date().isoformat(),
                          fake_eff.date().isoformat(), 0, COST_BPS)
            if o:
                vals.append(o["resid"])
        if vals:
            rows.append(float(np.mean(vals)))
    return rows


def main():
    recs, sha = load_manifest()
    px = load_prices([t for r in recs for t in r["affected_tickers"]])
    spy = px["SPY"]

    prim, n_names, n_exec = batch_values(recs, px, spy)
    net = [b["net"] for b in prim]
    gross = [b["resid"] for b in prim]
    adhoc = [b["net"] for b in prim if not b["quarterly"]]
    quart = [b["net"] for b in prim if b["quarterly"]]
    delayed = [b["net"] for b in batch_values(recs, px, spy, entry_offset=1)[0]]
    doubled = [b["net"] for b in batch_values(recs, px, spy, cost_bps=2 * COST_BPS)[0]]
    plac = placebo(recs, px, spy)

    # concentration: leave-one-batch-out mean range
    loo = [float((np.sum(net) - x) / (len(net) - 1)) for x in net] if len(net) > 1 else []
    conc = {"loo_min_bps": round(min(loo) * 1e4, 1), "loo_max_bps": round(max(loo) * 1e4, 1)} if loo else None

    S = stats(net)
    decision_reasons = []
    if not S or S["mean_bps"] <= 0:
        decision_reasons.append("primary net not > 0")
    if S and S["ci95_bps"][0] <= 0:
        decision_reasons.append("bootstrap lower bound <= 0")
    ds = stats(delayed)
    dd = stats(doubled)
    if not ds or ds["mean_bps"] <= 0:
        decision_reasons.append("fails delayed-entry")
    if not dd or dd["mean_bps"] <= 0:
        decision_reasons.append("fails doubled-cost")
    verdict = "REJECT_CLOSE_LEG" if decision_reasons else "ADVANCE_TO_100EVENT_CANDIDATE"

    report = {
        "test_id": "forced-flow-announcement-continuation-v1",
        "run_once": True, "manifest_sha256": sha,
        "spec": "docs/forced_flow_continuation_test_spec.md",
        "executable": {"batches_with_>=1_executable": len(prim),
                       "added_names_total": n_names, "added_names_executable": n_exec,
                       "note": ("non-executable = spinoff/recent-listing additions with no "
                                "pre-announcement trading history (no entry price in the "
                                "announcement->effective window). Executable subset = names "
                                "already trading (MidCap-400 promotions etc.).")},
        "gross_event_study_residual_bps": stats(gross),   # BEFORE costs, reported separately
        "primary_net_after_cost": S,
        "by_strata_net": {"ad_hoc": stats(adhoc), "quarterly_rebalance": stats(quart)},
        "stress_delayed_entry_net": ds,
        "stress_doubled_cost_net": dd,
        "placebo_same_name_random_window_resid_bps": stats(plac),
        "concentration_leave_one_batch_out_net": conc,
        "cost_model": f"{COST_BPS}bps/side x (long+short) x (entry+exit) = {2*COST_BPS}bps*(1+beta)",
        "decision_rule": "reject-only at n<100 executable events; promotion needs >=100",
        "verdict": verdict, "reject_reasons": decision_reasons,
        "power_note": ("n~%d executable batches -> MDE ~1.2-1.5%%; a null is uninformative "
                       "about tiny effects, NOT proof of zero. Consistent with the "
                       "disappearing-index-effect literature (NBER w30748)." % len(prim)),
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    print(f"executable batches: {len(prim)} (added names {n_exec}/{n_names})")
    print(f"GROSS resid (pre-cost): {report['gross_event_study_residual_bps']}")
    print(f"PRIMARY net: {S}")
    print(f"  ad-hoc net: {stats(adhoc)}")
    print(f"  quarterly net: {stats(quart)}")
    print(f"  delayed-entry net: {ds}")
    print(f"  doubled-cost net: {dd}")
    print(f"  placebo resid: {stats(plac)}")
    print(f"  concentration LOO: {conc}")
    print(f"VERDICT: {verdict}  reasons={decision_reasons}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
