"""Audit the LARGE strategy — the one survivor of leak-free testing.

The user's question: LARGE looked good in the holdout (14.1% CAGR, Sharpe 1.27,
-6.4% max DD, +1.21% factor-adj 63d CAR). Is it a real low-risk alpha sleeve, or
just under-invested tech beta? This computes:

  A. Portfolio-level
     - average gross exposure and % time in cash
     - factor-regression ALPHA and betas of the actual portfolio daily returns
       (the decisive test: is there alpha after market/size/value/momentum?)
     - return normalized to SPY / QQQ volatility (what scaling to full vol implies)
     - block-bootstrap CI on the portfolio's annualized return
  B. Per-trade factor-adjusted CAR by characteristic
     - new position vs add-on, owner (self/spouse), filing lag, trade size

Everything reuses the leak-free machinery (next-open, Carhart, cluster bootstrap,
2024+ holdout).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..config import DATA_DIR, BENCHMARK_TICKER
from ..db import connect
from ..portfolio import backtest_v2 as B
from .event_study_v2 import _cluster_bootstrap, HOLDOUT_START

QQQ = "QQQ"
FACTOR_COLS = ["mkt_rf", "smb", "hml", "mom"]


# ---------------------------------------------------------------------------
# A. Instrumented LARGE strategy — capture daily returns + gross exposure
# ---------------------------------------------------------------------------
def _run_large(con, cand=None, op=None, exclude_members=None,
               exclude_tickers=None) -> pd.DataFrame:
    if cand is None:
        cand = B._load_candidates(con)
    if op is None:
        op = B._panels(con)
    exclude_members = exclude_members or set()
    exclude_tickers = exclude_tickers or set()
    if exclude_members or exclude_tickers:
        cand = cand[~cand["member_key"].isin(exclude_members) &
                    ~cand["ticker"].isin(exclude_tickers)]
    strat = next(s for s in B.STRATEGIES if s.name == "large")
    cal = op.index
    cols = {t: i for i, t in enumerate(op.columns)}
    O = op.to_numpy()
    start_i = max(int(cal.searchsorted(pd.Timestamp("2015-01-01"))), 1)
    end_i = len(cal) - 1

    elig = cand[cand.apply(strat.eligible, axis=1)]
    by_fpos: dict[int, list] = {}
    for r in elig.to_dict("records"):
        fp = cal.searchsorted(r["filing_date"], side="right")
        if fp < len(cal):
            r["_prio"] = strat.priority(r)
            by_fpos.setdefault(fp, []).append(r)

    open_pos, w_prev, last_px = {}, {}, {}
    rows = []
    for t in range(start_i + 1, end_i + 1):
        ret = 0.0
        for tk, w in w_prev.items():
            ci = cols.get(tk)
            if ci is None:
                continue
            p0, p1 = B._px(O, ci, t - 1, last_px, tk), B._px(O, ci, t, last_px, tk)
            if p0 and p1 and p0 > 0:
                ret += w * (p1 / p0 - 1.0)
        for tk, s in list(open_pos.items()):
            ci = cols.get(tk)
            delisted = ci is not None and not np.isfinite(O[t:, ci]).any()
            if t >= s["exit_i"] or delisted:
                del open_pos[tk]
        if len(open_pos) < B.TARGET_N:
            pool = {}
            for fp in range(max(0, t - B.FRESH_TD), t + 1):
                for r in by_fpos.get(fp, []):
                    tk = r["ticker"]
                    if tk in open_pos or cols.get(tk) is None or not np.isfinite(O[t, cols[tk]]):
                        continue
                    if tk not in pool or r["_prio"] > pool[tk]["_prio"]:
                        pool[tk] = r
            for r in sorted(pool.values(), key=lambda x: x["_prio"], reverse=True):
                if len(open_pos) >= B.TARGET_N:
                    break
                open_pos[r["ticker"]] = {"exit_i": t + B.HOLD_TD}
        held = list(open_pos)
        invested = min(len(held) / B.TARGET_N, 1.0)
        if held:
            w = B._cap_weights(np.ones(len(held)), B.MAX_W)
            w = w / w.sum() * invested
            w_now = dict(zip(held, w))
        else:
            w_now = {}
        turn = 0.5 * sum(abs(w_now.get(tk, 0) - w_prev.get(tk, 0))
                         for tk in set(w_now) | set(w_prev))
        net = ret - turn * 2 * B.COST_BPS / 1e4
        rows.append({"date": cal[t], "ret": net, "gross": invested,
                     "n_held": len(held)})
        w_prev = w_now
    return pd.DataFrame(rows)


def _factor_regression(daily: pd.DataFrame, con) -> dict:
    fac = con.execute("SELECT * FROM factors ORDER BY date").df()
    fac["date"] = pd.to_datetime(fac["date"])
    d = daily.merge(fac, on="date", how="inner").dropna(subset=FACTOR_COLS)
    y = (d["ret"] - d["rf"]).to_numpy()
    X = np.column_stack([np.ones(len(d)), d[FACTOR_COLS].to_numpy()])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    dof = len(d) - X.shape[1]
    sigma2 = (resid @ resid) / dof
    cov = sigma2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    tstat = coef / se
    names = ["alpha_daily", *FACTOR_COLS]
    return {"n_days": int(len(d)),
            "alpha_annual": float(coef[0] * 252),
            "alpha_t": float(tstat[0]),
            "betas": {names[i]: {"beta": float(coef[i]), "t": float(tstat[i])}
                      for i in range(1, len(names))},
            "r2": float(1 - (resid @ resid) / (((y - y.mean()) ** 2).sum() + 1e-12))}


def _bench_daily(con, ticker) -> pd.Series:
    px = con.execute("SELECT date, open FROM prices WHERE ticker=? ORDER BY date",
                     [ticker]).df()
    px["date"] = pd.to_datetime(px["date"])
    s = px.set_index("date")["open"].pct_change()
    return s


def _block_bootstrap_ci(rets: np.ndarray, block: int = 21, n_boot: int = 2000,
                        seed: int = 11) -> dict:
    rng = np.random.default_rng(seed)
    n = len(rets)
    nblocks = n // block
    ann = []
    for _ in range(n_boot):
        starts = rng.integers(0, n - block, nblocks)
        sample = np.concatenate([rets[s:s + block] for s in starts])
        ann.append((1 + sample.mean()) ** 252 - 1)
    return {"mean_annual": float((1 + rets.mean()) ** 252 - 1),
            "ci_low": float(np.percentile(ann, 2.5)),
            "ci_high": float(np.percentile(ann, 97.5))}


def _portfolio_audit(con) -> dict:
    daily = _run_large(con)
    daily["date"] = pd.to_datetime(daily["date"])
    r = daily["ret"].to_numpy()
    ann_ret = (1 + r.mean()) ** 252 - 1
    ann_vol = r.std() * np.sqrt(252)
    sharpe = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan

    spy = _bench_daily(con, BENCHMARK_TICKER).reindex(daily["date"]).to_numpy()
    qqq = _bench_daily(con, QQQ).reindex(daily["date"]).to_numpy()
    spy_vol = np.nanstd(spy) * np.sqrt(252)
    qqq_vol = np.nanstd(qqq) * np.sqrt(252)

    reg = _factor_regression(daily, con)
    ho = pd.Timestamp(HOLDOUT_START)
    hold_r = daily[daily["date"] >= ho]["ret"].to_numpy()

    return {
        "avg_gross": float(daily["gross"].mean()),
        "avg_cash": float(1 - daily["gross"].mean()),
        "pct_time_fully_cash": float((daily["gross"] == 0).mean()),
        "avg_n_held": float(daily["n_held"].mean()),
        "ann_return": float(ann_ret), "ann_vol": float(ann_vol), "sharpe": float(sharpe),
        "spy_vol": float(spy_vol), "qqq_vol": float(qqq_vol),
        # scaling to a benchmark's vol multiplies BOTH return and drawdown by k
        "return_at_spy_vol": float(sharpe * spy_vol) if np.isfinite(sharpe) else None,
        "return_at_qqq_vol": float(sharpe * qqq_vol) if np.isfinite(sharpe) else None,
        "factor_regression": reg,
        "bootstrap_full": _block_bootstrap_ci(r),
        "bootstrap_holdout": _block_bootstrap_ci(hold_r) if len(hold_r) > 100 else None,
    }


# ---------------------------------------------------------------------------
# B. Per-trade factor-adjusted CAR by characteristic (large trades only)
# ---------------------------------------------------------------------------
def _trade_slices(con) -> dict:
    tables = [t[0] for t in con.execute("SHOW TABLES").fetchall()]
    if "event_v2" not in tables:
        return {}
    df = con.execute("""
        SELECT e.ticker, e.member_key, e.filing_date, e.amount_mid, e.is_repeat,
               e.disclosure_lag, e.car_ff_63 AS car, t.owner
        FROM event_v2 e JOIN trades t ON e.trade_id = t.trade_id
        WHERE e.car_ff_63 IS NOT NULL AND e.amount_mid >= 50000
    """).df()
    df["filing_date"] = pd.to_datetime(df["filing_date"])
    train = df[df["filing_date"] < HOLDOUT_START]

    def slc(sub):
        return _cluster_bootstrap(sub, "car", "ticker")

    out = {"n_large_trades": int(len(df))}
    out["new_vs_addon"] = {
        "new_position": slc(train[~train["is_repeat"]]),
        "add_on": slc(train[train["is_repeat"]])}
    out["by_owner"] = {o: slc(train[train["owner"] == o])
                       for o in ["self", "spouse", "joint"] if (train["owner"] == o).any()}
    lag_bins = [(0, 14, "<=14d"), (15, 30, "15-30d"), (31, 45, "31-45d"), (46, 400, ">45d")]
    out["by_filing_lag"] = {lbl: slc(train[(train["disclosure_lag"] >= lo) &
                                           (train["disclosure_lag"] <= hi)])
                            for lo, hi, lbl in lag_bins}
    size_bins = [(50e3, 100e3, "$50-100k"), (100e3, 250e3, "$100-250k"),
                 (250e3, 1e6, "$250k-1M"), (1e6, 1e12, "$1M+")]
    out["by_size"] = {lbl: slc(train[(train["amount_mid"] >= lo) &
                                     (train["amount_mid"] < hi)])
                      for lo, hi, lbl in size_bins}
    return out


def run(verbose: bool = True) -> dict:
    con = connect(read_only=True)
    audit = _portfolio_audit(con)
    slices = _trade_slices(con)
    con.close()
    out = {"portfolio": audit, "trade_slices": slices,
           "generated_at": datetime.now(timezone.utc).isoformat()}
    with open(DATA_DIR / "large_audit.json", "w") as f:
        json.dump(out, f, indent=2)
    if verbose:
        _print(out)
    return out


def _print(out):
    a = out["portfolio"]
    print("=== LARGE strategy audit ===")
    print(f"gross exposure avg={a['avg_gross']:.1%}  cash avg={a['avg_cash']:.1%}  "
          f"avg names held={a['avg_n_held']:.1f}")
    print(f"ann return={a['ann_return']:.1%}  vol={a['ann_vol']:.1%}  Sharpe={a['sharpe']:.2f}")
    print(f"return scaled to SPY vol ({a['spy_vol']:.0%})={a['return_at_spy_vol']:.1%}  "
          f"to QQQ vol ({a['qqq_vol']:.0%})={a['return_at_qqq_vol']:.1%}")
    r = a["factor_regression"]
    print(f"\nFactor regression (Carhart): alpha={r['alpha_annual']:+.2%}/yr "
          f"(t={r['alpha_t']:+.2f})  R2={r['r2']:.2f}")
    for k, v in r["betas"].items():
        print(f"  beta_{k}={v['beta']:+.2f} (t={v['t']:+.1f})")
    bf, bh = a["bootstrap_full"], a.get("bootstrap_holdout")
    print(f"\nbootstrap ann return full: {bf['mean_annual']:.1%} "
          f"[{bf['ci_low']:.1%}, {bf['ci_high']:.1%}]")
    if bh:
        print(f"bootstrap ann return holdout: {bh['mean_annual']:.1%} "
              f"[{bh['ci_low']:.1%}, {bh['ci_high']:.1%}]")
    s = out["trade_slices"]
    if s:
        print(f"\n--- factor-adj 63d CAR by characteristic (large trades, train) ---")
        for grp in ("new_vs_addon", "by_owner", "by_filing_lag", "by_size"):
            print(f"  {grp}:")
            for k, v in s[grp].items():
                if v.get("mean") is not None:
                    sig = "*" if (v["ci_low"] > 0 or v["ci_high"] < 0) else " "
                    print(f"    {k:14s} {v['mean']*100:+.2f}%{sig} "
                          f"[{v['ci_low']*100:+.2f},{v['ci_high']*100:+.2f}] n={v['n']}")


if __name__ == "__main__":
    run()
