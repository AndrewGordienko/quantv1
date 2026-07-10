"""Is LARGE's edge real, or an artifact of a few names/members and multiple testing?

The validation the review demanded before calling LARGE production-ready:

  1. Leave-one-politician-out / leave-one-ticker-out — drop the top contributors
     one at a time and see how the annualized return moves. Robust edges survive.
  2. Concentration — what share of large trades and of positive factor-adjusted
     CAR comes from the top 5 members / tickers. If <5 drive it, it's fragile.
  3. Paired block-bootstrap CI for LARGE minus a BETA-MATCHED SPY portfolio — the
     honest "excess return vs benchmark" test (raw-return CIs excluding zero do
     NOT prove excess return). active_t = r_large,t - beta * r_spy,t.
  4. Deflated Sharpe — haircut LARGE's Sharpe for the number of strategies/slices
     tried (multiple-testing / selection bias), via the expected-max-under-null.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..config import DATA_DIR, BENCHMARK_TICKER
from ..db import connect
from ..portfolio import backtest_v2 as B
from .large_audit import _run_large, _bench_daily

# Rough count of strategies + slices we have evaluated this project (for the
# multiple-testing haircut): 5 backtest strategies, ~5 event-study slices x2
# periods, ~13 audit slices, combo, intraday, reg -> ~30 independent-ish trials.
N_TRIALS = 30
EULER = 0.5772156649


def _ann(rets: np.ndarray) -> float:
    return float((1 + np.nanmean(rets)) ** 252 - 1)


def _leave_one_out(con) -> dict:
    cand = B._load_candidates(con)
    op = B._panels(con)
    base = _ann(_run_large(con, cand, op)["ret"].to_numpy())

    # top contributors by number of large trades
    large = cand[cand["amount_mid"] >= 50_000]
    top_m = large["member_key"].value_counts().head(15).index.tolist()
    top_t = large["ticker"].value_counts().head(15).index.tolist()

    lopo = []
    for m in top_m:
        r = _ann(_run_large(con, cand, op, exclude_members={m})["ret"].to_numpy())
        lopo.append({"drop": m, "ann_return": r, "delta": r - base})
    loto = []
    for t in top_t:
        r = _ann(_run_large(con, cand, op, exclude_tickers={t})["ret"].to_numpy())
        loto.append({"drop": t, "ann_return": r, "delta": r - base})

    lopo.sort(key=lambda x: x["delta"])
    loto.sort(key=lambda x: x["delta"])
    return {"base_ann_return": base,
            "leave_one_member_out_worst": lopo[:5],
            "leave_one_ticker_out_worst": loto[:5]}


def _concentration(con) -> dict:
    df = con.execute("""
        SELECT member_key, ticker, car_ff_63 AS car FROM event_v2
        WHERE car_ff_63 IS NOT NULL AND amount_mid >= 50000
    """).df()
    n = len(df)
    pos = df[df["car"] > 0]
    total_pos = pos["car"].sum()

    def share(col):
        by = df.groupby(col).size().sort_values(ascending=False)
        top5_trades = by.head(5).sum() / n
        by_car = pos.groupby(col)["car"].sum().sort_values(ascending=False)
        top5_car = by_car.head(5).sum() / total_pos if total_pos > 0 else np.nan
        return {"top5_share_of_trades": float(top5_trades),
                "top5_share_of_positive_car": float(top5_car),
                "n_distinct": int(df[col].nunique())}

    return {"n_large_trades": n, "by_member": share("member_key"),
            "by_ticker": share("ticker")}


def _paired_bootstrap_vs_beta_spy(con, block: int = 21, n_boot: int = 3000,
                                  seed: int = 13) -> dict:
    daily = _run_large(con)
    daily["date"] = pd.to_datetime(daily["date"])
    spy = _bench_daily(con, BENCHMARK_TICKER).reindex(daily["date"]).to_numpy()
    r = daily["ret"].to_numpy()
    mask = np.isfinite(r) & np.isfinite(spy)
    r, spy = r[mask], spy[mask]
    # estimate beta of LARGE on SPY
    beta = float(np.cov(r, spy)[0, 1] / np.var(spy))
    active = r - beta * spy                           # LARGE minus beta-matched SPY
    rng = np.random.default_rng(seed)
    n = len(active)
    nb = n // block
    boot = []
    for _ in range(n_boot):
        starts = rng.integers(0, n - block, nb)
        s = np.concatenate([active[i:i + block] for i in starts])
        boot.append(_ann(s))
    return {"beta_to_spy": beta,
            "active_ann_return": _ann(active),
            "ci_low": float(np.percentile(boot, 2.5)),
            "ci_high": float(np.percentile(boot, 97.5)),
            "prob_positive": float((np.array(boot) > 0).mean())}


def _deflated_sharpe(con, n_trials: int = N_TRIALS) -> dict:
    daily = _run_large(con)
    r = daily["ret"].to_numpy()
    r = r[np.isfinite(r)]
    T = len(r)
    sr = r.mean() / r.std()                            # per-period Sharpe
    sr_ann = sr * np.sqrt(252)
    # variance of the Sharpe estimator (Lo), with skew/kurtosis
    from scipy.stats import skew, kurtosis
    g3, g4 = float(skew(r)), float(kurtosis(r, fisher=False))
    se_sr = np.sqrt((1 - g3 * sr + (g4 - 1) / 4 * sr ** 2) / (T - 1))
    # expected maximum Sharpe of N null strategies (Bailey & Lopez de Prado)
    z1 = norm.ppf(1 - 1.0 / n_trials)
    z2 = norm.ppf(1 - 1.0 / (n_trials * np.e))
    sr0 = se_sr * ((1 - EULER) * z1 + EULER * z2)      # expected max under null (per-period)
    # deflated Sharpe ratio = P(true SR > expected-max-null)
    dsr = norm.cdf((sr - sr0) * np.sqrt(T - 1) /
                   np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2))
    return {"sharpe_annual": float(sr_ann), "n_trials": n_trials,
            "expected_max_null_sharpe_annual": float(sr0 * np.sqrt(252)),
            "deflated_sharpe_prob": float(dsr),
            "passes": bool(dsr > 0.95)}


def run(verbose: bool = True) -> dict:
    con = connect(read_only=True)
    out = {
        "leave_one_out": _leave_one_out(con),
        "concentration": _concentration(con),
        "vs_beta_matched_spy": _paired_bootstrap_vs_beta_spy(con),
        "deflated_sharpe": _deflated_sharpe(con),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    con.close()
    with open(DATA_DIR / "large_validation.json", "w") as f:
        json.dump(out, f, indent=2)
    if verbose:
        _print(out)
    return out


def _print(out):
    lo = out["leave_one_out"]
    print(f"=== LARGE validation ===\nbase ann return={lo['base_ann_return']:.1%}")
    print("worst leave-one-MEMBER-out:")
    for x in lo["leave_one_member_out_worst"][:3]:
        print(f"  drop {x['drop'][:24]:24s} -> {x['ann_return']:.1%} (delta {x['delta']:+.1%})")
    print("worst leave-one-TICKER-out:")
    for x in lo["leave_one_ticker_out_worst"][:3]:
        print(f"  drop {x['drop']:6s} -> {x['ann_return']:.1%} (delta {x['delta']:+.1%})")
    c = out["concentration"]
    print(f"\nconcentration (n={c['n_large_trades']}): "
          f"top-5 members = {c['by_member']['top5_share_of_trades']:.0%} of trades / "
          f"{c['by_member']['top5_share_of_positive_car']:.0%} of positive CAR; "
          f"top-5 tickers = {c['by_ticker']['top5_share_of_trades']:.0%} / "
          f"{c['by_ticker']['top5_share_of_positive_car']:.0%}")
    v = out["vs_beta_matched_spy"]
    print(f"\nvs beta-matched SPY (beta={v['beta_to_spy']:.2f}): active={v['active_ann_return']:+.1%} "
          f"CI[{v['ci_low']:+.1%},{v['ci_high']:+.1%}] P(>0)={v['prob_positive']:.2f}")
    d = out["deflated_sharpe"]
    print(f"deflated Sharpe: SR={d['sharpe_annual']:.2f} vs expected-max-null="
          f"{d['expected_max_null_sharpe_annual']:.2f} (N={d['n_trials']}) -> "
          f"DSR={d['deflated_sharpe_prob']:.2f} {'PASS' if d['passes'] else 'FAIL'}")


if __name__ == "__main__":
    run()
