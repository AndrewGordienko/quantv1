"""Robustness / fragility stress-test of the TSMOM lead — NOT new variants.

Protocol steps 8-9 applied to the one positive lead (tsmom_etf_diag_v1, net
Sharpe 0.66). These are stress tests of the FROZEN combined spec, not a new
signal search, so they do not add to the global strategy-trial count:

  1. Delayed entry  — enter +1 and +5 trading days late. A real edge survives
     modest execution delay; a microstructure artifact does not.
  2. Sub-period stability — Sharpe in three equal date thirds. Is 0.66 stable or
     concentrated in the 2022 trend year?
  3. Per-year net returns + 2022 concentration share.
  4. Probabilistic Sharpe Ratio PSR(0) — prob. the true Sharpe > 0 given n, skew,
     kurtosis. And the deflated benchmark E[max Sharpe|null] for the 8 TSMOM
     configs already tried, so the 0.66 is judged against multiple testing.

Output: data/tsmom_etf_stress.json.
"""

from __future__ import annotations

import importlib.util
import json
import math

import numpy as np
import pandas as pd
from scipy import stats as sps

from quantv1.config import DATA_DIR

# reuse the frozen strategy definition (module-level is import-safe)
_spec = importlib.util.spec_from_file_location(
    "tsmom_diag", str((DATA_DIR.parent / "scripts" / "tsmom_etf_diag.py")))
_d = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_d)

OUT = DATA_DIR / "tsmom_etf_stress.json"
ANN = _d.ANN


def combined_net(close, delay=1, bps=3.0, start=None, end=None):
    """Net daily return of the FROZEN combined-lookback TSMOM, with entry delay."""
    rets = close.pct_change()
    vol = rets.rolling(_d.VOL_WIN, min_periods=_d.VOL_WIN // 2).std() * np.sqrt(ANN)
    me = pd.DatetimeIndex(_d.month_ends(close.index))
    sig = sum(np.sign(close / close.shift(L) - 1.0) for L in _d.LOOKBACKS) / len(_d.LOOKBACKS)
    scale = (_d.TARGET_VOL / vol).clip(upper=_d.MAX_LEV)
    w_me = (sig * scale).reindex(me).where(scale.reindex(me).notna())
    w_daily = w_me.reindex(close.index).ffill()
    gross = (w_daily.shift(delay) * rets).sum(axis=1) / len(_d.INSTRUMENTS)
    turn = (w_me.fillna(0.0).diff().abs().sum(axis=1) / len(_d.INSTRUMENTS))
    cost = pd.Series(0.0, index=close.index)
    cost.loc[turn.index] = turn.values * (bps / 1e4)
    net = (gross - cost.reindex(close.index).fillna(0.0)).dropna()
    warm = close.index[max(_d.LOOKBACKS) + _d.VOL_WIN]
    net = net[net.index >= warm]
    if start is not None:
        net = net[net.index >= pd.Timestamp(start)]
    if end is not None:
        net = net[net.index <= pd.Timestamp(end)]
    return net


def sharpe(x):
    return round(float(x.mean() / x.std() * np.sqrt(ANN)), 3) if x.std() else None


def psr(x, sr_benchmark=0.0):
    """Probabilistic Sharpe Ratio: P(true Sharpe > sr_benchmark). Non-annualized."""
    x = np.asarray(x.dropna())
    n = len(x)
    sr = x.mean() / x.std()                       # per-observation Sharpe
    sk = float(sps.skew(x))
    ku = float(sps.kurtosis(x, fisher=False))     # non-excess
    sr_b = sr_benchmark / np.sqrt(ANN)
    denom = math.sqrt(max(1e-12, 1 - sk * sr + (ku - 1) / 4 * sr ** 2))
    z = (sr - sr_b) * math.sqrt(n - 1) / denom
    return round(float(sps.norm.cdf(z)), 4)


def expected_max_sharpe_null(trial_sharpes_ann, n_trials):
    """Deflated benchmark: E[max Sharpe] under the null given trial dispersion."""
    v = np.std([s for s in trial_sharpes_ann if s is not None], ddof=1)
    if not np.isfinite(v) or v == 0 or n_trials < 2:
        return None
    gamma = 0.5772156649
    e = math.e
    z = ((1 - gamma) * sps.norm.ppf(1 - 1.0 / n_trials)
         + gamma * sps.norm.ppf(1 - 1.0 / (n_trials * e)))
    return round(float(v * z), 3)


def main():
    close = _d.load()
    base = combined_net(close, delay=1)
    idx = base.index
    thirds = np.array_split(idx, 3)

    # trial Sharpes from the diag run (the 8 configs already counted)
    diag = json.load(open(DATA_DIR / "tsmom_etf_diag.json"))
    trial_sharpes = [v["net"]["sharpe"] for v in diag["results"].values()]

    per_year = {}
    for y, g in base.groupby(base.index.year):
        per_year[str(y)] = {"net_return": round(float((1 + g).prod() - 1), 4),
                            "sharpe": sharpe(g), "n_days": int(len(g))}
    total_growth = float((1 + base).prod())
    ex2022 = base[base.index.year != 2022]
    report = {
        "label": "STRESS_TEST_OF_FROZEN_TSMOM_LEAD",
        "not_new_variants": True,
        "baseline_full": {"sharpe": sharpe(base), "n_days": int(len(base)),
                          "date_span": [str(idx.min().date()), str(idx.max().date())]},
        "delayed_entry": {
            "delay_1d_baseline": sharpe(base),
            "delay_2d": sharpe(combined_net(close, delay=2)),
            "delay_6d": sharpe(combined_net(close, delay=6)),
        },
        "subperiod_stability": {
            f"third_{i+1}": {"span": [str(t[0].date()), str(t[-1].date())],
                             "sharpe": sharpe(base[base.index.isin(t)])}
            for i, t in enumerate(thirds)
        },
        "per_year": per_year,
        "concentration": {
            "sharpe_ex_2022": sharpe(ex2022),
            "sharpe_with_2022": sharpe(base),
            "note": "if Sharpe collapses without 2022, the edge is trend-regime concentrated",
        },
        "multiple_testing": {
            "psr_true_sharpe_gt_0": psr(base, 0.0),
            "psr_true_sharpe_gt_0p5": psr(base, 0.5),
            "n_tsmom_configs_tried": len(trial_sharpes),
            "deflated_benchmark_E_max_sharpe_null": expected_max_sharpe_null(
                trial_sharpes, len(trial_sharpes)),
            "observed_best_ann_sharpe": max([s for s in trial_sharpes if s is not None]),
        },
    }
    # verdict
    sub = [v["sharpe"] for v in report["subperiod_stability"].values()]
    stable = all(s is not None and s > 0 for s in sub)
    delay_ok = report["delayed_entry"]["delay_6d"] and report["delayed_entry"]["delay_6d"] > 0.3
    ex22_ok = report["concentration"]["sharpe_ex_2022"] and report["concentration"]["sharpe_ex_2022"] > 0.3
    report["verdict"] = ("ROBUST_MODEST_DIVERSIFIER" if (stable and delay_ok and ex22_ok)
                         else "FRAGILE_OR_REGIME_CONCENTRATED")
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    print(f"baseline Sharpe {report['baseline_full']['sharpe']}  n={report['baseline_full']['n_days']}")
    print("delayed entry:", report["delayed_entry"])
    print("subperiod:", {k: v["sharpe"] for k, v in report["subperiod_stability"].items()})
    print("ex-2022 Sharpe:", report["concentration"]["sharpe_ex_2022"],
          " with-2022:", report["concentration"]["sharpe_with_2022"])
    print("PSR(>0):", report["multiple_testing"]["psr_true_sharpe_gt_0"],
          " PSR(>0.5):", report["multiple_testing"]["psr_true_sharpe_gt_0p5"],
          " E[max|null]:", report["multiple_testing"]["deflated_benchmark_E_max_sharpe_null"])
    print("per-year:", {y: v["net_return"] for y, v in per_year.items()})
    print("VERDICT:", report["verdict"])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
