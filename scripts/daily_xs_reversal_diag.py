"""Daily cross-sectional residual momentum / reversal — GROSS-SIGNAL DIAGNOSTIC.

Backlog #2. This is NOT a promotion test. Per docs/point_in_time_panel_spec.md
the daily panel is survivorship-biased *by construction* (currently-listed names
backfilled to 2012; 51.5% of S&P 500 deletions absent). A dollar-neutral
long/short built on this panel is therefore biased in the OPTIMISTIC direction:
survivors are over-represented on both legs but the missing names are
disproportionately distressed index-exiters.

Decision logic (why this gate is worth running before paying for a clean panel):
  * NULL / negative gross signal here  -> DECISIVE KILL. Survivorship bias could
    not even manufacture a signal; a clean panel would only be worse.
  * POSITIVE gross signal here         -> INCONCLUSIVE. Could be bias. Justifies
    the cost of a survivorship-safe panel to retest honestly. Not tradeable.

Residualization: cross-sectional demeaning among the eligible liquid universe
each day (market-neutral by construction). Proper rolling-beta / sector-neutral
residualization is a refinement, not needed to read a GROSS signal. Frozen,
return-blind universe/signal rules below; costs applied on realized turnover.

Output: data/daily_xs_reversal_diag.json (diagnostic, clearly labelled).
"""

from __future__ import annotations

import json

import duckdb
import numpy as np
import pandas as pd

from quantv1.config import DB_PATH, DATA_DIR, SECTOR_ETFS

OUT = DATA_DIR / "daily_xs_reversal_diag.json"

# --- frozen universe rules -------------------------------------------------
MIN_PRICE = 5.0            # prior-day close floor (penny-stock exclusion)
MIN_ADV_USD = 20_000_000   # trailing 60d dollar ADV floor (liquid universe)
ADV_WINDOW = 60
DECILE = 0.10              # long top decile, short bottom decile
EXCLUDE = set(SECTOR_ETFS) | {"SPY", "QQQ", "DIA", "IWM", "VOO", "VTI"}

# --- frozen predeclared signal grid (variations counted globally) ----------
# (name, kind, lookback, skip)  kind: 'mom' = +sum residual, 'rev' = -sum
SIGNALS = [
    ("reversal_5d",   "rev", 5,  0),
    ("reversal_21d",  "rev", 21, 0),
    ("momentum_60d1", "mom", 60, 1),
    ("momentum_21d1", "mom", 21, 1),
]
HOLDS = [2, 5, 10]
COST_BPS = [10.0, 20.0]   # per unit notional traded (baseline, doubled stress)
ANN = 252


def load_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = con.execute(
        "SELECT ticker, date, close, volume FROM prices ORDER BY date"
    ).df()
    con.close()
    df = df[~df["ticker"].isin(EXCLUDE)]
    close = df.pivot(index="date", columns="ticker", values="close").sort_index()
    dv = (df.assign(dv=df["close"] * df["volume"])
            .pivot(index="date", columns="ticker", values="dv").sort_index())
    return close, dv


def build_masks(close: pd.DataFrame, dv: pd.DataFrame):
    ret = close.pct_change()
    prev_close = close.shift(1)
    adv = dv.rolling(ADV_WINDOW, min_periods=ADV_WINDOW // 2).mean().shift(1)
    elig = (adv > MIN_ADV_USD) & (prev_close > MIN_PRICE) & ret.notna()
    # cross-sectional (market) demeaning among the eligible universe each day
    r_elig = ret.where(elig)
    resid = r_elig.sub(r_elig.mean(axis=1), axis=0)
    return ret, elig, resid


def signal_matrix(resid: pd.DataFrame, kind: str, lb: int, skip: int) -> pd.DataFrame:
    s = resid.shift(skip).rolling(lb, min_periods=max(2, lb // 2)).sum()
    return -s if kind == "rev" else s


def long_short_weights(sig: pd.DataFrame, elig: pd.DataFrame) -> pd.DataFrame:
    """Dollar-neutral decile weights: +1 gross long, -1 gross short per day."""
    s = sig.where(elig)
    lo = s.rank(axis=1, pct=True)
    n = elig.sum(axis=1)
    top = lo.ge(1 - DECILE)
    bot = lo.le(DECILE)
    ntop = top.sum(axis=1).replace(0, np.nan)
    nbot = bot.sum(axis=1).replace(0, np.nan)
    w = top.div(ntop, axis=0).astype(float) - bot.div(nbot, axis=0).astype(float)
    # require a minimally broad book (breadth control)
    w = w.where((ntop >= 10) & (nbot >= 10) & (n >= 100))
    return w


def backtest(sig, ret, elig, hold, bps):
    w = long_short_weights(sig, elig).fillna(0.0)
    # overlapping book: average target weights over the last `hold` formation days
    book = sum(w.shift(h) for h in range(hold)) / hold
    fwd = ret.shift(-1)                        # enter at t, earn t->t+1
    pnl = (book * fwd).sum(axis=1)
    turnover = book.diff().abs().sum(axis=1)   # total notional traded
    cost = turnover * (bps / 1e4)
    net = pnl - cost
    valid = book.abs().sum(axis=1) > 0
    net, pnl, turnover = net[valid], pnl[valid], turnover[valid]
    if len(net) < 50:
        return None

    def stats(x):
        mu, sd = x.mean(), x.std()
        return {"ann_return": round(float(mu * ANN), 4),
                "ann_vol": round(float(sd * np.sqrt(ANN)), 4),
                "sharpe": round(float(mu / sd * np.sqrt(ANN)), 3) if sd else None}

    g, nstat = stats(pnl), stats(net)
    return {"n_dates": int(len(net)),
            "avg_turnover": round(float(turnover.mean()), 4),
            "gross": g, "net": nstat,
            "net_positive": bool(nstat["ann_return"] > 0 and (nstat["sharpe"] or 0) > 0)}


def main() -> None:
    close, dv = load_panel()
    ret, elig, resid = build_masks(close, dv)
    results = {}
    n_trials = 0
    best = {"sharpe": -99, "key": None}
    for name, kind, lb, skip in SIGNALS:
        sig = signal_matrix(resid, kind, lb, skip)
        for hold in HOLDS:
            for bps in COST_BPS:
                key = f"{name}|hold{hold}|{int(bps)}bps"
                res = backtest(sig, ret, elig, hold, bps)
                if res is None:
                    continue
                n_trials += 1
                results[key] = res
                sh = res["net"]["sharpe"] or -99
                if bps == COST_BPS[0] and sh > best["sharpe"]:
                    best = {"sharpe": sh, "key": key}
    any_pos = any(v["net_positive"] for v in results.values())
    report = {
        "label": "SURVIVORSHIP_INFLATED_DIAGNOSTIC",
        "not_a_promotion_test": True,
        "panel_contract": "PANEL_CONTRACT_INCOMPLETE (see pit_panel_audit.json)",
        "residualization": "cross-sectional market demeaning among eligible universe",
        "universe_rules": {"min_price": MIN_PRICE, "min_adv_usd": MIN_ADV_USD,
                           "adv_window": ADV_WINDOW, "decile": DECILE,
                           "excluded_etfs": sorted(EXCLUDE)},
        "n_variations_tried": n_trials,
        "avg_eligible_names": round(float(elig.sum(axis=1).replace(0, np.nan).mean()), 1),
        "date_span": [str(ret.index.min().date()), str(ret.index.max().date())],
        "best_net_sharpe_at_baseline_cost": best,
        "results": results,
        "verdict": ("GROSS_SIGNAL_PRESENT_INCONCLUSIVE" if any_pos
                    else "NO_GROSS_SIGNAL_KILL_CANDIDATE"),
        "interpretation": (
            "Positive here is INCONCLUSIVE (survivorship-optimistic bias); only a "
            "clean panel can confirm. Negative/null across the grid is a decisive "
            "kill for backlog #2 -- bias could not manufacture a signal."),
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"variations={n_trials} avg_names={report['avg_eligible_names']} "
          f"span={report['date_span']}")
    for k, v in results.items():
        print(f"  {k:34s} gross_Sh={v['gross']['sharpe']:>6}  "
              f"net_Sh={v['net']['sharpe']:>6}  net_ann={v['net']['ann_return']:>7}")
    print(f"VERDICT: {report['verdict']}  best={best}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
