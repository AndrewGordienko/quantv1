"""Track A: leak-free daily-bar tests around S&P 500 index-addition effective dates.

Two horizons, correctly separated by whether they are executable:

  * Effective-day open->close residual   -- DESCRIPTIVE EVENT STUDY. Entering at
    the effective-day open presumes the change was publicly known before that
    open; without a verified announcement time we cannot claim that, so this is
    not a deployable strategy.
  * D+1 open -> D+5 close residual        -- EXECUTABLE. The effective event is
    already complete and public by D+1; we enter at the next open and exit at a
    frozen horizon. Entry is NEVER the effective close.

Residuals use the market model vs SPY (sector ETFs are unavailable for most of
the added mid-caps, and no market-cap source exists). Inference clusters by the
effective-date batch. Matched controls are momentum + dollar-ADV neighbours that
were tradable and not themselves index changes near the event. Raw event-study
returns are reported separately from the after-cost portfolio ("fade") strategy.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..config import DATA_DIR, BENCHMARK_TICKER
from ..db import connect
from ..ingest.forced_flow import CENSUS_VERSION

PRE_BETA_DAYS = 60
MIN_BETA_OBS = 30
REVERSAL_ENTRY_LAG = 1        # D+1 open
REVERSAL_EXIT = 5             # D+5 close
DELAYED_ENTRY_LAG = 2         # robustness: D+2 open
COST_BPS_PER_SIDE = 15.0
N_CONTROLS = 5
CONTROL_EXCLUSION_DAYS = 10   # a control cannot be an index change within +/- this
PLACEBO_SHIFT_DAYS = 25       # placebo effective dates, shifted back
N_BOOT = 2000
MOMENTUM_DAYS = 60


def _load_daily(con) -> dict:
    df = con.execute(
        "SELECT ticker, date, open, close, volume FROM prices "
        "WHERE open IS NOT NULL AND close IS NOT NULL ORDER BY ticker, date").df()
    df["date"] = pd.to_datetime(df["date"])
    panel = {}
    for ticker, g in df.groupby("ticker"):
        panel[ticker] = {
            "date": g["date"].to_numpy(),
            "open": g["open"].to_numpy(dtype=float),
            "close": g["close"].to_numpy(dtype=float),
            "dollar": (g["close"].to_numpy(dtype=float) * g["volume"].to_numpy(dtype=float)),
        }
    return panel


def _idx_on_or_after(dates: np.ndarray, day: np.datetime64) -> int | None:
    i = int(np.searchsorted(dates, day, side="left"))
    return i if i < len(dates) else None


def _beta(panel: dict, ticker: str, eff_idx: int) -> float | None:
    stock, bench = panel[ticker], panel[BENCHMARK_TICKER]
    start = max(1, eff_idx - PRE_BETA_DAYS)
    dates = stock["date"][start:eff_idx]
    if len(dates) < MIN_BETA_OBS:
        return None
    b0 = int(np.searchsorted(bench["date"], dates[0], side="left"))
    b1 = int(np.searchsorted(bench["date"], dates[-1], side="right"))
    common, si, bi = np.intersect1d(dates, bench["date"][b0:b1],
                                    assume_unique=False, return_indices=True)
    if len(common) < MIN_BETA_OBS:
        return None
    sc = stock["close"][start:eff_idx][si]
    bc = bench["close"][b0:b1][bi]
    sr = sc[1:] / sc[:-1] - 1.0
    br = bc[1:] / bc[:-1] - 1.0
    var = float(np.var(br, ddof=1))
    if var <= 0:
        return None
    beta = float(np.cov(sr, br, ddof=1)[0, 1] / var)
    return float(np.clip(beta, 0.0, 3.0))


def _window_return(arr_open, arr_close, entry_idx, exit_idx) -> float | None:
    if entry_idx >= len(arr_open) or exit_idx >= len(arr_close):
        return None
    o, c = arr_open[entry_idx], arr_close[exit_idx]
    if not (np.isfinite(o) and np.isfinite(c) and o > 0):
        return None
    return float(c / o - 1.0)


def _residual(panel, ticker, eff_idx, beta, entry_lag, exit_offset):
    """Market-model residual over [eff+entry_lag open, eff+exit_offset close]."""
    stock, bench = panel[ticker], panel[BENCHMARK_TICKER]
    entry_i, exit_i = eff_idx + entry_lag, eff_idx + exit_offset
    stock_ret = _window_return(stock["open"], stock["close"], entry_i, exit_i)
    if stock_ret is None:
        return None
    entry_day, exit_day = stock["date"][entry_i], stock["date"][exit_i]
    be = int(np.searchsorted(bench["date"], entry_day, side="left"))
    bx = int(np.searchsorted(bench["date"], exit_day, side="left"))
    if be >= len(bench["date"]) or bx >= len(bench["date"]):
        return None
    if bench["date"][be] != entry_day or bench["date"][bx] != exit_day:
        return None
    bench_ret = _window_return(bench["open"], bench["close"], be, bx)
    if bench_ret is None:
        return None
    return stock_ret - beta * bench_ret


def _momentum_liquidity(panel, ticker, eff_idx):
    stock = panel[ticker]
    start = max(1, eff_idx - MOMENTUM_DAYS)
    if eff_idx - start < MOMENTUM_DAYS // 2:
        return None, None
    mom = _window_return(stock["close"], stock["close"], start, eff_idx - 1)
    dollar = float(np.nanmean(stock["dollar"][start:eff_idx]))
    return mom, dollar


def _cluster_bootstrap(values, batches, rng, n_boot=N_BOOT) -> dict:
    """Bootstrap the mean by resampling effective-date batches (the cluster)."""
    by = {}
    for v, b in zip(values, batches):
        if v is not None and np.isfinite(v):
            by.setdefault(b, []).append(float(v))
    flat = [v for vs in by.values() for v in vs]
    if len(by) < 3 or len(flat) < 10:
        return {"mean_bps": float(np.mean(flat) * 1e4) if flat else None,
                "ci_low": None, "ci_high": None, "n_obs": len(flat),
                "n_batches": len(by)}
    keys = list(by)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        picked = rng.choice(keys, len(keys), replace=True)
        boot[i] = np.mean(np.concatenate([by[k] for k in picked]))
    return {"mean_bps": float(np.mean(flat) * 1e4),
            "ci_low": float(np.percentile(boot, 2.5) * 1e4),
            "ci_high": float(np.percentile(boot, 97.5) * 1e4),
            "n_obs": len(flat), "n_batches": len(by)}


def run(verbose: bool = True) -> dict:
    con = connect()
    adds = con.execute("""
        SELECT ticker, effective_date, change_type, event_batch_id
        FROM forced_flow_events
        WHERE version=? AND event_type='addition' AND coverage_status='COVERED'
    """, [CENSUS_VERSION]).df()
    # every index-change ticker+date, to keep controls clean of contamination
    changes = con.execute("""
        SELECT ticker, effective_date FROM forced_flow_events WHERE version=?
    """, [CENSUS_VERSION]).fetchall()
    panel = _load_daily(con)
    con.close()

    change_days = {}
    for ticker, day in changes:
        change_days.setdefault(ticker, []).append(np.datetime64(pd.Timestamp(day)))
    universe = [t for t in panel if t != BENCHMARK_TICKER and len(panel[t]["date"]) > 200]

    rng = np.random.default_rng(7)
    rows = []
    for r in adds.itertuples(index=False):
        ticker = r.ticker
        if ticker not in panel:
            continue
        eff = np.datetime64(pd.Timestamp(r.effective_date))
        eff_idx = _idx_on_or_after(panel[ticker]["date"], eff)
        if eff_idx is None or eff_idx < PRE_BETA_DAYS or eff_idx + REVERSAL_EXIT >= len(panel[ticker]["date"]):
            continue
        beta = _beta(panel, ticker, eff_idx)
        if beta is None:
            continue
        rec = {
            "ticker": ticker, "batch": r.event_batch_id, "change_type": r.change_type,
            "eff_idx": eff_idx, "beta": beta,
            "event_day": _residual(panel, ticker, eff_idx, beta, 0, 0),
            "reversal": _residual(panel, ticker, eff_idx, beta, REVERSAL_ENTRY_LAG, REVERSAL_EXIT),
            "reversal_delayed": _residual(panel, ticker, eff_idx, beta, DELAYED_ENTRY_LAG, REVERSAL_EXIT),
        }
        # placebo: same machinery at a shifted, non-event date
        p_idx = eff_idx - PLACEBO_SHIFT_DAYS
        rec["placebo_reversal"] = (
            _residual(panel, ticker, p_idx, beta, REVERSAL_ENTRY_LAG, REVERSAL_EXIT)
            if p_idx > PRE_BETA_DAYS else None)
        # matched control (momentum + dollar-ADV neighbour, uncontaminated)
        rec["matched_reversal"] = _matched_control_residual(
            panel, universe, change_days, ticker, eff, eff_idx, beta, rng)
        rows.append(rec)

    result = _aggregate(rows, rng, verbose)
    with open(DATA_DIR / "forced_flow_reversal.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


def _matched_control_residual(panel, universe, change_days, ticker, eff, eff_idx, beta, rng):
    tgt_mom, tgt_dollar = _momentum_liquidity(panel, ticker, eff_idx)
    if tgt_mom is None or not tgt_dollar or tgt_dollar <= 0:
        return None
    cands = []
    for cand in universe:
        if cand == ticker:
            continue
        near = any(abs((eff - d) / np.timedelta64(1, "D")) <= CONTROL_EXCLUSION_DAYS
                   for d in change_days.get(cand, ()))
        if near:
            continue
        c_idx = _idx_on_or_after(panel[cand]["date"], eff)
        if c_idx is None or c_idx < PRE_BETA_DAYS or c_idx + REVERSAL_EXIT >= len(panel[cand]["date"]):
            continue
        c_mom, c_dollar = _momentum_liquidity(panel, cand, c_idx)
        if c_mom is None or not c_dollar or c_dollar <= 0:
            continue
        dist = abs(c_mom - tgt_mom) + abs(np.log(c_dollar) - np.log(tgt_dollar))
        cands.append((dist, cand, c_idx))
    if len(cands) < N_CONTROLS:
        return None
    cands.sort(key=lambda x: x[0])
    resids = []
    for _, cand, c_idx in cands[:N_CONTROLS]:
        cb = _beta(panel, cand, c_idx)
        if cb is None:
            continue
        res = _residual(panel, cand, c_idx, cb, REVERSAL_ENTRY_LAG, REVERSAL_EXIT)
        if res is not None:
            resids.append(res)
    return float(np.mean(resids)) if resids else None


def _fade_net(rows, rng):
    """Executable fade: at D+1 open take -sign(effective-day residual), exit D+5."""
    cost = 2.0 * COST_BPS_PER_SIDE / 1e4
    vals, batches = [], []
    for r in rows:
        if r["event_day"] is None or r["reversal"] is None:
            continue
        side = -1.0 if r["event_day"] > 0 else 1.0
        vals.append(side * r["reversal"] - cost)
        batches.append(r["batch"])
    net = _cluster_bootstrap(vals, batches, rng)
    arr = np.asarray([v for v in vals if v is not None])
    net["sharpe_per_event"] = (float(arr.mean() / arr.std(ddof=1))
                               if len(arr) > 1 and arr.std(ddof=1) > 0 else None)
    net["doubled_cost_mean_bps"] = (
        float(np.mean([v - cost for v in vals]) * 1e4) if vals else None)
    return net


def _aggregate(rows, rng, verbose):
    def col(name, subset=None):
        src = subset if subset is not None else rows
        return [r[name] for r in src], [r["batch"] for r in src]

    quarterly = [r for r in rows if r["change_type"] == "QUARTERLY_REBALANCE"]
    ad_hoc = [r for r in rows if r["change_type"] == "AD_HOC"]
    matched_diff = [(r["reversal"] - r["matched_reversal"])
                    if r["reversal"] is not None and r["matched_reversal"] is not None else None
                    for r in rows]

    result = {
        "test": "forced_flow_track_A_reversal",
        "census_version": CENSUS_VERSION,
        "n_events": len(rows),
        "clustering": "effective_date_batch",
        "residual_model": "market_model_vs_SPY",
        "caveats": [
            "sector ETFs unavailable for most added mid-caps; SPY market model only",
            "no market-cap source; controls matched on momentum + dollar-ADV proxy",
        ],
        "status_by_test": {
            "effective_day_open_to_close": "DESCRIPTIVE_EVENT_STUDY",
            "reversal_D1_open_to_D5_close": "EXECUTABLE",
            "closing_auction_pressure": "BLOCKED_NEEDS_INTRADAY",
            "announcement_to_effective_continuation": "BLOCKED_NEEDS_ANNOUNCEMENT_TIME",
        },
        "event_study": {
            "effective_day_residual": _cluster_bootstrap(*col("event_day"), rng),
        },
        "executable_reversal": {
            "added_residual_D1_D5": _cluster_bootstrap(*col("reversal"), rng),
            "added_residual_delayed_D2_D5": _cluster_bootstrap(*col("reversal_delayed"), rng),
            "matched_minus_control": _cluster_bootstrap(matched_diff, [r["batch"] for r in rows], rng),
            "placebo_shifted_date": _cluster_bootstrap(*col("placebo_reversal"), rng),
            "fade_strategy_net": _fade_net(rows, rng),
        },
        "by_change_type": {
            "quarterly_reversal": _cluster_bootstrap(*col("reversal", quarterly), rng),
            "ad_hoc_reversal": _cluster_bootstrap(*col("reversal", ad_hoc), rng),
        },
    }
    if verbose:
        rev = result["executable_reversal"]["added_residual_D1_D5"]
        fade = result["executable_reversal"]["fade_strategy_net"]
        plac = result["executable_reversal"]["placebo_shifted_date"]
        print(f"Track A: {len(rows)} events / {rev['n_batches']} batches")
        print(f"  event-study effective-day residual: "
              f"{result['event_study']['effective_day_residual']['mean_bps']} bps")
        print(f"  EXECUTABLE reversal D1->D5 residual: {rev['mean_bps']} bps "
              f"[{rev['ci_low']}, {rev['ci_high']}]")
        print(f"  placebo (shifted date) residual: {plac['mean_bps']} bps "
              f"[{plac['ci_low']}, {plac['ci_high']}]")
        print(f"  fade strategy net (after 2-side cost): {fade['mean_bps']} bps "
              f"[{fade['ci_low']}, {fade['ci_high']}], sharpe/event {fade['sharpe_per_event']}")
    return result


if __name__ == "__main__":
    run()
