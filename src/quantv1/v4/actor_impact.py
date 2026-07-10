"""Descriptive news-mention audit — explicitly not a B1/B2 actor test.

The input identifies people mentioned in secondary-news headlines, not people
who generated a public communication.  The output is therefore always labelled
``INVALID_PROXY_STUDY`` and cannot be interpreted as evidence for or against
actor effects.

The outcome calculation is still kept honest for diagnostic use: elapsed-time
horizons cannot cross the session, market/sector betas and pre-event volatility
are estimated from pre-event bars, and raw, residual, modeled-hedge and actually
executed hedge outcomes are reported separately.  Minute bars are not quotes,
so this study remains non-executable and the actually-hedged field is null.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..config import DATA_DIR, BENCHMARK_TICKER
from ..db import connect
from ..events.actors import EXTRACTION_VERSION
from .replay import BarPanel, to_ns

STUDY_STATUS = "INVALID_PROXY_STUDY"
HORIZONS_MINUTES = {"30m": 30, "2h": 120}
_MINUTE_NS = 60_000_000_000
_DAY_NS = 86_400_000_000_000
_PRE_EVENT_BARS = 10 * 390
_MIN_BETA_OBSERVATIONS = 100

SECTOR_ETF = {
    "communication services": "XLC",
    "consumer discretionary": "XLY",
    "consumer staples": "XLP",
    "energy": "XLE",
    "financials": "XLF",
    "health care": "XLV",
    "industrials": "XLI",
    "information technology": "XLK",
    "materials": "XLB",
    "real estate": "XLRE",
    "utilities": "XLU",
    "technology": "XLK",
    "financial services": "XLF",
}


def _session_id(timestamp_ns: int) -> int:
    # US regular-hours bars remain on one UTC calendar day.  If extended-hours
    # data is introduced, replace this with the exchange calendar session ID.
    return int(timestamp_ns) // _DAY_NS


def _aligned_open_close_return(panel: BarPanel, ticker: str,
                               start_ns: int, end_ns: int) -> float | None:
    if not panel.has(ticker) or _session_id(start_ns) != _session_id(end_ns):
        return None
    data = panel.data[ticker]
    i0 = int(np.searchsorted(data["ts"], start_ns, side="left"))
    i1 = int(np.searchsorted(data["ts"], end_ns, side="left"))
    if i0 >= len(data["ts"]) or i1 >= len(data["ts"]):
        return None
    if data["ts"][i0] != start_ns or data["ts"][i1] != end_ns:
        return None
    start, end = data["open"][i0], data["close"][i1]
    if not (np.isfinite(start) and np.isfinite(end) and start > 0):
        return None
    return float(end / start - 1.0)


def _pre_event_beta_vol(panel: BarPanel, ticker: str, benchmark: str,
                        entry_index: int) -> tuple[float | None, float | None, int]:
    """Estimate beta and one-minute vol using only bars before the event."""
    if not panel.has(benchmark):
        return None, None, 0
    stock = panel.data[ticker]
    bench = panel.data[benchmark]
    start = max(0, entry_index - _PRE_EVENT_BARS)
    stock_ts = stock["ts"][start:entry_index]
    if len(stock_ts) < _MIN_BETA_OBSERVATIONS + 1:
        return None, None, 0
    b0 = int(np.searchsorted(bench["ts"], stock_ts[0], side="left"))
    b1 = int(np.searchsorted(bench["ts"], stock_ts[-1], side="right"))
    common, stock_pos, bench_pos = np.intersect1d(
        stock_ts, bench["ts"][b0:b1], assume_unique=True, return_indices=True
    )
    if len(common) < _MIN_BETA_OBSERVATIONS + 1:
        return None, None, max(0, len(common) - 1)
    stock_close = stock["close"][start:entry_index][stock_pos]
    bench_close = bench["close"][b0:b1][bench_pos]
    same_session = (common[1:] // _DAY_NS) == (common[:-1] // _DAY_NS)
    valid = (same_session & np.isfinite(stock_close[1:]) & np.isfinite(stock_close[:-1]) &
             np.isfinite(bench_close[1:]) & np.isfinite(bench_close[:-1]) &
             (stock_close[:-1] > 0) & (bench_close[:-1] > 0))
    stock_returns = stock_close[1:][valid] / stock_close[:-1][valid] - 1.0
    bench_returns = bench_close[1:][valid] / bench_close[:-1][valid] - 1.0
    n = len(stock_returns)
    if n < _MIN_BETA_OBSERVATIONS:
        return None, None, n
    variance = float(np.var(bench_returns, ddof=1))
    beta = (float(np.cov(stock_returns, bench_returns, ddof=1)[0, 1] / variance)
            if variance > 0 else None)
    volatility = float(np.std(stock_returns, ddof=1))
    return beta, volatility, n


def _outcome(panel: BarPanel, ticker: str, public_time_ns: int,
             horizon_minutes: int, sector_benchmark: str | None) -> dict | None:
    data = panel.data[ticker]
    entry_index = panel.next_idx_after(ticker, public_time_ns)
    if entry_index is None:
        return None
    entry_ns = int(data["ts"][entry_index])
    target_ns = entry_ns + horizon_minutes * _MINUTE_NS
    exit_index = int(np.searchsorted(data["ts"], target_ns, side="left"))
    if exit_index >= len(data["ts"]):
        return None
    exit_ns = int(data["ts"][exit_index])
    if _session_id(entry_ns) != _session_id(exit_ns):
        return None
    # Gaps/halts do not turn a 2-hour target into an arbitrary later bar.
    if exit_ns - target_ns > _MINUTE_NS:
        return None
    raw_return = _aligned_open_close_return(panel, ticker, entry_ns, exit_ns)
    if raw_return is None:
        return None

    market_return = _aligned_open_close_return(panel, BENCHMARK_TICKER,
                                               entry_ns, exit_ns)
    market_beta, pre_vol, n_beta = _pre_event_beta_vol(
        panel, ticker, BENCHMARK_TICKER, entry_index
    )
    market_residual = (raw_return - market_beta * market_return
                       if market_beta is not None and market_return is not None else None)

    sector = sector_benchmark if sector_benchmark and panel.has(sector_benchmark) else None
    sector_return = (_aligned_open_close_return(panel, sector, entry_ns, exit_ns)
                     if sector else None)
    sector_beta, _, n_sector_beta = (_pre_event_beta_vol(panel, ticker, sector, entry_index)
                                     if sector else (None, None, 0))
    sector_residual = (raw_return - sector_beta * sector_return
                       if sector_beta is not None and sector_return is not None else None)

    horizon_vol = (pre_vol * np.sqrt(horizon_minutes)
                   if pre_vol is not None and pre_vol > 0 else None)
    modeled_hedge = sector_residual if sector_residual is not None else market_residual
    return {
        "entry_ns": entry_ns,
        "exit_ns": exit_ns,
        "same_session": True,
        "raw_return": raw_return,
        "market_benchmark": BENCHMARK_TICKER,
        "market_return": market_return,
        "market_beta": market_beta,
        "market_beta_residual": market_residual,
        "sector_benchmark": sector,
        "sector_return": sector_return,
        "sector_beta": sector_beta,
        "sector_beta_residual": sector_residual,
        "modeled_beta_hedged_return": modeled_hedge,
        "actually_hedged_return": None,
        "hedge_execution_status": "NOT_EXECUTED",
        "pre_event_minute_volatility": pre_vol,
        "raw_standardized": raw_return / horizon_vol if horizon_vol else None,
        "residual_standardized": modeled_hedge / horizon_vol
        if modeled_hedge is not None and horizon_vol else None,
        "beta_observations": n_beta,
        "sector_beta_observations": n_sector_beta,
        "execution_basis": "FIRST_POST_PUBLICATION_BAR_OPEN_PROXY",
        "executable": False,
    }


def _cluster_ci(values, clusters, rng, n_boot: int = 2000) -> dict:
    """One-way catalyst bootstrap; deliberately not called multiway."""
    pairs = [(float(value), cluster) for value, cluster in zip(values, clusters)
             if value is not None and np.isfinite(value)]
    if not pairs:
        return {"mean_bps": None, "ci_low": None, "ci_high": None, "n": 0,
                "clustering": "catalyst_one_way"}
    by: dict[str, list[float]] = {}
    for value, cluster in pairs:
        by.setdefault(str(cluster), []).append(value)
    vals = np.asarray([value for value, _ in pairs])
    if len(by) < 2 or len(vals) < 10:
        return {"mean_bps": float(vals.mean() * 1e4), "ci_low": None,
                "ci_high": None, "n": len(vals),
                "clustering": "catalyst_one_way"}
    keys = list(by)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        picked = rng.choice(keys, len(keys), replace=True)
        boot[b] = np.concatenate([by[key] for key in picked]).mean()
    return {
        "mean_bps": float(vals.mean() * 1e4),
        "ci_low": float(np.percentile(boot, 2.5) * 1e4),
        "ci_high": float(np.percentile(boot, 97.5) * 1e4),
        "n": len(vals), "clustering": "catalyst_one_way",
    }


def run(verbose: bool = True) -> dict:
    con = connect()
    actor_events = con.execute("""
        SELECT actor_id, ticker, public_time, catalyst_id
        FROM actor_events
        WHERE extraction_version=? AND actor_event_role='merely_mentioned'
          AND primary_hypothesis_eligible=FALSE
          AND ticker IS NOT NULL AND public_time IS NOT NULL
    """, [EXTRACTION_VERSION]).df()
    sectors = dict(con.execute("""
        SELECT ticker, lower(sector) FROM ticker_sectors
        WHERE ticker IS NOT NULL AND sector IS NOT NULL
    """).fetchall())
    panel = BarPanel(con, table="bars_minute")
    con.close()

    base = {
        "study_status": STUDY_STATUS,
        "is_b1_vs_b2_test": False,
        "primary_hypothesis_eligible": False,
        "reason": (
            "Secondary-news headline mentions do not identify actor-generated actions, "
            "quotes, stance, event semantics, or a matched semantic baseline."
        ),
        "inference_about_actor_effects": "NONE",
        "execution": {
            "executable": False,
            "reason": "minute bars are not contemporaneous executable quotes",
            "actual_hedge_returns_available": False,
        },
        "horizons": list(HORIZONS_MINUTES),
        "by_actor": {},
    }
    actor_events = actor_events[actor_events["ticker"].isin(panel.data)].reset_index(drop=True)
    if actor_events.empty:
        base["note"] = "no current context-only actor events with minute bars"
        with open(DATA_DIR / "v4_actor_impact.json", "w") as file:
            json.dump(base, file, indent=2)
        return base

    actor_events["public_time_ns"] = to_ns(actor_events["public_time"])
    records = []
    for row in actor_events.itertuples(index=False):
        record = {"actor": row.actor_id, "ticker": row.ticker,
                  "catalyst": row.catalyst_id}
        sector_benchmark = SECTOR_ETF.get(sectors.get(row.ticker, ""))
        for label, minutes in HORIZONS_MINUTES.items():
            record[label] = _outcome(panel, row.ticker, row.public_time_ns,
                                     minutes, sector_benchmark)
        records.append(record)

    rng = np.random.default_rng(11)
    frame = pd.DataFrame(records)
    for actor, group in frame.groupby("actor"):
        values, clusters = [], []
        for outcome, catalyst in zip(group["2h"], group["catalyst"]):
            if outcome and outcome["modeled_beta_hedged_return"] is not None:
                values.append(outcome["modeled_beta_hedged_return"])
                clusters.append(catalyst)
        if values:
            base["by_actor"][actor] = {
                "n": len(values),
                "descriptive_abs_modeled_residual_2h": _cluster_ci(
                    np.abs(values), clusters, rng
                ),
                "descriptive_signed_modeled_residual_2h": _cluster_ci(
                    values, clusters, rng
                ),
            }
    base["outcome_definitions"] = {
        "raw": "unhedged asset return",
        "residual": "pre-event beta-adjusted market/sector diagnostic",
        "modeled_hedge": "arithmetic hedge proxy, not an executed trade",
        "actually_hedged": "null until synchronized quotes and hedge executions exist",
        "standardization": "pre-event intraday volatility scaled to horizon",
        "session_rule": "reject if elapsed-time horizon leaves the session",
    }
    with open(DATA_DIR / "v4_actor_impact.json", "w") as file:
        json.dump(base, file, indent=2, default=str)

    if verbose:
        print(f"=== Actor news-mention audit: {STUDY_STATUS} ===")
        print("No B1/B2 comparison was performed; no inference about actor effects is valid.")
        print("Diagnostics use same-session, pre-event beta/vol-adjusted bar outcomes only.")
    return base


if __name__ == "__main__":
    run()
