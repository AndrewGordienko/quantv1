"""F1 research screen for latent intraday flow shocks.

This module deliberately implements only what the current data can support:
minute OHLCV/VWAP bars.  It is *not* an order-flow model and it never labels
the bar-direction volume proxy as buyer/seller initiated volume.  In
particular, there are no NBBO quotes, queue sizes, trade signs, or book depth
in ``bars_minute``.  The output is consequently a research report and signal
episodes, never executable orders.

The frozen F1 rule looks for a continuation fingerprint:

* a two-factor (market + sector excess) residual move;
* time-of-day relative volume and four-minute signed-volume persistence;
* peer confirmation and a one-sided CUSUM change point; and
* low price impact per unit of relative volume.

The paired reversal fingerprint is recorded as a diagnostic only.  It is not
traded: the repository has already rejected generic intraday fading.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from ..config import DATA_DIR, BENCHMARK_TICKER, SECTOR_ETFS
from ..db import connect


YAHOO_TO_ETF = {
    "Technology": "XLK", "Financial Services": "XLF", "Healthcare": "XLV",
    "Energy": "XLE", "Industrials": "XLI", "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP", "Utilities": "XLU", "Basic Materials": "XLB",
    "Real Estate": "XLRE", "Communication Services": "XLC",
}


@dataclass(frozen=True)
class Params:
    """Pre-registered F1 screen parameters; do not optimize them on outcomes."""

    beta_window: int = 120
    z_window: int = 60
    volume_days: int = 20
    flow_minutes: int = 4
    residual_z_min: float = 1.5
    relative_volume_min: float = 1.5
    score_abs_min: float = 3.0
    cusum_drift: float = 0.5
    cusum_threshold: float = 3.0
    cooldown_minutes: int = 60
    round_trip_cost_bps: float = 16.0
    holdout_start: str = "2026-01-01"
    seed: int = 7


def _regular_session(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep only XNYS regular-session minutes, DST-safe from UTC timestamps."""
    out = frame.copy()
    utc = pd.to_datetime(out["ts"], utc=True)
    local = utc.dt.tz_convert("America/New_York")
    mins = local.dt.hour * 60 + local.dt.minute
    out["session_date"] = local.dt.date
    out["slot"] = mins - (9 * 60 + 30)
    return out.loc[(local.dt.dayofweek < 5) & (out["slot"] >= 0) & (out["slot"] < 390)].copy()


def _past_z(x: pd.DataFrame, window: int) -> pd.DataFrame:
    """z-score against strictly earlier values; current observation is excluded."""
    mu = x.rolling(window, min_periods=window).mean().shift(1)
    sd = x.rolling(window, min_periods=window).std(ddof=0).shift(1)
    return (x - mu) / sd.replace(0, np.nan)


def _two_factor_residual(y: pd.DataFrame, market: pd.Series,
                         sector_excess: pd.DataFrame, window: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Rolling OLS residual with coefficients frozen before the current bar.

    The factors are market return and sector return in excess of the market,
    which avoids treating two highly collinear raw index returns as separate
    independent factors.  The closed-form 2x2 solution is vectorized over
    tickers and has no future observations in either beta.
    """
    x1 = pd.DataFrame(np.broadcast_to(market.to_numpy()[:, None], y.shape),
                      index=y.index, columns=y.columns)
    x2 = sector_excess.reindex(columns=y.columns)

    def mean(x):
        return x.rolling(window, min_periods=window).mean().shift(1)

    m1, m2, my = mean(x1), mean(x2), mean(y)
    v1 = mean(x1 * x1) - m1 * m1
    v2 = mean(x2 * x2) - m2 * m2
    c12 = mean(x1 * x2) - m1 * m2
    c1y = mean(x1 * y) - m1 * my
    c2y = mean(x2 * y) - m2 * my
    det = v1 * v2 - c12 * c12
    det = det.where(det.abs() > 1e-14)
    b_market = (c1y * v2 - c2y * c12) / det
    b_sector = (c2y * v1 - c1y * c12) / det
    return y - b_market * x1 - b_sector * x2, b_market, b_sector


def _cusum(z: pd.DataFrame, dates: pd.Series, drift: float) -> pd.DataFrame:
    """One-sided, direction-aware CUSUM; states reset at each cash session."""
    result = pd.DataFrame(np.nan, index=z.index, columns=z.columns)
    day = dates.to_numpy()
    for col in z:
        value = z[col].to_numpy(dtype=float)
        out = np.full(len(value), np.nan)
        pos = neg = 0.0
        last_day = None
        for i, v in enumerate(value):
            if day[i] != last_day:
                pos = neg = 0.0
                last_day = day[i]
            if not np.isfinite(v):
                continue
            pos = max(0.0, pos + v - drift)
            neg = min(0.0, neg + v + drift)
            out[i] = pos if v >= 0 else -neg
        result[col] = out
    return result


def _time_of_day_relative_volume(bars: pd.DataFrame, volume_days: int) -> pd.DataFrame:
    """Relative volume vs prior sessions at the same minute-of-session."""
    work = bars.sort_values(["ticker", "session_date", "slot"]).copy()
    work["expected_volume"] = work.groupby(["ticker", "slot"])["volume"].transform(
        lambda x: x.shift(1).rolling(volume_days, min_periods=max(5, volume_days // 2)).median())
    work["relative_volume"] = work["volume"] / work["expected_volume"].replace(0, np.nan)
    return work


def _episode_starts(candidates: pd.DataFrame, cooldown_minutes: int) -> pd.DataFrame:
    """Keep a single signal per ticker per shock episode without look-ahead."""
    if candidates.empty:
        return candidates.copy()
    keep = []
    gap = pd.Timedelta(minutes=cooldown_minutes)
    for _, group in candidates.sort_values(["ticker", "ts"]).groupby("ticker", sort=False):
        last = None
        for row in group.itertuples():
            if last is None or row.ts - last >= gap:
                keep.append(row.Index)
                last = row.ts
    return candidates.loc[keep].copy()


def _summary(values: pd.Series) -> dict:
    values = values.dropna()
    if values.empty:
        return {"n": 0, "mean_bps": None, "hit_rate": None, "t_stat": None}
    std = values.std(ddof=1)
    return {
        "n": int(len(values)), "mean_bps": float(values.mean() * 1e4),
        "hit_rate": float((values > 0).mean()),
        "t_stat": float(values.mean() / (std / np.sqrt(len(values)))) if std and np.isfinite(std) else None,
    }


def _outcome_report(episodes: pd.DataFrame, p: Params) -> dict:
    out: dict[str, object] = {}
    cost = p.round_trip_cost_bps / 1e4
    holdout = pd.Timestamp(p.holdout_start)
    for h in (5, 15, 30, 60):
        col = f"hedged_{h}m"
        if col not in episodes:
            continue
        net = episodes[col] - cost
        delayed = episodes.get(f"hedged_{h}m_delayed", pd.Series(np.nan, index=episodes.index)) - cost
        doubled = episodes[col] - 2 * cost
        out[f"{h}m"] = {
            "all": _summary(net),
            "train": _summary(net[episodes["ts"] < holdout]),
            "holdout": _summary(net[episodes["ts"] >= holdout]),
            "delayed_one_minute": _summary(delayed),
            "doubled_cost": _summary(doubled),
        }
    return out


def _controls(episodes: pd.DataFrame, features: pd.DataFrame, p: Params) -> dict:
    """Same-time and same-time-of-day controls; diagnostic, not significance tests."""
    if episodes.empty:
        return {"shuffled_time_30m": _summary(pd.Series(dtype=float)),
                "shuffled_ticker_30m": _summary(pd.Series(dtype=float))}
    rng = np.random.default_rng(p.seed)
    cost = p.round_trip_cost_bps / 1e4
    eligible = features.dropna(subset=["hedged_30m"])
    time_values = []
    for (_, slot), g in episodes.groupby(["ticker", "slot"]):
        pool = eligible[(eligible["ticker"] == g["ticker"].iloc[0]) & (eligible["slot"] == slot)]["hedged_30m"].to_numpy()
        if len(pool):
            time_values.extend(rng.choice(pool, len(g), replace=True) - cost)

    # A different name at the same timestamp retains market state and time of day,
    # while deliberately breaking the name-specific shock association.
    pool = eligible[["ts", "ticker", "hedged_30m"]].rename(
        columns={"ticker": "control_ticker", "hedged_30m": "control_return"})
    chosen = []
    for r in episodes[["ts", "ticker"]].itertuples(index=False):
        options = pool[(pool["ts"] == r.ts) & (pool["control_ticker"] != r.ticker)]["control_return"].to_numpy()
        if len(options):
            chosen.append(rng.choice(options) - cost)
    return {"shuffled_time_30m": _summary(pd.Series(time_values, dtype=float)),
            "shuffled_ticker_30m": _summary(pd.Series(chosen, dtype=float))}


def build_features(params: Params | None = None) -> tuple[pd.DataFrame, dict]:
    """Construct leak-free F1 features from the local minute-bar table."""
    p = params or Params()
    con = connect(read_only=True)
    bars = con.execute("""
        SELECT ticker, ts, open, close, volume, vwap
        FROM bars_minute
        WHERE close > 0 AND open > 0 AND volume >= 0
        ORDER BY ts, ticker
    """).df()
    sectors = dict(con.execute("SELECT ticker, sector FROM ticker_sectors").fetchall())
    con.close()
    if bars.empty:
        return pd.DataFrame(), {"status": "NO_MINUTE_BARS"}
    bars = _regular_session(bars)
    bars = _time_of_day_relative_volume(bars, p.volume_days)
    close = bars.pivot(index="ts", columns="ticker", values="close").sort_index()
    open_ = bars.pivot(index="ts", columns="ticker", values="open").reindex(close.index)
    volume = bars.pivot(index="ts", columns="ticker", values="relative_volume").reindex(close.index)
    vwap = bars.pivot(index="ts", columns="ticker", values="vwap").reindex(close.index)
    slots = bars.drop_duplicates("ts").set_index("ts")["slot"].reindex(close.index)
    dates = bars.drop_duplicates("ts").set_index("ts")["session_date"].reindex(close.index)

    etfs = set(SECTOR_ETFS) | {BENCHMARK_TICKER, "QQQ"}
    names = [t for t in close.columns if t not in etfs and YAHOO_TO_ETF.get(sectors.get(t)) in close.columns]
    if BENCHMARK_TICKER not in close or not names:
        return pd.DataFrame(), {"status": "INSUFFICIENT_FACTOR_UNIVERSE", "n_names": len(names)}

    ret = close.pct_change(fill_method=None)
    ret.loc[slots == 0] = np.nan                    # no overnight return in a minute signal
    market = ret[BENCHMARK_TICKER]
    sector = pd.DataFrame({t: ret[YAHOO_TO_ETF[sectors[t]]] - market for t in names}, index=ret.index)
    residual, beta_market, beta_sector = _two_factor_residual(ret[names], market, sector, p.beta_window)
    residual_z = _past_z(residual, p.z_window)

    relative_volume = volume[names]
    log_volume_z = _past_z(np.log(relative_volume.clip(lower=1e-6)), p.z_window)
    signed_rel_volume = np.sign(residual) * relative_volume
    flow = signed_rel_volume.rolling(p.flow_minutes, min_periods=p.flow_minutes).sum()
    flow_z = _past_z(flow, p.z_window)
    impact = residual.abs() / relative_volume.replace(0, np.nan)
    impact_z = _past_z(impact, p.z_window)

    peer_confirmation = pd.DataFrame(index=ret.index, columns=names, dtype=float)
    for ticker in names:
        peers = [x for x in names if x != ticker and sectors.get(x) == sectors.get(ticker)]
        if peers:
            peer_excess = ret[peers].mean(axis=1) - market
            # Keep this signed.  A negative peer residual confirms a negative
            # stock residual and must pull the directional score further down.
            peer_confirmation[ticker] = _past_z(peer_excess.to_frame(ticker), p.z_window)[ticker]
    peer_confirmation = peer_confirmation.fillna(0.0).clip(-3, 3)
    cusum = _cusum(residual_z, dates, p.cusum_drift)
    score = residual_z + 0.75 * log_volume_z + 0.75 * flow_z + 0.5 * peer_confirmation - 0.5 * impact_z

    sign = np.sign(residual)
    same_day = dates.eq(dates.shift(p.flow_minutes - 1))
    persistent = sign.rolling(p.flow_minutes, min_periods=p.flow_minutes).sum().abs().eq(p.flow_minutes)
    persistent = persistent.where(same_day, False)
    continuation = (
        residual_z.abs().ge(p.residual_z_min) & relative_volume.ge(p.relative_volume_min) &
        score.abs().ge(p.score_abs_min) & cusum.ge(p.cusum_threshold) & persistent &
        score.mul(sign).gt(0) & peer_confirmation.mul(sign).ge(0.25)
    )
    reversal = (residual_z.abs().ge(2.0) & relative_volume.lt(1.0) & impact_z.gt(1.0) &
                peer_confirmation.mul(sign).lt(0.25))

    frames = []
    for ticker in names:
        part = pd.DataFrame({
            "ts": close.index, "ticker": ticker, "session_date": dates.to_numpy(), "slot": slots.to_numpy(),
            "open": open_[ticker].to_numpy(), "close": close[ticker].to_numpy(),
            "vwap_distance": ((close[ticker] - vwap[ticker]) / vwap[ticker]).to_numpy(),
            "market_open": open_[BENCHMARK_TICKER].to_numpy(), "market_close": close[BENCHMARK_TICKER].to_numpy(),
            "sector_open": open_[YAHOO_TO_ETF[sectors[ticker]]].to_numpy(),
            "sector_close": close[YAHOO_TO_ETF[sectors[ticker]]].to_numpy(),
            "residual_z": residual_z[ticker].to_numpy(), "relative_volume": relative_volume[ticker].to_numpy(),
            "flow_z": flow_z[ticker].to_numpy(), "peer_confirmation": peer_confirmation[ticker].to_numpy(),
            "impact_z": impact_z[ticker].to_numpy(), "cusum": cusum[ticker].to_numpy(),
            "score": score[ticker].to_numpy(), "direction": sign[ticker].to_numpy(),
            "beta_market": beta_market[ticker].to_numpy(), "beta_sector": beta_sector[ticker].to_numpy(),
            "continuation": continuation[ticker].to_numpy(), "reversal": reversal[ticker].to_numpy(),
        })
        # Entry is at the next minute's open.  All factor betas above were fixed
        # before the signal bar; no current/future bar enters them.
        for h in (5, 15, 30, 60):
            same_session = part["session_date"].eq(part["session_date"].shift(-1)) & \
                part["session_date"].eq(part["session_date"].shift(-h))
            stock = part["close"].shift(-h) / part["open"].shift(-1) - 1
            market_r = part["market_close"].shift(-h) / part["market_open"].shift(-1) - 1
            sector_r = part["sector_close"].shift(-h) / part["sector_open"].shift(-1) - 1 - market_r
            part[f"hedged_{h}m"] = (part["direction"] *
                                     (stock - part["beta_market"] * market_r - part["beta_sector"] * sector_r)).where(same_session)
            delayed_same_session = part["session_date"].eq(part["session_date"].shift(-2)) & \
                part["session_date"].eq(part["session_date"].shift(-(h + 1)))
            stock_d = part["close"].shift(-(h + 1)) / part["open"].shift(-2) - 1
            market_d = part["market_close"].shift(-(h + 1)) / part["market_open"].shift(-2) - 1
            sector_d = part["sector_close"].shift(-(h + 1)) / part["sector_open"].shift(-2) - 1 - market_d
            part[f"hedged_{h}m_delayed"] = (part["direction"] *
                                             (stock_d - part["beta_market"] * market_d - part["beta_sector"] * sector_d)).where(delayed_same_session)
        frames.append(part)
    features = pd.concat(frames, ignore_index=True)
    metadata = {"status": "OK", "n_names": len(names), "names": names,
                "first_bar": str(close.index.min()), "last_bar": str(close.index.max())}
    return features, metadata


def run(params: Params | None = None, verbose: bool = True) -> dict:
    """Run the F1 screen and write a reproducible, non-deployable report."""
    p = params or Params()
    features, meta = build_features(p)
    if features.empty:
        return meta
    candidates = features[features["continuation"].fillna(False)].copy()
    episodes = _episode_starts(candidates, p.cooldown_minutes)
    report = {
        "status": "RESEARCH_SCREEN_ONLY",
        "engine": "latent_flow_shock_f1_bars_only",
        "params": asdict(p),
        "data": meta,
        "capabilities": {
            "implemented": ["regular-session factor residual", "rolling frozen betas", "CUSUM change point",
                            "time-of-day relative volume", "bar-direction volume proxy", "peer confirmation",
                            "VWAP-distance diagnostic", "impact proxy", "episode de-duplication",
                            "next-bar factor-hedged replay"],
            "unavailable_and_not_proxied": ["buyer/seller-initiated trade imbalance", "NBBO spread", "bid/ask queue imbalance",
                                             "microprice", "quote cancellations", "order-book depth", "options positioning",
                                             "auction imbalance", "Hawkes/lead-lag/HMM/supervised prediction"],
        },
        "warnings": ["bar-direction signed volume is not order-flow imbalance", "current 18-name stock universe is narrow and survivor-prone",
                     "this report is a frozen screening rule, not a trained continuation probability or an executable strategy",
                     "reversal flags are diagnostic only; generic intraday fade remains archived negative"],
        "candidate_minutes": int(len(candidates)), "episodes": int(len(episodes)),
        "reversal_diagnostic_minutes": int(features["reversal"].fillna(False).sum()),
        "outcomes": _outcome_report(episodes, p), "controls": _controls(episodes, features, p),
    }
    DATA_DIR.joinpath("latent_flow_f1.json").write_text(json.dumps(report, indent=2, default=str))
    if verbose:
        print(f"=== Latent Flow Shock F1 (bars only; research screen) ===\n"
              f"names={meta['n_names']} candidate minutes={len(candidates)} episodes={len(episodes)}")
        value = report["outcomes"].get("30m", {}).get("holdout", {})
        if value.get("n"):
            print(f"30m holdout: n={value['n']} mean={value['mean_bps']:+.2f}bps hit={value['hit_rate']:.1%}")
    return report


if __name__ == "__main__":
    run()
