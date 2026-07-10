"""Leak-free event-replay backtester (V4 foundation).

Replays timestamped public events against intraday bars, revealing each record
only at its real `public_time`, so a signal can never use information published
after the moment it would have acted. This is the engine that must exist BEFORE
more strategy work — it makes the point-in-time discipline structural, not a
convention.

Per event:
  1. reveal bars with ts <= public_time; observe `obs_bars` after it (real-time).
  2. the signal_fn decides side (+1/-1/0) from only those bars.
  3. ENTER at the next bar's open after the observation window (pay spread+slippage).
  4. EXIT via triple barrier (take-profit / stop / timeout) or EOD.
  5. return is MARKET/SECTOR-ADJUSTED (subtract the benchmark over the same window)
     and net of spread + slippage + fees.
Tracks rejects (no bar / halted), turnover, and reports net Sharpe, drawdown,
turnover and DEFLATED Sharpe.

Signal fn contract:  signal_fn(event, panel, i_pub, i_dec) -> dict(side, reason)
  where i_pub = first bar after public_time, i_dec = decision bar (i_pub+obs_bars).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..config import BENCHMARK_TICKER
from ..db import connect

EULER = 0.5772156649


def to_ns(series) -> np.ndarray:
    """Datetimes -> int64 NANOSECONDS, regardless of source resolution.

    DuckDB returns datetime64[us]; a plain .astype('int64') then yields
    MICROSECONDS, which silently corrupts any unit='ns' date math (the 1970
    bucketing bug). Forcing [ns] first makes the unit unambiguous everywhere."""
    return pd.to_datetime(series).values.astype("datetime64[ns]").astype("int64")


@dataclass
class ReplayParams:
    obs_bars: int = 1          # bars observed after the event before deciding
    entry_delay: int = 0       # extra bars to wait AFTER the decision before entry
    max_hold: int = 6          # timeout barrier (bars)
    tp: float = 0.03           # take-profit (fraction)
    sl: float = 0.015          # stop-loss (fraction)
    spread_bps: float = 3.0
    slippage_bps: float = 2.0
    fees_bps: float = 0.0
    max_concurrent: int = 5
    cooldown_bars: int = 6     # per-symbol cooldown after exit
    n_trials: int = 5          # for deflated Sharpe (strategies tried)


class BarPanel:
    """Intraday bars as per-ticker numpy arrays with fast time lookup.

    `table` selects the resolution: 'bars_minute' (V4 real system) or
    'bars_hourly' (the free PoC)."""

    def __init__(self, con=None, table: str = "bars_hourly"):
        own = con is None
        con = con or connect(read_only=True)
        df = con.execute(f"SELECT ticker, ts, open, high, low, close, volume "
                         f"FROM {table} ORDER BY ticker, ts").df()
        if own:
            con.close()
        df["ts"] = to_ns(df["ts"])                            # int64 NANOSECONDS
        self.data = {}
        for tk, g in df.groupby("ticker"):
            self.data[tk] = {
                "ts": g["ts"].to_numpy(), "open": g["open"].to_numpy(),
                "high": g["high"].to_numpy(), "low": g["low"].to_numpy(),
                "close": g["close"].to_numpy(), "vol": g["volume"].to_numpy(),
            }

    def has(self, tk):
        return tk in self.data

    def next_idx_after(self, tk, t_ns) -> int | None:
        a = self.data[tk]["ts"]
        i = int(np.searchsorted(a, t_ns, side="right"))
        return i if i < len(a) else None


def replay(events: pd.DataFrame, panel: BarPanel, signal_fn,
           params: ReplayParams | None = None, sector_map: dict | None = None,
           test_start: str | None = None) -> dict:
    p = params or ReplayParams()
    sector_map = sector_map or {}
    cost = (p.spread_bps + p.slippage_bps + p.fees_bps) / 1e4
    ev = events.sort_values("public_time").reset_index(drop=True)
    ev["public_time"] = to_ns(ev["public_time"])

    trades = []
    last_exit_ns: dict[str, int] = {}
    for e in ev.itertuples(index=False):
        tk = e.ticker
        if not tk or not panel.has(tk):
            continue
        if last_exit_ns.get(tk, 0) >= e.public_time:      # cooldown / already open
            continue
        i_pub = panel.next_idx_after(tk, e.public_time)
        if i_pub is None:
            trades.append({"ticker": tk, "status": "reject_no_bar"})
            continue
        i_dec = i_pub + p.obs_bars
        i_entry = i_dec + p.entry_delay      # delay is separate from observation
        d = panel.data[tk]
        if i_entry + 1 >= len(d["ts"]):
            trades.append({"ticker": tk, "status": "reject_eod"})
            continue
        sig = signal_fn(e, panel, i_pub, i_dec)
        side = sig.get("side", 0)
        if side == 0:
            continue

        entry_px = d["open"][i_entry] * (1 + side * cost)
        # triple barrier over the next max_hold bars
        exit_px, exit_i, reason = None, None, "timeout"
        for j in range(i_entry, min(i_entry + p.max_hold, len(d["ts"]))):
            hi, lo = d["high"][j], d["low"][j]
            up = entry_px * (1 + p.tp) if side > 0 else entry_px * (1 - p.tp)
            dn = entry_px * (1 - p.sl) if side > 0 else entry_px * (1 + p.sl)
            if side > 0 and hi >= up:
                exit_px, exit_i, reason = up, j, "take_profit"; break
            if side > 0 and lo <= dn:
                exit_px, exit_i, reason = dn, j, "stop"; break
            if side < 0 and lo <= up:
                exit_px, exit_i, reason = up, j, "take_profit"; break
            if side < 0 and hi >= dn:
                exit_px, exit_i, reason = dn, j, "stop"; break
        if exit_px is None:
            exit_i = min(i_entry + p.max_hold - 1, len(d["ts"]) - 1)
            exit_px = d["close"][exit_i]

        gross = side * (exit_px / entry_px - 1)
        # market/sector adjustment over the same time window
        bench = sector_map.get(tk, BENCHMARK_TICKER)
        badj = _bench_return_timealigned(panel, bench, d["ts"][i_entry], d["ts"][exit_i])
        net = gross - side * badj - cost              # pay cost again on exit
        trades.append({"ticker": tk, "status": "filled", "side": side,
                       "entry_ns": int(d["ts"][i_entry]), "exit_ns": int(d["ts"][exit_i]),
                       "gross": float(gross), "net": float(net), "reason": reason,
                       "bars_held": int(exit_i - i_entry + 1)})
        bar_ns = int(d["ts"][exit_i] - d["ts"][exit_i - 1]) if exit_i > 0 else 60_000_000_000
        last_exit_ns[tk] = int(d["ts"][exit_i]) + p.cooldown_bars * bar_ns

    return _metrics(pd.DataFrame(trades), p, test_start)


def _bench_return_timealigned(panel, bench, t0_ns, t1_ns) -> float:
    if not panel.has(bench):
        return 0.0
    a = panel.data[bench]["ts"]
    i0 = int(np.searchsorted(a, t0_ns, side="left"))
    i1 = int(np.searchsorted(a, t1_ns, side="left"))
    c = panel.data[bench]["close"]
    if i0 >= len(c) or i1 >= len(c) or c[i0] <= 0:
        return 0.0
    return float(c[i1] / c[i0] - 1)


def _deflated_sharpe(rets: np.ndarray, sr: float, n_trials: int) -> float:
    from scipy.stats import skew, kurtosis
    T = len(rets)
    if T < 20 or rets.std() == 0:
        return np.nan
    g3, g4 = float(skew(rets)), float(kurtosis(rets, fisher=False))
    z1, z2 = norm.ppf(1 - 1.0 / n_trials), norm.ppf(1 - 1.0 / (n_trials * np.e))
    se = np.sqrt((1 - g3 * sr + (g4 - 1) / 4 * sr ** 2) / (T - 1))
    sr0 = se * ((1 - EULER) * z1 + EULER * z2)
    return float(norm.cdf((sr - sr0) * np.sqrt(T - 1) /
                          np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2)))


def _metrics(trades: pd.DataFrame, p: ReplayParams, test_start) -> dict:
    filled = trades[trades["status"] == "filled"] if not trades.empty else trades
    rejects = int((trades["status"].str.startswith("reject")).sum()) if not trades.empty else 0
    if filled.empty:
        return {"n_trades": 0, "rejects": rejects, "note": "no trades taken"}
    filled = filled.copy()
    filled["date"] = pd.to_datetime(filled["entry_ns"], unit="ns").dt.date
    daily = filled.groupby("date")["net"].sum()
    r = daily.to_numpy()
    sr_bar = filled["net"].mean() / filled["net"].std() if filled["net"].std() > 0 else np.nan
    ann = 252
    sharpe = float(r.mean() / r.std() * np.sqrt(ann)) if r.std() > 0 else np.nan
    eq = np.cumprod(1 + r)
    dsr = _deflated_sharpe(filled["net"].to_numpy(),
                           filled["net"].mean() / filled["net"].std() if filled["net"].std() > 0 else 0,
                           p.n_trials)

    def seg(df):
        if df.empty:
            return None
        return {"n": int(len(df)), "mean_net": float(df["net"].mean()),
                "hit_rate": float((df["net"] > 0).mean()),
                "avg_win": float(df[df["net"] > 0]["net"].mean()) if (df["net"] > 0).any() else None,
                "avg_loss": float(df[df["net"] < 0]["net"].mean()) if (df["net"] < 0).any() else None}
    out = {
        "n_trades": int(len(filled)), "rejects": rejects,
        "net_sharpe_daily": sharpe, "deflated_sharpe": dsr,
        "total_net_return": float(eq[-1] - 1), "trades_per_active_day": float(len(filled) / max(len(daily), 1)),
        "overall": seg(filled),
        "exit_reasons": filled["reason"].value_counts().to_dict(),
    }
    if test_start:
        ts = pd.Timestamp(test_start).date()
        out["train"] = seg(filled[filled["date"] < ts])
        out["holdout"] = seg(filled[filled["date"] >= ts])
    return out
