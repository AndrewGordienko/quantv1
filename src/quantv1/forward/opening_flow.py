"""Frozen Opening Flow canary policy and its historical replay.

This is a prospective experiment, not a validated alpha model.  It uses only
information available at 10:00 New York time and deliberately has no parameter
search.  P0 is cash, P1 is the overnight-gap control, P2 adds the first
30-minute factor-residual confirmation, and P3 adds peer and relative-volume
confirmation.  The live order loop lives in ``opening_flow_live.py``.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

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

POLICY_VERSION = "opening-flow-v1"
CHAMPION_BOOK = "CASH_CHAMPION"
PROMOTION_AUTHORITY = False
POLICY = {
    "observation_close": "10:00 America/New_York",
    "entry": "next_minute_open",
    "exit": "15:50 America/New_York",
    "max_positions": 1,
    "round_trip_cost_bps": 16.0,
    "gap_multiple": 1.0,
    "residual_multiple": 0.50,
    "relative_volume_min": 1.25,
    "peer_multiple": 0.25,
    "expected_move_cost_multiple": 2.0,
    "daily_vol_window": 20,
    "beta_window": 60,
    "volume_window": 20,
}


def version() -> str:
    payload = json.dumps({"version": POLICY_VERSION, "policy": POLICY}, sort_keys=True)
    return f"{POLICY_VERSION}-{hashlib.sha256(payload.encode()).hexdigest()[:12]}"


def ensure_tables(con=None):
    own = con is None
    con = con or connect()
    con.execute("""
        CREATE TABLE IF NOT EXISTS opening_flow_decisions (
            decision_id VARCHAR PRIMARY KEY,
            decision_date DATE,
            decision_ts TIMESTAMP,
            book VARCHAR,
            policy_version VARCHAR,
            signal_as_of TIMESTAMP,
            ticker VARCHAR,
            action VARCHAR,
            side INTEGER,
            target_weight DOUBLE,
            status VARCHAR,
            entry_after TIMESTAMP,
            exit_by TIMESTAMP,
            reason JSON,
            features JSON
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS opening_flow_orders (
            order_id VARCHAR PRIMARY KEY,
            decision_id VARCHAR,
            book VARCHAR,
            ticker VARCHAR,
            side VARCHAR,
            qty DOUBLE,
            broker VARCHAR,
            status VARCHAR,
            submitted_at TIMESTAMP,
            filled_at TIMESTAMP,
            fill_price DOUBLE,
            raw JSON
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS opening_flow_marks (
            decision_id VARCHAR,
            mark_ts TIMESTAMP,
            ticker VARCHAR,
            price DOUBLE,
            pnl DOUBLE,
            exit_reason VARCHAR,
            raw JSON,
            PRIMARY KEY (decision_id, mark_ts)
        )
    """)
    if own:
        con.close()


def _regular(bars: pd.DataFrame) -> pd.DataFrame:
    b = bars.copy()
    b["ts"] = pd.to_datetime(b["ts"], utc=True)
    local = b["ts"].dt.tz_convert("America/New_York")
    minute = local.dt.hour * 60 + local.dt.minute
    b["session"] = local.dt.date
    b["slot"] = minute - 570
    return b[(local.dt.dayofweek < 5) & b["slot"].between(0, 389)].copy()


def _rolling_beta(stock: pd.Series, factor: pd.Series, window: int) -> pd.Series:
    cov = stock.rolling(window, min_periods=window).cov(factor).shift(1)
    var = factor.rolling(window, min_periods=window).var(ddof=0).shift(1)
    return cov / var.replace(0, np.nan)


def _daily_panel(prices: pd.DataFrame) -> pd.DataFrame:
    p = prices.copy()
    p["date"] = pd.to_datetime(p["date"])
    return p.pivot_table(index="date", columns="ticker", values="close").sort_index()


def _feature_rows(intraday: pd.DataFrame, daily: pd.DataFrame,
                  sectors: dict[str, str], names: list[str]) -> pd.DataFrame:
    """One row per ticker/session, using only bars through slot 29."""
    b = _regular(intraday)
    if b.empty:
        return pd.DataFrame()
    b = b[b["ticker"].isin(names + [BENCHMARK_TICKER] + list(SECTOR_ETFS))]
    dclose = _daily_panel(daily)
    # Aggregate once.  The previous prototype filtered the raw minute table
    # inside every ticker/day/peer loop, which made a two-year replay needlessly
    # expensive while producing identical features.
    b30 = b[b["slot"] <= 29].sort_values(["ticker", "session", "slot"])
    agg = (b30.groupby(["ticker", "session"], sort=False)
           .agg(open=("open", "first"), close=("close", "last"),
                volume30=("volume", "sum"), n=("close", "size"))
           .reset_index())
    agg["session"] = pd.to_datetime(agg["session"])
    agg = agg[agg["n"] >= 25]
    if agg.empty:
        return pd.DataFrame()
    agg["return30"] = agg["close"] / agg["open"] - 1
    agg["expected_volume"] = agg.groupby("ticker")["volume30"].transform(
        lambda x: x.shift(1).rolling(POLICY["volume_window"], min_periods=5).median())
    agg["relative_volume"] = agg["volume30"] / agg["expected_volume"].replace(0, np.nan)
    ret30 = agg.pivot(index="session", columns="ticker", values="return30")
    vol30 = agg.pivot(index="session", columns="ticker", values="relative_volume")
    session_index = pd.DatetimeIndex(sorted(agg["session"].unique()))
    records = []
    for ticker in names:
        sector = sectors.get(ticker)
        etf = YAHOO_TO_ETF.get(sector)
        if etf not in ret30 or BENCHMARK_TICKER not in ret30 or ticker not in ret30:
            continue
        # Daily beta estimates use only completed daily closes before this day.
        stock_daily = dclose.get(ticker, pd.Series(dtype=float)).reindex(dclose.index).pct_change()
        spy_daily = dclose.get(BENCHMARK_TICKER, pd.Series(dtype=float)).reindex(dclose.index).pct_change()
        sec_daily = dclose.get(etf, pd.Series(dtype=float)).reindex(dclose.index).pct_change()
        bm = _rolling_beta(stock_daily, spy_daily, POLICY["beta_window"])
        bs = _rolling_beta(stock_daily, sec_daily, POLICY["beta_window"])
        daily_vol = stock_daily.rolling(POLICY["daily_vol_window"], min_periods=10).std().shift(1)
        stock_rows = agg[agg["ticker"] == ticker].set_index("session")
        for day, row in stock_rows.iterrows():
            if day not in ret30.index or not all(x in ret30.columns for x in (ticker, BENCHMARK_TICKER, etf)):
                continue
            if day not in dclose.index or day not in bm.index:
                continue
            prev_close = dclose[ticker].loc[dclose.index < day].dropna()
            if prev_close.empty:
                continue
            gap = float(row.open) / float(prev_close.iloc[-1]) - 1
            rv = float(daily_vol.get(day, np.nan))
            if not np.isfinite(rv) or rv <= 0:
                continue
            r_stock = float(ret30.loc[day, ticker])
            r_spy = float(ret30.loc[day, BENCHMARK_TICKER])
            r_sec = float(ret30.loc[day, etf])
            beta_m = float(bm.get(day, np.nan))
            beta_s = float(bs.get(day, np.nan))
            if not np.isfinite(beta_m) or not np.isfinite(beta_s):
                continue
            residual = r_stock - beta_m * r_spy - beta_s * r_sec
            relvol = float(vol30.loc[day, ticker]) if ticker in vol30 else np.nan
            peers = [n for n in names if n != ticker and sectors.get(n) == sector and n in ret30]
            peer_values = [float(ret30.loc[day, n]) for n in peers if pd.notna(ret30.loc[day, n])]
            peer = float(np.mean(peer_values)) if peer_values else np.nan
            records.append({"date": day, "ticker": ticker, "sector": sector,
                            "gap": gap, "daily_vol": rv, "gap_z": gap / rv,
                            "residual": residual, "residual_z": residual / rv,
                            "relative_volume": relvol, "peer_return": peer,
                            "peer_z": peer / rv if np.isfinite(peer) else np.nan,
                            "beta_market": beta_m, "beta_sector": beta_s,
                            "liquidity_ok": True})
    return pd.DataFrame(records)


def _select(row: pd.Series, policy: str) -> bool:
    gap = abs(row.gap_z) >= POLICY["gap_multiple"]
    aligned = np.sign(row.gap) == np.sign(row.residual) and abs(row.residual_z) >= POLICY["residual_multiple"]
    # P1 has no residual feature by definition; its fixed expected-move proxy
    # is half the gap. P2/P3 use half the confirmed residual. This is a cost
    # gate, not a fitted return prediction.
    move_proxy = abs(row.gap) if policy == "P1" else abs(row.residual)
    expected = 0.5 * move_proxy >= POLICY["expected_move_cost_multiple"] * POLICY["round_trip_cost_bps"] / 1e4
    if policy == "P1":
        return bool(gap and expected and row.liquidity_ok)
    if policy == "P2":
        return bool(gap and aligned and expected and row.liquidity_ok)
    if policy == "P3":
        peer_ok = np.isfinite(row.peer_z) and np.sign(row.peer_return) == np.sign(row.gap) and abs(row.peer_z) >= POLICY["peer_multiple"]
        return bool(gap and aligned and peer_ok and row.relative_volume >= POLICY["relative_volume_min"] and expected and row.liquidity_ok)
    return False


def _replay(features: pd.DataFrame, intraday: pd.DataFrame, policy: str) -> pd.DataFrame:
    if policy == "P0" or features.empty:
        return pd.DataFrame(columns=["date", "ticker", "net"])
    b = _regular(intraday)
    chosen = []
    for date, group in features.groupby("date"):
        eligible = group[group.apply(lambda r: _select(r, policy), axis=1)].copy()
        if eligible.empty:
            continue
        eligible["rank"] = eligible["gap_z"].abs() + eligible["residual_z"].abs()
        row = eligible.sort_values(["rank", "ticker"], ascending=[False, True]).iloc[0]
        session_key = date.date() if isinstance(date, pd.Timestamp) else date
        x = b[(b["ticker"] == row.ticker) & (b["session"] == session_key)]
        entry = x[x["slot"] == 31]
        exit_ = x[x["slot"] == 380]
        if entry.empty or exit_.empty:
            continue
        side = int(np.sign(row.gap))
        gross = side * (float(exit_.iloc[0].close) / float(entry.iloc[0].open) - 1)
        net = gross - POLICY["round_trip_cost_bps"] / 1e4
        chosen.append({"date": str(date.date()), "ticker": row.ticker, "side": side,
                       "gross": gross, "net": net, "rank": float(row["rank"]),
                       "gap_z": float(row.gap_z), "residual_z": float(row.residual_z),
                       "relative_volume": float(row.relative_volume) if np.isfinite(row.relative_volume) else None})
    return pd.DataFrame(chosen)


def _metrics(trades: pd.DataFrame, dates: pd.Series | None = None) -> dict:
    if trades.empty:
        return {"n_trades": 0, "mean_net_bps": None, "hit_rate": None, "total_net": 0.0}
    net = trades["net"]
    return {"n_trades": int(len(net)), "mean_net_bps": float(net.mean() * 1e4),
            "hit_rate": float((net > 0).mean()), "total_net": float(net.sum()),
            "worst_trade_bps": float(net.min() * 1e4)}


def historical_screen(verbose: bool = True) -> dict:
    """Run the one frozen screen; no thresholds are searched or selected."""
    con = connect(read_only=True)
    sectors = dict(con.execute("SELECT ticker, sector FROM ticker_sectors").fetchall())
    available = set(r[0] for r in con.execute("SELECT DISTINCT ticker FROM bars_minute").fetchall())
    names = [t for t in available if t not in set(SECTOR_ETFS) | {BENCHMARK_TICKER}
             and YAHOO_TO_ETF.get(sectors.get(t)) in available]
    symbols = sorted(set(names) | {BENCHMARK_TICKER} | set(SECTOR_ETFS))
    placeholders = ",".join("?" for _ in symbols)
    # Pull only the first 30 minutes, the next-minute entry, and the 15:50
    # exit.  This is enough for the frozen policy/replay and avoids materializing
    # millions of irrelevant afternoon/extended-hours rows in pandas.
    # bars_minute stores UTC-naive timestamps.  Keep both possible US cash-open
    # windows (13:30 UTC in DST, 14:30 UTC in standard time), then let the
    # timezone-safe _regular() filter select the correct one for each date.
    bars = con.execute(f"""
        SELECT ticker, ts, open, high, low, close, volume
        FROM bars_minute
        WHERE ticker IN ({placeholders})
          AND ((CAST(ts AS TIME) BETWEEN TIME '13:30:00' AND TIME '14:01:00')
            OR (CAST(ts AS TIME) BETWEEN TIME '14:30:00' AND TIME '15:01:00')
            OR CAST(ts AS TIME) IN (TIME '19:50:00', TIME '20:50:00'))
    """, symbols).df()
    prices = con.execute("SELECT ticker, date, open, high, low, close, volume FROM prices "
                         "WHERE ticker IN (SELECT UNNEST(?)) ORDER BY date", [symbols]).df()
    con.close()
    bars = _regular(bars)
    features = _feature_rows(bars, prices, sectors, names)
    results = {"P0": {"n_trades": 0, "mean_net_bps": 0.0, "hit_rate": None, "total_net": 0.0}}
    candidate_counts = {p: int(features.apply(lambda r: _select(r, p), axis=1).sum())
                       if not features.empty else 0 for p in ("P1", "P2", "P3")}
    trade_rows = {}
    for policy in ("P1", "P2", "P3"):
        t = _replay(features, bars, policy)
        trade_rows[policy] = t
        results[policy] = _metrics(t)
    report = {"status": "PROSPECTIVE_POLICY_SCREEN", "policy_version": version(),
              "champion_book": CHAMPION_BOOK, "promotion_authority": PROMOTION_AUTHORITY,
              "policy": POLICY, "universe": names, "n_feature_rows": int(len(features)),
              "candidate_minutes": candidate_counts,
              "results": results,
              "data_limitations": ["historical minute bars have no NBBO spread; liquidity_ok is a bar-volume proxy",
                                    "no public-event contradiction feed is applied in this screen",
                                    "results are a fixed historical screen, not validation or a profit claim"],
              "trades": {k: v.to_dict(orient="records") for k, v in trade_rows.items()}}
    (DATA_DIR / "opening_flow_screen.json").write_text(json.dumps(report, indent=2, default=str))
    if verbose:
        print(f"Opening Flow {report['policy_version']} — feature rows={len(features)}")
        for k, v in results.items():
            print(f"  {k}: n={v['n_trades']} mean={v['mean_net_bps']}")
    return report


def record_decisions(rows: list[dict], decision_ts: datetime | None = None,
                     notional: float = 10_000.0) -> dict:
    """Append one immutable decision per book/ticker; return order candidates."""
    ensure_tables()
    ts = decision_ts or datetime.now(timezone.utc)
    con = connect()
    out = []
    for row in rows:
        book, ticker = row["book"], row.get("ticker")
        decision_id = hashlib.sha256(f"{book}|{ticker}|{ts.date()}|{POLICY_VERSION}".encode()).hexdigest()[:24]
        exists = con.execute("SELECT 1 FROM opening_flow_decisions WHERE decision_id=?", [decision_id]).fetchone()
        if exists:
            continue
        action = row.get("action", "NO TRADE")
        weight = min(float(row.get("target_weight", 0.0)), 0.10) if action != "NO TRADE" else 0.0
        con.execute("INSERT INTO opening_flow_decisions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", [
            decision_id, str(ts.date()), ts, book, row.get("policy_version", version()),
            row.get("signal_as_of", ts), ticker, action, int(row.get("side", 0)), weight,
            row.get("status", "RECORDED"), row.get("entry_after"), row.get("exit_by"),
            json.dumps(row.get("reason", {})), json.dumps(row.get("features", {}))])
        out.append({"decision_id": decision_id, "book": book, "ticker": ticker,
                    "action": action, "notional": notional * weight, "side": int(row.get("side", 0))})
    con.close()
    return {"recorded": out, "decision_ts": ts.isoformat(), "policy_version": version()}


def record_order(decision_id: str, book: str, ticker: str, side: str,
                 qty: float, result: dict, submitted_at: datetime | None = None) -> dict:
    """Record a broker submission/rejection; decisions remain immutable."""
    ensure_tables()
    ts = submitted_at or datetime.now(timezone.utc)
    broker_id = result.get("id") or f"paper-{decision_id}-{side.lower()}"
    con = connect()
    con.execute("INSERT OR REPLACE INTO opening_flow_orders VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", [
        broker_id, decision_id, book, ticker, side, qty, "alpaca_paper", result.get("status", "unknown"),
        ts, result.get("filled_at"), result.get("filled_avg_price"), json.dumps(result)])
    con.close()
    return {"order_id": broker_id, "status": result.get("status", "unknown")}
