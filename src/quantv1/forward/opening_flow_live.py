"""Alpaca-data -> Opening Flow decision -> paper-order loop.

The runner is paper-only and dry-run by default.  It records a NO TRADE decision
when data, quote quality, timing, or credentials fail; it never silently turns
an unavailable feed into a BUY.  The four books are always recorded separately,
while only P3 is eligible for the canary order when ``--send-orders`` is given.
"""

from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd

from .. import net
from ..config import BENCHMARK_TICKER, DATA_DIR, ROOT
from . import opening_flow


ET = ZoneInfo("America/New_York")
DATA_URL = "https://data.alpaca.markets"
PAPER_URL = "https://paper-api.alpaca.markets/v2"
BOOKS = ("CASH_CHAMPION", "OPENING_FLOW_P1", "OPENING_FLOW_P2", "OPENING_FLOW_P3")


def _env():
    p = ROOT / ".env"
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip() and not line.lstrip().startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
    return os.environ.get("ALPACA_KEY"), os.environ.get("ALPACA_SECRET")


def _request(url: str, key: str, secret: str, params: dict | None = None,
             method: str = "GET", body: dict | None = None):
    q = f"?{urllib.parse.urlencode(params)}" if params else ""
    payload = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url + q, data=payload, method=method,
                                 headers={"APCA-API-KEY-ID": key,
                                          "APCA-API-SECRET-KEY": secret,
                                          "Content-Type": "application/json",
                                          "User-Agent": net.DEFAULT_UA})
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.load(response)


class AlpacaPaper:
    """Guarded paper broker. Live trading URLs are intentionally impossible."""

    def __init__(self, send_orders: bool = False):
        self.key, self.secret = _env()
        self.send_orders = bool(send_orders)

    @property
    def ready(self):
        return bool(self.key and self.secret)

    def snapshots(self, symbols: list[str]) -> dict:
        if not self.ready:
            raise RuntimeError("ALPACA_KEY/ALPACA_SECRET missing")
        return _request(f"{DATA_URL}/v2/stocks/snapshots", self.key, self.secret,
                        {"symbols": ",".join(symbols), "feed": "iex"})

    def bars(self, symbols: list[str], start: datetime, end: datetime,
             timeframe: str = "1Min") -> dict:
        if not self.ready:
            raise RuntimeError("ALPACA_KEY/ALPACA_SECRET missing")
        return _request(f"{DATA_URL}/v2/stocks/bars", self.key, self.secret,
                        {"symbols": ",".join(symbols), "timeframe": timeframe,
                         "start": start.astimezone(timezone.utc).isoformat(),
                         "end": end.astimezone(timezone.utc).isoformat(),
                         "limit": 10000, "feed": "iex"})

    def positions(self) -> list[dict]:
        if not self.ready:
            return []
        return _request(f"{PAPER_URL}/positions", self.key, self.secret)

    def submit_market(self, ticker: str, side: str, qty: int,
                      client_order_id: str) -> dict:
        if qty <= 0:
            return {"status": "rejected_below_minimum", "ticker": ticker}
        if not self.ready:
            return {"status": "rejected_no_credentials", "ticker": ticker}
        body = {"symbol": ticker, "qty": str(qty), "side": side,
                "type": "market", "time_in_force": "day",
                "client_order_id": client_order_id}
        if not self.send_orders:
            return {"status": "dry_run", **body}
        # PAPER_URL is hard-coded above; there is no live endpoint override.
        return {"status": "submitted", **_request(f"{PAPER_URL}/orders", self.key,
                                                     self.secret, method="POST", body=body)}

    def close(self, ticker: str) -> dict:
        if not self.ready:
            return {"status": "rejected_no_credentials", "ticker": ticker}
        if not self.send_orders:
            return {"status": "dry_run_close", "ticker": ticker}
        return {"status": "submitted_close", **_request(f"{PAPER_URL}/positions/{ticker}",
                                                           self.key, self.secret, method="DELETE")}

    def order(self, order_id: str) -> dict:
        if not self.ready:
            return {"status": "rejected_no_credentials"}
        return _request(f"{PAPER_URL}/orders/{order_id}", self.key, self.secret)


def _rows(payload: dict, kind: str) -> list[dict]:
    out = []
    for ticker, values in (payload.get(kind) or {}).items():
        for bar in values:
            out.append({"ticker": ticker, "ts": bar["t"], "open": bar["o"],
                        "high": bar.get("h", bar["o"]), "low": bar.get("l", bar["o"]),
                        "close": bar["c"], "volume": bar.get("v", 0)})
    return out


def _universe() -> tuple[list[str], dict[str, str]]:
    from ..db import connect
    con = connect(read_only=True)
    sectors = dict(con.execute("SELECT ticker, sector FROM ticker_sectors").fetchall())
    available = {r[0] for r in con.execute("SELECT DISTINCT ticker FROM bars_minute").fetchall()}
    con.close()
    names = [t for t in sorted(available) if t not in set(opening_flow.SECTOR_ETFS) | {BENCHMARK_TICKER}
             and opening_flow.YAHOO_TO_ETF.get(sectors.get(t)) in available]
    return names, sectors


def _snapshot_quotes(snapshots: dict) -> dict[str, dict]:
    quotes = {}
    for ticker, snap in snapshots.items():
        q = snap.get("latestQuote") or {}
        bid, ask = float(q.get("bp") or 0), float(q.get("ap") or 0)
        mid = (bid + ask) / 2 if bid > 0 and ask > 0 else 0
        quotes[ticker] = {"bid": bid, "ask": ask, "bid_size": q.get("bs"),
                          "ask_size": q.get("as"),
                          "mid": mid,
                          "spread_bps": (ask - bid) / mid * 1e4 if mid else None}
    return quotes


def _decision_rows(features: pd.DataFrame, now: datetime, quotes: dict[str, dict],
                   notional: float = 10_000.0,
                   blocked_tickers: set[str] | None = None) -> list[dict]:
    today = pd.Timestamp(now.astimezone(ET).date())
    rows = [{"book": "CASH_CHAMPION", "ticker": None, "action": "NO TRADE",
             "status": "RECORDED", "reason": {"control": "cash"}, "features": {}}]
    day = features[features["date"] == today].copy()
    blocked_tickers = blocked_tickers or set()
    candidates = {}
    for policy, book in (("P1", "OPENING_FLOW_P1"), ("P2", "OPENING_FLOW_P2"), ("P3", "OPENING_FLOW_P3")):
        eligible = day[day.apply(lambda r: opening_flow._select(r, policy), axis=1)].copy() if not day.empty else day
        if not eligible.empty:
            eligible = eligible[~eligible["ticker"].isin(blocked_tickers)]
            eligible["rank"] = eligible.gap_z.abs() + eligible.residual_z.abs()
            eligible = eligible.sort_values(["rank", "ticker"], ascending=[False, True])
            for r in eligible.itertuples(index=False):
                quote = quotes.get(r.ticker, {})
                if quote.get("spread_bps") is not None and quote["spread_bps"] <= 15:
                    candidates[policy] = r
                    break
        r = candidates.get(policy)
        if r is None:
            rows.append({"book": book, "ticker": None, "action": "NO TRADE", "status": "RECORDED",
                         "reason": {"policy": policy, "why": "no_candidate_or_quote_gate"}, "features": {}})
            continue
        rows.append({"book": book, "ticker": r.ticker, "action": "BUY" if r.gap > 0 else "SELL",
                     "side": 1 if r.gap > 0 else -1, "target_weight": 0.05,
                     "status": "RECORDED", "reason": {"policy": policy, "rank": float(r.rank)},
                     "features": {k: getattr(r, k) for k in ("gap_z", "residual_z", "relative_volume", "peer_z")},
                     "entry_after": now + timedelta(minutes=1),
                     "exit_by": now.astimezone(ET).replace(hour=15, minute=50, second=0, microsecond=0).astimezone(timezone.utc)})
    return rows


def run_once(now: datetime | None = None, send_orders: bool = False,
             notional: float = 10_000.0, verbose: bool = True) -> dict:
    """One 10:00 decision pass. ``send_orders`` is explicit and paper-only."""
    now = now or datetime.now(timezone.utc)
    now = now if now.tzinfo else now.replace(tzinfo=timezone.utc)
    broker = AlpacaPaper(send_orders=send_orders)
    names, sectors = _universe()
    symbols = sorted(set(names) | {BENCHMARK_TICKER} | set(opening_flow.SECTOR_ETFS))
    start = now.astimezone(ET).replace(hour=9, minute=0, second=0, microsecond=0) - timedelta(days=35)
    try:
        intraday = _rows(broker.bars(symbols, start, now), "bars")
        daily = _rows(broker.bars(symbols, start, now, timeframe="1Day"), "bars")
        intraday_df = pd.DataFrame(intraday)
        daily_df = pd.DataFrame(daily).rename(columns={"ts": "date"})
        features = opening_flow._feature_rows(intraday_df, daily_df, sectors, names)
        snaps = broker.snapshots(symbols)
        quotes = _snapshot_quotes(snaps)
        from ..db import connect
        db = connect(read_only=True)
        open_utc = now.astimezone(ET).replace(hour=9, minute=30, second=0, microsecond=0).astimezone(timezone.utc)
        event_rows = db.execute("""SELECT DISTINCT ticker FROM events
                                  WHERE ticker IS NOT NULL AND source_time BETWEEN ? AND ?""",
                               [open_utc, now]).fetchall()
        db.close()
        rows = _decision_rows(features, now, quotes, notional,
                              {r[0] for r in event_rows})
    except Exception as exc:  # data failure is an explicit recorded no-trade
        rows = [{"book": book, "ticker": None, "action": "NO TRADE", "status": "REJECTED_DATA",
                 "reason": {"error": str(exc)}, "features": {}} for book in BOOKS]
    recorded = opening_flow.record_decisions(rows, decision_ts=now, notional=notional)
    orders = []
    for row in recorded["recorded"]:
        if row["book"] != "OPENING_FLOW_P3" or row["action"] == "NO TRADE":
            continue
        mid = float((quotes.get(row["ticker"]) or {}).get("mid") or 0)
        qty = max(0, int(row["notional"] // mid)) if mid > 0 else 0
        result = broker.submit_market(row["ticker"], "buy" if row["side"] > 0 else "sell", qty,
                                      f"of-{row['decision_id']}")
        opening_flow.record_order(row["decision_id"], row["book"], row["ticker"],
                                  "BUY" if row["side"] > 0 else "SELL", qty, result, now)
        orders.append({"decision_id": row["decision_id"], "ticker": row["ticker"], **result})
    if verbose:
        print(f"Opening Flow live pass {now.isoformat()} version={recorded['policy_version']}")
        print(f"  records={len(recorded['recorded'])} paper_orders={len(orders)} send_orders={send_orders}")
    return {"recorded": recorded, "orders": orders, "send_orders": send_orders,
            "paper_only": True, "n_symbols": len(symbols)}


def close_due(now: datetime | None = None, send_orders: bool = False,
              verbose: bool = True) -> dict:
    """Close today's P3 canary by the frozen 15:50 ET hard exit."""
    now = now or datetime.now(timezone.utc)
    broker = AlpacaPaper(send_orders=send_orders)
    from ..db import connect
    con = connect(read_only=True)
    rows = con.execute("""SELECT decision_id, book, ticker FROM opening_flow_decisions
                         WHERE decision_date=? AND book='OPENING_FLOW_P3'
                           AND action <> 'NO TRADE' AND ticker IS NOT NULL""",
                       [str(now.astimezone(ET).date())]).fetchall()
    con.close()
    closed = []
    for decision_id, book, ticker in rows:
        result = broker.close(ticker)
        opening_flow.record_order(decision_id, book, ticker, "SELL", 0, result, now)
        closed.append({"decision_id": decision_id, "ticker": ticker, **result})
    if verbose:
        print(f"Opening Flow close pass: {len(closed)} positions, send_orders={send_orders}")
    return {"closed": closed, "paper_only": True, "send_orders": send_orders}


def reconcile(now: datetime | None = None, verbose: bool = True) -> dict:
    """Pull actual Alpaca paper order statuses/fills into the ledger."""
    broker = AlpacaPaper(send_orders=False)
    from ..db import connect
    con = connect(read_only=True)
    pending = con.execute("""SELECT order_id FROM opening_flow_orders
                            WHERE broker='alpaca_paper' AND status IN
                            ('accepted','new','partially_filled','submitted')""").fetchall()
    con.close()
    updated = []
    if broker.ready:
        con = connect()
        for (order_id,) in pending:
            try:
                result = broker.order(order_id)
            except Exception as exc:  # retry next minute; preserve the prior state
                updated.append({"order_id": order_id, "status": f"poll_error:{exc}"})
                continue
            con.execute("UPDATE opening_flow_orders SET status=?, filled_at=?, fill_price=?, raw=? WHERE order_id=?",
                        [result.get("status", "unknown"), result.get("filled_at"),
                         result.get("filled_avg_price"), json.dumps(result), order_id])
            updated.append({"order_id": order_id, "status": result.get("status")})
        con.close()
    if verbose:
        print(f"Opening Flow reconcile: {len(updated)} orders")
    return {"updated": updated, "paper_only": True}


def mark_once(now: datetime | None = None, verbose: bool = True) -> dict:
    """Record a minute mark-to-market row for each filled P3 canary position."""
    now = now or datetime.now(timezone.utc)
    broker = AlpacaPaper(send_orders=False)
    from ..db import connect
    con = connect(read_only=True)
    fills = con.execute("""SELECT d.decision_id, d.ticker, d.side, d.target_weight,
                                  o.fill_price
                           FROM opening_flow_decisions d JOIN opening_flow_orders o
                             ON d.decision_id=o.decision_id
                           WHERE d.book='OPENING_FLOW_P3' AND d.action <> 'NO TRADE'
                             AND o.side='BUY' AND o.fill_price IS NOT NULL""").fetchall()
    con.close()
    if not fills or not broker.ready:
        return {"marked": 0, "paper_only": True, "note": "no filled orders or credentials"}
    symbols = sorted({r[1] for r in fills})
    snapshots = broker.snapshots(symbols)
    con = connect()
    marked = []
    for decision_id, ticker, side, weight, fill_price in fills:
        snap = snapshots.get(ticker) or {}
        last = (snap.get("latestTrade") or {}).get("p")
        quote = snap.get("latestQuote") or {}
        price = float(last or ((quote.get("bp", 0) + quote.get("ap", 0)) / 2))
        if price <= 0:
            continue
        pnl = int(side) * (price / float(fill_price) - 1) * float(weight)
        con.execute("INSERT OR REPLACE INTO opening_flow_marks VALUES (?,?,?,?,?,?,?)",
                    [decision_id, now, ticker, price, pnl, "minute_mark", json.dumps(snap)])
        marked.append({"decision_id": decision_id, "ticker": ticker, "price": price, "pnl": pnl})
    con.close()
    if verbose:
        print(f"Opening Flow mark: {len(marked)} positions")
    return {"marked": len(marked), "rows": marked, "paper_only": True}


def run_loop(send_orders: bool = False, notional: float = 10_000.0) -> None:
    """Keep one process alive from pre-open through the 15:50 hard exit."""
    decision_done = False
    close_done = False
    while True:
        now = datetime.now(timezone.utc)
        local = now.astimezone(ET)
        if local.weekday() >= 5:
            print("Opening Flow loop: weekend — no market session")
            return
        if local.hour == 10 and local.minute == 0 and not decision_done:
            run_once(now=now, send_orders=send_orders, notional=notional)
            decision_done = True
        elif (local.hour > 10 or (local.hour == 10 and local.minute >= 1)) and (local.hour < 15 or (local.hour == 15 and local.minute < 50)):
            reconcile(now=now, verbose=False)
            mark_once(now=now, verbose=False)
        elif local.hour == 15 and local.minute >= 50 and not close_done:
            close_due(now=now, send_orders=send_orders)
            reconcile(now=now, verbose=False)
            close_done = True
            return
        time.sleep(max(1, 60 - local.second))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--send-orders", action="store_true", help="submit to hard-coded Alpaca PAPER endpoint")
    parser.add_argument("--close", action="store_true", help="submit the frozen 15:50 ET close pass")
    parser.add_argument("--reconcile", action="store_true")
    parser.add_argument("--mark", action="store_true")
    parser.add_argument("--loop", action="store_true", help="stay alive through the 15:50 ET close")
    parser.add_argument("--notional", type=float, default=10_000.0)
    args = parser.parse_args()
    if args.loop:
        run_loop(send_orders=args.send_orders, notional=args.notional)
        result = {"status": "loop_finished", "paper_only": True}
    elif args.close:
        result = close_due(send_orders=args.send_orders)
    elif args.reconcile:
        result = reconcile()
    elif args.mark:
        result = mark_once()
    else:
        result = run_once(send_orders=args.send_orders, notional=args.notional)
    print(json.dumps(result, indent=2, default=str))
