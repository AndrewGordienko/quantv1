"""Interactive Brokers execution connector (PAPER first) — Canada-friendly.

Deliberately a thin, guarded wrapper: IBKR trades through a locally-running
TWS or IB Gateway, and `ib_insync` is an optional dependency (not installed by
default). This module imports it lazily and refuses to run against anything but
the PAPER port unless explicitly overridden — so it can exist in the repo now and
be wired up only when a strategy has cleared the forward-record bar.

Setup when ready:
  1. `uv add ib_insync`
  2. Install IB Gateway / TWS, log into the PAPER account, enable the API
     (default paper port 7497), add 127.0.0.1 to trusted IPs.
  3. Set IBKR_HOST/IBKR_PORT/IBKR_CLIENT_ID/IBKR_PAPER in .env.
  4. `place_book(target_book)` reads a forward book and submits paper orders,
     then fills should be reconciled back into the forward record.

This module NEVER computes a strategy — it only executes a target book that the
frozen forward tracker produced. Real-money use is gated on the pre-registered
evaluation passing.
"""

from __future__ import annotations

import os

from ..config import ROOT

PAPER_PORTS = {7497, 4002}     # TWS paper, Gateway paper


def _env():
    p = ROOT / ".env"
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
    return {
        "host": os.environ.get("IBKR_HOST", "127.0.0.1"),
        "port": int(os.environ.get("IBKR_PORT", "7497")),
        "client_id": int(os.environ.get("IBKR_CLIENT_ID", "1")),
        "paper": os.environ.get("IBKR_PAPER", "true").lower() == "true",
    }


def _connect(allow_live: bool = False):
    try:
        from ib_insync import IB
    except ImportError as e:
        raise RuntimeError("ib_insync not installed — run `uv add ib_insync` and "
                           "start IB Gateway/TWS (paper).") from e
    cfg = _env()
    if cfg["port"] not in PAPER_PORTS and not allow_live:
        raise RuntimeError(f"port {cfg['port']} is not a known PAPER port "
                           f"{PAPER_PORTS}; refusing (set allow_live=True to override).")
    ib = IB()
    ib.connect(cfg["host"], cfg["port"], clientId=cfg["client_id"], timeout=10)
    return ib, cfg


def account_summary() -> dict:
    ib, cfg = _connect()
    try:
        vals = {v.tag: v.value for v in ib.accountSummary()}
        return {"paper": cfg["paper"], "port": cfg["port"],
                "net_liquidation": vals.get("NetLiquidation"),
                "cash": vals.get("TotalCashValue"), "positions": len(ib.positions())}
    finally:
        ib.disconnect()


def place_book(target_book: dict, notional: float, dry_run: bool = True) -> dict:
    """Submit marketable-limit orders to reach the target weights (paper).

    `target_book` = {"positions": [{"ticker","weight"}...]}. With dry_run (default)
    it returns the order plan WITHOUT submitting — the safe default.
    """
    plan = []
    for pos in target_book.get("positions", []):
        plan.append({"ticker": pos["ticker"], "target_weight": pos["weight"],
                     "target_notional": round(pos["weight"] * notional, 2),
                     "side": "BUY"})
    if dry_run:
        return {"dry_run": True, "orders": plan}

    from ib_insync import Stock, MarketOrder
    ib, cfg = _connect()
    submitted = []
    try:
        for o in plan:
            contract = Stock(o["ticker"], "SMART", "USD")
            ib.qualifyContracts(contract)
            last = ib.reqMktData(contract)
            ib.sleep(1)
            px = last.last or last.close
            if not px:
                submitted.append({**o, "status": "no_price"})
                continue
            qty = int(o["target_notional"] // px)
            if qty <= 0:
                submitted.append({**o, "status": "below_min"})
                continue
            trade = ib.placeOrder(contract, MarketOrder("BUY", qty))
            submitted.append({**o, "qty": qty, "status": "submitted",
                              "order_id": trade.order.orderId})
        return {"dry_run": False, "paper": cfg["paper"], "orders": submitted}
    finally:
        ib.disconnect()


if __name__ == "__main__":
    print("IBKR connector — set up IB Gateway/TWS paper + `uv add ib_insync`, "
          "then call account_summary() / place_book(book, notional).")
