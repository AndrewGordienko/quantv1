"""Crypto TSMOM (BTC/ETH) paper-forward — immutable OOS record (PAPER ONLY).

Advances the shortlisted crypto-TSMOM candidate (trial crypto_tsmom_btc_eth_v1)
to a prospective record so the backtest Sharpe ~1.07 must now prove itself out of
sample. Daily trend overlay, NOT a day-trader. No real money.

Discipline (mirrors scripts/tsmom_paper_forward.py):
  * rules frozen + pinned to scripts/crypto_tsmom_backtest.py sha256;
  * the strategy is daily-rebalanced, so each finalized UTC daily bar is marked
    with the weight decided at the PRIOR finalized bar (no look-ahead);
  * marks include taker+slippage cost on turnover and funding (long pays when
    funding>0); only FINALIZED bars (date < today UTC) are ever marked, so an
    in-progress 24/7 day is never appended;
  * prospective outcomes only on/after PROSPECTIVE_START (2026-07-21);
  * append-only, idempotent.

Commands: arm (freeze + day-0 book, once) · run (mark finalized bars).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from quantv1.config import ROOT
from quantv1.ingest import crypto_perp as cp

_spec = importlib.util.spec_from_file_location(
    "cbt", str(ROOT / "scripts" / "crypto_tsmom_backtest.py"))
_b = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_b)

PAPER = ROOT / "goldset" / "crypto_tsmom_paper"
SPEC = PAPER / "spec_v1.json"
DECISIONS = PAPER / "decisions.jsonl"
MARKS = PAPER / "marks.jsonl"
PROSPECTIVE_START = "2026-07-21"
SPEC_VERSION = "crypto-tsmom-btc-eth-paper-v1"


def _rules_hash() -> str:
    return hashlib.sha256((ROOT / "scripts" / "crypto_tsmom_backtest.py").read_bytes()).hexdigest()


def _load(p):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def _weights(close: pd.Series) -> pd.Series:
    """Frozen crypto TSMOM weight series (same formula as the backtest sleeve)."""
    ret = close.pct_change()
    vol = ret.rolling(_b.VOL_WIN, min_periods=_b.VOL_WIN // 2).std() * np.sqrt(_b.ANN)
    sig = sum(np.sign(close / close.shift(L) - 1.0) for L in _b.LOOKBACKS) / len(_b.LOOKBACKS)
    return (sig * (_b.TARGET_VOL / vol).clip(upper=_b.MAX_LEV)).clip(-_b.MAX_LEV, _b.MAX_LEV)


def _fetch():
    """Fresh BTC/ETH daily close + daily funding; return dict sym -> (close, funding_daily)."""
    out = {}
    for s in _b.SYMBOLS:
        k = cp.fetch_klines(s).set_index("date")["close"].astype(float).sort_index()
        f = cp.fetch_funding(s)
        fd = f.set_index("ts")["funding_rate"].groupby(pd.Grouper(freq="D")).sum()
        fd.index = fd.index.normalize()
        out[s] = (k, fd)
    return out


def _finalized(idx: pd.DatetimeIndex, today_iso: str) -> pd.DatetimeIndex:
    return idx[idx < pd.Timestamp(today_iso)]


def arm() -> None:
    if DECISIONS.exists() and DECISIONS.read_text().strip():
        print("ALREADY ARMED — refusing to re-arm.")
        return
    PAPER.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    data = _fetch()
    book, refs = {}, {}
    as_of = None
    for s, (close, _) in data.items():
        fin = _finalized(close.index, today)
        w = _weights(close.loc[fin]).iloc[-1]
        book[s] = round(float(w), 5)
        refs[s] = round(float(close.loc[fin].iloc[-1]), 2)
        as_of = fin[-1].date().isoformat()
    SPEC.write_text(json.dumps({
        "spec_version": SPEC_VERSION, "capital": "PAPER_ONLY_NO_REAL_MONEY",
        "frozen_at": today, "prospective_start": PROSPECTIVE_START,
        "rules_source": "scripts/crypto_tsmom_backtest.py", "rules_code_sha256": _rules_hash(),
        "universe": _b.SYMBOLS, "lookbacks": _b.LOOKBACKS, "target_vol": _b.TARGET_VOL,
        "max_lev": _b.MAX_LEV, "vol_window": _b.VOL_WIN, "rebalance": "daily",
        "execution": "next daily bar", "fee_slippage_bps_per_side": _b.FEE_SLIP_BPS,
        "funding": "modeled (long pays when funding>0)",
        "note": "daily trend overlay, not a day-trader; PAPER ONLY",
    }, indent=2, sort_keys=True))
    dec = {"decision_date": today, "signal_as_of": as_of,
           "execution_session": PROSPECTIVE_START, "status": "ARMED_PRELAUNCH",
           "rules_code_sha256": _rules_hash(), "target_weights": book,
           "reference_close": refs,
           "note": "frozen day-0 book; daily-rebalanced going forward; NO return before execution."}
    with open(DECISIONS, "a") as f:
        f.write(json.dumps(dec, sort_keys=True) + "\n")
    print(f"ARMED. rules {_rules_hash()[:12]}  signal_as_of={as_of}  exec={PROSPECTIVE_START}")
    print(f"  day-0 book: {book}")


def run() -> None:
    if not _load(DECISIONS):
        print("NOT ARMED — run `arm` first.")
        return
    exec_session = _load(DECISIONS)[0]["execution_session"]
    today = date.today().isoformat()
    data = _fetch()
    marked = {m["date"] for m in _load(MARKS)}
    cost = _b.FEE_SLIP_BPS / 1e4
    # per-symbol daily net series (finalized only), then equal-weight portfolio
    sleeve = {}
    for s, (close, fd) in data.items():
        fin = _finalized(close.index, today)
        c = close.loc[fin]
        ret = c.pct_change()
        w = _weights(c)
        wp = w.shift(1)
        turn = wp.diff().abs()
        fnd = fd.reindex(c.index).fillna(0.0)
        sleeve[s] = (wp * ret) - (turn * cost) + (-wp * fnd)
    port = pd.concat(sleeve.values(), axis=1).mean(axis=1)
    new = 0
    for d, r in port.items():
        di = d.date().isoformat()
        if di < exec_session or di in marked or pd.isna(r):
            continue
        with open(MARKS, "a") as f:
            f.write(json.dumps({"date": di, "strategy": SPEC_VERSION,
                                "net_ret": round(float(r), 6)}, sort_keys=True) + "\n")
        new += 1
    tot = _load(MARKS)
    eq = float(np.prod([1 + m["net_ret"] for m in tot])) if tot else 1.0
    print(f"marked {new} finalized bar(s); {len(tot)} total; paper equity {eq:.4f}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "arm"
    {"arm": arm, "run": run, "mark": run}.get(cmd, arm)()
