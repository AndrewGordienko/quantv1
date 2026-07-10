"""Immutable forward paper-trading record (v2 — corrected).

Fixes from review:
  * version() content-hashes the actual strategy SOURCE files (+ git), so editing
    a rule changes the fingerprint.
  * Distinct date semantics: decision_date (real run date), signal_as_of (data
    cutoff), execution_session (next trading session; theoretical entry = its open).
  * Empty books are immutable via a decision HEADER row (recorded even with zero
    positions), keyed by (decision_date, strategy, version) so versions coexist.
  * Enters ONLY signals first observed after FORWARD_START and fresh within
    FRESH_TD — no stale historical disclosures. Pre-launch signals are excluded.
  * first_seen_at is the collector's REAL observation timestamp (from trades).
  * Full lifecycle: settle (paper fills), mark (positions, turnover, P&L,
    SPY/QQQ/beta-matched-SPY), exits (time stop) — all append-only.

Plain LARGE is the primary (paper-capital) strategy; the four discovered variants
are observational shadows with no capital and are never promoted until they win
prospectively.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import DATA_DIR, BENCHMARK_TICKER, ROOT
from ..db import connect
from ..portfolio.construct import _cap_weights

TARGET_N = 15
MAX_W = 0.08
HOLD_TD = 63
FRESH_TD = 10
LARGE_MIN = 50_000
FORWARD_START = pd.Timestamp("2026-07-10")   # frozen forward record begins here
SPREAD_BPS = 3.0
SLIPPAGE_BPS = 2.0
QQQ = "QQQ"
_JOURNAL = DATA_DIR / "forward" / "decisions.jsonl"

# Source files whose content defines the strategy — any edit changes the version.
_SOURCE_FILES = [
    ROOT / "src/quantv1/forward/tracker.py",
    ROOT / "src/quantv1/portfolio/construct.py",
]

STRATEGIES = {
    "LARGE":        {"capital": True,  "rule": lambda r: r["amount_mid"] >= LARGE_MIN},
    "LARGE_NEW":    {"capital": False, "rule": lambda r: r["amount_mid"] >= LARGE_MIN and r["is_new"]},
    "LARGE_SPOUSE": {"capital": False, "rule": lambda r: r["amount_mid"] >= LARGE_MIN and r["owner"] == "spouse"},
    "LARGE_15_30D": {"capital": False, "rule": lambda r: r["amount_mid"] >= LARGE_MIN and 15 <= (r["disclosure_lag"] or 0) <= 30},
    "LARGE_250K_1M": {"capital": False, "rule": lambda r: 250_000 <= r["amount_mid"] < 1_000_000},
}


# ---------------------------------------------------------------------------
# Version fingerprint (content hash of source + params, plus git if available)
# ---------------------------------------------------------------------------
def version() -> str:
    h = hashlib.sha1()
    for f in _SOURCE_FILES:
        try:
            h.update(f.read_bytes())
        except OSError:
            h.update(b"missing")
    h.update(json.dumps({"TARGET_N": TARGET_N, "MAX_W": MAX_W, "HOLD_TD": HOLD_TD,
                         "FRESH_TD": FRESH_TD, "LARGE_MIN": LARGE_MIN},
                        sort_keys=True).encode())
    fp = h.hexdigest()[:12]
    try:
        g = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=5)
        commit = g.stdout.strip() if g.returncode == 0 else "nogit"
    except Exception:  # noqa: BLE001
        commit = "nogit"
    return f"{fp}.{commit}"


# ---------------------------------------------------------------------------
# Calendars
# ---------------------------------------------------------------------------
def _price_cal(con):
    d = con.execute("SELECT DISTINCT date FROM prices ORDER BY date").df()["date"]
    return pd.to_datetime(d)


def _next_session(run_date: pd.Timestamp) -> pd.Timestamp:
    """Next US business day at/after run_date (holidays ignored for v1)."""
    d = run_date
    while d.weekday() >= 5:
        d += pd.Timedelta(days=1)
    return d


# ---------------------------------------------------------------------------
# Build a strategy's target book from NEWLY-OBSERVED signals only
# ---------------------------------------------------------------------------
def build_book(con, strategy: str, signal_as_of: pd.Timestamp) -> dict:
    """Only enters purchases FIRST SEEN after FORWARD_START and within FRESH_TD
    calendar-days of signal_as_of. Pre-launch/historical signals are excluded, so
    the forward record starts genuinely empty and fills as new disclosures land."""
    lo_seen = max(FORWARD_START, signal_as_of - pd.Timedelta(days=FRESH_TD * 1.5))
    rows = con.execute("""
        SELECT trade_id, member, member_key, ticker, filing_date, disclosure_lag,
               amount_mid, owner, first_seen_at
        FROM trades
        WHERE tx_type='purchase' AND ticker IS NOT NULL AND NOT filing_estimated
          AND amount_mid IS NOT NULL AND first_seen_at IS NOT NULL
          AND first_seen_at > ? AND first_seen_at >= ?
    """, [str(FORWARD_START), str(lo_seen)]).df()
    empty = {"strategy": strategy, "positions": [], "gross": 0.0, "cash": 1.0}
    if rows.empty:
        return empty

    rows["filing_date"] = pd.to_datetime(rows["filing_date"])
    prior = con.execute("""
        SELECT member_key, ticker, MIN(filing_date) AS fb
        FROM trades WHERE tx_type='purchase' AND ticker IS NOT NULL GROUP BY 1,2
    """).df()
    fb = {(r.member_key, r.ticker): pd.Timestamp(r.fb) for r in prior.itertuples(index=False)}
    rows["is_new"] = [fb.get((r.member_key, r.ticker), r.filing_date) >= r.filing_date
                      for r in rows.itertuples(index=False)]

    rule = STRATEGIES[strategy]["rule"]
    elig = rows[rows.apply(lambda r: rule(r), axis=1)]
    if elig.empty:
        return empty
    elig = (elig.sort_values(["ticker", "amount_mid"], ascending=[True, False])
            .drop_duplicates("ticker").sort_values("filing_date", ascending=False).head(TARGET_N))

    n = len(elig)
    w = _cap_weights(np.ones(n), MAX_W)
    w = w / w.sum() * min(n / TARGET_N, 1.0) if w.sum() > 0 else w
    px = dict(con.execute("SELECT ticker, arg_max(close, date) FROM prices "
                          "WHERE date <= ? GROUP BY ticker", [str(signal_as_of)]).fetchall())
    positions = []
    for wi, r in zip(w, elig.itertuples(index=False)):
        positions.append({
            "ticker": r.ticker, "weight": float(wi), "source_trade_id": r.trade_id,
            "source_member": r.member, "source_filing_date": str(r.filing_date.date()),
            "first_seen_at": str(r.first_seen_at),
            "decision_price": float(px.get(r.ticker)) if px.get(r.ticker) else None,
            "amount_mid": float(r.amount_mid), "owner": r.owner,
        })
    return {"strategy": strategy, "positions": positions,
            "gross": float(sum(p["weight"] for p in positions)),
            "cash": float(1 - sum(p["weight"] for p in positions))}


# ---------------------------------------------------------------------------
# Record decision (immutable, header-gated, version-aware)
# ---------------------------------------------------------------------------
def record_decision(run_date: str | None = None, verbose: bool = True) -> dict:
    con = connect()
    ts = datetime.now(timezone.utc)
    rd = pd.Timestamp(run_date) if run_date else pd.Timestamp(ts.date())
    signal_as_of = rd
    exec_session = _next_session(rd)
    ver = version()

    recorded = {}
    for strat in STRATEGIES:
        exists = con.execute("""SELECT COUNT(*) FROM forward_decision_headers
            WHERE decision_date=? AND strategy=? AND version=?""",
            [str(rd.date()), strat, ver]).fetchone()[0]
        if exists:
            recorded[strat] = "already recorded (immutable)"
            continue
        book = build_book(con, strat, signal_as_of)
        # header first (records even empty books)
        con.execute("INSERT INTO forward_decision_headers VALUES (?,?,?,?,?,?,?,?,?,?)",
                    [str(rd.date()), strat, ver, ts, str(signal_as_of.date()),
                     str(exec_session.date()), len(book["positions"]),
                     book["gross"], book["cash"], "RECORDED"])
        for p in book["positions"]:
            con.execute(f"INSERT INTO forward_decisions VALUES ({','.join(['?']*17)})",
                        [str(rd.date()), strat, ver, p["ticker"], ts,
                         str(signal_as_of.date()), str(exec_session.date()),
                         p["weight"], book["gross"], book["cash"], p["source_trade_id"],
                         p["source_member"], p["source_filing_date"], p["first_seen_at"],
                         p["decision_price"], None,
                         json.dumps({"amount_mid": p["amount_mid"], "owner": p["owner"]})])
        with open(_JOURNAL, "a") as f:
            f.write(json.dumps({"decision_date": str(rd.date()), "strategy": strat,
                                "version": ver, "recorded_ts": ts.isoformat(),
                                "signal_as_of": str(signal_as_of.date()),
                                "execution_session": str(exec_session.date()),
                                "status": "RECORDED", "book": book}) + "\n")
        recorded[strat] = f"{len(book['positions'])} positions, gross {book['gross']:.0%}"
    con.close()
    if verbose:
        print(f"Forward decision run={rd.date()} exec={exec_session.date()} version={ver}")
        for k, v in recorded.items():
            print(f"  [{'CAPITAL' if STRATEGIES[k]['capital'] else 'shadow '}] {k:14s} {v}")
    return {"decision_date": str(rd.date()), "version": ver,
            "execution_session": str(exec_session.date()), "recorded": recorded}


# ---------------------------------------------------------------------------
# Lifecycle: settle -> positions/exits -> mark  (append-only, inert until data)
# ---------------------------------------------------------------------------
def _open_on(con, ticker, session) -> float | None:
    r = con.execute("SELECT open FROM prices WHERE ticker=? AND date=?",
                    [ticker, str(session)]).fetchone()
    return float(r[0]) if r and r[0] is not None else None


def settle(verbose: bool = True) -> dict:
    """Record paper fills for any decision whose execution_session now has prices."""
    con = connect()
    pending = con.execute("""
        SELECT d.decision_date, d.strategy, d.ticker, d.execution_session, d.theoretical_entry
        FROM forward_decisions d
        LEFT JOIN forward_fills f
          ON d.decision_date=f.decision_date AND d.strategy=f.strategy AND d.ticker=f.ticker
        WHERE f.ticker IS NULL
    """).df()
    filled = 0
    for r in pending.itertuples(index=False):
        op = _open_on(con, r.ticker, r.execution_session)
        if op is None:
            continue                                 # session hasn't happened yet
        cost = (SPREAD_BPS + SLIPPAGE_BPS) / 1e4
        con.execute("INSERT OR REPLACE INTO forward_fills VALUES (?,?,?,?,?,?,?,?,?)",
                    [str(r.decision_date), str(r.execution_session), r.strategy, r.ticker,
                     "filled", op, op * (1 + cost), SPREAD_BPS, SLIPPAGE_BPS])
        filled += 1
    con.close()
    if verbose:
        print(f"settle: {filled} paper fills recorded")
    return {"filled": filled}


def mark(verbose: bool = True) -> dict:
    """Derive daily positions, turnover, P&L, exits and benchmarks from paper
    fills + prices. Derived tables are recomputed idempotently (INSERT OR REPLACE)
    — only DECISIONS are immutable; P&L is a deterministic function of them."""
    con = connect()
    fills = con.execute("""
        SELECT f.fill_date, f.strategy, f.ticker, f.fill_price, d.target_weight,
               d.execution_session, d.source_trade_id
        FROM forward_fills f JOIN forward_decisions d
          ON f.decision_date=d.decision_date AND f.strategy=d.strategy AND f.ticker=d.ticker
        WHERE f.status='filled'
    """).df()
    if fills.empty:
        con.close()
        if verbose:
            print("mark: no fills yet — nothing to mark")
        return {"marked_days": 0}

    close = con.execute("SELECT ticker, date, close FROM prices").df()
    close["date"] = pd.to_datetime(close["date"])
    cmat = close.pivot_table(index="date", columns="ticker", values="close")
    cal = cmat.index
    spy = cmat[BENCHMARK_TICKER] if BENCHMARK_TICKER in cmat else None
    qqq = cmat[QQQ] if QQQ in cmat else None
    fills["fill_date"] = pd.to_datetime(fills["fill_date"])
    fills["execution_session"] = pd.to_datetime(fills["execution_session"])

    marked = 0
    for strat, g in fills.groupby("strategy"):
        start = g["execution_session"].min()
        sess = cal[cal >= start]
        prev_w = {}
        equity = 1.0
        for i in range(1, len(sess)):
            s, p = sess[i], sess[i - 1]
            held = g[(g["execution_session"] <= s)]
            # exits at HOLD_TD trading days
            active = []
            for r in held.itertuples(index=False):
                entry_i = cal.searchsorted(r.execution_session)
                if i_here_hold(cal, entry_i, s) >= HOLD_TD:
                    _record_exit(con, s, strat, r, cmat, cal)
                else:
                    active.append(r)
            if not active:
                prev_w = {}
                continue
            wsum = sum(r.target_weight for r in active)
            ret = 0.0
            for r in active:
                if r.ticker in cmat and np.isfinite(cmat.at[p, r.ticker]) and np.isfinite(cmat.at[s, r.ticker]) and cmat.at[p, r.ticker] > 0:
                    ret += r.target_weight * (cmat.at[s, r.ticker] / cmat.at[p, r.ticker] - 1)
            now_w = {r.ticker: r.target_weight for r in active}
            turn = 0.5 * sum(abs(now_w.get(t, 0) - prev_w.get(t, 0))
                             for t in set(now_w) | set(prev_w))
            equity *= (1 + ret)
            spy_r = float(spy[s] / spy[p] - 1) if spy is not None and np.isfinite(spy[p]) and spy[p] > 0 else 0.0
            qqq_r = float(qqq[s] / qqq[p] - 1) if qqq is not None and np.isfinite(qqq[p]) and qqq[p] > 0 else 0.0
            con.execute("INSERT OR REPLACE INTO forward_pnl VALUES (?,?,?,?,?,?,?,?,?)",
                        [str(s.date()), strat, float(ret), float(equity), float(wsum),
                         float(turn), spy_r, qqq_r, 0.72 * spy_r])   # beta-matched SPY
            for t, w in now_w.items():
                con.execute("INSERT OR REPLACE INTO forward_positions VALUES (?,?,?,?,?)",
                            [str(s.date()), strat, t, float(w), None])
            prev_w = now_w
            marked += 1
    con.close()
    if verbose:
        print(f"mark: {marked} strategy-days marked")
    return {"marked_days": marked}


def i_here_hold(cal, entry_i, s) -> int:
    return int(cal.searchsorted(s) - entry_i)


def _record_exit(con, s, strat, r, cmat, cal):
    entry_px = r.fill_price
    exit_px = float(cmat.at[s, r.ticker]) if r.ticker in cmat and np.isfinite(cmat.at[s, r.ticker]) else None
    ret = (exit_px / entry_px - 1) if exit_px and entry_px else None
    con.execute("INSERT OR REPLACE INTO forward_exits VALUES (?,?,?,?,?,?,?)",
                [str(s.date()), strat, r.ticker, str(r.execution_session.date()),
                 "time_stop", r.source_trade_id, ret])


def evaluate() -> dict:
    con = connect(read_only=True)
    try:
        nd = con.execute("SELECT COUNT(DISTINCT date) FROM forward_pnl WHERE strategy='LARGE'").fetchone()[0]
        ne = con.execute("SELECT COUNT(*) FROM forward_exits WHERE strategy='LARGE'").fetchone()[0]
        first = con.execute("SELECT MIN(decision_date) FROM forward_decision_headers WHERE status='RECORDED'").fetchone()[0]
        heads = con.execute("SELECT COUNT(*) FROM forward_decision_headers WHERE status='RECORDED'").fetchone()[0]
    finally:
        con.close()
    months = nd / 21.0
    return {"forward_start": str(FORWARD_START.date()),
            "first_recorded_decision": str(first) if first else None,
            "decisions_recorded": int(heads),
            "trading_days_marked": int(nd), "approx_months": round(months, 1),
            "completed_positions": int(ne),
            "preliminary_review_ready": bool(months >= 6 and ne >= 50),
            "deployment_review_ready": bool(months >= 12 and ne >= 100),
            "primary_test": "active return vs beta-matched SPY (CI must exclude 0)",
            "gates": ["positive active-return CI", "DSR >= 0.95",
                      "acceptable realized costs", "no member-concentration worsening"],
            "status": ("insufficient forward data — keep recording; do NOT deploy real capital"
                       if months < 6 or ne < 50 else "preliminary review threshold reached")}


def current_book(strategy: str = "LARGE") -> dict:
    con = connect(read_only=True)
    signal_as_of = pd.Timestamp(datetime.now(timezone.utc).date())
    book = build_book(con, strategy, signal_as_of)
    con.close()
    book["version"] = version()
    book["signal_as_of"] = str(signal_as_of.date())
    book["execution_session"] = str(_next_session(signal_as_of).date())
    return book


# ---------------------------------------------------------------------------
# One-time migration: invalidate pre-launch record, add columns
# ---------------------------------------------------------------------------
def migrate(verbose: bool = True) -> None:
    import duckdb
    from ..config import DB_PATH
    con = duckdb.connect(str(DB_PATH))
    cols = [r[1] for r in con.execute("PRAGMA table_info('trades')").fetchall()]
    if "first_seen_at" not in cols:
        con.execute("ALTER TABLE trades ADD COLUMN first_seen_at TIMESTAMP")
    # drop the pre-launch (July-9) forward_decisions with the old schema
    con.execute("DROP TABLE IF EXISTS forward_decisions")
    con.execute("DROP TABLE IF EXISTS forward_decision_headers")
    con.close()
    # journal the invalidation (preserve the record, mark it invalid)
    with open(_JOURNAL, "a") as f:
        f.write(json.dumps({"marker": "PRELAUNCH_INVALID",
                            "note": "pre-launch July-9 decisions discarded; real "
                                    "forward record starts with first disclosure "
                                    "first-seen after 2026-07-10",
                            "ts": datetime.now(timezone.utc).isoformat()}) + "\n")
    connect().close()  # recreate tables with the new schema
    if verbose:
        print("migrated: trades.first_seen_at added; pre-launch forward record invalidated")


if __name__ == "__main__":
    migrate()
    record_decision()
    print()
    print(json.dumps(evaluate(), indent=2))
