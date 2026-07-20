"""Immutable paper-forward record for the frozen TSMOM overlay (PAPER ONLY).

The one honest positive from the 2026-07-20 sweep (robust modest diversifying
overlay, net Sharpe ~0.5-0.66, sub-gate). This stands up a prospective,
append-only record so out-of-sample track accrues day by day. No real money.

Discipline:
  * Rules are FROZEN and imported verbatim from scripts/tsmom_etf_diag.py; the
    spec pins a sha256 of that source so the ruleset is verifiable.
  * The prospective OUTCOME window starts on/after 2026-07-21 (PROSPECTIVE_START).
    The day-0 decision is pre-registered (ARMED) and executes at the next XNYS
    session open on/after PROSPECTIVE_START; NO return is recorded before it.
  * Append-only: `arm` refuses to re-arm; `mark` never overwrites a marked date.
  * Execution assumption: enter at next-session open, per-side cost 2 bps, marks
    hold the last decision's target weights (monthly rebalance cadence).

Commands:
  arm   -- freeze spec + pre-register the day-0 target book (run once)
  mark  -- append daily marks for sessions on/after execution (run each session)

Artifacts (committed, auditable):
  goldset/tsmom_paper/tsmom_spec_v1.json
  goldset/tsmom_paper/decisions.jsonl
  goldset/tsmom_paper/marks.jsonl
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass
import yfinance as yf

from quantv1.config import ROOT
import importlib.util

# import the FROZEN rules verbatim
_spec = importlib.util.spec_from_file_location(
    "tsmom_diag", str(ROOT / "scripts" / "tsmom_etf_diag.py"))
_d = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_d)

PAPER = ROOT / "goldset" / "tsmom_paper"
SPEC = PAPER / "tsmom_spec_v1.json"
DECISIONS = PAPER / "decisions.jsonl"
MARKS = PAPER / "marks.jsonl"
PROSPECTIVE_START = "2026-07-21"     # genuine prospective record begins here
COST_BPS_PER_SIDE = 2.0
SPEC_VERSION = "tsmom-etf-paper-v1"


def _rules_hash() -> str:
    return hashlib.sha256((ROOT / "scripts" / "tsmom_etf_diag.py").read_bytes()).hexdigest()


def _next_session_on_or_after(d: str) -> str:
    """Next XNYS trading session on/after date d (fallback: next weekday)."""
    start = date.fromisoformat(d)
    try:
        import exchange_calendars as xcals
        cal = xcals.get_calendar("XNYS")
        sessions = cal.sessions_in_range(d, (start + timedelta(days=10)).isoformat())
        if len(sessions):
            return pd.Timestamp(sessions[0]).date().isoformat()
    except Exception:
        pass
    while start.weekday() >= 5:
        start += timedelta(days=1)
    return start.isoformat()


def _fetch(today: str) -> dict:
    """Return {'open':df,'close':df} for the basket+SPY, through <= today (inclusive)."""
    tickers = _d.INSTRUMENTS + ["SPY"]
    df = yf.download(tickers, period="2y", auto_adjust=True, progress=False)
    lvl0 = df.columns.get_level_values(0)
    out = {}
    for field in ("Open", "Close"):
        f = df[field] if field in lvl0 else df
        out[field.lower()] = f.dropna(how="all").sort_index()[lambda x: x.index <= pd.Timestamp(today)]
    return out


def _target_weights(close: pd.DataFrame, as_of_ts: pd.Timestamp | None = None) -> tuple[dict, str]:
    """FROZEN combined-lookback vol-targeted TSMOM weights as of a given close date
    (default: last available close)."""
    inst = [t for t in _d.INSTRUMENTS if t in close.columns]
    px = close[inst]
    if as_of_ts is not None:
        px = px[px.index <= as_of_ts]
    rets = px.pct_change()
    vol = rets.rolling(_d.VOL_WIN, min_periods=_d.VOL_WIN // 2).std() * np.sqrt(_d.ANN)
    sig = sum(np.sign(px / px.shift(L) - 1.0) for L in _d.LOOKBACKS) / len(_d.LOOKBACKS)
    scale = (_d.TARGET_VOL / vol).clip(upper=_d.MAX_LEV)
    w = (sig * scale).iloc[-1] / len(_d.INSTRUMENTS)
    as_of = px.index[-1].date().isoformat()
    weights = {t: round(float(w[t]), 5) for t in inst if pd.notna(w[t])}
    return weights, as_of


def arm() -> None:
    if DECISIONS.exists() and DECISIONS.read_text().strip():
        print("ALREADY ARMED — decisions.jsonl exists; refusing to re-arm.")
        return
    PAPER.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    close = _fetch(today)["close"]
    weights, as_of = _target_weights(close)
    ref_close = {t: round(float(close[t].loc[close.index[-1]]), 4) for t in weights}
    exec_session = _next_session_on_or_after(PROSPECTIVE_START)
    gross_long = round(sum(v for v in weights.values() if v > 0), 4)
    gross_short = round(-sum(v for v in weights.values() if v < 0), 4)

    spec = {
        "spec_version": SPEC_VERSION, "capital": "PAPER_ONLY_NO_REAL_MONEY",
        "frozen_at": today, "prospective_start": PROSPECTIVE_START,
        "rules_source": "scripts/tsmom_etf_diag.py", "rules_code_sha256": _rules_hash(),
        "universe": _d.INSTRUMENTS, "basket_by_class": _d.BASKET,
        "lookbacks_days": _d.LOOKBACKS, "signal": "mean sign of cumulative return over lookbacks",
        "vol_target_per_instrument": _d.TARGET_VOL, "max_leverage_per_instrument": _d.MAX_LEV,
        "vol_window_days": _d.VOL_WIN, "weighting": "averaged across N instruments (equal notional slots)",
        "rebalance": "monthly (last trading day); initialized at prospective_start",
        "execution": "enter at next XNYS session open after decision",
        "cost_bps_per_side": COST_BPS_PER_SIDE, "benchmark": "SPY",
        "price_source": "yfinance auto_adjust (total-return proxy)",
    }
    SPEC.write_text(json.dumps(spec, indent=2, sort_keys=True))

    decision = {
        "decision_date": today, "signal_as_of": as_of, "execution_session": exec_session,
        "status": "ARMED_PRELAUNCH", "rules_code_sha256": _rules_hash(),
        "target_weights": weights, "reference_close": ref_close,
        "gross_long": gross_long, "gross_short": gross_short,
        "n_long": sum(1 for v in weights.values() if v > 0),
        "n_short": sum(1 for v in weights.values() if v < 0),
        "note": ("frozen day-0 book, pre-registered before the prospective window; "
                 "NO return recorded until execution_session open. PAPER ONLY."),
    }
    with open(DECISIONS, "a") as f:
        f.write(json.dumps(decision, sort_keys=True) + "\n")
    print(f"ARMED. spec frozen (rules {_rules_hash()[:12]}).")
    print(f"  signal_as_of={as_of}  execution={exec_session}  "
          f"gross L/S={gross_long}/{gross_short}  n L/S={decision['n_long']}/{decision['n_short']}")
    print("  target weights:", {k: v for k, v in sorted(weights.items(), key=lambda x: -abs(x[1]))})
    print(f"  wrote {SPEC.name}, {DECISIONS.name}")


def _load(p):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def run() -> None:
    """Idempotent, append-only: apply any due monthly rebalances, then mark all
    FINALIZED sessions since the first execution. Correctness fixes vs v0:
      * entry/rebalance day return = OPEN->CLOSE (we enter at the session open),
        NOT close->close (which wrongly included the pre-entry overnight move);
      * subtract turnover cost (cost_bps/side) on entry and each rebalance;
      * only mark sessions strictly BEFORE today, so an in-progress/unfinalized
        bar can never be appended to the immutable record.
    """
    decisions = _load(DECISIONS)
    if not decisions:
        print("NOT ARMED — run `arm` first.")
        return
    today = date.today().isoformat()
    px = _fetch(today)
    close, open_ = px["close"], px["open"]
    fin = close.index[close.index < pd.Timestamp(today)]        # FINALIZED sessions only
    first_exec = decisions[0]["execution_session"]
    if len(fin) == 0 or fin[-1].date().isoformat() < first_exec:
        print(f"PRE-LAUNCH / no finalized session on-or-after {first_exec} yet (today {today}).")
        return

    # --- due monthly rebalances: at each month-end finalized session, target as-of
    #     its close, executed at the next session open (append a REBALANCE decision) ---
    dec_execs = {d["execution_session"] for d in decisions}
    month_ends = pd.Series(fin, index=fin).groupby([fin.year, fin.month]).last().tolist()
    for me in month_ends:
        if me.date().isoformat() < first_exec:
            continue
        nxt = close.index[close.index > me]
        if len(nxt) == 0:
            continue
        exec_next = nxt[0].date().isoformat()
        if exec_next <= first_exec or exec_next in dec_execs:
            continue
        w, asof = _target_weights(close, as_of_ts=me)
        rec = {"decision_date": me.date().isoformat(), "signal_as_of": asof,
               "execution_session": exec_next, "status": "REBALANCE",
               "rules_code_sha256": _rules_hash(), "target_weights": w,
               "note": "monthly rebalance (last trading day of month)"}
        with open(DECISIONS, "a") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
        decisions.append(rec)
        dec_execs.add(exec_next)

    # --- mark finalized sessions using the book active on each date ---
    books = sorted(decisions, key=lambda d: d["execution_session"])
    marked = {m["date"] for m in _load(MARKS)}
    cost = COST_BPS_PER_SIDE / 1e4
    new = 0
    for i, bk in enumerate(books):
        w = pd.Series(bk["target_weights"], dtype=float)
        inst = [t for t in w.index if t in close.columns]
        w = w.reindex(inst).fillna(0.0)
        w_prev = pd.Series(books[i - 1]["target_weights"], dtype=float) if i > 0 else pd.Series(dtype=float)
        turnover = float((w - w_prev.reindex(inst).fillna(0.0)).abs().sum())
        start = bk["execution_session"]
        end = books[i + 1]["execution_session"] if i + 1 < len(books) else None
        for d in fin:
            di = d.date().isoformat()
            if di < start or (end and di >= end) or di in marked:
                continue
            if di == start:                                     # entry/rebalance day: open->close
                if d not in open_.index:
                    continue
                r = float((w * (close.loc[d, inst] / open_.loc[d, inst] - 1.0)).sum()) - turnover * cost
                spy = float(close.loc[d, "SPY"] / open_.loc[d, "SPY"] - 1.0) if d in open_.index else None
                first = True
            else:                                               # holding day: close->close
                r = float((w * close[inst].pct_change().loc[d]).sum())
                spy = float(close["SPY"].pct_change().loc[d])
                first = False
            with open(MARKS, "a") as f:
                f.write(json.dumps({"date": di, "strategy": SPEC_VERSION, "ret": round(r, 6),
                                    "spy_ret": round(spy, 6) if spy is not None else None,
                                    "entry_or_rebalance_day": first, "book_execution": start},
                                   sort_keys=True) + "\n")
            marked.add(di)
            new += 1
    tot = _load(MARKS)
    equity = float(np.prod([1 + m["ret"] for m in tot])) if tot else 1.0
    print(f"marked {new} new session(s); {len(tot)} total; paper equity index {equity:.4f}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    {"arm": arm, "run": run, "mark": run}.get(cmd, run)()
