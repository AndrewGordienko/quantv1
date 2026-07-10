"""Flagship experiment: does INSIDER confirmation improve a congressional buy?

The methodology review's clearest lawful thesis:

    large or repeated congressional purchase + a corroborating public signal +
    limited prior price reaction  >>  "a politician bought stock".

Here the corroborating signal is a corporate insider's OPEN-MARKET purchase
(Form 4 code P) in the same name near the congressional disclosure. We compare
the factor-adjusted 63-day CAR of congressional purchases that ARE vs ARE NOT
insider-confirmed, using the leak-free machinery: next-open CARs from
`event_v2`, cluster-robust block-bootstrap CIs, and a locked 2024+ holdout.

Confirmation is point-in-time: an insider filing only counts if its public
`source_time` is within CONFIRM_WINDOW days of the congressional filing.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..config import DATA_DIR
from ..db import connect
from .event_study_v2 import _cluster_bootstrap, HOLDOUT_START

CONFIRM_WINDOW = 30     # days between insider filing and congress filing
CAR = "car_ff_63"


def _insider_index(con) -> dict:
    rows = con.execute("""
        SELECT ticker, source_time FROM events WHERE layer='F' AND event_type='insider_buy'
    """).fetchall()
    idx: dict[str, list] = {}
    for tk, st in rows:
        idx.setdefault(tk, []).append(pd.Timestamp(st))
    for tk in idx:
        idx[tk].sort()
    return idx


def _confirmed(idx: dict, ticker: str, when: pd.Timestamp, window: int) -> bool:
    lst = idx.get(ticker)
    if not lst:
        return False
    lo, hi = when - pd.Timedelta(days=window), when + pd.Timedelta(days=window)
    import bisect
    i = bisect.bisect_left(lst, lo)
    return i < len(lst) and lst[i] <= hi


def run(verbose: bool = True) -> dict:
    con = connect(read_only=True)
    tables = [t[0] for t in con.execute("SHOW TABLES").fetchall()]
    if "event_v2" not in tables:
        con.close()
        raise RuntimeError("run event_study_v2 first (need event_v2 table)")
    ev = con.execute(f"""
        SELECT trade_id, member_key, ticker, filing_date, amount_mid, is_repeat,
               mom20, {CAR} AS car
        FROM event_v2 WHERE {CAR} IS NOT NULL
    """).df()
    idx = _insider_index(con)
    n_insider = con.execute("SELECT COUNT(*) FROM events WHERE layer='F'").fetchone()[0]
    con.close()

    ev["filing_date"] = pd.to_datetime(ev["filing_date"])
    ev["confirmed"] = [
        _confirmed(idx, r.ticker, r.filing_date, CONFIRM_WINDOW)
        for r in ev.itertuples(index=False)
    ]
    ev["is_large"] = ev["amount_mid"] >= 50_000

    def compare(df: pd.DataFrame, label: str) -> dict:
        conf = df[df["confirmed"]]
        unconf = df[~df["confirmed"]]
        return {
            "label": label,
            "confirmed": _cluster_bootstrap(conf, "car", "ticker"),
            "unconfirmed": _cluster_bootstrap(unconf, "car", "ticker"),
            "confirmed_by_member": _cluster_bootstrap(conf, "car", "member_key"),
        }

    train = ev[ev["filing_date"] < HOLDOUT_START]
    hold = ev[ev["filing_date"] >= HOLDOUT_START]

    report = {
        "all_train": compare(train, "all congress buys (train)"),
        "large_train": compare(train[train["is_large"]], "large congress buys (train)"),
        "large_repeat_train": compare(
            train[train["is_large"] & train["is_repeat"]], "large+repeat (train)"),
        "all_holdout": compare(hold, "all congress buys (2024+ holdout)"),
        "large_holdout": compare(hold[hold["is_large"]], "large congress buys (2024+ holdout)"),
    }
    out = {"confirm_window_days": CONFIRM_WINDOW, "car": CAR,
           "n_insider_events": n_insider,
           "n_confirmed": int(ev["confirmed"].sum()), "n_total": int(len(ev)),
           "report": report, "generated_at": datetime.now(timezone.utc).isoformat()}
    with open(DATA_DIR / "combo_experiment.json", "w") as f:
        json.dump(out, f, indent=2)

    if verbose:
        print(f"=== Insider-confirmation experiment (±{CONFIRM_WINDOW}d, {CAR}) ===")
        print(f"insider events={n_insider}  confirmed {out['n_confirmed']}/{out['n_total']} "
              f"congress buys\n")
        for key, r in report.items():
            c, u = r["confirmed"], r["unconfirmed"]
            if c["mean"] is None or u["mean"] is None:
                print(f"  {r['label']:34s} (insufficient n)")
                continue
            lift = c["mean"] - u["mean"]
            sig = "*" if (c["ci_low"] > 0 or c["ci_high"] < 0) else " "
            print(f"  {r['label']:34s} confirmed={c['mean']*100:+.2f}%{sig} "
                  f"[{c['ci_low']*100:+.2f},{c['ci_high']*100:+.2f}] n={c['n']}  |  "
                  f"unconfirmed={u['mean']*100:+.2f}% n={u['n']}  |  lift={lift*100:+.2f}pp")
    return out


if __name__ == "__main__":
    run()
