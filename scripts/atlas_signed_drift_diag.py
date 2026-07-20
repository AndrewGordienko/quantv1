"""SEC Atlas SIGNED post-event drift — DISCOVERY DIAGNOSTIC (return-blind sign map).

The directive's crux: the Atlas predicts "something moves", not "which way".
But the taxonomy already encodes fundamental direction in the event *type*
(guidance_raised +, secondary_offering -, buyback_authorization +, restatement
-, activist_13d +). So a SIGNED fundamental prior needs no new extraction; it is
frozen below from the event type alone (SIGN_MAP_VERSION), defined WITHOUT
looking at any return.

Tradability framing (why gap != drift):
  * You learn the event, then enter at the NEXT OPEN. The prev_close->next_open
    GAP is the market's immediate reaction and is NOT tradeable.
  * The tradeable quantity is the DRIFT from next_open onward. Under-reaction
    (drift continues in the fundamental direction) is the only tradeable edge;
    efficient pricing => drift ~ 0; over-reaction => drift reverses.

Caveats (this is DISCOVERY DIAGNOSTIC, not a test):
  * current-ticker-linked, survivorship-biased ~40% priced subset (see
    pit_panel_audit.json); 2022-2024 discovery split only.
  * per-family n is mostly < 100 event clusters -> below the power gate. A null
    here deprioritizes; a clean signed drift motivates the annotation+PIT-price
    gates, it does not authorize a trade.

Output: data/atlas_signed_drift_diag.json.
"""

from __future__ import annotations

import hashlib
import json

import duckdb
import numpy as np
import pandas as pd

from quantv1.config import DB_PATH, DATA_DIR

OUT = DATA_DIR / "atlas_signed_drift_diag.json"

# --- FROZEN structural sign map (return-blind; fundamental direction only) ---
# Only UNAMBIGUOUS directions are signed. Ambiguous types (dividend_change,
# buyback_change, restructuring, layoffs, ceo/cfo_departure, merger_announced,
# auditor_change, regulatory_decision, cyber) are EXCLUDED, not guessed.
SIGN_MAP_VERSION = "atlas-structural-sign-v1"
STRUCTURAL_SIGN: dict[str, int] = {
    # capital return
    "buyback_authorization": +1,
    # financing / dilution
    "secondary_offering": -1, "convertible_offering": -1,
    # guidance
    "guidance_raised": +1, "guidance_lowered": -1, "guidance_withdrawn": -1,
    # restatement / controls
    "restatement": -1, "internal_control_failure": -1, "material_weakness": -1,
    # activist / ownership
    "activist_13d": +1, "ownership_increase": +1, "ownership_decrease": -1,
    # going concern / distress
    "going_concern_warning": -1, "liquidity_warning": -1, "default_notice": -1,
    "bankruptcy_risk": -1,
    # commercial (unambiguous win/loss)
    "major_customer_win": +1, "major_customer_loss": -1, "contract_termination": -1,
    # insider clusters
    "insider_buy_cluster": +1, "insider_sell_cluster": -1,
    # cyber (negative direction)
    "cyber_incident": -1, "data_breach": -1,
}
# The directive's five frozen families for the eventual signed test.
DIRECTIVE_FAMILIES = {"guidance", "financing_dilution", "capital_return",
                      "restatement_controls", "activist_ownership"}
ROUND_TRIP_COST_BPS = 30.0   # nominal net hurdle for a drift to be worth trading


def sign_map_hash() -> str:
    return hashlib.sha256(
        json.dumps(STRUCTURAL_SIGN, sort_keys=True).encode()).hexdigest()[:12]


def _resid(stock, spy, da, pa, db, pb):
    """Residual return (stock - market) from (date da, price col pa) to (db, pb)."""
    if da not in stock.index or db not in stock.index:
        return None
    if da not in spy.index or db not in spy.index:
        return None
    s = stock.loc[db, pb] / stock.loc[da, pa] - 1.0
    m = spy.loc[db, "close"] / spy.loc[da, "open" if pa == "open" else "close"] - 1.0
    return float(s - m)


def load():
    con = duckdb.connect(str(DB_PATH), read_only=True)
    ev = con.execute("""
        SELECT atlas_event_id, ticker, event_family, event_type, public_time
        FROM atlas_events WHERE status='VERIFIED'
    """).df()
    tickers = sorted(set(ev["ticker"]) | {"SPY"})
    px = con.execute("""SELECT ticker,date,open,close FROM prices
                        WHERE ticker IN (SELECT UNNEST(?)) ORDER BY date""",
                     [tickers]).df()
    con.close()
    price = {t: g.set_index("date").sort_index() for t, g in px.groupby("ticker")}
    return ev, price


def event_measures(ev, price):
    spy = price.get("SPY")
    rows = []
    for r in ev.itertuples():
        s = STRUCTURAL_SIGN.get(r.event_type)
        if s is None:
            continue
        stock = price.get(r.ticker)
        if stock is None:
            continue
        d = pd.Timestamp(r.public_time)
        d = (d.tz_convert(None) if d.tzinfo is not None else d).normalize()
        after = stock.index[stock.index > d]
        before = stock.index[stock.index <= d]
        if len(after) < 6 or len(before) < 1:
            continue
        prev_close_date = before[-1]
        d1, d5 = after[0], after[4]
        d21 = after[20] if len(after) > 20 else None
        gap = _resid(stock, spy, prev_close_date, "close", d1, "open")
        drift5 = _resid(stock, spy, d1, "open", d5, "close")
        drift21 = _resid(stock, spy, d1, "open", d21, "close") if d21 is not None else None
        if gap is None or drift5 is None:
            continue
        rows.append({"family": r.event_family, "event_type": r.event_type,
                     "ticker": r.ticker, "sign": s,
                     "sgap": s * gap, "sdrift5": s * drift5,
                     "sdrift21": (s * drift21) if drift21 is not None else np.nan})
    return pd.DataFrame(rows)


def boot_ci(x, n=2000):
    x = np.asarray([v for v in x if v == v])  # drop NaN
    if len(x) < 5:
        return None
    idx = (np.arange(len(x)) * 2654435761) % len(x)  # deterministic seed-free base
    means = []
    rng = np.random.default_rng(12345)
    for _ in range(n):
        means.append(rng.choice(x, size=len(x), replace=True).mean())
    lo, hi = np.percentile(means, [2.5, 97.5])
    return [round(float(lo) * 1e4, 1), round(float(hi) * 1e4, 1)]  # bps


def summarize(df):
    out = []
    for fam, g in df.groupby("family"):
        rec = {"family": fam, "n": int(len(g)),
               "in_directive_five": fam in DIRECTIVE_FAMILIES,
               "mean_sgap_bps": round(float(g["sgap"].mean()) * 1e4, 1),
               "mean_sdrift5_bps": round(float(g["sdrift5"].mean()) * 1e4, 1),
               "sdrift5_ci_bps": boot_ci(g["sdrift5"]),
               "mean_sdrift21_bps": round(float(np.nanmean(g["sdrift21"])) * 1e4, 1)
               if g["sdrift21"].notna().any() else None,
               "sdrift5_pos_share": round(float((g["sdrift5"] > 0).mean()), 3),
               "n_tickers": int(g["ticker"].nunique())}
        ci = rec["sdrift5_ci_bps"]
        rec["drift5_sig_positive"] = bool(ci and ci[0] > 0)
        rec["clears_costs_gross"] = bool(rec["mean_sdrift5_bps"] > ROUND_TRIP_COST_BPS)
        out.append(rec)
    return sorted(out, key=lambda r: -r["mean_sdrift5_bps"])


def main():
    ev, price = load()
    df = event_measures(ev, price)
    fam = summarize(df)
    signed = [r for r in fam if r["in_directive_five"]]
    hits = [r for r in signed if r["drift5_sig_positive"] and r["clears_costs_gross"]]
    report = {
        "label": "DISCOVERY_DIAGNOSTIC_SURVIVORSHIP_BIASED",
        "not_a_promotion_test": True,
        "sign_map_version": SIGN_MAP_VERSION, "sign_map_hash": sign_map_hash(),
        "structural_sign": STRUCTURAL_SIGN,
        "measure": ("sgap=signed prev_close->next_open gap (untradeable); "
                    "sdrift5=signed next_open->day5_close (tradeable); "
                    "positive drift = under-reaction/continuation"),
        "round_trip_cost_hurdle_bps": ROUND_TRIP_COST_BPS,
        "n_signed_events_priced": int(len(df)),
        "by_family": fam,
        "directive_five_with_signed_positive_drift_clearing_costs":
            [r["family"] for r in hits],
        "verdict": ("SIGNED_DRIFT_LEAD_PRESENT" if hits
                    else "NO_TRADEABLE_SIGNED_DRIFT_IN_DISCOVERY"),
        "interpretation": (
            "Any positive family here is a survivorship-biased DISCOVERY lead, "
            "under-powered (n<100), not a trade. A null across the directive five "
            "means the signed-drift hypothesis is not worth the annotation+PIT "
            "price-panel cost; a clear lead justifies those gates."),
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    print(f"signed priced events: {len(df)}")
    print(f"{'family':22s} {'n':>4} {'sgap':>7} {'sdrift5':>8} {'ci_bps':>16} "
          f"{'d21':>7} {'cost?':>6}")
    for r in fam:
        star = " *5" if r["in_directive_five"] else ""
        print(f"{r['family']:22s} {r['n']:>4} {r['mean_sgap_bps']:>7} "
              f"{r['mean_sdrift5_bps']:>8} {str(r['sdrift5_ci_bps']):>16} "
              f"{str(r['mean_sdrift21_bps']):>7} {str(r['clears_costs_gross']):>6}{star}")
    print(f"VERDICT: {report['verdict']}  hits={report['directive_five_with_signed_positive_drift_clearing_costs']}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
