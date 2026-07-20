"""Point-in-time daily panel data-contract audit (read-only).

Protocol rule #2 ("verify the point-in-time data contract before reading
outcomes") applied to the *daily* panel that every cross-sectional and event
strategy in the backlog depends on. This script does not compute a single
return. It measures, with real numbers, how far the current `prices` panel is
from a survivorship-safe, corporate-action-correct, point-in-time contract.

The required contract (from goldset/sec_event_atlas_price_feasibility.json):
    raw_ohlcv, adjusted_ohlcv (both), split_dividend_factors, listing_intervals,
    delisting_evidence, last_trade, terminal_status, filing_era_symbol.

Output: data/pit_panel_audit.json (durable, re-runnable, return-blind).
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb

from quantv1.config import DB_PATH, DATA_DIR, ROOT

OUT = DATA_DIR / "pit_panel_audit.json"
SP500_CHANGES = ROOT / "goldset" / "forced_flow" / "sp500_changes_since_2019.csv"

# Columns that, if present in `prices`, would satisfy pieces of the contract.
ADJUSTED_HINT_COLS = {"adj_close", "adjusted_close", "adjclose", "close_adj"}
FACTOR_HINT_COLS = {"split_factor", "dividend", "adj_factor", "cum_factor",
                    "split_ratio", "cash_dividend"}
# Tables that, if present, would supply the missing survivorship pieces.
CONTRACT_TABLES = {
    "delisting": ["delisting", "delistings", "delisting_returns",
                  "security_terminal_status"],
    "listing_intervals": ["listing_intervals", "security_master",
                          "security_listing"],
    "corporate_actions": ["corporate_actions", "splits", "dividends",
                          "distributions"],
    "pit_sectors": ["sector_snapshots", "ticker_sector_history",
                    "market_cap_history"],
}


def _tables(con) -> set[str]:
    return {r[0] for r in con.execute("SHOW TABLES").fetchall()}


def _cols(con, table: str) -> list[str]:
    return [r[1] for r in con.execute(f"PRAGMA table_info('{table}')").fetchall()]


def _sp500_survivorship(con) -> dict:
    """Cross-check S&P 500 index changes against `prices` coverage.

    Deletions are the population *guaranteed* to have real 2012-2026 history
    (they were large-cap constituents). If they are absent from the panel, the
    panel is survivorship-biased by construction, not merely stale. We also
    check whether any *present* deletion actually terminates near its removal
    (a real delisting stop) vs. extends to the panel end (survived / reused
    symbol). The add/delete asymmetry measures whether the bias is sign-neutral.
    """
    import csv
    if not SP500_CHANGES.exists():
        return {"status": "SOURCE_ABSENT", "path": str(SP500_CHANGES)}
    removes: set[str] = set()
    adds: set[str] = set()
    with open(SP500_CHANGES) as f:
        for row in csv.DictReader(f):
            for t in row.get("remove", "").split(","):
                if t.strip():
                    removes.add(t.strip())
            for t in row.get("add", "").split(","):
                if t.strip():
                    adds.add(t.strip())
    gmax = con.execute("SELECT MAX(date) FROM prices").fetchone()[0]

    def cover(tickers: set[str]) -> dict:
        present = missing = ends_early = extends = 0
        missing_ex: list[str] = []
        for t in sorted(tickers):
            mn, mx, n = con.execute(
                "SELECT MIN(date), MAX(date), COUNT(*) FROM prices WHERE ticker=?",
                [t]).fetchone()
            if not n:
                missing += 1
                if len(missing_ex) < 40:
                    missing_ex.append(t)
                continue
            present += 1
            if (gmax - mx).days > 90:
                ends_early += 1
            else:
                extends += 1
        return {"n": len(tickers), "present": present, "missing": missing,
                "present_ends_early": ends_early, "present_extends_to_end": extends,
                "pct_missing": round(100.0 * missing / len(tickers), 1)
                if tickers else None, "missing_examples": missing_ex}

    return {"status": "OK", "deletions": cover(removes), "additions": cover(adds),
            "note": ("deletions with 0 'ends_early' means the panel records NO "
                     "delisting terminations; high deletion pct_missing = "
                     "survivorship absence; add/delete asymmetry = non-neutral bias")}


def audit() -> dict:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    tables = _tables(con)
    report: dict = {
        "db_path": str(DB_PATH),
        "requirements": [
            "raw_ohlcv", "adjusted_ohlcv", "split_dividend_factors",
            "listing_intervals", "delisting_evidence", "last_trade",
            "terminal_status", "filing_era_symbol",
        ],
        "tables_present": sorted(tables),
    }

    # --- prices panel shape -------------------------------------------------
    price_cols = _cols(con, "prices") if "prices" in tables else []
    price_col_set = {c.lower() for c in price_cols}
    n_tickers, dmin, dmax, n_rows = con.execute(
        "SELECT COUNT(DISTINCT ticker), MIN(date), MAX(date), COUNT(*) FROM prices"
    ).fetchone()

    # Survivorship proxy: a ticker whose last observation is well before the
    # global panel end either delisted, was renamed, or went stale. WITHOUT a
    # delisting-return table these names silently vanish from any as-of-date
    # universe -> classic survivorship bias. Quantify the magnitude.
    gap_rows = con.execute(
        """
        WITH span AS (
            SELECT ticker, MIN(date) AS first_date, MAX(date) AS last_date,
                   COUNT(*) AS n
            FROM prices GROUP BY ticker
        ),
        g AS (SELECT MAX(date) AS gmax FROM prices)
        SELECT
            SUM(CASE WHEN last_date < (SELECT gmax FROM g) - INTERVAL 30 DAY
                     THEN 1 ELSE 0 END) AS stale_30d,
            SUM(CASE WHEN last_date < (SELECT gmax FROM g) - INTERVAL 180 DAY
                     THEN 1 ELSE 0 END) AS stale_180d,
            SUM(CASE WHEN last_date < (SELECT gmax FROM g) - INTERVAL 365 DAY
                     THEN 1 ELSE 0 END) AS stale_365d,
            COUNT(*) AS total
        FROM span
        """
    ).fetchone()

    report["prices"] = {
        "columns": price_cols,
        "n_tickers": n_tickers,
        "date_min": str(dmin),
        "date_max": str(dmax),
        "n_rows": n_rows,
        "has_raw_and_adjusted_together": bool(
            price_col_set & ADJUSTED_HINT_COLS) and "close" in price_col_set,
        "adjusted_column_present": sorted(price_col_set & ADJUSTED_HINT_COLS),
        "split_dividend_factor_columns": sorted(price_col_set & FACTOR_HINT_COLS),
        "adjustment_provenance": {
            "source": "src/quantv1/ingest/prices.py (yfinance auto_adjust=True)",
            "close_is": "fully back-adjusted (split+dividend)",
            "raw_unadjusted_stored": False,
            "point_in_time_defect": ("back-adjustment is retroactive: a historical "
                                     "row's close reflects ALL later splits/divs, so "
                                     "it is not the level observable at that date; "
                                     "no raw close means as-of price filters, lot "
                                     "sizing and split detection are impossible"),
            "universe_construction": ("delisted/renamed tickers 'simply return "
                                      "nothing from yfinance' -> absent, not stale"),
        },
        "survivorship_proxy": {
            "note": ("tickers whose last daily bar precedes the panel end by "
                     "the given horizon: candidates for delist/rename/stale "
                     "that vanish from as-of-date universes without a "
                     "delisting-return table. LOW values here do NOT imply low "
                     "survivorship bias -- see sp500_survivorship for absence."),
            "tickers_stale_gt_30d": gap_rows[0],
            "tickers_stale_gt_180d": gap_rows[1],
            "tickers_stale_gt_365d": gap_rows[2],
            "total_tickers": gap_rows[3],
            "pct_stale_gt_365d": round(100.0 * (gap_rows[2] or 0) / gap_rows[3], 1)
            if gap_rows[3] else None,
        },
        "sp500_survivorship": _sp500_survivorship(con),
    }

    # --- ticker_sectors: is it point-in-time? -------------------------------
    if "ticker_sectors" in tables:
        ts_cols = _cols(con, "ticker_sectors")
        ts_n = con.execute("SELECT COUNT(*) FROM ticker_sectors").fetchone()[0]
        report["ticker_sectors"] = {
            "columns": ts_cols,
            "n_rows": ts_n,
            "is_point_in_time": any(
                c.lower() in {"as_of", "valid_from", "valid_to", "known_at",
                              "snapshot_date"} for c in ts_cols),
            "note": ("single current snapshot keyed by ticker only -> NOT "
                     "point-in-time; market_cap/sector as-of history absent"),
        }

    # --- intraday universes (breadth check) ---------------------------------
    for tbl, tscol in (("bars_minute", "ts"), ("bars_hourly", "ts")):
        if tbl in tables:
            n_sym, tmin, tmax, rows = con.execute(
                f"SELECT COUNT(DISTINCT ticker), MIN({tscol}), MAX({tscol}), "
                f"COUNT(*) FROM {tbl}"
            ).fetchone()
            report[tbl] = {
                "n_symbols": n_sym,
                "ts_min": str(tmin),
                "ts_max": str(tmax),
                "n_rows": rows,
            }

    # --- factors availability (for residualization) -------------------------
    if "factors" in tables:
        fmin, fmax, frows = con.execute(
            "SELECT MIN(date), MAX(date), COUNT(*) FROM factors"
        ).fetchone()
        report["factors"] = {"date_min": str(fmin), "date_max": str(fmax),
                             "n_rows": frows}

    # --- contract-table presence -------------------------------------------
    contract = {}
    for req, candidates in CONTRACT_TABLES.items():
        present = [t for t in candidates if t in tables]
        contract[req] = {"satisfied": bool(present), "tables": present}
    report["contract_tables"] = contract

    # --- overall verdict ----------------------------------------------------
    unmet = []
    if not report["prices"]["has_raw_and_adjusted_together"]:
        unmet.append("raw_and_adjusted_together")
    if not report["prices"]["split_dividend_factor_columns"]:
        unmet.append("split_dividend_factors")
    for req in ("delisting", "listing_intervals", "corporate_actions",
                "pit_sectors"):
        if not contract[req]["satisfied"]:
            unmet.append(req)
    report["verdict"] = {
        "status": "PANEL_CONTRACT_INCOMPLETE" if unmet else "PANEL_CONTRACT_OK",
        "unmet_requirements": unmet,
        "interpretation": (
            "Full-universe daily momentum / reversal / stat-arb results computed "
            "on this panel may be survivorship- and corporate-action artifacts. "
            "Diagnostics permitted; promotion gated until the contract is met."
        ),
    }
    con.close()
    return report


def main() -> None:
    report = audit()
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True))
    v = report["verdict"]
    p = report["prices"]
    print(f"prices: {p['n_tickers']} tickers, {p['date_min']}..{p['date_max']}, "
          f"{p['n_rows']:,} rows")
    print(f"survivorship: {p['survivorship_proxy']['tickers_stale_gt_365d']} "
          f"/ {p['survivorship_proxy']['total_tickers']} tickers stale >1y "
          f"({p['survivorship_proxy']['pct_stale_gt_365d']}%)")
    print(f"verdict: {v['status']}  unmet={v['unmet_requirements']}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
