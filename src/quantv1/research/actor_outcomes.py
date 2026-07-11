"""Deterministic residual-return outcomes for primary-source actor events.

Populates ``actor_event_outcomes`` from ``bars_minute`` by reusing the exact
same-session, pre-event beta/volatility engine that the actor news-mention
audit uses (:mod:`quantv1.v4.actor_impact`).  There is no text and no model
here: this is pure market data.  It is the half of the actor B1/B2 pipeline
that needs neither a manifest transcript nor an LLM, so it can be built and
verified before any feature extraction exists.

Hedge benchmark rule:
  * SPY (the market proxy itself) is skipped -- a self-hedge residual is
    degenerate, so exposures should react *against* the market, not be it.
  * Sector ETFs (XL*, and QQQ/DIA/IWM broad proxies) hedge against SPY.
  * Single names hedge against their sector ETF, falling back to SPY.

The written ``sector_beta_residual`` is the target column that
:func:`quantv1.research.actor_b2.load_frame` reads.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json

from ..config import BENCHMARK_TICKER, DATA_DIR
from ..db import connect
from ..v4.actor_impact import SECTOR_ETF, _outcome
from ..v4.replay import BarPanel, to_ns

OUTCOME_VERSION = "fed-primary-outcomes-v1"
HORIZONS_MINUTES = {"30m": 30, "1h": 60, "2h": 120}
PRIMARY_ROLES = ("speaker_author", "direct_public_action", "verified_decision_maker")
_BROAD_PROXIES = {"QQQ", "DIA", "IWM"}


def _hedge_benchmark(ticker: str, sectors: dict[str, str], panel: BarPanel) -> str | None:
    """Pick the beta-hedge benchmark whose residual we treat as the target."""
    upper = ticker.upper()
    if upper == BENCHMARK_TICKER:
        return None
    spy = BENCHMARK_TICKER if panel.has(BENCHMARK_TICKER) else None
    if upper.startswith("XL") or upper in _BROAD_PROXIES:
        return spy
    sector_etf = SECTOR_ETF.get(sectors.get(upper, ""))
    if sector_etf and panel.has(sector_etf):
        return sector_etf
    return spy


def build(outcome_version: str = OUTCOME_VERSION, verbose: bool = True) -> dict:
    """Compute and persist residual-return outcomes for primary-source events."""
    con = connect()
    events = con.execute(f"""
        SELECT actor_event_id, actor_id, ticker, public_time
        FROM actor_events
        WHERE primary_hypothesis_eligible=TRUE
          AND actor_event_role IN {PRIMARY_ROLES}
          AND ticker IS NOT NULL AND public_time IS NOT NULL
    """).df()
    sectors = {str(ticker).upper(): sector for ticker, sector in con.execute("""
        SELECT ticker, lower(sector) FROM ticker_sectors
        WHERE ticker IS NOT NULL AND sector IS NOT NULL
    """).fetchall()}
    panel = BarPanel(con, table="bars_minute")
    now = datetime.now(timezone.utc).replace(tzinfo=None)

    summary = {
        "outcome_version": outcome_version,
        "horizons": list(HORIZONS_MINUTES),
        "primary_events": int(len(events)),
        "events_with_bars": 0,
        "rows_written": 0,
        "non_null_sector_residual": {label: 0 for label in HORIZONS_MINUTES},
        "skipped_no_bars": 0,
        "skipped_no_hedge": 0,
    }
    if events.empty:
        summary["note"] = ("no primary-source actor events yet -- ingest a Fed "
                           "manifest via quantv1.ingest.fed_primary first")
        con.close()
        _write_summary(summary)
        if verbose:
            print(f"actor outcomes: {summary['note']}")
        return summary

    events = events[events["ticker"].str.upper().isin(
        {t.upper() for t in panel.data}
    )].reset_index(drop=True)
    summary["events_with_bars"] = int(len(events))
    events["public_time_ns"] = to_ns(events["public_time"])

    con.execute("BEGIN TRANSACTION")
    try:
        con.execute("DELETE FROM actor_event_outcomes WHERE outcome_version=?",
                    [outcome_version])
        for row in events.itertuples(index=False):
            ticker = str(row.ticker)
            if not panel.has(ticker):
                summary["skipped_no_bars"] += 1
                continue
            hedge = _hedge_benchmark(ticker, sectors, panel)
            if hedge is None:
                summary["skipped_no_hedge"] += 1
                continue
            for label, minutes in HORIZONS_MINUTES.items():
                outcome = _outcome(panel, ticker, int(row.public_time_ns),
                                   minutes, hedge)
                if outcome is None:
                    continue
                metadata = {
                    "hedge_benchmark": hedge,
                    "market_benchmark": outcome["market_benchmark"],
                    "market_beta": outcome["market_beta"],
                    "sector_beta": outcome["sector_beta"],
                    "beta_observations": outcome["beta_observations"],
                    "sector_beta_observations": outcome["sector_beta_observations"],
                    "pre_event_minute_volatility": outcome["pre_event_minute_volatility"],
                    "entry_ns": outcome["entry_ns"],
                    "exit_ns": outcome["exit_ns"],
                    "executable": outcome["executable"],
                }
                con.execute("""
                    INSERT INTO actor_event_outcomes
                        (actor_event_id, ticker, horizon, outcome_version,
                         public_time, raw_return, market_beta_residual,
                         sector_beta_residual, standardized_residual,
                         modeled_beta_hedged_return, actually_hedged_return,
                         execution_source, quote_source, created_at, metadata)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT DO NOTHING
                """, [row.actor_event_id, ticker, label, outcome_version,
                      row.public_time, outcome["raw_return"],
                      outcome["market_beta_residual"],
                      outcome["sector_beta_residual"],
                      outcome["residual_standardized"],
                      outcome["modeled_beta_hedged_return"],
                      outcome["actually_hedged_return"],
                      outcome["execution_basis"], "bars_minute", now,
                      json.dumps(metadata)])
                summary["rows_written"] += 1
                if outcome["sector_beta_residual"] is not None:
                    summary["non_null_sector_residual"][label] += 1
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        con.close()
        raise
    con.close()
    _write_summary(summary)
    if verbose:
        print(f"actor outcomes ({outcome_version}): {summary['rows_written']} rows "
              f"from {summary['events_with_bars']} events with bars; "
              f"2h non-null residuals: {summary['non_null_sector_residual']['2h']}")
    return summary


def _write_summary(summary: dict) -> None:
    with open(DATA_DIR / "actor_event_outcomes_build.json", "w") as file:
        json.dump(summary, file, indent=2)


if __name__ == "__main__":
    build()
