"""End-to-end daily pipeline: ingest -> research -> model -> portfolio -> persist.

Runs every stage sequentially because DuckDB is single-writer; nothing here may
overlap with a running API write (the API is read-only, so it can stay up).

Usage:
    uv run python scripts/daily_update.py                # full run
    uv run python scripts/daily_update.py --skip-ingest  # reuse cached data
    uv run python scripts/daily_update.py --no-backtest  # skip the slow backtest
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import pandas as pd

from quantv1.config import DATA_DIR
from quantv1.db import connect


def _persist_portfolio(book: pd.DataFrame, as_of) -> None:
    con = connect()
    con.execute("DELETE FROM portfolios WHERE as_of_date = ?", [as_of])
    if not book.empty:
        rows = []
        for r in book.itertuples(index=False):
            rationale = {
                "members": getattr(r, "members", []),
                "n_members": int(getattr(r, "n_members", 0) or 0),
                "contribs": getattr(r, "contribs", None),
            }
            rows.append([as_of, r.ticker, float(r.weight), float(r.score),
                         json.dumps(rationale)])
        con.executemany("INSERT INTO portfolios VALUES (?,?,?,?,?)", rows)
    con.close()


def _write_event_study() -> None:
    from quantv1.research import event_study as ES
    out = ES.run_report(purchases_only=True)

    def frame_to_records(df):
        return json.loads(df.to_json(orient="records"))

    payload = {k: frame_to_records(v) for k, v in out["summaries"].items()}
    payload["generated_at"] = datetime.now(timezone.utc).isoformat()
    with open(DATA_DIR / "event_study.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  event study -> {DATA_DIR / 'event_study.json'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-ingest", action="store_true")
    ap.add_argument("--no-backtest", action="store_true")
    ap.add_argument("--research-v2", action="store_true",
                    help="run leak-free v2 research (factors, persistence, event study, backtest)")
    ap.add_argument("--sector-limit", type=int, default=300,
                    help="max ticker sector lookups this run")
    args = ap.parse_args()

    t0 = datetime.now()
    print(f"=== quantv1 daily update @ {t0:%Y-%m-%d %H:%M} ===")

    # 1. Ingest (each stage is its own writer; run strictly sequentially) -----
    if not args.skip_ingest:
        from quantv1.ingest import stockwatcher, prices, committees, sectors
        print("[1/8] trades");    stockwatcher.ingest()
        print("[2/8] prices");    prices.ingest(incremental=True)
        print("[3/8] committees"); committees.ingest()
        print("[4/8] sectors");   sectors.ingest(limit=args.sector_limit)
    else:
        print("[1-4/8] ingest skipped")

    # 2. Skill leaderboard ----------------------------------------------------
    print("[5/8] skill scores")
    from quantv1.research import skill
    skill.run()

    # 3. Train signal model ---------------------------------------------------
    print("[6/8] train model")
    from quantv1.model import train
    tr = train.run()
    for name, m in tr["metrics"].items():
        print(f"      {name}: AUC={m['mean_auc']:.3f} IC={m['mean_ic']:.3f}")

    # 4. Score recent disclosures + build today's portfolio -------------------
    print("[7/8] score + construct portfolio")
    from quantv1.model import predict
    from quantv1.portfolio import construct
    from quantv1.research.returns import PriceStore
    store = PriceStore()
    scored = predict.score(store=store)
    if not scored.empty:
        predict.persist(scored)
        book = construct.construct(scored)
        as_of = pd.to_datetime(scored["filing_date"]).max().date()
        _persist_portfolio(book, as_of)
        print(f"      portfolio as of {as_of}: {len(book)} positions")
    else:
        print("      no recent trades to score")

    # 5. Backtest + event study ----------------------------------------------
    if not args.no_backtest:
        print("[8/8] backtest")
        from quantv1.portfolio import backtest
        backtest.run(verbose=False)
    _write_event_study()

    # v2: leak-free research engine (factor-adjusted, next-open, holdout) --------
    if args.research_v2:
        print("[v2] factors")
        if not args.skip_ingest:
            from quantv1.ingest import factors
            factors.ingest()
        print("[v2] skill persistence")
        from quantv1.research import skill_persistence
        skill_persistence.run(verbose=False)
        print("[v2] factor-adjusted event study")
        from quantv1.research import event_study_v2
        event_study_v2.run(verbose=False)
        print("[v2] leak-free backtest (4 experiments + holdout)")
        from quantv1.portfolio import backtest_v2
        backtest_v2.run(verbose=False)

        # v3: event store (entity resolution + P/F layers) + flagship experiment
        print("[v3] entity resolution + event store")
        if not args.skip_ingest:
            from quantv1.ingest import sec_entities, edgar_form4
            sec_entities.ingest(verbose=False)
            edgar_form4.ingest(verbose=False)
        from quantv1.events import store
        store.populate_congress(verbose=False)
        print("[v3] flagship insider-confirmation experiment")
        from quantv1.research import combo_experiment
        combo_experiment.run(verbose=False)

        # G layer: government contracts + contract experiments
        print("[v3] government contracts (USAspending)")
        try:
            if not args.skip_ingest:
                from quantv1.ingest import usaspending
                usaspending.ingest()
            from quantv1.research import gov_experiment
            gov_experiment.run(verbose=False)
        except Exception as e:  # noqa: BLE001 - USAspending endpoint is flaky
            print(f"      G-layer skipped: {e}")

        # fast-trigger proof-of-concept (hourly)
        print("[v3] intraday fast-trigger PoC (hourly)")
        if not args.skip_ingest:
            from quantv1.ingest import bars_hourly
            bars_hourly.ingest(verbose=False)
        from quantv1.strategies import intraday_meanrev
        intraday_meanrev.run(verbose=False)

        # MGRM: zero-vendor management-guidance revision vs reaction (EERM M1/M2
        # stay BLOCKED_DATA_ECONOMICALLY_INACCESSIBLE). Report the data gate and
        # append forward public guidance; fit only if the extraction gate passes.
        print("[mgrm] guidance data gate + forward collector")
        try:
            import json as _json
            from quantv1.config import DATA_DIR
            from quantv1.ingest import guidance, guidance_goldset
            from quantv1.research import mgrm
            gold = guidance_goldset.certify()
            (DATA_DIR / "mgrm_goldset_audit.json").write_text(
                _json.dumps(gold, indent=2, default=str))
            if not args.skip_ingest:
                guidance.collect_forward()
            audit = mgrm.extraction_audit()
            (DATA_DIR / "mgrm_audit.json").write_text(
                _json.dumps(audit, indent=2, default=str))
            mgrm.run(verbose=False)
            print(f"      MGRM gate: {audit['status']} "
                  f"(fit allowed: {audit['g1_g2_fitting_allowed']}); "
                  f"extractor gold-set: {gold['status']}")
        except Exception as e:  # noqa: BLE001 - SEC endpoint / empty gate is non-fatal
            print(f"      MGRM skipped: {e}")

    print(f"=== done in {(datetime.now() - t0).total_seconds():.0f}s ===")


if __name__ == "__main__":
    main()
