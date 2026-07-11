"""Management Guidance Revision-Reaction Mismatch (MGRM) sprint driver.

Zero-vendor pivot: only automatically retrievable public SEC/IR data is used.
EERM M1/M2 stay paused (BLOCKED_DATA_ECONOMICALLY_INACCESSIBLE); this program is
independent and never reads or writes the vendor-consensus tables.

Pipeline order:
  uv run python scripts/mgrm_sprint.py discover  [--tickers AAPL,MSFT] [--force]
  uv run python scripts/mgrm_sprint.py extract    [--max-documents N] [--no-ai]
  uv run python scripts/mgrm_sprint.py link
  uv run python scripts/mgrm_sprint.py audit
  uv run python scripts/mgrm_sprint.py features --coarse|--full [--include-retrospective-holdout]
  uv run python scripts/mgrm_sprint.py run [--lock-spec] [--retrospective-holdout]
  uv run python scripts/mgrm_sprint.py forward           # daily forward collector
"""

from __future__ import annotations

import argparse
import json

from quantv1.config import DATA_DIR
from quantv1.ingest import guidance
from quantv1.research import mgrm


def _print(payload) -> None:
    print(json.dumps(payload, indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    discover = sub.add_parser("discover")
    discover.add_argument("--tickers", help="comma-separated subset of the frozen universe")
    discover.add_argument("--start")
    discover.add_argument("--end")
    discover.add_argument("--force", action="store_true")

    extract = sub.add_parser("extract")
    extract.add_argument("--max-documents", type=int)
    extract.add_argument("--no-ai", action="store_true")
    extract.add_argument("--force", action="store_true")

    sub.add_parser("link")
    sub.add_parser("audit")
    sub.add_parser("forward")

    features = sub.add_parser("features")
    mode = features.add_mutually_exclusive_group(required=True)
    mode.add_argument("--coarse", action="store_true")
    mode.add_argument("--full", action="store_true")
    features.add_argument("--include-retrospective-holdout", action="store_true")

    run = sub.add_parser("run")
    run.add_argument("--lock-spec", action="store_true")
    run.add_argument("--retrospective-holdout", action="store_true")

    args = parser.parse_args()

    if args.command == "discover":
        from datetime import date
        from quantv1.ingest.earnings import SAMPLE_END, SAMPLE_START, build_universe
        universe = build_universe()
        if args.tickers:
            wanted = {t.strip().upper() for t in args.tickers.split(",") if t.strip()}
            universe = [c for c in universe if c["ticker"].upper() in wanted]
        kwargs = {}
        if args.start:
            kwargs["start"] = date.fromisoformat(args.start)
        if args.end:
            kwargs["end"] = date.fromisoformat(args.end)
        _print(guidance.discover_filings(universe=universe, force=args.force, **kwargs))
    elif args.command == "extract":
        _print(guidance.extract_documents(max_documents=args.max_documents,
                                          force=args.force, use_ai=not args.no_ai))
    elif args.command == "link":
        _print(guidance.link_previous_guidance())
    elif args.command == "audit":
        audit = mgrm.extraction_audit()
        (DATA_DIR / "mgrm_audit.json").write_text(json.dumps(audit, indent=2, default=str))
        _print(audit)
    elif args.command == "forward":
        _print(guidance.collect_forward())
    elif args.command == "features":
        _print({"rows": len(mgrm.build_features(
            mode="full" if args.full else "coarse",
            include_retrospective_holdout=args.include_retrospective_holdout,
        ))})
    elif args.command == "run":
        _print(mgrm.run(lock_spec=args.lock_spec,
                        retrospective_holdout=args.retrospective_holdout))


if __name__ == "__main__":
    main()
