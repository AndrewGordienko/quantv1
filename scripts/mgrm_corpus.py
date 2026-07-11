"""MGRM real-filing gold-set corpus driver (data phase).

  uv run python scripts/mgrm_corpus.py targets [--per-sector 4]
  uv run python scripts/mgrm_corpus.py acquire --per-sector 4 [--start Y-M-D --end Y-M-D]
  uv run python scripts/mgrm_corpus.py select
  uv run python scripts/mgrm_corpus.py freeze     # manifest + dev prefill + skeletons
  uv run python scripts/mgrm_corpus.py distribution

Certification is NOT run here. After freezing: a human fills the skeletons, the
extractor is tuned on the development set and frozen, then certification runs via
scripts/mgrm_sprint.py goldset with a configured MGRM_LLM_PROVIDER.
"""

from __future__ import annotations

import argparse
import json

from quantv1.ingest import mgrm_corpus


def _print(payload) -> None:
    print(json.dumps(payload, indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("targets", "acquire"):
        p = sub.add_parser(name)
        p.add_argument("--per-sector", type=int, default=4)
        if name == "acquire":
            p.add_argument("--start")
            p.add_argument("--end")
            p.add_argument("--force", action="store_true")
    sub.add_parser("select")
    sub.add_parser("freeze")
    sub.add_parser("distribution")
    args = parser.parse_args()

    if args.command == "targets":
        targets = mgrm_corpus.crawl_targets(per_sector=args.per_sector)
        _print({"companies": len(targets),
                "tickers": [t["ticker"] for t in targets]})
    elif args.command == "acquire":
        from datetime import date
        from quantv1.ingest import guidance
        from quantv1.ingest.earnings import SAMPLE_END, SAMPLE_START
        targets = mgrm_corpus.crawl_targets(per_sector=args.per_sector)
        kwargs = {"universe": targets, "force": args.force}
        kwargs["start"] = date.fromisoformat(args.start) if args.start else SAMPLE_START
        kwargs["end"] = date.fromisoformat(args.end) if args.end else SAMPLE_END
        _print(guidance.discover_filings(**kwargs))
    elif args.command == "select":
        selection = mgrm_corpus.select_corpus()
        _print(mgrm_corpus.distribution(selection))
    elif args.command == "freeze":
        _print(mgrm_corpus.freeze())
    elif args.command == "distribution":
        _print(mgrm_corpus.distribution())


if __name__ == "__main__":
    main()
