"""Focused earnings-alpha sprint driver.

Examples:
  uv run python scripts/earnings_sprint.py universe
  uv run python scripts/earnings_sprint.py sec-events
  uv run python scripts/earnings_sprint.py classify-sec
  uv run python scripts/earnings_sprint.py releases reviewed_releases.jsonl
  uv run python scripts/earnings_sprint.py consensus vendor_snapshots.jsonl
  uv run python scripts/earnings_sprint.py actuals vendor_actuals.jsonl
  uv run python scripts/earnings_sprint.py windows --max-events 25
  uv run python scripts/earnings_sprint.py audit
  uv run python scripts/earnings_sprint.py run
"""

from __future__ import annotations

import argparse

from quantv1.ingest import earnings


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("universe")
    sub.add_parser("sec-events")
    classify = sub.add_parser("classify-sec")
    classify.add_argument("--max-events", type=int)
    classify.add_argument("--force", action="store_true")
    for name in ("releases", "sec-classifications", "universe-metadata",
                 "consensus", "actuals",
                 "guidance", "options", "positioning", "calls"):
        command = sub.add_parser(name)
        command.add_argument("manifest")
    windows = sub.add_parser("windows")
    windows.add_argument("--max-events", type=int)
    windows.add_argument("--include-conservative", action="store_true")
    windows.add_argument("--with-quotes", action="store_true")
    windows.add_argument("--before", help="UTC ISO timestamp; excludes later events")
    windows.add_argument("--sample-modulus", type=int)
    windows.add_argument("--sample-remainder", type=int, default=0)
    windows.add_argument("--workers", type=int, default=1)
    windows.add_argument("--force", action="store_true")
    sub.add_parser("preflight")
    sub.add_parser("features")
    sub.add_parser("audit")
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--lock-spec", action="store_true")
    run_parser.add_argument("--final-test", action="store_true")
    args = parser.parse_args()

    if args.command == "universe":
        universe = earnings.build_universe()
        print({"companies": len(universe),
               "unseen": sum(row["company_bucket"] == "UNSEEN_COMPANY"
                             for row in universe)})
    elif args.command == "sec-events":
        print(earnings.acquire_sec_events())
    elif args.command == "classify-sec":
        print(earnings.classify_sec_candidates(max_events=args.max_events,
                                               force=args.force))
    elif args.command == "releases":
        print(earnings.ingest_release_manifest(args.manifest))
    elif args.command == "sec-classifications":
        print(earnings.ingest_sec_classification_manifest(args.manifest))
    elif args.command == "universe-metadata":
        print(earnings.ingest_universe_metadata_manifest(args.manifest))
    elif args.command == "consensus":
        print(earnings.ingest_consensus_manifest(args.manifest))
    elif args.command == "actuals":
        print(earnings.ingest_actuals_manifest(args.manifest))
    elif args.command == "guidance":
        print(earnings.ingest_guidance_manifest(args.manifest))
    elif args.command == "options":
        print(earnings.ingest_options_manifest(args.manifest))
    elif args.command == "positioning":
        print(earnings.ingest_positioning_manifest(args.manifest))
    elif args.command == "calls":
        print(earnings.ingest_call_manifest(args.manifest))
    elif args.command == "windows":
        from datetime import datetime
        from quantv1.v4.earnings_windows import fetch_bar_windows_parallel, fetch_windows
        before = (datetime.fromisoformat(args.before.replace("Z", "+00:00"))
                  .replace(tzinfo=None) if args.before else None)
        if args.workers > 1:
            if args.with_quotes or args.max_events:
                parser.error("parallel windows supports bars-only full selection only")
            print(fetch_bar_windows_parallel(
                include_conservative=args.include_conservative, before=before,
                sample_modulus=args.sample_modulus,
                sample_remainder=args.sample_remainder, force=args.force,
                workers=args.workers,
            ))
        else:
            print(fetch_windows(include_conservative=args.include_conservative,
                                max_events=args.max_events, force=args.force,
                                with_quotes=args.with_quotes, before=before,
                                sample_modulus=args.sample_modulus,
                                sample_remainder=args.sample_remainder))
    elif args.command == "preflight":
        from datetime import datetime, timedelta, timezone
        from quantv1.v4.earnings_windows import entitlement_preflight
        end = datetime.now(timezone.utc).replace(tzinfo=None)
        print(entitlement_preflight("AAPL", end - timedelta(days=1), end))
    elif args.command == "features":
        from quantv1.research.earnings_alpha import build_feature_frame
        print({"features": len(build_feature_frame())})
    elif args.command == "audit":
        import json
        from quantv1.research.earnings_alpha import structured_data_audit
        print(json.dumps(structured_data_audit(), indent=2))
    elif args.command == "run":
        from quantv1.research.earnings_alpha import run
        print(run(lock_spec=args.lock_spec, final_test=args.final_test))


if __name__ == "__main__":
    main()
