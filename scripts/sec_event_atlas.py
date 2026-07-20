"""SEC Event Atlas manifest ingest and unsigned Stage-1 validation."""

from __future__ import annotations

import argparse
import json

from quantv1.events.atlas import ingest_manifest, unsigned_validation


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    ing = sub.add_parser("ingest")
    ing.add_argument("manifest")
    sub.add_parser("unsigned")
    args = parser.parse_args()
    result = ingest_manifest(args.manifest) if args.command == "ingest" else unsigned_validation()
    print(json.dumps(result, indent=2, default=str))
