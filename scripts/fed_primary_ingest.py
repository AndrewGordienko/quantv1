"""Ingest a reviewed official-source Fed B2/B3 manifest."""

from __future__ import annotations

import argparse

from quantv1.ingest.fed_primary import ingest_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", help="JSON or JSONL manifest; see docs/fed_primary_manifest.md")
    args = parser.parse_args()
    ingest_manifest(args.manifest)


if __name__ == "__main__":
    main()
