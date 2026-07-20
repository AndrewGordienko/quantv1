"""Prepare/score the SEC Event Atlas human-label queue."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from quantv1.events.goldset import score_goldset


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("path", nargs="?", default="goldset/sec_event_atlas_goldset_skeleton.jsonl")
    p.add_argument("--report", default="goldset/sec_event_atlas_goldset_score.json")
    args = p.parse_args()
    report = score_goldset(args.path)
    Path(args.report).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

