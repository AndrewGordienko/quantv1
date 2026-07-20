"""Audit an immutable F2 trades-and-NBBO sample; does not build F2 features."""

from __future__ import annotations

import argparse
import json

from quantv1.ingest.microstructure_audit import audit_manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", help="path to a frozen F2 sample manifest JSON")
    args = parser.parse_args()
    result = audit_manifest(args.manifest)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "ACCEPTED_FOR_F2_FEATURE_RESEARCH" else 2)
