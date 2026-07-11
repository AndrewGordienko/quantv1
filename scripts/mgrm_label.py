"""Launch the local MGRM labelling interface (data phase).

  uv run python scripts/mgrm_label.py            # http://127.0.0.1:8010
  uv run python scripts/mgrm_label.py --port 8010

Label the 20 DEVELOPMENT documents first; keep the 30 certification documents
sealed until the extractor and provider/model are frozen. Certification
documents never display prefill or extractor output.
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    args = parser.parse_args()
    import uvicorn
    from quantv1.labeling import create_app
    uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
