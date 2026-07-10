"""Daily forward-record runner — records the immutable paper decision each day.

Run once per trading day (after the disclosure feeds update). It records that
day's frozen books before market open; it NEVER edits a prior day's decision.

Cron (weekdays, before the open, after ingest):
    30 8 * * 1-5  cd /path/to/quantv1 && uv run python scripts/daily_update.py --skip-ingest \
                  && uv run python scripts/forward_tick.py
"""

from __future__ import annotations

from quantv1.forward import tracker


def main():
    tracker.record_decision()   # immutable snapshot of today's books
    tracker.settle()            # paper fills for any decision whose session has prices
    tracker.mark()              # daily positions, turnover, P&L, exits, benchmarks
    ev = tracker.evaluate()
    print(f"\nforward status: {ev['decisions_recorded']} decisions recorded, "
          f"{ev['trading_days_marked']} days marked, "
          f"{ev['completed_positions']} completed positions — {ev['status']}")


if __name__ == "__main__":
    main()
