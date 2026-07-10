"""Low-priority resumable USAspending fetch — safe to run on a daily cron.

Fetches pending (uncached) months of large federal contracts within a short time
budget, caching each success and retrying failed months next run. Verified TLS
(truststore), records first_seen_at. When new contract months land, refreshes
the G-layer experiments.

Cron example (daily, quiet):
    0 6 * * *  cd /path/to/quantv1 && uv run python scripts/usaspending_tick.py

macOS launchd: schedule this command; it self-limits runtime.
"""

from __future__ import annotations

from quantv1.ingest import usaspending


def main():
    stats = usaspending.tick(time_budget_s=120)      # short budget; resumes next run
    if stats["cached"] and stats["g_contract_events"] > 0:
        # new contract data — refresh the experiments that depend on it
        from quantv1.research import gov_experiment
        gov_experiment.run(verbose=False)
        print("refreshed gov_experiment with new contract events")


if __name__ == "__main__":
    main()
