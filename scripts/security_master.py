"""Security master utilities: build intervals or write a coverage audit."""
import argparse, json
from quantv1.ingest.security_master import build_intervals, build_phase_a_pilot, coverage_audit, linkage_failure_decomposition, write_report

def main():
    p = argparse.ArgumentParser(); p.add_argument("command", choices=["build", "coverage", "pilot", "decompose"]); p.add_argument("input", nargs="?"); p.add_argument("output", nargs="?")
    p.add_argument("--root", default=None)
    p.add_argument("--output", default=None)
    a = p.parse_args()
    if a.command == "pilot":
        mappings, conflicts = build_phase_a_pilot(a.root)
        payload = {"status": "PILOT_CURRENT_CACHE_ONLY", "mappings": mappings, "conflicts": conflicts,
                   "warning": "Not point-in-time price coverage; delisting records require explicit source evidence."}
        with open(a.output or "goldset/sec_event_atlas_security_master_pilot.json", "w") as handle: json.dump(payload, handle, indent=2)
        print(json.dumps({"mappings": len(mappings), "conflicts": len(conflicts)}))
        return
    if a.command == "decompose":
        payload = json.load(open(a.input)); events = payload["events"]; mappings = payload["mappings"]
        report = linkage_failure_decomposition(events, mappings, a.root)
        with open(a.output or "goldset/sec_event_atlas_linkage_failure_decomposition.json", "w") as handle: json.dump(report, handle, indent=2, sort_keys=True)
        print(json.dumps({k: report[k] for k in ("unmapped_tags", "unmapped_accessions", "by_category")}))
        return
    rows = [json.loads(x) for x in open(a.input) if x.strip()]
    if a.command == "build":
        mappings, conflicts = build_intervals(rows); json.dump({"mappings": mappings, "conflicts": conflicts}, open(a.output, "w"), indent=2)
    else:
        payload = json.load(open(a.input)); write_report(a.output, coverage_audit(payload["events"], payload["mappings"], payload.get("price_windows")))
if __name__ == "__main__": main()
