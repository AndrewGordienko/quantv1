"""Fail-closed intake audit for an immutable F2 trades-and-NBBO sample.

This is intentionally a *data gate*, not an order-flow feature builder.  It
does not calculate OFI, trade signs, microprice, labels, or returns.  A vendor
sample must prove its ordering and historical-completeness claims in a frozen
manifest before any F2 research is allowed to start.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


VERSION = "latent-flow-f2-microstructure-audit-v1"
SELECTION_VERSION = "latent-flow-f2-pilot-selection-v1"
FILE_TYPES = ("quotes", "trades", "halts", "corporate_actions")
SEQUENCE_DOMAIN_FIELDS = {"ticker", "venue", "channel", "session"}
REQUIRED_COLUMNS = {
    "quotes": {"ticker", "exchange_ts", "sequence", "bid", "ask", "bid_size", "ask_size", "condition_codes"},
    "trades": {"ticker", "exchange_ts", "sequence", "price", "size", "condition_codes", "correction_code", "cancellation_code"},
    "halts": {"ticker", "start_ts", "end_ts", "halt_code"},
    "corporate_actions": {"ticker", "effective_date", "adjustment_type", "adjustment_factor", "source"},
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"unsupported sample format: {path.suffix}; use CSV or Parquet")


def _truth(value) -> bool:
    return value is True


def _manifest_errors(manifest: dict, root: Path) -> list[str]:
    errors = []
    for key in ("version", "provider", "dataset_id", "snapshot_id", "retrieved_at", "source_documentation_url",
                "historical_availability", "ordering", "quote_completeness", "selection", "files"):
        if not manifest.get(key):
            errors.append(f"MISSING_MANIFEST_FIELD:{key}")
    if manifest.get("version") != VERSION:
        errors.append("BAD_MANIFEST_VERSION")
    history = manifest.get("historical_availability") or {}
    if not (_truth(history.get("point_in_time")) and history.get("available_from") and history.get("documentation_url")):
        errors.append("HISTORY_NOT_PROVEN")
    ordering = manifest.get("ordering") or {}
    domain = ordering.get("sequence_domain")
    if not (_truth(ordering.get("exchange_timestamp")) and _truth(ordering.get("sequence_numbers")) and
            isinstance(domain, list) and "session" in domain and set(domain) <= SEQUENCE_DOMAIN_FIELDS and
            len(domain) == len(set(domain)) and ordering.get("documentation_url")):
        errors.append("ORDERING_NOT_PROVEN")
    complete = manifest.get("quote_completeness") or {}
    if not (_truth(complete.get("nbbo_updates_complete")) and complete.get("methodology") and
            complete.get("documentation_url")):
        errors.append("QUOTE_COMPLETENESS_NOT_PROVEN")
    conditions = manifest.get("conditions") or {}
    for key in ("trade_condition_codes_documented", "quote_condition_codes_documented",
                "corrections_and_cancellations_included"):
        if not _truth(conditions.get(key)):
            errors.append(f"CONDITION_NOT_PROVEN:{key}")

    files = manifest.get("files") or {}
    for kind in FILE_TYPES:
        entry = files.get(kind) or {}
        raw_path, expected = entry.get("path"), entry.get("sha256")
        if not raw_path or not isinstance(expected, str) or len(expected) != 64:
            errors.append(f"MISSING_FILE_PROVENANCE:{kind}")
            continue
        path = (root / raw_path).resolve()
        if not path.is_file():
            errors.append(f"MISSING_FILE:{kind}")
        elif _sha256(path) != expected.lower():
            errors.append(f"HASH_MISMATCH:{kind}")
    return errors


def _load_selection(manifest: dict, root: Path, errors: list[str]) -> dict | None:
    """Verify the pre-download selection artifact before reading observations."""
    entry = manifest.get("selection") or {}
    raw_path, expected = entry.get("path"), entry.get("sha256")
    if not raw_path or not isinstance(expected, str) or len(expected) != 64:
        errors.append("SELECTION_NOT_PINNED")
        return None
    path = (root / raw_path).resolve()
    if not path.is_file():
        errors.append("MISSING_SELECTION_MANIFEST")
        return None
    if _sha256(path) != expected.lower():
        errors.append("SELECTION_HASH_MISMATCH")
        return None
    try:
        selection = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        errors.append("BAD_SELECTION_MANIFEST")
        return None
    stocks = selection.get("stock_symbols")
    etfs = selection.get("sector_etfs")
    sessions = selection.get("sessions")
    if not (selection.get("version") == SELECTION_VERSION and isinstance(stocks, list) and
            len(stocks) == len(set(stocks)) and 10 <= len(stocks) <= 20 and
            selection.get("benchmark") == "SPY" and isinstance(etfs, dict) and
            set(etfs) == set(stocks) and isinstance(sessions, list) and 20 <= len(sessions) <= 40 and
            len(sessions) == len(set(sessions)) and _truth(selection.get("no_return_or_move_selection"))):
        errors.append("BAD_SELECTION_RULES")
        return None
    created, retrieved = pd.to_datetime(selection.get("created_at"), utc=True, errors="coerce"), \
        pd.to_datetime(manifest.get("retrieved_at"), utc=True, errors="coerce")
    if pd.isna(created) or pd.isna(retrieved) or created >= retrieved:
        errors.append("SELECTION_NOT_PRE_DOWNLOAD")
    return selection


def _timestamps(frame: pd.DataFrame, column: str, kind: str, errors: list[str]) -> pd.Series:
    ts = pd.to_datetime(frame[column], utc=True, errors="coerce")
    if ts.isna().any():
        errors.append(f"BAD_TIMESTAMP:{kind}")
    return ts


def _market_file_errors(frame: pd.DataFrame, kind: str, sequence_domain: list[str]) -> tuple[list[str], pd.Series | None]:
    errors: list[str] = []
    missing = REQUIRED_COLUMNS[kind] - set(frame.columns)
    if missing:
        return [f"MISSING_COLUMNS:{kind}:{','.join(sorted(missing))}"], None
    ts = _timestamps(frame, "exchange_ts", kind, errors)
    seq = pd.to_numeric(frame["sequence"], errors="coerce")
    if seq.isna().any() or (seq < 0).any() or (seq % 1 != 0).any():
        errors.append(f"BAD_SEQUENCE:{kind}")
    if frame["ticker"].isna().any() or frame["ticker"].astype(str).str.strip().eq("").any():
        errors.append(f"BAD_TICKER:{kind}")
    if frame["condition_codes"].isna().any() or frame["condition_codes"].astype(str).str.strip().eq("").any():
        errors.append(f"MISSING_CONDITION_CODE:{kind}")
    if ts.isna().any() or seq.isna().any():
        return errors, ts

    work = pd.DataFrame({"ticker": frame["ticker"].astype(str), "ts": ts, "sequence": seq.astype("int64")})
    work["session"] = work["ts"].dt.date
    group_keys = []
    for field in sequence_domain:
        if field == "session":
            group_keys.append("session")
        elif field not in frame.columns:
            errors.append(f"MISSING_SEQUENCE_DOMAIN_COLUMN:{kind}:{field}")
        else:
            work[field] = frame[field].astype(str)
            group_keys.append(field)
    if len(group_keys) != len(sequence_domain):
        return errors, ts
    if work.duplicated(group_keys + ["ts", "sequence"]).any():
        errors.append(f"DUPLICATE_ORDER_KEY:{kind}")
    # The raw file must already preserve the vendor-documented sequence domain.
    # Sorting it here would conceal precisely the ordering defect F2 cannot tolerate.
    for _, group in work.groupby(group_keys, sort=False):
        if not group["ts"].is_monotonic_increasing or not group["sequence"].is_monotonic_increasing:
            errors.append(f"NON_MONOTONIC_ORDER:{kind}")
            break
    return errors, ts


def _content_errors(frames: dict[str, pd.DataFrame]) -> list[str]:
    errors: list[str] = []
    quotes, trades = frames["quotes"], frames["trades"]
    for col in ("bid", "ask", "bid_size", "ask_size"):
        value = pd.to_numeric(quotes[col], errors="coerce")
        if value.isna().any() or (value < 0).any():
            errors.append(f"BAD_QUOTE_VALUE:{col}")
    bid, ask = pd.to_numeric(quotes["bid"], errors="coerce"), pd.to_numeric(quotes["ask"], errors="coerce")
    if (bid <= 0).any() or (ask <= 0).any() or (ask < bid).any():
        errors.append("CROSSED_OR_NONPOSITIVE_QUOTE")
    for col in ("price", "size"):
        value = pd.to_numeric(trades[col], errors="coerce")
        if value.isna().any() or (value <= 0).any():
            errors.append(f"BAD_TRADE_VALUE:{col}")
    for kind in ("halts", "corporate_actions"):
        missing = REQUIRED_COLUMNS[kind] - set(frames[kind].columns)
        if missing:
            errors.append(f"MISSING_COLUMNS:{kind}:{','.join(sorted(missing))}")
    return errors


def audit_manifest(path: str | Path) -> dict:
    """Audit an immutable sample manifest and fail closed on any missing proof."""
    manifest_path = Path(path)
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": "REJECTED", "version": VERSION, "errors": [f"BAD_MANIFEST:{exc}"]}
    root = manifest_path.parent
    errors = _manifest_errors(manifest, root)
    selection = _load_selection(manifest, root, errors)
    frames: dict[str, pd.DataFrame] = {}
    for kind in FILE_TYPES:
        entry = (manifest.get("files") or {}).get(kind) or {}
        candidate = root / str(entry.get("path", ""))
        if not candidate.is_file():
            continue
        try:
            frames[kind] = _read(candidate)
        except (OSError, ValueError, ImportError) as exc:
            errors.append(f"UNREADABLE_FILE:{kind}:{exc}")
    if set(frames) == set(FILE_TYPES):
        domain = (manifest.get("ordering") or {}).get("sequence_domain", [])
        quote_errors, quote_ts = _market_file_errors(frames["quotes"], "quotes", domain)
        trade_errors, trade_ts = _market_file_errors(frames["trades"], "trades", domain)
        errors.extend(quote_errors + trade_errors + _content_errors(frames))
        if quote_ts is not None and trade_ts is not None and not quote_ts.isna().any() and not trade_ts.isna().any():
            tickers = set(frames["quotes"]["ticker"].astype(str)) | set(frames["trades"]["ticker"].astype(str))
            sessions = set(quote_ts.dt.date) | set(trade_ts.dt.date)
            if selection:
                stocks = set(selection["stock_symbols"])
                expected = stocks | {selection["benchmark"]} | set(selection["sector_etfs"].values())
                if tickers != expected:
                    errors.append("SELECTION_SYMBOL_MISMATCH")
                selected_sessions = set(pd.to_datetime(selection["sessions"], utc=True).date)
                if sessions != selected_sessions:
                    errors.append("SELECTION_SESSION_MISMATCH")
            else:
                errors.append("SELECTION_NOT_VERIFIED")
    else:
        errors.append("SAMPLE_FILES_INCOMPLETE")
    errors = sorted(set(errors))
    return {
        "status": "ACCEPTED_FOR_F2_FEATURE_RESEARCH" if not errors else "REJECTED",
        "version": VERSION, "provider": manifest.get("provider"), "dataset_id": manifest.get("dataset_id"),
        "snapshot_id": manifest.get("snapshot_id"), "errors": errors,
        "note": "Completeness is verified from the vendor's frozen attestation and documentation; this audit cannot independently reconstruct a missing feed update.",
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Audit a frozen F2 trades/NBBO sample manifest")
    parser.add_argument("manifest")
    args = parser.parse_args()
    print(json.dumps(audit_manifest(args.manifest), indent=2))
