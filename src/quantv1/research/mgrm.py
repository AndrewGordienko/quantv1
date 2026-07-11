"""Management Guidance Revision-Reaction Mismatch research harness."""

from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import duckdb
import numpy as np
import pandas as pd

from ..config import DATA_DIR, ROOT
from ..db import connect
from ..ingest.earnings import (
    RETROSPECTIVE_HOLDOUT_START,
    VALIDATION_START,
)
from ..ingest.guidance import EXTRACTOR_VERSION, LINKER_VERSION
from ..ingest.guidance_goldset import certification_status
from .earnings_alpha import (
    MIN_TRAIN,
    MIN_VALIDATION,
    PRICE_CATEGORICAL,
    PRICE_NUMERIC,
    _available_model_specs as _unused_eerm_specs,
    _deflated_sharpe,
    _evaluate_cell,
    _fit,
    _git_hash,
    _load_feature_frame,
    _load_feature_metadata,
    _metrics,
    _stability_and_concentration,
    permutation_controls,
)
from .protocol import (
    BETA_VERSION,
    MIN_PERMUTATION_FRACTION,
    TARGET_VERSION,
    clustered_mean_ci,
    evaluate_power,
    execution_cost_estimate,
    first_session_after,
    power_requirements,
)


HYPOTHESIS = "Management Guidance Revision-Reaction Mismatch"
VERSION = "mgrm-v1"
ARTIFACT_VERSION = "mgrm-features-v2"
FEATURE_PATH = DATA_DIR / "mgrm_features.parquet"
METADATA_PATH = DATA_DIR / "mgrm_features_metadata.json"
REPORT_PATH = DATA_DIR / "mgrm_report.json"
SPEC_LOCK_PATH = DATA_DIR / "mgrm_model_spec_lock.json"
HOLDOUT_REPORT_PATH = DATA_DIR / "mgrm_retrospective_holdout_report.json"

MIN_DOCUMENT_COVERAGE = 0.90
MIN_EXTRACTION_AGREEMENT = 0.80
MIN_PREVIOUS_MATCH = 0.60
METRICS = ["revenue", "eps", "ebitda", "operating_income", "gross_margin",
           "bookings", "arr", "capex"]

GUIDANCE_NUMERIC = [
    "guidance_metric_count", "raised_count", "lowered_count",
    "reaffirmed_count", "withdrawn_count", "initiated_count", "guidance_withdrawn",
    "mean_midpoint_revision", "mean_range_width_change",
] + [f"{metric}_revision" for metric in METRICS] + [
    f"{metric}_range_width_change" for metric in METRICS
] + ["guidance_revision_score"]
MISMATCH_NUMERIC = ["guidance_reaction_mismatch"]
GUIDANCE_CATEGORICAL = ["guidance_direction"]


class MGRMError(RuntimeError):
    pass


def _write_frame(frame: pd.DataFrame) -> None:
    con = duckdb.connect(":memory:")
    try:
        con.register("_mgrm", frame)
        FEATURE_PATH.unlink(missing_ok=True)
        con.execute(f"COPY _mgrm TO '{FEATURE_PATH}' (FORMAT PARQUET)")
    finally:
        con.close()


def _read_frame() -> pd.DataFrame:
    if not FEATURE_PATH.exists():
        return pd.DataFrame()
    con = duckdb.connect(":memory:")
    try:
        return con.execute("SELECT * FROM read_parquet(?)", [str(FEATURE_PATH)]).df()
    finally:
        con.close()


def _guidance_rows() -> pd.DataFrame:
    con = connect(read_only=True)
    try:
        return con.execute("""
            SELECT x.extraction_id,x.earnings_event_id,x.ticker,x.metric,
                   x.guidance_period,x.lower_value,x.upper_value,x.midpoint,
                   x.guidance_status,x.stated_action,x.public_time,
                   l.previous_extraction_id,l.midpoint_revision,
                   l.range_width_change,l.revision_classification,l.link_status
            FROM mgrm_guidance_extractions x
            LEFT JOIN mgrm_guidance_links l USING (extraction_id)
            WHERE x.agreement_status='AGREED' AND x.earnings_event_id IS NOT NULL
              AND (l.link_status IS NULL OR
                   l.link_status NOT IN ('DUPLICATE_EXHIBIT',
                                         'UNKNOWN_PERIOD_NOT_LINKABLE'))
            ORDER BY x.public_time,x.ticker,x.metric,x.extraction_id
        """).df()
    finally:
        con.close()


MIN_REVISION_HISTORY = 5


def _hierarchical_z(revision: float, histories: list[list[float]]) -> float:
    """Standardize a numeric revision using the finest level with enough history.

    Frozen backoff: company -> sector -> metric. A level qualifies only with at
    least ``MIN_REVISION_HISTORY`` past observations; if none qualifies the
    normalized score is left missing (NaN) rather than returning the raw
    revision as if it were already standardized.
    """
    for history in histories:
        valid = np.asarray([item for item in history if np.isfinite(item)],
                           dtype=float)
        if len(valid) >= MIN_REVISION_HISTORY:
            std = valid.std(ddof=1)
            return float((revision - valid.mean()) / std) if std > 1e-12 else 0.0
    return np.nan


def structured_guidance_features(market: pd.DataFrame,
                                 guidance: pd.DataFrame) -> pd.DataFrame:
    """Create point-in-time expanding guidance scores without future fitting."""
    data = market.copy()
    if guidance.empty or data.empty:
        return pd.DataFrame()
    sectors = data[["earnings_event_id", "sector"]].drop_duplicates()
    rows = guidance.merge(sectors, on="earnings_event_id", how="left")
    rows["public_time"] = pd.to_datetime(rows["public_time"], utc=True)
    rows = rows.sort_values(["public_time", "ticker", "metric", "extraction_id"])
    company_history: dict[tuple[str, str], list[float]] = defaultdict(list)
    sector_history: dict[tuple[str, str], list[float]] = defaultdict(list)
    metric_history: dict[str, list[float]] = defaultdict(list)
    scores = []
    for record in rows.to_dict(orient="records"):
        # Only genuine numeric midpoint revisions are standardized. Non-numeric
        # RAISED/LOWERED/WITHDRAWN language is NOT converted into a fabricated
        # magnitude; it survives as the separate action counts and withdrawal
        # indicator below. Missing revision -> missing normalized score.
        revision = record.get("midpoint_revision")
        if revision is not None and np.isfinite(revision):
            company_key = (str(record["ticker"]), str(record["metric"]))
            sector_key = (str(record.get("sector")), str(record["metric"]))
            score = _hierarchical_z(float(revision), [
                company_history[company_key],
                sector_history[sector_key],
                metric_history[str(record["metric"])],
            ])
            company_history[company_key].append(float(revision))
            sector_history[sector_key].append(float(revision))
            metric_history[str(record["metric"])].append(float(revision))
        else:
            score = np.nan
        scores.append(score)
    rows["normalized_revision_score"] = scores

    event_records = []
    for event_id, group in rows.groupby("earnings_event_id", sort=False):
        record = {"earnings_event_id": event_id,
                  "guidance_metric_count": int(group.metric.nunique())}
        classifications = group["revision_classification"].fillna(
            group["stated_action"]
        )
        for action in ("RAISED", "LOWERED", "REAFFIRMED", "WITHDRAWN", "INITIATED"):
            record[f"{action.lower()}_count"] = int((classifications == action).sum())
        record["guidance_withdrawn"] = int(record["withdrawn_count"] > 0)
        record["mean_midpoint_revision"] = float(group.midpoint_revision.mean()) \
            if group.midpoint_revision.notna().any() else np.nan
        record["mean_range_width_change"] = float(group.range_width_change.mean()) \
            if group.range_width_change.notna().any() else np.nan
        record["guidance_revision_score"] = float(group.normalized_revision_score.mean()) \
            if group.normalized_revision_score.notna().any() else np.nan
        for metric in METRICS:
            selected = group[group.metric == metric]
            record[f"{metric}_revision"] = (
                float(selected.midpoint_revision.mean())
                if selected.midpoint_revision.notna().any() else np.nan
            )
            record[f"{metric}_range_width_change"] = (
                float(selected.range_width_change.mean())
                if selected.range_width_change.notna().any() else np.nan
            )
        score = record["guidance_revision_score"]
        record["guidance_direction"] = (
            "POSITIVE" if np.isfinite(score) and score > 0 else
            "NEGATIVE" if np.isfinite(score) and score < 0 else "MIXED_OR_FLAT"
        )
        event_records.append(record)
    structured = pd.DataFrame(event_records)
    result = data.merge(structured, on="earnings_event_id", how="inner")
    result["guidance_reaction_mismatch"] = (
        result["guidance_revision_score"] - result["reaction_score"]
    )
    return result


def _executable_sample(market: pd.DataFrame | None,
                       guidance: pd.DataFrame | None = None) -> pd.DataFrame:
    """Join guidance onto the market frame and keep rows with a usable target."""
    guidance = guidance if guidance is not None else _guidance_rows()
    if market is None or len(market) == 0 or guidance.empty:
        return pd.DataFrame()
    frame = structured_guidance_features(market, guidance)
    if frame.empty or "target_beta_hedged_5d" not in frame:
        return pd.DataFrame()
    return frame.dropna(subset=["target_beta_hedged_5d"]).copy()


def _sample_power(sample: pd.DataFrame) -> dict:
    """evaluate_power() on eligible feature rows, plus executability reporting."""
    if not len(sample):
        power = power_requirements(np.nan)
        gate = evaluate_power(power, unique_trades=0, unique_tickers=0,
                              unique_dates=0, events_by_year={})
        return {"unique_executable_events": 0, "unique_tickers": 0,
                "unique_announcement_dates": 0, "events_by_year": {},
                "effective_sample_size": gate["effective_sample_size"],
                "quote_complete_coverage": 0.0, "long_deployable": 0,
                "short_deployable": 0, "power_requirements": power, "gate": gate}
    entry = pd.to_datetime(sample["entry_time"], utc=True)
    events_by_year = {str(year): int(count)
                      for year, count in entry.dt.year.value_counts().items()}
    unique_events = int(sample.earnings_event_id.nunique())
    unique_tickers = int(sample.ticker.nunique())
    unique_dates = int(entry.dt.date.astype(str).nunique())
    quote_coverage = float(pd.to_numeric(
        sample.get("quote_complete", pd.Series(False, index=sample.index)),
        errors="coerce").fillna(0).mean())
    records = sample.to_dict("records")
    long_deployable = int(sum(bool(execution_cost_estimate(row, 1).get("deployable"))
                              for row in records))
    short_deployable = int(sum(bool(execution_cost_estimate(row, -1).get("deployable"))
                               for row in records))
    power = power_requirements(float(sample["target_beta_hedged_5d"].std(ddof=1)))
    gate = evaluate_power(power, unique_trades=unique_events,
                          unique_tickers=unique_tickers, unique_dates=unique_dates,
                          events_by_year=events_by_year)
    return {"unique_executable_events": unique_events,
            "unique_tickers": unique_tickers,
            "unique_announcement_dates": unique_dates,
            "events_by_year": events_by_year,
            "effective_sample_size": gate["effective_sample_size"],
            "quote_complete_coverage": quote_coverage,
            "long_deployable": long_deployable, "short_deployable": short_deployable,
            "power_requirements": power, "gate": gate}


def extraction_audit(market: pd.DataFrame | None = None) -> dict:
    market = market.copy() if market is not None else _load_feature_frame()
    con = connect(read_only=True)
    try:
        counts = {
            "eligible_filings": int(con.execute(
                "SELECT COUNT(*) FROM mgrm_filings WHERE status='ELIGIBLE'"
            ).fetchone()[0]),
            "preserved_documents": int(con.execute(
                "SELECT COUNT(*) FROM mgrm_documents WHERE status='PRESERVED'"
            ).fetchone()[0]),
            "eligible_filings_with_document": int(con.execute(
                "SELECT COUNT(DISTINCT d.accession_number) FROM mgrm_documents d "
                "JOIN mgrm_filings f USING (accession_number) "
                "WHERE d.status='PRESERVED' AND f.status='ELIGIBLE'"
            ).fetchone()[0]),
            "deterministic_extractions": int(con.execute(
                "SELECT COUNT(*) FROM mgrm_guidance_extractions"
            ).fetchone()[0]),
            "agreed_extractions": int(con.execute(
                "SELECT COUNT(*) FROM mgrm_guidance_extractions "
                "WHERE agreement_status='AGREED'"
            ).fetchone()[0]),
            "linked_extractions": int(con.execute(
                "SELECT COUNT(*) FROM mgrm_guidance_links WHERE link_status='LINKED'"
            ).fetchone()[0]),
        }
        events = con.execute("""
            SELECT COUNT(DISTINCT earnings_event_id) FROM mgrm_filings
            WHERE status='ELIGIBLE' AND earnings_event_id IS NOT NULL
        """).fetchone()[0]
    finally:
        con.close()
    document_coverage = (counts["eligible_filings_with_document"] /
                         counts["eligible_filings"]
                         if counts["eligible_filings"] else 0.0)
    agreement = (counts["agreed_extractions"] / counts["deterministic_extractions"]
                 if counts["deterministic_extractions"] else 0.0)
    previous_match = (counts["linked_extractions"] / counts["agreed_extractions"]
                      if counts["agreed_extractions"] else 0.0)

    # Power is evaluated on the ACTUAL executable feature rows (target present),
    # via the corrected protocol gate -- unique tickers, announcement dates,
    # events-per-year and effective sample size, not a bare event count.
    guidance = _guidance_rows()
    accepted_events = int(guidance.earnings_event_id.nunique()) if len(guidance) else 0
    sample = _executable_sample(market, guidance)
    sample_power = _sample_power(sample)
    power = sample_power.pop("power_requirements")
    power_gate = sample_power.pop("gate")

    certification = certification_status()
    gates = {
        "document_coverage": document_coverage >= MIN_DOCUMENT_COVERAGE,
        "deterministic_ai_agreement": agreement >= MIN_EXTRACTION_AGREEMENT,
        "previous_guidance_match": previous_match >= MIN_PREVIOUS_MATCH,
        "power": bool(power_gate["passes"]),
        "goldset_certified": bool(certification["certified"]),
    }
    return {
        "status": "READY" if all(gates.values()) else "BLOCKED",
        "counts": {**counts, "linked_earnings_events": int(events),
                   "accepted_guidance_events": accepted_events},
        "rates": {"document_coverage": document_coverage,
                  "extraction_agreement": agreement,
                  "previous_guidance_match": previous_match},
        "thresholds": {"document_coverage": MIN_DOCUMENT_COVERAGE,
                       "extraction_agreement": MIN_EXTRACTION_AGREEMENT,
                       "previous_guidance_match": MIN_PREVIOUS_MATCH},
        "power_requirements": power, "power_gate": power_gate,
        "sample_power": sample_power, "certification": certification,
        "goldset_certified": bool(certification["certified"]), "gates": gates,
        "g1_g2_fitting_allowed": all(gates.values()),
        "extractor_version": EXTRACTOR_VERSION,
        "linker_version": LINKER_VERSION,
    }


def build_features(*, mode: str, include_retrospective_holdout: bool = False) -> pd.DataFrame:
    if mode not in {"coarse", "full"}:
        raise MGRMError("feature mode must be coarse or full")
    if include_retrospective_holdout and not SPEC_LOCK_PATH.exists():
        raise MGRMError("MGRM retrospective features require the MGRM model lock")
    source_metadata = _load_feature_metadata()
    if mode == "full" and source_metadata.get("mode") != "full":
        raise MGRMError("MGRM full mode requires a full market feature artifact")
    # G1/G2 feature promotion fails closed without a valid extractor
    # certification (absent / stale / wrong provider / below threshold).
    certification = certification_status()
    if mode == "full" and not certification["certified"]:
        raise MGRMError(
            "MGRM full (promotable) features require a valid extractor "
            f"certification; got {certification['reason']}. Coarse diagnostic "
            "features remain available."
        )
    market = _load_feature_frame()
    if not include_retrospective_holdout and not market.empty:
        market = market[market.time_bucket != "RETROSPECTIVE_HOLDOUT_TIME"].copy()
    frame = structured_guidance_features(market, _guidance_rows())
    if mode == "coarse" and not frame.empty:
        frame = frame[frame.earnings_event_id.map(
            lambda value: int(hashlib.sha256(str(value).encode()).hexdigest()[:8], 16) % 4 == 0
        )].copy()
    if not frame.empty:
        frame["artifact_mode"] = mode
        frame["artifact_version"] = ARTIFACT_VERSION
        frame["hypothesis"] = HYPOTHESIS
        _write_frame(frame)
    else:
        FEATURE_PATH.unlink(missing_ok=True)
    metadata = {
        "artifact_version": ARTIFACT_VERSION, "mode": mode,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rows": len(frame), "includes_retrospective_holdout":
            include_retrospective_holdout,
        "source_market_artifact": source_metadata,
        "target_version": TARGET_VERSION, "beta_version": BETA_VERSION,
        "extractor_version": EXTRACTOR_VERSION, "linker_version": LINKER_VERSION,
        "goldset_certification": certification,
        "promotion_eligible": (mode == "full" and
                               source_metadata.get("mode") == "full" and
                               source_metadata.get("target_version") == TARGET_VERSION and
                               source_metadata.get("beta_version") == BETA_VERSION and
                               certification["certified"]),
        "code_hash": _git_hash(),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2, default=str))
    return frame


def _specs(structured_allowed: bool) -> dict[str, tuple[list[str], list[str]]]:
    specs = {"G0_reaction_only": (PRICE_NUMERIC, PRICE_CATEGORICAL)}
    if structured_allowed:
        specs["G1_structured_guidance"] = (
            list(dict.fromkeys(PRICE_NUMERIC + GUIDANCE_NUMERIC)),
            list(dict.fromkeys(PRICE_CATEGORICAL + GUIDANCE_CATEGORICAL)),
        )
        specs["G2_guidance_reaction_mismatch"] = (
            list(dict.fromkeys(PRICE_NUMERIC + GUIDANCE_NUMERIC + MISMATCH_NUMERIC)),
            list(dict.fromkeys(PRICE_CATEGORICAL + GUIDANCE_CATEGORICAL)),
        )
    return specs


def _dataset_hash(frame: pd.DataFrame) -> str:
    content = frame[["earnings_event_id", "target_beta_hedged_5d",
                     "guidance_revision_score"]].astype(str) \
        .sort_values("earnings_event_id").to_csv(index=False)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def run(frame: pd.DataFrame | None = None, *, lock_spec: bool = False,
        retrospective_holdout: bool = False, verbose: bool = True) -> dict:
    if retrospective_holdout and HOLDOUT_REPORT_PATH.exists():
        return {"status": "MGRM_HOLDOUT_ALREADY_OPENED",
                "result": json.loads(HOLDOUT_REPORT_PATH.read_text())}
    if retrospective_holdout and not SPEC_LOCK_PATH.exists():
        return {"status": "MGRM_HOLDOUT_BLOCKED_SPEC_NOT_LOCKED"}
    if retrospective_holdout:
        certification = certification_status()
        if not certification["certified"]:
            return {"status": "MGRM_HOLDOUT_BLOCKED_UNCERTIFIED",
                    "certification": certification}
    data = frame.copy() if frame is not None else _read_frame()
    audit = extraction_audit(data)
    if data.empty:
        result = {"status": "MGRM_BLOCKED_EXTRACTION_DATA", "audit": audit,
                  "g1_g2_fitted": False,
                  "retrospective_holdout_outcomes_evaluated": False}
        REPORT_PATH.write_text(json.dumps(result, indent=2, default=str))
        return result
    data["entry_time"] = pd.to_datetime(data["entry_time"], utc=True)
    usable = data.dropna(subset=["target_beta_hedged_5d"]).sort_values("entry_time")
    train = usable[usable.time_bucket == "TRAIN_TIME"].copy()
    validation = usable[usable.time_bucket == "VALIDATION_TIME"].copy()
    holdout = usable[usable.time_bucket == "RETROSPECTIVE_HOLDOUT_TIME"].copy()
    preholdout = usable[usable.time_bucket != "RETROSPECTIVE_HOLDOUT_TIME"].copy()
    counts = {"features": len(data), "train": len(train),
              "validation": len(validation), "retrospective_holdout": len(holdout)}
    if retrospective_holdout:
        metadata = json.loads(METADATA_PATH.read_text()) if METADATA_PATH.exists() else {}
        if (metadata.get("mode") != "full" or
                not metadata.get("includes_retrospective_holdout")):
            return {"status": "MGRM_HOLDOUT_BLOCKED_ARTIFACT"}
        spec = json.loads(SPEC_LOCK_PATH.read_text())
        numeric, categorical = spec["numeric"], spec["categorical"]
        from .earnings_alpha import _pipeline
        model = _pipeline(numeric, categorical)
        model.set_params(**spec["elastic_net_params"])
        model.fit(preholdout[numeric + categorical],
                  preholdout["target_beta_hedged_5d"])
        result = {"status": "MGRM_RETROSPECTIVE_HOLDOUT_OPENED",
                  "model_name": spec["model_name"], "counts": counts,
                  "result": _evaluate_cell(
                      holdout, model.predict(holdout[numeric + categorical])
                  ), "spec_lock": spec,
                  "retrospective_holdout_outcomes_evaluated": True}
        HOLDOUT_REPORT_PATH.write_text(json.dumps(result, indent=2, default=str))
        return result
    if len(train) < MIN_TRAIN or len(validation) < MIN_VALIDATION:
        result = {"status": "MGRM_BLOCKED_SAMPLE_POWER", "counts": counts,
                  "required": {"train": MIN_TRAIN, "validation": MIN_VALIDATION},
                  "audit": audit, "g1_g2_fitted": False,
                  "retrospective_holdout_outcomes_evaluated": False}
        REPORT_PATH.write_text(json.dumps(result, indent=2, default=str))
        return result

    specs = _specs(audit["g1_g2_fitting_allowed"])
    models = {}
    params = {}
    predictions = {}
    results = {}
    for name, (numeric, categorical) in specs.items():
        models[name], params[name] = _fit(train, numeric, categorical)
        predictions[name] = models[name].predict(validation[numeric + categorical])
        results[name] = _evaluate_cell(validation, predictions[name])
        results[name]["permutation_controls"] = permutation_controls(
            models[name], validation, numeric, categorical
        )
    structured = "G2_guidance_reaction_mismatch" in specs
    candidate = "G2_guidance_reaction_mismatch" if structured else "G0_reaction_only"
    candidate_result = results[candidate]
    gates = {"extraction_data_gate": audit["g1_g2_fitting_allowed"],
             "g2_tested": structured}
    if structured:
        g1, g2 = results["G1_structured_guidance"], results[candidate]
        lift = (np.square(validation.target_beta_hedged_5d -
                          predictions["G1_structured_guidance"]) -
                np.square(validation.target_beta_hedged_5d - predictions[candidate]))
        lift_ci = clustered_mean_ci(validation, lift)
        gates.update({
            "g2_rmse_lift_vs_g1": g2["predictive"]["rmse"] < g1["predictive"]["rmse"],
            "g2_net_lift_vs_g1": g2["portfolio"]["net_return"] >
                                 g1["portfolio"]["net_return"],
            "g2_clustered_loss_lift_lower_above_zero":
                lift_ci["confidence_interval"][0] is not None and
                lift_ci["confidence_interval"][0] > 0,
        })
    else:
        lift_ci = {"status": "NOT_TESTED", "confidence_interval": [None, None]}
    permutation_fraction = (
        candidate_result["permutation_controls"].get("grouping", {})
        .get("permuted_rows", 0) / len(validation)
    )
    stability = _stability_and_concentration(candidate_result["portfolio"])
    dsr = _deflated_sharpe(candidate_result["portfolio"]["daily_returns"], 32)
    bootstrap = candidate_result["cluster_bootstrap"].get("confidence_intervals", {})
    hac_ci = candidate_result["hac"].get("annualized_alpha_ci", [None, None])
    metadata = json.loads(METADATA_PATH.read_text()) if METADATA_PATH.exists() else {}
    quote_coverage = float(pd.to_numeric(
        validation.get("quote_complete", pd.Series(False, index=validation.index)),
        errors="coerce").fillna(0).mean())
    gates.update({
        "full_artifact": metadata.get("promotion_eligible") is True,
        "target_beta_version_frozen":
            metadata.get("target_version") == TARGET_VERSION and
            metadata.get("beta_version") == BETA_VERSION,
        "executable_quote_coverage_at_least_95pct": quote_coverage >= 0.95,
        "goldset_certified": audit["gates"]["goldset_certified"],
        "power_gate_passed": bool(audit["power_gate"]["passes"]),
        "positive_net_return": candidate_result["portfolio"]["net_return"] > 0,
        "net_sharpe_above_1": (candidate_result["portfolio"].get("sharpe_annual") or 0) > 1,
        "deflated_sharpe_above_0_95": dsr["passes"],
        "positive_delayed_entry": candidate_result["delayed_entry"]["net_return"] > 0,
        "positive_doubled_costs": candidate_result["doubled_costs"]["net_return"] > 0,
        "stable_and_unconcentrated": stability["passes"],
        "permutation_change_coverage": permutation_fraction >= MIN_PERMUTATION_FRACTION,
        "bootstrap_alpha_lower_above_zero":
            bootstrap.get("annualized_alpha", [None])[0] is not None and
            bootstrap["annualized_alpha"][0] > 0,
        "hac_alpha_lower_above_zero": hac_ci[0] is not None and hac_ci[0] > 0,
    })
    passed = all(gates.values()) and candidate == "G2_guidance_reaction_mismatch"
    dataset_hash = _dataset_hash(preholdout)
    experiment_id = hashlib.sha256(
        f"{VERSION}|{dataset_hash}|{candidate}|{_git_hash()}".encode()
    ).hexdigest()[:20]
    result = {"status": "MGRM_VALIDATION_PASSED" if passed else
                         "MGRM_VALIDATION_BLOCKED",
              "experiment_id": experiment_id, "dataset_hash": dataset_hash,
              "counts": counts, "audit": audit, "candidate_model": candidate,
              "models": results, "elastic_net_params": params,
              "gates": gates, "m2_style_lift": lift_ci,
              "g1_g2_fitted": structured, "catboost_allowed": passed,
              "retrospective_holdout_outcomes_evaluated": False}
    con = connect()
    con.execute("""
        INSERT INTO mgrm_experiments
            (experiment_id,created_at,status,dataset_hash,code_hash,model_name,
             holdout_definition,metrics,promotion_gates)
        VALUES (?,?,?,?,?,?,?,?,?) ON CONFLICT DO NOTHING
    """, [experiment_id, datetime.now(timezone.utc).replace(tzinfo=None),
          result["status"], dataset_hash, _git_hash(), candidate,
          json.dumps({"retrospective_start": str(RETROSPECTIVE_HOLDOUT_START)}),
          json.dumps(results, default=str), json.dumps(gates, default=str)])
    con.close()
    if lock_spec:
        if not passed:
            result["specification_locked"] = False
        else:
            if SPEC_LOCK_PATH.exists():
                raise MGRMError("MGRM model specification already locked")
            numeric, categorical = specs[candidate]
            lock = {"model_spec_locked_at": datetime.now(timezone.utc).isoformat(),
                    "model_name": candidate, "numeric": numeric,
                    "categorical": categorical,
                    "elastic_net_params": params[candidate],
                    "target_version": TARGET_VERSION,
                    "beta_version": BETA_VERSION,
                    "extractor_version": EXTRACTOR_VERSION,
                    "linker_version": LINKER_VERSION,
                    "validation_experiment_id": experiment_id}
            lock["prospective_forward_start"] = first_session_after(
                lock["model_spec_locked_at"]
            )
            SPEC_LOCK_PATH.write_text(json.dumps(lock, indent=2))
            result["specification_locked"] = True
    REPORT_PATH.write_text(json.dumps(result, indent=2, default=str))
    if verbose:
        print(result["status"], candidate, gates)
    return result
