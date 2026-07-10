"""Valid B1-versus-B2 predictive comparison for primary-source actor events.

B1 conditions on event semantics, stance, magnitude, asset, sector, regime,
volatility and time of day.  B2 adds regularized actor identity and actor×event
type/topic terms.  Ridge penalties give the actor terms a shared zero-centered
Gaussian prior (partial pooling); rare actors shrink toward the population.

The comparison is evaluated separately on held-out time and held-out speakers.
Actor permutations are restricted to actor-event role × semantic event type.
This module refuses context-only news mentions upstream via the loader.
"""

from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, GroupShuffleSplit, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..config import DATA_DIR
from ..db import connect

FORBIDDEN_MODEL_FEATURES = {
    "authority", "actor_power", "legacy_authority_prior", "exposure_confidence",
}
BASE_CATEGORICAL = [
    "semantic_event_type", "stance_label", "ticker", "sector", "regime",
    "topic", "time_of_day_bucket", "actor_event_role", "actor_asset_channel",
]
BASE_NUMERIC = ["stance", "magnitude", "pre_event_volatility"]
ACTOR_CATEGORICAL = ["actor_id", "actor_event_interaction", "actor_topic_interaction"]


@dataclass
class FittedComparison:
    model: Pipeline
    metrics: dict
    predictions: np.ndarray
    alpha: float


def load_frame(feature_version: str, outcome_version: str,
               horizon: str = "2h") -> pd.DataFrame:
    """Load only primary-eligible events with semantic features and outcomes."""
    con = connect(read_only=True)
    frame = con.execute("""
        SELECT ae.actor_event_id, ae.actor_id, ae.ticker, ae.public_time,
               ae.actor_event_role, ae.event_type, ae.catalyst_id,
               COALESCE(json_extract_string(ae.metadata, '$.actor_asset_channel'),
                        'unknown') AS actor_asset_channel,
               f.semantic_event_type, f.stance, f.magnitude, f.topic,
               f.regime, f.sector, f.pre_event_volatility,
               f.time_of_day_bucket,
               o.sector_beta_residual AS target_return
        FROM actor_events ae
        JOIN actor_event_features f ON f.actor_event_id=ae.actor_event_id
        JOIN actor_event_outcomes o
          ON o.actor_event_id=ae.actor_event_id AND o.ticker=ae.ticker
        WHERE ae.primary_hypothesis_eligible=TRUE
          AND ae.actor_event_role IN
              ('speaker_author','direct_public_action','verified_decision_maker')
          AND f.feature_version=? AND o.outcome_version=? AND o.horizon=?
          AND o.sector_beta_residual IS NOT NULL
    """, [feature_version, outcome_version, horizon]).df()
    con.close()
    return frame


def _prepare(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "actor_id", "ticker", "public_time", "actor_event_role",
        "semantic_event_type", "stance", "magnitude", "topic", "regime",
        "sector", "pre_event_volatility", "time_of_day_bucket",
        "actor_asset_channel", "target_return",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"actor B2 frame is missing columns: {sorted(missing)}")
    data = frame.copy()
    data["public_time"] = pd.to_datetime(data["public_time"], utc=True)
    data = data.sort_values("public_time").reset_index(drop=True)
    if not data["actor_event_role"].isin(
        ["speaker_author", "direct_public_action", "verified_decision_maker"]
    ).all():
        raise ValueError("context-only actor-event roles cannot enter B2")
    data["stance_label"] = pd.cut(
        pd.to_numeric(data["stance"], errors="coerce"),
        [-np.inf, -0.15, 0.15, np.inf], labels=["negative", "neutral", "positive"],
    ).astype(str)
    for column in BASE_CATEGORICAL + ["actor_id"]:
        data[column] = data[column].fillna("unknown").astype(str)
    for column in BASE_NUMERIC + ["target_return"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=BASE_NUMERIC + ["target_return"])
    data["actor_event_interaction"] = (
        data["actor_id"] + "::" + data["semantic_event_type"]
    )
    data["actor_topic_interaction"] = data["actor_id"] + "::" + data["topic"]
    data["catalyst_day"] = (
        data.get("catalyst_id", pd.Series(index=data.index, dtype=object))
        .fillna(data.get("actor_event_id", pd.Series(data.index, index=data.index)).astype(str))
        .astype(str) + "::" + data["public_time"].dt.strftime("%Y-%m-%d")
    )
    return data.reset_index(drop=True)


def _pipeline(categorical: list[str], numeric: list[str], alpha=None) -> Pipeline:
    transform = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("numeric", StandardScaler(), numeric),
    ])
    ridge = Ridge(alpha=1.0 if alpha is None else alpha)
    return Pipeline([("features", transform), ("ridge", ridge)])


def _trading_metrics(actual: np.ndarray, predicted: np.ndarray,
                     cost_bps: float) -> dict:
    cost = cost_bps / 1e4
    position = np.where(np.abs(predicted) > cost, np.sign(predicted), 0.0)
    net = position * actual - np.abs(position) * cost
    active = position != 0
    active_net = net[active]
    return {
        "n_events": int(len(actual)), "n_trades": int(active.sum()),
        "mean_net_bps_per_event": float(net.mean() * 1e4) if len(net) else None,
        "mean_net_bps_per_trade": float(active_net.mean() * 1e4) if len(active_net) else None,
        "total_net_return": float(active_net.sum()) if len(active_net) else 0.0,
        "hit_rate": float((active_net > 0).mean()) if len(active_net) else None,
        "event_information_ratio": (
            float(active_net.mean() / active_net.std(ddof=1))
            if len(active_net) > 1 and active_net.std(ddof=1) > 0 else None
        ),
        "cost_bps": cost_bps,
    }


def _fit(train: pd.DataFrame, test: pd.DataFrame, *, b2: bool,
         cost_bps: float, fixed_alpha: float | None = None) -> FittedComparison:
    categorical = BASE_CATEGORICAL + (ACTOR_CATEGORICAL if b2 else [])
    numeric = BASE_NUMERIC
    if fixed_alpha is None:
        splits = min(5, max(2, len(train) // 20))
        if len(train) <= splits:
            raise ValueError("not enough training observations for time-series CV")
        search = GridSearchCV(
            _pipeline(categorical, numeric),
            {"ridge__alpha": np.logspace(-2, 3, 10)},
            scoring="neg_mean_squared_error", cv=TimeSeriesSplit(n_splits=splits),
            n_jobs=1,
        )
        search.fit(train[categorical + numeric], train["target_return"])
        model = search.best_estimator_
        alpha = float(search.best_params_["ridge__alpha"])
    else:
        alpha = float(fixed_alpha)
        model = _pipeline(categorical, numeric, alpha)
        model.fit(train[categorical + numeric], train["target_return"])
    predicted = model.predict(test[categorical + numeric])
    actual = test["target_return"].to_numpy()
    train_predicted = model.predict(train[categorical + numeric])
    sigma = max(float(np.std(train["target_return"].to_numpy() - train_predicted, ddof=1)), 1e-9)
    gaussian_loss = float(np.mean(
        0.5 * np.log(2 * np.pi * sigma ** 2) + 0.5 * ((actual - predicted) / sigma) ** 2
    ))
    metrics = {
        "n_train": int(len(train)), "n_test": int(len(test)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mae": float(mean_absolute_error(actual, predicted)),
        "gaussian_predictive_loss": gaussian_loss,
        "ridge_alpha": alpha,
        "net_trading": _trading_metrics(actual, predicted, cost_bps),
    }
    return FittedComparison(model, metrics, predicted, alpha)


def _split_time(data: pd.DataFrame, test_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    cut = max(1, min(len(data) - 1, int(len(data) * (1 - test_fraction))))
    return data.iloc[:cut], data.iloc[cut:]


def _split_speakers(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if data["actor_id"].nunique() < 4:
        raise ValueError("held-out-speaker evaluation needs at least four speakers")
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=19)
    train_index, test_index = next(splitter.split(data, groups=data["actor_id"]))
    return (data.iloc[train_index].sort_values("public_time"),
            data.iloc[test_index].sort_values("public_time"))


def permute_actor_within_role_event(data: pd.DataFrame,
                                    rng: np.random.Generator) -> pd.DataFrame:
    """Permutation null that preserves actor-event role and event semantics."""
    permuted = data.copy()
    for _, indices in permuted.groupby(
        ["actor_event_role", "semantic_event_type"], dropna=False
    ).groups.items():
        positions = np.asarray(list(indices))
        permuted.loc[positions, "actor_id"] = rng.permutation(
            permuted.loc[positions, "actor_id"].to_numpy()
        )
    permuted["actor_event_interaction"] = (
        permuted["actor_id"] + "::" + permuted["semantic_event_type"]
    )
    permuted["actor_topic_interaction"] = permuted["actor_id"] + "::" + permuted["topic"]
    return permuted


def matched_strata_summary(data: pd.DataFrame) -> dict:
    """Audit support for the required within-condition matched comparisons."""
    frame = data.copy()
    frame["volatility_bin"] = pd.qcut(
        frame["pre_event_volatility"].rank(method="first"),
        q=min(5, len(frame)), labels=False, duplicates="drop",
    )
    keys = ["ticker", "semantic_event_type", "stance_label", "volatility_bin",
            "time_of_day_bucket", "regime"]
    eligible = 0
    observations = 0
    contrasts: dict[tuple[str, str], list[float]] = {}
    for _, group in frame.groupby(keys, dropna=False):
        if group["actor_id"].nunique() >= 2:
            eligible += 1
            observations += len(group)
            actor_means = group.groupby("actor_id")["target_return"].mean()
            actors = sorted(actor_means.index)
            for i, left in enumerate(actors):
                for right in actors[i + 1:]:
                    contrasts.setdefault((left, right), []).append(
                        float(actor_means[left] - actor_means[right])
                    )
    pairwise = {
        f"{left}_minus_{right}": {
            "n_matched_strata": len(values),
            "mean_difference_bps": float(np.mean(values) * 1e4),
        }
        for (left, right), values in sorted(contrasts.items())
    }
    return {"matched_strata": eligible, "matched_observations": observations,
            "strata_definition": keys, "pairwise_actor_contrasts": pairwise}


def _evaluation(train: pd.DataFrame, test: pd.DataFrame, cost_bps: float,
                n_permutations: int, seed: int) -> dict:
    b1 = _fit(train, test, b2=False, cost_bps=cost_bps)
    b2 = _fit(train, test, b2=True, cost_bps=cost_bps)
    result = {
        "B1_semantic": b1.metrics,
        "B2_hierarchical_actor": b2.metrics,
        "B2_minus_B1": {
            "rmse_improvement": b1.metrics["rmse"] - b2.metrics["rmse"],
            "predictive_loss_improvement": (
                b1.metrics["gaussian_predictive_loss"] -
                b2.metrics["gaussian_predictive_loss"]
            ),
            "net_bps_per_event_improvement": (
                b2.metrics["net_trading"]["mean_net_bps_per_event"] -
                b1.metrics["net_trading"]["mean_net_bps_per_event"]
            ),
        },
    }
    if n_permutations:
        rng = np.random.default_rng(seed)
        observed = result["B2_minus_B1"]["rmse_improvement"]
        null = []
        for _ in range(n_permutations):
            permuted = permute_actor_within_role_event(train, rng)
            fit = _fit(permuted, test, b2=True, cost_bps=cost_bps,
                       fixed_alpha=b2.alpha)
            null.append(b1.metrics["rmse"] - fit.metrics["rmse"])
        result["within_role_event_actor_permutation"] = {
            "n": n_permutations, "observed_rmse_improvement": observed,
            "null_mean": float(np.mean(null)),
            "one_sided_p": float((1 + np.sum(np.asarray(null) >= observed)) /
                                 (n_permutations + 1)),
        }
    return result


def run(frame: pd.DataFrame | None = None, *, feature_version: str | None = None,
        outcome_version: str | None = None, horizon: str = "2h",
        cost_bps: float = 5.0, n_permutations: int = 100,
        verbose: bool = True) -> dict:
    if frame is None:
        if not feature_version or not outcome_version:
            raise ValueError("feature_version and outcome_version are required when frame is omitted")
        frame = load_frame(feature_version, outcome_version, horizon)
    data = _prepare(frame)
    if len(data) < 50:
        raise ValueError("valid B1/B2 evaluation requires at least 50 primary-source observations")
    time_train, time_test = _split_time(data)
    speaker_train, speaker_test = _split_speakers(data)
    result = {
        "study_status": "VALID_PRIMARY_SOURCE_B1_B2",
        "hypothesis": (
            "Does actor identity improve held-out market-impact prediction after "
            "conditioning on communication semantics, asset and market state?"
        ),
        "authority_metadata_used": False,
        "forbidden_model_features": sorted(FORBIDDEN_MODEL_FEATURES),
        "actor_shrinkage": "shared zero-centered Gaussian/ridge prior",
        "held_out_time": _evaluation(time_train, time_test, cost_bps,
                                     n_permutations, 31),
        "held_out_speakers": _evaluation(speaker_train, speaker_test, cost_bps,
                                         n_permutations, 37),
        "matched_comparison_audit": matched_strata_summary(data),
        "coefficient_inference": {
            "required_clusters": ["catalyst_day", "ticker", "actor"],
            "note": "predictive comparison is primary; use multiway covariance for coefficient claims",
        },
    }
    with open(DATA_DIR / "actor_b2_primary.json", "w") as file:
        json.dump(result, file, indent=2)
    if verbose:
        time_lift = result["held_out_time"]["B2_minus_B1"]
        speaker_lift = result["held_out_speakers"]["B2_minus_B1"]
        print("=== Primary-source B1 vs B2 ===")
        print(f"held-out time RMSE improvement: {time_lift['rmse_improvement']:+.6f}")
        print(f"held-out speaker RMSE improvement: {speaker_lift['rmse_improvement']:+.6f}")
    return result
