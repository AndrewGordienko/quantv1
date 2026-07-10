"""Multiway cluster-robust covariance for linear event-study models.

Implements the inclusion/exclusion sandwich estimator of Cameron, Gelbach and
Miller.  Actor studies should pass separate cluster arrays for catalyst/day,
ticker and actor.  This is materially different from resampling catalysts alone.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np


def _cluster_meat(x: np.ndarray, residuals: np.ndarray,
                  keys: np.ndarray) -> tuple[np.ndarray, int]:
    grouped: dict[object, np.ndarray] = {}
    for row, residual, key in zip(x, residuals, keys):
        score = row * residual
        grouped[key] = grouped.get(key, np.zeros(x.shape[1])) + score
    meat = sum((np.outer(score, score) for score in grouped.values()),
               start=np.zeros((x.shape[1], x.shape[1])))
    return meat, len(grouped)


def multiway_cluster_covariance(
    x,
    residuals,
    clusters: dict[str, object],
    *,
    finite_sample: bool = True,
) -> np.ndarray:
    """Return covariance clustered across every supplied dimension.

    Parameters
    ----------
    x:
        ``n × k`` design matrix used for the fitted linear model.
    residuals:
        Length-``n`` fitted residual vector.
    clusters:
        Mapping such as ``{"catalyst_day": ..., "ticker": ..., "actor": ...}``.
        Intersection cluster terms are constructed automatically.
    finite_sample:
        Apply the usual component-wise cluster degrees-of-freedom correction.
    """
    design = np.asarray(x, dtype=float)
    errors = np.asarray(residuals, dtype=float).reshape(-1)
    if design.ndim != 2 or len(errors) != design.shape[0]:
        raise ValueError("x must be n×k and residuals must have length n")
    if not clusters:
        raise ValueError("at least one cluster dimension is required")
    arrays = {name: np.asarray(values, dtype=object).reshape(-1)
              for name, values in clusters.items()}
    if any(len(values) != design.shape[0] for values in arrays.values()):
        raise ValueError("every cluster array must have length n")
    n, k = design.shape
    bread = np.linalg.pinv(design.T @ design)
    names = list(arrays)
    combined_meat = np.zeros((k, k))
    for width in range(1, len(names) + 1):
        sign = 1 if width % 2 else -1
        for selected in combinations(names, width):
            if width == 1:
                keys = arrays[selected[0]]
            else:
                keys = np.empty(n, dtype=object)
                keys[:] = list(zip(*(arrays[name] for name in selected)))
            meat, n_groups = _cluster_meat(design, errors, keys)
            if finite_sample and n_groups > 1 and n > k:
                meat *= (n_groups / (n_groups - 1)) * ((n - 1) / (n - k))
            combined_meat += sign * meat
    covariance = bread @ combined_meat @ bread
    return (covariance + covariance.T) / 2
