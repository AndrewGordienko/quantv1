"""Turn scored disclosures into a target portfolio.

Inputs: a frame of recently-filed, model-scored purchases (one row per trade,
with `score`, `ticker`, `sector`, `member`, optional `contribs`). Output: a
target book — a set of tickers with weights that respect a per-name cap and a
per-sector cap, taking the highest-scoring names.

Multiple members may disclose the same ticker; we collapse to one position per
ticker keeping the max score and recording every member behind it (the crowd
behind a name is itself part of the thesis).
"""

from __future__ import annotations

import pandas as pd

from ..config import MAX_POSITION_WEIGHT, MAX_SECTOR_WEIGHT, TOP_K


def _collapse_by_ticker(scored: pd.DataFrame) -> pd.DataFrame:
    agg = (scored.sort_values("score", ascending=False)
           .groupby("ticker")
           .agg(score=("score", "max"),
                sector=("sector", "first"),
                members=("member", lambda s: sorted(set(s))),
                n_members=("member", "nunique"))
           .reset_index())
    # carry the top contributor's SHAP rationale if present
    if "contribs" in scored.columns:
        top = (scored.sort_values("score", ascending=False)
               .drop_duplicates("ticker")[["ticker", "contribs"]])
        agg = agg.merge(top, on="ticker", how="left")
    return agg.sort_values("score", ascending=False).reset_index(drop=True)


def construct(scored: pd.DataFrame, *, top_k: int = TOP_K,
              score_threshold: float = 0.5,
              max_position: float = MAX_POSITION_WEIGHT,
              max_sector: float = MAX_SECTOR_WEIGHT,
              weighting: str = "score") -> pd.DataFrame:
    """Return a target book with columns [ticker, weight, score, sector, ...]."""
    if scored.empty:
        return scored.assign(weight=[])

    cand = _collapse_by_ticker(scored)
    cand = cand[cand["score"] >= score_threshold]
    if cand.empty:
        return cand.assign(weight=pd.Series(dtype=float))

    # Greedy pick top names subject to the per-sector cap.
    picks, sector_wt = [], {}
    for r in cand.itertuples(index=False):
        if len(picks) >= top_k:
            break
        sec = r.sector or "Unknown"
        if sector_wt.get(sec, 0.0) + max_position > max_sector + 1e-9:
            continue  # would breach sector cap; skip to next name
        picks.append(r)
        sector_wt[sec] = sector_wt.get(sec, 0.0) + max_position
    if not picks:
        return cand.head(0).assign(weight=pd.Series(dtype=float))

    book = pd.DataFrame(picks)
    if weighting == "equal":
        raw = pd.Series(1.0, index=book.index)
    else:  # score-weighted (scores are probabilities, already in [0,1])
        raw = book["score"].clip(lower=1e-6)
    w = _cap_weights(raw.to_numpy(), max_position)
    book = book.assign(weight=w).sort_values("weight", ascending=False)
    return book.reset_index(drop=True)


def _cap_weights(raw: "np.ndarray", cap: float) -> "np.ndarray":
    """Normalize `raw` to weights that never exceed `cap`, via water-filling.

    Excess from capped names is redistributed to uncapped names (not blindly
    renormalized, which re-breaches the cap). If there aren't enough names to
    absorb it (n * cap < 1), the leftover stays as CASH — weights sum to < 1
    rather than being force-invested. Fixes the clip-then-renormalize bug.
    """
    import numpy as np
    w = raw / raw.sum() if raw.sum() > 0 else np.zeros_like(raw)
    for _ in range(100):
        over = w > cap + 1e-12
        if not over.any():
            break
        excess = float((w[over] - cap).sum())
        w[over] = cap
        under = ~over & (w > 0)
        if not under.any():
            break  # everyone capped -> remainder becomes cash
        w[under] += excess * (w[under] / w[under].sum())
    return np.minimum(w, cap)


def diff_books(old: pd.DataFrame, new: pd.DataFrame) -> dict:
    """Buy/sell/rebalance deltas between yesterday's and today's target books."""
    old_w = dict(zip(old["ticker"], old["weight"])) if not old.empty else {}
    new_w = dict(zip(new["ticker"], new["weight"])) if not new.empty else {}
    buys = [{"ticker": t, "weight": round(new_w[t], 4)}
            for t in new_w if t not in old_w]
    sells = [{"ticker": t, "old_weight": round(old_w[t], 4)}
             for t in old_w if t not in new_w]
    rebal = [{"ticker": t, "old_weight": round(old_w[t], 4),
              "new_weight": round(new_w[t], 4)}
             for t in new_w if t in old_w and abs(new_w[t] - old_w[t]) > 0.005]
    return {"buys": buys, "sells": sells, "rebalances": rebal}
