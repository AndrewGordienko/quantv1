"""Politician skill scoring via empirical-Bayes shrinkage.

The trap: most members have very few trades, so ranking by raw mean return puts
whoever got lucky on 3 trades at the top. We instead treat each member's true
per-trade skill mu_i as drawn from a population prior N(mu0, tau^2), and shrink
each member's noisy sample mean toward the population mean in proportion to how
uncertain that member's estimate is (few trades -> shrink hard; many trades ->
trust the data). This is the classic James-Stein / batting-average estimator.

We report a posterior mean AND a 95% credible interval, so the leaderboard can
show uncertainty bars instead of pretending a 4-trade member's rank is precise.

Signal is measured on PURCHASES only (the deliberate act), using filing-date
abnormal returns at the label horizon.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..config import LABEL_HORIZON
from ..db import connect
from .returns import PriceStore


def eb_shrink(df: pd.DataFrame) -> pd.DataFrame:
    """Empirical-Bayes shrink per-member mean abnormal returns.

    Input: a frame with columns [member_key, member, ar] (one row per trade).
    Output: per-member frame with raw_car, shrunk_car, ci_low/high, hit_rate, n.
    Pure function (no DB) so the backtest can call it with any point-in-time slice.
    """
    # Group by member_key only; a single member can appear under slightly
    # different display spellings that normalize to the same key.
    name_by_key = df.groupby("member_key")["member"].agg(
        lambda s: s.value_counts().idxmax())
    grp = df.groupby("member_key")["ar"]
    stats = grp.agg(["mean", "var", "count"]).reset_index()
    stats = stats.rename(columns={"mean": "raw_car", "var": "within_var",
                                  "count": "n_purchases"})
    stats["member"] = stats["member_key"].map(name_by_key)
    if len(stats) < 2:
        stats["shrunk_car"] = stats["raw_car"]
        stats["ci_low"] = stats["ci_high"] = stats["raw_car"]
        stats["hit_rate"] = np.nan
        return stats

    pooled_var = float(df["ar"].var(ddof=1)) if len(df) > 2 else 1e-4
    stats["within_var"] = stats["within_var"].fillna(pooled_var)
    stats["se2"] = np.where(
        stats["n_purchases"] >= 5,
        stats["within_var"] / stats["n_purchases"],
        pooled_var / stats["n_purchases"],
    )
    mu0 = float(np.average(stats["raw_car"], weights=stats["n_purchases"]))
    grand_var = float(stats["raw_car"].var(ddof=1))
    tau2 = max(grand_var - float(stats["se2"].mean()), 1e-6)
    w = tau2 / (tau2 + stats["se2"])
    stats["shrunk_car"] = w * stats["raw_car"] + (1 - w) * mu0
    post_sd = np.sqrt(1.0 / (1.0 / tau2 + 1.0 / stats["se2"]))
    stats["ci_low"] = stats["shrunk_car"] - 1.96 * post_sd
    stats["ci_high"] = stats["shrunk_car"] + 1.96 * post_sd
    hit = df.assign(win=(df["ar"] > 0)).groupby("member_key")["win"].mean()
    stats["hit_rate"] = stats["member_key"].map(hit)
    stats.attrs["hyperparams"] = {"mu0": mu0, "tau2": tau2, "pooled_var": pooled_var}
    return stats


def _member_trade_ars(con, store: PriceStore, horizon: int) -> pd.DataFrame:
    trades = con.execute("""
        SELECT trade_id, member, member_key, ticker, filing_date
        FROM trades
        WHERE tx_type = 'purchase' AND ticker IS NOT NULL
        ORDER BY filing_date
    """).df()
    ars = [store.abnormal_return(r.ticker, r.filing_date, horizon, beta=1.0)
           for r in trades.itertuples(index=False)]
    trades["ar"] = ars
    return trades.dropna(subset=["ar"])


def compute(con=None, store: PriceStore | None = None,
            horizon: int = LABEL_HORIZON) -> pd.DataFrame:
    own = con is None
    con = con or connect(read_only=True)
    store = store or PriceStore(con)

    df = _member_trade_ars(con, store, horizon)
    n_all = dict(con.execute(
        "SELECT member_key, COUNT(*) FROM trades GROUP BY 1"
    ).fetchall())
    if own:
        con.close()

    stats = eb_shrink(df)
    stats["n_trades"] = stats["member_key"].map(n_all).fillna(stats["n_purchases"])
    stats = stats.sort_values("shrunk_car", ascending=False).reset_index(drop=True)
    return stats


def persist(stats: pd.DataFrame) -> None:
    con = connect()
    now = datetime.now(timezone.utc)
    con.execute("DELETE FROM skill_scores")
    con.executemany(
        "INSERT INTO skill_scores VALUES (?,?,?,?,?,?,?,?,?,?)",
        [[r.member_key, r.member, int(r.n_trades), int(r.n_purchases),
          float(r.raw_car), float(r.shrunk_car), float(r.ci_low),
          float(r.ci_high), float(r.hit_rate) if pd.notna(r.hit_rate) else None, now]
         for r in stats.itertuples(index=False)],
    )
    con.close()


def run() -> pd.DataFrame:
    store = PriceStore()
    stats = compute(store=store)
    persist(stats)
    return stats


if __name__ == "__main__":
    s = run()
    hp = s.attrs["hyperparams"]
    print(f"hyperparams: mu0={hp['mu0']:.4f} tau={np.sqrt(hp['tau2']):.4f}")
    print("\n=== Top 15 by shrunk CAR (>=5 purchases) ===")
    top = s[s["n_purchases"] >= 5].head(15)
    print(top[["member", "n_purchases", "raw_car", "shrunk_car",
               "ci_low", "ci_high", "hit_rate"]].to_string(index=False))
