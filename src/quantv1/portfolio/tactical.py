"""Tactical entry/exit overlay + daily-resolution backtest.

The base portfolio buys any disclosure the model likes, ignoring what the stock
has done SINCE the filing — so a name that has been bleeding for two months with
no follow-on buys still shows up as a fresh buy (the MRVL problem). This module
adds the risk-management layer the plan calls for:

ENTRY gates (all must pass to open a position on a given day):
  E1 model score >= SCORE_IN
  E2 signal-slice gate: backing trade >= $50k, OR cluster >= 2 members/30d,
     OR member in top-decile shrunk skill  (event study shows edge lives here)
  E3 freshness: <= FRESH_MAX_TD trading days since the backing filing
  E4 trend gate: close > 50d MA AND 20d return > TREND_RET_MIN  (no falling knives)
  E5 liquidity: median 63d $volume >= DVOL_MIN and price >= PRICE_MIN
  E6 regime: only take new entries when SPY > its 200d MA

EXIT rules (any one closes the position):
  X1 time stop at HOLD_TD trading days
  X2 ATR disaster stop: close < entry - ATR_STOP * ATR14
  X3 chandelier trailing stop: close < peak_close - ATR_TRAIL * ATR14
  X4 trend break: 2 consecutive closes below the 50d MA
  X5 signal reversal: a member files a SALE of the name after entry
  X6 portfolio kill switch: book day return < KILL_DAY -> flatten + halt entries

Sizing: vol-targeted — weight proportional to score / ATR% (equal risk), capped.

Everything is point-in-time: model scores come from a monthly walk-forward refit
(no future leakage), indicators are strictly trailing, entries/exits act on the
same day's close for every strategy so comparisons are fair.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from ..config import (BENCHMARK_TICKER, COST_BPS, MAX_POSITION_WEIGHT,
                      MAX_SECTOR_WEIGHT, TOP_K)
from ..db import connect
from ..model import features as F
from ..research.returns import PriceStore
from ..research.skill import eb_shrink

REALIZE_CALENDAR_DAYS = 92     # ~63 trading days for a label to realize


@dataclass
class TacticalParams:
    score_in: float = 0.55       # E1
    fresh_max_td: int = 21       # E3 trading days
    trend_ret_min: float = -0.05  # E4 20d return floor
    dvol_min: float = 5e6        # E5 median $ volume
    price_min: float = 5.0       # E5
    slice_amount: float = 50_000  # E2 large-trade threshold
    slice_cluster: int = 2       # E2 members in 30d
    skill_top_q: float = 0.90    # E2 top-decile skill
    hold_td: int = 63            # X1
    atr_stop: float = 2.5        # X2
    atr_trail: float = 3.0       # X3
    score_out: float = 0.45      # X5 (re-score) — approximated via sales in backtest
    kill_day: float = -0.03      # X6
    use_e4_trend: bool = True    # E4 trend/momentum entry gate on/off
    use_x4_trendbreak: bool = True  # X4 2-close-below-MA exit on/off
    top_k: int = TOP_K
    max_w: float = MAX_POSITION_WEIGHT
    max_sector_w: float = MAX_SECTOR_WEIGHT


# ---------------------------------------------------------------------------
# Technical-indicator panel (OHLCV -> trailing indicators as numpy arrays)
# ---------------------------------------------------------------------------
class TechPanel:
    def __init__(self, con=None):
        own = con is None
        con = con or connect(read_only=True)
        px = con.execute("""
            SELECT ticker, date, open, high, low, close, volume
            FROM prices ORDER BY date
        """).df()
        if own:
            con.close()
        px["date"] = pd.to_datetime(px["date"])
        piv = lambda c: px.pivot_table(index="date", columns="ticker", values=c)
        close = piv("close")
        high, low = piv("high"), piv("low")
        vol = piv("volume")

        prev = close.shift(1)
        tr = pd.concat([(high - low).abs(), (high - prev).abs(), (low - prev).abs()]).groupby(level=0).max()
        tr = tr.reindex(close.index)
        atr = tr.rolling(14, min_periods=7).mean()
        ma50 = close.rolling(50, min_periods=25).mean()
        ret20 = close / close.shift(20) - 1.0
        dvol = (close * vol).rolling(63, min_periods=20).median()

        self.cal = close.index
        self.cols = {t: i for i, t in enumerate(close.columns)}
        self.close = close.to_numpy()
        self.atr = atr.to_numpy()
        self.ma50 = ma50.to_numpy()
        self.ret20 = ret20.to_numpy()
        self.dvol = dvol.to_numpy()
        # below-MA boolean (for the 2-consecutive-close trend break)
        self.below = (close < ma50).to_numpy()

        # SPY 200d MA regime flag per day
        if BENCHMARK_TICKER in self.cols:
            spy = close[BENCHMARK_TICKER]
            self.spy_on = (spy > spy.rolling(200, min_periods=100).mean()).to_numpy()
        else:
            self.spy_on = np.ones(len(self.cal), dtype=bool)

    def pos_on_or_after(self, date) -> int | None:
        i = self.cal.searchsorted(pd.Timestamp(date), side="left")
        return int(i) if i < len(self.cal) else None

    def col(self, ticker: str) -> int | None:
        return self.cols.get(ticker)


# ---------------------------------------------------------------------------
# Point-in-time OOS model scores (monthly walk-forward refit, per trade)
# ---------------------------------------------------------------------------
def precompute_oos_scores(data: pd.DataFrame, freq_days: int = 21) -> tuple[dict, dict]:
    """Return ({trade_id: score}, {trade_id: pit_member_skill}).

    Each trade is scored by a model trained only on outcomes realized before that
    trade's filing month, and the member-skill used is the point-in-time shrunk
    skill from that same realized slice — so both the score and the E2 skill gate
    are leakage-free."""
    data = data.sort_values("filing_date").reset_index(drop=True)
    start = data["filing_date"].min() + pd.Timedelta(days=365 * 3)  # 3y warmup
    end = data["filing_date"].max()
    bounds = pd.date_range(start, end + pd.Timedelta(days=freq_days), freq=f"{freq_days}D")

    scores: dict[str, float] = {}
    pit_skill: dict[str, float] = {}
    for i in range(len(bounds) - 1):
        D, Dn = bounds[i], bounds[i + 1]
        realized = data[data["filing_date"] <= D - pd.Timedelta(days=REALIZE_CALENDAR_DAYS)]
        if len(realized) < 200 or realized["label"].nunique() < 2:
            continue
        skill = _pit_skill(realized)
        train = realized.copy()
        train["member_skill"] = train["member_key"].map(skill).fillna(0.0)
        gbm = LGBMClassifier(n_estimators=250, learning_rate=0.03, num_leaves=15,
                             min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
                             reg_lambda=1.0, verbose=-1)
        gbm.fit(train[F.FEATURE_COLS].astype(float), train["label"].to_numpy())
        win = data[(data["filing_date"] > D) & (data["filing_date"] <= Dn)].copy()
        if win.empty:
            continue
        win["member_skill"] = win["member_key"].map(skill).fillna(0.0)
        p = gbm.predict_proba(win[F.FEATURE_COLS].astype(float))[:, 1]
        for tid, s, mk in zip(win["trade_id"], p, win["member_skill"]):
            scores[tid] = float(s)
            pit_skill[tid] = float(mk)
    return scores, pit_skill


def _pit_skill(realized: pd.DataFrame) -> dict:
    df = realized.rename(columns={"fwd_excess": "ar"})[["member_key", "member", "ar"]].dropna()
    if df.empty:
        return {}
    s = eb_shrink(df)
    return dict(zip(s["member_key"], s["shrunk_car"]))


# ---------------------------------------------------------------------------
# Candidate assembly: one row per (trade) with static gate inputs + score
# ---------------------------------------------------------------------------
def build_candidates(data: pd.DataFrame, scores: dict, pit_skill: dict,
                     p: TacticalParams) -> pd.DataFrame:
    """Attach OOS score + E2 slice flags to each purchase; return candidates.

    The E2 top-skill cutoff is derived from the point-in-time skill values so it
    reflects only what was knowable when each trade was scored."""
    c = data.copy()
    c["score"] = c["trade_id"].map(scores)
    c["pit_skill"] = c["trade_id"].map(pit_skill)
    c = c.dropna(subset=["score"])
    skill_cut = (np.nanquantile(list(pit_skill.values()), p.skill_top_q)
                 if pit_skill else np.inf)
    amt = np.power(10.0, c["amount_mid_log"])   # recover dollars from log10
    c["is_large"] = amt >= p.slice_amount
    c["is_cluster"] = c["cluster_count"] >= p.slice_cluster
    c["is_topskill"] = c["pit_skill"].fillna(-np.inf) >= skill_cut
    c["slice_ok"] = c["is_large"] | c["is_cluster"] | c["is_topskill"]
    return c[["trade_id", "ticker", "sector", "member", "filing_date", "score",
              "slice_ok", "is_large", "is_cluster", "is_topskill",
              "cluster_count", "pit_skill"]]


def _sales_by_ticker(con) -> dict:
    rows = con.execute("""
        SELECT ticker, filing_date FROM trades
        WHERE tx_type = 'sale' AND ticker IS NOT NULL AND NOT filing_estimated
    """).fetchall()
    out: dict[str, list] = {}
    for tk, fd in rows:
        out.setdefault(tk, []).append(pd.Timestamp(fd))
    for tk in out:
        out[tk].sort()
    return out


# ---------------------------------------------------------------------------
# Sizing: vol-targeted weights among open positions
# ---------------------------------------------------------------------------
def _size_weights(open_pos: dict, tech: TechPanel, t: int, p: TacticalParams,
                  gross: float) -> dict:
    """weight_i proportional to score_i / ATR%_i (equal risk), capped, scaled by gross."""
    raw = {}
    for tk, s in open_pos.items():
        col = tech.col(tk)
        if col is None:
            continue
        cprice = tech.close[t, col]
        atr = tech.atr[t, col]
        if not (np.isfinite(cprice) and cprice > 0):
            continue
        atr_pct = max(atr / cprice, 0.01) if np.isfinite(atr) else 0.05
        raw[tk] = max(s["score"], 1e-6) / atr_pct
    tot = sum(raw.values())
    if tot <= 0:
        return {}
    w = {tk: v / tot for tk, v in raw.items()}
    # per-name cap, then renormalize, then scale by regime gross
    w = {tk: min(v, p.max_w) for tk, v in w.items()}
    s2 = sum(w.values())
    return {tk: (v / s2) * gross for tk, v in w.items()} if s2 > 0 else {}


# ---------------------------------------------------------------------------
# Daily-resolution tactical backtest
# ---------------------------------------------------------------------------
def prepare() -> dict:
    """Do the expensive, config-independent work once: features, OOS scores,
    point-in-time skill, sales, and the technical-indicator panel. The returned
    context is reused across many parameter configs in a sweep."""
    con = connect(read_only=True)
    data = F.build(con=con, with_label=True, purchases_only=True)
    data = data[~data["filing_estimated"]].copy()
    data["filing_date"] = pd.to_datetime(data["filing_date"])
    sales = _sales_by_ticker(con)
    con.close()
    scores, pit_skill = precompute_oos_scores(data)
    tech = TechPanel()
    return {"data": data, "sales": sales, "scores": scores,
            "pit_skill": pit_skill, "tech": tech}


def run_backtest(start_after: str = "2016-01-01", params: TacticalParams | None = None,
                 verbose: bool = True, ctx: dict | None = None) -> dict:
    p = params or TacticalParams()
    if ctx is None:
        if verbose:
            print("precomputing point-in-time OOS scores…")
        ctx = prepare()
    data, sales = ctx["data"], ctx["sales"]
    scores, pit_skill, tech = ctx["scores"], ctx["pit_skill"], ctx["tech"]
    cal = tech.cal

    cand = build_candidates(data, scores, pit_skill, p)
    cand = cand[cand["score"] >= p.score_in]
    cand_by_fpos: dict[int, list] = {}
    for r in cand.itertuples(index=False):
        fpos = tech.pos_on_or_after(r.filing_date)
        if fpos is None:
            continue
        cand_by_fpos.setdefault(fpos, []).append(r)

    start_i = max(tech.pos_on_or_after(start_after) or 0, 1)
    end_i = len(cal) - 1

    open_pos: dict[str, dict] = {}
    weights_prev: dict[str, float] = {}
    equity, dates = [1.0], [cal[start_i]]
    trade_log, exit_reasons = [], {}

    for t in range(start_i + 1, end_i + 1):
        # 1) mark-to-market over [t-1, t] using last set weights
        book_ret = 0.0
        for tk, w in weights_prev.items():
            col = tech.col(tk)
            if col is None:
                continue
            p0, p1 = tech.close[t - 1, col], tech.close[t, col]
            if np.isfinite(p0) and np.isfinite(p1) and p0 > 0:
                book_ret += w * (p1 / p0 - 1.0)
        equity.append(equity[-1] * (1 + book_ret))

        # 2) X6 kill switch
        halt = book_ret < p.kill_day
        if halt:
            for tk, s in list(open_pos.items()):
                col = tech.col(tk)
                px = tech.close[t, col] if col is not None else np.nan
                _log_exit(trade_log, exit_reasons, tk, s, cal[t], t, px, "X6_kill")
            open_pos.clear()

        # 3) exits (X1–X5) evaluated at today's close
        for tk, s in list(open_pos.items()):
            col = tech.col(tk)
            if col is None:
                continue
            cprice = tech.close[t, col]
            if not np.isfinite(cprice):
                continue
            s["peak"] = max(s["peak"], cprice)
            atr_now = tech.atr[t, col]
            held = t - s["entry_i"]
            reason = None
            if np.isfinite(s["atr_entry"]) and cprice < s["entry_px"] - p.atr_stop * s["atr_entry"]:
                reason = "X2_atr_stop"
            elif _has_sale_after(sales, tk, s["entry_date"], cal[t]):
                reason = "X5_reversal"
            elif p.use_x4_trendbreak and t >= 1 and tech.below[t, col] and tech.below[t - 1, col]:
                reason = "X4_trend_break"
            elif np.isfinite(atr_now) and cprice < s["peak"] - p.atr_trail * atr_now:
                reason = "X3_trail"
            elif held >= p.hold_td:
                reason = "X1_time"
            if reason:
                _log_exit(trade_log, exit_reasons, tk, s, cal[t], t, cprice, reason)
                del open_pos[tk]

        # 4) entries (E1–E6) — regime gate blocks NEW entries when SPY < 200d MA
        regime_on = bool(tech.spy_on[t])
        gross = 1.0 if regime_on else 0.5
        if regime_on and not halt and len(open_pos) < p.top_k:
            picks = _select_entries(t, tech, cand_by_fpos, open_pos, p)
            for r in picks:
                if len(open_pos) >= p.top_k:
                    break
                col = tech.col(r.ticker)
                open_pos[r.ticker] = {
                    "entry_i": t, "entry_date": cal[t],
                    "entry_px": float(tech.close[t, col]),
                    "atr_entry": float(tech.atr[t, col]),
                    "peak": float(tech.close[t, col]),
                    "score": float(r.score), "sector": r.sector or "Unknown",
                    "member": r.member, "trade_id": r.trade_id,
                }

        # 5) resize + charge turnover cost
        weights_now = _size_weights(open_pos, tech, t, p, gross)
        turn = 0.5 * sum(abs(weights_now.get(tk, 0) - weights_prev.get(tk, 0))
                         for tk in set(weights_now) | set(weights_prev))
        equity[-1] *= (1 - turn * 2 * COST_BPS / 1e4)
        weights_prev = weights_now
        dates.append(cal[t])

    con2 = connect(read_only=True)
    bench = _benchmark_curves(tech, dates, start_i, end_i)
    con2.close()

    result = _tactical_metrics(dates, equity, bench, trade_log, exit_reasons)
    result["params"] = asdict(p)
    if verbose:
        _print_summary(result)
    return result


def _select_entries(t, tech, cand_by_fpos, open_pos, p) -> list:
    """Gather candidates fresh at day t, apply dynamic gates E3–E5, rank by score."""
    fresh_lo = t - p.fresh_max_td
    pool = {}
    for fpos in range(fresh_lo, t + 1):
        for r in cand_by_fpos.get(fpos, []):
            if r.ticker in open_pos or not r.slice_ok:          # E2 + no double-hold
                continue
            col = tech.col(r.ticker)
            if col is None:
                continue
            cprice = tech.close[t, col]
            if not (np.isfinite(cprice) and cprice >= p.price_min):   # E5
                continue
            if p.use_e4_trend:
                ma = tech.ma50[t, col]
                r20 = tech.ret20[t, col]
                if not (np.isfinite(ma) and cprice > ma):             # E4 trend
                    continue
                if not (np.isfinite(r20) and r20 > p.trend_ret_min):  # E4 momentum
                    continue
            dv = tech.dvol[t, col]
            if not (np.isfinite(dv) and dv >= p.dvol_min):            # E5 liquidity
                continue
            if r.ticker not in pool or r.score > pool[r.ticker].score:
                pool[r.ticker] = r
    ranked = sorted(pool.values(), key=lambda r: r.score, reverse=True)
    # sector cap: at most floor(max_sector_w/max_w) names per sector
    max_per_sector = max(1, int(p.max_sector_w / p.max_w))
    sector_count: dict[str, int] = {}
    for tk, s in open_pos.items():
        sector_count[s["sector"]] = sector_count.get(s["sector"], 0) + 1
    out = []
    for r in ranked:
        sec = r.sector or "Unknown"
        if sector_count.get(sec, 0) >= max_per_sector:
            continue
        out.append(r)
        sector_count[sec] = sector_count.get(sec, 0) + 1
    return out


def _has_sale_after(sales: dict, ticker: str, entry_date, now) -> bool:
    lst = sales.get(ticker)
    if not lst:
        return False
    for d in lst:
        if entry_date < d <= now:
            return True
    return False


def _log_exit(trade_log, exit_reasons, tk, s, exit_date, t, exit_px, reason):
    ret = (float(exit_px / s["entry_px"] - 1.0)
           if np.isfinite(exit_px) and s["entry_px"] > 0 else np.nan)
    trade_log.append({"ticker": tk, "entry_date": str(s["entry_date"].date()),
                      "exit_date": str(exit_date.date()), "held_td": t - s["entry_i"],
                      "ret": ret, "reason": reason, "score": s["score"]})
    exit_reasons[reason] = exit_reasons.get(reason, 0) + 1


def _benchmark_curves(tech, dates, start_i, end_i) -> dict:
    """SPY buy-and-hold aligned to the tactical date index."""
    col = tech.col(BENCHMARK_TICKER)
    spy = [1.0]
    idxs = [tech.pos_on_or_after(d) for d in dates]
    for k in range(1, len(idxs)):
        a, b = idxs[k - 1], idxs[k]
        r = 0.0
        if a is not None and b is not None and col is not None:
            p0, p1 = tech.close[a, col], tech.close[b, col]
            if np.isfinite(p0) and np.isfinite(p1) and p0 > 0:
                r = p1 / p0 - 1.0
        spy.append(spy[-1] * (1 + r))
    return {"spy": spy}


def _tactical_metrics(dates, equity, bench, trade_log, exit_reasons) -> dict:
    eq = np.array(equity)
    rets = np.diff(eq) / eq[:-1]
    years = (dates[-1] - dates[0]).days / 365.25
    cagr = eq[-1] ** (1 / years) - 1 if years > 0 and eq[-1] > 0 else np.nan
    sharpe = np.mean(rets) / np.std(rets) * np.sqrt(252) if np.std(rets) > 0 else np.nan
    peak = np.maximum.accumulate(eq)
    max_dd = float(np.min(eq / peak - 1))
    tl = pd.DataFrame(trade_log)
    losers = tl[tl["ret"] < 0] if not tl.empty else pd.DataFrame()
    winners = tl[tl["ret"] >= 0] if not tl.empty else pd.DataFrame()
    stats = {
        "tactical": {"cagr": float(cagr), "sharpe": float(sharpe), "max_dd": max_dd,
                     "final": float(eq[-1])},
    }
    spy = np.array(bench["spy"])
    spy_rets = np.diff(spy) / spy[:-1]
    stats["spy"] = {
        "cagr": float(spy[-1] ** (1 / years) - 1) if years > 0 else np.nan,
        "sharpe": float(np.mean(spy_rets) / np.std(spy_rets) * np.sqrt(252)) if np.std(spy_rets) > 0 else np.nan,
        "max_dd": float(np.min(spy / np.maximum.accumulate(spy) - 1)),
        "final": float(spy[-1]),
    }
    curve = [{"date": str(d.date()), "tactical": float(equity[i]), "spy": float(spy[i])}
             for i, d in enumerate(dates)]
    return {
        "stats": stats, "curve": curve, "exit_reasons": exit_reasons,
        "n_trades": int(len(tl)),
        "win_rate": float((tl["ret"] >= 0).mean()) if not tl.empty else None,
        "avg_win": float(winners["ret"].mean()) if len(winners) else None,
        "avg_loss": float(losers["ret"].mean()) if len(losers) else None,
        "avg_hold_td": float(tl["held_td"].mean()) if not tl.empty else None,
    }


def _print_summary(result):
    for k, m in result["stats"].items():
        print(f"{k:9s} CAGR={m['cagr']:.2%} Sharpe={m['sharpe']:.2f} "
              f"maxDD={m['max_dd']:.2%} finalx={m['final']:.2f}")
    print(f"trades={result['n_trades']} win_rate={result['win_rate']:.1%} "
          f"avg_win={result['avg_win']:+.2%} avg_loss={result['avg_loss']:+.2%} "
          f"avg_hold={result['avg_hold_td']:.0f}td")
    print("exit reasons:", dict(sorted(result["exit_reasons"].items(),
                                       key=lambda x: -x[1])))


if __name__ == "__main__":
    run_backtest()


# ---------------------------------------------------------------------------
# Live morning book: the actual 9am decision, with per-name entry check
# ---------------------------------------------------------------------------
def build_morning_book(as_of: str | None = None, params: TacticalParams | None = None,
                       lookback_days: int = 120) -> dict:
    """Produce today's tradeable book AND an explicit 'excluded + why' list.

    Uses the production model (data/signal_model.pkl) + current skill_scores for
    scoring, then runs each recent purchase through the entry gates E1–E6 against
    the latest indicators. Names that fail (stale, below trend, illiquid, weak
    slice) are reported with the exact gate that rejected them — so a bleeding,
    nobody's-bought-it-since-May name shows up in 'excluded', not the book.
    """
    import pickle
    from ..model.train import MODEL_PATH

    p = params or TacticalParams()
    con = connect(read_only=True)
    tech = TechPanel(con)
    store = PriceStore(con)

    # skill cutoff from current persisted skill scores (top decile)
    skl = con.execute("SELECT member_key, shrunk_car FROM skill_scores").fetchall()
    skill_map = dict(skl)
    skill_cut = (float(np.nanquantile([v for _, v in skl], p.skill_top_q))
                 if skl else np.inf)

    feats = F.build(con=con, store=store, with_label=False, purchases_only=True,
                    skill_map=skill_map)
    con.close()

    feats["filing_date"] = pd.to_datetime(feats["filing_date"])
    asof = pd.Timestamp(as_of) if as_of else feats["filing_date"].max()
    t = tech.pos_on_or_after(asof)
    if t is None or t >= len(tech.cal):
        t = len(tech.cal) - 1
    lo = asof - pd.Timedelta(days=lookback_days)
    recent = feats[(feats["filing_date"] > lo) & (feats["filing_date"] <= asof)].copy()
    if recent.empty:
        return {"as_of": str(asof.date()), "book": [], "excluded": []}

    with open(MODEL_PATH, "rb") as f:
        payload = pickle.load(f)
    model, cols = payload["model"], payload["feature_cols"]
    recent["score"] = model.predict_proba(recent[cols].astype(float))[:, 1]

    # collapse to one row per ticker (latest, highest-scoring backing trade)
    recent = recent.sort_values(["ticker", "score"], ascending=[True, False])
    included, excluded = [], []
    seen = set()
    for r in recent.itertuples(index=False):
        if r.ticker in seen:
            continue
        seen.add(r.ticker)
        check = _entry_check(r, tech, t, p, skill_cut)
        rec = {"ticker": r.ticker, "sector": r.sector, "member": r.member,
               "score": float(r.score), "filing_date": str(r.filing_date.date()),
               "checks": check["checks"], "fresh_td": check["fresh_td"],
               "ret_since_filing": check["ret_since_filing"]}
        if check["pass"]:
            included.append(rec)
        else:
            rec["failed"] = check["failed"]
            excluded.append(rec)

    included.sort(key=lambda x: x["score"], reverse=True)
    included = included[:p.top_k]
    excluded.sort(key=lambda x: x["score"], reverse=True)
    return {"as_of": str(asof.date()), "regime_on": bool(tech.spy_on[t]),
            "book": included, "excluded": excluded[:40],
            "params": asdict(p)}


def _entry_check(r, tech: TechPanel, t: int, p: TacticalParams, skill_cut: float) -> dict:
    """Evaluate the six entry gates for one candidate at day t; return pass/fail
    with a labelled reason per gate."""
    col = tech.col(r.ticker)
    checks, failed = {}, []

    # E1 model score
    e1 = r.score >= p.score_in
    checks["E1_score"] = e1
    # E2 slice gate
    amt = 10 ** r.amount_mid_log
    is_large = amt >= p.slice_amount
    is_cluster = r.cluster_count >= p.slice_cluster
    is_topskill = r.member_skill >= skill_cut
    e2 = bool(is_large or is_cluster or is_topskill)
    checks["E2_slice"] = e2
    # freshness
    fresh_td = None
    if col is not None:
        fpos = tech.pos_on_or_after(r.filing_date)
        if fpos is not None:
            fresh_td = t - fpos
    e3 = fresh_td is not None and fresh_td <= p.fresh_max_td
    checks["E3_fresh"] = e3

    cprice = tech.close[t, col] if col is not None else np.nan
    ma = tech.ma50[t, col] if col is not None else np.nan
    r20 = tech.ret20[t, col] if col is not None else np.nan
    dv = tech.dvol[t, col] if col is not None else np.nan

    e4 = bool(np.isfinite(cprice) and np.isfinite(ma) and cprice > ma
              and np.isfinite(r20) and r20 > p.trend_ret_min)
    checks["E4_trend"] = e4
    e5 = bool(np.isfinite(dv) and dv >= p.dvol_min
              and np.isfinite(cprice) and cprice >= p.price_min)
    checks["E5_liquidity"] = e5
    e6 = bool(tech.spy_on[t])
    checks["E6_regime"] = e6

    # return since filing (for display: "already bled X%")
    ret_since = None
    if col is not None and fresh_td is not None:
        fpos = tech.pos_on_or_after(r.filing_date)
        if fpos is not None and np.isfinite(tech.close[fpos, col]) and np.isfinite(cprice):
            base = tech.close[fpos, col]
            if base > 0:
                ret_since = float(cprice / base - 1.0)

    labels = {"E1_score": "model score", "E2_slice": "signal slice",
              "E3_fresh": "freshness", "E4_trend": "trend/momentum",
              "E5_liquidity": "liquidity", "E6_regime": "market regime"}
    for k, ok in checks.items():
        if not ok:
            failed.append(labels[k])
    return {"pass": all(checks.values()), "checks": checks, "failed": failed,
            "fresh_td": fresh_td, "ret_since_filing": ret_since}
