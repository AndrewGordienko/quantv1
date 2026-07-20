"""FastAPI backend: serves JSON from DuckDB for the dashboard.

Read-only and stateless — every request opens a short-lived read-only DuckDB
connection so it never contends with the daily writer for the file lock. All
heavy computation happens offline in scripts/daily_update.py; the API just reads
the derived tables (portfolios, signals, skill_scores, backtest_equity).
"""

from __future__ import annotations

import json

import duckdb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from ..config import DB_PATH

app = FastAPI(title="quantv1 — Congressional Alpha Engine")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


def q(sql: str, params: list | None = None) -> list[dict]:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        cur = con.execute(sql, params or [])
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        con.close()


def _loads(v):
    if isinstance(v, str):
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            return v
    return v


@app.get("/api/health")
def health():
    try:
        n = q("SELECT COUNT(*) AS n FROM trades")[0]["n"]
        return {"ok": True, "trades": n}
    except duckdb.Error as e:
        raise HTTPException(503, str(e))


@app.get("/api/stats")
def stats():
    row = q("""
        SELECT COUNT(*) AS trades,
               COUNT(DISTINCT member_key) AS members,
               COUNT(DISTINCT ticker) AS tickers,
               MAX(filing_date) AS latest_filing
        FROM trades
    """)[0]
    row["prices"] = q("SELECT COUNT(*) AS n FROM prices")[0]["n"]
    return row


@app.get("/api/portfolio/today")
def portfolio_today():
    dates = q("SELECT DISTINCT as_of_date FROM portfolios ORDER BY as_of_date DESC LIMIT 2")
    if not dates:
        return {"as_of_date": None, "positions": [], "deltas": {}}
    latest = dates[0]["as_of_date"]
    positions = q("""
        SELECT ticker, weight, score, rationale FROM portfolios
        WHERE as_of_date = ? ORDER BY weight DESC
    """, [latest])
    # Attach each name's latest close so the client can size share counts.
    tickers = [p["ticker"] for p in positions]
    price_map = {}
    if tickers:
        ph = ",".join(["?"] * len(tickers))
        for r in q(f"""
            SELECT ticker, arg_max(close, date) AS price, max(date) AS price_date
            FROM prices WHERE ticker IN ({ph}) GROUP BY ticker
        """, tickers):
            price_map[r["ticker"]] = (r["price"], str(r["price_date"]))
    for p in positions:
        p["rationale"] = _loads(p["rationale"])
        pr = price_map.get(p["ticker"])
        p["price"] = pr[0] if pr else None
        p["price_date"] = pr[1] if pr else None
    prev = q("SELECT ticker, weight FROM portfolios WHERE as_of_date = ?",
             [dates[1]["as_of_date"]]) if len(dates) > 1 else []
    old = {r["ticker"]: r["weight"] for r in prev}
    new = {p["ticker"]: p["weight"] for p in positions}
    deltas = {
        "buys": [t for t in new if t not in old],
        "sells": [t for t in old if t not in new],
    }
    return {"as_of_date": str(latest), "positions": positions, "deltas": deltas}


@app.get("/api/feed")
def feed(limit: int = 60):
    rows = q("""
        SELECT s.filing_date, s.ticker, s.member, s.score, s.contribs,
               t.tx_date, t.amount_mid, t.chamber, t.disclosure_lag
        FROM signals s JOIN trades t USING (trade_id)
        ORDER BY s.filing_date DESC, s.score DESC
        LIMIT ?
    """, [limit])
    for r in rows:
        r["contribs"] = _loads(r["contribs"])
        r["filing_date"] = str(r["filing_date"])
        r["tx_date"] = str(r["tx_date"])
    return rows


@app.get("/api/leaderboard")
def leaderboard(min_purchases: int = 5):
    rows = q("""
        SELECT member, n_trades, n_purchases, raw_car, shrunk_car,
               ci_low, ci_high, hit_rate, member_key
        FROM skill_scores
        WHERE n_purchases >= ?
        ORDER BY shrunk_car DESC
    """, [min_purchases])
    return rows


@app.get("/api/member/{member_key}")
def member(member_key: str):
    info = q("""
        SELECT member, chamber, party, state, committees
        FROM members WHERE member_key = ?
    """, [member_key])
    skill = q("SELECT * FROM skill_scores WHERE member_key = ?", [member_key])
    trades = q("""
        SELECT tx_date, filing_date, ticker, tx_type, amount_mid, owner
        FROM trades WHERE member_key = ?
        ORDER BY filing_date DESC LIMIT 300
    """, [member_key])
    for t in trades:
        t["tx_date"] = str(t["tx_date"])
        t["filing_date"] = str(t["filing_date"])
    prof = info[0] if info else {}
    if prof.get("committees"):
        prof["committees"] = _loads(prof["committees"])
    return {"profile": prof, "skill": skill[0] if skill else None, "trades": trades}


@app.get("/api/ticker/{ticker}")
def ticker(ticker: str):
    tk = ticker.upper()
    trades = q("""
        SELECT member, chamber, tx_type, tx_date, filing_date, amount_mid
        FROM trades WHERE ticker = ? ORDER BY filing_date DESC
    """, [tk])
    for t in trades:
        t["tx_date"] = str(t["tx_date"])
        t["filing_date"] = str(t["filing_date"])
    prices = q("""
        SELECT date, close FROM prices WHERE ticker = ? AND date >= '2018-01-01'
        ORDER BY date
    """, [tk])
    for p in prices:
        p["date"] = str(p["date"])
    sector = q("SELECT sector, industry FROM ticker_sectors WHERE ticker = ?", [tk])
    return {"ticker": tk, "sector": sector[0] if sector else None,
            "trades": trades, "prices": prices}


@app.get("/api/research/backtest")
def backtest():
    curve = q("SELECT strategy, date, equity FROM backtest_equity ORDER BY date")
    series: dict[str, list] = {}
    for r in curve:
        series.setdefault(r["strategy"], []).append(
            {"date": str(r["date"]), "equity": r["equity"]})
    metrics = {}
    try:
        from ..config import DATA_DIR
        with open(DATA_DIR / "model_metrics.json") as f:
            metrics = json.load(f)
    except FileNotFoundError:
        pass
    return {"series": series, "model_metrics": metrics}


def _read_json(name: str, default):
    from ..config import DATA_DIR
    try:
        with open(DATA_DIR / name) as f:
            return json.load(f)
    except FileNotFoundError:
        return default


@app.get("/api/tactical/morning")
def tactical_morning():
    """Today's tactical book + the excluded-and-why list (MRVL-style names)."""
    return _read_json("morning_book.json",
                      {"as_of": None, "book": [], "excluded": [],
                       "note": "run daily_update --morning to generate"})


@app.get("/api/research/tactical")
def research_tactical():
    """Tactical daily backtest: equity curve, exit-reason mix, trade stats."""
    return _read_json("tactical.json", {"note": "run the tactical backtest"})


@app.get("/api/research/tactical-sweep")
def research_tactical_sweep():
    """Sensitivity table: what each layer of stops does to CAGR/Sharpe/DD."""
    return _read_json("tactical_sweep.json", {"configs": []})


@app.get("/api/research/entry-timing")
def research_entry_timing():
    """Alpha-decay-by-entry-delay study + conditioning on post-filing price action."""
    return _read_json("entry_timing.json", {"note": "run the entry-timing study"})


@app.get("/api/research/skill-persistence")
def research_skill_persistence():
    """Does politician skill persist year-to-year? (Spearman rank, next-open entry.)"""
    return _read_json("skill_persistence.json", {"note": "run skill_persistence"})


@app.get("/api/research/event-study-v2")
def research_event_study_v2():
    """Factor-adjusted, cluster-robust CARs by slice, with a locked 2024+ holdout."""
    return _read_json("event_study_v2.json", {"note": "run event_study_v2"})


@app.get("/api/research/backtest-v2")
def research_backtest_v2():
    """Leak-free rule-strategy backtest (next-open, cash, delistings) + holdout."""
    return _read_json("backtest_v2.json", {"note": "run backtest_v2"})


@app.get("/api/research/combo")
def research_combo():
    """Flagship experiment: does insider (Form 4) confirmation improve a congress buy?"""
    return _read_json("combo_experiment.json", {"note": "run combo_experiment"})


@app.get("/api/research/gov")
def research_gov():
    """G-layer: contract event study (standalone) + congress-buy-followed-by-contract."""
    return _read_json("gov_experiment.json", {"note": "run gov_experiment"})


@app.get("/api/research/large-audit")
def research_large_audit():
    """Deep audit of the LARGE strategy: gross/cash, factor alpha/beta, vol-scaling,
    bootstrap CIs, and per-trade-characteristic CAR slices."""
    return _read_json("large_audit.json", {"note": "run large_audit"})


@app.get("/api/research/reg")
def research_reg():
    """Does a Federal Register significant rule in the sector improve a LARGE buy?"""
    return _read_json("reg_experiment.json", {"note": "run reg_experiment"})


@app.get("/api/strategy/large-sleeve")
def strategy_large_sleeve():
    """Deployable vol-targeted LARGE sleeve: default vs aggressive configs."""
    return _read_json("large_sleeve.json", {"note": "run large_sleeve"})


@app.get("/api/research/large-validation")
def research_large_validation():
    """LARGE validation: leave-one-out, concentration, vs beta-matched SPY, deflated Sharpe."""
    return _read_json("large_validation.json", {"note": "run large_validation"})


@app.get("/api/research/intraday")
def research_intraday():
    """Hourly sector-relative mean-reversion PoC: gross vs net, cost sweep, holdout."""
    return _read_json("intraday_meanrev.json", {"note": "run intraday_meanrev"})


@app.get("/api/research/latent-flow")
def research_latent_flow():
    """F1 latent-flow-shock bar-only screen; research-only, not executable flow."""
    return _read_json("latent_flow_f1.json", {"note": "run scripts/latent_flow_sprint.py",
                                                "status": "NOT_RUN"})


@app.get("/api/research/v4-replay")
def research_v4_replay():
    """V4 leak-free event-replay PoC (Federal Register -> sector ETF, hourly)."""
    return _read_json("v4_event_reaction_poc.json", {"note": "run v4.event_reaction_poc"})


@app.get("/api/research/mgrm")
def research_mgrm():
    """MGRM: management guidance revision vs market reaction. Zero-vendor public
    SEC/IR data only; G0/G1/G2 nested elastic nets gated on the extraction audit."""
    return _read_json("mgrm_report.json",
                      {"note": "run scripts/mgrm_sprint.py discover/extract/link/features/run",
                       "status": "MGRM_NOT_RUN"})


@app.get("/api/research/event-atlas")
def research_event_atlas():
    """SEC Event Atlas taxonomy/unsigned Stage-1 diagnostic."""
    return _read_json("sec_event_atlas_unsigned.json",
                      {"status": "BLOCKED_NO_ATLAS_MANIFEST",
                       "note": "run scripts/sec_event_atlas.py ingest then unsigned"})


@app.get("/api/research/mgrm-audit")
def research_mgrm_audit():
    """MGRM data gate: document coverage, extraction/AI agreement, previous-guidance
    match rate, and power-derived sample requirement."""
    return _read_json("mgrm_audit.json", {"note": "run scripts/mgrm_sprint.py audit",
                                          "status": "BLOCKED"})


@app.get("/api/research/mgrm-goldset")
def research_mgrm_goldset():
    """MGRM extractor gold-set audit: detection precision/recall and field-level
    accuracy vs frozen labels; certification gates the historical pilot."""
    return _read_json("mgrm_goldset_audit.json",
                      {"note": "run scripts/mgrm_sprint.py goldset",
                       "status": "NO_GOLDSET"})


@app.get("/api/forward/book")
def forward_book():
    """The live paper book to trade: primary LARGE + the 4 observational shadows."""
    from ..forward import tracker
    strategies = ["LARGE", "LARGE_NEW", "LARGE_SPOUSE", "LARGE_15_30D", "LARGE_250K_1M"]
    books = {}
    for s in strategies:
        b = tracker.current_book(s)
        # attach latest price for share sizing (reuse portfolio price map)
        books[s] = b
    return {"books": books, "capital_strategy": "LARGE",
            "version": books["LARGE"].get("version")}


@app.get("/api/forward/evaluate")
def forward_evaluate():
    """Pre-registered evaluation status of the frozen forward record."""
    from ..forward import tracker
    return tracker.evaluate()


@app.get("/api/forward/decisions")
def forward_decisions(limit: int = 200):
    """The immutable recorded decisions (append-only history)."""
    rows = q("""
        SELECT decision_date, strategy, version, ticker, target_weight,
               source_member, source_filing_date, first_seen_at, decision_price
        FROM forward_decisions ORDER BY decision_date DESC, strategy, target_weight DESC
        LIMIT ?
    """, [limit])
    for r in rows:
        for k in ("decision_date", "source_filing_date", "first_seen_at"):
            r[k] = str(r[k])
    return rows


@app.get("/api/forward/opening-flow")
def forward_opening_flow(limit: int = 200):
    """Opening Flow canary decisions, paper orders, marks, and fixed screen."""
    from ..config import DATA_DIR
    try:
        decisions = q("""SELECT decision_id, decision_date, decision_ts, book,
                                policy_version, ticker, action, side, target_weight,
                                status, reason, features
                         FROM opening_flow_decisions
                         ORDER BY decision_ts DESC, book LIMIT ?""", [limit])
        orders = q("""SELECT order_id, decision_id, book, ticker, side, qty,
                            broker, status, submitted_at, filled_at, fill_price
                     FROM opening_flow_orders ORDER BY submitted_at DESC LIMIT ?""", [limit])
        marks = q("""SELECT decision_id, mark_ts, ticker, price, pnl, exit_reason
                    FROM opening_flow_marks ORDER BY mark_ts DESC LIMIT ?""", [limit])
    except duckdb.Error:
        decisions, orders, marks = [], [], []
    for row in decisions:
        row["reason"], row["features"] = _loads(row["reason"]), _loads(row["features"])
        for key in ("decision_date", "decision_ts", "signal_as_of", "entry_after", "exit_by"):
            if key in row and row[key] is not None:
                row[key] = str(row[key])
    for row in orders + marks:
        for key in ("submitted_at", "filled_at", "mark_ts"):
            if key in row and row[key] is not None:
                row[key] = str(row[key])
    screen = _read_json("opening_flow_screen.json", {"status": "NOT_RUN"})
    return {"books": ["CASH_CHAMPION", "OPENING_FLOW_P1", "OPENING_FLOW_P2", "OPENING_FLOW_P3"],
            "decisions": decisions, "orders": orders, "marks": marks, "screen": screen}


@app.get("/api/events/summary")
def events_summary():
    """Point-in-time event store: layer counts + resolved-entity coverage."""
    try:
        layers = q("SELECT layer, COUNT(*) AS n FROM events GROUP BY layer ORDER BY 1")
        ent = q("SELECT COUNT(*) AS n FROM sec_entities")[0]["n"]
        return {"layers": layers, "sec_entities": ent}
    except duckdb.Error as e:
        raise HTTPException(503, str(e))


@app.get("/api/research/event-study")
def event_study_endpoint():
    """On-demand event-study summary (cached file if present)."""
    from ..config import DATA_DIR
    try:
        with open(DATA_DIR / "event_study.json") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"note": "run scripts/daily_update.py to generate event-study output"}
