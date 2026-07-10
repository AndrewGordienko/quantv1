# TRADING_PLAN.md — From research dashboard to live (paper) trading

Goal: connect the congressional-signal engine to a broker API so it trades
during the day — entering names the model likes, and **exiting time-sensitively
instead of holding losers** (the MRVL problem). Buy side AND sell side, fully
automated, with safety rails.

---

## 0. Honest framing before we wire money

Our own research says: the *average* disclosure has no edge (event study ~0,
model AUC 0.55, backtest trailed SPY). The edge, if any, is in **slices**:
large trades ($250k+ bucket: +6.3% @ 63d, t=2.65), cluster buys, top-skill
members. Intraday execution and stops do NOT create alpha — they control risk
and stop us bleeding on stale/broken signals. Expectation: a tighter, safer
version of the strategy; not a money printer. **Paper trade only** until the
live paper track record beats its own backtest expectations for 3+ months.

---

## 1. Broker: Alpaca (paper)

- **Why**: free paper-trading API, commission-free, first-class Python SDK
  (`alpaca-py`), supports **bracket orders** (entry + take-profit + stop-loss
  as one atomic order — the exit lives server-side even if our process dies).
- Setup: create account at alpaca.markets → generate PAPER keys →
  `.env` with `ALPACA_KEY`, `ALPACA_SECRET`, `ALPACA_PAPER=true`
  (never commit; add `.env` to .gitignore).
- Data: Alpaca's free IEX feed is enough for stop monitoring at our frequency;
  daily bars still come from yfinance.

---

## 2. The quant algo stack

### 2.1 ENTRY — "should this name be bought *today*?" (all must pass)

| # | Algorithm | Rule (initial params) | Why |
|---|---|---|---|
| E1 | **Disclosure signal** | LightGBM score ≥ 0.55 (existing model) | the base alpha |
| E2 | **Signal-slice gate** | backing trade ≥ $50k midpoint, OR cluster ≥ 2 members in 30d, OR member in top-decile shrunk skill | event study shows edge lives ONLY here |
| E3 | **Freshness decay** | days since latest backing filing ≤ 21 trading days; weight × (1 − d/63) | drift window is 63d; entering late = less left |
| E4 | **Trend gate** (time-series momentum) | last close > 50d MA AND 20d return > −5% | never catch falling knives (kills MRVL-style entries) |
| E5 | **Liquidity floor** | median $ volume (63d) ≥ $5M; price ≥ $5 | slippage + honest fills |
| E6 | **Regime filter** | SPY > 200d MA → full gross; below → 50% gross, no new entries on VIX-spike days | classic trend overlay; cuts 2022-style bleeds |

### 2.2 EXIT — time-sensitive sells (ANY one triggers)

| # | Algorithm | Rule (initial params) | Type |
|---|---|---|---|
| X1 | **Time stop** | exit at 63 trading days after entry, no exceptions | signal expiry |
| X2 | **ATR stop-loss** | exit if price < entry − 2.5 × ATR(14) (≈ vol-adjusted −8–12%) | disaster stop — *sent as bracket order at entry* |
| X3 | **Chandelier trailing stop** | exit if price < (highest close since entry) − 3 × ATR(14) | lock in winners |
| X4 | **Trend-break** | 2 consecutive closes below 50d MA → exit next open | the "it's already going down" rule |
| X5 | **Signal reversal** | a backing member files a SALE of the name → exit; OR re-scored model score < 0.45 → exit | the thesis died |
| X6 | **Portfolio kill switch** | day P&L < −3% → flatten all, halt for the day; drawdown < −15% from peak → cut gross to 50% until recovered | survival |

### 2.3 SIZING — vol-targeted risk parity (lite)

- Per-name weight ∝ score × freshness ÷ ATR% (equal *risk*, not equal dollars)
- Caps: 8% per name, 30% per sector, ≤ 20 names
- Portfolio vol target ~12–15% annualized; scale gross exposure to hit it
- Whole shares via `floor(weight × equity / price)` (calculator logic we built)

### 2.4 EXECUTION

- **Never market orders at the open.** Enter 9:35–10:00 ET with marketable
  limit orders (limit = last price + 10–20bps); unfilled after 15 min → cancel.
- Entries as **bracket orders**: limit entry + attached X2 stop + optional
  take-profit at +3.5 × ATR.
- Client-order-IDs = deterministic hash (date+ticker+side) → idempotent retries.
- On every start: reconcile local book vs broker positions; broker is truth.

---

## 3. Daily schedule (all times ET, weekdays)

| Time | Job | What it does |
|---|---|---|
| 8:30 | `morning_run` | ingest new disclosures + incremental prices → re-score → build target book through gates E1–E6 → compute diff vs broker positions → write order plan |
| 9:35 | `execute` | submit exit orders first (X4/X5 flags from last night, anything time-stopped), then entry bracket orders |
| every 15 min 9:35–15:55 | `risk_monitor` | mark positions vs X3 trailing/X4 trend levels using live quotes; fire exits; check X6 kill switch |
| 16:10 | `eod_report` | pull fills + P&L from Alpaca, persist to DuckDB (`live_trades`, `live_equity`), refresh dashboard Live page |

Scheduling: two options — `launchd` plists on the Mac (survives reboots) or a
single long-running `scheduler.py` using APScheduler. Start with cron/launchd:
simplest, observable, each job idempotent.

---

## 4. Build phases (the actual instructions)

**Phase A — prove the rules in backtest first (no broker code).**
1. Upgrade `portfolio/backtest.py` to a **daily** event loop (currently 21-day).
2. Implement E1–E6 entries, X1–X6 exits, vol sizing as a `tactical` strategy.
3. Run 2016→2026: compare tactical vs current model vs naive vs SPY on CAGR,
   Sharpe, max-DD, and **avg loss per losing trade** (the stat the stops fix).
4. Keep only rules that survive: if a gate doesn't improve Sharpe/DD, drop it —
   fewer parameters, less overfitting. This step decides the final param set.

**Phase B — broker layer.**
5. `uv add alpaca-py python-dotenv`.
6. `src/quantv1/live/broker.py` — thin wrapper: get_positions, get_equity,
   submit_bracket, submit_limit, cancel_all, get_fills. Paper endpoint only,
   asserts `ALPACA_PAPER=true` unless an env override is set.
7. `src/quantv1/live/orders.py` — target-book → order-plan diff engine
   (buys, sells, resizes; respects whole shares, min $200 per order).

**Phase C — the three jobs.**
8. `src/quantv1/live/morning.py` (decision), `execute.py` (orders),
   `monitor.py` (intraday exits + kill switch), `eod.py` (reporting).
9. New DuckDB tables: `live_orders`, `live_fills`, `live_equity`.
10. launchd plists in `ops/` + a `make live-install` to load them.

**Phase D — dashboard Live page.**
11. API: `/api/live/positions`, `/api/live/orders`, `/api/live/equity`.
12. Frontend: broker positions vs target book side-by-side, today's fills,
    live P&L curve, kill-switch status banner, every exit tagged with WHICH
    rule fired (X1–X6) — so you can audit "why did it sell."

**Phase E — go-live checklist (paper).**
- [ ] Phase A backtest shows tactical ≥ model on Sharpe AND smaller max-DD
- [ ] Dry-run mode prints order plan without submitting for 1 week
- [ ] Kill switch tested by simulating a −3% day
- [ ] Reconciliation handles: partial fills, rejected orders, halted stocks
- [ ] 3 months green paper trading before even discussing real dollars

---

## 5. Known constraints

- **PDT rule**: real accounts under $25k are limited to 3 day-trades per 5
  days. We swing-trade (hold days–weeks), so exits normally don't day-trade,
  but same-day entry+stop-out counts — the monitor must track this on small
  real accounts. Irrelevant on paper.
- Disclosure data updates daily, not intraday — intraday work is *risk
  management only*; there is no intraday alpha source here.
- yfinance prices lag ~1 day; live marks come from Alpaca quotes.
