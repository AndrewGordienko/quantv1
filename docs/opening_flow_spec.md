# Opening Flow — prospective paper canary

This is an executable forward experiment, not a profitability claim. It is
deliberately separate from rejected F1 and from the existing daily LARGE
tracker.

`CASH_CHAMPION` is the current champion. P3 is a challenger with no promotion
authority; Monday's paper activity exists only to test the live loop.

## Frozen policy

At 10:00 New York time, the runner builds four books:

- `CASH_CHAMPION`: no trade control;
- `OPENING_FLOW_P1`: material overnight gap only;
- `OPENING_FLOW_P2`: gap plus 30-minute market/sector residual agreement;
- `OPENING_FLOW_P3`: P2 plus same-sector peer confirmation and first-30-minute
  relative volume.

The constants are frozen in `quantv1.forward.opening_flow.POLICY`: one position,
5% simulated NAV, 16 bps round-trip cost, 1× volatility gap, 1.25× relative
volume, and a 0.5× move proxy that must exceed 2× cost. The candidate is entered
at the next minute open and hard-exited at 15:50 ET. No averaging down,
leverage, or overnight holding is allowed. Only P3 is eligible to submit the
single canary order; P1/P2 remain shadow books.

The historical screen is fixed and compares P0–P3 exactly once:

```bash
uv run python scripts/opening_flow_screen.py
```

The current local screen produced 436 P1, 168 P2, and 27 P3 trades. Mean net
returns were −11.0, −8.1, and −16.0 bps respectively after the frozen cost.
This is a screen, not validation; P3 is shadow-only.

## Live loop

Create a paper-only Alpaca `.env` with `ALPACA_KEY` and `ALPACA_SECRET`. The
runner is dry-run by default and hard-codes the Alpaca paper trading endpoint:

```bash
# 10:00 ET decision pass; records all four books
uv run python scripts/opening_flow_live.py

# Keep one process alive from pre-open through the 15:50 ET hard exit (dry-run)
uv run python scripts/opening_flow_live.py --loop

# Explicitly submit only the small P3 canary to Alpaca paper; no promotion authority
uv run python scripts/opening_flow_live.py --loop --send-orders --notional 10000

# Run each minute after submission / at the close
uv run python scripts/opening_flow_live.py --reconcile
uv run python scripts/opening_flow_live.py --mark
uv run python scripts/opening_flow_live.py --close --send-orders
```

No credentials produce explicit `REJECTED_DATA`/`rejected_no_credentials` rows,
not an implied order. Decisions, broker submissions, fill status and minute
marks are stored in `opening_flow_decisions`, `opening_flow_orders`, and
`opening_flow_marks`. They are exposed at `/api/forward/opening-flow` and on the
localhost **Opening Flow** page for replay after the close.

The historical minute bars do not contain NBBO spreads, so their liquidity gate
is only a volume proxy. Live decisions require a quote spread no wider than 15
bps. Public-event contradiction filtering is not yet claimed; the limitation is
recorded in the screen artifact.

## Evening discipline

The runner only records decisions and observed outcomes. It never changes the
constants after a loss, removes a ticker because of its return, or promotes a
book after one day. The screen artifact and all decision/order/mark rows carry
the immutable policy version. Threshold changes require a new weekly version
and a separate champion/challenger comparison.
