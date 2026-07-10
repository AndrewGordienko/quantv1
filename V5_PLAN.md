# V5 — expectations, repricing and forced flows

## Current evidence

There is no positive signal in the completed work. Politician copying, politician
skill persistence, insider confirmation, generic news continuation/fade and
intraday mean reversion failed. Powerful-person mentions are an invalid proxy,
and primary-source behavior remains untested. Behavior is optional context, not
a standalone strategy.

The first V5 bar-cost proxy screen is also negative: a deterministic Tier-2 SEC
sample (684 event windows; 477 train, 204 validation after the outcome embargo)
rejected price/reaction-only elastic net on validation. This is not a verdict on
earnings expectation errors: point-in-time EPS/revenue consensus and guidance
remain absent, so structured surprise has not yet been tested. The final time
test remains unopened.

## Economic mechanisms

V5 tests two mechanisms:

1. **Expectation errors:** new structured information differs from priced
   expectations and is processed incompletely.
2. **Forced flows:** index funds, systematic funds or dealers must transact
   independent of discretionary opinion.

The active slate is:

| Track | Horizon | Priority |
|---|---:|---:|
| Earnings expectation and repricing | 30 minutes–20 days | 1 |
| Index/ETF forced flows | 1–10 days | 2 |
| Analyst revision diffusion | 2–30 days | 3 |
| Actor-state surprise as incremental context | 5 minutes–5 days | 4 |
| Event-conditioned peer diffusion | 30 minutes–5 days | 5 |

## Earnings model layers

Each layer must beat the prior layer on purged, grouped untouched data:

1. Financial surprise only.
2. Financial surprise plus initial 5/30/120-minute abnormal reaction.
3. Add pre-event options-implied move and positioning.
4. Add within-executive transcript/Q&A behavioral changes.
5. Add supplier/customer/competitor reactions.
6. Add regime and conformal uncertainty/abstention.

The primary economic horizon is five trading days after a next-quote entry 30
minutes into the first liquid post-release session. Intraday two-hour and 1-day,
20-day outcomes are secondary. Event windows, companies and transcripts remain
in one split group; the last year and deterministic unseen companies are locked.

CatBoost/quantile models are gated behind elastic-net lift. Deep RL is excluded.
Behavior advances only if the full financial/reaction model plus behavior beats
the same model without behavior.

## Promotion

Positive untouched net alpha; net Sharpe above 1; deflated-Sharpe probability
above 0.95; doubled-cost survival; positive/stable years, sectors and size groups;
no company/event/executive concentration; and realistic next-quote portfolio
accounting. If earnings and forced flows both fail, stop slicing this public-data
stack rather than returning to generic sentiment.
