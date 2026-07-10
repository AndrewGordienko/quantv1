# ACTOR_ENGINE.md — Actor State & Influence Engine

Not a "personality model" (that invites storytelling and overfitting). An engine
that measures **observable public behavior** and its market consequences:

The primary falsifiable question is:

> Conditional on what was said, the event type, the asset and market conditions,
> does knowing who generated the public communication and how unusual it was for
> them improve the predicted market-impact distribution?

Hand-assigned authority scores are registry metadata only. They are not predictive
features until replaced with measured components such as jurisdiction, ownership,
historical follow-through and rolling out-of-sample impact.

The falsifiable hypothesis (vs "news sentiment moves stocks"): the market
processes the SAME public event differently depending on WHO generated it, how
unusual it is FOR THEM, their current credibility, and how influence propagates —
and sometimes misprices it. Only lawful PUBLIC professional communications;
never private location/conversations or MNPI. (SEC Reg-FD recognizes designated
company social channels as valid public disclosure.)

## The four mispricings to hunt

1. **Actor-specific meaning** — the same sentence matters differently from a Fed
   chair vs a regulator vs a CEO vs an uninvolved commentator.
2. **Deviation from personal baseline** — Musk sounding aggressive is routine; a
   normally-restrained regulator turning forceful is informative. Normalize every
   feature WITHIN the actor.
3. **Credibility / follow-through** — some actors threaten and rarely act; others
   speak rarely and reliably act. Learn the difference.
4. **Reaction mismatch (most tradeable)** — high-impact event + weak reaction →
   continuation; low-impact rhetoric + extreme reaction → reversal.

## Algorithm ladder (each stage must beat the prior on UNTOUCHED data)

- **B0** price + market context only
- **B1** + event type, stance, magnitude, ticker, sector and regime
- **B2** + hierarchically-shrunk actor identity, actor×event type and actor×topic
- **B3** + actor behavioral state         ← does "emotion"/state add anything?
- **B4** + influence graph
- **B5** + audio/video emotion (last, noisiest)

If B2≤B1, identity adds nothing. If B3≤B2, the state/emotion layer is useless.
Build in this order; do NOT jump to graphs or multimodal first.

## Models (simplest first — NOT a GNN first)

1. **Hierarchical actor-impact** (Stage 1): residual return ~ event_type + stance
   + magnitude + ticker + sector + regime + baseline_deviation + measured
   credibility + measured company exposure + actor×event_type + actor×topic.
   Partial pooling lets frequent actors get individual
   effects while a 3-event politician can't look magically predictive.
2. **Hidden semi-Markov / Bayesian change-point** (state, not emotion labels):
   routine → escalating → negotiating → committed → reversing. The TRANSITION is
   the feature.
3. **Marked Hawkes / survival** — event chains (meeting→statement→proposal→agency
   action→outcome) and how fast informational relevance decays.
4. **Temporal heterogeneous graph** — only after 1–3 show something; predict which
   connected assets under-reacted.
5. **Reaction/execution** — expected impact − observed reaction − spread −
   slippage − hedge; trade only if the lower CB stays positive after costs.

Also: double-ML / synthetic control / matched event studies (causality vs
coincidence); mixture-of-experts (CEO posts vs central-bank speeches vs rulings);
conformal/Bayesian uncertainty (refuse uncertain trades); online Bayesian updating
(credibility drift).

## Features (measurable rolling, never labels like "narcissistic")

- **Stable actor profile:** formal authority, ownership/voting power, regulatory
  jurisdiction, audience reach, historical asset impact, topic specialization,
  follow-through rate, reversal rate, response latency, comm frequency, credibility
  decay.
- **Dynamic state:** current topics, stance per company/industry, certainty vs
  hedging, urgency, cooperation vs conflict, escalation, novelty vs own history,
  frequency anomaly, recent meetings, recent contradiction, distance from own
  language baseline.
- **Multimodal (B5 only):** transcript semantics, pace, pauses, pitch variation,
  intensity, interruptions, facial-action changes, word-vs-delivery gap. Normalize
  WITHIN actor (never compare Musk's face to Powell's). Emotion classifiers are
  noisy — use only after text+actor works.

## Influence graph (TIME-VALID edges — or today leaks into the past)

Nodes: people (CEOs, politicians, regulators, central bankers, activists, agency
heads), orgs (companies, agencies, committees, parties, suppliers, customers,
contractors), assets (stocks, ETFs, commodities, FX, options), events (meetings,
speeches, posts, filings, hearings, contracts, earnings, rulings, investigations).
Edges: leads / regulates / sits_on / oversees / contracts_with / buys_from /
mentioned / met_with / affects / exposes. EVERY edge carries `valid_from`,
`valid_to`, `source`, `first_seen_at`.

## Controls (the whole point)

Compare B1 and B2 on untouched predictive loss and net trading outcomes. Match
events within ticker, event type, stance, volatility, time of day and market
regime. Shuffle actor identity only WITHIN actor-event role and event type;
shuffle state within actor; shift timestamps preserving time of day; hold out
entire speakers and time periods. Use multiway covariance/bootstrap across
catalyst/day, ticker and actor.

## Where to start (~20 high-information actors, not everyone famous)

Central bankers (standardized press conferences, exact timestamps, clean asset
exposure) and founder/CEOs with frequent public comms + earnings calls are the
CLEANEST first targets. Add a few regulators with direct jurisdiction and
executive officials affecting tariffs/procurement/regulation. Political meetings
are a slower contextual layer (contents often disclosed later).

## First build (this is the concrete sequence)

1. Actor + time-valid alias/role/exposure tables.              [schema — DONE]
2. Public actor-event ingestion with EXACT timestamps.        [data — the crux]
3. Actor baseline + credibility features (rolling, within-actor).
4. Hierarchical Bayesian actor-impact event study (Stage 1).
5. Actor-identity permutation test (B1 vs B2).
6. Behavioral-state / change-point model (B2 vs B3).
7. Only then graph propagation, then multimodal.

## The data crux (honest)

Polygon news is COMPANY-tagged, not ACTOR-action-tagged. Its headline-mention
audit is `INVALID_PROXY_STUDY`, not a B2 test and not evidence against actor
effects. Every actor event needs a participation role: speaker/author, directly
quoted, direct public action, verified decision maker, meeting participant,
subject of story or merely mentioned. Only speaker/author, direct public action
and verified decision-maker events enter the primary hypothesis.

Cleanest sources with exact timestamps + asset exposure:
- **Fed speaker panel (B2)**: public speeches and remarks from Chairs, Governors,
  regional Reserve Bank presidents and other voting participants. Test semantic
  B1 against hierarchical speaker B2 on held-out time and held-out speakers.
- **Chair press conferences (B3)**: timestamped transcript/audio segments,
  separated into prepared remarks, questions and answers. Test hawkish/dovish
  change, certainty/hedging, deviation from personal baseline and within-event
  state transitions.
- **Earnings calls**: scheduled, timestamped, per-company (transcripts need a source).

The minimum Fed asset set is Treasury duration plus financials (for example IEF,
TLT and XLF); a final study should use rates/futures data appropriate to monetary
policy shocks, not SPY alone.

Reuses everything already built: the catalyst store, the leak-free replay harness
(next-open, catalyst-clustered bootstrap, pessimistic barriers, controls), the
Polygon minute bars. This is a NEW hypothesis on the SAME rigorous substrate.
