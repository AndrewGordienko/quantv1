# ACTOR_ENGINE.md — Actor State & Influence Engine

Not a "personality model" (that invites storytelling and overfitting). An engine
that measures **observable public behavior** and its market consequences:

```
Expected impact = actor_power × event_importance × surprise_vs_actor_baseline
                  × credibility/follow-through × company_exposure × market_receptivity
Tradeable edge  = expected impact − reaction already priced
```

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
- **B1** + event text/type
- **B2** + actor identity & power        ← does famous-person identity add edge?
- **B3** + actor behavioral state         ← does "emotion"/state add anything?
- **B4** + influence graph
- **B5** + audio/video emotion (last, noisiest)

If B2≤B1, identity adds nothing. If B3≤B2, the state/emotion layer is useless.
Build in this order; do NOT jump to graphs or multimodal first.

## Models (simplest first — NOT a GNN first)

1. **Hierarchical Bayesian actor-impact** (Stage 1): residual return ~ event_type
   + stance + actor_power + baseline_deviation + credibility + company_exposure +
   regime + actor×topic. Partial pooling lets frequent actors get individual
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

Shuffle actor identity WITHIN role+event-type; shuffle state WITHIN actor; shift
timestamps preserving time-of-day; scheduled meetings vs matched non-meeting dates;
text-only vs +audio/video; hold out entire actors, companies AND time periods;
cluster inference by catalyst/day AND actor.

## Where to start (~20 high-information actors, not everyone famous)

Central bankers (standardized press conferences, exact timestamps, clean asset
exposure) and founder/CEOs with frequent public comms + earnings calls are the
CLEANEST first targets. Add a few regulators with direct jurisdiction and
executive officials affecting tariffs/procurement/regulation. Political meetings
are a slower contextual layer (contents often disclosed later).

## First build (this is the concrete sequence)

1. Actor + time-valid relationship tables.                    [schema — STARTED]
2. Public actor-event ingestion with EXACT timestamps.        [data — the crux]
3. Actor baseline + credibility features (rolling, within-actor).
4. Hierarchical Bayesian actor-impact event study (Stage 1).
5. Actor-identity permutation test (B1 vs B2).
6. Behavioral-state / change-point model (B2 vs B3).
7. Only then graph propagation, then multimodal.

## The data crux (honest)

Polygon news is COMPANY-tagged, not ACTOR-tagged — attribution is the hard part.
Cleanest sources with exact timestamps + asset exposure:
- **Fed / central banks**: FOMC statements + press-conference times (public).
- **Earnings calls**: scheduled, timestamped, per-company (transcripts need a source).
- Actor mentions extractable from headlines for a curated ~20 actors as a first proxy.

Reuses everything already built: the catalyst store, the leak-free replay harness
(next-open, catalyst-clustered bootstrap, pessimistic barriers, controls), the
Polygon minute bars. This is a NEW hypothesis on the SAME rigorous substrate.
