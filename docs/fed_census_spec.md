# Fed Communication Census — Frozen Specification (task #3)

**Rules artifact:** `goldset/fed_census_rules.json`
**Rules version:** `fed-census-rules-v1`
**Rules SHA-256:** `88a01e666c5a4d2af4f2a71a4bdbd88e8cd4b4515d648816a0451f0f156117b7`

This spec is frozen **before** any enumeration, feature extraction, or outcome
inspection. It defines a *deterministic census* of every qualifying Federal
Reserve public communication in the window where we can measure market
reaction. Nothing about actor features or event outcomes may be run until the
accepted manifest **and** the rejection ledger are frozen and committed.

## Why this exists

The actor B0–B4 study asks whether *who spoke* and *how they deviated* predicts
market reaction beyond *what was said*. That question is only answerable on a
census that was fixed before anyone looked at returns — otherwise inclusion
choices leak the answer. Hand-picking "interesting" Powell events would
manufacture a result. So we enumerate the whole roster's whole output in the
window and log every rejection with a reason.

## Census window — derived, not chosen

**2024-07-15 → 2026-07-09** (UTC, 537 trading days).

Derived as the intersection of `SPY`, `QQQ`, `XLF` minute-bar coverage in
`bars_minute` — the assets whose reaction we can actually compute. The window
is a property of the data, not a modelling choice.

### Market-data reality (a real limitation, stated plainly)

`TLT` and `IEF` have **zero** minute bars. Treasury duration is the cleanest
Fed transmission channel, and `fed_primary._validate` still *requires* an
IEF/TLT + XLF exposure on every record — so those tickers are **recorded** as
affected assets, but **no outcome rows can be computed** for them. Every
measurable outcome is therefore **equity-ETF spillover** (SPY/QQQ/XLF), which
is noisier than the rate response. Any B2 result is a statement about equity
spillover, not the rates market.

## The timestamp rule (the part that decides tradability)

The usable time is **when the exact content the model consumes became public**,
never when the speaker started. Backdating a completed transcript to the live
start would hand the model the entire future appearance at t=0 — catastrophic
look-ahead. Every event records four times and a mode:

| field | meaning |
|---|---|
| `scheduled_time` | calendar-announced start |
| `live_start_time` | when the person actually began speaking |
| `content_first_public_time` | when the **exact consumed content** first became public — the decision-eligible timestamp |
| `transcript_publication_time` | when the completed transcript was published |

`content_mode` ∈ `{PREPARED_TEXT_RELEASE, TIMESTAMPED_CAPTIONS, LIVE_STREAM,
POST_EVENT_TRANSCRIPT}`. A `POST_EVENT_TRANSCRIPT`'s
`content_first_public_time == transcript_publication_time` and **must not** be
backdated to `live_start_time`.

## Tradability tiers — the real question

The total count of archived transcripts is **not** the tradable sample. What
matters is how many events were machine-readable *in real time*:

| tier | content_mode | what it enables |
|---|---|---|
| `immediate_text` | `PREPARED_TEXT_RELEASE` | same-day text-based trading |
| `streaming_live` | `TIMESTAMPED_CAPTIONS`, `LIVE_STREAM` | streaming/live trading only |
| `research_only` | `POST_EVENT_TRANSCRIPT` | backtest/research only, not real-time |

The `immediate_text` (+ where historically-timestamped, `streaming_live`)
subset determines whether this can become a same-day trading system. The census
report must give the count per tier.

## Institutional vs individual

FOMC institutional statements are **separate non-speaker events**
(`actor_id = fomc_committee`), never attributed to a person. The 2:00pm
statement and the 2:30pm Chair press conference are **distinct events** with
distinct content and distinct timestamps.

## Actor universe

Every FOMC participant — Board of Governors + the 12 Reserve Bank presidents —
serving at any point in the window, per the official Fed roster. Deterministic
from the roster; no hand-selection.

### Census v1 scope: BOARD_ONLY

The first census version enumerates **only `federalreserve.gov`** — the Board of
Governors (including the Chair) plus `fomc_committee` institutional events. The
12 Reserve Bank presidents and their regional sites are **deferred** to a later
version (reason code `OUT_OF_SCOPE_V1`), because establishing an exact
`content_first_public_time` per regional site is far harder and would swamp the
first pass with low-timestamp-quality candidates. This trades breadth for
timestamp quality and the cleanest `immediate_text` count.

> **Power implication:** Board-only may not reach ≥6 adequately represented
> speakers. If it doesn't, census v1 is a **descriptive pilot** and a later
> full-roster version is required before any validation claim.

> **Gap found:** the `actors` registry currently holds only 2 central bankers
> (Powell, Warsh). The in-scope Board roster must be registered during
> enumeration.

## Deliverables (all frozen before features/outcomes)

1. **Accepted manifest** — fields per `goldset/fed_census_rules.json`, with
   source URLs, `document_sha256`, and timestamp provenance.
2. **Rejection ledger** — every excluded candidate with an exact reason code.
3. **Coverage tables** — by speaker, event type, year, timing precision,
   content mode, tradability tier.
4. **Unique-event counts** — never event-ticker rows.
5. **Macro-overlap flags** — FOMC/CPI/PCE/payrolls/GDP/earnings sharing a session.
6. **Frozen census version/hash** — SHA-256 over manifest + rejection ledger.

## Power gate

- ≥ **75** exact-time unique events **and** ≥ **6** adequately represented speakers → proceed to the B0–B4 study.
- Otherwise → classify **DESCRIPTIVE PILOT**. Do **not** relax timestamp
  requirements to reach the gate.

## Downstream discipline (recorded here so it is not forgotten)

- Sample power is counted in **unique events**. N events across SPY/QQQ/XLF are still N shocks.
- All statistics **cluster by event** (correlated multi-asset reactions).
- The B0–B4 comparison uses **leave-one-speaker-out** validation so the model
  cannot win merely by learning "Powell events are volatile."
- Until validated, the system emits **research/shadow decisions only** — there
  is no actor-derived buy/sell signal.
