import {
  ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, Legend,
} from "recharts";
import { useApi, fmtPct, fmtNum } from "../api.js";
import { PageHead, Stat, Loading } from "../components/bits.jsx";

const tt = {
  background: "#fff", border: "1px solid #e4e7ec", borderRadius: 9, fontSize: 12,
  boxShadow: "0 4px 16px rgba(16,24,40,0.08)",
};

export default function Evidence() {
  const { data: persist } = useApi("/research/skill-persistence");
  const { data: es } = useApi("/research/event-study-v2");
  const { data: bt } = useApi("/research/backtest-v2");
  const { data: combo } = useApi("/research/combo");
  const { data: intra } = useApi("/research/intraday");
  const { data: audit } = useApi("/research/large-audit");
  const { data: reg } = useApi("/research/reg");
  const { data: valid } = useApi("/research/large-validation");
  const { data: sleeve } = useApi("/strategy/large-sleeve");

  return (
    <>
      <PageHead eyebrow="Leak-free research (v2/v3)" title="Evidence">
        The honest scoreboard after fixing the v1 leaks: next-open execution,
        Carhart factor-adjusted returns, cluster-robust confidence intervals, and a
        locked 2024–2026 holdout. Short version: the apparent edge was mostly growth-factor
        exposure, not political alpha — and combining public signals hasn’t changed that yet.
      </PageHead>

      {audit && audit.portfolio && <LargeAudit a={audit} />}
      {valid && valid.deflated_sharpe && <Validation v={valid} />}
      {sleeve && sleeve.configs && <Sleeve s={sleeve} />}
      {reg && reg.train && <RegInteraction r={reg} />}
      {persist && <Persistence p={persist} />}
      {es && <FactorStudy es={es} />}
      {combo && combo.report && <Combo c={combo} />}
      {bt && <BacktestV2 bt={bt} />}
      {intra && intra.full && <Intraday x={intra} />}
      {!persist && !es && !bt && <Loading />}
    </>
  );
}

/* Flagship: insider (Form 4) confirmation of a congress buy */
function Combo({ c }) {
  const rows = [
    ["all_train", "All buys (train)"],
    ["large_train", "Large buys (train)"],
    ["all_holdout", "All buys (2024+ holdout)"],
    ["large_holdout", "Large buys (holdout)"],
  ];
  return (
    <div className="card" style={{ marginBottom: 20 }}>
      <div className="card-h">
        <h3>Flagship: does insider buying confirm a congress buy?</h3>
        <small>Form 4 open-market purchase within ±{c.confirm_window_days}d · 63d factor-adj CAR</small>
      </div>
      <div className="grid grid-3" style={{ marginBottom: 16 }}>
        <Stat label="Insider events (F layer)" value={c.n_insider_events?.toLocaleString()} />
        <Stat label="Confirmed congress buys" value={`${c.n_confirmed} / ${c.n_total}`}
          sub="insider buy within ±30d" />
        <Stat label="Verdict" value="No lift" tone="neg" sub="sign flips train↔holdout" />
      </div>
      <table>
        <thead><tr>
          <th>Slice</th><th className="num">Confirmed CAR</th>
          <th className="num">Unconfirmed CAR</th><th className="num">Lift</th><th className="num">n conf.</th>
        </tr></thead>
        <tbody>
          {rows.map(([k, label]) => {
            const r = c.report[k]; if (!r) return null;
            const cf = r.confirmed, un = r.unconfirmed;
            if (!cf || cf.mean == null || !un || un.mean == null)
              return (<tr key={k}><td>{label}</td><td className="num faint" colSpan={4}>insufficient n</td></tr>);
            const lift = cf.mean - un.mean;
            return (
              <tr key={k}>
                <td>{label}</td>
                <td className={"num mono " + (cf.mean >= 0 ? "pos" : "neg")}>{fmtPct(cf.mean, 2)}</td>
                <td className={"num mono " + (un.mean >= 0 ? "dim" : "neg")}>{fmtPct(un.mean, 2)}</td>
                <td className={"num mono " + (lift >= 0 ? "pos" : "neg")}>{fmtPct(lift, 2)}</td>
                <td className="num mono faint">{cf.n}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="faint" style={{ fontSize: 11.5, marginTop: 10 }}>
        Combining the P (politician) and F (insider) layers of the event store. The lift is
        negative in training and positive in holdout — no stable signal. The multi-source engine
        is built; this particular combination isn’t the edge.
      </p>
    </div>
  );
}

/* 1. Skill persistence */
function Persistence({ p }) {
  const rho = p.pooled_spearman;
  const persists = rho != null && Math.abs(rho) > 0.2 && p.pooled_p < 0.05;
  return (
    <div className="card" style={{ marginBottom: 20 }}>
      <div className="card-h"><h3>Does politician skill persist?</h3>
        <small>rank members in year t, check year t+1 · next-open entry</small></div>
      <div className="grid grid-3" style={{ marginBottom: 16 }}>
        <Stat label="Pooled rank correlation" value={fmtNum(rho, 3)}
          tone={persists ? "pos" : "neg"} sub={`p = ${fmtNum(p.pooled_p, 2)}, n = ${p.pooled_n}`} />
        <Stat label="Top-quartile → next year"
          value={fmtPct(p.transition?.top_quartile_next_year_mean_ar, 2)}
          sub={`n = ${p.transition?.n_top}`} />
        <Stat label="Bottom-quartile → next year"
          value={fmtPct(p.transition?.bottom_quartile_next_year_mean_ar, 2)}
          sub={`n = ${p.transition?.n_bottom}`} />
      </div>
      <div className="pill" style={{ background: "var(--bg)" }}>
        {persists
          ? "Skill persists — 'best politicians' is a tradeable trait."
          : "Verdict: no persistence (ρ ≈ 0). “Follow the best politicians” is not tradeable — rank on trade features instead."}
      </div>
      <table style={{ marginTop: 16 }}>
        <thead><tr><th>Year → next</th><th className="num">Spearman</th>
          <th className="num">p</th><th className="num">members</th></tr></thead>
        <tbody>
          {(p.year_pairs || []).map((yp) => (
            <tr key={yp.year}>
              <td className="dim">{yp.year} → {yp.next}</td>
              <td className={"num mono " + (yp.spearman >= 0 ? "pos" : "neg")}>{fmtNum(yp.spearman, 3)}</td>
              <td className="num mono faint">{fmtNum(yp.p, 2)}</td>
              <td className="num mono faint">{yp.n_members}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* 2. Factor-adjusted event study */
const SLICES = [
  ["overall", "All purchases"],
  ["large_trades", "Large (≥ $50k)"],
  ["fast_filings", "Fast filed (≤ 14d)"],
  ["repeat_purchases", "Repeat conviction"],
  ["large_repeat_momentum", "Large + repeat + momentum"],
];

function FactorStudy({ es }) {
  const horizons = es.horizons || [5, 21, 63];
  const h = 63;
  const train = es.report?.[`ff_${h}`]?.train || {};
  const hold = es.report?.[`ff_${h}`]?.holdout_2024plus || {};
  const sig = (s) => s && s.ci_low != null && (s.ci_low > 0 || s.ci_high < 0);

  return (
    <div className="card" style={{ marginBottom: 20 }}>
      <div className="card-h"><h3>Factor-adjusted alpha ({h}-day CAR)</h3>
        <small>Carhart 4-factor · 95% CI clustered by member · * = excludes 0</small></div>
      <table>
        <thead><tr>
          <th>Slice</th><th className="num">Mean CAR</th>
          <th>95% CI (member-clustered)</th><th className="num">n</th>
          <th className="num">Holdout ’24+</th>
        </tr></thead>
        <tbody>
          {SLICES.map(([key, label]) => {
            const s = (train[key]?.by_member) || train[key];
            const hs = (hold[key]?.by_member) || hold[key];
            if (!s || s.mean == null) return null;
            return (
              <tr key={key}>
                <td>{label}</td>
                <td className={"num mono " + (s.mean >= 0 ? "pos" : "neg")}>
                  {fmtPct(s.mean, 2)}{sig(s) ? " *" : ""}
                </td>
                <td className="mono faint">
                  [{fmtPct(s.ci_low, 2)}, {fmtPct(s.ci_high, 2)}]
                </td>
                <td className="num mono faint">{s.n}</td>
                <td className={"num mono " + (hs?.mean >= 0 ? "dim" : "neg")}>
                  {hs && hs.mean != null ? fmtPct(hs.mean, 2) : "—"}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="faint" style={{ fontSize: 11.5, marginTop: 10 }}>
        v1 reported the $250k–1M slice at +6.3%. After next-open entry, factor adjustment and
        clustering, the large-trade slice is ~+1.3% and only marginally significant — and it does
        not hold up in the 2024+ holdout.
      </p>
    </div>
  );
}

/* 3. Leak-free backtest */
const STRATS = [
  ["large", "#0d9488"], ["repeat", "#4f6bed"], ["fast", "#d97706"],
  ["naive", "#9aa0ad"], ["combo", "#7c3aed"], ["spy", "#111827"], ["qqq", "#16a34a"],
];

function BacktestV2({ bt }) {
  const r = bt.results || {};
  const order = ["naive", "large", "fast", "repeat", "combo", "spy", "qqq"];
  return (
    <div className="card">
      <div className="card-h"><h3>Leak-free backtest — strategies vs SPY & QQQ</h3>
        <small>next-open · cash allowed · delistings kept · locked 2024+ holdout</small></div>
      <div style={{ overflowX: "auto" }}>
        <table>
          <thead><tr>
            <th>Strategy</th>
            <th className="num">Full CAGR</th><th className="num">Full Sharpe</th>
            <th className="num">Holdout CAGR</th><th className="num">Holdout Sharpe</th>
            <th className="num">Holdout MaxDD</th>
          </tr></thead>
          <tbody>
            {order.map((k) => {
              const m = r[k]; if (!m) return null;
              const bench = k === "spy" || k === "qqq";
              return (
                <tr key={k} style={bench ? { fontWeight: 600 } : null}>
                  <td className="tick" style={{ color: bench ? "var(--text)" : "var(--accent-2)" }}>{k.toUpperCase()}</td>
                  <td className="num mono">{fmtPct(m.full?.cagr, 1)}</td>
                  <td className="num mono">{fmtNum(m.full?.sharpe, 2)}</td>
                  <td className="num mono">{fmtPct(m.holdout?.cagr, 1)}</td>
                  <td className="num mono">{fmtNum(m.holdout?.sharpe, 2)}</td>
                  <td className="num mono neg">{fmtPct(m.holdout?.max_dd, 1)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {bt.curve && (
        <ResponsiveContainer width="100%" height={300} style={{ marginTop: 16 }}>
          <LineChart data={bt.curve} margin={{ top: 8, right: 12, bottom: 0, left: 0 }}>
            <CartesianGrid stroke="#eef0f3" vertical={false} />
            <XAxis dataKey="date" tick={{ fill: "#8a919e", fontSize: 11 }} minTickGap={70} stroke="#e4e7ec" />
            <YAxis tick={{ fill: "#8a919e", fontSize: 11 }} stroke="#e4e7ec" width={40}
              tickFormatter={(v) => v.toFixed(1) + "×"} />
            <Tooltip contentStyle={tt} formatter={(v) => fmtNum(v, 2) + "×"} />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            {STRATS.map(([k, c]) => (
              <Line key={k} type="monotone" dataKey={k} name={k.toUpperCase()} stroke={c}
                strokeWidth={k === "spy" || k === "qqq" ? 2 : 1.3} dot={false}
                strokeDasharray={k === "qqq" ? "5 4" : null} />
            ))}
          </LineChart>
        </ResponsiveContainer>
      )}
      <p className="faint" style={{ fontSize: 11.5, marginTop: 10 }}>
        Rule strategies beat SPY on the full period but QQQ (growth) beats them all — the edge is
        mostly factor exposure. Out of sample no strategy beats the risk-matched growth benchmark.
      </p>
    </div>
  );
}

/* Intraday fast-trigger proof-of-concept */
function Intraday({ x }) {
  const sweep = x.cost_sweep || {};
  return (
    <div className="card" style={{ marginBottom: 20 }}>
      <div className="card-h">
        <h3>Fast-trigger PoC — hourly sector-relative mean reversion</h3>
        <small>next-bar exec · {x.n_names} names · 2025+ holdout · the fast layer, prototyped</small>
      </div>
      <div className="grid grid-3" style={{ marginBottom: 16 }}>
        <Stat label="Net Sharpe (full)" value={fmtNum(x.full?.sharpe, 2)}
          tone={x.full?.sharpe > 0 ? "pos" : "neg"} sub={`at ${x.params?.cost_bps_side}bps/side`} />
        <Stat label="Turnover / bar" value={fmtNum(x.avg_turnover_per_bar, 2)}
          sub="53% of book, every hour" />
        <Stat label="Verdict" value="Costs kill it" tone="neg" sub="~0 gross edge anyway" />
      </div>
      <table>
        <thead><tr><th>Cost / side</th>{Object.keys(sweep).map((c) =>
          <th key={c} className="num">{c}</th>)}</tr></thead>
        <tbody>
          <tr>
            <td className="dim">Sharpe</td>
            {Object.values(sweep).map((v, i) =>
              <td key={i} className={"num mono " + (v.sharpe > 0 ? "pos" : "neg")}>{fmtNum(v.sharpe, 2)}</td>)}
          </tr>
        </tbody>
      </table>
      <p className="faint" style={{ fontSize: 11.5, marginTop: 10 }}>
        Gross Sharpe is ~0 and every basis point of cost makes it worse — hourly reversal in liquid
        large-caps has no retail edge. The harness (next-bar exec, spreads, cost sweep, holdout) is
        the reusable scaffolding; a real fast layer needs minute data and an event-driven signal,
        not generic reversal.
      </p>
    </div>
  );
}

/* LARGE strategy deep audit — the one survivor */
function LargeAudit({ a }) {
  const p = a.portfolio, r = p.factor_regression, bf = p.bootstrap_full, bh = p.bootstrap_holdout;
  const slices = a.trade_slices || {};
  const sig = (v) => v && v.ci_low != null && (v.ci_low > 0 || v.ci_high < 0);
  const sliceRows = [];
  for (const [grp, label] of [["new_vs_addon", "New vs add-on"], ["by_owner", "By owner"],
      ["by_filing_lag", "By filing lag"], ["by_size", "By trade size"]]) {
    const g = slices[grp] || {};
    for (const [k, v] of Object.entries(g)) {
      if (v && v.mean != null) sliceRows.push({ grp: label, k, v });
    }
  }
  return (
    <div className="card" style={{ marginBottom: 20, borderColor: "rgba(13,148,136,0.4)" }}>
      <div className="card-h">
        <h3>LARGE strategy audit — the one survivor</h3>
        <small>the low-risk sleeve, stress-tested</small>
      </div>
      <div className="grid grid-4" style={{ marginBottom: 16 }}>
        <Stat label="Factor alpha (Carhart)" value={fmtPct(r.alpha_annual, 1) + "/yr"}
          tone={r.alpha_annual > 0 ? "pos" : "neg"} sub={`t = ${fmtNum(r.alpha_t, 2)} · R² ${fmtNum(r.r2, 2)}`} />
        <Stat label="Market beta" value={fmtNum(r.betas?.mkt_rf?.beta, 2)}
          sub="low → not tech beta" />
        <Stat label="Avg gross / cash" value={`${fmtPct(p.avg_gross, 0)} / ${fmtPct(p.avg_cash, 0)}`}
          sub={`${fmtNum(p.avg_n_held, 1)} names held`} />
        <Stat label="Return @ QQQ vol" value={fmtPct(p.return_at_qqq_vol, 1)}
          sub={`Sharpe ${fmtNum(p.sharpe, 2)}`} />
      </div>
      <div className="grid grid-2" style={{ marginBottom: 16 }}>
        <div className="stat">
          <div className="label">Bootstrap annual return — full</div>
          <div className="value" style={{ fontSize: 20 }}>{fmtPct(bf.mean_annual, 1)}</div>
          <div className="sub">95% CI [{fmtPct(bf.ci_low, 1)}, {fmtPct(bf.ci_high, 1)}]</div>
        </div>
        {bh && <div className="stat">
          <div className="label">Bootstrap annual return — 2024+ holdout</div>
          <div className="value" style={{ fontSize: 20 }}>{fmtPct(bh.mean_annual, 1)}</div>
          <div className="sub">95% CI [{fmtPct(bh.ci_low, 1)}, {fmtPct(bh.ci_high, 1)}]</div>
        </div>}
      </div>
      <table>
        <thead><tr><th>Group</th><th>Slice</th><th className="num">Factor-adj 63d CAR</th>
          <th>95% CI</th><th className="num">n</th></tr></thead>
        <tbody>
          {sliceRows.map((row, i) => (
            <tr key={i}>
              <td className="faint">{row.grp}</td>
              <td>{row.k}</td>
              <td className={"num mono " + (row.v.mean >= 0 ? "pos" : "neg")}>
                {fmtPct(row.v.mean, 2)}{sig(row.v) ? " *" : ""}
              </td>
              <td className="mono faint">[{fmtPct(row.v.ci_low, 2)}, {fmtPct(row.v.ci_high, 2)}]</td>
              <td className="num mono faint">{row.v.n}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="faint" style={{ fontSize: 11.5, marginTop: 10 }}>
        LARGE has positive Carhart alpha (~+8%/yr, t≈1.9 — borderline) with LOW market beta (0.32),
        so its −6% drawdown and returns are NOT just tech exposure. ~62% invested, robustly positive
        bootstrap CIs. Signal concentrates in new positions, $250k–1M trades, and spouse accounts
        (in-sample slices — treat with multiple-comparison caution).
      </p>
    </div>
  );
}

/* FR-rule x LARGE interaction */
function RegInteraction({ r }) {
  const seg = (s) => r[s] || {};
  const row = (name, seg) => {
    const lo = seg.low_reg_activity, hi = seg.high_reg_activity;
    if (!lo || lo.mean == null || !hi || hi.mean == null) return null;
    return (
      <tr key={name}>
        <td>{name}</td>
        <td className={"num mono " + (hi.mean >= 0 ? "pos" : "neg")}>{fmtPct(hi.mean, 2)}</td>
        <td className={"num mono " + (lo.mean >= 0 ? "pos" : "neg")}>{fmtPct(lo.mean, 2)}</td>
        <td className={"num mono " + (hi.mean - lo.mean >= 0 ? "pos" : "neg")}>{fmtPct(hi.mean - lo.mean, 2)}</td>
      </tr>
    );
  };
  return (
    <div className="card" style={{ marginBottom: 20 }}>
      <div className="card-h">
        <h3>Do government events improve LARGE? (Federal Register)</h3>
        <small>{r.n_reg_rules} significant rules · high vs low sector regulatory activity</small>
      </div>
      <table>
        <thead><tr><th>Period</th><th className="num">High-activity CAR</th>
          <th className="num">Low-activity CAR</th><th className="num">Lift</th></tr></thead>
        <tbody>{row("Train", seg("train"))}{row("2024+ holdout", seg("holdout"))}</tbody>
      </table>
      <p className="faint" style={{ fontSize: 11.5, marginTop: 10 }}>
        Lift flips sign (train +1.4pp, holdout −2.6pp) — no stable interaction. Federal Register is
        only sector-mapped; per-company rule extraction (LLM) is the real next step. Per-company
        USAspending contracts are the stronger G-layer test (ingest pending).
      </p>
    </div>
  );
}

/* LARGE validation — is the edge real? */
function Validation({ v }) {
  const d = v.deflated_sharpe, b = v.vs_beta_matched_spy, c = v.concentration;
  return (
    <div className="card" style={{ marginBottom: 20 }}>
      <div className="card-h">
        <h3>Is LARGE actually proven? Validation</h3>
        <small>leave-one-out · concentration · vs beta-matched SPY · deflated Sharpe</small>
      </div>
      <div className="grid grid-4" style={{ marginBottom: 16 }}>
        <Stat label="Deflated Sharpe" value={fmtNum(d.deflated_sharpe_prob, 2)}
          tone={d.passes ? "pos" : "neg"}
          sub={d.passes ? "clears N=" + d.n_trials + " trials" : "FAILS multiple-testing"} />
        <Stat label="Sharpe vs null-max" value={`${fmtNum(d.sharpe_annual, 2)} / ${fmtNum(d.expected_max_null_sharpe_annual, 2)}`}
          sub={`best-of-${d.n_trials} under null`} />
        <Stat label="vs beta-matched SPY" value={fmtPct(b.active_ann_return, 1) + "/yr"}
          tone={b.ci_low > 0 ? "pos" : "warn"}
          sub={`CI [${fmtPct(b.ci_low, 1)}, ${fmtPct(b.ci_high, 1)}] · P(>0) ${fmtNum(b.prob_positive, 2)}`} />
        <Stat label="Top-5 members" value={fmtPct(c.by_member.top5_share_of_positive_car, 0)}
          tone="warn" sub="of positive CAR (concentrated)" />
      </div>
      <div className="grid grid-2">
        <div>
          <div className="eyebrow">Worst leave-one-member-out</div>
          <table>
            <tbody>
              {v.leave_one_out.leave_one_member_out_worst.slice(0, 4).map((x, i) => (
                <tr key={i}><td>{x.drop}</td>
                  <td className="num mono">{fmtPct(x.ann_return, 1)}</td>
                  <td className={"num mono " + (x.delta >= 0 ? "pos" : "neg")}>{fmtPct(x.delta, 1)}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
        <div>
          <div className="eyebrow">Worst leave-one-ticker-out</div>
          <table>
            <tbody>
              {v.leave_one_out.leave_one_ticker_out_worst.slice(0, 4).map((x, i) => (
                <tr key={i}><td className="tick">{x.drop}</td>
                  <td className="num mono">{fmtPct(x.ann_return, 1)}</td>
                  <td className={"num mono " + (x.delta >= 0 ? "pos" : "neg")}>{fmtPct(x.delta, 1)}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      <p className="faint" style={{ fontSize: 11.5, marginTop: 12 }}>
        Verdict: strongest candidate, NOT proven. Robust to any single name, but ~55% of positive CAR
        comes from 5 members; beats beta-matched SPY by ~4%/yr but the CI includes zero (P≈0.91); and
        Sharpe 0.88 fails the deflated-Sharpe bar after ~{d.n_trials} strategies were tried. Deploy in
        paper and let the frozen forward record (from {v.forward_freeze || "2026-07-10"}) be the judge.
      </p>
    </div>
  );
}

/* Deployable vol-targeted sleeve */
function Sleeve({ s }) {
  const rows = [["unlevered_base", "Unlevered (base)", s.unlevered_base],
    ["default", "Default paper (16% vol)", s.configs.default],
    ["aggressive", "Aggressive paper (22% vol)", s.configs.aggressive]];
  return (
    <div className="card" style={{ marginBottom: 20 }}>
      <div className="card-h">
        <h3>Deployable LARGE sleeve v1 (paper)</h3>
        <small>original rule unchanged · volatility-targeted · freeze {s.forward_freeze}</small>
      </div>
      <table>
        <thead><tr><th>Config</th><th className="num">CAGR</th><th className="num">Vol</th>
          <th className="num">Sharpe</th><th className="num">Max DD</th>
          <th className="num">Avg gross</th><th className="num">% days levered</th></tr></thead>
        <tbody>
          {rows.map(([k, label, c]) => c && (
            <tr key={k} style={k === "unlevered_base" ? { fontWeight: 600 } : null}>
              <td>{label}</td>
              <td className="num mono">{fmtPct(c.cagr, 1)}</td>
              <td className="num mono">{fmtPct(c.vol, 1)}</td>
              <td className="num mono">{fmtNum(c.sharpe, 2)}</td>
              <td className="num mono neg">{fmtPct(c.max_dd, 1)}</td>
              <td className="num mono dim">{fmtPct(c.avg_gross, 0)}</td>
              <td className="num mono faint">{c.pct_time_levered != null ? fmtPct(c.pct_time_levered, 0) : "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="faint" style={{ fontSize: 11.5, marginTop: 10 }}>
        Vol-targeting did NOT improve LARGE here (Sharpe drops from 0.88 → 0.79/0.72) — it cut exposure
        into high-vol periods that rebounded, and leverage worsened drawdowns. Note the full-period max
        DD is −36% (the −6% figure was holdout-only). Deployable v1 ≈ the unlevered book at ~62% gross.
      </p>
    </div>
  );
}
