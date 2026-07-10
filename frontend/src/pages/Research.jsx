import { useState, useMemo } from "react";
import {
  ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, Legend, Area, AreaChart,
} from "recharts";
import { useApi, fmtPct, fmtNum } from "../api.js";
import { PageHead, Stat, Loading, Empty } from "../components/bits.jsx";

const usd = (x, d = 0) =>
  x == null ? "—" : `$${x.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d })}`;

export default function Research() {
  const { data, loading } = useApi("/research/backtest");
  const { data: es } = useApi("/research/event-study");
  if (loading) return <Loading />;

  const series = data?.series || {};
  const merged = mergeSeries(series);
  const metrics = data?.model_metrics?.metrics || {};
  const gbm = metrics.lightgbm;

  return (
    <>
      <PageHead eyebrow="Evidence" title="Research & Performance">
        Everything here is point-in-time: entries at the filing date, the model refit at each
        rebalance on only-already-realized outcomes, benchmarked against SPY and a naive
        copy-everything book.
      </PageHead>

      {merged.length > 0 && <DCASimulator series={series} />}

      {gbm && (
        <div className="grid grid-4" style={{ marginBottom: 20 }}>
          <Stat label="Model AUC (OOS)" value={fmtNum(gbm.mean_auc, 3)}
            sub="walk-forward mean" tone={gbm.mean_auc > 0.55 ? "pos" : null} />
          <Stat label="Rank IC" value={fmtNum(gbm.mean_ic, 3)} sub="score vs realized excess" />
          <Stat label="Base rate" value={fmtPct(data?.model_metrics?.base_rate, 0)}
            sub="beat SPY unconditionally" />
          <Stat label="Train rows" value={data?.model_metrics?.n_train?.toLocaleString()} />
        </div>
      )}

      {merged.length > 0 ? (
        <div className="card" style={{ marginBottom: 20 }}>
          <div className="card-h"><h3>Backtest equity</h3>
            <small>growth of $1 (lump sum), monthly rebalance, 10bps/side</small></div>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={merged} margin={{ top: 8, right: 12, bottom: 0, left: 0 }}>
              <CartesianGrid stroke="#eef0f3" vertical={false} />
              <XAxis dataKey="date" tick={{ fill: "#8a919e", fontSize: 11 }} minTickGap={70} stroke="#e4e7ec" />
              <YAxis tick={{ fill: "#8a919e", fontSize: 11 }} stroke="#e4e7ec" width={40} />
              <Tooltip contentStyle={tt} labelStyle={{ color: "#58606d" }} formatter={(v) => fmtNum(v, 2) + "×"} />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              <Line type="monotone" dataKey="model" name="Model" stroke="#0d9488" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="naive" name="Naive copy" stroke="#4f6bed" strokeWidth={1.4} dot={false} />
              <Line type="monotone" dataKey="spy" name="SPY" stroke="#8a919e" strokeWidth={1.4} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      ) : (
        <Empty>No backtest yet — run the daily update with backtesting enabled.</Empty>
      )}

      {es?.overall && <EventStudy es={es} />}
    </>
  );
}

/* ---- Dollar-cost-averaging: invest $X/week following the algo ---- */
function DCASimulator({ series }) {
  const [weekly, setWeekly] = useState(() => {
    const s = Number(localStorage.getItem("quantv1_weekly"));
    return s > 0 ? s : 100;
  });
  const setW = (v) => { setWeekly(v); localStorage.setItem("quantv1_weekly", String(v)); };

  // Rebalances are ~21 calendar days (≈3 weeks); each contribution aggregates
  // the weekly amount over that interval so the input reads as "per week".
  const perRebalance = weekly * 3;

  const sim = useMemo(() => dca(series, perRebalance), [series, perRebalance]);
  if (!sim) return null;

  const { curve, final } = sim;
  const contributed = final.contributed;
  const modelVal = final.model;
  const spyVal = final.spy;
  const profit = modelVal - contributed;
  const years = curve.length ? (new Date(curve[curve.length - 1].date) - new Date(curve[0].date)) / 3.156e10 : 0;

  return (
    <div className="card" style={{ marginBottom: 20 }}>
      <div className="card-h">
        <h3>If you invested this every week, following the algo</h3>
        <small>dollar-cost averaging into the model portfolio since {curve[0]?.date}</small>
      </div>

      <div className="calc" style={{ marginBottom: 20, boxShadow: "none", background: "var(--bg)" }}>
        <div className="calc-field">
          <label>Amount per week</label>
          <div className="calc-input">
            <span>$</span>
            <input type="number" min="0" step="10" value={weekly}
              onChange={(e) => setW(Math.max(0, Number(e.target.value)))} />
          </div>
        </div>
        <div className="calc-presets">
          {[25, 50, 100, 250].map((v) => (
            <button key={v} className={"preset" + (weekly === v ? " active" : "")}
              onClick={() => setW(v)}>${v}</button>
          ))}
        </div>
        <div className="calc-summary">
          <div className="item">
            <div className="k">Total invested</div>
            <div className="v" style={{ color: "var(--text-dim)" }}>{usd(contributed)}</div>
          </div>
          <div className="item">
            <div className="k">Worth now (algo)</div>
            <div className="v" style={{ color: "var(--accent)" }}>{usd(modelVal)}</div>
          </div>
          <div className="item">
            <div className="k">Profit</div>
            <div className="v" style={{ color: profit >= 0 ? "var(--pos)" : "var(--neg)" }}>
              {profit >= 0 ? "+" : ""}{usd(profit)}
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-3" style={{ marginBottom: 18 }}>
        <Stat label="Algo portfolio" value={usd(modelVal)}
          sub={`${fmtNum(modelVal / contributed, 2)}× money in`} tone="accent" />
        <Stat label="Same $ into SPY" value={usd(spyVal)}
          sub={`${fmtNum(spyVal / contributed, 2)}× money in`} />
        <Stat label="Invested over" value={`${fmtNum(years, 1)} yrs`}
          sub={`${curve.length} contributions of ${usd(perRebalance)}`} />
      </div>

      <ResponsiveContainer width="100%" height={240}>
        <AreaChart data={curve} margin={{ top: 6, right: 12, bottom: 0, left: 0 }}>
          <defs>
            <linearGradient id="gModel" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#0d9488" stopOpacity={0.25} />
              <stop offset="100%" stopColor="#0d9488" stopOpacity={0.02} />
            </linearGradient>
          </defs>
          <CartesianGrid stroke="#eef0f3" vertical={false} />
          <XAxis dataKey="date" tick={{ fill: "#8a919e", fontSize: 11 }} minTickGap={70} stroke="#e4e7ec" />
          <YAxis tick={{ fill: "#8a919e", fontSize: 11 }} stroke="#e4e7ec" width={54}
            tickFormatter={(v) => "$" + (v / 1000).toFixed(0) + "k"} />
          <Tooltip contentStyle={tt} labelStyle={{ color: "#58606d" }} formatter={(v) => usd(v)} />
          <Legend wrapperStyle={{ fontSize: 12 }} />
          <Area type="monotone" dataKey="model" name="Algo value" stroke="#0d9488" strokeWidth={2} fill="url(#gModel)" />
          <Line type="monotone" dataKey="contributed" name="Money invested" stroke="#8a919e"
            strokeWidth={1.4} strokeDasharray="4 4" dot={false} />
          <Line type="monotone" dataKey="spy" name="SPY" stroke="#4f6bed" strokeWidth={1.2} dot={false} />
        </AreaChart>
      </ResponsiveContainer>
      <p className="faint" style={{ fontSize: 11.5, marginTop: 10 }}>
        Contributions are pooled at each ~3-week rebalance. Past backtest performance is not a
        prediction — over this period the algo returned roughly the same as naive copying and
        trailed a lump-sum SPY buy-and-hold.
      </p>
    </div>
  );
}

// DCA: contribute `c` at each equity point; each contribution compounds by the
// series' growth from that date to the end. Returns a value-over-time curve.
function dca(series, c) {
  const model = series.model, spy = series.spy;
  if (!model?.length) return null;
  const n = model.length;
  const curve = [];
  for (let t = 0; t < n; t++) {
    let mv = 0, sv = 0;
    for (let i = 0; i <= t; i++) {
      mv += c * (model[t].equity / model[i].equity);
      if (spy?.[i]) sv += c * (spy[t].equity / spy[i].equity);
    }
    curve.push({ date: model[t].date, model: mv, spy: sv, contributed: c * (t + 1) });
  }
  const last = curve[curve.length - 1];
  return { curve, final: { model: last.model, spy: last.spy, contributed: last.contributed } };
}

function EventStudy({ es }) {
  const rows = es.by_amount || [];
  const horizons = ["ar_5", "ar_21", "ar_63", "ar_126"];
  return (
    <div className="card">
      <div className="card-h"><h3>Event study — CAR by trade size</h3>
        <small>mean abnormal return after filing, House purchases</small></div>
      <table>
        <thead>
          <tr>
            <th>Amount bucket</th><th className="num">n</th>
            {horizons.map((h) => <th key={h} className="num">{h.replace("ar_", "")}d</th>)}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i}>
              <td>{r.amount_bucket}</td>
              <td className="num mono dim">{r.n}</td>
              {horizons.map((h) => {
                const v = r[`${h}_mean`];
                return <td key={h} className={"num mono " + (v >= 0 ? "pos" : "neg")}>{fmtPct(v, 2)}</td>;
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function mergeSeries(series) {
  const byDate = {};
  for (const strat of Object.keys(series)) {
    for (const pt of series[strat]) {
      byDate[pt.date] = byDate[pt.date] || { date: pt.date };
      byDate[pt.date][strat] = pt.equity;
    }
  }
  return Object.values(byDate).sort((a, b) => a.date.localeCompare(b.date));
}

const tt = {
  background: "#ffffff", border: "1px solid #e4e7ec", borderRadius: 9, fontSize: 12,
  boxShadow: "0 4px 16px rgba(16,24,40,0.08)",
};
