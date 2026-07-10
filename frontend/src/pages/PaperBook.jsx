import { useState, useMemo } from "react";
import { Link } from "react-router-dom";
import { useApi, fmtPct } from "../api.js";
import { PageHead, Stat, Loading, Empty } from "../components/bits.jsx";

const usd = (x, d = 0) =>
  x == null ? "—" : `$${x.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d })}`;

export default function PaperBook() {
  const { data, loading } = useApi("/forward/book");
  const { data: evald } = useApi("/forward/evaluate");
  const [capital, setCapital] = useState(() => Number(localStorage.getItem("quantv1_paper")) || 10000);
  const setCap = (v) => { setCapital(v); localStorage.setItem("quantv1_paper", String(v)); };

  if (loading) return <><Head /><Loading /></>;
  const large = data?.books?.LARGE;
  const shadows = ["LARGE_NEW", "LARGE_SPOUSE", "LARGE_15_30D", "LARGE_250K_1M"];

  return (
    <>
      <Head version={data?.version} />

      {evald && (
        <div className="card" style={{ marginBottom: 20, borderColor: "rgba(217,119,6,0.4)",
          background: "rgba(217,119,6,0.04)" }}>
          <div className="row" style={{ justifyContent: "space-between", flexWrap: "wrap", gap: 12 }}>
            <div>
              <div className="eyebrow" style={{ color: "var(--warn)" }}>Pre-registered evaluation</div>
              <div style={{ fontSize: 14, fontWeight: 600 }}>{evald.status}</div>
              <div className="faint" style={{ fontSize: 12, marginTop: 4 }}>
                Day {evald.trading_days_elapsed} · {evald.completed_positions} completed positions ·
                preliminary review at 6mo / 50 positions · deployment at 12mo / 100.
                Primary test: {evald.primary_test}.
              </div>
            </div>
            <div className="pill" style={{ alignSelf: "center" }}>frozen {evald.forward_start}</div>
          </div>
        </div>
      )}

      {/* Primary LARGE book */}
      <div className="card" style={{ marginBottom: 20, borderColor: "rgba(13,148,136,0.4)" }}>
        <div className="card-h">
          <h3>LARGE — primary paper book</h3>
          <small>original rule (≥ $50k), unchanged · what to trade in paper today</small>
        </div>
        <div className="calc" style={{ marginBottom: 18, boxShadow: "none", background: "var(--bg)" }}>
          <div className="calc-field">
            <label>Paper capital</label>
            <div className="calc-input"><span>$</span>
              <input type="number" min="0" step="1000" value={capital}
                onChange={(e) => setCap(Math.max(0, Number(e.target.value)))} /></div>
          </div>
          <div className="calc-summary">
            <div className="item"><div className="k">Gross</div>
              <div className="v">{fmtPct(large?.gross, 0)}</div></div>
            <div className="item"><div className="k">Cash</div>
              <div className="v" style={{ color: "var(--text-dim)" }}>{fmtPct(large?.cash, 0)}</div></div>
            <div className="item"><div className="k">Positions</div>
              <div className="v">{large?.positions?.length || 0}</div></div>
          </div>
        </div>
        {large?.positions?.length ? <BookTable positions={large.positions} capital={capital} />
          : <Empty>No large disclosures in the current window — the book is in cash. That is correct behavior.</Empty>}
      </div>

      {/* Shadow books */}
      <div className="card">
        <div className="card-h"><h3>Shadow variants — observational, no capital</h3>
          <small>discovered in the audit · tracked, never funded until they win prospectively</small></div>
        <table>
          <thead><tr><th>Strategy</th><th className="num">Positions</th><th className="num">Gross</th><th>Tickers</th></tr></thead>
          <tbody>
            {shadows.map((s) => {
              const b = data?.books?.[s];
              return (
                <tr key={s}>
                  <td className="mono">{s}</td>
                  <td className="num mono">{b?.positions?.length || 0}</td>
                  <td className="num mono dim">{fmtPct(b?.gross, 0)}</td>
                  <td className="dim" style={{ fontSize: 12.5 }}>
                    {(b?.positions || []).map((p) => p.ticker).join(", ") || "—"}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </>
  );
}

function BookTable({ positions, capital }) {
  const sized = useMemo(() => positions.map((p) => {
    const alloc = p.weight * capital;
    const shares = p.decision_price ? Math.floor(alloc / p.decision_price) : 0;
    return { ...p, alloc, shares, cost: shares * (p.decision_price || 0) };
  }), [positions, capital]);
  return (
    <table>
      <thead><tr>
        <th>Ticker</th><th>Weight</th><th className="num">Price</th><th className="num">Shares</th>
        <th className="num">Amount</th><th>Source (member · filed)</th>
      </tr></thead>
      <tbody>
        {sized.map((p) => (
          <tr key={p.ticker}>
            <td><Link className="tick" to={`/ticker/${p.ticker}`}>{p.ticker}</Link></td>
            <td className="mono">{fmtPct(p.weight)}</td>
            <td className="num mono dim">{p.decision_price ? usd(p.decision_price, 2) : "—"}</td>
            <td className="num mono" style={{ fontWeight: 600 }}>{p.shares || "—"}</td>
            <td className="num mono">{usd(p.cost)}</td>
            <td className="dim" style={{ fontSize: 12.5 }}>{p.source_member} · {p.source_filing_date}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function Head({ version }) {
  return (
    <PageHead eyebrow="Frozen forward record" title="Paper Book">
      The immutable paper-trading record. Plain LARGE is the primary strategy; decisions are recorded
      before execution and never edited. This is a PAPER book — real capital waits for the
      pre-registered forward evaluation. {version && <span className="mono faint">v{version}</span>}
    </PageHead>
  );
}
