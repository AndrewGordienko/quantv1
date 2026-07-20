import { useApi, fmtNum } from "../api.js";
import { Empty, Loading, PageHead, Stat } from "../components/bits.jsx";

export default function OpeningFlow() {
  const { data, loading } = useApi("/forward/opening-flow");
  if (loading) return <Loading />;
  const screen = data?.screen || {};
  const results = screen.results || {};
  const decisions = data?.decisions || [];
  const orders = data?.orders || [];
  return (
    <>
      <PageHead eyebrow="Prospective experiment" title="Opening Flow">
        Frozen 10:00 ET canary. CASH_CHAMPION is the champion; P1/P2/P3 are
        recorded separately. P3 has no promotion authority and can only submit
        a small paper order with explicit credentials.
      </PageHead>
      <div className="grid grid-4" style={{ marginBottom: 20 }}>
        {["P0", "P1", "P2", "P3"].map((p) => {
          const r = results[p] || {};
          return <Stat key={p} label={p} value={r.n_trades == null ? "—" : r.n_trades}
            sub={r.mean_net_bps == null ? "not screened" : `${fmtNum(r.mean_net_bps, 1)} bps mean net`}
            tone={p === "P0" ? null : (r.mean_net_bps > 0 ? "pos" : "neg")} />;
        })}
      </div>
      <div className="card" style={{ marginBottom: 20 }}>
        <div className="card-h"><h3>Immutable decisions</h3><small>{decisions.length} rows</small></div>
        {!decisions.length ? <Empty>No live decision yet — run the 10:00 ET paper tick.</Empty> : (
          <table><thead><tr><th>Time</th><th>Book</th><th>Ticker</th><th>Action</th><th>Status</th></tr></thead>
            <tbody>{decisions.map((d) => <tr key={d.decision_id}><td className="mono dim">{d.decision_ts}</td>
              <td>{d.book}</td><td className="tick">{d.ticker || "—"}</td>
              <td className={d.action === "BUY" ? "pos" : d.action === "SELL" ? "neg" : "dim"}>{d.action}</td>
              <td>{d.status}</td></tr>)}</tbody></table>
        )}
      </div>
      <div className="card">
        <div className="card-h"><h3>Paper orders and marks</h3><small>fills populate after broker reconciliation</small></div>
        {!orders.length ? <Empty>No paper orders recorded.</Empty> : <table><thead><tr><th>Book</th><th>Ticker</th><th>Side</th><th>Qty</th><th>Status</th></tr></thead>
          <tbody>{orders.map((o) => <tr key={o.order_id}><td>{o.book}</td><td className="tick">{o.ticker}</td><td>{o.side}</td><td>{o.qty}</td><td>{o.status}</td></tr>)}</tbody></table>}
      </div>
    </>
  );
}
