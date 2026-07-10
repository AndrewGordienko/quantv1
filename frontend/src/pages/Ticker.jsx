import { useParams, Link } from "react-router-dom";
import {
  ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip,
  ReferenceDot, CartesianGrid,
} from "recharts";
import { useApi, fmtMoney } from "../api.js";
import { PageHead, Stat, Loading, Empty } from "../components/bits.jsx";

export default function Ticker() {
  const { ticker } = useParams();
  const { data, loading } = useApi(`/ticker/${ticker}`, [ticker]);
  if (loading) return <Loading />;
  if (!data) return <Empty>Not found.</Empty>;

  const prices = data.prices || [];
  const priceByDate = Object.fromEntries(prices.map((p) => [p.date, p.close]));
  const buys = (data.trades || []).filter((t) => t.tx_type === "purchase");
  const sells = (data.trades || []).filter((t) => t.tx_type === "sale");
  const markers = (data.trades || [])
    .map((t) => ({ date: t.filing_date, close: priceByDate[t.filing_date], type: t.tx_type }))
    .filter((m) => m.close != null);

  return (
    <>
      <PageHead eyebrow="Security" title={data.ticker}>
        {data.sector?.sector && data.sector.sector !== "Unknown"
          ? `${data.sector.sector} · ${data.sector.industry}`
          : "Congressional trading activity and disclosure markers."}
      </PageHead>

      <div className="grid grid-4" style={{ marginBottom: 20 }}>
        <Stat label="Total trades" value={data.trades.length} />
        <Stat label="Purchases" value={buys.length} tone="pos" />
        <Stat label="Sales" value={sells.length} tone="neg" />
        <Stat label="Distinct members" value={new Set(data.trades.map((t) => t.member)).size} />
      </div>

      {prices.length > 0 && (
        <div className="card" style={{ marginBottom: 20 }}>
          <div className="card-h"><h3>Price & disclosures</h3>
            <small>▲ purchase · ▼ sale (at filing date)</small></div>
          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={prices} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
              <CartesianGrid stroke="#eef0f3" vertical={false} />
              <XAxis dataKey="date" tick={{ fill: "#8a919e", fontSize: 11 }}
                minTickGap={60} stroke="#e4e7ec" />
              <YAxis tick={{ fill: "#8a919e", fontSize: 11 }} stroke="#e4e7ec"
                domain={["auto", "auto"]} width={48} />
              <Tooltip contentStyle={tooltipStyle} labelStyle={{ color: "#58606d" }} />
              <Line type="monotone" dataKey="close" stroke="#4f6bed" strokeWidth={1.5} dot={false} />
              {markers.map((m, i) => (
                <ReferenceDot key={i} x={m.date} y={m.close} r={3.5}
                  fill={m.type === "purchase" ? "#16a34a" : "#dc2626"} stroke="none" />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      <div className="card">
        <div className="card-h"><h3>Who traded it</h3></div>
        <table>
          <thead>
            <tr><th>Filed</th><th>Member</th><th>Chamber</th><th>Type</th><th className="num">Size</th></tr>
          </thead>
          <tbody>
            {data.trades.map((t, i) => (
              <tr key={i}>
                <td className="dim">{t.filing_date}</td>
                <td>{t.member}</td>
                <td className="faint">{t.chamber}</td>
                <td>
                  <span className={"pill " + (t.tx_type === "purchase" ? "buy" : t.tx_type === "sale" ? "sell" : "")}>
                    {t.tx_type}
                  </span>
                </td>
                <td className="num mono">{fmtMoney(t.amount_mid)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}

const tooltipStyle = {
  background: "#ffffff", border: "1px solid #e4e7ec", borderRadius: 9, fontSize: 12,
  boxShadow: "0 4px 16px rgba(16,24,40,0.08)",
};
