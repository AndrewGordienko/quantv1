import { Link } from "react-router-dom";
import { useApi, fmtMoney, daysAgo } from "../api.js";
import { PageHead, ScoreBar, RationaleChips, Loading, Empty } from "../components/bits.jsx";

export default function Feed() {
  const { data, loading } = useApi("/feed?limit=80");
  return (
    <>
      <PageHead eyebrow="Disclosures" title="Live Feed">
        The newest congressional purchases as they are filed, each scored by the model in real
        time. &quot;Stale&quot; shows how many days passed between the trade and its public disclosure —
        the edge you never get back.
      </PageHead>
      {loading ? <Loading /> : !data?.length ? (
        <Empty>No scored disclosures yet.</Empty>
      ) : (
        <div className="card">
          <table>
            <thead>
              <tr>
                <th>Filed</th>
                <th>Member</th>
                <th>Ticker</th>
                <th className="num">Size</th>
                <th>Stale</th>
                <th style={{ width: 130 }}>Score</th>
                <th>Thesis</th>
              </tr>
            </thead>
            <tbody>
              {data.map((r, i) => (
                <tr key={i}>
                  <td className="dim">{daysAgo(r.filing_date)}</td>
                  <td>{r.member}</td>
                  <td><Link className="tick" to={`/ticker/${r.ticker}`}>{r.ticker}</Link></td>
                  <td className="num mono">{fmtMoney(r.amount_mid)}</td>
                  <td className={r.disclosure_lag > 40 ? "neg" : "dim"}>
                    {r.disclosure_lag != null ? `${r.disclosure_lag}d` : "—"}
                  </td>
                  <td><ScoreBar score={r.score} /></td>
                  <td style={{ maxWidth: 260 }}><RationaleChips contribs={r.contribs} max={3} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </>
  );
}
