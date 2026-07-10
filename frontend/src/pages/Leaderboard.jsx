import { Link } from "react-router-dom";
import { useApi, fmtPct } from "../api.js";
import { PageHead, CIBar, PartyPill, Loading, Empty } from "../components/bits.jsx";

export default function Leaderboard() {
  const { data, loading } = useApi("/leaderboard?min_purchases=5");
  return (
    <>
      <PageHead eyebrow="Who is actually good" title="Skill Leaderboard">
        Per-member skill measured as the abnormal return after their purchases, then shrunk
        toward the population mean by empirical Bayes so a lucky 5-trade member can&apos;t top a
        proven 200-trade one. The bar shows the 95% credible interval — width is uncertainty.
      </PageHead>
      {loading ? <Loading /> : !data?.length ? (
        <Empty>No skill scores yet.</Empty>
      ) : <Table rows={data} />}
    </>
  );
}

function Table({ rows }) {
  const lo = Math.min(...rows.map((r) => r.ci_low), 0);
  const hi = Math.max(...rows.map((r) => r.ci_high), 0);
  const pad = (hi - lo) * 0.05;
  const domain = [lo - pad, hi + pad];
  return (
    <div className="card">
      <table>
        <thead>
          <tr>
            <th style={{ width: 30 }}></th>
            <th>Member</th>
            <th className="num">Buys</th>
            <th className="num">Raw</th>
            <th className="num">Shrunk skill</th>
            <th>95% credible interval</th>
            <th className="num">Hit rate</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={r.member_key} className="clickable"
              onClick={() => (window.location.href = `/member/${r.member_key}`)}>
              <td className="faint mono">{i + 1}</td>
              <td>
                <Link to={`/member/${r.member_key}`} onClick={(e) => e.stopPropagation()}>
                  {r.member}
                </Link>
              </td>
              <td className="num mono dim">{r.n_purchases}</td>
              <td className="num mono dim">{fmtPct(r.raw_car, 1)}</td>
              <td className={"num mono " + (r.shrunk_car >= 0 ? "pos" : "neg")}>
                {fmtPct(r.shrunk_car, 2)}
              </td>
              <td><CIBar low={r.ci_low} high={r.ci_high} point={r.shrunk_car} domain={domain} /></td>
              <td className="num mono dim">{fmtPct(r.hit_rate, 0)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
