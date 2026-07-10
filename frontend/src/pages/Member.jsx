import { useParams, Link } from "react-router-dom";
import { useApi, fmtPct, fmtMoney } from "../api.js";
import { PageHead, Stat, PartyPill, Loading, Empty } from "../components/bits.jsx";

export default function Member() {
  const { key } = useParams();
  const { data, loading } = useApi(`/member/${key}`, [key]);
  if (loading) return <Loading />;
  if (!data?.profile?.member) return <Empty>Member not found.</Empty>;

  const { profile, skill, trades } = data;
  const jur = profile.committees?.jurisdiction_sectors || [];
  const committees = profile.committees?.committees || [];

  return (
    <>
      <PageHead eyebrow="Member profile" title={profile.member}>
        <span className="row" style={{ gap: 8 }}>
          <PartyPill party={profile.party} />
          {profile.chamber && <span className="pill">{profile.chamber}</span>}
          {profile.state && <span className="pill">{profile.state}</span>}
        </span>
      </PageHead>

      <div className="grid grid-4" style={{ marginBottom: 20 }}>
        <Stat label="Total trades" value={skill?.n_trades ?? trades.length} />
        <Stat label="Purchases scored" value={skill?.n_purchases ?? "—"} />
        <Stat label="Shrunk skill" value={fmtPct(skill?.shrunk_car, 2)}
          tone={skill?.shrunk_car >= 0 ? "pos" : "neg"}
          sub={skill ? `raw ${fmtPct(skill.raw_car, 1)}` : ""} />
        <Stat label="Hit rate vs SPY" value={fmtPct(skill?.hit_rate, 0)} />
      </div>

      {committees.length > 0 && (
        <div className="card" style={{ marginBottom: 20 }}>
          <div className="card-h"><h3>Committees</h3>
            <small>{jur.length} sectors under jurisdiction</small></div>
          <div className="chips" style={{ marginBottom: 12 }}>
            {committees.map((c) => <span key={c} className="chip">{c}</span>)}
          </div>
          {jur.length > 0 && (
            <div className="chips">
              {jur.map((s) => <span key={s} className="chip up">{s}</span>)}
            </div>
          )}
        </div>
      )}

      <div className="card">
        <div className="card-h"><h3>Trade history</h3><small>{trades.length} most recent</small></div>
        <table>
          <thead>
            <tr>
              <th>Filed</th><th>Traded</th><th>Ticker</th><th>Type</th>
              <th className="num">Size</th><th>Owner</th>
            </tr>
          </thead>
          <tbody>
            {trades.map((t, i) => (
              <tr key={i}>
                <td className="dim">{t.filing_date}</td>
                <td className="faint">{t.tx_date}</td>
                <td><Link className="tick" to={`/ticker/${t.ticker}`}>{t.ticker}</Link></td>
                <td>
                  <span className={"pill " + (t.tx_type === "purchase" ? "buy" : t.tx_type === "sale" ? "sell" : "")}>
                    {t.tx_type}
                  </span>
                </td>
                <td className="num mono">{fmtMoney(t.amount_mid)}</td>
                <td className="faint">{t.owner}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}
