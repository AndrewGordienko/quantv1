import { fmtNum } from "../api.js";

export function PageHead({ eyebrow, title, children }) {
  return (
    <div className="page-head">
      {eyebrow && <div className="eyebrow">{eyebrow}</div>}
      <h2>{title}</h2>
      {children && <p>{children}</p>}
    </div>
  );
}

export function Stat({ label, value, sub, tone }) {
  return (
    <div className="stat">
      <div className="label">{label}</div>
      <div className="value" style={tone ? { color: `var(--${tone})` } : null}>{value}</div>
      {sub && <div className="sub">{sub}</div>}
    </div>
  );
}

export function ScoreBar({ score }) {
  return (
    <div className="row" style={{ gap: 8 }}>
      <div className="scorebar" style={{ flex: 1 }}>
        <div style={{ width: `${Math.round((score || 0) * 100)}%` }} />
      </div>
      <span className="mono" style={{ fontSize: 12.5, minWidth: 34 }}>
        {score == null ? "—" : score.toFixed(2)}
      </span>
    </div>
  );
}

// Contribution chips from SHAP values.
export function RationaleChips({ contribs, max = 4 }) {
  if (!contribs || !contribs.length) return <span className="faint">—</span>;
  return (
    <div className="chips">
      {contribs.slice(0, max).map((c, i) => (
        <span key={i} className={"chip " + (c.contribution >= 0 ? "up" : "down")}>
          {c.contribution >= 0 ? "↑" : "↓"} {c.label}
        </span>
      ))}
    </div>
  );
}

export function PartyPill({ party }) {
  if (!party) return null;
  const dem = /democr/i.test(party);
  const rep = /republic/i.test(party);
  return <span className={"pill " + (dem ? "dem" : rep ? "rep" : "")}>{party}</span>;
}

// Credible-interval bar for the leaderboard (shared x-domain passed in).
export function CIBar({ low, high, point, domain }) {
  const [dlo, dhi] = domain;
  const span = dhi - dlo || 1;
  const pos = (v) => `${((v - dlo) / span) * 100}%`;
  const zero = pos(0);
  return (
    <div className="cibar">
      <div className="zero" style={{ left: zero }} />
      <div className="range" style={{ left: pos(low), width: `${((high - low) / span) * 100}%` }} />
      <div className="pt" style={{ left: pos(point) }} title={fmtNum(point * 100, 2) + "%"} />
    </div>
  );
}

export function Loading() { return <div className="loading">loading…</div>; }
export function Empty({ children }) { return <div className="empty">{children}</div>; }
