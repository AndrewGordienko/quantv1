import { NavLink, Outlet } from "react-router-dom";
import { useApi } from "../api.js";

const NAV = [
  { to: "/", label: "Today", ico: "◉", end: true },
  { to: "/paper", label: "Paper Book", ico: "◈" },
  { to: "/opening-flow", label: "Opening Flow", ico: "↗" },
  { to: "/feed", label: "Live Feed", ico: "≋" },
  { to: "/leaderboard", label: "Leaderboard", ico: "▲" },
  { to: "/research", label: "Research", ico: "⌗" },
  { to: "/evidence", label: "Evidence", ico: "✓" },
];

export default function Layout() {
  const { data: stats } = useApi("/stats");
  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-dot" />
          <div>
            <h1>quantv1</h1>
            <span>congressional alpha</span>
          </div>
        </div>
        <nav>
          {NAV.map((n) => (
            <NavLink key={n.to} to={n.to} end={n.end}
              className={({ isActive }) => "nav-link" + (isActive ? " active" : "")}>
              <span className="nav-ico">{n.ico}</span>
              {n.label}
            </NavLink>
          ))}
        </nav>
        <div className="sidebar-foot">
          {stats ? (
            <small>
              {stats.trades?.toLocaleString()} trades · {stats.members} members
              <br />
              latest filing {stats.latest_filing}
            </small>
          ) : (
            <small>connecting…</small>
          )}
        </div>
      </aside>
      <main className="main">
        <Outlet />
      </main>
    </div>
  );
}
