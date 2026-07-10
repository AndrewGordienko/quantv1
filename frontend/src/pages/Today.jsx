import { useState, useMemo, Fragment } from "react";
import { Link } from "react-router-dom";
import { useApi, fmtPct } from "../api.js";
import { PageHead, ScoreBar, RationaleChips, Loading, Empty } from "../components/bits.jsx";

const PRESETS = [1000, 10000, 50000, 100000];
const usd = (x, d = 0) =>
  x == null ? "—" : `$${x.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d })}`;

export default function Today() {
  const { data, loading } = useApi("/portfolio/today");
  const [open, setOpen] = useState(null);
  const [capital, setCapital] = useState(() => {
    const saved = Number(localStorage.getItem("quantv1_capital"));
    return saved > 0 ? saved : 10000;
  });

  const setCap = (v) => {
    setCapital(v);
    localStorage.setItem("quantv1_capital", String(v));
  };

  // Auto-recalculated on every render from `capital` + live prices.
  const sized = useMemo(() => {
    if (!data?.positions) return [];
    return data.positions.map((p) => {
      const alloc = p.weight * capital;
      const shares = p.price ? Math.floor(alloc / p.price) : 0;
      const cost = shares * (p.price || 0);
      return { ...p, alloc, shares, cost };
    });
  }, [data, capital]);

  const deployed = sized.reduce((s, p) => s + p.cost, 0);
  const cash = capital - deployed;

  if (loading) return <><Head /><Loading /></>;
  if (!data || !data.positions?.length)
    return (
      <>
        <Head />
        <Empty>
          No portfolio yet. Run <code>scripts/daily_update.py</code> to generate today&apos;s book.
        </Empty>
      </>
    );

  const { deltas, as_of_date } = data;
  const topWeight = Math.max(...sized.map((p) => p.weight));

  return (
    <>
      <Head asOf={as_of_date} />

      {/* Investment calculator */}
      <div className="calc">
        <div className="calc-field">
          <label>Investment amount</label>
          <div className="calc-input">
            <span>$</span>
            <input
              type="number" min="0" step="100" value={capital}
              onChange={(e) => setCap(Math.max(0, Number(e.target.value)))}
            />
          </div>
        </div>
        <div className="calc-presets">
          {PRESETS.map((v) => (
            <button key={v} className={"preset" + (capital === v ? " active" : "")}
              onClick={() => setCap(v)}>
              {v >= 1000 ? `$${v / 1000}k` : `$${v}`}
            </button>
          ))}
        </div>
        <div className="calc-summary">
          <div className="item">
            <div className="k">Deployed</div>
            <div className="v">{usd(deployed)}</div>
          </div>
          <div className="item">
            <div className="k">Cash left</div>
            <div className="v" style={{ color: "var(--text-dim)" }}>{usd(cash)}</div>
          </div>
          <div className="item">
            <div className="k">Positions</div>
            <div className="v">{sized.filter((p) => p.shares > 0).length}</div>
          </div>
        </div>
      </div>

      <div className="grid grid-4" style={{ marginBottom: 20 }}>
        <Stat label="New buys" value={deltas.buys?.length || 0} tone="pos"
          sub={deltas.buys?.slice(0, 3).join(", ") || "none"} />
        <Stat label="Exits" value={deltas.sells?.length || 0} tone="neg"
          sub={deltas.sells?.slice(0, 3).join(", ") || "none"} />
        <Stat label="Top conviction" value={sized[0]?.ticker}
          sub={`${fmtPct(sized[0]?.weight)} · score ${sized[0]?.score?.toFixed(2)}`} />
        <Stat label="As of" value={as_of_date} sub="latest disclosures" />
      </div>

      <div className="card">
        <div className="card-h">
          <h3>Target portfolio</h3>
          <small>share counts update live with the amount above · click a row for the thesis</small>
        </div>
        <table>
          <thead>
            <tr>
              <th style={{ width: 30 }}></th>
              <th>Ticker</th>
              <th>Weight</th>
              <th className="num">Price</th>
              <th className="num">Shares</th>
              <th className="num">Amount</th>
              <th>Score</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {sized.map((p, i) => {
              const r = p.rationale || {};
              const isOpen = open === i;
              return (
                <Fragment key={p.ticker}>
                  <tr className="clickable" onClick={() => setOpen(isOpen ? null : i)}>
                    <td className="faint mono">{i + 1}</td>
                    <td>
                      <Link className="tick" to={`/ticker/${p.ticker}`} onClick={(e) => e.stopPropagation()}>
                        {p.ticker}
                      </Link>
                    </td>
                    <td style={{ minWidth: 120 }}>
                      <div className="row" style={{ gap: 8 }}>
                        <div className="scorebar" style={{ flex: 1 }}>
                          <div style={{ width: `${(p.weight / topWeight) * 100}%` }} />
                        </div>
                        <span className="mono faint" style={{ fontSize: 12 }}>{fmtPct(p.weight)}</span>
                      </div>
                    </td>
                    <td className="num mono dim">{p.price ? usd(p.price, 2) : "—"}</td>
                    <td className="num mono" style={{ fontWeight: 600 }}>{p.price ? p.shares : "—"}</td>
                    <td className="num mono">{usd(p.cost)}</td>
                    <td style={{ width: 120 }}><ScoreBar score={p.score} /></td>
                    <td className="faint">{isOpen ? "▲" : "▼"}</td>
                  </tr>
                  {isOpen && (
                    <tr className="expand-row">
                      <td colSpan={8}>
                        <div className="spread" style={{ alignItems: "flex-start", gap: 24 }}>
                          <div style={{ flex: 1 }}>
                            <div className="eyebrow">Why this is in the book</div>
                            <RationaleChips contribs={r.contribs} max={6} />
                          </div>
                          <div style={{ flex: 1 }}>
                            <div className="eyebrow">Disclosed by</div>
                            <div className="chips">
                              {(r.members || []).map((m) => (
                                <span key={m} className="chip">{m}</span>
                              ))}
                            </div>
                          </div>
                          <div style={{ minWidth: 150 }}>
                            <div className="eyebrow">Order</div>
                            <div className="dim" style={{ fontSize: 13 }}>
                              {p.shares} sh × {usd(p.price, 2)}<br />
                              = <strong>{usd(p.cost)}</strong>
                              <div className="faint" style={{ fontSize: 11.5, marginTop: 4 }}>
                                price as of {p.price_date || "—"}
                              </div>
                            </div>
                          </div>
                        </div>
                      </td>
                    </tr>
                  )}
                </Fragment>
              );
            })}
          </tbody>
        </table>
      </div>
    </>
  );
}

// local Stat (kept here to allow tone styling)
function Stat({ label, value, sub, tone }) {
  return (
    <div className="stat">
      <div className="label">{label}</div>
      <div className="value" style={tone ? { color: `var(--${tone})` } : null}>{value}</div>
      {sub && <div className="sub">{sub}</div>}
    </div>
  );
}

function Head({ asOf }) {
  return (
    <PageHead eyebrow="Daily recommendation" title="Today's Portfolio">
      A model-scored book built from congressional purchases disclosed in the last 90 days,
      capped per name and per sector. Enter an amount below to see exactly how many shares of
      each to buy — it recalculates as you type. {asOf && <span className="tag-live" style={{ marginLeft: 8 }}>live</span>}
    </PageHead>
  );
}
