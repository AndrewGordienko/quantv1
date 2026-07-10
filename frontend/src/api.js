import { useEffect, useState } from "react";

export async function api(path) {
  const res = await fetch(`/api${path}`);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

// Small data-fetching hook with loading/error state.
export function useApi(path, deps = []) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);
  useEffect(() => {
    let alive = true;
    setLoading(true);
    api(path)
      .then((d) => alive && (setData(d), setError(null)))
      .catch((e) => alive && setError(e.message))
      .finally(() => alive && setLoading(false));
    return () => (alive = false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  return { data, error, loading };
}

export const fmtPct = (x, d = 1) =>
  x == null || isNaN(x) ? "—" : `${(x * 100).toFixed(d)}%`;
export const fmtMoney = (x) => {
  if (x == null) return "—";
  if (x >= 1e6) return `$${(x / 1e6).toFixed(1)}M`;
  if (x >= 1e3) return `$${(x / 1e3).toFixed(0)}k`;
  return `$${x}`;
};
export const fmtNum = (x, d = 2) => (x == null || isNaN(x) ? "—" : x.toFixed(d));
export const daysAgo = (dateStr) => {
  const d = Math.floor((Date.now() - new Date(dateStr)) / 86400000);
  return d <= 0 ? "today" : d === 1 ? "1d ago" : `${d}d ago`;
};
