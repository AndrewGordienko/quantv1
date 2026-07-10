"""Ingest Fama-French 3-factor + momentum daily series into DuckDB `factors`.

Source: Ken French's data library (public, free). We need real factor returns
so the event study can measure political *alpha* — the residual after removing
market, size, value and momentum exposure — instead of counting a tech/growth
tilt as political skill (the beta=1-vs-SPY problem in v1).

Values in the source CSVs are in percent; we store decimals.
"""

from __future__ import annotations

import io
import urllib.request
import zipfile

import pandas as pd

from ..db import connect

FF3_URL = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
           "F-F_Research_Data_Factors_daily_CSV.zip")
MOM_URL = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
           "F-F_Momentum_Factor_daily_CSV.zip")
_UA = {"User-Agent": "quantv1-ingest"}


def _fetch_csv(url: str) -> str:
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=90) as r:
        blob = r.read()
    zf = zipfile.ZipFile(io.BytesIO(blob))
    name = zf.namelist()[0]
    return zf.read(name).decode("latin-1")


def _parse_daily(text: str, cols: list[str]) -> pd.DataFrame:
    """Extract the daily block: rows whose first token is an 8-digit date."""
    rows = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        tok = parts[0]
        if len(tok) == 8 and tok.isdigit():
            try:
                vals = [float(x) for x in parts[1:1 + len(cols)]]
            except ValueError:
                continue
            if len(vals) == len(cols):
                rows.append([pd.to_datetime(tok, format="%Y%m%d").date(), *vals])
    df = pd.DataFrame(rows, columns=["date", *cols])
    # source is in percent
    for c in cols:
        df[c] = df[c] / 100.0
    return df


def ingest(verbose: bool = True) -> dict:
    ff3 = _parse_daily(_fetch_csv(FF3_URL), ["mkt_rf", "smb", "hml", "rf"])
    mom = _parse_daily(_fetch_csv(MOM_URL), ["mom"])
    df = ff3.merge(mom, on="date", how="left")
    df["mom"] = df["mom"].fillna(0.0)
    df = df[["date", "mkt_rf", "smb", "hml", "mom", "rf"]]

    con = connect()
    con.execute("DELETE FROM factors")
    con.executemany(
        "INSERT INTO factors VALUES (?,?,?,?,?,?)",
        df.itertuples(index=False, name=None),
    )
    n = con.execute("SELECT COUNT(*) FROM factors").fetchone()[0]
    rng = con.execute("SELECT MIN(date), MAX(date) FROM factors").fetchone()
    con.close()
    if verbose:
        print(f"Factors: {n} daily rows, {rng[0]} .. {rng[1]}")
    return {"rows": n, "start": str(rng[0]), "end": str(rng[1])}


if __name__ == "__main__":
    ingest()
