"""Crypto perpetual-futures ingest — BTC/ETH only, free Binance USD-M API.

The venue pivot (docs/strategy_crypto.md): crypto perps give FREE OHLCV + funding
+ order books, no PDT, symmetric shorts — which unblocks the intraday research the
equity engine was data-gated on. We restrict to BTC/ETH majors, which sidesteps
crypto's worst backtest traps (survivorship from dead tokens, wash-traded volume
on thin pairs).

Fetches daily klines and the full funding-rate history (funding is a real,
recurring cost/return term unique to perps — it must be modeled, not ignored).
Stores Parquet under data/crypto/ (gitignored, regenerable). Public endpoints,
no key required.
"""

from __future__ import annotations

import time
from pathlib import Path

try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass
import pandas as pd
import requests

from ..config import DATA_DIR

FAPI = "https://fapi.binance.com"
OUT = DATA_DIR / "crypto"
UA = {"User-Agent": "Mozilla/5.0 (quantv1 crypto research)"}
SYMBOLS = ("BTCUSDT", "ETHUSDT")


def _get(path: str, params: dict) -> list:
    r = requests.get(FAPI + path, params=params, headers=UA, timeout=25)
    r.raise_for_status()
    return r.json()


def fetch_klines(symbol: str, interval: str = "1d") -> pd.DataFrame:
    """Full history of klines, paginated forward from the listing date."""
    rows, start = [], 0
    while True:
        batch = _get("/fapi/v1/klines",
                     {"symbol": symbol, "interval": interval,
                      "startTime": start, "limit": 1500})
        if not batch:
            break
        rows.extend(batch)
        nxt = batch[-1][0] + 1
        if nxt <= start or len(batch) < 1500:
            start = nxt
            if len(batch) < 1500:
                break
        start = nxt
        time.sleep(0.25)
    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time",
            "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]
    df = pd.DataFrame(rows, columns=cols)
    df["date"] = pd.to_datetime(df["open_time"], unit="ms")
    for c in ("open", "high", "low", "close", "volume", "quote_volume", "taker_buy_base"):
        df[c] = df[c].astype(float)
    df["trades"] = df["trades"].astype(int)
    return df[["date", "open", "high", "low", "close", "volume", "quote_volume",
               "trades", "taker_buy_base"]].drop_duplicates("date").sort_values("date")


def fetch_funding(symbol: str) -> pd.DataFrame:
    """Full funding-rate history (every ~8h), paginated FORWARD from listing.

    Binance treats startTime<=0 as unset (returns only the most-recent page), so
    we page forward from a concrete early timestamp until we reach the present.
    """
    rows = []
    start = int(pd.Timestamp("2019-09-01", tz="UTC").timestamp() * 1000)
    now = int(time.time() * 1000)
    while start < now:
        batch = _get("/fapi/v1/fundingRate",
                     {"symbol": symbol, "startTime": start, "limit": 1000})
        if not batch:
            break
        rows.extend(batch)
        last = batch[-1]["fundingTime"]
        if last <= start:
            break
        start = last + 1
        if len(batch) < 1000:
            break
        time.sleep(0.25)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["fundingTime"], unit="ms")
    df["funding_rate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    df["mark_price"] = pd.to_numeric(df.get("markPrice"), errors="coerce")
    return df[["ts", "funding_rate", "mark_price"]].dropna(subset=["funding_rate"]) \
             .drop_duplicates("ts").sort_values("ts")


def ingest(symbols=SYMBOLS) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    report = {}
    for s in symbols:
        k = fetch_klines(s)
        f = fetch_funding(s)
        k.to_csv(OUT / f"{s}_1d.csv", index=False)
        f.to_csv(OUT / f"{s}_funding.csv", index=False)
        report[s] = {
            "klines": len(k), "klines_range": [str(k["date"].min().date()),
                                               str(k["date"].max().date())],
            "funding": len(f), "funding_range": [str(f["ts"].min().date()),
                                                 str(f["ts"].max().date())] if len(f) else None,
            "avg_daily_funding_bps": round(float(f["funding_rate"].mean()) * 1e4 * 3, 3)
            if len(f) else None,   # ~3 funding periods/day
        }
    return report


if __name__ == "__main__":
    import json
    print(json.dumps(ingest(), indent=2))
