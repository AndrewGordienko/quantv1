"""Crypto trade-flow ingest — free Binance USD-M aggTrades archives.

The real crypto day-trade engine is microstructure/flow absorption, whose core
input is trade-sign order-flow imbalance (OFI). Binance publishes free daily
aggTrades archives (data.binance.vision) that include the aggressor side
(``is_buyer_maker``) — so OFI is HISTORICALLY testable for free, which is what
the equity engine needed paid NBBO for. (Full L2 book replay still needs a live
collector; trade-level OFI does not.)

`ofi_bars` is a pure, testable aggregation: aggressor-buy volume minus
aggressor-sell volume, normalized. Bars are NOT order flow inferred from candle
volume — this is real signed trade flow (the Latent-Flow rejection was about the
former; this is the latter).
"""

from __future__ import annotations

import io
import zipfile
from datetime import date, timedelta
from pathlib import Path

try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass
import pandas as pd
import requests

from ..config import DATA_DIR

BASE = "https://data.binance.vision/data/futures/um/daily/aggTrades"
OUT = DATA_DIR / "crypto"
UA = {"User-Agent": "Mozilla/5.0 (quantv1 crypto research)"}
COLS = ["agg_trade_id", "price", "quantity", "first_trade_id", "last_trade_id",
        "transact_time", "is_buyer_maker"]


def download_aggtrades(symbol: str, day: str) -> pd.DataFrame:
    """Download + parse one day of aggTrades (aggressor side included)."""
    url = f"{BASE}/{symbol}/{symbol}-aggTrades-{day}.zip"
    r = requests.get(url, headers=UA, timeout=60)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    name = z.namelist()[0]
    head = z.open(name).readline().decode("utf-8", "replace").lower()
    has_header = "price" in head or "transact" in head
    df = pd.read_csv(z.open(name), header=0 if has_header else None,
                     names=None if has_header else COLS)
    df.columns = [c.strip().lower() for c in df.columns]
    ren = {"transact_time": "transact_time", "time": "transact_time",
           "qty": "quantity", "is_buyer_maker": "is_buyer_maker"}
    df = df.rename(columns={k: v for k, v in ren.items() if k in df.columns})
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["transact_time"] = pd.to_numeric(df["transact_time"], errors="coerce")
    df["is_buyer_maker"] = df["is_buyer_maker"].astype(str).str.lower().isin(["true", "1"])
    return df.dropna(subset=["price", "quantity", "transact_time"])


def ofi_bars(trades: pd.DataFrame, freq: str = "1min") -> pd.DataFrame:
    """Pure aggregation -> signed order-flow bars.

    aggressor is the BUYER when is_buyer_maker is False (buyer took the offer).
    OFI = (buy_vol - sell_vol) / (buy_vol + sell_vol) in [-1, 1].
    """
    df = trades.copy()
    df["ts"] = pd.to_datetime(df["transact_time"], unit="ms")
    aggressor_buy = ~df["is_buyer_maker"].astype(bool)
    df["buy_q"] = df["quantity"].where(aggressor_buy, 0.0)
    df["sell_q"] = df["quantity"].where(~aggressor_buy, 0.0)
    df["notional"] = df["price"] * df["quantity"]
    g = df.set_index("ts").groupby(pd.Grouper(freq=freq))
    bars = pd.DataFrame({
        "trades": g.size(),
        "volume": g["quantity"].sum(),
        "buy_vol": g["buy_q"].sum(),
        "sell_vol": g["sell_q"].sum(),
        "vwap": g["notional"].sum() / g["quantity"].sum().replace(0, pd.NA),
        "close": g["price"].last(),
    })
    tot = bars["buy_vol"] + bars["sell_vol"]
    bars["ofi"] = ((bars["buy_vol"] - bars["sell_vol"]) / tot.replace(0, pd.NA)).fillna(0.0)
    return bars[bars["trades"] > 0]


def ingest_sample(symbol: str = "BTCUSDT", days=("2026-07-19",), freq="1min") -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    frames = []
    for d in days:
        frames.append(ofi_bars(download_aggtrades(symbol, d), freq))
    bars = pd.concat(frames).sort_index()
    path = OUT / f"{symbol}_ofi_{freq}.csv"
    bars.to_csv(path)
    return {"symbol": symbol, "days": list(days), "freq": freq, "bars": len(bars),
            "mean_abs_ofi": round(float(bars["ofi"].abs().mean()), 4),
            "range": [str(bars.index.min()), str(bars.index.max())],
            "wrote": str(path)}


def _existing_days(path: Path) -> set:
    if not path.exists():
        return set()
    idx = pd.read_csv(path, index_col=0, parse_dates=True).index
    return set(pd.Series(idx).dt.date.astype(str))


def ingest_range(symbol: str = "BTCUSDT", days_back: int = 14,
                 freqs=("1min", "5min")) -> dict:
    """Resumable multi-day OFI accumulation. Downloads each new day ONCE (serving
    all freqs), skips days already stored, discards raw after computing bars."""
    OUT.mkdir(parents=True, exist_ok=True)
    end = date.today() - timedelta(days=1)          # today's archive may not exist yet
    want = sorted((end - timedelta(days=i)).isoformat() for i in range(days_back))
    paths = {f: OUT / f"{symbol}_ofi_{f}.csv" for f in freqs}
    have = {f: _existing_days(paths[f]) for f in freqs}
    added, missing = [], []
    for d in want:
        if all(d in have[f] for f in freqs):
            continue
        try:
            trades = download_aggtrades(symbol, d)
        except Exception:
            missing.append(d)
            continue
        for f in freqs:
            if d in have[f]:
                continue
            bars = ofi_bars(trades, f)
            existing = pd.read_csv(paths[f], index_col=0, parse_dates=True) if paths[f].exists() else None
            combined = pd.concat([x for x in (existing, bars) if x is not None]).sort_index()
            combined = combined[~combined.index.duplicated(keep="last")]
            combined.to_csv(paths[f])
        added.append(d)
        del trades
    return {"symbol": symbol, "added_days": added, "missing_days": missing,
            "coverage_days": {f: len(_existing_days(paths[f])) for f in freqs}}


if __name__ == "__main__":
    import json
    print(json.dumps(ingest_sample(), indent=2))
