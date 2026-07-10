"""Shared return / abnormal-return engine.

Everything quantitative (event study, skill scores, model labels, backtest)
computes returns through this one module so the conventions are identical:

* Entry is the first trading day on or after an "as of" date (the filing date).
  You cannot act on a disclosure before you have seen it, so returns are always
  measured from the filing date forward, never the transaction date.
* Horizons are in TRADING days, taken off SPY's calendar.
* Abnormal return = stock return - beta * market return over the same window.
  Beta defaults to 1.0 (market-adjusted return) — the standard, robust choice
  that avoids noisy per-name beta estimation; a CAPM beta is available opt-in.

Prices are pivoted into a wide date x ticker close matrix once and reused.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import BENCHMARK_TICKER
from ..db import connect


class PriceStore:
    def __init__(self, con=None):
        own = con is None
        con = con or connect(read_only=True)
        px = con.execute(
            "SELECT ticker, date, close, volume FROM prices ORDER BY date"
        ).df()
        if own:
            con.close()
        px["date"] = pd.to_datetime(px["date"])
        self.close = px.pivot_table(index="date", columns="ticker", values="close")
        dollar = px.assign(dv=px["close"] * px["volume"])
        self.dollar_vol = dollar.pivot_table(index="date", columns="ticker", values="dv")
        # SPY defines the trading calendar and the market benchmark.
        self.cal = self.close.index
        self.spy = self.close[BENCHMARK_TICKER] if BENCHMARK_TICKER in self.close else None
        self._ret = None  # lazily-built daily return matrix for beta

    # -- calendar helpers ---------------------------------------------------
    def _pos_on_or_after(self, date) -> int | None:
        date = pd.Timestamp(date)
        i = self.cal.searchsorted(date, side="left")
        return int(i) if i < len(self.cal) else None

    def has(self, ticker: str) -> bool:
        return ticker in self.close.columns

    def entry_price(self, ticker: str, date) -> tuple[pd.Timestamp, float] | None:
        """First valid close on/after `date` for `ticker`."""
        if ticker not in self.close.columns:
            return None
        i = self._pos_on_or_after(date)
        if i is None:
            return None
        col = self.close[ticker].to_numpy()
        idx = self.cal
        for j in range(i, len(idx)):
            v = col[j]
            if np.isfinite(v):
                return idx[j], float(v)
        return None

    # -- returns ------------------------------------------------------------
    def forward_return(self, ticker: str, date, horizon: int) -> float | None:
        """Return of `ticker` from entry (on/after date) to `horizon` trading days later."""
        if ticker not in self.close.columns:
            return None
        i = self._pos_on_or_after(date)
        if i is None:
            return None
        col = self.close[ticker].to_numpy()
        n = len(col)
        # locate entry
        e = next((j for j in range(i, n) if np.isfinite(col[j])), None)
        if e is None:
            return None
        x = e + horizon
        if x >= n:
            return None
        entry, exit_ = col[e], col[x]
        if not (np.isfinite(entry) and np.isfinite(exit_)) or entry <= 0:
            return None
        return float(exit_ / entry - 1.0)

    def market_return(self, date, horizon: int) -> float | None:
        return self.forward_return(BENCHMARK_TICKER, date, horizon)

    def abnormal_return(self, ticker: str, date, horizon: int,
                        beta: float = 1.0) -> float | None:
        r = self.forward_return(ticker, date, horizon)
        m = self.market_return(date, horizon)
        if r is None or m is None:
            return None
        return r - beta * m

    def beats_market(self, ticker: str, date, horizon: int) -> int | None:
        ar = self.abnormal_return(ticker, date, horizon, beta=1.0)
        return None if ar is None else int(ar > 0)

    # -- beta (opt-in) ------------------------------------------------------
    def _daily_returns(self) -> pd.DataFrame:
        if self._ret is None:
            self._ret = self.close.pct_change()
        return self._ret

    def beta(self, ticker: str, asof, window: int = 252) -> float:
        """Trailing CAPM beta vs SPY estimated over `window` days before asof."""
        if ticker not in self.close.columns or self.spy is None:
            return 1.0
        i = self._pos_on_or_after(asof)
        if i is None or i < 30:
            return 1.0
        ret = self._daily_returns()
        lo = max(0, i - window)
        s = ret[ticker].iloc[lo:i]
        m = ret[BENCHMARK_TICKER].iloc[lo:i]
        d = pd.concat([s, m], axis=1).dropna()
        if len(d) < 30 or d.iloc[:, 1].var() == 0:
            return 1.0
        b = float(np.cov(d.iloc[:, 0], d.iloc[:, 1])[0, 1] / d.iloc[:, 1].var())
        # clamp to a sane range to avoid blow-ups from thin data
        return float(np.clip(b, -1.0, 3.0))

    def median_dollar_volume(self, ticker: str, asof, window: int = 63) -> float | None:
        if ticker not in self.dollar_vol.columns:
            return None
        i = self._pos_on_or_after(asof)
        if i is None:
            return None
        lo = max(0, i - window)
        s = self.dollar_vol[ticker].iloc[lo:i].dropna()
        return float(s.median()) if len(s) else None
