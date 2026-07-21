"""Tests for the crypto trade-flow OFI aggregation (pure; no network)."""

import unittest

import pandas as pd

from quantv1.ingest.crypto_trades import ofi_bars


class TestOFI(unittest.TestCase):
    def _trades(self):
        base = 1_700_000_000_000  # ms, all within one minute
        return pd.DataFrame({
            "price": [100.0, 101.0, 100.5],
            "quantity": [10.0, 5.0, 5.0],
            "transact_time": [base, base + 1000, base + 2000],
            # aggressor BUY when is_buyer_maker is False
            "is_buyer_maker": [False, True, False],   # buy 10, sell 5, buy 5
        })

    def test_signed_imbalance(self):
        b = ofi_bars(self._trades(), freq="1min")
        self.assertEqual(len(b), 1)
        row = b.iloc[0]
        self.assertAlmostEqual(row["buy_vol"], 15.0)
        self.assertAlmostEqual(row["sell_vol"], 5.0)
        self.assertAlmostEqual(row["ofi"], (15.0 - 5.0) / 20.0)   # +0.5
        self.assertEqual(row["trades"], 3)
        self.assertAlmostEqual(row["close"], 100.5)

    def test_all_sells_gives_minus_one(self):
        t = self._trades()
        t["is_buyer_maker"] = [True, True, True]
        self.assertAlmostEqual(ofi_bars(t).iloc[0]["ofi"], -1.0)

    def test_vwap(self):
        b = ofi_bars(self._trades())
        vwap = (100*10 + 101*5 + 100.5*5) / 20
        self.assertAlmostEqual(b.iloc[0]["vwap"], vwap)


if __name__ == "__main__":
    unittest.main()
