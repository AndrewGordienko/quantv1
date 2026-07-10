from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from quantv1.research.actor_b2 import _prepare, permute_actor_within_role_event


class ActorB2Tests(unittest.TestCase):
    def test_actor_permutation_stays_within_role_and_event_type(self):
        n = 24
        frame = pd.DataFrame({
            "actor_event_id": [f"e{i}" for i in range(n)],
            "actor_id": [f"actor_{i % 4}" for i in range(n)],
            "ticker": ["TLT"] * n,
            "public_time": pd.date_range("2025-01-01", periods=n, tz="UTC"),
            "actor_event_role": ["speaker_author"] * 12 + ["direct_public_action"] * 12,
            "semantic_event_type": (["speech"] * 6 + ["remarks"] * 6) * 2,
            "stance": np.linspace(-1, 1, n), "magnitude": 0.5,
            "topic": "inflation", "regime": "normal", "sector": "rates",
            "pre_event_volatility": 0.01, "time_of_day_bucket": "afternoon",
            "actor_asset_channel": "monetary_policy",
            "target_return": np.linspace(-0.01, 0.01, n),
        })
        data = _prepare(frame)
        before = {
            key: sorted(group["actor_id"])
            for key, group in data.groupby(["actor_event_role", "semantic_event_type"])
        }
        permuted = permute_actor_within_role_event(data, np.random.default_rng(2))
        after = {
            key: sorted(group["actor_id"])
            for key, group in permuted.groupby(["actor_event_role", "semantic_event_type"])
        }
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
