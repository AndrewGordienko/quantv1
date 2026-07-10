from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from quantv1.db import connect
from quantv1.events import actors


class ActorRegistryTests(unittest.TestCase):
    def test_temporal_role_aliases_and_context_only_extraction(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "test.duckdb"
            with patch("quantv1.db.DB_PATH", db_path):
                actors.register(verbose=False)
                con = connect()
                rows = [
                    ("g", "2024-06-01 12:00:00", "SEC Chair announces market rule"),
                    ("u", "2025-03-01 12:00:00", "SEC Chair discusses capital formation"),
                    ("a", "2025-05-01 12:00:00", "SEC Chair discusses digital assets"),
                    ("h", "2025-05-01 12:01:00", "Huang discusses artificial intelligence"),
                ]
                for event_id, timestamp, headline in rows:
                    con.execute("""
                        INSERT INTO events
                            (event_id,layer,event_type,ticker,entity,direction,magnitude,
                             novelty,effective_date,source_time,source_url,payload)
                        VALUES (?,'N','news','SPY','wire',0,0.5,1,
                                CAST(? AS TIMESTAMP)::DATE,CAST(? AS TIMESTAMP),
                                'https://example.test/article',?)
                    """, [event_id, timestamp, timestamp, json.dumps({"title": headline})])
                con.close()
                actors.extract_from_news(verbose=False)
                con = connect(read_only=True)
                extracted = con.execute("""
                    SELECT source_event_id, actor_id, actor_event_role,
                           primary_hypothesis_eligible
                    FROM actor_events WHERE extraction_version=? ORDER BY source_event_id
                """, [actors.EXTRACTION_VERSION]).fetchall()
                con.close()
                mapping = {source: actor_id for source, actor_id, _, _ in extracted}
                self.assertEqual(mapping["g"], "gary_gensler")
                self.assertEqual(mapping["u"], "mark_uyeda")
                self.assertEqual(mapping["a"], "paul_atkins")
                self.assertNotIn("h", mapping)  # surname-only Huang is deliberately unsafe
                self.assertTrue(all(role == "merely_mentioned" and eligible is False
                                    for _, _, role, eligible in extracted))

    def test_gensler_and_atkins_are_distinct_sourced_roles(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "test.duckdb"
            with patch("quantv1.db.DB_PATH", db_path):
                actors.register(verbose=False)
                con = connect(read_only=True)
                roles = con.execute("""
                    SELECT actor_id, valid_from, valid_to, source FROM actor_roles
                    WHERE actor_id IN ('gary_gensler','paul_atkins') ORDER BY actor_id
                """).fetchall()
                actor_columns = {row[0] for row in con.execute("DESCRIBE actors").fetchall()}
                con.close()
                self.assertEqual({row[0] for row in roles}, {"gary_gensler", "paul_atkins"})
                self.assertTrue(all(row[3].startswith("https://www.sec.gov/") for row in roles))
                self.assertNotIn("authority", actor_columns)


if __name__ == "__main__":
    unittest.main()
