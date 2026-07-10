from __future__ import annotations

from datetime import datetime, timedelta
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from quantv1.db import connect
from quantv1.events import catalysts
from quantv1.v4.news_reaction import news_events


def _row(event_id, ticker, when, url, title, tickers=None):
    return (event_id, ticker, when, url, "wire",
            json.dumps({"title": title, "tickers": tickers or [ticker]}))


class CatalystTests(unittest.TestCase):
    def test_near_duplicate_requires_window_and_entity_overlap(self):
        at = datetime(2026, 1, 2, 10)
        base = "Tesla announces expanded battery partnership with Rivian for commercial vehicles"
        edit = "Tesla announces an expanded battery partnership with Rivian for new commercial vehicles"
        rows = [
            _row("a", "TSLA", at, "https://wire/a", base, ["TSLA", "RIVN"]),
            _row("b", "RIVN", at + timedelta(hours=4), "https://wire/b", edit,
                 ["TSLA", "RIVN"]),
            _row("c", "RIVN", at + timedelta(hours=20), "https://wire/c", edit,
                 ["TSLA", "RIVN"]),
            _row("d", "AAPL", at + timedelta(hours=1), "https://wire/d",
                 "Apple announces an expanded software partnership with Adobe for new services",
                 ["AAPL"]),
        ]
        revisions, groups = catalysts.cluster_rows(rows)
        grouped_ids = [{revisions[index].revision_id for index in group} for group in groups]
        ids = {revision.identity: revision.revision_id for revision in revisions}
        self.assertTrue(any({ids["url:https://wire/a"], ids["url:https://wire/b"]} <= group
                            for group in grouped_ids))
        self.assertFalse(any({ids["url:https://wire/a"], ids["url:https://wire/c"]} <= group
                             for group in grouped_ids))
        self.assertFalse(any({ids["url:https://wire/a"], ids["url:https://wire/d"]} <= group
                             for group in grouped_ids))

    def test_full_headline_avoids_first_ten_word_collision(self):
        at = datetime(2026, 1, 2, 10)
        prefix = "Markets update stocks rise as investors watch Federal Reserve policy"
        rows = [
            _row("a", "TSLA", at, "https://wire/a",
                 prefix + " after Tesla delivers record quarterly vehicles"),
            _row("b", "TSLA", at + timedelta(minutes=5), "https://wire/b",
                 prefix + " before Tesla faces a new supplier lawsuit"),
        ]
        _, groups = catalysts.cluster_rows(rows)
        self.assertEqual(len(groups), 2)

    def test_transitive_chain_cannot_exceed_total_cluster_span(self):
        at = datetime(2026, 1, 2, 0)
        headline = "Tesla announces expanded battery partnership with Rivian for vehicles"
        rows = [
            _row("a", "TSLA", at, "https://wire/a", headline),
            _row("b", "TSLA", at + timedelta(hours=8), "https://wire/b", headline),
            _row("c", "TSLA", at + timedelta(hours=16), "https://wire/c", headline),
        ]
        _, groups = catalysts.cluster_rows(rows)
        self.assertEqual(sorted(map(len, groups)), [1, 2])

    def test_build_keeps_late_ticker_gate_and_old_version(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "test.duckdb"
            with patch("quantv1.db.DB_PATH", db_path):
                con = connect()
                con.execute("""
                    INSERT INTO catalysts
                        (catalyst_id,cluster_version,earliest_public_time,headline,
                         n_article_revisions,n_assets,created_at,metadata)
                    VALUES ('old-catalyst','old-version','2025-01-01','old',1,1,
                            '2025-01-01','{}')
                """)
                at = datetime(2026, 1, 2, 10)
                title = "Tesla announces expanded battery partnership with Rivian for commercial vehicles"
                edit = "Tesla announces an expanded battery partnership with Rivian for commercial vehicles"
                events = [
                    ("e1", "TSLA", at, "https://wire/1", title, ["TSLA"]),
                    ("e2", "TSLA", at + timedelta(hours=4), "https://wire/2", edit,
                     ["TSLA", "RIVN"]),
                    ("e3", "RIVN", at + timedelta(hours=4), "https://wire/2", edit,
                     ["TSLA", "RIVN"]),
                ]
                for event_id, ticker, when, url, headline, tickers in events:
                    con.execute("""
                        INSERT INTO events
                            (event_id,layer,event_type,ticker,entity,direction,magnitude,
                             novelty,effective_date,source_time,source_url,payload)
                        VALUES (?,'N','news',?,'wire',0,0.5,1,?,?,?,?)
                    """, [event_id, ticker, when.date(), when, url,
                          json.dumps({"title": headline, "tickers": tickers})])
                con.close()
                stats = catalysts.build(verbose=False)
                self.assertEqual(stats["raw_ticker_event_rows"], 3)
                self.assertEqual(stats["raw_articles"], 2)
                self.assertEqual(stats["unique_article_revisions"], 2)
                self.assertEqual(stats["catalysts"], 1)
                self.assertEqual(stats["catalyst_ticker_observations"], 2)
                con = connect(read_only=True)
                self.assertEqual(con.execute(
                    "SELECT COUNT(*) FROM catalysts WHERE catalyst_id='old-catalyst'"
                ).fetchone()[0], 1)
                links = dict(con.execute("""
                    SELECT ticker, first_link_public_time FROM catalyst_assets
                    WHERE build_id=?
                """, [stats["build_id"]]).fetchall())
                self.assertEqual(links["TSLA"], at)
                self.assertEqual(links["RIVN"], at + timedelta(hours=4))
                as_of_11 = catalysts.assets_as_of(
                    con, at + timedelta(hours=1), build_id=stats["build_id"]
                )
                self.assertEqual(set(as_of_11["ticker"]), {"TSLA"})
                observed = news_events(
                    con, build_id=stats["build_id"]
                ).set_index("ticker")["public_time"].to_dict()
                self.assertEqual(observed["TSLA"].to_pydatetime(), at)
                self.assertEqual(observed["RIVN"].to_pydatetime(), at + timedelta(hours=4))
                con.close()
                rerun = catalysts.build(verbose=False)
                self.assertEqual(rerun["build_id"], stats["build_id"])
                con = connect(read_only=True)
                self.assertEqual(con.execute("""
                    SELECT max(n) FROM (
                        SELECT event_id,count(*) n FROM catalyst_events
                        WHERE build_id=? GROUP BY event_id
                    )
                """, [stats["build_id"]]).fetchone()[0], 1)
                con.close()


if __name__ == "__main__":
    unittest.main()
