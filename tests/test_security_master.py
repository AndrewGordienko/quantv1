import unittest
from quantv1.ingest.security_master import extract_cover_mapping, build_intervals, coverage_audit, select_trade_class

class SecurityMasterTests(unittest.TestCase):
    def test_inline_xbrl_priority(self):
        doc = '<ix:nonNumeric name="dei:TradingSymbol">ABC</ix:nonNumeric><ix:nonNumeric name="dei:SecurityExchangeName">Nasdaq</ix:nonNumeric>'
        self.assertEqual(extract_cover_mapping(doc)["method"], "INLINE_XBRL")

    def test_multiple_listed_classes_are_retained(self):
        doc = ('<ix:nonNumeric contextRef="c" name="dei:TradingSymbol">AAA</ix:nonNumeric>'
               '<ix:nonNumeric contextRef="c" name="dei:TradingSymbol">AAA.P</ix:nonNumeric>')
        mapping = extract_cover_mapping(doc)
        self.assertEqual(mapping["listed_tickers"], ["AAA", "AAA.P"])

    def test_trade_rule_caps_once_at_issuer(self):
        rows = [{"cik": "1", "security_id": "CIK:1:CLASS:COMMONA", "instrument_class": "COMMON", "ticker": "AAA", "status": "ACTIVE"},
                {"cik": "1", "security_id": "CIK:1:CLASS:COMMONB", "instrument_class": "COMMON", "ticker": "AAB", "status": "ACTIVE"}]
        got = select_trade_class(rows, {"CIK:1:CLASS:COMMONA": 10, "CIK:1:CLASS:COMMONB": 20})
        self.assertEqual(got["1"]["ticker"], "AAB")
        self.assertTrue(got["1"]["issuer_exposure_cap"])
        self.assertEqual(select_trade_class(rows), {})

    def test_conflict_fails_closed(self):
        rows = [
            {"cik":"1", "accession_number":"a", "source_public_time":"2024-01-01T00:00:00Z", "mapping":{"ticker":"AAA","exchange":"NASDAQ","method":"INLINE_XBRL","confidence":"HIGH"}},
            {"cik":"1", "accession_number":"b", "source_public_time":"2024-01-01T00:00:00Z", "mapping":{"ticker":"BBB","exchange":"NASDAQ","method":"COVER_TEXT","confidence":"MEDIUM"}},
        ]
        _, conflicts = build_intervals(rows)
        self.assertTrue(conflicts)

    def test_coverage_gate_requires_delisting_and_rates(self):
        events = [{"atlas_event_id":"e", "cik":"1", "public_time":"2024-01-02T00:00:00Z", "event_family":"guidance"}]
        mappings = [{"cik":"0000000001", "ticker":"AAA", "exchange":"NASDAQ", "valid_from":"2024-01-01T00:00:00+00:00", "valid_to":"2025-01-01T00:00:00+00:00", "status":"DELISTED"}]
        report = coverage_audit(events, mappings, {"e": True})
        self.assertTrue(report["promotion_gate"])

if __name__ == '__main__': unittest.main()
