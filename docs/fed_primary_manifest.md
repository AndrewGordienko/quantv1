# Fed primary-source manifest

The importer deliberately requires reviewed records. A page date is not an
event timestamp, and press-conference transcript paragraphs do not have reliable
intraday timing unless aligned to the official audio/video.

Each JSON/JSONL communication has this shape:

```json
{
  "sample": "B2_FED_SPEAKER_PANEL",
  "actor": {
    "actor_id": "speaker_stable_id",
    "name": "Full Name",
    "actor_type": "central_banker"
  },
  "institutional_role": {
    "organization": "Federal Reserve Bank or Board",
    "role": "Governor or President",
    "valid_from": "YYYY-MM-DD",
    "valid_to": null,
    "source": "https://official-fed-role-source"
  },
  "public_time": "2026-01-01T14:00:00-05:00",
  "timestamp_precision": "exact",
  "communication_type": "speech",
  "actor_event_role": "speaker_author",
  "title": "Official title",
  "source_url": "https://official-fed-transcript-source",
  "transcript": "Primary-source transcript text",
  "topics": ["inflation", "labor"],
  "asset_exposures": [
    {"ticker": "TLT", "channel": "monetary_policy", "confidence": 1.0},
    {"ticker": "XLF", "channel": "monetary_policy", "confidence": 0.8}
  ]
}
```

The B2 panel must contain multiple speakers across Chairs, Governors, regional
Reserve Bank presidents and other voting participants. B3 records use
`sample="B3_CHAIR_PRESS_CONFERENCE"`,
`communication_type="chair_press_conference"`, and add monotone exact-time
`segments` with `segment_role` equal to `prepared`, `question`, or `answer`.
Question segments may have no actor ID; answer/prepared segments identify the
Chair. Include official audio offsets when available.

The pilot rejects records without at least one Treasury-duration instrument
(`IEF` or `TLT`) and `XLF`. It does not treat SPY as a sufficient monetary-policy
outcome.
