"""Shared networking setup.

This machine sits behind a TLS-inspecting proxy whose CA is trusted by the OS
keychain but not by the uv-managed Python's bundled certs. `truststore` makes
Python use the operating-system trust store, so TLS stays VERIFIED (no disabling
certificate checks) while still trusting the corporate proxy. Import this module
before making HTTPS requests.
"""

from __future__ import annotations

import truststore

truststore.inject_into_ssl()

DEFAULT_UA = "quantv1 research (andrew.gordienko05@gmail.com)"
