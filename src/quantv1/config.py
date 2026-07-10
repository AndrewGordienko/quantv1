"""Central configuration and shared constants."""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
DB_PATH = DATA_DIR / "quantv1.duckdb"

DATA_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Data sources
# ---------------------------------------------------------------------------
# Senate/House Stock Watcher publish the "all transactions" aggregates parsed
# from the official EFD (Senate) and House Clerk disclosure portals, committed
# directly to GitHub. Served via raw.githubusercontent.com.
#
# NOTE (verified 2026-07): the Senate aggregate under timothycarambat is stale
# (transactions end ~2020) and has NO disclosure date. The House feed under
# TattooedHead is fresh (through 2026) and includes disclosure_date + amount_mid.
# House is therefore the primary, point-in-time-clean dataset; Senate is kept
# for historical breadth with an *estimated* filing date.
SENATE_WATCHER_URL = (
    "https://raw.githubusercontent.com/timothycarambat/"
    "senate-stock-watcher-data/master/aggregate/all_transactions.json"
)
HOUSE_WATCHER_URL = (
    "https://raw.githubusercontent.com/TattooedHead/"
    "house-stock-watcher-data/main/data/all_transactions.json"
)

# Median senate disclosure lag used to estimate senate filing dates (days).
SENATE_ESTIMATED_LAG_DAYS = 30

BENCHMARK_TICKER = "SPY"
SECTOR_ETFS = ["XLK", "XLE", "XLV", "XLF", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC"]

# ---------------------------------------------------------------------------
# Modeling constants
# ---------------------------------------------------------------------------
# Disclosure amounts are reported as ranges. Map the reported range string to
# (low, high) dollar bounds; midpoint (log-scaled) is used as the size proxy.
AMOUNT_RANGES: dict[str, tuple[float, float]] = {
    "$1,001 -": (1_001, 15_000),
    "$1,001 - $15,000": (1_001, 15_000),
    "$15,001 - $50,000": (15_001, 50_000),
    "$50,001 - $100,000": (50_001, 100_000),
    "$100,001 - $250,000": (100_001, 250_000),
    "$250,001 - $500,000": (250_001, 500_000),
    "$500,001 - $1,000,000": (500_001, 1_000_000),
    "$1,000,001 - $5,000,000": (1_000_001, 5_000_000),
    "$5,000,001 - $25,000,000": (5_000_001, 25_000_000),
    "$25,000,001 - $50,000,000": (25_000_001, 50_000_000),
    "$50,000,000 +": (50_000_000, 100_000_000),
    "Over $50,000,000": (50_000_000, 100_000_000),
}

# Event-study horizons in trading days.
HORIZONS = [5, 21, 63, 126]

# Label horizon for the supervised signal model (trading days after filing).
LABEL_HORIZON = 63

# Cluster window: members buying same ticker within this many calendar days.
CLUSTER_WINDOW_DAYS = 30

# Portfolio construction
LOOKBACK_DAYS = 90          # disclosures filed within this window are eligible
TOP_K = 20                  # max holdings
MAX_POSITION_WEIGHT = 0.08  # per-name cap
MAX_SECTOR_WEIGHT = 0.30    # per-sector cap
HOLD_HORIZON = 63           # trading-day exit horizon
COST_BPS = 10               # per-side transaction cost, basis points
