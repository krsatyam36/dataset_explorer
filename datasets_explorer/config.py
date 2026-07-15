import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Ollama settings — runs fully local, no API key
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
# Best tool-using model that fits on 4 GB VRAM + 24 GB RAM (slow but smart).
# Falls back if not pulled — the agent will downgrade automatically.
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:14b")
OLLAMA_FALLBACK_MODELS = [
    "qwen2.5:14b",           # 14B, ~9GB — default, clean Ollama tool calls
    "mistral-small:latest",  # 24B, ~14GB, very strong tool use (CPU-slow)
    "qwen2.5:7b",            # 7B, ~5GB, fast fallback
    "qwen2.5-coder:latest",  # 7B coder variant, JSON-in-content tool calls
    "gpt-oss:20b",           # NOTE: harmony tool format conflicts with Ollama parser
    "llama3.2:latest",       # 3B, last-resort
]

# Global storage — same DB no matter which directory you run from
DATA_DIR = Path(os.environ.get("DATASET_SEARCH_DIR", str(Path.home() / ".dataset_search")))
DB_PATH = DATA_DIR / "results.db"
LOG_DIR = DATA_DIR / "logs"

# CSV scrape output location — fixed path inside the project tree, NOT under DATA_DIR.
# One CSV per run, filename ddmmyy-hhmmss_scrape.csv (local time at run start).
CSV_DIR = Path(os.environ.get(
    "DATASET_SEARCH_CSV_DIR",
    str(Path.home() / "datasets_explorer" / "logs" / "csv"),
))

# Defaults for the search agent
DEFAULT_HOURS = float(os.environ.get("DEFAULT_HOURS", "2.0"))
DEFAULT_DEPTH = int(os.environ.get("DEFAULT_DEPTH", "2"))

# Per-depth iteration caps. Depth 3 is effectively unlimited.
DEPTH_ITERATION_CAPS = {1: 80, 2: 200, 3: 2000}

# OSINT balancing — domains the agent over-uses by default.
# For every dataset stored from one of these, it must store
# MAINSTREAM_BALANCE_RATIO datasets from other domains before storing
# another mainstream one.
MAINSTREAM_DOMAINS = ("kaggle.com", "huggingface.co", "github.com")
MAINSTREAM_BALANCE_RATIO = 2

# Force advanced query operator usage. After this many consecutive broad
# (no site:/filetype:/inurl:) web_search calls, the next broad call is rejected.
MAX_CONSECUTIVE_BROAD_QUERIES = 2

# Optional Brave Search API key. If set, web_search falls back to Brave when
# DuckDuckGo errors or returns nothing. Get a free key at https://api.search.brave.com/.
BRAVE_API_KEY = os.environ.get("BRAVE_API_KEY", "").strip()

# Phase-A gate: research tools (arxiv_search, read_pdf) are blocked until either
# the agent has stored this many datasets, OR has issued this many distinct
# portal-targeted web_search calls. Forces portal-first behavior.
PHASE_A_MIN_STORES = 2
PHASE_A_MIN_PORTAL_SEARCHES = 4

# Hosts treated as "portals" for the Phase-A gate (presence of any in a query's
# `site:` filter counts as a portal search).
PORTAL_HOSTS = (
    "figshare.com", "zenodo.org", "dataverse.harvard.edu", "ieee-dataport.org",
    "earthdata.nasa.gov", "earthexplorer.usgs.gov", "scihub.copernicus.eu",
    "paperswithcode.com", "openaerialmap.org", "universe.roboflow.com",
    "data.gov", "catalog.data.gov", "registry.opendata.aws",
    "european-data.europa.eu", "data.gov.uk", "data.gc.ca",
)

# Federated search: number of parallel queries to suggest the agent runs concurrently.
FEDERATED_PARALLEL_QUERIES = int(os.environ.get("FEDERATED_PARALLEL_QUERIES", "1"))
