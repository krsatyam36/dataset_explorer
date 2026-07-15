import csv
import json
import logging
import re
from collections import deque
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from .config import (
    MAINSTREAM_DOMAINS,
    MAINSTREAM_BALANCE_RATIO,
    MAX_CONSECUTIVE_BROAD_QUERIES,
    BRAVE_API_KEY,
    PHASE_A_MIN_STORES,
    PHASE_A_MIN_PORTAL_SEARCHES,
    PORTAL_HOSTS,
)
from .models import Dataset, DatasetSource
from .storage import Storage
from .utils import RateLimiter

logger = logging.getLogger("datasets_explorer")

# pypdf is verbose with malformed-PDF warnings on stderr; route them to our log instead.
_pypdf_log = logging.getLogger("pypdf")
_pypdf_log.setLevel(logging.ERROR)
_pypdf_log.propagate = False

# Ollama / OpenAI tool format: {"type": "function", "function": {"name", "description", "parameters"}}
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the internet using DuckDuckGo. Returns a list of results with title, url, and snippet. "
                "Use ADVANCED OPERATORS in nearly every query — site:, filetype:, inurl:, intitle:, and negative -site: filters. "
                "Examples of GOOD queries: "
                "'site:figshare.com aircraft satellite dataset', "
                "'site:catalog.data.gov filetype:csv aerial imagery', "
                "'site:edu inurl:dataset remote sensing aircraft', "
                "'inurl:dataverse military aircraft', "
                "'\"MAR20 dataset\" download', "
                "'aerial imagery dataset -site:kaggle.com -site:huggingface.co -site:github.com'. "
                "Avoid broad queries with no operators — they return surface-web SEO sludge."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query string"},
                    "max_results": {
                        "type": "integer",
                        "description": "Number of results to return (max 20), default 10",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_page",
            "description": (
                "Fetch and parse a webpage. Returns cleaned text content and links found on the page. "
                "Use this to read dataset descriptions, find download links, check license info, "
                "and discover related datasets. Avoid fetching the same URL twice."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to fetch"},
                    "extract_links": {
                        "type": "boolean",
                        "description": "Whether to extract links from the page (default true)",
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "store_dataset",
            "description": (
                "Store a confirmed dataset in the results database. Call this ONLY when you have "
                "verified it is a real, accessible dataset — not just a search result snippet. "
                "Assign relevance_score 0.0-1.0 based on how well it matches the user's query. "
                "Score 0.9+ = highly relevant (correct subject AND format). "
                "Score 0.6-0.8 = partial match. "
                "Score below 0.5 = tangentially related. "
                "Do NOT store duplicates — if you already stored a URL, skip it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Dataset name (the 'what is it')"},
                    "url": {"type": "string", "description": "Canonical landing/page URL where you found it"},
                    "download_url": {
                        "type": "string",
                        "description": "Direct dataset download/access URL if known and distinct from url. e.g. 'https://kaggle.com/.../download'",
                    },
                    "source": {
                        "type": "string",
                        "enum": [
                            "kaggle", "huggingface", "roboflow", "ieee_dataport",
                            "nasa_earthdata", "usgs", "copernicus", "github",
                            "zenodo", "papers_with_code", "open_aerial_map", "generic",
                        ],
                        "description": "Which platform hosts this dataset",
                    },
                    "formats": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Data formats: jpeg, tiff, geotiff, cog, npy, png, csv, hdf5, shapefile",
                    },
                    "description": {"type": "string"},
                    "license": {"type": "string", "description": "e.g. CC BY 4.0, MIT, Public Domain"},
                    "size_human": {"type": "string", "description": "e.g. '12.4 GB'"},
                    "num_samples": {"type": "integer", "description": "Number of images/samples if known"},
                    "date_range_start": {"type": "string", "description": "e.g. '2018'"},
                    "date_range_end": {"type": "string", "description": "e.g. '2024'"},
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tags: aircraft, military, SAR, EO, multispectral, annotated, YOLO, COCO, etc.",
                    },
                    "relevance_score": {
                        "type": "number",
                        "description": "How well this dataset matches the user's query (0.0 to 1.0)",
                    },
                    "relevance_reasoning": {
                        "type": "string",
                        "description": "Brief explanation of the relevance score",
                    },
                    "license_spdx": {
                        "type": "string",
                        "description": "SPDX license id if you saw it on the page (e.g. 'CC-BY-4.0', 'MIT', 'Apache-2.0', 'CC0-1.0', 'GPL-3.0', 'unknown'). Use 'unknown' rather than guessing.",
                    },
                    "license_commercial_ok": {
                        "type": "boolean",
                        "description": "True if the license clearly permits commercial use (CC-BY/MIT/Apache/CC0); False if explicitly non-commercial (CC-BY-NC); omit if unclear.",
                    },
                    "doi": {
                        "type": "string",
                        "description": "DOI if shown on the page, e.g. '10.5281/zenodo.7331974' (no 'https://doi.org/' prefix).",
                    },
                    "authors": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Author / dataset-creator names if listed on the page.",
                    },
                    "institution": {
                        "type": "string",
                        "description": "Hosting institution / lab / company (e.g. 'NASA', 'TU Munich', 'Airbus DS').",
                    },
                    "country": {
                        "type": "string",
                        "description": "ISO 3166-1 alpha-2 country code of origin if known (e.g. 'US', 'CN', 'DE'). Matters for export-control review.",
                    },
                },
                "required": ["name", "url", "relevance_score", "relevance_reasoning"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_page_js",
            "description": (
                "Fetch a page using a HEADLESS BROWSER (Playwright) — required for JS-rendered "
                "sites where regular fetch_page returns empty/skeletal HTML. Use for: "
                "Roboflow Universe, Mendeley Data, some university lab pages, single-page apps. "
                "Slower than fetch_page (~3-8s); only use when fetch_page returned blank or stub content."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to fetch with a real browser"},
                    "wait_ms": {"type": "integer", "description": "Extra wait after load (default 1500ms)"},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_pdf",
            "description": (
                "Download and extract text from an academic paper PDF (arXiv, MDPI, PMC, IEEE, .edu, etc.). "
                "Use this AFTER portal searches are exhausted — papers introduce datasets in their body and "
                "cite primary sources (e.g. MAR20, DOTA, FAIR1M, RarePlanes, xView, DIOR, VEDAI). "
                "Returns cleaned text plus any URLs found in the document. "
                "After reading, run a HIGHLY SPECIFIC web_search for each dataset name you extract "
                "(e.g. '\"MAR20\" download', '\"FAIR1M\" benchmark site:edu')."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Direct URL to a .pdf or paper landing page that links to a PDF"},
                    "max_pages": {
                        "type": "integer",
                        "description": "Limit pages to read (default 12). The first ~10 pages of a paper "
                                       "usually contain the dataset section.",
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_github_readme",
            "description": (
                "Fetch and parse a GitHub repository's README. Use for repos that look like dataset hosts, "
                "benchmark code, or curated 'awesome-X' lists. Returns README text plus all links it contains. "
                "Datasets are often hosted via Zenodo/Drive/OneDrive links inside READMEs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "repo_url": {
                        "type": "string",
                        "description": "GitHub repo URL, e.g. 'https://github.com/owner/repo'",
                    },
                },
                "required": ["repo_url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "arxiv_search",
            "description": (
                "Search arXiv directly via its API for academic papers. More reliable than DDG for academic content. "
                "Returns title, abstract, PDF URL, and authors. After you find relevant papers, "
                "call read_pdf on each to extract the datasets they introduce or benchmark on."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Topic/keywords (no site: operators needed)"},
                    "max_results": {"type": "integer", "description": "Number of papers to return (max 20), default 10"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "zenodo_search",
            "description": (
                "Search Zenodo's REST API directly for datasets. FAR more reliable than "
                "'web_search site:zenodo.org' because it returns structured metadata: "
                "title, description, DOI, license, creators, file formats. "
                "Use this as your PRIMARY zenodo discovery method — call web_search site:zenodo.org "
                "only as a fallback. After finding records, use the returned 'url' "
                "(https://zenodo.org/records/<id>) to store_dataset directly."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search keywords (no site: operators needed)"},
                    "max_results": {"type": "integer", "description": "Number of results (max 20), default 10"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mark_search_complete",
            "description": (
                "Call this when you have exhaustively searched all major sources and believe "
                "you have found all available relevant datasets. Provide a summary of sources checked."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Summary of search results and sources checked",
                    },
                    "total_found": {"type": "integer", "description": "Total datasets stored"},
                },
                "required": ["summary", "total_found"],
            },
        },
    },
]


class ToolExecutor:
    def __init__(
        self,
        storage: Storage,
        query_id: int,
        rate_limiter: RateLimiter,
        on_dataset_stored=None,
        seen_urls: Optional[set[str]] = None,
        csv_path: Optional[Path] = None,
        subject: str = "",
    ):
        self.storage = storage
        self.query_id = query_id
        self.rate_limiter = rate_limiter
        self.on_dataset_stored = on_dataset_stored
        self._http = httpx.Client(
            timeout=30.0,
            follow_redirects=True,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (compatible; datasets_explorer/1.0; "
                    "dataset discovery research tool)"
                )
            },
        )
        # URLs already fetched in THIS run (avoid double-fetching same URL)
        self._fetched_urls: set[str] = set()
        # URLs that already exist in the global DB from prior runs (avoid rediscovery)
        self.seen_urls: set[str] = set(seen_urls or [])
        # Dedup stats for the run summary
        self.skipped_fetch_already_seen = 0
        self.skipped_store_duplicate = 0
        # OSINT balancing: track stored mainstream vs alternative for the 1:N rule.
        self.mainstream_stored = 0
        self.alternative_stored = 0
        # Force advanced search operator usage.
        self._recent_query_was_broad: deque[bool] = deque(maxlen=MAX_CONSECUTIVE_BROAD_QUERIES + 1)
        # Recent web/arxiv queries (normalized, tagged by tool) for repeat detection.
        self._recent_queries: deque[str] = deque(maxlen=14)
        # URLs already read by read_pdf / read_github_readme this run.
        self._read_pdf_urls: set[str] = set()
        self._read_readme_urls: set[str] = set()
        # mark_search_complete cooldown — after a rejection, don't accept it for N iters.
        self._mark_complete_cooldown_until: int = 0
        self._iteration_seen: int = 0
        # Phase-A gate counters.
        self._stores_so_far: int = 0
        self._portal_search_count: int = 0
        # Subject the user asked for — used to decide whether perspective check applies.
        self.subject = subject or ""
        self._subject_wants_aerial = bool(re.search(
            r"\b(aerial|satellite|overhead|nadir|drone|uav|remote sensing|orthoimagery|"
            r"orthomosaic|earth observation|sentinel|landsat|maxar|planet|airbus)\b",
            self.subject, re.IGNORECASE,
        ))
        # Canonical-URL dedup so doi.org/10.5281/zenodo.X aliases zenodo.org/records/X.
        self._seen_canonical: set[str] = {self._canonical_url(u) for u in self.seen_urls}
        # Track which web_search query first surfaced each URL.
        self._url_to_query: dict[str, str] = {}
        # Per-run CSV of confirmed datasets.
        self.csv_path = csv_path
        self._csv_serial = 0
        if csv_path is not None:
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            new_file = not csv_path.exists()
            self._csv_fh = csv_path.open("a", newline="", encoding="utf-8")
            self._csv_writer = csv.writer(self._csv_fh)
            if new_file:
                self._csv_writer.writerow(
                    ["S.No", "Website name", "What does the website", "Link"]
                )
                self._csv_fh.flush()
        else:
            self._csv_fh = None
            self._csv_writer = None

    def execute(self, tool_name: str, tool_input: dict) -> Any:
        dispatch = {
            "web_search": self._web_search,
            "fetch_page": self._fetch_page,
            "fetch_page_js": self._fetch_page_js,
            "read_pdf": self._read_pdf,
            "read_github_readme": self._read_github_readme,
            "arxiv_search": self._arxiv_search,
            "zenodo_search": self._zenodo_search,
            "store_dataset": self._store_dataset,
            "mark_search_complete": self._mark_search_complete,
        }
        fn = dispatch.get(tool_name)
        if fn is None:
            return {"error": f"Unknown tool: {tool_name}"}
        return fn(tool_input)

    @staticmethod
    def _query_uses_operator(q: str) -> bool:
        ql = q.lower()
        return bool(re.search(r"\b(site:|filetype:|inurl:|intitle:|ext:)", ql))

    @staticmethod
    def _domain_of(url: str) -> str:
        try:
            host = (urlparse(url).hostname or "").lower()
            if host.startswith("www."):
                host = host[4:]
            return host
        except Exception:
            return ""

    @classmethod
    def _is_mainstream(cls, url: str) -> bool:
        host = cls._domain_of(url)
        return any(host == d or host.endswith("." + d) for d in MAINSTREAM_DOMAINS)

    # Aggregator/scraper sites that re-host other people's datasets without
    # being the authoritative source. Storing these is almost always noise.
    AGGREGATOR_DOMAINS = (
        "gts.ai", "innovatiana.com", "datasetninja.com",
        "wandb.ai", "cnas.org", "simuletic.com", "thegrenze.com",
        "libguides.utdallas.edu", "researchgate.net", "academia.edu",
        "semanticscholar.org",  # paper search hub, not a dataset host
    )

    # URL path fragments that indicate "this is a paper/report/blog about a
    # dataset, not the dataset itself". Stored URLs containing any of these
    # in the path are rejected. NOTE: '/articles/' is intentionally excluded
    # — figshare uses /articles/dataset/ for real datasets.
    _NON_DATASET_PATH_TOKENS = (
        "/reports/", "/publications/", "/publication/", "/blog/", "/news/",
        "/posts/", "/wp-content/", "/papers/", "/library/",
        "/pdf/",  # arxiv-style /pdf/<id> paper URLs
    )

    # URL paths that indicate a site root / listing page rather than a real dataset.
    _HOMEPAGE_PATHS = {
        "", "/", "/datasets", "/datasets/", "/dataset", "/dataset/",
        "/data", "/data/", "/search", "/search/", "/browse", "/browse/",
    }

    # Synthetic-data red-flag tokens. Word-boundary regex applied to name+desc+tags.
    _SYNTHETIC_RE = re.compile(
        r"\b(synthetic|simulated|simulation[- ]based|cgi|game[- ]engine|"
        r"airsim|unreal\s*engine|unity\s*engine|gan[- ]generated|"
        r"procedurally\s*generated|rendered)\b",
        re.IGNORECASE,
    )
    # Same shape, applied to OUTGOING queries so we don't even ask about synthetic data.
    _SYNTHETIC_QUERY_RE = re.compile(
        r"\b(synthetic|simulated|simulation|cgi|airsim|unreal|unity|"
        r"gan[- ]generated|rendered|procedurally\s*generated)\b",
        re.IGNORECASE,
    )

    # Ground-perspective red-flags — only fire when the user's subject mentions
    # aerial/satellite/overhead/UAV/drone/remote-sensing/etc.
    _GROUND_PERSPECTIVE_RE = re.compile(
        r"\b(wikimedia\s+commons|google\s+image\s+search|spotter\s+photos?|"
        r"plane[- ]?spott(?:ing|er)|planespotter|on[- ]tarmac|airshow|"
        r"side[- ]view|side\s+view|ground[- ]level|ground\s+level|"
        r"ground[- ]based|ground\s+based|from\s+below|from\s+the\s+ground|"
        r"frontal\s+view|cockpit\s+view|museum\s+aircraft|static\s+display)\b",
        re.IGNORECASE,
    )
    # Required positive phrases when the user wants aerial — must appear in name/desc/tags.
    _AERIAL_GREEN_RE = re.compile(
        r"\b(aerial|satellite|overhead|nadir|top[- ]down|orthoimagery|orthomosaic|"
        r"orthophoto|drone\s+imagery|uav\s+imagery|remote\s+sensing|earth\s+observation|"
        r"geotiff|cog\b|tif\s+tile|sentinel|landsat|maxar|planet\s+labs?|airbus|"
        r"copernicus|spaceborne|airborne|bird['’]?s[- ]eye)\b",
        re.IGNORECASE,
    )

    @classmethod
    def _canonical_url(cls, url: str) -> str:
        """Normalize URLs so common aliases collapse to one key for dedup.

        Examples that should collapse:
          - https://doi.org/10.5281/zenodo.7331974   ->  zenodo.org/records/7331974
          - https://www.zenodo.org/records/7331974/  ->  zenodo.org/records/7331974
          - https://Figshare.com/articles/dataset/X  ->  figshare.com/articles/dataset/x
        """
        if not url:
            return ""
        try:
            p = urlparse(url.strip())
            host = (p.hostname or "").lower()
            if host.startswith("www."):
                host = host[4:]
            path = (p.path or "").rstrip("/").lower()
            # DOI -> Zenodo
            if host == "doi.org":
                m = re.match(r"/10\.5281/zenodo\.(\d+)", path)
                if m:
                    return f"zenodo.org/records/{m.group(1)}"
            return f"{host}{path}"
        except Exception:
            return url.lower().rstrip("/")

    @classmethod
    def _is_homepage_url(cls, url: str) -> bool:
        try:
            path = (urlparse(url).path or "").rstrip("/").lower() + ("/" if url.endswith("/") else "")
        except Exception:
            return False
        # Treat empty / single-slash / common listing roots as homepage.
        plain = path.rstrip("/")
        return plain in {p.rstrip("/") for p in cls._HOMEPAGE_PATHS}

    @classmethod
    def _is_aggregator(cls, url: str) -> bool:
        host = cls._domain_of(url)
        return any(host == d or host.endswith("." + d) for d in cls.AGGREGATOR_DOMAINS)

    @classmethod
    def _looks_like_paper_or_report(cls, url: str) -> bool:
        """True if the URL is a paper PDF / report / blog post — NOT a dataset host."""
        try:
            p = urlparse(url)
            path = (p.path or "").lower()
        except Exception:
            return False
        if path.endswith(".pdf"):
            return True
        # Grenze uses servej.php?fn=...pdf — query string holds the file.
        if "servej.php" in path or path.endswith(".php"):
            return True
        for token in cls._NON_DATASET_PATH_TOKENS:
            if token in path:
                return True
        return False

    @classmethod
    def _looks_synthetic(cls, *texts: str) -> bool:
        blob = " ".join(t for t in texts if t)
        return bool(cls._SYNTHETIC_RE.search(blob))

    def _url_reachable(self, url: str) -> bool:
        """Cheap liveness check. Accepts 2xx/3xx and gated-but-real (401/403/405).

        - If we already fetched it successfully this run, skip the network call.
        - HEAD first; on 405 / connection oddities, retry with a small GET.
        """
        if not url:
            return False
        if url in self._fetched_urls:
            return True
        try:
            r = self._http.head(url, timeout=10.0)
            if r.status_code < 400 or r.status_code in (401, 403):
                return True
            if r.status_code in (405, 501):
                # Server doesn't allow HEAD — retry with a tiny GET.
                r = self._http.get(url, timeout=10.0, headers={"Range": "bytes=0-0"})
                return r.status_code < 400 or r.status_code in (401, 403, 416)
            return False
        except Exception as e:
            logger.info(f"[url_reachable] {url} failed: {e}")
            return False

    def _web_search(self, inp: dict) -> list[dict] | dict:
        query = (inp or {}).get("query")
        if not query or not isinstance(query, str):
            return {"error": "Missing required argument 'query' (string). Pass {\"query\": \"...\", \"max_results\": 10}."}

        # Reject queries that ask for synthetic / simulated data — we never store it,
        # so don't waste DDG quota looking for it.
        if self._SYNTHETIC_QUERY_RE.search(query):
            logger.info(f"[web_search] REJECTED synthetic-themed query: {query!r}")
            return {
                "error": (
                    "REJECTED: This query targets SYNTHETIC / SIMULATED / CGI / AirSim-style data. "
                    "We only collect REAL-WORLD imagery. Drop the synthetic/simulated/CGI/Unreal/Unity "
                    "keywords and re-issue with terms like 'real', 'satellite', 'aerial', 'annotated'."
                ),
                "rejected": True,
            }
        # DDG (and Brave) do NOT honor `site:X OR site:Y OR site:Z` syntax — this
        # silently returns junk. Force the agent to issue them as separate queries.
        if re.search(r"\bsite:\S+\s+OR\s+site:", query, re.IGNORECASE) or \
           re.search(r"\bsite:\([^)]*\|", query):
            logger.info(f"[web_search] REJECTED malformed OR-site query: {query!r}")
            return {
                "error": (
                    "REJECTED: DuckDuckGo does not honor 'site:X OR site:Y' across site filters. "
                    "Issue them as SEPARATE queries — one site: per call."
                ),
                "rejected": True,
            }

        # Reject exact-repeat queries within the last N search calls
        # (covers both web_search and arxiv_search via shared deque).
        norm_q = "web:" + " ".join(query.lower().split())
        if norm_q in self._recent_queries:
            logger.info(f"[web_search] REJECTED repeat query: {query!r}")
            return {
                "error": (
                    "REJECTED: You already issued this exact search recently. "
                    "Vary the query — try synonyms (aircraft → airplane/jet/fighter/bomber/UAV/drone), "
                    "swap the source (figshare ↔ zenodo ↔ dataverse ↔ catalog.data.gov ↔ arxiv ↔ ahmia.fi), "
                    "or chase a specific dataset name in quotes (e.g. '\"MAR20 dataset\" download'). "
                    "Do not reissue an identical query."
                ),
                "rejected": True,
            }
        self._recent_queries.append(norm_q)

        # Forced query operator rotation: block runs of broad queries.
        is_broad = not self._query_uses_operator(query)
        recent = list(self._recent_query_was_broad)
        if is_broad and len(recent) >= MAX_CONSECUTIVE_BROAD_QUERIES and all(recent[-MAX_CONSECUTIVE_BROAD_QUERIES:]):
            logger.info(f"[web_search] REJECTED broad query (no operator): {query!r}")
            return {
                "error": (
                    "REJECTED: Broad searches yield low-quality results. "
                    "Use advanced dorks like 'site:edu inurl:dataset', 'filetype:csv site:gov', "
                    "'site:figshare.com <topic>', or append negative filters "
                    "'-site:kaggle.com -site:huggingface.co -site:github.com' to physically block "
                    "mainstream surface-web hits. Reissue this search with a site:/filetype:/inurl: operator."
                ),
                "rejected": True,
            }
        self._recent_query_was_broad.append(is_broad)

        max_results = min((inp or {}).get("max_results", 10), 20)
        logger.info(f"[web_search] {query!r} max={max_results}")

        # Try DDG first.
        ddg_results: list[dict] = []
        ddg_error: str = ""
        try:
            from ddgs import DDGS
            self.rate_limiter.wait("ddg")
            with DDGS() as ddgs:
                ddg_results = [
                    {"title": r["title"], "url": r["href"], "snippet": r["body"]}
                    for r in ddgs.text(query, max_results=max_results)
                ]
        except Exception as e:
            ddg_error = str(e)
            logger.warning(f"[web_search] DDG error: {e}")

        # Fall back to Brave if DDG failed or returned nothing — only if a key is set.
        if not ddg_results and BRAVE_API_KEY:
            try:
                self.rate_limiter.wait("brave")
                logger.info(f"[web_search] falling back to Brave for {query!r}")
                r = self._http.get(
                    "https://api.search.brave.com/res/v1/web/search",
                    params={"q": query, "count": max_results},
                    headers={"Accept": "application/json", "X-Subscription-Token": BRAVE_API_KEY},
                    timeout=15.0,
                )
                if r.status_code == 200:
                    data = r.json()
                    web = (data.get("web") or {}).get("results") or []
                    ddg_results = [
                        {"title": w.get("title", ""), "url": w.get("url", ""), "snippet": w.get("description", "")}
                        for w in web
                    ]
                else:
                    logger.warning(f"[web_search] Brave HTTP {r.status_code}")
            except Exception as e:
                logger.warning(f"[web_search] Brave error: {e}")

        if not ddg_results:
            return [{"error": ddg_error or "Both DDG and Brave returned no results."}]

        # Remember the discovery query for each URL so we can write it to the CSV later.
        for r_ in ddg_results:
            href = r_.get("url")
            if href and href not in self._url_to_query:
                self._url_to_query[href] = query
        # Phase-A bookkeeping: count this as a "portal search" if its site: filter
        # targets a known portal host.
        ql = query.lower()
        if any(("site:" + h) in ql or ("site:www." + h) in ql for h in PORTAL_HOSTS):
            self._portal_search_count += 1
        return ddg_results

    def _fetch_page(self, inp: dict) -> dict:
        url = (inp or {}).get("url")
        if not url or not isinstance(url, str):
            return {"error": "Missing required argument 'url' (string). Pass {\"url\": \"https://...\"}.", "links": [], "status_code": 0}
        if url in self.seen_urls:
            self.skipped_fetch_already_seen += 1
            logger.info(f"[fetch_page] SKIP — URL already in global DB: {url}")
            return {
                "skipped": True,
                "reason": "already_in_global_db",
                "message": (
                    f"This URL is ALREADY in the global database from a previous search. "
                    f"Do NOT fetch or store it. Pick a DIFFERENT URL — find datasets you have not seen before."
                ),
                "url": url,
                "links": [],
                "status_code": 0,
            }
        if url in self._fetched_urls:
            return {"text": "[already fetched in this run — skip this URL]", "links": [], "status_code": 0}
        self._fetched_urls.add(url)
        try:
            self.rate_limiter.wait(url)
            logger.info(f"[fetch_page] {url}")
            resp = self._http.get(url)
            soup = BeautifulSoup(resp.text, "lxml")
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)[:8000]
            links = []
            if inp.get("extract_links", True):
                base = f"{resp.url.scheme}://{resp.url.host}"
                for a in soup.find_all("a", href=True)[:60]:
                    href = a["href"]
                    if href.startswith("http"):
                        links.append(href)
                    elif href.startswith("/"):
                        links.append(base + href)
                links = list(dict.fromkeys(links))[:50]
            return {"text": text, "links": links, "status_code": resp.status_code}
        except Exception as e:
            logger.warning(f"[fetch_page] Error fetching {url}: {e}")
            return {"error": str(e), "links": [], "status_code": 0}

    def _fetch_page_js(self, inp: dict) -> dict:
        """JS-rendering fetch via Playwright (lazy-imported so the dependency is optional).

        If Playwright is not installed we transparently fall back to fetch_page so
        the agent doesn't waste an iteration on a re-issue.
        """
        url = (inp or {}).get("url")
        if not url or not isinstance(url, str):
            return {"error": "Missing required argument 'url' (string).", "links": [], "status_code": 0}
        try:
            from playwright.sync_api import sync_playwright  # type: ignore
        except ImportError:
            logger.info(f"[fetch_page_js] Playwright unavailable — falling back to fetch_page for {url}")
            # Don't pre-add to _fetched_urls; let _fetch_page do its own dedup.
            res = self._fetch_page({"url": url, "extract_links": True})
            if isinstance(res, dict):
                res = dict(res)
                res["fallback"] = "fetch_page (playwright not installed)"
            return res
        if url in self._fetched_urls:
            return {"text": "[already fetched in this run — skip]", "links": [], "status_code": 0}
        self._fetched_urls.add(url)
        wait_ms = int((inp or {}).get("wait_ms", 1500) or 1500)
        wait_ms = max(0, min(wait_ms, 10000))
        try:
            self.rate_limiter.wait(url)
            logger.info(f"[fetch_page_js] {url}")
            with sync_playwright() as pw:
                browser = pw.chromium.launch(headless=True)
                page = browser.new_page(user_agent="Mozilla/5.0 (compatible; datasets_explorer/1.0)")
                page.goto(url, wait_until="networkidle", timeout=30000)
                if wait_ms:
                    page.wait_for_timeout(wait_ms)
                html = page.content()
                final_url = page.url
                browser.close()
            soup = BeautifulSoup(html, "lxml")
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)[:8000]
            base = "/".join(final_url.split("/")[:3])
            links = []
            for a in soup.find_all("a", href=True)[:80]:
                href = a["href"]
                if href.startswith("http"):
                    links.append(href)
                elif href.startswith("/"):
                    links.append(base + href)
            links = list(dict.fromkeys(links))[:60]
            return {"text": text, "links": links, "status_code": 200, "rendered_url": final_url}
        except Exception as e:
            logger.warning(f"[fetch_page_js] Error: {e}")
            return {"error": str(e), "links": [], "status_code": 0}

    # ---- Research-phase tools (PDF, GitHub README, arXiv) ------------------

    @staticmethod
    def _resolve_pdf_url(url: str, html: str = "") -> str:
        """Best-effort: turn a paper landing-page URL into a direct PDF URL."""
        # arXiv abstract page → PDF
        m = re.match(r"https?://arxiv\.org/abs/([\w.\-]+)", url, re.IGNORECASE)
        if m:
            return f"https://arxiv.org/pdf/{m.group(1)}.pdf"
        # MDPI articles have a /pdf variant
        m = re.match(r"https?://www\.mdpi\.com/(\d+-\d+/\d+/\d+/\d+)/?$", url)
        if m:
            return f"https://www.mdpi.com/{m.group(1)}/pdf"
        # If we have HTML, look for an explicit PDF link.
        if html:
            try:
                soup = BeautifulSoup(html, "lxml")
                # citation_pdf_url meta tag is the academic-paper standard
                meta = soup.find("meta", attrs={"name": "citation_pdf_url"})
                if meta and meta.get("content"):
                    return meta["content"]
                a = soup.find("a", href=re.compile(r"\.pdf($|\?)", re.IGNORECASE))
                if a and a.get("href"):
                    href = a["href"]
                    if href.startswith("http"):
                        return href
            except Exception:
                pass
        return url  # fall through; caller will try as-is

    def _read_pdf(self, inp: dict) -> dict:
        gate = self._phase_a_gate_block("read_pdf")
        if gate:
            return gate
        url = (inp or {}).get("url")
        max_pages = int((inp or {}).get("max_pages", 12) or 12)
        max_pages = max(1, min(max_pages, 50))
        if not url or not isinstance(url, str):
            return {"error": "Missing required argument 'url' (string)."}

        # Per-run dedup: refuse to re-extract the same PDF.
        if url in self._read_pdf_urls:
            return {
                "error": (
                    "Already read this PDF earlier in the run. Move on — read a DIFFERENT paper "
                    "or follow up on a dataset name you previously extracted."
                ),
                "rejected": True,
                "links": [],
            }

        # Refuse URLs that obviously won't yield a paper PDF.
        host = self._domain_of(url)
        if host in ("figshare.com",) or host.endswith(".figshare.com"):
            self._read_pdf_urls.add(url)
            return {
                "error": (
                    "figshare URLs are dataset landing pages, not papers. Use fetch_page on this URL instead."
                ),
                "rejected": True,
                "links": [],
            }
        if host in ("sciencedirect.com", "www.sciencedirect.com"):
            self._read_pdf_urls.add(url)
            return {
                "error": "ScienceDirect papers are login-walled — skip and try arXiv / MDPI / openaccess.thecvf.com.",
                "rejected": True,
                "links": [],
            }
        if host in ("ieeexplore.ieee.org", "ieee.org") or host.endswith(".ieee.org"):
            self._read_pdf_urls.add(url)
            return {
                "error": (
                    "IEEE Xplore papers are login-walled — skip read_pdf on ieee.org. "
                    "Use the dataset's primary host (Zenodo / figshare / GitHub repo) instead, "
                    "or arXiv preprints of the same paper."
                ),
                "rejected": True,
                "links": [],
            }
        if url.lower().endswith((".zip", ".tar", ".tar.gz", ".tgz", ".rar", ".7z")):
            self._read_pdf_urls.add(url)
            return {
                "error": (
                    f"'{url}' is a data archive, not a PDF. Don't try to read_pdf on archives — "
                    f"download via the dataset host or just store_dataset using the landing page URL."
                ),
                "rejected": True,
                "links": [],
            }
        self._read_pdf_urls.add(url)

        try:
            from pypdf import PdfReader
        except ImportError:
            return {"error": "pypdf not installed. Install with: pip install pypdf"}
        try:
            self.rate_limiter.wait(url)
            target = self._resolve_pdf_url(url)
            logger.info(f"[read_pdf] {target}")
            resp = self._http.get(target, timeout=45.0)
            ctype = (resp.headers.get("content-type") or "").lower()
            # Landing page → resolve again with the actual HTML
            if "pdf" not in ctype and "html" in ctype:
                target = self._resolve_pdf_url(url, resp.text)
                if target == url:
                    return {
                        "error": "Could not locate a PDF link on this landing page. "
                                 "Try fetch_page first to read the HTML, or pass the direct PDF URL.",
                        "links": [],
                    }
                resp = self._http.get(target, timeout=45.0)
                ctype = (resp.headers.get("content-type") or "").lower()
            if "pdf" not in ctype and not resp.content[:5].startswith(b"%PDF-"):
                return {"error": f"URL did not return a PDF (content-type={ctype})", "links": []}

            from io import BytesIO
            reader = PdfReader(BytesIO(resp.content))
            total = len(reader.pages)
            pages_read = min(total, max_pages)
            text_chunks: list[str] = []
            for i in range(pages_read):
                try:
                    text_chunks.append(reader.pages[i].extract_text() or "")
                except Exception as page_err:
                    text_chunks.append(f"[page {i+1} extraction failed: {page_err}]")
            text = "\n".join(text_chunks)
            text = re.sub(r"[ \t]+", " ", text)
            text = re.sub(r"\n{3,}", "\n\n", text)
            text = text[:15000]

            # Surface URLs that appear in the text — this is how you find dataset hosts.
            urls = re.findall(r"https?://[\w\-./%~?#=&+,:@]+", text)
            urls = [u.rstrip(".,);]") for u in urls]
            urls = list(dict.fromkeys(urls))[:60]

            return {
                "text": text,
                "links": urls,
                "pdf_url": target,
                "pages_total": total,
                "pages_read": pages_read,
            }
        except Exception as e:
            logger.warning(f"[read_pdf] Error: {e}")
            return {"error": str(e), "links": []}

    def _read_github_readme(self, inp: dict) -> dict:
        repo_url = (inp or {}).get("repo_url") or (inp or {}).get("url")
        if not repo_url or not isinstance(repo_url, str):
            return {"error": "Missing required argument 'repo_url'."}
        m = re.match(r"https?://github\.com/([^/]+)/([^/?#]+)", repo_url.strip())
        if not m:
            return {"error": f"Not a GitHub repo URL: {repo_url}"}
        owner, repo = m.group(1), m.group(2).removesuffix(".git")
        repo_key = f"{owner}/{repo}"
        if repo_key in self._read_readme_urls:
            return {
                "error": (
                    f"Already read README for {repo_key}. Don't re-read it — instead, run a "
                    f"web_search for one of the SPECIFIC dataset names you saw in it, or chase a link "
                    f"from it that you haven't fetched yet."
                ),
                "rejected": True,
                "links": [],
            }
        self._read_readme_urls.add(repo_key)
        # Try the raw README on common branches/filenames; first hit wins.
        candidates = [
            f"https://raw.githubusercontent.com/{owner}/{repo}/main/README.md",
            f"https://raw.githubusercontent.com/{owner}/{repo}/master/README.md",
            f"https://raw.githubusercontent.com/{owner}/{repo}/main/Readme.md",
            f"https://raw.githubusercontent.com/{owner}/{repo}/main/README.rst",
            f"https://raw.githubusercontent.com/{owner}/{repo}/master/README.rst",
        ]
        text = ""
        used = ""
        for c in candidates:
            try:
                self.rate_limiter.wait(c)
                r = self._http.get(c, timeout=15.0)
                if r.status_code == 200 and r.text.strip():
                    text = r.text
                    used = c
                    break
            except Exception:
                continue
        if not text:
            return {"error": f"No README found for {owner}/{repo}", "links": []}
        # Extract URLs and Markdown links so the agent can chase data hosts (Zenodo, Drive, OneDrive…)
        urls = re.findall(r"https?://[\w\-./%~?#=&+,:@]+", text)
        urls = [u.rstrip(".,);]") for u in urls]
        urls = list(dict.fromkeys(urls))[:80]
        return {
            "text": text[:15000],
            "links": urls,
            "readme_url": used,
            "repo": f"{owner}/{repo}",
            "next_step_hint": (
                "READMEs of dataset/awesome-list repos commonly name 5-30 datasets. "
                "From the text above, EXTRACT each dataset name and run a separate web_search "
                "for each one in quotes (e.g. '\"FAIR1M\" download'). Don't read this README again — "
                "chase the names you found."
            ),
        }

    def _phase_a_gate_block(self, tool_label: str) -> Optional[dict]:
        """Returns a rejection dict if research-phase tools are blocked by the
        Phase-A gate, or None if the gate is open."""
        if (self._stores_so_far >= PHASE_A_MIN_STORES or
                self._portal_search_count >= PHASE_A_MIN_PORTAL_SEARCHES):
            return None
        return {
            "error": (
                f"REJECTED: {tool_label} is blocked until Phase A (portal sweep) is satisfied. "
                f"Status: {self._stores_so_far} stores (need >= {PHASE_A_MIN_STORES}), "
                f"{self._portal_search_count} portal searches (need >= {PHASE_A_MIN_PORTAL_SEARCHES}). "
                f"Do portal searches FIRST: web_search with site:figshare.com / site:zenodo.org / "
                f"site:dataverse.harvard.edu / site:catalog.data.gov / site:registry.opendata.aws / "
                f"site:european-data.europa.eu / site:ieee-dataport.org / site:openaerialmap.org / "
                f"site:universe.roboflow.com / site:paperswithcode.com — one site: per call. "
                f"Drill into promising hits with fetch_page and store_dataset."
            ),
            "rejected": True,
            "links": [],
        }

    def _arxiv_search(self, inp: dict) -> list[dict] | dict:
        gate = self._phase_a_gate_block("arxiv_search")
        if gate:
            return gate
        query = (inp or {}).get("query")
        if not query or not isinstance(query, str):
            return {"error": "Missing required argument 'query'."}
        if self._SYNTHETIC_QUERY_RE.search(query):
            logger.info(f"[arxiv_search] REJECTED synthetic-themed query: {query!r}")
            return {
                "error": (
                    "REJECTED: This arxiv_search asks for SYNTHETIC / SIMULATED data. "
                    "We never store synthetic. Search for papers about real-world annotated imagery instead."
                ),
                "rejected": True,
            }
        norm_q = "arxiv:" + " ".join(query.lower().split())
        if norm_q in self._recent_queries:
            logger.info(f"[arxiv_search] REJECTED repeat query: {query!r}")
            return {
                "error": (
                    "REJECTED: You already ran this exact arxiv_search. Vary the keywords "
                    "(synonyms, narrower phrasing) or move on to read_pdf on papers you already found."
                ),
                "rejected": True,
            }
        self._recent_queries.append(norm_q)
        max_results = min(int((inp or {}).get("max_results", 10) or 10), 20)
        try:
            import urllib.parse as up
            self.rate_limiter.wait("arxiv.org")
            params = up.urlencode({
                "search_query": f"all:{query}",
                "start": 0,
                "max_results": max_results,
                "sortBy": "relevance",
                "sortOrder": "descending",
            })
            url = f"http://export.arxiv.org/api/query?{params}"
            logger.info(f"[arxiv_search] {query!r} max={max_results}")
            r = self._http.get(url, timeout=20.0)
            soup = BeautifulSoup(r.text, "lxml-xml") if r.text else None
            if soup is None:
                return {"error": "arXiv returned empty response", "results": []}
            entries = soup.find_all("entry")
            results = []
            for e in entries:
                arxiv_id = (e.id.text if e.id else "").split("/abs/")[-1].strip()
                pdf = ""
                for link in e.find_all("link"):
                    if link.get("title") == "pdf":
                        pdf = link.get("href", "")
                        break
                if not pdf and arxiv_id:
                    pdf = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
                results.append({
                    "title": (e.title.text if e.title else "").strip(),
                    "abstract": (e.summary.text if e.summary else "").strip()[:1000],
                    "url": (e.id.text if e.id else "").strip(),
                    "pdf_url": pdf,
                    "authors": [a.find("name").text for a in e.find_all("author") if a.find("name")][:5],
                })
            # Record discovery query for the abstract URLs so the CSV column 3 stays meaningful.
            for r_ in results:
                if r_["url"] and r_["url"] not in self._url_to_query:
                    self._url_to_query[r_["url"]] = f"arxiv:{query}"
                if r_["pdf_url"] and r_["pdf_url"] not in self._url_to_query:
                    self._url_to_query[r_["pdf_url"]] = f"arxiv:{query}"
            return results
        except Exception as e:
            logger.warning(f"[arxiv_search] Error: {e}")
            return {"error": str(e), "results": []}

    def _zenodo_search(self, inp: dict) -> list[dict] | dict:
        query = (inp or {}).get("query")
        if not query or not isinstance(query, str):
            return {"error": "Missing required argument 'query'."}
        if self._SYNTHETIC_QUERY_RE.search(query):
            return {
                "error": "REJECTED: synthetic-themed query. Search for real-world imagery instead.",
                "rejected": True,
            }
        norm_q = "zenodo:" + " ".join(query.lower().split())
        if norm_q in self._recent_queries:
            return {
                "error": "REJECTED: You already ran this exact zenodo_search. Vary your keywords.",
                "rejected": True,
            }
        self._recent_queries.append(norm_q)
        # Always counts as a portal search for Phase-A gate.
        self._portal_search_count += 1
        max_results = min(int((inp or {}).get("max_results", 10) or 10), 20)
        try:
            self.rate_limiter.wait("zenodo.org")
            logger.info(f"[zenodo_search] {query!r} max={max_results}")
            r = self._http.get(
                "https://zenodo.org/api/records",
                params={"q": query, "type": "dataset", "size": max_results, "sort": "bestmatch"},
                timeout=20.0,
            )
            if r.status_code != 200:
                return {"error": f"Zenodo API returned HTTP {r.status_code}", "results": []}
            data = r.json()
            hits = (data.get("hits") or {}).get("hits") or []
            results = []
            for h in hits:
                meta = h.get("metadata") or {}
                record_id = h.get("id", "")
                doi = h.get("doi") or meta.get("doi") or ""
                creators = [c.get("name", "") for c in (meta.get("creators") or [])[:5]]
                license_id = (meta.get("license") or {}).get("id") or ""
                flist = h.get("files") or []
                fmts = list({f.get("type", "") for f in flist if f.get("type")})
                results.append({
                    "title": meta.get("title", ""),
                    "description": (meta.get("description") or "")[:600],
                    "url": f"https://zenodo.org/records/{record_id}",
                    "doi": doi,
                    "license": license_id,
                    "creators": creators,
                    "formats": fmts,
                    "publication_date": meta.get("publication_date", ""),
                })
            if not results:
                return {"message": "No datasets found on Zenodo for this query.", "results": []}
            return results
        except Exception as e:
            logger.warning(f"[zenodo_search] Error: {e}")
            return {"error": str(e), "results": []}

    def _store_dataset(self, inp: dict) -> dict:
        try:
            inp = inp or {}
            url = (inp.get("url") or "").strip()
            name = (inp.get("name") or "").strip()
            description = (inp.get("description") or "").strip()
            tags = inp.get("tags") or []
            if not isinstance(tags, list):
                tags = [str(tags)]

            # Pre-validate required fields so a malformed call does not crash the loop.
            if not name:
                return {
                    "success": False, "rejected": True, "reason": "missing_name",
                    "message": "REJECTED: 'name' is required and must be a non-empty string.",
                }
            if not url or not url.startswith(("http://", "https://")):
                return {
                    "success": False, "rejected": True, "reason": "missing_url",
                    "message": "REJECTED: 'url' must be a real http(s) URL pointing to the dataset's landing page.",
                }
            if "relevance_score" not in inp or "relevance_reasoning" not in inp:
                return {
                    "success": False, "rejected": True, "reason": "missing_relevance",
                    "message": "REJECTED: relevance_score (0..1) and relevance_reasoning are required.",
                }
            # num_samples must be int if present.
            ns = inp.get("num_samples")
            if ns is not None and not isinstance(ns, int):
                try:
                    inp["num_samples"] = int(str(ns).replace(",", "").strip())
                except Exception:
                    inp["num_samples"] = None

            # Reject homepage / listing-page URLs masquerading as datasets.
            if self._is_homepage_url(url):
                logger.info(f"[store_dataset] REJECTED homepage URL: {url}")
                return {
                    "success": False, "rejected": True, "reason": "homepage_url",
                    "message": (
                        f"REJECTED: '{url}' is a site root or listing page, not a dataset page. "
                        f"Drill into a specific dataset entry (e.g. /datasets/<slug> or /records/<id>) "
                        f"and store that URL instead."
                    ),
                }

            # Reject URLs that point to a paper / report / blog post about
            # a dataset rather than the dataset itself.
            if self._looks_like_paper_or_report(url):
                logger.info(f"[store_dataset] REJECTED paper/report URL: {url}")
                return {
                    "success": False, "rejected": True, "reason": "paper_or_report",
                    "message": (
                        f"REJECTED: '{url}' looks like a paper PDF, report, or blog post about "
                        f"a dataset — NOT the dataset's primary host. Find the actual dataset "
                        f"landing page (Zenodo, figshare, dataverse, IEEE DataPort, .edu lab page, "
                        f"or the GitHub repo's data link) and store that instead."
                    ),
                }

            # Reject low-quality aggregator/scraper sites.
            if self._is_aggregator(url):
                logger.info(f"[store_dataset] REJECTED aggregator URL: {url}")
                return {
                    "success": False, "rejected": True, "reason": "aggregator_domain",
                    "message": (
                        f"REJECTED: '{self._domain_of(url)}' is a third-party aggregator that re-hosts "
                        f"other people's datasets. Find the PRIMARY source (the original Zenodo / "
                        f"figshare / IEEE DataPort / .edu / .gov / Roboflow Universe page) and store that."
                    ),
                }

            # Verify the URL is actually reachable before saving.
            if not self._url_reachable(url):
                logger.info(f"[store_dataset] REJECTED unreachable URL: {url}")
                return {
                    "success": False, "rejected": True, "reason": "unreachable_url",
                    "message": (
                        f"REJECTED: '{url}' returned a non-OK status (404, DNS failure, or connection error). "
                        f"Do not store dead URLs. Find the dataset's current canonical landing page first "
                        f"(via fetch_page) and store the URL you actually verified."
                    ),
                }

            # Reject synthetic / simulated datasets.
            if self._looks_synthetic(name, description, " ".join(tags)):
                logger.info(f"[store_dataset] REJECTED synthetic dataset: {name!r}")
                return {
                    "success": False, "rejected": True, "reason": "synthetic",
                    "message": (
                        "REJECTED: This dataset appears to be SYNTHETIC / SIMULATED / CGI / "
                        "GAN-generated / game-engine-rendered. Real-world imagery only. "
                        "Skip this and find a dataset built from genuine satellite / aerial captures."
                    ),
                }

            # Perspective check — fires only when the user's subject mentions
            # aerial/satellite/overhead/UAV/drone. Aircraft-detection models trained
            # on ground photos are useless for overhead imagery, so we reject
            # ground-perspective stores even when the URL/score look fine.
            if self._subject_wants_aerial:
                blob = " ".join([name, description, " ".join(tags)])
                if self._GROUND_PERSPECTIVE_RE.search(blob):
                    logger.info(f"[store_dataset] REJECTED ground-perspective: {name!r}")
                    return {
                        "success": False, "rejected": True, "reason": "ground_perspective",
                        "message": (
                            "REJECTED: This dataset appears to be GROUND-PERSPECTIVE imagery "
                            "(planespotter shots, on-tarmac photos, side views, museum displays, "
                            "Wikimedia Commons / Google Image Search aggregations). The user wants "
                            "AERIAL / SATELLITE / OVERHEAD imagery — top-down captures from "
                            "satellites / drones / aircraft. Ground photos of static planes are "
                            "out-of-distribution for overhead aircraft detection. Find a dataset "
                            "with genuine overhead / satellite / drone imagery instead."
                        ),
                    }
                if not self._AERIAL_GREEN_RE.search(blob):
                    # Soft warning only — cap score instead of hard-rejecting.
                    # Hard-rejecting here caused 0 stores because qwen models often omit
                    # "aerial"/"satellite" from the description even for real overhead datasets.
                    logger.info(f"[store_dataset] no aerial signal in description for {name!r} — capping score to 0.55")
                    inp = dict(inp)
                    inp["relevance_score"] = min(float(inp.get("relevance_score", 0.5)), 0.55)
                    inp["relevance_reasoning"] = (
                        "[perspective unverified — no aerial/satellite keyword in description; "
                        "update description if this is confirmed overhead imagery] "
                        + (inp.get("relevance_reasoning") or "")
                    )

            download_url = inp.get("download_url")

            # 1:N mainstream balancing rule (OSINT enforcement).
            if url and self._is_mainstream(url):
                required = self.mainstream_stored * MAINSTREAM_BALANCE_RATIO
                if self.alternative_stored < required:
                    deficit = required - self.alternative_stored
                    logger.info(
                        f"[store_dataset] REJECTED mainstream store — need {deficit} more alternative-domain "
                        f"datasets (mainstream={self.mainstream_stored}, alternative={self.alternative_stored})"
                    )
                    return {
                        "success": False,
                        "rejected": True,
                        "reason": "mainstream_quota",
                        "message": (
                            f"REJECTED: Mainstream domain limit reached. "
                            f"For every mainstream link (Kaggle/HF/GitHub), you must find "
                            f"{MAINSTREAM_BALANCE_RATIO} datasets from alternative domains. "
                            f"You have stored {self.mainstream_stored} mainstream and only "
                            f"{self.alternative_stored} alternative — find {deficit} more from "
                            f"academic archives (figshare, dataverse, zenodo), government portals "
                            f"(catalog.data.gov, registry.opendata.aws, european-data.europa.eu), "
                            f"or decentralized repos before returning to these sites."
                        ),
                    }

            # Reject duplicates against the global DB (canonicalized so DOI
            # aliases, www-prefixes, and trailing slashes all collapse).
            canon = self._canonical_url(url)
            canon_dl = self._canonical_url(download_url) if download_url else ""
            if canon in self._seen_canonical or (canon_dl and canon_dl in self._seen_canonical) \
               or url in self.seen_urls:
                self.skipped_store_duplicate += 1
                logger.info(f"[store_dataset] SKIP duplicate URL (canonical={canon}): {url}")
                return {
                    "skipped": True,
                    "reason": "already_stored",
                    "message": (
                        f"This dataset is ALREADY stored (canonical URL: {canon}). "
                        f"DOI aliases and slug variants count as duplicates. "
                        f"Find a DIFFERENT dataset — try a new source or a new keyword."
                    ),
                    "url": url,
                }

            source_val = inp.get("source", "generic")
            try:
                source = DatasetSource(source_val)
            except ValueError:
                source = DatasetSource.GENERIC

            # Trust Card v1: pull license/provenance fields if the LLM provided them.
            spdx = (inp.get("license_spdx") or "").strip() or None
            commercial = inp.get("license_commercial_ok")
            if isinstance(commercial, str):
                commercial = commercial.lower() in ("true", "yes", "1")
            doi = (inp.get("doi") or "").strip() or None
            if doi:
                # Strip any URL prefix the LLM may have included
                doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi).strip("/ ")
            authors = inp.get("authors") or []
            if not isinstance(authors, list):
                authors = [str(authors)]
            institution = (inp.get("institution") or "").strip() or None
            country = (inp.get("country") or "").strip().upper() or None
            if country and len(country) > 2:
                country = country[:2]  # coerce to alpha-2

            dataset = Dataset(
                query_id=self.query_id,
                name=inp["name"],
                url=inp["url"],
                download_url=inp.get("download_url"),
                source=source,
                formats=inp.get("formats", []),
                description=inp.get("description", ""),
                license=inp.get("license"),
                size_human=inp.get("size_human"),
                num_samples=inp.get("num_samples"),
                date_range_start=inp.get("date_range_start"),
                date_range_end=inp.get("date_range_end"),
                tags=inp.get("tags", []),
                relevance_score=float(inp["relevance_score"]),
                relevance_reasoning=inp["relevance_reasoning"],
                license_spdx=spdx,
                license_commercial_ok=commercial if isinstance(commercial, bool) else None,
                doi=doi,
                authors=[str(a) for a in authors[:20]],
                institution=institution,
                country=country,
            )
            saved_id = self.storage.save_dataset(dataset)
            dataset.id = saved_id
            self.seen_urls.add(dataset.url)
            self._seen_canonical.add(self._canonical_url(dataset.url))
            if dataset.download_url:
                self.seen_urls.add(dataset.download_url)
                self._seen_canonical.add(self._canonical_url(dataset.download_url))
            if self._is_mainstream(dataset.url):
                self.mainstream_stored += 1
            else:
                self.alternative_stored += 1
            self._stores_so_far += 1
            logger.info(
                f"[store_dataset] id={saved_id} score={dataset.relevance_score:.2f} name={dataset.name!r}"
            )
            # Append a row to the per-run CSV (S.No, Website name, What does the website, Link).
            if self._csv_writer is not None:
                self._csv_serial += 1
                try:
                    self._csv_writer.writerow([
                        self._csv_serial,
                        dataset.name,
                        (dataset.description or "")[:2000],
                        dataset.url,
                    ])
                    self._csv_fh.flush()
                except Exception as csv_err:
                    logger.warning(f"CSV write error: {csv_err}")
            if self.on_dataset_stored is not None:
                try:
                    self.on_dataset_stored(dataset)
                except Exception as cb_err:
                    logger.warning(f"on_dataset_stored callback error: {cb_err}")
            return {"success": True, "dataset_id": saved_id, "message": f"Stored: {inp['name']}"}
        except Exception as e:
            logger.warning(f"[store_dataset] Error: {e}")
            return {"success": False, "error": str(e)}

    def _mark_search_complete(self, inp: dict) -> dict:
        logger.info(
            f"[mark_search_complete] total={inp.get('total_found')} "
            f"summary={str(inp.get('summary', ''))[:100]}"
        )
        return {"done": True, "summary": inp.get("summary", ""), "total": inp.get("total_found", 0)}
# feat/search-reproducibility: Add search-reproducibility helper function
# feat/search-reproducibility: Wire search-reproducibility into search pipeline
# feat/search-reproducibility: Add tests for search-reproducibility feature
