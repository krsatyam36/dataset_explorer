# dataset_search

An autonomous, OSINT-style CLI agent that scours the public internet for ML
datasets matching a free-form subject. Runs entirely locally on Ollama, persists
findings to SQLite, exposes a live web dashboard, and produces per-run CSV
exports plus a Trust Card (license + provenance metadata) for every dataset it
saves.

Designed to behave like an investigator, not a Kaggle scraper.

```bash
dataset_search "satellite imagery of military aircraft dataset" --format all --depth 3 --unlimited
# → opens a live dashboard at http://127.0.0.1:7860
# → saves CSV to ~/datasets_explorer/logs/csv/<ddmmyy-hhmmss>_scrape.csv
# → persists to ~/.dataset_search/results.db
```

---

## Installation

```bash
cd ~/datasets_explorer
pip install --user --break-system-packages -e .
pip install --user --break-system-packages pypdf

# Optional: JS-rendered fetch for SPAs (Roboflow Universe, Mendeley, etc.)
pip install --user --break-system-packages playwright
playwright install chromium

# Required runtime: Ollama with qwen2.5:14b pulled
ollama pull qwen2.5:14b
```

`.env` defaults (override as needed):

```
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=qwen2.5:14b
DATASET_SEARCH_DIR=~/.dataset_search           # DB + logs
DATASET_SEARCH_CSV_DIR=~/datasets_explorer/logs/csv
BRAVE_API_KEY=                                 # optional: Brave Search fallback
DEFAULT_DEPTH=2
DEFAULT_HOURS=2.0
```

Phase-A gate constants live in `config.py` (not env-overridable by default):
`PHASE_A_MIN_STORES=2`, `PHASE_A_MIN_PORTAL_SEARCHES=8`, plus a `PORTAL_HOSTS`
tuple listing every host treated as a portal for the gate.

---

## Usage

```bash
dataset_search "<subject>" \
  [--format FMT[,FMT]|all] \
  [--time "<range>"] \
  [--depth 1|2|3] \
  [--model NAME] \
  [--hours N | --unlimited] \
  [--min-relevance 0.0..1.0] \
  [--web/--no-web] \
  [--web-port 7860]
```

`<subject>` is positional and required — a free-form description of what you
need. Be specific.

### `dataset_search` flags

| Flag | Type | Default | What it does |
|---|---|---|---|
| `<subject>` | string | — | (positional) Free-form topic to search for. Quote it. |
| `--format <fmt>` | string | `all` | Required dataset format(s). Comma-separate multiple, or `all` for any. e.g. `geotiff,tiff,cog`. The agent rejects datasets that clearly don't expose one of these formats. |
| `--time <range>` | string | (empty) | Free-form time-range filter. e.g. `"2020-2024"`, `"last 5 years"`. Passed to the LLM as a soft preference. |
| `--depth <1\|2\|3>` | int | `2` | Search aggressiveness. `1` fast first pass (~80 iters). `2` deep (~200 iters, min 25 stores). `3` frontier (cap 2000 iters, min 50 stores, full A→E phase pipeline incl. arXiv + READMEs + deep web). |
| `--model <name>` | string | from `.env` (`qwen2.5:14b`) | Ollama model name. Auto-falls-back through `OLLAMA_FALLBACK_MODELS` if the requested model isn't pulled. |
| `--hours <N>` | float | `2.0` | Max search duration in hours. Ignored when `--unlimited` is set. |
| `--unlimited` | flag | off | No time cap. Runs until `mark_search_complete` is accepted or you Ctrl+C. Findings save continuously, so Ctrl+C never loses data. |
| `--min-relevance <0..1>` | float | `0.0` | Threshold for the live console findings stream. Does **not** affect what gets stored — every confirmed dataset is always saved to the DB. |
| `--web` / `--no-web` | flag | `--web` | Serve the live dashboard at `http://127.0.0.1:<port>` during the run. |
| `--web-port <N>` | int | `7860` | Port for the dashboard. Auto-bumps to the next free port if busy. |
| `-h`, `--help` | flag | — | Show help and exit. |

Findings save to disk **continuously** — Ctrl+C never loses data.

### Common invocations

```bash
# Frontier-mode aircraft search, no time limit, dashboard on:
dataset_search "satellite imagery of military aircraft" --depth 3 --unlimited

# Fast triage with format filter and a time window:
dataset_search "SAR ship detection annotated" \
  --format geotiff,tiff --time "2020-2024" --depth 1 --hours 0.5

# Deep run pinned to a specific model, dashboard off (headless box):
dataset_search "aerial orthoimagery building footprints" \
  --depth 2 --unlimited --model mistral-small:latest --no-web

# Frontier run on a custom port, only show high-confidence stores in console:
dataset_search "multispectral UAV imagery" \
  --depth 3 --unlimited --min-relevance 0.7 --web-port 8080
```

---

## What the agent actually does

### Discovery workflow (depth 3 — frontier)

| Phase | Iterations | Tools |
|---|---|---|
| **A — Portals** *(hard-gated)* | 1–40 | `web_search` on figshare, zenodo, dataverse, .gov, .edu, registry.opendata.aws, paperswithcode, OpenAerialMap, IEEE DataPort, Roboflow Universe. Research-phase tools (`arxiv_search`, `read_pdf`) are **rejected** until ≥2 stores OR ≥8 distinct portal-targeted searches have happened — forces portal-first behavior. |
| **B — Research papers** | 41–110 | `arxiv_search` → `read_pdf` extracts dataset names from paper bodies → quoted follow-up `web_search` for each name |
| **C — README mining** | 111–180 | `read_github_readme` on `awesome-X` lists and benchmark repos |
| **D — Deep web / OSINT** | 181–260 | Ahmia, Tor2Web gateways, Wayback-archived OSINT forums, Bellingcat, IntelX, r/OSINT |
| **E — Link chasing** | 261+ | Multi-hop from already-found datasets |

### Three-axis relevance scoring

The system prompt forces the model to grade every potential store along three
axes and *multiply* them — one weak axis collapses the score:

1. **Content match** — is the subject really represented?
2. **Perspective match** — if the user mentioned aerial/satellite/overhead/UAV/drone,
   ground photos score 0.0 on this axis (not stored).
3. **Labeling quality** — bbox/masks/class labels score 1.0; raw imagery 0.5; image-only ≤0.3.

`relevance_score = content × perspective × labeling`. ≥0.9 only when all three are ≥0.9.

### Tools the agent has

| Tool | Purpose |
|---|---|
| `web_search(query)` | DuckDuckGo (with Brave fallback if `BRAVE_API_KEY` set). Required to use `site:` / `filetype:` / `inurl:` operators. |
| `fetch_page(url)` | HTML scraper |
| `fetch_page_js(url)` | Playwright-backed fetch for JS-rendered SPAs (Roboflow Universe, Mendeley, etc.). **Auto-falls-back to `fetch_page` when Playwright is not installed** — no wasted iteration. |
| `arxiv_search(query)` | Direct arXiv API — cleaner academic discovery than DDG |
| `read_pdf(url, max_pages=12)` | Extracts text from academic PDFs (auto-resolves arXiv / MDPI / PMC landing pages → PDF) |
| `read_github_readme(repo_url)` | Fetches README, returns text + every linked URL (Zenodo / Drive / OneDrive data hosts) |
| `store_dataset(...)` | Confirmed dataset write (gated by every guardrail below) |
| `mark_search_complete(...)` | Ends the run — only accepted when min stores reached and not on cooldown |

---

## Guardrails

The agent's output is filtered by a stack of code-level checks. None of these
rely solely on prompt instructions — every rule is enforced in `tools.py`.

### Workflow enforcement
- **Phase A gate** — `arxiv_search` and `read_pdf` are rejected until either
  ≥2 datasets have been stored OR ≥8 distinct portal-targeted `web_search`
  calls have been issued (a portal-targeted query is one whose `site:` filter
  matches `PORTAL_HOSTS` in `config.py`). Stops the agent from skipping Phase A
  to chase arXiv detours.
- **`mark_search_complete` cooldown** — after one rejection (insufficient
  stores), the next 20 iterations refuse it outright. Stops the spam.

### Source diversity
- **1:2 mainstream balancing** — for every store from `kaggle.com` /
  `huggingface.co` / `github.com`, the agent must store 2 from alternative
  domains. Mainstream stores violating this ratio are rejected with a message
  pointing the agent to figshare / zenodo / dataverse / .gov.
- **Operator rotation** — 3 consecutive `web_search` calls without a `site:` /
  `filetype:` / `inurl:` operator → next broad query rejected.
- **Repeat-query rejection** — last 14 normalized queries deduped across
  `web_search` and `arxiv_search`.
- **Per-URL dedup** — `read_pdf` and `read_github_readme` refuse to re-read the
  same target within a run.
- **Within-iteration dedup** — duplicate tool calls within a single model turn
  collapsed to one (fixes the "fetch_page X 8× in one iteration" issue).
- **Multilingual nudge** — system prompt instructs the model to also issue
  queries in Chinese / German / French / Russian / Spanish after exhausting
  English on a portal.

### Query hygiene (rejected before they hit the search engine)
- **Synthetic-themed queries** — `web_search` and `arxiv_search` reject any
  query containing `synthetic`, `simulated`, `simulation`, `cgi`, `airsim`,
  `unreal`, `unity`, `gan-generated`, `rendered`, `procedurally generated`.
  Stops the agent from drifting into synthetic territory before DDG quota is
  even spent.
- **Malformed `site:X OR site:Y` queries** — rejected with a message telling
  the agent to issue separate calls (DDG silently ignores OR'd site filters).
- **IEEE Xplore PDFs** — `read_pdf` short-circuits on `ieee.org` /
  `ieeexplore.ieee.org` (login-walled, like ScienceDirect) with a hint to use
  the dataset's primary host or arXiv preprint instead.

### Content quality
- **No synthetic data** — name + description + tags scanned for
  `synthetic|simulated|cgi|airsim|unreal|unity|gan-generated|rendered`.
  Code-level rejection (prompt-only didn't hold).
- **Perspective check** *(fires automatically when the user's subject contains
  aerial/satellite/overhead/UAV/drone/remote-sensing keywords)*:
  - **Ground-perspective rejection** — stores whose name+description+tags
    mention `Wikimedia Commons`, `Google Image Search`, `planespotter`,
    `on-tarmac`, `airshow`, `side-view`, `ground-level`, `from below`,
    `cockpit view`, `museum aircraft`, or `static display` are rejected as
    out-of-distribution for overhead detection models.
  - **Aerial-signal requirement** — stores must mention at least one of
    `aerial / satellite / overhead / nadir / top-down / orthoimagery /
    orthomosaic / drone imagery / UAV imagery / remote sensing / GeoTIFF /
    COG / Sentinel / Landsat / Maxar / Planet / Airbus / Copernicus /
    spaceborne / airborne / bird's-eye`. Missing this signal → rejected with
    a "fetch_page again to confirm or skip" message.
- **No paper / report / blog URLs** — paths ending in `.pdf`, or containing
  `/reports/`, `/publications/`, `/papers/`, `/pdf/`, `servej.php` are rejected
  with "find the dataset's primary host, not the paper about it."
- **Aggregator blocklist** — `gts.ai`, `innovatiana.com`, `datasetninja.com`,
  `wandb.ai`, `cnas.org`, `simuletic.com`, `thegrenze.com`, `researchgate.net`,
  `academia.edu`, `semanticscholar.org`, `libguides.utdallas.edu`.
- **Homepage rejection** — paths like `/`, `/datasets`, `/data`, `/search`
  blocked.

### URL hygiene
- **Reachability check** — every `store_dataset` HEAD-probes the URL; rejects
  404 / DNS failure / connection error. Accepts 401/403/405 (gated but real).
- **Canonicalization for dedup** — `doi.org/10.5281/zenodo.X` collapses to
  `zenodo.org/records/X`; `www.` and trailing `/` stripped; lowercased. Catches
  the "same dataset under 5 URL variants" problem.

### Lifecycle
- **Robust LLM call** — `chat()` wrapped in try/except; if response can't be
  parsed (gpt-oss harmony issue), inject a corrective user message and continue
  rather than crash the whole run.
- **`gpt-oss` harmony suppression** — passes `think=False` when model name
  contains `gpt-oss` (gpt-oss + Ollama tool-calling is structurally broken — see
  Ollama issue #12187; recommended path is llama.cpp + llama-server if you must
  use gpt-oss).

---

## Trust Card v1 — License & Provenance

For every confirmed store, the LLM opportunistically extracts and persists:

| Column | Example |
|---|---|
| `license_spdx` | `CC-BY-4.0`, `MIT`, `Apache-2.0`, `unknown` |
| `license_commercial_ok` | `1` / `0` / null |
| `doi` | `10.5281/zenodo.7331974` |
| `authors` | JSON array of names |
| `institution` | `NASA`, `TU Munich`, `Airbus DS` |
| `country` | ISO alpha-2: `US`, `CN`, `DE` (matters for ITAR / export-control review) |

These fields auto-migrate onto existing DBs. Query them directly:

```bash
sqlite3 ~/.dataset_search/results.db \
  "SELECT name, license_spdx, country, institution, doi
   FROM datasets WHERE license_spdx IS NOT NULL ORDER BY id DESC LIMIT 20;"
```

**Roadmap (not yet built):** cross-dataset perceptual-hash leakage detection,
quality signals (resolution histogram, label noise estimation), URL liveness
watch.

---

## Web dashboard

Every run launches a live dashboard at **http://127.0.0.1:7860** (Server-Sent
Events, pure stdlib, no external deps). Disable with `--no-web`.

Sections:

| Panel | What's shown |
|---|---|
| **Header** | Subject · model · depth · live indicator · running elapsed clock |
| **6 KPI tiles** | Iteration / cap · Stored / needed · Mainstream:Alternative ratio · Searches (with rejection count) · PDFs + READMEs read · Sites visited + total fetches |
| **Live activity feed** | Every tool call, timestamped and color-coded; clickable URLs |
| **Where the agent is now** | The action the agent is executing this very moment |
| **Confirmed datasets** | Each store with score, name, URL, and trust-card tags (license, country, institution, DOI) |
| **Agent thinking** | Collapsible per-iteration dropdowns containing the model's full free-text reasoning before it picked a tool |
| **Sites visited** | Every domain touched, with hit counts |

Tabs auto-reconnect on disconnect; events replay from history when opened
mid-run.

---

## Outputs

| Path | Format | Purpose |
|---|---|---|
| `~/.dataset_search/results.db` | SQLite | Authoritative store; survives across runs |
| `~/.dataset_search/logs/search_<UTC>.log` | Plain text | Per-run debug log (ollama, ddg, agent decisions, pypdf warnings) |
| `~/datasets_explorer/logs/csv/<ddmmyy-hhmmss>_scrape.csv` | CSV | Per-run findings: `S.No`, `Website name`, `What does the website`, `Link` |
| Web dashboard | HTTP / SSE | Real-time visibility while the run is in flight |

---

## `datasets_explorer` — companion CLI

A second entry point for browsing past runs and managing already-stored
datasets. Same database as `dataset_search`.

```bash
datasets_explorer <subcommand> [options]
```

### Subcommands

#### `search` — alternative entry to start a new search

```bash
datasets_explorer search "<query>" \
  [--hours N] \
  [--min-relevance 0.0..1.0] \
  [--model NAME]
```

| Flag | Type | Default | What it does |
|---|---|---|---|
| `<query>` | string | — | (positional) Subject to search for. |
| `--hours <N>` | float | `2.0` | Max search duration. |
| `--min-relevance <0..1>` | float | `0.5` | Console-display threshold (does not affect storage). |
| `--model <name>` | string | from config | Ollama model name. |

> Prefer `dataset_search` for new runs — it has the full flag set (depth, web,
> CSV, etc.). `datasets_explorer search` is kept as a thin alias.

#### `queries` — list every run in the DB

```bash
datasets_explorer queries
```

Shows query id, raw subject, status (`running` / `completed` / `interrupted`),
start time, and dataset count for every search ever executed.

#### `results` — browse stored datasets

```bash
datasets_explorer results \
  [--query-id N] \
  [--min-relevance 0.0..1.0] \
  [--format <fmt>] \
  [--limit N]
```

| Flag | Type | Default | What it does |
|---|---|---|---|
| `--query-id <N>` | int | (none) | Filter to one specific search run. |
| `--min-relevance <0..1>` | float | `0.0` | Hide stores below this score. |
| `--format <fmt>` | string | (none) | Filter by data format token (e.g. `geotiff`, `jpeg`, `npy`). |
| `--limit <N>` | int | `50` | Max rows to display. |

#### `review` — annotate a dataset

```bash
datasets_explorer review <dataset_id> \
  [--notes "..."] \
  [--reviewed]
```

| Flag | Type | Default | What it does |
|---|---|---|---|
| `<dataset_id>` | int | — | (positional) The dataset's primary-key id (visible in `results`). |
| `--notes "..."` | string | (none) | Free-form note to attach. Useful for "downloaded", "training-set candidate", "license-blocked", etc. |
| `--reviewed` | flag | off | Mark this dataset as reviewed by a human (filterable later). |

### Direct SQLite access

The DB is plain SQLite — query it with the `sqlite3` CLI for anything the
subcommands don't cover:

```bash
# All non-mainstream datasets with a license
sqlite3 ~/.dataset_search/results.db \
  "SELECT name, license_spdx, country, url
   FROM datasets
   WHERE license_spdx IS NOT NULL
     AND url NOT LIKE '%kaggle.com%'
     AND url NOT LIKE '%huggingface.co%'
     AND url NOT LIKE '%github.com%'
   ORDER BY relevance_score DESC
   LIMIT 30;"
```

---

## Architecture

```
datasets_explorer/
├── search_cli.py     # Click CLI + live console stream + web dashboard wiring
├── agent.py          # System prompt builder + iteration loop
├── tools.py          # All tool implementations + guardrails (the bulk of the logic)
├── storage.py        # SQLite layer + auto-migration
├── models.py         # Pydantic models (Dataset, SearchQuery)
├── web_ui.py         # Pure-stdlib HTTP server + embedded HTML dashboard
├── config.py         # Paths, defaults, OSINT constants, BRAVE_API_KEY
├── utils.py          # Rate limiter + logging setup
└── cli.py            # Legacy `datasets_explorer` subcommands (queries, results, review)
```

---

## Known limitations

- Single LLM provider (Ollama). gpt-oss is broken with Ollama tool calling;
  recommended fallback is llama.cpp + llama-server.
- Single primary search engine (DDG); Brave is opt-in via `BRAVE_API_KEY`.
- SQLite → single user, single host. Postgres + a queue would be needed for
  multi-tenant or scheduled runs.
- No tests, no CI, no `pyproject.toml`, no schema migrations beyond the
  auto-`ALTER TABLE` shim. This is a working prototype, not production
  software.

---

## Hardware notes

Tested on HP Pavilion 15 (Ubuntu 24.04, no GPU). `qwen2.5:14b` is the practical
default — `mistral-small:24b` produces better tool calls but is CPU-bound on
this hardware (~3 cores at 308%, seconds per token). Models >14B are not
recommended without GPU acceleration.
