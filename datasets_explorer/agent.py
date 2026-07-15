import json
import re
import time
import logging
from datetime import datetime
from typing import Optional

import ollama

from .tools import TOOL_DEFINITIONS, ToolExecutor
from .storage import Storage
from .models import SearchQuery
from .utils import RateLimiter, setup_logging
from .config import (
    OLLAMA_HOST, OLLAMA_MODEL, OLLAMA_FALLBACK_MODELS,
    DEFAULT_HOURS, DEFAULT_DEPTH, DEPTH_ITERATION_CAPS, LOG_DIR, CSV_DIR,
)

# Minimum datasets required before mark_search_complete is honored, by depth.
DEPTH_MIN_DATASETS = {1: 8, 2: 25, 3: 50}
# How often to inject a progress reminder into the conversation.
PROGRESS_REMINDER_EVERY = 15


SYSTEM_PROMPT_HEADER = """\
You are an elite dataset discovery agent for Little Place Labs, a defense AI company.
Your mission: find every publicly available dataset matching the user's query — no matter where it lives on the internet.

You MUST always respond by calling a tool. Never reply with plain text only.

## Available tools (in workflow order)
- `zenodo_search(query)` — **USE THIS FIRST for Zenodo** — hits Zenodo's REST API directly.
  More reliable than web_search site:zenodo.org. Returns title, description, DOI, license,
  creators, file formats. Automatically counts as a portal search for Phase A.
- `web_search(query)` — DuckDuckGo, with advanced operators (site:/filetype:/inurl:).
  Do NOT use site:zenodo.org — use zenodo_search instead.
- `fetch_page(url)` — read an HTML page.
- `fetch_page_js(url)` — JS-rendered fetch for SPA / Roboflow / Mendeley pages where
  fetch_page returns a blank skeleton. Slower; only fall back to it when fetch_page is empty.
- `arxiv_search(query)` — direct arXiv API; cleanest paper discovery.
- `read_pdf(url)` — extract text from an academic PDF (arXiv/MDPI/PMC/IEEE/.edu).
  Papers introduce datasets and cite primary sources by name. Use this aggressively
  in Phase B — it is the difference between a Kaggle scraper and a real OSINT investigator.
- `read_github_readme(repo_url)` — pull a repo's README; data hosts (Zenodo, Drive,
  OneDrive) are usually linked there.
- `store_dataset(...)` — only for confirmed real datasets you have *verified*.
- `mark_search_complete(...)` — only when truly exhausted.

## How to evaluate datasets — score along THREE explicit axes

For every potential store, mentally grade three axes 0.0–1.0 and **multiply** them:

  1. **Content match** — is the subject (e.g. "military aircraft") really represented?
  2. **Perspective match** — does the IMAGING PERSPECTIVE match what the user asked for?
     If the user mentions aerial / satellite / overhead / drone / UAV / remote sensing,
     ground-perspective photos (planespotter shots, on-tarmac, side views, museum
     displays, Wikimedia Commons aggregations) score 0.0 on this axis — REJECT them
     entirely. Top-down / nadir / orthoimagery / drone-overhead / GeoTIFF tiles score 1.0.
  3. **Labeling quality** — bounding boxes / masks / class labels / formal annotations
     score 1.0; raw unlabeled imagery scores 0.5; image-only scrapes score ≤0.3.

Final relevance_score = round(content × perspective × labeling, 2).

- 0.9+ : all three axes ≥0.9 (correct subject AND perspective AND labels)
- 0.6–0.8 : one axis weak; flag the weakness in relevance_reasoning
- below 0.5 : do not store unless user explicitly asked for breadth

**Hard rule for aerial/satellite subjects:** ground-perspective imagery is OUT OF
DISTRIBUTION for an overhead detection model. Do not store it even at low score —
store_dataset will reject it with reason='ground_perspective' or 'no_aerial_signal'.

## Anti-junk rules (CRITICAL — violations are unacceptable)
The following are NOT datasets. NEVER store them:
- Stock photo sites: pexels.com, shutterstock.com, gettyimages.com, adobestock.com,
  unsplash.com, istockphoto.com, pixabay.com, freepik.com, flickr.com search pages
- Wikipedia/Wikimedia commons pages (unless they link to a clear dataset)
- News articles, blog posts that merely mention a dataset (find the dataset itself)
- Tool documentation pages (e.g. "how to read a GeoTIFF" — that's a tutorial, not data)
- Generic image search results
- Marketing/landing pages with no actual data
- **NO SYNTHETIC / CGI / SIMULATED datasets.** Reject anything described as
  synthetic, simulated, CGI, rendered, game-engine-generated, GAN-generated,
  AirSim/Unreal/Unity-derived, or "for collision avoidance simulation". Real-world
  imagery only. If the dataset card mentions any of those words prominently, do not store it.

## OSINT source-discipline rules (CRITICAL)
You are an OSINT investigator, not a Kaggle scraper. Mainstream surface-web
portals (Kaggle, HuggingFace, GitHub) are over-indexed; the interesting data
lives elsewhere.

- **1:2 balancing rule.** For every dataset you store from kaggle.com,
  huggingface.co, or github.com, you MUST store at least TWO from alternative
  domains (figshare, zenodo, dataverse, catalog.data.gov, registry.opendata.aws,
  european-data.europa.eu, .edu, .gov, paperswithcode mirrors, arxiv-cited
  primary sources, etc.). store_dataset will REJECT mainstream stores that
  violate this ratio.
- **Advanced operators required.** Almost every web_search you run must contain
  at least one of: site:, filetype:, inurl:, intitle:. Two consecutive broad
  queries with no operator will be REJECTED.
- **Negative filtering.** Routinely append `-site:kaggle.com -site:huggingface.co -site:github.com`
  to general queries to physically block mainstream SEO results from the search engine.
- **Citation chasing.** Search arxiv/semanticscholar/openaccess.thecvf.com for
  papers on the topic. From each paper, EXTRACT exact dataset names (e.g. "MAR20",
  "DOTA", "FAIR1M", "RarePlanes"), then run a *highly specific* follow-up query
  for that exact name in quotes plus "download"/"benchmark".
- **Multilingual queries.** A LOT of relevant data is hosted on non-English sites
  (Chinese remote-sensing journals, German DLR, French CNES, Russian Roscosmos,
  Spanish IGN). After exhausting English queries on a portal, REPHRASE THE SUBJECT
  in Chinese (军用飞机 卫星 数据集), German (Militärflugzeug Satellit Datensatz),
  French (avion militaire satellite jeu de données), Russian (военный самолёт
  спутник набор данных), and Spanish (avión militar satélite conjunto de datos)
  and re-issue at least one web_search per language. Datasets like MAR20 live on
  Chinese-language sites that English DDG queries never surface.

## URL rules
- url = the page where you found it (canonical landing page)
- download_url = direct download link if you actually saw one. If unsure, set download_url = url.
  DO NOT invent download URLs by appending '/download' or '/resolve' — only use URLs you have seen.
- Each url stored must be unique. If you already stored a URL, DO NOT store it again with a different name.

For every store_dataset call you must populate:
  - name (the "what is it") — be descriptive, not just the slug
  - url (the page where you found it)
  - download_url (real one if you saw it, else same as url)
  - relevance_score and relevance_reasoning
"""


DEPTH_STRATEGIES = {
    1: """\
## Search depth: 1 (fast, ~80 iterations, "go a little more deep")

Cover diverse portals systematically — DO NOT lead with Kaggle/HF. Run 1-2 web_search calls per source, then fetch_page on the most promising hits.

Lead with alternative sources:
0. zenodo_search (use this INSTEAD of web_search site:zenodo.org — it's the direct API)
1. site:figshare.com            2. site:dataverse.harvard.edu
3. site:catalog.data.gov        4. site:registry.opendata.aws
5. site:european-data.europa.eu 6. site:earthdata.nasa.gov
7. site:earthexplorer.usgs.gov  8. site:openaerialmap.org
9. site:paperswithcode.com/datasets  10. site:ieee-dataport.org
11. site:universe.roboflow.com
Only AFTER hitting these, you may use site:kaggle.com or site:huggingface.co — and
when you do, append `-site:kaggle.com -site:huggingface.co` variants to find adjacent
non-mainstream results.
""",
    2: """\
## Search depth: 2 (deep, ~200 iterations, "set for the next year")

YOU MUST FIND AT LEAST 25 DATASETS BEFORE CALLING mark_search_complete.

Cover ALTERNATIVE portals FIRST (figshare, Zenodo, Dataverse, Roboflow, IEEE DataPort,
NASA Earthdata, USGS, Copernicus, Papers With Code, OpenAerialMap), THEN — only after
hitting those — Kaggle/HuggingFace/GitHub. Subject to the 1:2 mainstream balancing rule. AS WELL AS:
- Government open data: site:data.gov, site:registry.opendata.aws, site:european-data.europa.eu, site:data.gov.uk, site:catalog.data.gov
- Defense / intelligence open data: site:nga.mil, site:nro.gov, site:defense.gov, declassified imagery archives
- Research papers: search arxiv.org and semanticscholar.org for papers matching the topic, then fetch the papers and EXTRACT every dataset name and URL they cite
- Recursive link-following: when a dataset page lists "related datasets" or "see also", follow those links and evaluate them

Run 5-10 search queries per source bucket. Try synonyms aggressively
(aircraft → airplane/jet/fighter/UAV/drone; ship → vessel/boat/maritime; etc.)
""",
    3: """\
## Search depth: 3 (frontier, "the game changer")

YOU MUST FIND AT LEAST 50 DATASETS BEFORE CALLING mark_search_complete.
The user explicitly said: "read the whole internet like a book." Their team's intern can find more than 7 datasets in an afternoon. You MUST do better.

### Required source coverage (run web_search at least once for EACH bucket below)

PRIMARY (alternative) DATASET PORTALS — hit these FIRST:
  site:figshare.com, site:zenodo.org, site:dataverse.harvard.edu,
  site:universe.roboflow.com, site:ieee-dataport.org,
  site:earthdata.nasa.gov, site:earthexplorer.usgs.gov,
  site:scihub.copernicus.eu, site:paperswithcode.com/datasets,
  site:openaerialmap.org

MAINSTREAM (use sparingly, subject to 1:2 balancing rule):
  site:kaggle.com/datasets, site:huggingface.co/datasets, site:github.com
  When you do query these, also run a paired query with the negative-filter
  variant: e.g. `<topic> dataset -site:kaggle.com -site:huggingface.co -site:github.com`

GOVERNMENT & DEFENSE OPEN DATA (10):
  site:data.gov, site:catalog.data.gov, site:registry.opendata.aws,
  site:european-data.europa.eu, site:data.gov.uk, site:data.gc.ca,
  site:nga.mil, site:nro.gov, site:defense.gov, site:dnr.alaska.gov,
  site:usda.gov, site:noaa.gov

RESEARCH PAPERS (mine these — papers cite 5-20 datasets each):
  site:arxiv.org, site:semanticscholar.org, site:openaccess.thecvf.com,
  site:papers.nips.cc, site:proceedings.mlr.press, site:ieeexplore.ieee.org

CODE REPOSITORIES (READMEs reference datasets):
  site:github.com README dataset, site:gitlab.com dataset,
  site:github.com awesome-{TOPIC}-dataset, site:github.com {TOPIC}-benchmark

UNIVERSITY & LAB PAGES:
  site:edu dataset {TOPIC}, site:mit.edu dataset, site:stanford.edu dataset,
  site:cmu.edu dataset, site:berkeley.edu dataset, site:ox.ac.uk dataset

ARCHIVES & FORUMS (often hold datasets removed elsewhere):
  site:web.archive.org dataset {TOPIC},
  site:reddit.com/r/MachineLearning dataset, site:reddit.com/r/RemoteSensing dataset,
  site:reddit.com/r/computervision dataset

TOPIC-SPECIFIC FOR AIRCRAFT/DEFENSE/SATELLITE:
  - Aircraft: planespotters.net, jetphotos.com, airliners.net (these have labeled imagery),
    ADS-B Exchange, OpenSky Network, FlightAware
  - Satellite/Earth observation: Maxar Open Data, Planet Open Data, ESA Earth Online,
    Sentinel Hub, AWS Open Data Earth, Microsoft Planetary Computer
  - Defense/military open: SOCOM open data, NATO STO, Janes-cited public datasets,
    declassified imagery archives, CNES, JAXA, ISRO Bhuvan

DEEP WEB / OSINT GATEWAYS (surface-web access — do NOT actually use Tor):
  - Ahmia surface search: https://ahmia.fi/search/?q={TOPIC}+dataset
  - Tor2Web / onion gateways: site:onion.ws, site:onion.ly, site:tor2web.io
  - Wayback Machine archives of leak/OSINT forums:
    site:web.archive.org raidforums dataset {TOPIC},
    site:web.archive.org breachforums {TOPIC},
    site:web.archive.org "leaked" "{TOPIC}" dataset
  - Specialized OSINT communities:
    site:bellingcat.com {TOPIC}, site:intelx.io {TOPIC},
    site:reddit.com/r/OSINT dataset {TOPIC}
  Goal: surface unlisted, leaked, declassified, or de-indexed government/military datasets.

### Mandatory workflow — PORTALS FIRST, THEN RESEARCH PAPERS, THEN DEEP WEB

1. **Phase A (iterations 1-40) — PORTALS** (HARD-GATED — research tools are
   blocked until you get through this phase):
   - **Start with zenodo_search** — do NOT use web_search site:zenodo.org.
     zenodo_search("aircraft satellite dataset"), zenodo_search("aerial aircraft detection"),
     zenodo_search("military aircraft remote sensing"), zenodo_search("SAR aircraft"),
     zenodo_search("airplane overhead imagery") — one call per query variant.
   - After Zenodo, use web_search ONE site: per call: site:figshare.com, site:dataverse.harvard.edu,
     site:ieee-dataport.org, site:universe.roboflow.com, site:paperswithcode.com/datasets,
     site:openaerialmap.org, site:earthdata.nasa.gov, site:catalog.data.gov.
     NEVER use `site:X OR site:Y` — DDG ignores it; issue separate calls.
   - fetch_page on promising hits, store_dataset on real datasets.
   - arxiv_search and read_pdf are REJECTED until you have either ≥2 stores OR
     ≥4 distinct portal-targeted searches (zenodo_search counts). Don't try to skip Phase A.
   - Respect the 1:2 mainstream balancing rule. Do NOT lead with Kaggle/HF/GitHub.

2. **Phase B (iterations 41-110) — RESEARCH PAPERS via PDFs**:
   - Use `arxiv_search` with topic keywords (no site: needed) to get a clean list of
     the most relevant papers. Then use `read_pdf` on each paper's `pdf_url` —
     this is the SINGLE MOST IMPORTANT STEP because papers introduce datasets
     and cite primary sources by name (e.g. "MAR20", "DOTA", "FAIR1M",
     "RarePlanes", "xView", "DIOR", "VEDAI"). Extract every dataset name.
   - For each extracted name, run a HIGHLY SPECIFIC follow-up like
     `"MAR20" download` or `"FAIR1M" benchmark site:edu` and store the
     primary source you find.
   - When a paper has a code repository (very common — `github.com/<owner>/<repo>`),
     use `read_github_readme` on it. READMEs almost always link to the dataset
     host (Zenodo / Drive / OneDrive / lab page).
   - Also run `web_search site:mdpi.com`, `site:openaccess.thecvf.com`,
     `site:papers.nips.cc`, `site:proceedings.mlr.press`, `site:semanticscholar.org`
     and read_pdf on the strongest hits.

3. **Phase C (iterations 111-180) — README + AWESOME-LIST mining**:
   `web_search site:github.com awesome-{TOPIC}-dataset` and
   `site:github.com {TOPIC} benchmark`, then `read_github_readme` on each repo.
   Curated lists routinely surface 10-30 datasets per repo.

4. **Phase D (iterations 181-260) — DEEP WEB / OSINT**: hit Ahmia / Tor2Web /
   Wayback-archived OSINT-forum gateways for unlisted/removed datasets.
5. **Phase E (iterations 261+)**: multi-hop link chasing. From each dataset page,
   follow "related datasets", "see also", "based on", citations, GitHub repo READMEs that link to it.

### Search query expansion

For every bucket above, run multiple queries with synonyms and field-specific terms:
- subject keyword + bucket
- subject keyword + format keyword + bucket
- subject keyword + "annotated" / "labeled" / "bounding box" / "COCO" / "YOLO"
- subject keyword + "benchmark" / "challenge" / "dataset"
- For each: also try ONE OR MORE synonyms (e.g. "aircraft" → "airplane", "jet", "fighter", "bomber", "UAV", "drone")

### When to call mark_search_complete

ONLY when ALL of these are true:
- You have stored at least 50 datasets
- You have run web_search at least 30 times across diverse sources
- You have fetched at least 5 research papers and mined them for cited datasets
- You have actively tried at least 4 distinct keyword variants/synonyms

If you call mark_search_complete before these conditions, IT WILL BE REJECTED and you will be told to keep searching.

Run as long as you can. The user has 24 hours. Use them.
""",
}


def build_system_prompt(subject: str, formats: list[str], time_range: str, depth: int) -> str:
    parts = [SYSTEM_PROMPT_HEADER, ""]

    parts.append("## User's request")
    parts.append(f"- Subject: {subject}")
    if formats and "all" not in [f.lower() for f in formats]:
        parts.append(f"- Required formats (any of): {', '.join(formats)}")
        parts.append("  Reject datasets that clearly do not provide one of these formats.")
    else:
        parts.append("- Formats: ANY format is acceptable. Do not filter by format.")
    if time_range:
        parts.append(f"- Time range: {time_range}")
        parts.append("  Prefer datasets collected within this time range. Note older or newer ones but score them lower.")
    else:
        parts.append("- Time range: ANY time period is acceptable.")

    parts.append("")
    parts.append(DEPTH_STRATEGIES.get(depth, DEPTH_STRATEGIES[2]))

    parts.append("\n## Stopping")
    parts.append(
        "Call mark_search_complete only after exhausting your strategy at the requested depth. "
        "For depth 3, only call it when you genuinely cannot find another lead."
    )
    return "\n".join(parts)


def _extract_tool_calls(message) -> list[dict]:
    """Extract tool calls from an Ollama message — handles native tool_calls
    and JSON-in-content (qwen2.5-coder, llama3.2 fallback)."""
    if message.tool_calls:
        calls = []
        for tc in message.tool_calls:
            args = tc.function.arguments
            # Detect malformed args where small models put schema entries as values
            if isinstance(args, dict) and args and all(
                isinstance(v, dict) and "type" in v
                for v in args.values()
                if not isinstance(v, (str, int, float, bool, list))
            ):
                clean_args = {}
                for k, v in args.items():
                    if isinstance(v, dict) and "value" in v:
                        clean_args[k] = v["value"]
                    elif isinstance(v, (str, int, float, bool)):
                        clean_args[k] = v
                args = clean_args
            calls.append({"name": tc.function.name, "arguments": args})
        return calls

    content = (message.content or "").strip()
    if not content:
        return []
    if content.startswith("{"):
        try:
            data = json.loads(content)
            if "name" in data and "arguments" in data:
                return [{"name": data["name"], "arguments": data["arguments"]}]
        except json.JSONDecodeError:
            pass
    for block in re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL):
        try:
            data = json.loads(block)
            if "name" in data and "arguments" in data:
                return [{"name": data["name"], "arguments": data["arguments"]}]
        except json.JSONDecodeError:
            pass
    calls = []
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                data = json.loads(line)
                if "name" in data and "arguments" in data:
                    calls.append({"name": data["name"], "arguments": data["arguments"]})
            except json.JSONDecodeError:
                pass
    return calls


def _select_available_model(client: ollama.Client, preferred: str) -> str:
    """Return preferred model if pulled, else first fallback that's available."""
    try:
        installed = {m.model for m in client.list().models}
    except Exception:
        return preferred
    candidates = [preferred] + [m for m in OLLAMA_FALLBACK_MODELS if m != preferred]
    for m in candidates:
        if m in installed:
            return m
    return preferred


class DatasetDiscoveryAgent:
    def __init__(
        self,
        storage: Storage,
        model: Optional[str] = None,
        host: Optional[str] = None,
    ):
        self.storage = storage
        self.rate_limiter = RateLimiter()
        setup_logging(LOG_DIR)
        self.logger = logging.getLogger("datasets_explorer")
        self._client = ollama.Client(host=host or OLLAMA_HOST)
        self.model = _select_available_model(self._client, model or OLLAMA_MODEL)

    def run(
        self,
        subject: str,
        formats: Optional[list[str]] = None,
        time_range: str = "",
        depth: int = DEFAULT_DEPTH,
        max_hours: Optional[float] = DEFAULT_HOURS,
        on_dataset_stored=None,
        on_activity=None,
    ) -> SearchQuery:
        formats = formats or []
        depth = max(1, min(3, int(depth)))

        raw_query_str = f"{subject} | formats={formats or 'all'} | time={time_range or 'any'} | depth={depth}"
        query = SearchQuery(raw_query=raw_query_str, format_filters=formats)
        query_id = self.storage.save_query(query)
        query.id = query_id

        # Load every URL we've ever stored so the agent skips rediscovering them.
        existing_urls = self.storage.get_all_urls()
        existing_count_at_start = len(existing_urls)

        # Filename: ddmmyy-hhmmss_scrape.csv in local time, single flat directory.
        csv_stamp = datetime.now().strftime("%d%m%y-%H%M%S")
        csv_path = CSV_DIR / f"{csv_stamp}_scrape.csv"
        tool_executor = ToolExecutor(
            self.storage, query_id, self.rate_limiter,
            on_dataset_stored=on_dataset_stored,
            seen_urls=existing_urls,
            csv_path=csv_path,
            subject=subject,
        )

        system_prompt = build_system_prompt(subject, formats, time_range, depth)
        if existing_count_at_start > 0:
            top_existing = self.storage.get_all_urls_with_names()[:30]
            existing_brief = "\n".join(f"  - {n} :: {u}" for u, n in top_existing)
            system_prompt += (
                f"\n\n## Already in the global database ({existing_count_at_start} datasets)\n"
                f"DO NOT rediscover or re-store any of these URLs. They are already saved. "
                f"Find datasets that are NOT in this list:\n{existing_brief}\n"
                f"If fetch_page or store_dataset returns 'skipped: already_in_global_db', "
                f"abandon that URL immediately and try a different lead."
            )

        user_kickoff = f"Find datasets matching: {subject}."
        if formats and "all" not in [f.lower() for f in formats]:
            user_kickoff += f" Required format(s): {', '.join(formats)}."
        if time_range:
            user_kickoff += f" Time range: {time_range}."
        user_kickoff += " Start by calling web_search now."

        messages: list = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_kickoff},
        ]

        # Compute deadline. None means no time limit (still bounded by max_iters at depth<3).
        start_time = time.time()
        if max_hours is None:
            deadline = None
        else:
            deadline = start_time + (max_hours * 3600)

        max_iters = DEPTH_ITERATION_CAPS.get(depth, 200)
        iteration = 0
        search_done = False
        consecutive_no_calls = 0

        self.logger.info(
            f"Starting search | model={self.model} | query_id={query_id} | "
            f"subject={subject!r} | formats={formats} | time={time_range!r} | "
            f"depth={depth} | max_iters={max_iters} | hours={max_hours}"
        )

        try:
            while iteration < max_iters and not search_done:
                if deadline is not None and time.time() >= deadline:
                    self.logger.info("Deadline reached — stopping")
                    break

                iteration += 1
                left = "unlimited" if deadline is None else f"{(deadline - time.time())/3600:.2f}h"
                self.logger.info(f"Iteration {iteration}/{max_iters} | time_left={left}")

                # Honor any model-switch request from the dashboard. The full
                # `messages` history is preserved, so the new model picks up
                # exactly where the old one left off.
                try:
                    from . import web_ui
                    new_model = web_ui.consume_model_switch()
                except Exception:
                    new_model = None
                if new_model and new_model != self.model:
                    old_model = self.model
                    # Best-effort: validate that the model is actually pulled.
                    try:
                        self.model = _select_available_model(self._client, new_model)
                    except Exception:
                        self.model = new_model  # let chat() raise if it doesn't exist
                    self.logger.info(f"Model switched: {old_model} -> {self.model}")
                    if on_activity is not None:
                        try:
                            on_activity({
                                "type": "model_switched",
                                "from": old_model,
                                "to": self.model,
                                "iteration": iteration,
                                "elapsed": time.time() - start_time,
                            })
                        except Exception:
                            pass
                    # Also push a system note into the conversation so the
                    # incoming model knows it's mid-run and what to focus on.
                    messages.append({
                        "role": "user",
                        "content": (
                            f"[orchestrator note] You have just been switched in mid-run "
                            f"(from {old_model} to {self.model}). The full conversation "
                            f"history above is yours to continue from. Pick up where the "
                            f"previous model left off — do not restart the search plan."
                        ),
                    })

                if on_activity is not None:
                    try:
                        on_activity({
                            "type": "iter_start",
                            "iteration": iteration,
                            "elapsed": time.time() - start_time,
                        })
                    except Exception:
                        pass

                try:
                    chat_kwargs = dict(
                        model=self.model,
                        messages=messages,
                        tools=TOOL_DEFINITIONS,
                    )
                    # gpt-oss emits reasoning ("harmony" channel) before tool calls,
                    # which Ollama's tool-call parser rejects. Disable thinking for it.
                    if "gpt-oss" in self.model.lower():
                        chat_kwargs["think"] = False
                    response = self._client.chat(**chat_kwargs)
                except TypeError:
                    # older ollama client without `think` kwarg
                    chat_kwargs.pop("think", None)
                    response = self._client.chat(**chat_kwargs)
                except Exception as chat_err:
                    err_text = str(chat_err)
                    self.logger.warning(f"chat error (iteration {iteration}): {err_text[:300]}")
                    # If the model produced prose instead of a tool call, push a
                    # corrective message and retry on the next iteration rather than
                    # killing the whole run.
                    messages.append({
                        "role": "user",
                        "content": (
                            "Your previous reply was rejected because it contained prose "
                            "instead of a clean tool call. Reply with EXACTLY ONE tool call "
                            "(web_search, fetch_page, store_dataset, or mark_search_complete) "
                            "and NO commentary, no chain-of-thought, no JSON wrappers."
                        ),
                    })
                    continue

                tool_calls = _extract_tool_calls(response.message)

                # Surface the model's free-text content as a "thinking" event so
                # the web UI can show what the agent is reasoning about. This is
                # the chunk of black-box reasoning the user wanted visible.
                content_text = (response.message.content or "").strip()
                if content_text and on_activity is not None:
                    try:
                        on_activity({
                            "type": "thinking",
                            "text": content_text[:4000],
                            "iteration": iteration,
                            "elapsed": time.time() - start_time,
                        })
                    except Exception:
                        pass

                if response.message.tool_calls:
                    messages.append(response.message)
                else:
                    messages.append({
                        "role": "assistant",
                        "content": response.message.content or "",
                    })

                if tool_calls:
                    consecutive_no_calls = 0
                    # Dedup tool calls within this single iteration — the model
                    # sometimes emits the same fetch_page 8x in one turn.
                    seen_within_iter: set[tuple[str, str]] = set()
                    deduped_calls = []
                    for call in tool_calls:
                        key = (
                            call["name"],
                            (call.get("arguments") or {}).get("url")
                            or (call.get("arguments") or {}).get("repo_url")
                            or (call.get("arguments") or {}).get("query")
                            or "",
                        )
                        if key in seen_within_iter:
                            continue
                        seen_within_iter.add(key)
                        deduped_calls.append(call)
                    tool_calls = deduped_calls
                    for call in tool_calls:
                        tool_name = call["name"]
                        tool_args = call["arguments"]
                        self.logger.info(f"  Tool: {tool_name}({list(tool_args.keys())})")
                        if on_activity is not None:
                            try:
                                on_activity({
                                    "type": "tool_call",
                                    "name": tool_name,
                                    "args": tool_args,
                                    "iteration": iteration,
                                    "elapsed": time.time() - start_time,
                                })
                            except Exception:
                                pass

                        if tool_name == "mark_search_complete":
                            stored_now = len(self.storage.get_datasets(query_id=query_id))
                            min_needed = DEPTH_MIN_DATASETS.get(depth, 25)
                            cooldown_until = getattr(tool_executor, "_mark_complete_cooldown_until", 0)
                            if iteration < cooldown_until:
                                self.logger.info(
                                    f"  Rejected mark_search_complete (cooldown until iter {cooldown_until})"
                                )
                                messages.append({
                                    "role": "tool",
                                    "content": json.dumps({
                                        "rejected": True,
                                        "reason": "cooldown",
                                        "message": (
                                            f"REJECTED: mark_search_complete is on cooldown until iteration "
                                            f"{cooldown_until}. Stop trying to end the search. Run more "
                                            f"web_search / read_pdf / read_github_readme calls to find new datasets."
                                        ),
                                    }, default=str),
                                })
                                continue
                            if stored_now < min_needed:
                                self.logger.info(
                                    f"  Rejected mark_search_complete: have {stored_now}/{min_needed}"
                                )
                                # Set a 20-iteration cooldown so the model doesn't spam this tool.
                                tool_executor._mark_complete_cooldown_until = iteration + 20
                                rejection = {
                                    "rejected": True,
                                    "stored": stored_now,
                                    "minimum_required": min_needed,
                                    "message": (
                                        f"REJECTED. You have only stored {stored_now} datasets but "
                                        f"depth {depth} requires at least {min_needed}. "
                                        f"mark_search_complete is now ON COOLDOWN for the next 20 iterations — "
                                        f"do NOT call it again until iteration {iteration + 20}. "
                                        f"Keep searching: try sources you have not yet hit, mine arxiv papers, "
                                        f"chase awesome-list READMEs."
                                    ),
                                }
                                messages.append({
                                    "role": "tool",
                                    "content": json.dumps(rejection, default=str),
                                })
                                continue
                            search_done = True

                        result = tool_executor.execute(tool_name, tool_args)
                        if on_activity is not None and isinstance(result, dict):
                            if result.get("rejected") or result.get("error") or (
                                tool_name == "store_dataset" and result.get("success") is False
                            ):
                                try:
                                    on_activity({
                                        "type": "tool_result",
                                        "name": tool_name,
                                        "status": "rejected" if result.get("rejected") else "error",
                                        "message": result.get("message") or result.get("error") or "",
                                        "iteration": iteration,
                                        "elapsed": time.time() - start_time,
                                    })
                                except Exception:
                                    pass
                        messages.append({
                            "role": "tool",
                            "content": json.dumps(result, default=str)[:4000],
                        })
                else:
                    consecutive_no_calls += 1
                    self.logger.info(f"  No tool calls (consecutive={consecutive_no_calls})")
                    if consecutive_no_calls >= 3:
                        messages.append({
                            "role": "user",
                            "content": (
                                "Stop talking. Call web_search now with a real query. "
                                f'For example: web_search(query="{subject} dataset site:figshare.com") '
                                f'or web_search(query="{subject} filetype:csv site:catalog.data.gov").'
                            ),
                        })
                        consecutive_no_calls = 0
                    else:
                        messages.append({
                            "role": "user",
                            "content": "Call a tool now. Use web_search, fetch_page, or store_dataset.",
                        })

                # Every N iterations, inject a progress reminder so the model
                # doesn't drift, get stuck on one source, or wind down too early.
                if iteration > 0 and iteration % PROGRESS_REMINDER_EVERY == 0 and not search_done:
                    stored_now = len(self.storage.get_datasets(query_id=query_id))
                    min_needed = DEPTH_MIN_DATASETS.get(depth, 25)
                    sources_seen = {
                        d.source for d in self.storage.get_datasets(query_id=query_id, limit=500)
                    }
                    ms = getattr(tool_executor, "mainstream_stored", 0)
                    alt = getattr(tool_executor, "alternative_stored", 0)
                    reminder = (
                        f"PROGRESS CHECK — iteration {iteration}. "
                        f"Stored {stored_now} (need {min_needed}). "
                        f"Mainstream:Alternative = {ms}:{alt} (target ratio 1:2 — alternative must be >= 2*mainstream). "
                        f"Sources hit: {sorted(sources_seen) or '[none yet]'}. "
                        f"Sources NOT YET hit — go search them now: "
                        f"arxiv.org papers (mine cited datasets), figshare.com, dataverse.harvard.edu, "
                        f"zenodo.org, catalog.data.gov, registry.opendata.aws, european-data.europa.eu, "
                        f"Ahmia/Tor2Web gateways, Wayback Machine archived OSINT forums, "
                        f"university lab pages. "
                        f"Append `-site:kaggle.com -site:huggingface.co -site:github.com` to broad queries. "
                        f"Try synonyms: aircraft → airplane/jet/fighter/UAV/drone. "
                        f"Do not stop. Call web_search now with site:/filetype:/inurl: operators."
                    )
                    messages.append({"role": "user", "content": reminder})

                if len(messages) > 44:
                    messages = messages[:2] + messages[-40:]

        except KeyboardInterrupt:
            self.logger.info("Interrupted by user — saving state")
            count = len(self.storage.get_datasets(query_id=query_id))
            elapsed = time.time() - start_time
            self.storage.update_query_status(query_id, "interrupted", datetime.utcnow())
            self.storage.update_datasets_found(query_id, count)
            query.status = "interrupted"
            query.datasets_found = count
            query.elapsed_seconds = elapsed
            query.iterations = iteration
            query.dedup_skipped_fetch = tool_executor.skipped_fetch_already_seen
            query.dedup_skipped_store = tool_executor.skipped_store_duplicate
            query.existing_at_start = existing_count_at_start
            return query

        status = "completed" if search_done else "interrupted"
        count = len(self.storage.get_datasets(query_id=query_id))
        elapsed = time.time() - start_time
        self.storage.update_query_status(query_id, status, datetime.utcnow())
        self.storage.update_datasets_found(query_id, count)
        self.logger.info(
            f"Search finished | status={status} | datasets={count} | "
            f"iterations={iteration} | elapsed={elapsed:.0f}s | "
            f"skipped_fetch={tool_executor.skipped_fetch_already_seen} | "
            f"skipped_store={tool_executor.skipped_store_duplicate}"
        )
        query.status = status
        query.datasets_found = count
        query.elapsed_seconds = elapsed
        query.iterations = iteration
        query.dedup_skipped_fetch = tool_executor.skipped_fetch_already_seen
        query.dedup_skipped_store = tool_executor.skipped_store_duplicate
        query.existing_at_start = existing_count_at_start
        if on_activity is not None:
            try:
                sources_seen = {d.source for d in self.storage.get_datasets(query_id=query_id, limit=500)}
                on_activity({
                    "type": "run_complete",
                    "status": status,
                    "stored": count,
                    "iteration": iteration,
                    "elapsed": elapsed,
                    "sources": list(sources_seen),
                    "mainstream": getattr(tool_executor, "mainstream_stored", 0),
                    "alternative": getattr(tool_executor, "alternative_stored", 0),
                })
            except Exception:
                pass
        return query
# feat/agent-summary: Refine agent-summary implementation
# feat/agent-summary: Add agent-summary config option
# feat/agent-summary: Test agent-summary edge cases
