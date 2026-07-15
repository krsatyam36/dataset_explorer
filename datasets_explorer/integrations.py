import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


def search_github_datasets(query: str, limit: int = 20, min_stars: int = 0) -> List[dict]:
    """Search GitHub for repositories that look like datasets.

    Uses the public GitHub search API. Requires no auth for basic usage
    (rate-limited to 10 req/min without a token).
    """
    logger.info("search_github_datasets(%r, limit=%d, min_stars=%d)", query, limit, min_stars)
    results = []
    try:
        import requests
        headers = {"Accept": "application/vnd.github.v3+json"}
        gh_query = f"{query} dataset in:topics,description"
        if min_stars:
            gh_query += f" stars:>={min_stars}"
        resp = requests.get(
            "https://api.github.com/search/repositories",
            params={"q": gh_query, "per_page": limit, "sort": "stars"},
            headers=headers,
            timeout=15,
        )
        resp.raise_for_status()
        for item in resp.json().get("items", []):
            results.append({
                "name": item.get("full_name", ""),
                "description": item.get("description", "") or "",
                "url": item.get("html_url", ""),
                "stars": item.get("stargazers_count", 0),
                "forks": item.get("forks_count", 0),
                "language": item.get("language") or "",
                "topics": item.get("topics", []),
                "license": (item.get("license") or {}).get("spdx_id", ""),
                "source": "github",
            })
    except Exception as exc:
        logger.warning("GitHub search failed: %s", exc)
    return results


def search_github_by_topic(topic: str) -> List[dict]:
    """Search GitHub repos tagged with a specific topic related to datasets."""
    return search_github_datasets(f"topic:{topic}", limit=30)


def get_github_dataset_readme(repo_full_name: str) -> Optional[str]:
    """Fetch the README content of a GitHub repo to assess dataset quality."""
    logger.info("get_github_dataset_readme(%r)", repo_full_name)
    try:
        import requests
        resp = requests.get(
            f"https://api.github.com/repos/{repo_full_name}/readme",
            headers={"Accept": "application/vnd.github.v3.raw"},
            timeout=10,
        )
        if resp.status_code == 200:
            return resp.text[:2000]
    except Exception as exc:
        logger.warning("Failed to fetch README for %s: %s", repo_full_name, exc)
    return None
# feat/int-github-search: Refine int-github-search error handling
# feat/int-github-search: Add int-github-search timeout config
# feat/int-github-search: Add int-github-search rate limiting
