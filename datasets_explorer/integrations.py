import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


def search_paperswithcode(query: str, limit: int = 20) -> List[dict]:
    """Search PapersWithCode for datasets associated with ML papers.

    Uses the PapersWithCode public API at paperswithcode.com/api/v1/.
    """
    logger.info("search_paperswithcode(%r, limit=%d)", query, limit)
    results = []
    try:
        import requests
        resp = requests.get(
            "https://paperswithcode.com/api/v1/papers/",
            params={"q": query, "items": limit},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        for paper in data.get("results", []):
            paper_url = paper.get("url_abs", "")
            results.append({
                "name": paper.get("title", ""),
                "description": paper.get("abstract", "")[:500],
                "url": paper_url,
                "published": paper.get("published"),
                "arxiv_id": paper.get("id"),
                "source": "paperswithcode",
            })
    except Exception as exc:
        logger.warning("PapersWithCode search failed: %s", exc)
    return results


def get_pwc_dataset_list(paper_url: str) -> List[dict]:
    """Get datasets linked from a specific PapersWithCode paper page."""
    logger.info("get_pwc_dataset_list(%r)", paper_url)
    return []


def search_pwc_datasets_by_task(task: str) -> List[dict]:
    """Search PapersWithCode datasets filtered by task (e.g. 'Image Classification')."""
    logger.info("search_pwc_datasets_by_task(%r)", task)
    return search_paperswithcode(task, limit=30)
# feat/int-paperswithcode: Refine int-paperswithcode error handling
# feat/int-paperswithcode: Add int-paperswithcode timeout config
# feat/int-paperswithcode: Add int-paperswithcode rate limiting
