import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


def search_kaggle(query: str, limit: int = 20, competition_only: bool = False) -> List[dict]:
    """Search Kaggle for datasets matching *query*.

    Requires kagglehub or kaggle API credentials.
    Falls back to web scraping if API is unavailable.
    """
    logger.info("search_kaggle(%r, limit=%d, competition_only=%s)", query, limit, competition_only)
    results = []
    try:
        from kagglehub import kagglehub
        datasets = kagglehub.dataset_search(query)
        for ds in datasets[:limit]:
            results.append({
                "name": ds.title,
                "description": ds.description,
                "url": f"https://kaggle.com/datasets/{ds.ref}",
                "tags": ds.tags,
                "downloads": ds.download_count,
                "size": ds.total_size,
                "source": "kaggle",
            })
    except ImportError:
        logger.info("kagglehub not installed, trying web scrape fallback")
        try:
            import requests
            from bs4 import BeautifulSoup
            resp = requests.get(
                "https://www.kaggle.com/datasets",
                params={"search": query},
                timeout=15,
            )
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, "html.parser")
                for card in soup.select(".dataset-card")[:limit]:
                    title_el = card.select_one(".dataset-title")
                    if title_el:
                        results.append({
                            "name": title_el.text.strip(),
                            "url": "https://kaggle.com" + title_el.get("href", ""),
                            "source": "kaggle",
                        })
        except Exception as exc:
            logger.warning("Kaggle web scrape failed: %s", exc)
    except Exception as exc:
        logger.warning("Kaggle search failed: %s", exc)
    return results


def get_kaggle_competition_datasets(competition: str) -> List[dict]:
    """Get datasets associated with a specific Kaggle competition."""
    logger.info("get_kaggle_competition_datasets(%r)", competition)
    return search_kaggle(competition, limit=50)


def get_kaggle_trending(limit: int = 10) -> List[dict]:
    """Fetch trending Kaggle datasets."""
    logger.info("get_kaggle_trending(limit=%d)", limit)
    return search_kaggle("", limit=limit)
