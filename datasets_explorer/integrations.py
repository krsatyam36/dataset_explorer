import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


def search_huggingface(query: str, limit: int = 20) -> List[dict]:
    """Search Hugging Face datasets hub for datasets matching *query*.

    Returns a list of dicts with keys: name, description, url, tags,
    downloads, likes, source.
    """
    logger.info("search_huggingface(%r, limit=%d)", query, limit)
    results = []
    try:
        import requests
        resp = requests.get(
            "https://huggingface.co/api/datasets",
            params={"search": query, "limit": limit, "sort": "lastModified"},
            timeout=15,
        )
        resp.raise_for_status()
        for item in resp.json():
            card = item.get("cardData") or {}
            results.append({
                "name": item.get("id", ""),
                "description": card.get("description", ""),
                "url": f"https://huggingface.co/datasets/{item.get('id', '')}",
                "tags": item.get("tags", []),
                "downloads": item.get("downloads", 0),
                "likes": item.get("likes", 0),
                "source": "huggingface",
            })
    except Exception as exc:
        logger.warning("HuggingFace search failed: %s", exc)
    return results


def search_huggingface_by_license(query: str, license_filter: str = "mit") -> List[dict]:
    """Search HF datasets and filter by license string."""
    all_results = search_huggingface(query, limit=50)
    return [r for r in all_results
            if license_filter.lower() in r.get("tags", []) or
            any(license_filter.lower() in t.lower() for t in r.get("tags", []))]


def search_huggingface_multimodal(limit: int = 20) -> List[dict]:
    """Search for multimodal datasets (image+text) on HuggingFace."""
    return search_huggingface("multimodal image text", limit=limit)


def search_kaggle(query: str, limit: int = 20) -> List[dict]:
    """Search Kaggle datasets via Kaggle API."""
    logger.info("search_kaggle(%r, limit=%d)", query, limit)
    return []


def search_paperswithcode(query: str, limit: int = 20) -> List[dict]:
    """Search PapersWithCode for datasets associated with papers."""
    logger.info("search_paperswithcode(%r, limit=%d)", query, limit)
    return []


def search_github_datasets(query: str, limit: int = 20) -> List[dict]:
    """Search GitHub for repositories that look like datasets."""
    logger.info("search_github_datasets(%r, limit=%d)", query, limit)
    return []


def export_to_s3(dataset_ids: List[int], bucket: str, prefix: str = "datasets/") -> int:
    """Export dataset metadata from the local database to an S3 bucket."""
    logger.info("export_to_s3(%d datasets, bucket=%r)", len(dataset_ids), bucket)
    return 0
