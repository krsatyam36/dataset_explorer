import logging
from typing import List, Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


def enrich_with_whois(urls: List[str]) -> Dict[str, dict]:
    """Perform WHOIS lookups for a list of dataset URLs.

    Returns a dict mapping each URL to its domain info: registrar, org,
    creation date, etc.
    """
    logger.info("enrich_with_whois(%d urls)", len(urls))
    results = {}
    try:
        import whois as whois_lib
        seen_domains = set()
        for url in urls:
            from urllib.parse import urlparse
            domain = urlparse(url).netloc
            if domain in seen_domains:
                continue
            seen_domains.add(domain)
            try:
                w = whois_lib.whois(domain)
                results[url] = {
                    "domain": domain,
                    "registrar": w.registrar,
                    "org": w.org,
                    "creation_date": str(w.creation_date) if w.creation_date else None,
                    "expiration_date": str(w.expiration_date) if w.expiration_date else None,
                    "country": w.country,
                    "name": w.name,
                }
            except Exception:
                results[url] = {"domain": domain, "error": "lookup failed"}
    except ImportError:
        logger.warning("whois library not installed")
    return results


def get_registrar_stats(storage_datasets) -> Dict[str, int]:
    """Compute registrar distribution from stored datasets.

    Args:
        storage_datasets: iterable of Dataset objects from storage.
    Returns:
        dict mapping registrar -> count.
    """
    urls = [d.url for d in storage_datasets if d.url]
    whois_data = enrich_with_whois(urls)
    registrars = {}
    for info in whois_data.values():
        reg = info.get("registrar") or "unknown"
        registrars[reg] = registrars.get(reg, 0) + 1
    return registrars


def get_domain_country_stats(storage_datasets) -> Dict[str, int]:
    """Compute country distribution from WHOIS lookups on stored datasets."""
    urls = [d.url for d in storage_datasets if d.url]
    whois_data = enrich_with_whois(urls)
    countries = {}
    for info in whois_data.values():
        c = info.get("country") or "unknown"
        countries[c] = countries.get(c, 0) + 1
    return countries
# feat/analytics-whois: Refine analytics-whois output format
# feat/analytics-whois: Add analytics-whois edge case handling
