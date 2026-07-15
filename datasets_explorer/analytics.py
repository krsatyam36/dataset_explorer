import logging
from typing import Dict, List, Optional
from collections import Counter
from datetime import datetime

logger = logging.getLogger(__name__)


def get_discovery_timeline(storage_datasets) -> Dict[str, int]:
    """Count datasets discovered per calendar day."""
    daily: Counter = Counter()
    for ds in storage_datasets:
        if ds.discovered_at:
            day = ds.discovered_at.strftime("%Y-%m-%d")
            daily[day] += 1
    return dict(sorted(daily.items()))


def get_date_range_coverage(storage_datasets) -> Dict[str, int]:
    """Count how many datasets have date_range_start populated per year."""
    yearly: Counter = Counter()
    for ds in storage_datasets:
        if ds.date_range_start:
            year = ds.date_range_start[:4]
            yearly[year] += 1
    return dict(sorted(yearly.items()))


def get_collection_rate(storage_datasets) -> Dict:
    """Compute overall collection velocity metrics."""
    timeline = get_discovery_timeline(storage_datasets)
    days = list(timeline.keys())
    if not days:
        return {"total": 0, "days_active": 0, "avg_per_day": 0}
    total = sum(timeline.values())
    n_days = len(days)
    return {
        "total": total,
        "days_active": n_days,
        "avg_per_day": round(total / n_days, 2) if n_days else 0,
        "max_in_day": max(timeline.values()),
        "date_range": f"{days[0]} -> {days[-1]}" if len(days) > 1 else days[0],
    }
