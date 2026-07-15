import logging
from typing import Dict, List, Set
from collections import Counter
from math import log2

logger = logging.getLogger(__name__)


def get_source_distribution(storage_datasets) -> Dict[str, int]:
    """Count datasets per source."""
    src_counter: Counter = Counter()
    for ds in storage_datasets:
        src = ds.source.value if hasattr(ds.source, "value") else str(ds.source)
        src_counter[src] += 1
    return dict(src_counter.most_common())


def shannon_entropy(counts: Dict[str, int]) -> float:
    """Compute Shannon entropy as a diversity metric."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        if c == 0:
            continue
        p = c / total
        h -= p * log2(p)
    return round(h, 4)


def simpson_index(counts: Dict[str, int]) -> float:
    """Simpson's Diversity Index (1 - D). Higher = more diverse."""
    total = sum(counts.values())
    if total <= 1:
        return 1.0
    d = sum(c * (c - 1) for c in counts.values()) / (total * (total - 1))
    return round(1 - d, 4)


def source_diversity_report(storage_datasets) -> Dict:
    """Generate a full source diversity report with multiple metrics."""
    dist = get_source_distribution(storage_datasets)
    return {
        "source_distribution": dist,
        "num_sources": len(dist),
        "shannon_entropy": shannon_entropy(dist),
        "simpson_index": simpson_index(dist),
        "dominant_source": max(dist, key=dist.get) if dist else None,
        "dominant_fraction": round(max(dist.values()) / sum(dist.values()), 4) if dist else 0,
    }
