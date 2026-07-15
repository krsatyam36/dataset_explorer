import logging
from typing import Dict, List, Tuple
from collections import Counter

logger = logging.getLogger(__name__)


def get_format_distribution(storage_datasets) -> Dict[str, int]:
    """Compute the distribution of dataset formats across stored datasets.

    Args:
        storage_datasets: iterable of Dataset objects from storage.
    Returns:
        dict mapping format extension -> count of datasets using it.
    """
    format_counter: Counter = Counter()
    for ds in storage_datasets:
        if ds.formats:
            for fmt in ds.formats:
                format_counter[fmt.lower()] += 1
        else:
            format_counter["unknown"] += 1
    return dict(format_counter.most_common())


def get_format_co_occurrence(storage_datasets) -> List[Tuple[str, str, int]]:
    """Find format pairs that frequently appear together (co-occurrence)."""
    pairs: Counter = Counter()
    for ds in storage_datasets:
        fmts = sorted(set(f.lower() for f in ds.formats))
        for i in range(len(fmts)):
            for j in range(i + 1, len(fmts)):
                pairs[(fmts[i], fmts[j])] += 1
    return [(a, b, count) for (a, b), count in pairs.most_common(20)]


def get_format_size_correlation(storage_datasets) -> Dict[str, int]:
    """Compute average size (bytes) per format for datasets that report size."""
    fmt_sizes: Dict[str, list] = {}
    for ds in storage_datasets:
        if ds.size_bytes and ds.formats:
            for fmt in ds.formats:
                fmt_sizes.setdefault(fmt.lower(), []).append(ds.size_bytes)
    return {fmt: int(sum(sizes) / len(sizes)) for fmt, sizes in fmt_sizes.items()}
# feat/analytics-format-stats: Refine analytics-format-stats output format
# feat/analytics-format-stats: Add analytics-format-stats edge case handling
