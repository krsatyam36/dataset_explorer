import logging
from typing import Dict, List, Optional
from collections import Counter

logger = logging.getLogger(__name__)

# Common open-source licenses with their commercial-friendliness.
LICENSE_COMMERCIAL_OK = {
    "cc0-1.0", "mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause",
    "unlicense", "wtfpl", "isc",
}
LICENSE_COMMERCIAL_MAYBE = {
    "cc-by-4.0", "cc-by-sa-4.0", "lgpl-2.1", "lgpl-3.0", "mpl-2.0",
}
LICENSE_COMMERCIAL_NO = {
    "cc-by-nc-4.0", "cc-by-nc-sa-4.0", "gpl-2.0", "gpl-3.0", "agpl-3.0",
    "odbl", "unknown",
}


def get_license_distribution(storage_datasets) -> Dict[str, int]:
    """Count how many datasets use each license."""
    lic_counter: Counter = Counter()
    for ds in storage_datasets:
        lic = (ds.license_spdx or ds.license or "unknown").lower().strip()
        lic_counter[lic] += 1
    return dict(lic_counter.most_common())


def get_license_compliance_summary(storage_datasets) -> Dict[str, int]:
    """Categorize datasets as commercial-ok, commercial-maybe, or commercial-no."""
    categories: Dict[str, int] = {"commercial_ok": 0, "commercial_maybe": 0, "commercial_no": 0}
    for ds in storage_datasets:
        lic = (ds.license_spdx or ds.license or "unknown").lower().strip()
        if lic in LICENSE_COMMERCIAL_OK:
            categories["commercial_ok"] += 1
        elif lic in LICENSE_COMMERCIAL_MAYBE:
            categories["commercial_maybe"] += 1
        else:
            categories["commercial_no"] += 1
    return categories


def get_datasets_with_missing_license(storage_datasets) -> List[int]:
    """Return IDs of datasets that are missing license information."""
    return [d.id for d in storage_datasets if not d.license and not d.license_spdx and d.id is not None]
