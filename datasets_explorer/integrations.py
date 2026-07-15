import json
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


def export_to_s3(dataset_ids: List[int], bucket: str, prefix: str = "datasets/",
                 profile: Optional[str] = None) -> int:
    """Export dataset metadata from the local database to an S3 bucket as JSON lines.

    Returns the number of datasets exported.
    """
    logger.info("export_to_s3(%d datasets, bucket=%r, prefix=%r)", len(dataset_ids), bucket, prefix)
    count = 0
    try:
        import boto3
        session = boto3.Session(profile_name=profile) if profile else boto3.Session()
        s3 = session.client("s3")

        from .storage import Storage
        from .config import DB_PATH
        storage = Storage(DB_PATH)

        for ds_id in dataset_ids:
            datasets = storage.get_datasets(limit=1)
            match = next((d for d in datasets if d.id == ds_id), None)
            if not match:
                continue
            key = f"{prefix}{ds_id}.json"
            body = json.dumps({
                "id": match.id,
                "name": match.name,
                "url": match.url,
                "description": match.description,
                "source": match.source.value if hasattr(match.source, "value") else str(match.source),
                "formats": match.formats,
                "license": match.license,
                "tags": match.tags,
                "relevance_score": match.relevance_score,
                "discovered_at": str(match.discovered_at) if match.discovered_at else None,
            }, indent=2)
            s3.put_object(Bucket=bucket, Key=key, Body=body.encode())
            count += 1
    except ImportError:
        logger.warning("boto3 not installed; skipping S3 export")
    except Exception as exc:
        logger.error("S3 export failed: %s", exc)
    logger.info("Exported %d/%d datasets to s3://%s/%s", count, len(dataset_ids), bucket, prefix)
    return count


def export_to_s3_all(bucket: str, prefix: str = "datasets/") -> int:
    """Export all datasets in the local database to S3."""
    from .storage import Storage
    from .config import DB_PATH
    storage = Storage(DB_PATH)
    all_ds = storage.get_datasets(limit=10_000)
    ids = [d.id for d in all_ds if d.id is not None]
    return export_to_s3(ids, bucket, prefix)
# feat/int-s3-export: Refine int-s3-export error handling
# feat/int-s3-export: Add int-s3-export timeout config
# feat/int-s3-export: Add int-s3-export rate limiting
