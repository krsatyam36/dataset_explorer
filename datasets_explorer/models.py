from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum


class DatasetFormat(str, Enum):
    JPEG = "jpeg"
    TIFF = "tiff"
    GEOTIFF = "geotiff"
    COG = "cog"
    NPY = "npy"
    PNG = "png"
    CSV = "csv"
    JSON = "json"
    HDF5 = "hdf5"
    SHAPEFILE = "shapefile"
    UNKNOWN = "unknown"


class DatasetSource(str, Enum):
    KAGGLE = "kaggle"
    HUGGINGFACE = "huggingface"
    ROBOFLOW = "roboflow"
    IEEE_DATAPORT = "ieee_dataport"
    NASA_EARTHDATA = "nasa_earthdata"
    USGS = "usgs"
    COPERNICUS = "copernicus"
    GITHUB = "github"
    ZENODO = "zenodo"
    PAPERS_WITH_CODE = "papers_with_code"
    OPEN_AERIAL_MAP = "open_aerial_map"
    GENERIC = "generic"


class Dataset(BaseModel):
    reproducibility_score: Optional[float] = None  # 0-1: how reproducible (has code + data + instructions)
    id: Optional[int] = None
    name: str
    url: str
    download_url: Optional[str] = None
    source: DatasetSource = DatasetSource.GENERIC
    formats: List[str] = []
    size_bytes: Optional[int] = None
    size_human: Optional[str] = None
    license: Optional[str] = None
    description: str = ""
    tags: List[str] = []
    date_range_start: Optional[str] = None
    date_range_end: Optional[str] = None
    num_samples: Optional[int] = None
    relevance_score: float = Field(0.0, ge=0.0, le=1.0)
    relevance_reasoning: str = ""
    query_id: Optional[int] = None
    discovered_at: datetime = Field(default_factory=datetime.utcnow)
    reviewed: bool = False
    notes: str = ""
    # ---- Trust Card v1 (License + Provenance) -----------------------------
    # Filled in opportunistically by the LLM during store_dataset based on the
    # page text it has already read. None when not extractable.
    license_spdx: Optional[str] = None       # e.g. "CC-BY-4.0", "MIT", "unknown"
    license_commercial_ok: Optional[bool] = None  # may we use it commercially?
    doi: Optional[str] = None                # e.g. "10.5281/zenodo.7331974"
    authors: List[str] = []                  # paper authors / dataset creators
    institution: Optional[str] = None        # hosting org / lab
    country: Optional[str] = None            # ISO country of origin (best-effort)


class SearchQuery(BaseModel):
    id: Optional[int] = None
    raw_query: str
    parsed_intent: str = ""
    format_filters: List[str] = []
    keywords: List[str] = []
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    status: str = "running"
    datasets_found: int = 0
    session_log: str = ""
    # Runtime stats — populated by the agent on completion (not persisted to DB).
    elapsed_seconds: Optional[float] = None
    iterations: Optional[int] = None
    dedup_skipped_fetch: int = 0
    dedup_skipped_store: int = 0
    existing_at_start: int = 0
