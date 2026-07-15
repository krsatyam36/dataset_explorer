"""Tests for Pydantic models."""

from datasets_explorer.models import Dataset, SearchQuery, DatasetFormat, DatasetSource


class TestDataset:
    def test_minimal_dataset(self):
        d = Dataset(name="Test", url="https://example.com/ds", relevance_score=0.5, relevance_reasoning="ok")
        assert d.id is None
        assert d.name == "Test"
        assert d.url == "https://example.com/ds"
        assert d.relevance_score == 0.5
        assert d.formats == []
        assert d.tags == []
        assert d.reviewed is False
        assert d.notes == ""

    def test_score_bounds(self):
        d = Dataset(name="A", url="https://a.com", relevance_score=0.0, relevance_reasoning="worst")
        assert d.relevance_score == 0.0
        d2 = Dataset(name="B", url="https://b.com", relevance_score=1.0, relevance_reasoning="best")
        assert d2.relevance_score == 1.0

    def test_trust_card_defaults(self):
        d = Dataset(name="TC", url="https://example.com/tc", relevance_score=0.5, relevance_reasoning="ok")
        assert d.license_spdx is None
        assert d.license_commercial_ok is None
        assert d.doi is None
        assert d.authors == []
        assert d.institution is None
        assert d.country is None

    def test_trust_card_values(self):
        d = Dataset(
            name="TC Full", url="https://example.com/tc",
            relevance_score=0.9, relevance_reasoning="good",
            license_spdx="MIT",
            license_commercial_ok=True,
            doi="10.5281/zenodo.12345",
            authors=["Author One"],
            institution="Test Lab",
            country="DE",
        )
        assert d.license_spdx == "MIT"
        assert d.license_commercial_ok is True
        assert d.doi == "10.5281/zenodo.12345"
        assert d.authors == ["Author One"]
        assert d.institution == "Test Lab"
        assert d.country == "DE"


class TestSearchQuery:
    def test_minimal_query(self):
        q = SearchQuery(raw_query="find me datasets")
        assert q.raw_query == "find me datasets"
        assert q.status == "running"
        assert q.format_filters == []
        assert q.keywords == []
        assert q.datasets_found == 0

    def test_full_query(self):
        q = SearchQuery(
            raw_query="satellite imagery",
            parsed_intent="find satellite imagery datasets",
            format_filters=["geotiff", "tiff"],
            keywords=["satellite", "aerial"],
            status="completed",
            datasets_found=10,
        )
        assert q.status == "completed"
        assert q.datasets_found == 10
        assert "geotiff" in q.format_filters

    def test_query_id_assignment(self):
        q = SearchQuery(raw_query="test", id=42)
        assert q.id == 42


class TestEnums:
    def test_dataset_format_values(self):
        assert DatasetFormat.GEOTIFF.value == "geotiff"
        assert DatasetFormat.JPEG.value == "jpeg"
        assert DatasetFormat.TIFF.value == "tiff"
        assert DatasetFormat.UNKNOWN.value == "unknown"

    def test_dataset_source_values(self):
        assert DatasetSource.KAGGLE.value == "kaggle"
        assert DatasetSource.ZENODO.value == "zenodo"
        assert DatasetSource.GENERIC.value == "generic"
