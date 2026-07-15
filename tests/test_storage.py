"""Tests for the storage layer."""

import json
from datetime import datetime
from datasets_explorer.storage import Storage
from datasets_explorer.models import Dataset, SearchQuery, DatasetSource


class TestStorage:
    def test_save_and_get_query(self, storage: Storage):
        q = SearchQuery(raw_query="test query")
        qid = storage.save_query(q)
        assert qid > 0
        queries = storage.list_queries()
        assert len(queries) >= 1
        assert queries[0].raw_query == "test query"

    def test_save_and_get_dataset(self, storage: Storage):
        d = Dataset(
            name="Test Dataset",
            url="https://example.com/dataset/1",
            relevance_score=0.95,
            relevance_reasoning="excellent",
            query_id=1,
        )
        did = storage.save_dataset(d)
        assert did > 0
        datasets = storage.get_datasets(query_id=1)
        assert len(datasets) == 1
        assert datasets[0].name == "Test Dataset"
        assert datasets[0].url == "https://example.com/dataset/1"

    def test_dataset_dedup_by_url(self, storage: Storage):
        d1 = Dataset(name="A", url="https://example.com/ds", relevance_score=0.8, relevance_reasoning="ok", query_id=1)
        d2 = Dataset(name="A (dupe)", url="https://example.com/ds", relevance_score=0.6, relevance_reasoning="worse", query_id=1)
        storage.save_dataset(d1)
        storage.save_dataset(d2)
        datasets = storage.get_datasets(query_id=1)
        assert len(datasets) == 1  # deduped
        assert datasets[0].relevance_score == 0.8  # higher score kept

    def test_get_datasets_by_relevance(self, storage: Storage):
        for i, score in enumerate([0.3, 0.9, 0.6]):
            storage.save_dataset(Dataset(
                name=f"DS {i}",
                url=f"https://example.com/ds/{i}",
                relevance_score=score,
                relevance_reasoning="test",
                query_id=1,
            ))
        datasets = storage.get_datasets(query_id=1)
        assert [d.relevance_score for d in datasets] == [0.9, 0.6, 0.3]

    def test_get_datasets_min_relevance(self, storage: Storage):
        for i, score in enumerate([0.3, 0.9, 0.6]):
            storage.save_dataset(Dataset(
                name=f"DS {i}",
                url=f"https://example.com/ds/{i}",
                relevance_score=score,
                relevance_reasoning="test",
                query_id=1,
            ))
        datasets = storage.get_datasets(query_id=1, min_relevance=0.5)
        assert len(datasets) == 2
        assert all(d.relevance_score >= 0.5 for d in datasets)

    def test_trust_card_fields_persisted(self, storage: Storage):
        d = Dataset(
            name="Trust Card Test",
            url="https://example.com/tc",
            relevance_score=0.8,
            relevance_reasoning="good",
            query_id=1,
            license_spdx="CC-BY-4.0",
            license_commercial_ok=True,
            doi="10.5281/zenodo.12345",
            authors=["Alice", "Bob"],
            institution="NASA",
            country="US",
        )
        storage.save_dataset(d)
        datasets = storage.get_datasets(query_id=1)
        assert len(datasets) == 1
        ds = datasets[0]
        assert ds.license_spdx == "CC-BY-4.0"
        assert ds.license_commercial_ok is True
        assert ds.doi == "10.5281/zenodo.12345"
        assert ds.authors == ["Alice", "Bob"]
        assert ds.institution == "NASA"
        assert ds.country == "US"

    def test_get_all_urls(self, storage: Storage):
        storage.save_dataset(Dataset(
            name="A", url="https://a.com", download_url="https://a.com/download",
            relevance_score=0.5, relevance_reasoning="test", query_id=1,
        ))
        storage.save_dataset(Dataset(
            name="B", url="https://b.com",
            relevance_score=0.5, relevance_reasoning="test", query_id=1,
        ))
        urls = storage.get_all_urls()
        assert "https://a.com" in urls
        assert "https://a.com/download" in urls
        assert "https://b.com" in urls

    def test_get_all_urls_with_names(self, storage: Storage):
        storage.save_dataset(Dataset(
            name="Test DS", url="https://test.com",
            relevance_score=0.9, relevance_reasoning="best", query_id=1,
        ))
        items = storage.get_all_urls_with_names()
        assert len(items) == 1
        assert items[0] == ("https://test.com", "Test DS")

    def test_update_query_status(self, storage: Storage):
        qid = storage.save_query(SearchQuery(raw_query="test"))
        storage.update_query_status(qid, "completed", datetime.utcnow())
        queries = storage.list_queries()
        match = [q for q in queries if q.id == qid]
        assert len(match) == 1
        assert match[0].status == "completed"
        assert match[0].completed_at is not None

    def test_update_datasets_found(self, storage: Storage):
        qid = storage.save_query(SearchQuery(raw_query="test"))
        storage.update_datasets_found(qid, 42)
        queries = storage.list_queries()
        match = [q for q in queries if q.id == qid]
        assert match[0].datasets_found == 42

    def test_update_dataset_notes(self, storage: Storage):
        did = storage.save_dataset(Dataset(
            name="Notes Test", url="https://example.com/notes",
            relevance_score=0.5, relevance_reasoning="test", query_id=1,
        ))
        storage.update_dataset_notes(did, "Downloaded and reviewed")
        datasets = storage.get_datasets()
        match = [d for d in datasets if d.id == did]
        assert match[0].notes == "Downloaded and reviewed"

    def test_mark_reviewed(self, storage: Storage):
        did = storage.save_dataset(Dataset(
            name="Review Test", url="https://example.com/review",
            relevance_score=0.5, relevance_reasoning="test", query_id=1,
        ))
        storage.mark_reviewed(did, True)
        datasets = storage.get_datasets()
        match = [d for d in datasets if d.id == did]
        assert match[0].reviewed is True

    def test_filter_by_format(self, storage: Storage):
        storage.save_dataset(Dataset(
            name="GeoTIFF Dataset", url="https://example.com/geo",
            formats=["geotiff", "tiff"],
            relevance_score=0.8, relevance_reasoning="test", query_id=1,
        ))
        storage.save_dataset(Dataset(
            name="JPEG Dataset", url="https://example.com/jpg",
            formats=["jpeg"],
            relevance_score=0.6, relevance_reasoning="test", query_id=1,
        ))
        datasets = storage.get_datasets(query_id=1, formats=["geotiff"])
        assert len(datasets) == 1
        assert datasets[0].name == "GeoTIFF Dataset"

    def test_filter_by_reviewed(self, storage: Storage):
        d1 = Dataset(name="A", url="https://a.com", reviewed=True, relevance_score=0.5, relevance_reasoning="test", query_id=1)
        d2 = Dataset(name="B", url="https://b.com", reviewed=False, relevance_score=0.5, relevance_reasoning="test", query_id=1)
        storage.save_dataset(d1)
        storage.save_dataset(d2)
        reviewed = storage.get_datasets(reviewed=True)
        assert all(d.reviewed for d in reviewed)
        unreviewed = storage.get_datasets(reviewed=False)
        assert all(not d.reviewed for d in unreviewed)
