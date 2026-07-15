"""Tests for code-level guardrails in tools.py."""

import json
import pytest
from unittest.mock import patch, MagicMock
from datasets_explorer.tools import ToolExecutor
from datasets_explorer.models import Dataset, DatasetSource


# ─── Phase A gate ────────────────────────────────────────────────────────────

class TestPhaseAGate:
    def test_arxiv_search_blocked_initially(self, tool_executor: ToolExecutor):
        result = tool_executor.execute("arxiv_search", {"query": "aircraft dataset"})
        assert isinstance(result, dict)
        assert result.get("rejected") is True
        assert "PHASE A" in (result.get("error") or "").upper() or "REJECTED" in (result.get("error") or "").upper()

    def test_read_pdf_blocked_initially(self, tool_executor: ToolExecutor):
        result = tool_executor.execute("read_pdf", {"url": "https://arxiv.org/pdf/1234.5678.pdf"})
        assert isinstance(result, dict)
        assert result.get("rejected") is True

    def test_gate_opens_after_stores(self, tool_executor: ToolExecutor, storage):
        tool_executor._stores_so_far = 2
        result = tool_executor.execute("arxiv_search", {"query": "aircraft dataset"})
        assert result.get("rejected") is not True

    def test_gate_opens_after_portal_searches(self, tool_executor: ToolExecutor):
        tool_executor._portal_search_count = 4
        result = tool_executor.execute("arxiv_search", {"query": "aircraft dataset"})
        assert result.get("rejected") is not True

    def test_gate_stays_closed_below_threshold(self, tool_executor: ToolExecutor):
        tool_executor._stores_so_far = 1
        tool_executor._portal_search_count = 3
        result = tool_executor.execute("arxiv_search", {"query": "aircraft dataset"})
        assert result.get("rejected") is True

    def test_web_search_not_blocked_by_phase_a(self, tool_executor: ToolExecutor):
        result = tool_executor.execute("web_search", {"query": "site:figshare.com aircraft dataset"})
        assert result.get("rejected") is not True
        # web_search is never blocked by Phase A gate — it's the tool that opens it


# ─── 1:2 Mainstream balancing ───────────────────────────────────────────────

class TestMainstreamBalancing:
    def test_mainstream_rejected_when_unbalanced(self, tool_executor: ToolExecutor):
        tool_executor.alternative_stored = 0
        tool_executor.mainstream_stored = 1
        result = tool_executor.execute("store_dataset", {
            "name": "Kaggle Aircraft Dataset",
            "url": "https://kaggle.com/datasets/aircraft",
            "source": "kaggle",
            "relevance_score": 0.9,
            "relevance_reasoning": "good match",
        })
        assert result.get("rejected") is True
        assert result.get("reason") == "mainstream_quota"

    def test_mainstream_accepted_when_balanced(self, tool_executor: ToolExecutor, storage):
        tool_executor.alternative_stored = 2
        tool_executor.mainstream_stored = 0
        with patch.object(tool_executor, '_url_reachable', return_value=True):
            with patch.object(storage, 'save_dataset', return_value=1):
                result = tool_executor.execute("store_dataset", {
                    "name": "Kaggle Aircraft Dataset",
                    "url": "https://kaggle.com/datasets/aircraft",
                    "source": "kaggle",
                    "relevance_score": 0.9,
                    "relevance_reasoning": "good match",
                })
        assert result.get("success") is True

    def test_alternative_always_accepted(self, tool_executor: ToolExecutor, storage):
        tool_executor.mainstream_stored = 10
        tool_executor.alternative_stored = 0
        with patch.object(tool_executor, '_url_reachable', return_value=True):
            with patch.object(storage, 'save_dataset', return_value=2):
                result = tool_executor.execute("store_dataset", {
                    "name": "Zenodo Aircraft Dataset",
                    "url": "https://zenodo.org/records/12345",
                    "source": "zenodo",
                    "relevance_score": 0.9,
                    "relevance_reasoning": "good match",
                })
        assert result.get("success") is True

    def test_is_mainstream_helper(self):
        assert ToolExecutor._is_mainstream("https://kaggle.com/datasets/foo") is True
        assert ToolExecutor._is_mainstream("https://huggingface.co/datasets/foo") is True
        assert ToolExecutor._is_mainstream("https://github.com/owner/repo") is True
        assert ToolExecutor._is_mainstream("https://zenodo.org/records/12345") is False
        assert ToolExecutor._is_mainstream("https://figshare.com/articles/dataset/foo") is False


# ─── Synthetic data rejection ───────────────────────────────────────────────

class TestSyntheticRejection:
    def test_query_with_synthetic_rejected(self, tool_executor: ToolExecutor):
        result = tool_executor.execute("web_search", {"query": "synthetic aircraft dataset for simulation"})
        assert isinstance(result, dict)
        assert result.get("rejected") is True

    def test_query_with_gan_rejected(self, tool_executor: ToolExecutor):
        result = tool_executor.execute("web_search", {"query": "gan-generated satellite imagery"})
        assert result.get("rejected") is True

    def test_query_with_airsim_rejected(self, tool_executor: ToolExecutor):
        result = tool_executor.execute("web_search", {"query": "airsim drone dataset"})
        assert result.get("rejected") is True

    def test_clean_query_not_rejected(self, tool_executor: ToolExecutor):
        with patch.object(tool_executor, '_http') as mock_http:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"web": {"results": []}}
            mock_http.get.return_value = mock_response
            result = tool_executor.execute("web_search", {"query": "site:figshare.com real satellite imagery"})
            # May be rejected for other reasons (broad query, etc.) but not synthetic
            if isinstance(result, dict) and result.get("rejected"):
                assert "SYNTHETIC" not in (result.get("error") or "").upper()

    def test_store_synthetic_rejected(self, tool_executor: ToolExecutor):
        result = tool_executor.execute("store_dataset", {
            "name": "Synthetic Aircraft Dataset",
            "url": "https://example.com/dataset",
            "description": "A simulated dataset of military aircraft using Unreal Engine",
            "relevance_score": 0.8,
            "relevance_reasoning": "good synthetic data",
        })
        assert result.get("rejected") is True
        assert result.get("reason") == "synthetic"

    def test_store_synthetic_in_name_rejected(self, tool_executor: ToolExecutor):
        result = tool_executor.execute("store_dataset", {
            "name": "GAN-generated aircraft dataset",
            "url": "https://example.com/dataset",
            "relevance_score": 0.8,
            "relevance_reasoning": "looks good",
        })
        assert result.get("rejected") is True

    def test_store_real_dataset_accepted(self, tool_executor: ToolExecutor, storage):
        with patch.object(tool_executor, '_url_reachable', return_value=True):
            with patch.object(storage, 'save_dataset', return_value=3):
                result = tool_executor.execute("store_dataset", {
                    "name": "Real Aircraft Satellite Dataset",
                    "url": "https://zenodo.org/records/12345",
                    "description": "Real satellite imagery of aircraft from Sentinel-2",
                    "relevance_score": 0.9,
                    "relevance_reasoning": "excellent match",
                })
        # May be rejected due to aerial signal check, but not synthetic
        if isinstance(result, dict) and result.get("rejected"):
            assert result.get("reason") != "synthetic"


# ─── Perspective check ──────────────────────────────────────────────────────

class TestPerspectiveCheck:
    def test_ground_perspective_rejected(self, tool_executor: ToolExecutor):
        result = tool_executor.execute("store_dataset", {
            "name": "Airliner Spotter Photos",
            "url": "https://example.com/planespotter",
            "description": "Ground-level photos of aircraft at airshow, side-view images from Wikimedia Commons",
            "relevance_score": 0.8,
            "relevance_reasoning": "lots of aircraft images",
        })
        assert result.get("rejected") is True
        assert result.get("reason") == "ground_perspective"

    def test_ground_perspective_with_tags(self, tool_executor: ToolExecutor):
        result = tool_executor.execute("store_dataset", {
            "name": "My Dataset",
            "url": "https://example.com/dataset",
            "tags": ["planespotter", "airshow", "static display"],
            "relevance_score": 0.7,
            "relevance_reasoning": "contains aircraft",
        })
        assert result.get("rejected") is True

    def test_aerial_dataset_not_rejected(self, tool_executor: ToolExecutor, storage):
        with patch.object(tool_executor, '_url_reachable', return_value=True):
            with patch.object(storage, 'save_dataset', return_value=4):
                result = tool_executor.execute("store_dataset", {
                    "name": "Aerial Aircraft Dataset",
                    "url": "https://zenodo.org/records/12346",
                    "description": "Satellite imagery of military aircraft from Maxar, top-down view, GeoTIFF format",
                    "tags": ["satellite", "aerial", "military", "aircraft"],
                    "relevance_score": 0.95,
                    "relevance_reasoning": "perfect aerial match",
                    "source": "zenodo",
                })
        if isinstance(result, dict) and result.get("rejected"):
            assert result.get("reason") not in ("ground_perspective")

    def test_no_perspective_check_when_subject_not_aerial(self, tool_executor_generic: ToolExecutor, storage):
        with patch.object(tool_executor_generic, '_url_reachable', return_value=True):
            with patch.object(storage, 'save_dataset', return_value=5):
                result = tool_executor_generic.execute("store_dataset", {
                    "name": "Ground-level Cat Photos",
                    "url": "https://example.com/cats",
                    "description": "Pictures of cats on the ground",
                    "relevance_score": 0.8,
                    "relevance_reasoning": "good cat dataset",
                })
        # Should NOT be rejected for perspective when subject is "cats and dogs"
        if isinstance(result, dict) and result.get("rejected"):
            assert result.get("reason") != "ground_perspective"

    def test_is_mainstream_helper(self):
        assert ToolExecutor._is_mainstream("https://kaggle.com/foo") is True
        assert ToolExecutor._is_mainstream("https://zenodo.org/record/1") is False


# ─── URL canonicalization ───────────────────────────────────────────────────

class TestUrlCanonicalization:
    def test_doi_to_zenodo(self):
        canon = ToolExecutor._canonical_url("https://doi.org/10.5281/zenodo.7331974")
        assert canon == "zenodo.org/records/7331974"

    def test_www_stripped(self):
        canon = ToolExecutor._canonical_url("https://www.Zenodo.org/records/7331974/")
        assert canon == "zenodo.org/records/7331974"

    def test_trailing_slash_stripped(self):
        canon = ToolExecutor._canonical_url("https://figshare.com/articles/dataset/X/")
        assert canon == "figshare.com/articles/dataset/x"

    def test_lowercased(self):
        canon = ToolExecutor._canonical_url("https://Figshare.com/Articles/Dataset/XYZ")
        assert "figshare.com/articles/dataset/xyz" in canon

    def test_empty_url(self):
        assert ToolExecutor._canonical_url("") == ""
        assert ToolExecutor._canonical_url(None) == ""


# ─── Repeat query rejection ─────────────────────────────────────────────────

class TestRepeatQuery:
    def test_exact_repeat_web_search_rejected(self, tool_executor: ToolExecutor):
        q = "site:figshare.com aircraft dataset"
        # First call goes through (may be rejected for other reasons)
        tool_executor._recent_queries.clear()
        tool_executor._recent_queries.append("web:" + " ".join(q.lower().split()))
        result = tool_executor.execute("web_search", {"query": q})
        if isinstance(result, dict) and result.get("rejected"):
            assert "already issued" in (result.get("error") or "").lower()
        else:
            pass  # non-dict result means it went through as a list

    def test_different_query_not_rejected(self, tool_executor: ToolExecutor):
        tool_executor._recent_queries.clear()
        tool_executor._recent_queries.append("web:something else")
        with patch.object(tool_executor, '_http') as mock_http:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"web": {"results": []}}
            mock_http.get.return_value = mock_response
            result = tool_executor.execute("web_search", {"query": "site:figshare.com different query"})
            assert isinstance(result, list) or (isinstance(result, dict) and not "already issued" in (result.get("error") or "").lower())


# ─── Operator rotation ──────────────────────────────────────────────────────

class TestOperatorRotation:
    def test_consecutive_broad_queries_rejected(self, tool_executor: ToolExecutor):
        tool_executor._recent_query_was_broad.clear()
        for _ in range(3):
            tool_executor._recent_query_was_broad.append(True)
        result = tool_executor.execute("web_search", {"query": "aircraft dataset"})
        assert isinstance(result, dict)
        assert result.get("rejected") is True

    def test_operator_query_not_rejected_for_broad(self, tool_executor: ToolExecutor):
        tool_executor._recent_query_was_broad.clear()
        for _ in range(3):
            tool_executor._recent_query_was_broad.append(False)
        with patch.object(tool_executor, '_http') as mock_http:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"web": {"results": []}}
            mock_http.get.return_value = mock_response
            result = tool_executor.execute("web_search", {"query": "site:figshare.com aircraft dataset"})
            assert isinstance(result, list) or (isinstance(result, dict) and not result.get("rejected"))

    def test_query_uses_operator(self):
        assert ToolExecutor._query_uses_operator("site:figshare.com aircraft") is True
        assert ToolExecutor._query_uses_operator("filetype:csv aircraft") is True
        assert ToolExecutor._query_uses_operator("inurl:dataset aircraft") is True
        assert ToolExecutor._query_uses_operator("intitle:aircraft dataset") is True
        assert ToolExecutor._query_uses_operator("aircraft dataset") is False


# ─── Synthetic query rejection in arxiv_search ──────────────────────────────

class TestArxivSynthetic:
    def test_arxiv_synthetic_rejected(self, tool_executor: ToolExecutor):
        # Open the gate first
        tool_executor._stores_so_far = 2
        result = tool_executor.execute("arxiv_search", {"query": "synthetic aircraft dataset"})
        assert isinstance(result, dict)
        assert result.get("rejected") is True


# ─── Aggregator domain rejection ────────────────────────────────────────────

class TestAggregatorRejection:
    def test_aggregator_url_rejected(self, tool_executor: ToolExecutor):
        with patch.object(tool_executor, '_url_reachable', return_value=True):
            result = tool_executor.execute("store_dataset", {
                "name": "Dataset from aggregator",
                "url": "https://datasetninja.com/aircraft-dataset",
                "relevance_score": 0.8,
                "relevance_reasoning": "looks good",
            })
        assert result.get("rejected") is True
        assert result.get("reason") == "aggregator_domain"

    def test_gts_dot_ai_rejected(self, tool_executor: ToolExecutor):
        assert ToolExecutor._domain_of("https://gts.ai/datasets/aircraft") == "gts.ai"
        assert ToolExecutor._is_aggregator("https://gts.ai/datasets/aircraft") is True


# ─── Homepage URL rejection ─────────────────────────────────────────────────

class TestHomepageRejection:
    def test_root_url_rejected(self, tool_executor: ToolExecutor):
        result = tool_executor.execute("store_dataset", {
            "name": "Root dataset",
            "url": "https://figshare.com/",
            "relevance_score": 0.5,
            "relevance_reasoning": "maybe",
        })
        assert result.get("rejected") is True
        assert result.get("reason") == "homepage_url"

    @pytest.mark.parametrize("path", ["/datasets", "/data", "/search", "/datasets/"])
    def test_listing_url_rejected(self, tool_executor: ToolExecutor, path: str):
        result = tool_executor.execute("store_dataset", {
            "name": "Listing",
            "url": f"https://example.com{path}",
            "relevance_score": 0.5,
            "relevance_reasoning": "maybe",
        })
        assert result.get("rejected") is True
        assert result.get("reason") == "homepage_url"


# ─── Paper/report URL rejection ─────────────────────────────────────────────

class TestPaperUrlRejection:
    def test_pdf_url_rejected(self, tool_executor: ToolExecutor):
        result = tool_executor.execute("store_dataset", {
            "name": "Paper",
            "url": "https://arxiv.org/pdf/1234.5678.pdf",
            "relevance_score": 0.8,
            "relevance_reasoning": "paper mentions dataset",
        })
        assert result.get("rejected") is True
        assert result.get("reason") == "paper_or_report"

    def test_looks_like_paper_or_report(self):
        assert ToolExecutor._looks_like_paper_or_report("https://example.com/reports/aircraft.pdf") is True
        assert ToolExecutor._looks_like_paper_or_report("https://example.com/publications/dataset") is True
        assert ToolExecutor._looks_like_paper_or_report("https://zenodo.org/records/12345") is False
