"""Unit tests for individual tools — use mock data, no real models needed."""
import pytest
from unittest.mock import patch, MagicMock


class TestProductLookup:
    def test_returns_confidence_response_structure(self):
        """Every tool must return result, confidence, source_count, evidence."""
        from utils.helpers import confidence_response
        resp = confidence_response(result=[], confidence=0.5, source_count=0)
        assert "result" in resp
        assert "confidence" in resp
        assert 0.0 <= resp["confidence"] <= 1.0

    def test_empty_query_returns_empty_results(self):
        """Empty product DB returns empty results gracefully."""
        from tools.product_lookup import product_lookup
        with patch("tools.product_lookup._load_products") as mock:
            import pandas as pd
            mock.return_value = (pd.DataFrame(), None)
            result = product_lookup.invoke({"query": "earphones", "filters": {}})
            assert result["result"] == []
            assert result["confidence"] == 0.0


class TestFakeReviewFilter:
    def test_heuristic_flags_short_reviews(self):
        """Very short reviews should get low authenticity score via heuristic."""
        from tools.fake_review_filter import _extract_features
        reviews = [{"text": "good", "rating": 5.0, "verified_purchase": 0}]
        features = _extract_features(reviews)
        assert features.shape == (1, 10)
        assert features[0][0] == len("good")  # char length feature


class TestSafetyValidator:
    def test_passes_when_no_products(self):
        from services.safety_validator import SafetyValidator
        validator = SafetyValidator()
        result = validator.validate([], "This is a test response")
        assert result.passed is True

    def test_fails_on_low_authenticity(self):
        from services.safety_validator import SafetyValidator
        validator = SafetyValidator()
        products = [{"authenticity_score": 0.2}]
        result = validator.validate(products, "response")
        assert result.passed is False
        assert "review_authentic" in result.failed_checks
