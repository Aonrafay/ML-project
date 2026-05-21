"""Tests for fact_check module.

Covers the verifier, Google Fact Check API integration, and ClaimBuster
API integration using mocks for external HTTP calls and settings.
"""

import pytest
from unittest.mock import MagicMock, patch
from fact_check.verifier import verify_claim, _is_flagged
from fact_check.claimbuster import claimbuster_check
from fact_check.google_factcheck import google_fact_check


# ─── _is_flagged Tests ────────────────────────────────────────────────────


class TestIsFlagged:
    """Tests for the _is_flagged helper function."""

    def test_flagged_false_verdict(self):
        """'false' verdict should be flagged."""
        assert _is_flagged({"verdict": "false"}) is True

    def test_flagged_mostly_false_verdict(self):
        """'mostly false' verdict should be flagged."""
        assert _is_flagged({"verdict": "mostly false"}) is True

    def test_flagged_mixture_verdict(self):
        """'mixture' verdict should be flagged."""
        assert _is_flagged({"verdict": "mixture"}) is True

    def test_flagged_disputed_verdict(self):
        """'disputed' verdict should be flagged."""
        assert _is_flagged({"verdict": "disputed"}) is True

    def test_flagged_unproven_verdict(self):
        """'unproven' verdict should be flagged."""
        assert _is_flagged({"verdict": "unproven"}) is True

    def test_not_flagged_true_verdict(self):
        """'true' verdict should NOT be flagged."""
        assert _is_flagged({"verdict": "true"}) is False

    def test_not_flagged_no_match_verdict(self):
        """'no_match' verdict should NOT be flagged."""
        assert _is_flagged({"verdict": "no_match"}) is False

    def test_not_flagged_empty_verdict(self):
        """Empty verdict should NOT be flagged."""
        assert _is_flagged({"verdict": ""}) is False

    def test_not_flagged_missing_verdict(self):
        """Missing verdict key should NOT be flagged."""
        assert _is_flagged({}) is False

    def test_flagged_case_insensitive(self):
        """Flagging should be case-insensitive."""
        assert _is_flagged({"verdict": "False"}) is True
        assert _is_flagged({"verdict": "FALSE"}) is True

    def test_flagged_partial_match(self):
        """Verdict containing a flagged keyword should be flagged."""
        assert _is_flagged({"verdict": "mostly false according to reviewers"}) is True


# ─── ClaimBuster Tests ────────────────────────────────────────────────────


class TestClaimBuster:
    """Tests for the claimbuster_check function using mocked settings and HTTP."""

    @patch("fact_check.claimbuster.get_settings")
    def test_no_api_key_returns_error(self, mock_settings):
        """Should return error dict when API key is not configured."""
        mock_settings.return_value = MagicMock(claimbuster_api_key="")
        result = claimbuster_check("Test claim text")
        assert "error" in result
        assert "not configured" in result["error"].lower()

    @patch("fact_check.claimbuster.httpx.Client")
    @patch("fact_check.claimbuster.get_settings")
    def test_successful_check(self, mock_settings, mock_client_cls):
        """Should return score and checkworthy flag on success."""
        mock_settings.return_value = MagicMock(claimbuster_api_key="test_key")

        # Mock HTTP response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"score": 0.85}
        mock_response.raise_for_status = MagicMock()

        mock_client_instance = mock_client_cls.return_value
        mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
        mock_client_instance.__exit__ = MagicMock(return_value=False)
        mock_client_instance.get.return_value = mock_response

        result = claimbuster_check("The president announced new policies")
        assert result["score"] == 0.85
        assert result["checkworthy"] is True
        assert "raw" in result

    @patch("fact_check.claimbuster.httpx.Client")
    @patch("fact_check.claimbuster.get_settings")
    def test_low_score_not_checkworthy(self, mock_settings, mock_client_cls):
        """Low score should result in checkworthy=False."""
        mock_settings.return_value = MagicMock(claimbuster_api_key="test_key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"score": 0.3}
        mock_response.raise_for_status = MagicMock()

        mock_client_instance = mock_client_cls.return_value
        mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
        mock_client_instance.__exit__ = MagicMock(return_value=False)
        mock_client_instance.get.return_value = mock_response

        result = claimbuster_check("Hello world")
        assert result["score"] == 0.3
        assert result["checkworthy"] is False

    @patch("fact_check.claimbuster.httpx.Client")
    @patch("fact_check.claimbuster.get_settings")
    def test_api_error_returns_error_dict(self, mock_settings, mock_client_cls):
        """Should return error dict on API failure."""
        mock_settings.return_value = MagicMock(claimbuster_api_key="test_key")

        mock_client_instance = mock_client_cls.return_value
        mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
        mock_client_instance.__exit__ = MagicMock(return_value=False)
        mock_client_instance.get.side_effect = Exception("Connection error")

        result = claimbuster_check("Test claim")
        assert "error" in result
        assert result["score"] == 0
        assert result["checkworthy"] is False


# ─── Google Fact Check Tests ──────────────────────────────────────────────


class TestGoogleFactCheck:
    """Tests for the google_fact_check function using mocked settings and HTTP."""

    @patch("fact_check.google_factcheck.get_settings")
    def test_no_api_key_returns_error(self, mock_settings):
        """Should return error dict when API key is not configured."""
        mock_settings.return_value = MagicMock(google_fact_check_api_key="")
        result = google_fact_check("Test claim")
        assert "error" in result
        assert "not configured" in result["error"].lower()

    @patch("fact_check.google_factcheck.httpx.Client")
    @patch("fact_check.google_factcheck.get_settings")
    def test_no_matching_claims(self, mock_settings, mock_client_cls):
        """Should return no_match verdict when no claims are found."""
        mock_settings.return_value = MagicMock(google_fact_check_api_key="test_key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"claims": []}
        mock_response.raise_for_status = MagicMock()

        mock_client_instance = mock_client_cls.return_value
        mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
        mock_client_instance.__exit__ = MagicMock(return_value=False)
        mock_client_instance.get.return_value = mock_response

        result = google_fact_check("Obscure claim with no matches")
        assert result["verdict"] == "no_match"
        assert result["claims"] == []
        assert result["confidence"] == 0

    @patch("fact_check.google_factcheck.httpx.Client")
    @patch("fact_check.google_factcheck.get_settings")
    def test_matching_claims_returned(self, mock_settings, mock_client_cls):
        """Should return verdict and claims when matches are found."""
        mock_settings.return_value = MagicMock(google_fact_check_api_key="test_key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "claims": [
                {
                    "text": "Test claim text",
                    "claimReview": [
                        {
                            "textualRating": "false",
                            "publisher": {"name": "Snopes"},
                            "url": "https://snopes.com/test",
                            "reviewDate": "2024-01-01",
                        }
                    ],
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()

        mock_client_instance = mock_client_cls.return_value
        mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
        mock_client_instance.__exit__ = MagicMock(return_value=False)
        mock_client_instance.get.return_value = mock_response

        result = google_fact_check("Test claim")
        assert result["verdict"] == "false"
        assert len(result["claims"]) == 1
        assert result["claims"][0]["verdict"] == "false"
        assert result["claims"][0]["publisher"] == "Snopes"
        assert result["confidence"] == 0.8

    @patch("fact_check.google_factcheck.httpx.Client")
    @patch("fact_check.google_factcheck.get_settings")
    def test_multiple_claims_limited_to_five(self, mock_settings, mock_client_cls):
        """Should limit results to at most 5 claims."""
        mock_settings.return_value = MagicMock(google_fact_check_api_key="test_key")

        claims_data = [
            {
                "text": f"Claim {i}",
                "claimReview": [
                    {
                        "textualRating": "false",
                        "publisher": {"name": f"Publisher {i}"},
                        "url": f"https://example.com/{i}",
                        "reviewDate": "2024-01-01",
                    }
                ],
            }
            for i in range(10)
        ]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"claims": claims_data}
        mock_response.raise_for_status = MagicMock()

        mock_client_instance = mock_client_cls.return_value
        mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
        mock_client_instance.__exit__ = MagicMock(return_value=False)
        mock_client_instance.get.return_value = mock_response

        result = google_fact_check("Test claim")
        assert len(result["claims"]) <= 5

    @patch("fact_check.google_factcheck.httpx.Client")
    @patch("fact_check.google_factcheck.get_settings")
    def test_api_error_returns_error_dict(self, mock_settings, mock_client_cls):
        """Should return error dict on API failure."""
        mock_settings.return_value = MagicMock(google_fact_check_api_key="test_key")

        mock_client_instance = mock_client_cls.return_value
        mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
        mock_client_instance.__exit__ = MagicMock(return_value=False)
        mock_client_instance.get.side_effect = Exception("Connection timeout")

        result = google_fact_check("Test claim")
        assert "error" in result
        assert result["verdict"] == "error"
        assert result["confidence"] == 0


# ─── Verifier Integration Tests ────────────────────────────────────────────


class TestVerifier:
    """Tests for the verify_claim function that integrates ClaimBuster + Google Fact Check."""

    @patch("fact_check.verifier.google_fact_check")
    @patch("fact_check.verifier.claimbuster_check")
    def test_checkworthy_claim_gets_fact_checked(self, mock_claimbuster, mock_google):
        """A checkworthy claim should trigger Google Fact Check."""
        mock_claimbuster.return_value = {
            "score": 0.85,
            "checkworthy": True,
        }
        mock_google.return_value = {
            "verdict": "false",
            "claims": [{"verdict": "false", "publisher": "Snopes"}],
            "confidence": 0.8,
        }

        result = verify_claim("The president announced new policies")
        assert result["claim"] == "The president announced new policies"
        assert result["checkworthiness"]["checkworthy"] is True
        assert result["fact_check"]["verdict"] == "false"
        assert result["flagged"] is True
        mock_google.assert_called_once()

    @patch("fact_check.verifier.google_fact_check")
    @patch("fact_check.verifier.claimbuster_check")
    def test_not_checkworthy_claim_skips_fact_check(self, mock_claimbuster, mock_google):
        """A non-checkworthy claim should NOT trigger Google Fact Check."""
        mock_claimbuster.return_value = {
            "score": 0.2,
            "checkworthy": False,
        }

        result = verify_claim("Hello world")
        assert result["checkworthiness"]["checkworthy"] is False
        assert result["fact_check"]["verdict"] == "not_checkworthy"
        assert result["fact_check"]["claims"] == []
        assert result["flagged"] is False
        mock_google.assert_not_called()

    @patch("fact_check.verifier.google_fact_check")
    @patch("fact_check.verifier.claimbuster_check")
    def test_flagged_when_false_verdict(self, mock_claimbuster, mock_google):
        """verify_claim should set flagged=True when verdict is 'false'."""
        mock_claimbuster.return_value = {
            "score": 0.9,
            "checkworthy": True,
        }
        mock_google.return_value = {
            "verdict": "false",
            "claims": [],
            "confidence": 0.8,
        }

        result = verify_claim("Fake news claim")
        assert result["flagged"] is True

    @patch("fact_check.verifier.google_fact_check")
    @patch("fact_check.verifier.claimbuster_check")
    def test_not_flagged_when_true_verdict(self, mock_claimbuster, mock_google):
        """verify_claim should set flagged=False when verdict is 'true'."""
        mock_claimbuster.return_value = {
            "score": 0.9,
            "checkworthy": True,
        }
        mock_google.return_value = {
            "verdict": "true",
            "claims": [],
            "confidence": 0.8,
        }

        result = verify_claim("Verified true claim")
        assert result["flagged"] is False

    @patch("fact_check.verifier.google_fact_check")
    @patch("fact_check.verifier.claimbuster_check")
    def test_verify_claim_returns_complete_structure(self, mock_claimbuster, mock_google):
        """verify_claim should return a dict with all expected keys."""
        mock_claimbuster.return_value = {
            "score": 0.75,
            "checkworthy": True,
        }
        mock_google.return_value = {
            "verdict": "mixture",
            "claims": [{"verdict": "mixture"}],
            "confidence": 0.6,
        }

        result = verify_claim("Partially true claim")
        assert "claim" in result
        assert "checkworthiness" in result
        assert "fact_check" in result
        assert "flagged" in result