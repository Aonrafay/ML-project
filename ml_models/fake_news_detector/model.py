"""Fake News Detector model wrapper.

Provides a higher-level interface for fake news detection with batch
processing, confidence thresholds, detailed analysis, and result
aggregation capabilities.
"""

from ml_models.fake_news.predict import predict_fake_news, get_classifier
from typing import Optional


# Default confidence threshold for classification
DEFAULT_CONFIDENCE_THRESHOLD = 50.0  # percentage

# Labels for classification results
LABEL_FAKE = "fake"
LABEL_REAL = "real"


class FakeNewsDetector:
    """Higher-level fake news detection wrapper.

    Wraps the base RoBERTa prediction module with additional features
    like batch processing, confidence thresholds, and detailed analysis.
    """

    def __init__(
        self,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ):
        """Initialize the FakeNewsDetector.

        Args:
            confidence_threshold: Minimum confidence (%) to accept a classification.
                Results below this threshold are marked as "uncertain".
        """
        self.confidence_threshold = confidence_threshold
        self._classifier = None

    def classify(self, text: str) -> dict:
        """Classify a single text as fake or real news.

        Args:
            text: The text to classify.

        Returns:
            Dict with label, confidence, is_uncertain flag, and details.
        """
        result = predict_fake_news(text)
        label = result["label"]
        confidence = result["confidence"]

        is_uncertain = confidence < self.confidence_threshold

        return {
            "text": text[:200],
            "label": label,
            "confidence": confidence,
            "is_uncertain": is_uncertain,
            "effective_label": "uncertain" if is_uncertain else label,
            "confidence_threshold": self.confidence_threshold,
        }

    def classify_batch(self, texts: list[str]) -> list[dict]:
        """Classify a batch of texts.

        Args:
            texts: List of texts to classify.

        Returns:
            List of classification result dicts.
        """
        return [self.classify(text) for text in texts]

    def classify_detailed(self, text: str) -> dict:
        """Classify with detailed analysis including all scores.

        Args:
            text: The text to classify.

        Returns:
            Dict with full classification details and text metadata.
        """
        classifier = self._ensure_classifier()
        raw_results = classifier(text)

        # Process all label scores
        all_scores = {}
        for result in raw_results:
            label = result["label"]
            score = result["score"]
            if "NEGATIVE" in label or "FAKE" in label.upper():
                all_scores["fake"] = round(score * 100, 2)
            else:
                all_scores["real"] = round(score * 100, 2)

        # Determine primary classification
        primary = predict_fake_news(text)
        label = primary["label"]
        confidence = primary["confidence"]
        is_uncertain = confidence < self.confidence_threshold

        return {
            "text": text[:500],
            "text_length": len(text),
            "primary_label": label,
            "primary_confidence": confidence,
            "is_uncertain": is_uncertain,
            "effective_label": "uncertain" if is_uncertain else label,
            "all_scores": all_scores,
            "confidence_threshold": self.confidence_threshold,
        }

    def get_statistics(self, results: list[dict]) -> dict:
        """Aggregate statistics from a list of classification results.

        Args:
            results: List of classification result dicts (from classify or classify_batch).

        Returns:
            Dict with aggregated statistics.
        """
        total = len(results)
        if total == 0:
            return {
                "total": 0,
                "fake_count": 0,
                "real_count": 0,
                "uncertain_count": 0,
                "avg_confidence": 0.0,
                "distribution": {},
            }

        fake_count = sum(1 for r in results if r["label"] == LABEL_FAKE)
        real_count = sum(1 for r in results if r["label"] == LABEL_REAL)
        uncertain_count = sum(1 for r in results if r.get("is_uncertain", False))
        avg_confidence = sum(r["confidence"] for r in results) / total

        distribution = {
            LABEL_FAKE: round(fake_count / total * 100, 2),
            LABEL_REAL: round(real_count / total * 100, 2),
            "uncertain": round(uncertain_count / total * 100, 2),
        }

        return {
            "total": total,
            "fake_count": fake_count,
            "real_count": real_count,
            "uncertain_count": uncertain_count,
            "avg_confidence": round(avg_confidence, 2),
            "distribution": distribution,
        }

    def _ensure_classifier(self):
        """Ensure the classifier is loaded and return it."""
        if self._classifier is None:
            self._classifier = get_classifier()
        return self._classifier


# Singleton instance for convenience
_default_detector = FakeNewsDetector()


def classify_text(text: str, confidence_threshold: Optional[float] = None) -> dict:
    """Classify text using the default detector.

    Args:
        text: The text to classify.
        confidence_threshold: Override threshold for this call.

    Returns:
        Classification result dict.
    """
    if confidence_threshold is not None:
        detector = FakeNewsDetector(confidence_threshold=confidence_threshold)
        return detector.classify(text)
    return _default_detector.classify(text)


def classify_texts(texts: list[str], confidence_threshold: Optional[float] = None) -> list[dict]:
    """Classify a batch of texts using the default detector.

    Args:
        texts: List of texts to classify.
        confidence_threshold: Override threshold for this call.

    Returns:
        List of classification result dicts.
    """
    if confidence_threshold is not None:
        detector = FakeNewsDetector(confidence_threshold=confidence_threshold)
        return detector.classify_batch(texts)
    return _default_detector.classify_batch(texts)