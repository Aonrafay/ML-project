"""Sentiment Analyzer model wrapper.

Provides a higher-level interface for sentiment analysis with
VADER/DistilBERT model switching, batch processing, aggregation,
and timeline analysis capabilities.
"""

from ml_models.sentiment.predict import predict_sentiment, get_distilbert_classifier
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from typing import Optional
from datetime import datetime


# Sentiment labels
LABEL_POSITIVE = "positive"
LABEL_NEUTRAL = "neutral"
LABEL_NEGATIVE = "negative"

# Model modes
MODE_DISTILBERT = "distilbert"
MODE_VADER = "vader"
MODE_AUTO = "auto"  # Try DistilBERT first, fall back to VADER

# Default settings
DEFAULT_MODE = MODE_AUTO
DEFAULT_BATCH_SIZE = 32


class SentimentAnalyzer:
    """Higher-level sentiment analysis wrapper.

    Wraps the base DistilBERT + VADER prediction module with additional
    features like model switching, batch processing, aggregation, and
    timeline analysis.
    """

    def __init__(
        self,
        mode: str = DEFAULT_MODE,
    ):
        """Initialize the SentimentAnalyzer.

        Args:
            mode: Analysis mode — "distilbert", "vader", or "auto".
                "auto" tries DistilBERT first and falls back to VADER.
        """
        self.mode = mode
        self.vader = SentimentIntensityAnalyzer()
        self._distilbert = None

    def analyze(self, text: str) -> dict:
        """Analyze sentiment of a single text.

        Args:
            text: The text to analyze.

        Returns:
            Dict with label, score, model used, and VADER details.
        """
        if self.mode == MODE_VADER:
            return self._analyze_vader(text)
        elif self.mode == MODE_DISTILBERT:
            return self._analyze_distilbert(text)
        else:  # MODE_AUTO
            return self._analyze_auto(text)

    def analyze_batch(self, texts: list[str]) -> list[dict]:
        """Analyze sentiment for a batch of texts.

        Args:
            texts: List of texts to analyze.

        Returns:
            List of sentiment analysis result dicts.
        """
        return [self.analyze(text) for text in texts]

    def analyze_with_details(self, text: str) -> dict:
        """Analyze sentiment with full VADER score breakdown.

        Args:
            text: The text to analyze.

        Returns:
            Dict with primary analysis plus VADER component scores.
        """
        primary = self.analyze(text)
        vader_scores = self.vader.polarity_scores(text)

        return {
            "text": text[:200],
            "text_length": len(text),
            "label": primary["label"],
            "score": primary["score"],
            "model": primary["model"],
            "vader_scores": {
                "positive": vader_scores["pos"],
                "negative": vader_scores["neg"],
                "neutral": vader_scores["neu"],
                "compound": vader_scores["compound"],
            },
        }

    def get_aggregation(self, results: list[dict]) -> dict:
        """Aggregate sentiment results into a summary.

        Args:
            results: List of sentiment analysis result dicts.

        Returns:
            Dict with sentiment distribution and average scores.
        """
        total = len(results)
        if total == 0:
            return {
                "total": 0,
                "distribution": {},
                "average_score": 0.0,
                "dominant_sentiment": None,
            }

        positive_count = sum(1 for r in results if r["label"] == LABEL_POSITIVE)
        negative_count = sum(1 for r in results if r["label"] == LABEL_NEGATIVE)
        neutral_count = sum(1 for r in results if r["label"] == LABEL_NEUTRAL)

        distribution = {
            LABEL_POSITIVE: round(positive_count / total * 100, 2),
            LABEL_NEGATIVE: round(negative_count / total * 100, 2),
            LABEL_NEUTRAL: round(neutral_count / total * 100, 2),
        }

        avg_score = sum(r["score"] for r in results) / total

        # Determine dominant sentiment
        counts = {
            LABEL_POSITIVE: positive_count,
            LABEL_NEGATIVE: negative_count,
            LABEL_NEUTRAL: neutral_count,
        }
        dominant = max(counts, key=counts.get)

        return {
            "total": total,
            "distribution": distribution,
            "average_score": round(avg_score, 2),
            "dominant_sentiment": dominant,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "neutral_count": neutral_count,
        }

    def build_timeline(
        self,
        texts_with_timestamps: list[dict],
    ) -> list[dict]:
        """Build a sentiment timeline from texts with timestamps.

        Args:
            texts_with_timestamps: List of dicts with "text" and "timestamp" keys.

        Returns:
            List of dicts with timestamp, sentiment, and score, sorted by time.
        """
        timeline = []
        for item in texts_with_timestamps:
            text = item["text"]
            timestamp = item.get("timestamp", datetime.utcnow().isoformat())
            result = self.analyze(text)
            timeline.append({
                "timestamp": timestamp,
                "sentiment": result["label"],
                "score": result["score"],
                "model": result["model"],
            })

        # Sort by timestamp
        timeline.sort(key=lambda x: x["timestamp"])
        return timeline

    def _analyze_vader(self, text: str) -> dict:
        """Analyze using VADER only."""
        scores = self.vader.polarity_scores(text)
        compound = scores["compound"]

        if compound >= 0.05:
            label = LABEL_POSITIVE
        elif compound <= -0.05:
            label = LABEL_NEGATIVE
        else:
            label = LABEL_NEUTRAL

        return {
            "label": label,
            "score": round(abs(compound) * 100, 2),
            "model": MODE_VADER,
        }

    def _analyze_distilbert(self, text: str) -> dict:
        """Analyze using DistilBERT only."""
        classifier = self._ensure_distilbert()

        if classifier is None:
            # DistilBERT not available, fall back to VADER
            return self._analyze_vader(text)

        try:
            results = classifier(text)[0]
            label_map = {"LABEL_0": LABEL_NEGATIVE, "LABEL_1": LABEL_NEUTRAL, "LABEL_2": LABEL_POSITIVE}
            best = max(results, key=lambda x: x["score"])
            return {
                "label": label_map.get(best["label"], LABEL_NEUTRAL),
                "score": round(best["score"] * 100, 2),
                "model": MODE_DISTILBERT,
            }
        except Exception:
            return self._analyze_vader(text)

    def _analyze_auto(self, text: str) -> dict:
        """Analyze using auto mode (DistilBERT first, VADER fallback)."""
        result = predict_sentiment(text)
        return {
            "label": result["label"],
            "score": result["score"],
            "model": result.get("model", MODE_AUTO),
        }

    def _ensure_distilbert(self):
        """Ensure DistilBERT classifier is loaded."""
        if self._distilbert is None:
            self._distilbert = get_distilbert_classifier()
        return self._distilbert


# Singleton instance for convenience
_default_analyzer = SentimentAnalyzer()


def analyze_sentiment(text: str, mode: Optional[str] = None) -> dict:
    """Analyze sentiment of a text using the default analyzer.

    Args:
        text: The text to analyze.
        mode: Override analysis mode for this call.

    Returns:
        Sentiment analysis result dict.
    """
    if mode is not None:
        analyzer = SentimentAnalyzer(mode=mode)
        return analyzer.analyze(text)
    return _default_analyzer.analyze(text)


def analyze_batch_sentiment(texts: list[str], mode: Optional[str] = None) -> list[dict]:
    """Analyze sentiment for a batch of texts.

    Args:
        texts: List of texts to analyze.
        mode: Override analysis mode for this call.

    Returns:
        List of sentiment analysis result dicts.
    """
    if mode is not None:
        analyzer = SentimentAnalyzer(mode=mode)
        return analyzer.analyze_batch(texts)
    return _default_analyzer.analyze_batch(texts)


def aggregate_sentiment(results: list[dict]) -> dict:
    """Aggregate sentiment results using the default analyzer.

    Args:
        results: List of sentiment analysis result dicts.

    Returns:
        Aggregation summary dict.
    """
    analyzer = SentimentAnalyzer()
    return analyzer.get_aggregation(results)