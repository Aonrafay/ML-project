"""Tests for ml_models module.

Covers SentimentAnalyzer, FakeNewsDetector, and TopicModeler
using mocks for the underlying transformer/BERTopic models.
"""

import pytest
from unittest.mock import MagicMock, patch
from ml_models.sentiment_analyzer.model import (
    SentimentAnalyzer,
    LABEL_POSITIVE,
    LABEL_NEGATIVE,
    LABEL_NEUTRAL,
    MODE_DISTILBERT,
    MODE_VADER,
    MODE_AUTO,
    analyze_sentiment,
    analyze_batch_sentiment,
    aggregate_sentiment,
)
from ml_models.fake_news_detector.model import (
    FakeNewsDetector,
    LABEL_FAKE,
    LABEL_REAL,
    DEFAULT_CONFIDENCE_THRESHOLD,
    classify_text,
    classify_texts,
)
from ml_models.topic_modeler.model import (
    TopicModeler,
    DEFAULT_MIN_TOPIC_SIZE,
    DEFAULT_TOP_N_WORDS,
    analyze_topics,
    get_topic_overview,
    detect_topic_trends,
)


# ─── SentimentAnalyzer Tests ──────────────────────────────────────────────


class TestSentimentAnalyzerVader:
    """Tests for SentimentAnalyzer in VADER mode (no model dependency)."""

    def setup_method(self):
        self.analyzer = SentimentAnalyzer(mode=MODE_VADER)

    def test_analyze_positive(self):
        """VADER should detect positive sentiment."""
        result = self.analyzer.analyze("I absolutely love this amazing product!")
        assert result["label"] == LABEL_POSITIVE
        assert result["score"] > 0
        assert result["model"] == MODE_VADER

    def test_analyze_negative(self):
        """VADER should detect negative sentiment."""
        result = self.analyzer.analyze("This is terrible and I hate it so much")
        assert result["label"] == LABEL_NEGATIVE
        assert result["score"] > 0
        assert result["model"] == MODE_VADER

    def test_analyze_neutral(self):
        """VADER should detect neutral sentiment for mild text."""
        result = self.analyzer.analyze("The meeting is scheduled for tomorrow")
        assert result["label"] in [LABEL_NEUTRAL, LABEL_POSITIVE, LABEL_NEGATIVE]
        assert result["model"] == MODE_VADER

    def test_analyze_returns_dict(self):
        """analyze should return a dict with label, score, and model."""
        result = self.analyzer.analyze("Test text")
        assert "label" in result
        assert "score" in result
        assert "model" in result

    def test_analyze_batch(self):
        """analyze_batch should return results for each text."""
        texts = [
            "I love this!",
            "This is awful",
            "Okay, not bad",
        ]
        results = self.analyzer.analyze_batch(texts)
        assert len(results) == 3
        assert all("label" in r for r in results)

    def test_analyze_with_details(self):
        """analyze_with_details should include VADER component scores."""
        result = self.analyzer.analyze_with_details("I love this amazing product!")
        assert "text" in result
        assert "label" in result
        assert "score" in result
        assert "model" in result
        assert "vader_scores" in result
        assert "positive" in result["vader_scores"]
        assert "negative" in result["vader_scores"]
        assert "neutral" in result["vader_scores"]
        assert "compound" in result["vader_scores"]
        assert "text_length" in result

    def test_get_aggregation(self):
        """get_aggregation should compute distribution and dominant sentiment."""
        results = [
            {"label": LABEL_POSITIVE, "score": 80.0},
            {"label": LABEL_POSITIVE, "score": 70.0},
            {"label": LABEL_NEGATIVE, "score": 60.0},
            {"label": LABEL_NEUTRAL, "score": 40.0},
        ]
        agg = self.analyzer.get_aggregation(results)
        assert agg["total"] == 4
        assert agg["positive_count"] == 2
        assert agg["negative_count"] == 1
        assert agg["neutral_count"] == 1
        assert agg["dominant_sentiment"] == LABEL_POSITIVE
        assert "distribution" in agg
        assert "average_score" in agg

    def test_get_aggregation_empty(self):
        """get_aggregation should handle empty results list."""
        agg = self.analyzer.get_aggregation([])
        assert agg["total"] == 0
        assert agg["distribution"] == {}
        assert agg["average_score"] == 0.0
        assert agg["dominant_sentiment"] is None

    def test_build_timeline(self):
        """build_timeline should return sorted timeline entries."""
        texts_with_timestamps = [
            {"text": "Great news!", "timestamp": "2024-01-01T10:00:00"},
            {"text": "Bad news", "timestamp": "2024-01-01T12:00:00"},
        ]
        timeline = self.analyzer.build_timeline(texts_with_timestamps)
        assert len(timeline) == 2
        assert all("timestamp" in t for t in timeline)
        assert all("sentiment" in t for t in timeline)
        assert all("score" in t for t in timeline)
        # Should be sorted by timestamp
        assert timeline[0]["timestamp"] <= timeline[1]["timestamp"]


class TestSentimentAnalyzerAuto:
    """Tests for SentimentAnalyzer in AUTO mode with mocked DistilBERT."""

    @patch("ml_models.sentiment_analyzer.model.predict_sentiment")
    def test_auto_mode_calls_predict_sentiment(self, mock_predict):
        """AUTO mode should call predict_sentiment."""
        mock_predict.return_value = {
            "label": "positive",
            "score": 85.0,
            "model": "distilbert",
        }
        analyzer = SentimentAnalyzer(mode=MODE_AUTO)
        result = analyzer.analyze("I love this!")
        assert result["label"] == "positive"
        assert result["model"] == "distilbert"
        mock_predict.assert_called_once_with("I love this!")

    @patch("ml_models.sentiment_analyzer.model.predict_sentiment")
    def test_auto_mode_fallback_to_vader(self, mock_predict):
        """AUTO mode should return VADER result when predict returns vader model."""
        mock_predict.return_value = {
            "label": "positive",
            "score": 75.0,
            "model": "vader",
        }
        analyzer = SentimentAnalyzer(mode=MODE_AUTO)
        result = analyzer.analyze("I love this!")
        assert result["model"] == "vader"


class TestSentimentAnalyzerDistilBERT:
    """Tests for SentimentAnalyzer in DistilBERT mode with mocked classifier."""

    @patch("ml_models.sentiment_analyzer.model.get_distilbert_classifier")
    def test_distilbert_mode_with_classifier(self, mock_get_classifier):
        """DistilBERT mode should use the classifier when available."""
        mock_classifier = MagicMock()
        mock_classifier.return_value = [
            [{"label": "LABEL_2", "score": 0.9}, {"label": "LABEL_0", "score": 0.05}, {"label": "LABEL_1", "score": 0.05}]
        ]
        mock_get_classifier.return_value = mock_classifier

        analyzer = SentimentAnalyzer(mode=MODE_DISTILBERT)
        result = analyzer.analyze("I love this!")
        assert result["label"] == LABEL_POSITIVE
        assert result["model"] == MODE_DISTILBERT

    @patch("ml_models.sentiment_analyzer.model.get_distilbert_classifier")
    def test_distilbert_mode_fallback_when_no_classifier(self, mock_get_classifier):
        """DistilBERT mode should fall back to VADER when classifier is None."""
        mock_get_classifier.return_value = None

        analyzer = SentimentAnalyzer(mode=MODE_DISTILBERT)
        result = analyzer.analyze("I love this amazing product!")
        # Falls back to VADER
        assert result["model"] == MODE_VADER
        assert result["label"] == LABEL_POSITIVE

    @patch("ml_models.sentiment_analyzer.model.get_distilbert_classifier")
    def test_distilbert_mode_fallback_on_exception(self, mock_get_classifier):
        """DistilBERT mode should fall back to VADER when classifier throws."""
        mock_classifier = MagicMock()
        mock_classifier.side_effect = RuntimeError("Model error")
        mock_get_classifier.return_value = mock_classifier

        analyzer = SentimentAnalyzer(mode=MODE_DISTILBERT)
        result = analyzer.analyze("I love this amazing product!")
        assert result["model"] == MODE_VADER


class TestSentimentConvenienceFunctions:
    """Tests for module-level convenience functions."""

    @patch("ml_models.sentiment_analyzer.model.predict_sentiment")
    def test_analyze_sentiment_default(self, mock_predict):
        """analyze_sentiment should use default analyzer."""
        mock_predict.return_value = {
            "label": "positive",
            "score": 85.0,
            "model": "auto",
        }
        result = analyze_sentiment("I love this!")
        assert result["label"] == "positive"

    def test_analyze_sentiment_vader_mode(self):
        """analyze_sentiment with mode='vader' should use VADER."""
        result = analyze_sentiment("I love this amazing product!", mode=MODE_VADER)
        assert result["model"] == MODE_VADER
        assert result["label"] == LABEL_POSITIVE

    def test_analyze_batch_sentiment(self):
        """analyze_batch_sentiment should process multiple texts."""
        results = analyze_batch_sentiment(
            ["I love this!", "This is terrible"],
            mode=MODE_VADER,
        )
        assert len(results) == 2

    def test_aggregate_sentiment(self):
        """aggregate_sentiment should aggregate results."""
        results = [
            {"label": LABEL_POSITIVE, "score": 80.0},
            {"label": LABEL_NEGATIVE, "score": 60.0},
        ]
        agg = aggregate_sentiment(results)
        assert agg["total"] == 2
        assert agg["positive_count"] == 1
        assert agg["negative_count"] == 1


# ─── FakeNewsDetector Tests ───────────────────────────────────────────────


class TestFakeNewsDetector:
    """Tests for the FakeNewsDetector class with mocked classifier."""

    @patch("ml_models.fake_news_detector.model.predict_fake_news")
    def test_classify_real_news(self, mock_predict):
        """Should classify real news correctly."""
        mock_predict.return_value = {"label": LABEL_REAL, "confidence": 95.0}
        detector = FakeNewsDetector(confidence_threshold=50.0)
        result = detector.classify("Scientists confirm new discovery")
        assert result["label"] == LABEL_REAL
        assert result["confidence"] == 95.0
        assert result["is_uncertain"] is False
        assert result["effective_label"] == LABEL_REAL

    @patch("ml_models.fake_news_detector.model.predict_fake_news")
    def test_classify_fake_news(self, mock_predict):
        """Should classify fake news correctly."""
        mock_predict.return_value = {"label": LABEL_FAKE, "confidence": 88.0}
        detector = FakeNewsDetector(confidence_threshold=50.0)
        result = detector.classify("Conspiracy theory about aliens")
        assert result["label"] == LABEL_FAKE
        assert result["confidence"] == 88.0
        assert result["is_uncertain"] is False
        assert result["effective_label"] == LABEL_FAKE

    @patch("ml_models.fake_news_detector.model.predict_fake_news")
    def test_classify_uncertain_result(self, mock_predict):
        """Low confidence should result in 'uncertain' effective_label."""
        mock_predict.return_value = {"label": LABEL_FAKE, "confidence": 30.0}
        detector = FakeNewsDetector(confidence_threshold=50.0)
        result = detector.classify("Ambiguous text")
        assert result["is_uncertain"] is True
        assert result["effective_label"] == "uncertain"
        assert result["confidence_threshold"] == 50.0

    @patch("ml_models.fake_news_detector.model.predict_fake_news")
    def test_classify_returns_text_snippet(self, mock_predict):
        """classify should include a truncated text snippet."""
        mock_predict.return_value = {"label": LABEL_REAL, "confidence": 90.0}
        detector = FakeNewsDetector()
        long_text = "A" * 500
        result = detector.classify(long_text)
        assert len(result["text"]) <= 200

    @patch("ml_models.fake_news_detector.model.predict_fake_news")
    def test_classify_batch(self, mock_predict):
        """classify_batch should classify multiple texts."""
        mock_predict.side_effect = [
            {"label": LABEL_REAL, "confidence": 90.0},
            {"label": LABEL_FAKE, "confidence": 85.0},
        ]
        detector = FakeNewsDetector()
        results = detector.classify_batch(["Real news text", "Fake news text"])
        assert len(results) == 2
        assert results[0]["label"] == LABEL_REAL
        assert results[1]["label"] == LABEL_FAKE

    @patch("ml_models.fake_news_detector.model.predict_fake_news")
    @patch("ml_models.fake_news_detector.model.get_classifier")
    def test_classify_detailed(self, mock_get_classifier, mock_predict):
        """classify_detailed should return all scores and metadata."""
        mock_classifier = MagicMock()
        mock_classifier.return_value = [
            {"label": "NEGATIVE", "score": 0.15},
            {"label": "POSITIVE", "score": 0.85},
        ]
        mock_get_classifier.return_value = mock_classifier
        mock_predict.return_value = {"label": LABEL_REAL, "confidence": 85.0}

        detector = FakeNewsDetector()
        result = detector.classify_detailed("Detailed analysis text")
        assert "primary_label" in result
        assert "primary_confidence" in result
        assert "all_scores" in result
        assert "is_uncertain" in result
        assert "effective_label" in result
        assert "text_length" in result
        assert result["text_length"] > 0

    @patch("ml_models.fake_news_detector.model.predict_fake_news")
    def test_get_statistics(self, mock_predict):
        """get_statistics should aggregate classification results."""
        results = [
            {"label": LABEL_FAKE, "confidence": 80.0, "is_uncertain": False},
            {"label": LABEL_FAKE, "confidence": 70.0, "is_uncertain": False},
            {"label": LABEL_REAL, "confidence": 90.0, "is_uncertain": False},
            {"label": LABEL_REAL, "confidence": 30.0, "is_uncertain": True},
        ]
        detector = FakeNewsDetector()
        stats = detector.get_statistics(results)
        assert stats["total"] == 4
        assert stats["fake_count"] == 2
        assert stats["real_count"] == 2
        assert stats["uncertain_count"] == 1
        assert "distribution" in stats
        assert "avg_confidence" in stats

    @patch("ml_models.fake_news_detector.model.predict_fake_news")
    def test_get_statistics_empty(self, mock_predict):
        """get_statistics should handle empty results."""
        detector = FakeNewsDetector()
        stats = detector.get_statistics([])
        assert stats["total"] == 0
        assert stats["fake_count"] == 0
        assert stats["real_count"] == 0
        assert stats["avg_confidence"] == 0.0

    def test_default_confidence_threshold(self):
        """Default confidence threshold should be 50.0."""
        detector = FakeNewsDetector()
        assert detector.confidence_threshold == DEFAULT_CONFIDENCE_THRESHOLD

    def test_custom_confidence_threshold(self):
        """Custom confidence threshold should be stored."""
        detector = FakeNewsDetector(confidence_threshold=75.0)
        assert detector.confidence_threshold == 75.0


class TestFakeNewsConvenienceFunctions:
    """Tests for module-level convenience functions."""

    @patch("ml_models.fake_news_detector.model.predict_fake_news")
    def test_classify_text_default(self, mock_predict):
        """classify_text should use default detector."""
        mock_predict.return_value = {"label": LABEL_REAL, "confidence": 90.0}
        result = classify_text("Real news text")
        assert result["label"] == LABEL_REAL

    @patch("ml_models.fake_news_detector.model.predict_fake_news")
    def test_classify_text_custom_threshold(self, mock_predict):
        """classify_text with custom threshold should override default."""
        mock_predict.return_value = {"label": LABEL_FAKE, "confidence": 40.0}
        result = classify_text("Uncertain text", confidence_threshold=50.0)
        assert result["is_uncertain"] is True

    @patch("ml_models.fake_news_detector.model.predict_fake_news")
    def test_classify_texts_batch(self, mock_predict):
        """classify_texts should classify multiple texts."""
        mock_predict.side_effect = [
            {"label": LABEL_REAL, "confidence": 90.0},
            {"label": LABEL_FAKE, "confidence": 85.0},
        ]
        results = classify_texts(["Text 1", "Text 2"])
        assert len(results) == 2


# ─── TopicModeler Tests ───────────────────────────────────────────────────


class TestTopicModeler:
    """Tests for the TopicModeler class with mocked BERTopic."""

    @patch("ml_models.topic_modeler.model.predict_topics")
    def test_analyze(self, mock_predict):
        """analyze should return enriched topic results."""
        mock_predict.return_value = [
            {"text": "AI advances in 2024", "topic": 0, "keywords": ["AI", "technology", "advances"]},
            {"text": "Climate change impacts", "topic": 1, "keywords": ["climate", "change", "impacts"]},
        ]
        modeler = TopicModeler()
        results = modeler.analyze(["AI advances in 2024", "Climate change impacts"])
        assert len(results) == 2
        assert results[0]["topic"] == 0
        assert results[0]["topic_label"] == "AI / technology / advances"
        assert results[0]["confidence"] > 0
        assert results[1]["topic"] == 1

    @patch("ml_models.topic_modeler.model.predict_topics")
    def test_analyze_single(self, mock_predict):
        """analyze_single should return a single enriched result."""
        mock_predict.return_value = [
            {"text": "AI news", "topic": 0, "keywords": ["AI", "news"]},
        ]
        modeler = TopicModeler()
        result = modeler.analyze_single("AI news")
        assert result["topic"] == 0
        assert "topic_label" in result
        assert "confidence" in result

    @patch("ml_models.topic_modeler.model.predict_topics")
    def test_analyze_single_with_context(self, mock_predict):
        """analyze_single with context should analyze combined texts and return last result."""
        mock_predict.return_value = [
            {"text": "Climate change", "topic": 0, "keywords": ["climate"]},
            {"text": "AI news", "topic": 1, "keywords": ["AI", "news"]},
        ]
        modeler = TopicModeler()
        result = modeler.analyze_single("AI news", context_texts=["Climate change"])
        assert result["topic"] == 1
        assert "AI" in result["topic_label"]

    @patch("ml_models.topic_modeler.model.predict_topics")
    def test_analyze_single_empty_result(self, mock_predict):
        """analyze_single should return default dict when no results."""
        mock_predict.return_value = []
        modeler = TopicModeler()
        result = modeler.analyze_single("Short text")
        assert result["topic"] == -1
        assert result["topic_label"] == "unclassified"
        assert result["confidence"] == 0.0

    @patch("ml_models.topic_modeler.model.predict_topics")
    def test_get_topic_summary(self, mock_predict):
        """get_topic_summary should group results by topic."""
        mock_predict.return_value = [
            {"text": "AI news 1", "topic": 0, "keywords": ["AI", "technology"]},
            {"text": "AI news 2", "topic": 0, "keywords": ["AI", "technology"]},
            {"text": "Climate news", "topic": 1, "keywords": ["climate", "change"]},
            {"text": "Random post", "topic": -1, "keywords": []},
        ]
        modeler = TopicModeler()
        summary = modeler.get_topic_summary(["AI news 1", "AI news 2", "Climate news", "Random post"])
        assert summary["total_texts"] == 4
        assert summary["total_topics"] == 2  # topics 0 and 1 (excluding -1)
        assert summary["outlier_count"] == 1
        assert len(summary["topics"]) == 3  # topic 0, topic 1, and outlier -1

    @patch("ml_models.topic_modeler.model.predict_topics")
    def test_detect_trends_no_comparison(self, mock_predict):
        """detect_trends without previous texts should mark all as new."""
        mock_predict.return_value = [
            {"text": "AI news", "topic": 0, "keywords": ["AI"]},
            {"text": "Random", "topic": -1, "keywords": []},
        ]
        modeler = TopicModeler()
        trends = modeler.detect_trends(["AI news", "Random"])
        assert trends["has_comparison"] is False
        # Only non-outlier topics should appear
        assert len(trends["trending_topics"]) == 1
        assert trends["trending_topics"][0]["is_new"] is True
        assert trends["trending_topics"][0]["growth"] is None

    @patch("ml_models.topic_modeler.model.predict_topics")
    def test_detect_trends_with_comparison(self, mock_predict):
        """detect_trends with previous texts should calculate growth."""
        # First call for current texts
        # Second call for previous texts
        mock_predict.side_effect = [
            # Current
            [
                {"text": "AI news 1", "topic": 0, "keywords": ["AI"]},
                {"text": "AI news 2", "topic": 0, "keywords": ["AI"]},
                {"text": "New topic", "topic": 2, "keywords": ["new"]},
            ],
            # Previous
            [
                {"text": "AI old", "topic": 0, "keywords": ["AI"]},
            ],
        ]
        modeler = TopicModeler()
        trends = modeler.detect_trends(
            current_texts=["AI news 1", "AI news 2", "New topic"],
            previous_texts=["AI old"],
        )
        assert trends["has_comparison"] is True
        # Topic 0 existed before — should have growth
        topic_0 = [t for t in trends["trending_topics"] if t["topic"] == 0][0]
        assert topic_0["growth"] is not None
        assert topic_0["is_new"] is False
        # Topic 2 is new
        topic_2 = [t for t in trends["trending_topics"] if t["topic"] == 2][0]
        assert topic_2["is_new"] is True
        assert topic_2["growth"] is None

    def test_enrich_result_outlier(self):
        """_enrich_result should label outlier topics as 'unclassified'."""
        modeler = TopicModeler()
        result = modeler._enrich_result({"text": "test", "topic": -1, "keywords": []})
        assert result["topic_label"] == "unclassified"
        assert result["confidence"] == 0.0

    def test_enrich_result_with_keywords(self):
        """_enrich_result should create label from top 3 keywords."""
        modeler = TopicModeler(top_n_words=5)
        result = modeler._enrich_result({
            "text": "test",
            "topic": 0,
            "keywords": ["AI", "technology", "machine", "learning", "deep"],
        })
        assert result["topic_label"] == "AI / technology / machine"
        assert result["confidence"] == 1.0  # 5 keywords / 5 top_n_words

    def test_enrich_result_no_keywords(self):
        """_enrich_result should use topic ID when no keywords."""
        modeler = TopicModeler()
        result = modeler._enrich_result({"text": "test", "topic": 3, "keywords": []})
        assert result["topic_label"] == "topic_3"
        assert result["confidence"] == 0.0

    def test_default_settings(self):
        """TopicModeler should have default min_topic_size and top_n_words."""
        modeler = TopicModeler()
        assert modeler.min_topic_size == 10
        assert modeler.top_n_words == 5

    def test_custom_settings(self):
        """TopicModeler should accept custom settings."""
        modeler = TopicModeler(min_topic_size=20, top_n_words=10)
        assert modeler.min_topic_size == 20
        assert modeler.top_n_words == 10


class TestTopicModelerConvenienceFunctions:
    """Tests for module-level convenience functions."""

    @patch("ml_models.topic_modeler.model.predict_topics")
    def test_analyze_topics(self, mock_predict):
        """analyze_topics should use default modeler."""
        mock_predict.return_value = [
            {"text": "AI news", "topic": 0, "keywords": ["AI"]},
        ]
        from ml_models.topic_modeler.model import analyze_topics

        results = analyze_topics(["AI news"])
        assert len(results) == 1

    @patch("ml_models.topic_modeler.model.predict_topics")
    def test_analyze_topics_custom_min_size(self, mock_predict):
        """analyze_topics with custom min_topic_size should create new modeler."""
        mock_predict.return_value = [
            {"text": "AI news", "topic": 0, "keywords": ["AI"]},
        ]
        from ml_models.topic_modeler.model import analyze_topics

        results = analyze_topics(["AI news"], min_topic_size=20)
        assert len(results) == 1
        # Verify custom min_topic_size was passed
        mock_predict.assert_called_with(["AI news"], min_topic_size=20)

    @patch("ml_models.topic_modeler.model.predict_topics")
    def test_get_topic_overview(self, mock_predict):
        """get_topic_overview should return a topic summary."""
        mock_predict.return_value = [
            {"text": "AI news", "topic": 0, "keywords": ["AI"]},
        ]
        from ml_models.topic_modeler.model import get_topic_overview

        summary = get_topic_overview(["AI news"])
        assert "total_texts" in summary
        assert "topics" in summary

    @patch("ml_models.topic_modeler.model.predict_topics")
    def test_detect_topic_trends(self, mock_predict):
        """detect_topic_trends should detect trends."""
        mock_predict.return_value = [
            {"text": "AI news", "topic": 0, "keywords": ["AI"]},
        ]
        from ml_models.topic_modeler.model import detect_topic_trends

        trends = detect_topic_trends(["AI news"])
        assert "trending_topics" in trends
        assert "has_comparison" in trends


# ─── Base predict module Tests ────────────────────────────────────────────


class TestSentimentPredict:
    """Tests for the base sentiment predict module (VADER fallback)."""

    def test_predict_sentiment_returns_dict(self):
        """predict_sentiment should return a dict with label, score, model."""
        from ml_models.sentiment.predict import predict_sentiment

        result = predict_sentiment("I love this amazing technology!")
        assert "label" in result
        assert "score" in result
        assert "model" in result

    def test_predict_sentiment_positive(self):
        """predict_sentiment should detect positive sentiment."""
        from ml_models.sentiment.predict import predict_sentiment

        result = predict_sentiment("I absolutely love this new technology!")
        assert result["label"] == "positive"

    def test_predict_sentiment_negative(self):
        """predict_sentiment should detect negative sentiment."""
        from ml_models.sentiment.predict import predict_sentiment

        result = predict_sentiment("This is the worst thing ever, I hate it")
        assert result["label"] == "negative"

    def test_predict_sentiment_neutral(self):
        """predict_sentiment should handle neutral text."""
        from ml_models.sentiment.predict import predict_sentiment

        result = predict_sentiment("The meeting is scheduled for tomorrow")
        assert result["label"] in ["neutral", "positive", "negative"]
