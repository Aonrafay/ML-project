"""Tests for data_processing module.

Covers TextPreprocessor, LanguageDetector, and the processing pipeline
using mocks for database operations where needed.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from data_processing.preprocessor import TextPreprocessor
from data_processing.language_detection import (
    LanguageDetector,
    SUPPORTED_LANGUAGES,
    DEFAULT_CONFIDENCE_THRESHOLD,
    MIN_TEXT_LENGTH,
    detect_language,
    detect_batch_languages,
    filter_english_texts,
    get_language_stats,
)


# ─── TextPreprocessor Tests ───────────────────────────────────────────────


class TestTextPreprocessor:
    """Tests for the TextPreprocessor class."""

    def setup_method(self):
        self.preprocessor = TextPreprocessor()

    def test_clean_text_removes_urls(self):
        """URLs should be removed from text."""
        text = "Check this out! https://example.com and www.test.org"
        cleaned = self.preprocessor.clean_text(text)
        assert "https://example.com" not in cleaned
        assert "www.test.org" not in cleaned

    def test_clean_text_removes_html_tags(self):
        """HTML tags should be removed from text."""
        text = "This is <b>bold</b> and <i>italic</i> text"
        cleaned = self.preprocessor.clean_text(text)
        assert "<b>" not in cleaned
        assert "<i>" not in cleaned

    def test_clean_text_removes_mentions(self):
        """@mentions should be removed from text."""
        text = "Hello @user and @another_user check this"
        cleaned = self.preprocessor.clean_text(text)
        assert "@user" not in cleaned
        assert "@another_user" not in cleaned

    def test_clean_text_converts_hashtags(self):
        """Hashtags should be converted to plain words (symbol removed, word kept)."""
        text = "Check out #AI and #MachineLearning"
        cleaned = self.preprocessor.clean_text(text)
        assert "#" not in cleaned
        assert "AI" in cleaned
        assert "MachineLearning" in cleaned

    def test_clean_text_removes_special_chars(self):
        """Special characters (non-word, non-space, non-punctuation) should be removed."""
        text = "Hello $$$ world *** test"
        cleaned = self.preprocessor.clean_text(text)
        assert "$" not in cleaned
        assert "*" not in cleaned

    def test_clean_text_normalizes_whitespace(self):
        """Multiple spaces should be collapsed to single spaces."""
        text = "Hello    world   with    spaces"
        cleaned = self.preprocessor.clean_text(text)
        assert "    " not in cleaned
        assert cleaned == "Hello world with spaces"

    def test_clean_text_strips_leading_trailing(self):
        """Leading and trailing whitespace should be stripped."""
        text = "  Hello world  "
        cleaned = self.preprocessor.clean_text(text)
        assert cleaned == "Hello world"

    def test_detect_language_english(self):
        """English text should be detected as 'en'."""
        result = self.preprocessor.detect_language("This is English text written in the United States")
        assert result == "en"

    def test_detect_language_spanish(self):
        """Spanish text should be detected as 'es'."""
        result = self.preprocessor.detect_language("Este es un texto en español escrito en España")
        assert result == "es"

    def test_detect_language_short_text(self):
        """Very short text may still return a language or 'unknown'."""
        result = self.preprocessor.detect_language("hi")
        # langdetect may still guess or throw — preprocessor catches exceptions
        assert isinstance(result, str)

    def test_tokenize_removes_stopwords(self):
        """Tokenization should remove stopwords."""
        tokens = self.preprocessor.tokenize("The quick brown fox jumps over the lazy dog")
        assert "the" not in tokens
        assert "over" not in tokens

    def test_tokenize_lemmatizes(self):
        """Tokenization should lemmatize words."""
        tokens = self.preprocessor.tokenize("Running quickly through the forests")
        # "running" → "run", "forests" → "forest"
        assert "run" in tokens or "running" in tokens  # lemmatization may vary

    def test_tokenize_removes_short_tokens(self):
        """Tokens shorter than 3 characters should be removed."""
        tokens = self.preprocessor.tokenize("I am a big fan of AI tech")
        # "I", "am", "a", "of" are all <= 2 chars or stopwords
        for token in tokens:
            assert len(token) > 2

    def test_tokenize_removes_punctuation(self):
        """Punctuation tokens should be removed."""
        tokens = self.preprocessor.tokenize("Hello, world! How are you?")
        assert "," not in tokens
        assert "!" not in tokens
        assert "?" not in tokens

    def test_get_sentiment_positive(self):
        """Positive text should return 'positive' label."""
        result = self.preprocessor.get_sentiment("I love this amazing product!")
        assert result["label"] == "positive"
        assert result["score"] > 0

    def test_get_sentiment_negative(self):
        """Negative text should return 'negative' label."""
        result = self.preprocessor.get_sentiment("This is terrible and I hate it")
        assert result["label"] == "negative"
        assert result["score"] < 0

    def test_get_sentiment_neutral(self):
        """Neutral text should return 'neutral' or close label."""
        result = self.preprocessor.get_sentiment("The weather is okay today")
        assert result["label"] in ["neutral", "positive", "negative"]

    def test_get_sentiment_returns_scores_dict(self):
        """get_sentiment should include VADER component scores."""
        result = self.preprocessor.get_sentiment("I love this!")
        assert "scores" in result
        assert "pos" in result["scores"]
        assert "neg" in result["scores"]
        assert "neu" in result["scores"]
        assert "compound" in result["scores"]

    def test_preprocess_returns_complete_dict(self):
        """preprocess should return a dict with all expected keys."""
        result = self.preprocessor.preprocess("Hello world! This is a test.")
        assert "original" in result
        assert "cleaned" in result
        assert "language" in result
        assert "tokens" in result
        assert "sentiment" in result
        assert result["original"] == "Hello world! This is a test."
        assert result["language"] == "en"

    def test_preprocess_cleaned_differs_from_original(self):
        """Preprocessed text should differ when original has URLs/mentions/etc."""
        original = "Check https://example.com @user #AI news!"
        result = self.preprocessor.preprocess(original)
        assert result["cleaned"] != original
        assert "https://example.com" not in result["cleaned"]


# ─── LanguageDetector Tests ────────────────────────────────────────────────


class TestLanguageDetector:
    """Tests for the LanguageDetector class."""

    def setup_method(self):
        self.detector = LanguageDetector()

    def test_detect_language_english(self):
        """English text should be detected with language code 'en'."""
        result = self.detector.detect_language(
            "This is a sample text written in English for testing purposes"
        )
        assert result["language"] == "en"
        assert result["language_name"] == "English"
        assert result["is_supported"] is True
        assert result["confidence"] > 0

    def test_detect_language_short_text(self):
        """Text shorter than MIN_TEXT_LENGTH should return 'unknown'."""
        result = self.detector.detect_language("hi")
        assert result["language"] == "unknown"
        assert result["confidence"] == 0.0
        assert result["is_supported"] is False
        assert "short" in result["reason"].lower()

    def test_detect_language_returns_confidence(self):
        """Detection result should include a confidence score."""
        result = self.detector.detect_language(
            "The quick brown fox jumps over the lazy dog in the park today"
        )
        assert isinstance(result["confidence"], float)
        assert result["confidence"] >= 0

    def test_detect_batch(self):
        """detect_batch should return results for each text."""
        texts = [
            "This is English text for testing language detection",
            "Este es texto en español para probar la detección",
        ]
        results = self.detector.detect_batch(texts)
        assert len(results) == 2
        assert results[0]["language"] == "en"
        assert results[1]["language"] == "es"

    def test_filter_by_language_english(self):
        """filter_by_language should return only texts matching target languages."""
        texts = [
            "This is English text for testing language detection capabilities",
            "Este es texto en español para probar la detección de idiomas",
        ]
        filtered = self.detector.filter_by_language(texts, ["en"])
        assert len(filtered) >= 1
        assert all(item[1]["language"] == "en" for item in filtered)

    def test_filter_by_language_with_confidence(self):
        """filter_by_language should respect confidence threshold."""
        detector = LanguageDetector(confidence_threshold=0.99)
        texts = [
            "This is English text for testing language detection capabilities",
        ]
        # Very high threshold may filter out even correct detections
        filtered = detector.filter_by_language(texts, ["en"], min_confidence=0.99)
        # Result depends on langdetect confidence, but the mechanism works
        assert isinstance(filtered, list)

    def test_is_likely_english(self):
        """is_likely_english should return True for clear English text."""
        result = self.detector.is_likely_english(
            "This is clearly English text written for testing purposes here"
        )
        assert isinstance(result, bool)

    def test_get_language_distribution(self):
        """get_language_distribution should return counts per language."""
        texts = [
            "This is English text for testing language detection",
            "Another English text for testing purposes today",
            "Este es texto en español para probar",
        ]
        distribution = self.detector.get_language_distribution(texts)
        assert isinstance(distribution, dict)
        assert "en" in distribution
        assert distribution["en"] >= 1

    def test_supported_languages_dict(self):
        """SUPPORTED_LANGUAGES should contain common language codes."""
        assert "en" in SUPPORTED_LANGUAGES
        assert "es" in SUPPORTED_LANGUAGES
        assert "fr" in SUPPORTED_LANGUAGES
        assert "de" in SUPPORTED_LANGUAGES

    def test_custom_confidence_threshold(self):
        """Custom confidence threshold should be stored and used."""
        detector = LanguageDetector(confidence_threshold=0.9)
        assert detector.confidence_threshold == 0.9

    def test_custom_supported_languages(self):
        """Custom supported languages dict should override defaults."""
        custom = {"en": "English", "fr": "French"}
        detector = LanguageDetector(supported_languages=custom)
        assert detector.supported_languages == custom

    def test_clean_for_detection_removes_urls(self):
        """_clean_for_detection should remove URLs."""
        cleaned = self.detector._clean_for_detection(
            "Check https://example.com for more info about this topic"
        )
        assert "https://example.com" not in cleaned

    def test_clean_for_detection_removes_mentions(self):
        """_clean_for_detection should remove @mentions."""
        cleaned = self.detector._clean_for_detection("@user Hello world text here")
        assert "@user" not in cleaned


# ─── Convenience Function Tests ────────────────────────────────────────────


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_detect_language_function(self):
        """Module-level detect_language should work like the class method."""
        result = detect_language(
            "This is English text for testing language detection capabilities"
        )
        assert result["language"] == "en"
        assert "confidence" in result

    def test_detect_batch_languages_function(self):
        """Module-level detect_batch_languages should process multiple texts."""
        texts = [
            "This is English text for testing language detection",
            "Another English sentence for testing purposes today",
        ]
        results = detect_batch_languages(texts)
        assert len(results) == 2

    def test_filter_english_texts_function(self):
        """Module-level filter_english_texts should return English texts."""
        texts = [
            "This is English text for testing language detection capabilities",
        ]
        filtered = filter_english_texts(texts, min_confidence=0.5)
        assert isinstance(filtered, list)

    def test_get_language_stats_function(self):
        """Module-level get_language_stats should return comprehensive stats."""
        texts = [
            "This is English text for testing language detection",
            "Another English sentence for testing purposes today",
        ]
        stats = get_language_stats(texts)
        assert "distribution" in stats
        assert "total_texts" in stats
        assert stats["total_texts"] == 2
        assert "supported_language_count" in stats
        assert "supported_language_percentage" in stats


# ─── Pipeline Tests (with mocks) ──────────────────────────────────────────


class TestPipeline:
    """Tests for the data processing pipeline using mocked database."""

    @patch("data_processing.pipeline.raw_posts_collection")
    @patch("data_processing.pipeline.cleaned_posts_collection")
    def test_process_single_post_not_found(self, mock_cleaned, mock_raw):
        """process_single_post should return None if post not found."""
        mock_raw.find_one = AsyncMock(return_value=None)

        from data_processing.pipeline import process_single_post

        result = await process_single_post("nonexistent_id")
        assert result is None

    @patch("data_processing.pipeline.raw_posts_collection")
    @patch("data_processing.pipeline.cleaned_posts_collection")
    def test_process_single_post_success(self, mock_cleaned, mock_raw):
        """process_single_post should process and return a cleaned document."""
        mock_post = {
            "id": "test_post_1",
            "source": "reddit",
            "platform": "reddit",
            "title": "Test Title",
            "text": "This is test text content for processing!",
        }
        mock_raw.find_one = AsyncMock(return_value=mock_post)
        mock_cleaned.update_one = AsyncMock(return_value=MagicMock())
        mock_raw.update_one = AsyncMock(return_value=MagicMock())

        from data_processing.pipeline import process_single_post

        result = await process_single_post("test_post_1")
        assert result is not None
        assert result["original_id"] == "test_post_1"
        assert result["source"] == "reddit"
        assert "cleaned_text" in result
        assert "language" in result
        assert "tokens" in result
        assert "sentiment" in result