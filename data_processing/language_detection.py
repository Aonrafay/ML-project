"""Language detection module for SocialPulse AI.

Provides advanced language detection, filtering, and confidence scoring
for social media text processing. Extends the basic langdetect integration
in the preprocessor with additional features like batch detection,
confidence thresholds, and language filtering.
"""

from langdetect import detect, detect_langs, LangDetectException
from typing import Optional
import re


# Supported languages with their codes and names
SUPPORTED_LANGUAGES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese",
    "ar": "Arabic",
    "hi": "Hindi",
    "tr": "Turkish",
    "nl": "Dutch",
    "pl": "Polish",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "el": "Greek",
    "cs": "Czech",
}

# Minimum text length for reliable detection
MIN_TEXT_LENGTH = 20

# Default confidence threshold for language detection
DEFAULT_CONFIDENCE_THRESHOLD = 0.7


class LanguageDetector:
    """Advanced language detection with confidence scoring and filtering.

    Provides methods for single and batch language detection, confidence
    estimation, and language-based filtering of social media posts.
    """

    def __init__(
        self,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        supported_languages: Optional[dict] = None,
    ):
        """Initialize the LanguageDetector.

        Args:
            confidence_threshold: Minimum confidence score to accept a detection.
            supported_languages: Dict of language codes to names. Defaults to SUPPORTED_LANGUAGES.
        """
        self.confidence_threshold = confidence_threshold
        self.supported_languages = supported_languages or SUPPORTED_LANGUAGES

    def detect_language(self, text: str) -> dict:
        """Detect the language of a single text with confidence scoring.

        Args:
            text: The text to analyze.

        Returns:
            Dict with language code, name, confidence, and is_supported flag.
        """
        cleaned = self._clean_for_detection(text)

        if len(cleaned) < MIN_TEXT_LENGTH:
            return {
                "language": "unknown",
                "language_name": "Unknown",
                "confidence": 0.0,
                "is_supported": False,
                "reason": "Text too short for reliable detection",
            }

        try:
            # Get primary detection
            primary_lang = detect(cleaned)

            # Get probability distribution across languages
            lang_probs = detect_langs(cleaned)
            confidence = 0.0

            for prob in lang_probs:
                if prob.lang == primary_lang:
                    confidence = prob.prob
                    break

            # If no exact match found in probs, use highest probability
            if confidence == 0.0 and lang_probs:
                confidence = max(lang_probs, key=lambda x: x.prob).prob

            return {
                "language": primary_lang,
                "language_name": self.supported_languages.get(primary_lang, "Unknown"),
                "confidence": round(confidence, 4),
                "is_supported": primary_lang in self.supported_languages,
                "reason": None,
            }

        except LangDetectException:
            return {
                "language": "unknown",
                "language_name": "Unknown",
                "confidence": 0.0,
                "is_supported": False,
                "reason": "Language detection failed",
            }

    def detect_batch(self, texts: list[str]) -> list[dict]:
        """Detect languages for a batch of texts.

        Args:
            texts: List of texts to analyze.

        Returns:
            List of detection result dicts.
        """
        return [self.detect_language(text) for text in texts]

    def filter_by_language(
        self,
        texts: list[str],
        target_languages: list[str],
        min_confidence: Optional[float] = None,
    ) -> list[tuple[str, dict]]:
        """Filter texts by target languages with confidence threshold.

        Args:
            texts: List of texts to filter.
            target_languages: List of language codes to keep (e.g., ["en", "es"]).
            min_confidence: Override confidence threshold for this filter.

        Returns:
            List of (text, detection_result) tuples matching the criteria.
        """
        threshold = min_confidence or self.confidence_threshold
        results = []

        for text in texts:
            detection = self.detect_language(text)
            if (
                detection["language"] in target_languages
                and detection["confidence"] >= threshold
            ):
                results.append((text, detection))

        return results

    def is_likely_english(self, text: str, min_confidence: Optional[float] = None) -> bool:
        """Quick check if text is likely English.

        Args:
            text: The text to check.
            min_confidence: Override confidence threshold.

        Returns:
            True if text is detected as English above the confidence threshold.
        """
        threshold = min_confidence or self.confidence_threshold
        detection = self.detect_language(text)
        return detection["language"] == "en" and detection["confidence"] >= threshold

    def get_language_distribution(self, texts: list[str]) -> dict[str, int]:
        """Get the distribution of languages across a batch of texts.

        Args:
            texts: List of texts to analyze.

        Returns:
            Dict mapping language codes to counts.
        """
        distribution = {}
        for text in texts:
            detection = self.detect_language(text)
            lang = detection["language"]
            distribution[lang] = distribution.get(lang, 0) + 1
        return distribution

    def _clean_for_detection(self, text: str) -> str:
        """Clean text for more reliable language detection.

        Removes URLs, mentions, hashtags, and emoji that can confuse
        language detection algorithms.

        Args:
            text: Raw text to clean.

        Returns:
            Cleaned text suitable for language detection.
        """
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove @mentions
        text = re.sub(r'@\w+', '', text)
        # Remove hashtags (keep the word part)
        text = re.sub(r'#(\w+)', r'\1', text)
        # Remove emoji and special unicode
        text = re.sub(r'[^\w\s.,!?\'"-]', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text


# Singleton instance for convenience
_default_detector = LanguageDetector()


def detect_language(text: str) -> dict:
    """Detect the language of a text using the default detector.

    Args:
        text: The text to analyze.

    Returns:
        Dict with language code, name, confidence, and is_supported flag.
    """
    return _default_detector.detect_language(text)


def detect_batch_languages(texts: list[str]) -> list[dict]:
    """Detect languages for a batch of texts using the default detector.

    Args:
        texts: List of texts to analyze.

    Returns:
        List of detection result dicts.
    """
    return _default_detector.detect_batch(texts)


def filter_english_texts(texts: list[str], min_confidence: float = 0.7) -> list[tuple[str, dict]]:
    """Filter texts to only include those detected as English.

    Args:
        texts: List of texts to filter.
        min_confidence: Minimum confidence threshold.

    Returns:
        List of (text, detection_result) tuples for English texts.
    """
    detector = LanguageDetector(confidence_threshold=min_confidence)
    return detector.filter_by_language(texts, ["en"], min_confidence)


def get_language_stats(texts: list[str]) -> dict:
    """Get comprehensive language statistics for a batch of texts.

    Args:
        texts: List of texts to analyze.

    Returns:
        Dict with distribution, total count, and supported language percentage.
    """
    detector = LanguageDetector()
    distribution = detector.get_language_distribution(texts)
    total = len(texts)
    supported_count = sum(
        distribution.get(lang, 0) for lang in detector.supported_languages
    )

    return {
        "distribution": distribution,
        "total_texts": total,
        "supported_language_count": supported_count,
        "supported_language_percentage": round(supported_count / total * 100, 2) if total > 0 else 0.0,
        "unknown_count": distribution.get("unknown", 0),
    }