"""Fake News Detector — Higher-level wrapper for fake news detection.

Provides batch processing, confidence thresholds, and detailed analysis
beyond the basic predict module.
"""

from ml_models.fake_news_detector.model import FakeNewsDetector

__all__ = ["FakeNewsDetector"]