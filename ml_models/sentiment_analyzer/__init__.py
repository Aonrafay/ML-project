"""Sentiment Analyzer — Higher-level wrapper for sentiment analysis.

Provides VADER/DistilBERT switching, batch processing, aggregation,
and timeline analysis beyond the basic predict module.
"""

from ml_models.sentiment_analyzer.model import SentimentAnalyzer

__all__ = ["SentimentAnalyzer"]