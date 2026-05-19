"""Topic Modeler model wrapper.

Provides a higher-level interface for topic modeling with topic
labeling, trend detection, batch analysis, and topic comparison
capabilities beyond the base BERTopic module.
"""

from ml_models.topic_modeling.predict import predict_topics, get_topic_model
from typing import Optional
from datetime import datetime


# Default settings
DEFAULT_MIN_TOPIC_SIZE = 10
DEFAULT_TOP_N_WORDS = 5


class TopicModeler:
    """Higher-level topic modeling wrapper.

    Wraps the base BERTopic prediction module with additional features
    like topic labeling, trend detection, and batch analysis.
    """

    def __init__(
        self,
        min_topic_size: int = DEFAULT_MIN_TOPIC_SIZE,
        top_n_words: int = DEFAULT_TOP_N_WORDS,
    ):
        """Initialize the TopicModeler.

        Args:
            min_topic_size: Minimum documents per topic for BERTopic.
            top_n_words: Number of top keywords to extract per topic.
        """
        self.min_topic_size = min_topic_size
        self.top_n_words = top_n_words
        self._model = None

    def analyze(self, texts: list[str]) -> list[dict]:
        """Analyze topics for a list of texts.

        Args:
            texts: List of texts to analyze.

        Returns:
            List of topic analysis result dicts with labeled topics.
        """
        raw_results = predict_topics(texts, min_topic_size=self.min_topic_size)
        return [self._enrich_result(r) for r in raw_results]

    def analyze_single(self, text: str, context_texts: Optional[list[str]] = None) -> dict:
        """Analyze topic for a single text, optionally with context.

        If context_texts are provided, the text is analyzed together with
        the context for better topic assignment.

        Args:
            text: The text to analyze.
            context_texts: Optional additional texts for context.

        Returns:
            Topic analysis result dict.
        """
        if context_texts:
            all_texts = context_texts + [text]
            results = predict_topics(all_texts, min_topic_size=self.min_topic_size)
            # Return the result for the target text (last item)
            if results:
                return self._enrich_result(results[-1])

        # Single text analysis (may assign to outlier topic -1)
        results = predict_topics([text], min_topic_size=self.min_topic_size)
        if results:
            return self._enrich_result(results[0])

        return {
            "text": text[:200],
            "topic": -1,
            "topic_label": "unclassified",
            "keywords": [],
            "confidence": 0.0,
        }

    def get_topic_summary(self, texts: list[str]) -> dict:
        """Get a summary of all topics discovered in the texts.

        Args:
            texts: List of texts to analyze.

        Returns:
            Dict with topic summary, counts, and top keywords per topic.
        """
        results = self.analyze(texts)

        # Group by topic
        topic_groups = {}
        for r in results:
            topic_id = r["topic"]
            if topic_id not in topic_groups:
                topic_groups[topic_id] = {
                    "topic": topic_id,
                    "label": r["topic_label"],
                    "keywords": r["keywords"],
                    "count": 0,
                    "texts": [],
                }
            topic_groups[topic_id]["count"] += 1
            topic_groups[topic_id]["texts"].append(r["text"][:100])

        # Sort by count (most popular topics first)
        sorted_topics = sorted(
            topic_groups.values(),
            key=lambda x: x["count"],
            reverse=True,
        )

        total = len(results)
        outlier_count = topic_groups.get(-1, {}).get("count", 0)

        return {
            "total_texts": total,
            "total_topics": len(topic_groups) - (1 if -1 in topic_groups else 0),
            "outlier_count": outlier_count,
            "topics": sorted_topics,
        }

    def detect_trends(
        self,
        current_texts: list[str],
        previous_texts: Optional[list[str]] = None,
    ) -> dict:
        """Detect trending topics by comparing current vs previous texts.

        Args:
            current_texts: Current batch of texts.
            previous_texts: Previous batch of texts for comparison.

        Returns:
            Dict with trending topics and their growth metrics.
        """
        current_summary = self.get_topic_summary(current_texts)

        if previous_texts is None:
            # No comparison data — return current topics as new trends
            return {
                "trending_topics": [
                    {
                        "topic": t["topic"],
                        "label": t["label"],
                        "keywords": t["keywords"],
                        "current_count": t["count"],
                        "growth": None,
                        "is_new": True,
                    }
                    for t in current_summary["topics"]
                    if t["topic"] != -1
                ],
                "total_current_texts": current_summary["total_texts"],
                "has_comparison": False,
            }

        previous_summary = self.get_topic_summary(previous_texts)

        # Build previous topic lookup
        prev_topic_counts = {}
        for t in previous_summary["topics"]:
            prev_topic_counts[t["topic"]] = t["count"]

        # Calculate growth for each current topic
        trending = []
        for t in current_summary["topics"]:
            if t["topic"] == -1:
                continue

            current_count = t["count"]
            prev_count = prev_topic_counts.get(t["topic"], 0)

            if prev_count > 0:
                growth = round((current_count - prev_count) / prev_count * 100, 2)
                is_new = False
            else:
                growth = None
                is_new = True

            trending.append({
                "topic": t["topic"],
                "label": t["label"],
                "keywords": t["keywords"],
                "current_count": current_count,
                "previous_count": prev_count,
                "growth": growth,
                "is_new": is_new,
            })

        # Sort by growth (highest first), new topics first
        trending.sort(key=lambda x: (not x["is_new"], -(x["growth"] or 0)))

        return {
            "trending_topics": trending,
            "total_current_texts": current_summary["total_texts"],
            "total_previous_texts": previous_summary["total_texts"],
            "has_comparison": True,
        }

    def _enrich_result(self, raw_result: dict) -> dict:
        """Enrich a raw topic result with a human-readable label.

        Args:
            raw_result: Raw result from predict_topics.

        Returns:
            Enriched result dict with topic_label and confidence.
        """
        topic_id = raw_result["topic"]
        keywords = raw_result.get("keywords", [])

        # Generate a human-readable label from keywords
        if topic_id == -1:
            topic_label = "unclassified"
        elif keywords:
            # Use top keywords to form a label
            topic_label = " / ".join(keywords[:3])
        else:
            topic_label = f"topic_{topic_id}"

        # Estimate confidence based on keyword presence
        confidence = len(keywords) / self.top_n_words if keywords else 0.0

        return {
            "text": raw_result.get("text", ""),
            "topic": topic_id,
            "topic_label": topic_label,
            "keywords": keywords,
            "confidence": round(confidence, 2),
        }

    def _ensure_model(self):
        """Ensure the BERTopic model is loaded."""
        if self._model is None:
            self._model = get_topic_model()
        return self._model


# Singleton instance for convenience
_default_modeler = TopicModeler()


def analyze_topics(texts: list[str], min_topic_size: Optional[int] = None) -> list[dict]:
    """Analyze topics for texts using the default modeler.

    Args:
        texts: List of texts to analyze.
        min_topic_size: Override minimum topic size.

    Returns:
        List of topic analysis result dicts.
    """
    if min_topic_size is not None:
        modeler = TopicModeler(min_topic_size=min_topic_size)
        return modeler.analyze(texts)
    return _default_modeler.analyze(texts)


def get_topic_overview(texts: list[str]) -> dict:
    """Get a topic summary for texts using the default modeler.

    Args:
        texts: List of texts to analyze.

    Returns:
        Topic summary dict.
    """
    return _default_modeler.get_topic_summary(texts)


def detect_topic_trends(
    current_texts: list[str],
    previous_texts: Optional[list[str]] = None,
) -> dict:
    """Detect trending topics using the default modeler.

    Args:
        current_texts: Current batch of texts.
        previous_texts: Previous batch for comparison.

    Returns:
        Trend detection result dict.
    """
    return _default_modeler.detect_trends(current_texts, previous_texts)