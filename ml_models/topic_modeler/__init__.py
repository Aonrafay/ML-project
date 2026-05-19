"""Topic Modeler — Higher-level wrapper for topic modeling.

Provides topic labeling, trend detection, and batch analysis
beyond the basic BERTopic predict module.
"""

from ml_models.topic_modeler.model import TopicModeler

__all__ = ["TopicModeler"]