"""Tests for data_collection module.

Covers RedditCollector, TwitterCollector, Deduplicator, and Scheduler
using mocks for external API calls (PRAW, Tweepy) and database operations.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from data_collection.deduplication import Deduplicator


# ─── Deduplicator Tests ───────────────────────────────────────────────────


class TestDeduplicator:
    """Tests for the Deduplicator class."""

    def test_get_text_hash_consistency(self):
        """Same text (case-insensitive) should produce the same hash."""
        dedup = Deduplicator()
        hash1 = dedup.get_text_hash("Hello World")
        hash2 = dedup.get_text_hash("hello world")
        assert hash1 == hash2

    def test_get_text_hash_different_texts(self):
        """Different texts should produce different hashes."""
        dedup = Deduplicator()
        hash1 = dedup.get_text_hash("First text")
        hash2 = dedup.get_text_hash("Second text")
        assert hash1 != hash2

    def test_get_text_hash_whitespace_normalization(self):
        """Leading/trailing whitespace should not affect the hash."""
        dedup = Deduplicator()
        hash1 = dedup.get_text_hash("  Hello World  ")
        hash2 = dedup.get_text_hash("Hello World")
        assert hash1 == hash2

    def test_is_similar_identical_texts(self):
        """Identical texts should be considered similar."""
        dedup = Deduplicator(similarity_threshold=0.85)
        assert dedup.is_similar("The quick brown fox", "the quick brown fox") is True

    def test_is_similar_different_texts(self):
        """Very different texts should not be considered similar."""
        dedup = Deduplicator(similarity_threshold=0.85)
        assert dedup.is_similar("Politics news today", "Machine learning advances") is False

    def test_is_similar_slightly_modified_texts(self):
        """Texts with minor differences may still be similar depending on threshold."""
        dedup = Deduplicator(similarity_threshold=0.5)
        result = dedup.is_similar(
            "The president announced new policies today",
            "The president announced new policies yesterday",
        )
        # With a low threshold, slightly modified texts should match
        assert result is True

    def test_filter_duplicates_removes_exact_duplicates(self):
        """Exact duplicate posts should be filtered out."""
        dedup = Deduplicator()
        posts = [
            {"id": "1", "title": "Breaking News", "text": "Something happened"},
            {"id": "2", "title": "Breaking News", "text": "Something happened"},
            {"id": "3", "title": "Different News", "text": "Another thing"},
        ]
        unique = dedup.filter_duplicates(posts)
        assert len(unique) == 2
        assert unique[0]["id"] == "1"
        assert unique[1]["id"] == "3"

    def test_filter_duplicates_all_unique(self):
        """All unique posts should pass through."""
        dedup = Deduplicator()
        posts = [
            {"id": "1", "title": "News A", "text": "Event A happened"},
            {"id": "2", "title": "News B", "text": "Event B happened"},
            {"id": "3", "title": "News C", "text": "Event C happened"},
        ]
        unique = dedup.filter_duplicates(posts)
        assert len(unique) == 3

    def test_filter_duplicates_empty_list(self):
        """Empty input should return empty output."""
        dedup = Deduplicator()
        unique = dedup.filter_duplicates([])
        assert unique == []

    def test_filter_duplicates_short_texts_skip_similarity(self):
        """Short texts (<50 chars) should only use hash dedup, not similarity."""
        dedup = Deduplicator(similarity_threshold=0.5)
        posts = [
            {"id": "1", "title": "Hi", "text": "Short"},  # combined < 50 chars
            {"id": "2", "title": "Hey", "text": "Short"},  # different title, same text
        ]
        unique = dedup.filter_duplicates(posts)
        # Both should be kept since combined text hashes differ
        assert len(unique) == 2

    def test_clear_cache(self):
        """Clearing cache should reset seen hashes and texts."""
        dedup = Deduplicator()
        dedup.filter_duplicates([{"id": "1", "title": "Test", "text": "test"}])
        assert len(dedup.seen_hashes) > 0
        assert len(dedup.seen_texts) > 0

        dedup.clear_cache()
        assert len(dedup.seen_hashes) == 0
        assert len(dedup.seen_texts) == 0

    def test_clear_cache_then_refilter(self):
        """After clearing cache, previously seen posts should be accepted again."""
        dedup = Deduplicator()
        posts = [{"id": "1", "title": "Test", "text": "test content here"}]
        unique1 = dedup.filter_duplicates(posts)
        assert len(unique1) == 1

        # Same posts filtered again — should be deduplicated
        unique2 = dedup.filter_duplicates(posts)
        assert len(unique2) == 0

        # After clearing cache, they should be accepted again
        dedup.clear_cache()
        unique3 = dedup.filter_duplicates(posts)
        assert len(unique3) == 1

    def test_custom_similarity_threshold(self):
        """Custom similarity threshold should be respected."""
        strict = Deduplicator(similarity_threshold=0.99)
        loose = Deduplicator(similarity_threshold=0.3)

        text1 = "The president announced new policies today"
        text2 = "The president announced new policies yesterday"

        # Strict threshold — minor change should NOT be similar
        assert strict.is_similar(text1, text2) is False
        # Loose threshold — minor change SHOULD be similar
        assert loose.is_similar(text1, text2) is True


# ─── RedditCollector Tests (with mocks) ────────────────────────────────────


class TestRedditCollector:
    """Tests for RedditCollector using mocked PRAW."""

    @patch("data_collection.reddit_collector.praw.Reddit")
    @patch("data_collection.reddit_collector.get_settings")
    def test_init_creates_reddit_instance(self, mock_settings, mock_reddit_cls):
        """RedditCollector should initialize a PRAW Reddit instance."""
        mock_settings.return_value = MagicMock(
            reddit_client_id="test_id",
            reddit_client_secret="test_secret",
            reddit_user_agent="test_agent",
        )
        from data_collection.reddit_collector import RedditCollector

        collector = RedditCollector()
        mock_reddit_cls.assert_called_once()

    @patch("data_collection.reddit_collector.praw.Reddit")
    @patch("data_collection.reddit_collector.get_settings")
    def test_fetch_hot_posts_returns_posts(self, mock_settings, mock_reddit_cls):
        """fetch_hot_posts should return a list of post dicts."""
        mock_settings.return_value = MagicMock(
            reddit_client_id="test_id",
            reddit_client_secret="test_secret",
            reddit_user_agent="test_agent",
        )
        from data_collection.reddit_collector import RedditCollector

        # Create mock submission
        mock_submission = MagicMock()
        mock_submission.id = "abc123"
        mock_submission.title = "Test Reddit Post"
        mock_submission.selftext = "This is the post body"
        mock_submission.author = "test_user"
        mock_submission.score = 100
        mock_submission.num_comments = 25
        mock_submission.permalink = "/r/test/comments/abc123"
        mock_submission.created_utc = 1700000000.0

        # Configure mock subreddit
        mock_reddit_instance = mock_reddit_cls.return_value
        mock_subreddit = MagicMock()
        mock_subreddit.hot.return_value = [mock_submission]
        mock_reddit_instance.subreddit.return_value = mock_subreddit

        collector = RedditCollector()
        posts = collector.fetch_hot_posts("test", limit=1)

        assert len(posts) == 1
        assert posts[0]["id"] == "reddit_abc123"
        assert posts[0]["source"] == "reddit"
        assert posts[0]["platform"] == "reddit"
        assert posts[0]["title"] == "Test Reddit Post"
        assert posts[0]["text"] == "This is the post body"
        assert posts[0]["upvotes"] == 100
        assert posts[0]["comments_count"] == 25

    @patch("data_collection.reddit_collector.praw.Reddit")
    @patch("data_collection.reddit_collector.get_settings")
    def test_fetch_hot_posts_handles_errors(self, mock_settings, mock_reddit_cls):
        """fetch_hot_posts should handle API errors gracefully."""
        mock_settings.return_value = MagicMock(
            reddit_client_id="test_id",
            reddit_client_secret="test_secret",
            reddit_user_agent="test_agent",
        )
        from data_collection.reddit_collector import RedditCollector

        mock_reddit_instance = mock_reddit_cls.return_value
        mock_subreddit = MagicMock()
        mock_subreddit.hot.side_effect = Exception("API error")
        mock_reddit_instance.subreddit.return_value = mock_subreddit

        collector = RedditCollector()
        posts = collector.fetch_hot_posts("test")
        assert posts == []

    @patch("data_collection.reddit_collector.praw.Reddit")
    @patch("data_collection.reddit_collector.get_settings")
    def test_extract_hashtags(self, mock_settings, mock_reddit_cls):
        """_extract_hashtags should find hashtags in text."""
        mock_settings.return_value = MagicMock(
            reddit_client_id="test_id",
            reddit_client_secret="test_secret",
            reddit_user_agent="test_agent",
        )
        from data_collection.reddit_collector import RedditCollector

        collector = RedditCollector()
        hashtags = collector._extract_hashtags("Check out #AI and #MachineLearning!")
        assert "#AI" in hashtags
        assert "#MachineLearning" in hashtags

    @patch("data_collection.reddit_collector.praw.Reddit")
    @patch("data_collection.reddit_collector.get_settings")
    def test_get_post_hash(self, mock_settings, mock_reddit_cls):
        """get_post_hash should return a consistent MD5 hash."""
        mock_settings.return_value = MagicMock(
            reddit_client_id="test_id",
            reddit_client_secret="test_secret",
            reddit_user_agent="test_agent",
        )
        from data_collection.reddit_collector import RedditCollector

        collector = RedditCollector()
        hash1 = collector.get_post_hash("Test text")
        hash2 = collector.get_post_hash("test text")  # case-insensitive
        assert hash1 == hash2
        assert len(hash1) == 32  # MD5 hash length


# ─── TwitterCollector Tests (with mocks) ───────────────────────────────────


class TestTwitterCollector:
    """Tests for TwitterCollector using mocked Tweepy."""

    @patch("data_collection.twitter_collector.tweepy.Client")
    @patch("data_collection.twitter_collector.get_settings")
    def test_init_creates_client(self, mock_settings, mock_client_cls):
        """TwitterCollector should initialize a Tweepy Client."""
        mock_settings.return_value = MagicMock(
            x_bearer_token="test_bearer",
            x_api_key="test_key",
            x_api_secret="test_secret",
            x_access_token="test_token",
            x_access_token_secret="test_token_secret",
        )
        from data_collection.twitter_collector import TwitterCollector

        collector = TwitterCollector()
        mock_client_cls.assert_called_once()

    @patch("data_collection.twitter_collector.tweepy.Client")
    @patch("data_collection.twitter_collector.get_settings")
    def test_fetch_trending_tweets_returns_tweets(self, mock_settings, mock_client_cls):
        """fetch_trending_tweets should return a list of tweet dicts."""
        mock_settings.return_value = MagicMock(
            x_bearer_token="test_bearer",
            x_api_key="test_key",
            x_api_secret="test_secret",
            x_access_token="test_token",
            x_access_token_secret="test_token_secret",
        )
        from data_collection.twitter_collector import TwitterCollector

        # Create mock tweet
        mock_tweet = MagicMock()
        mock_tweet.id = "12345"
        mock_tweet.text = "This is a test tweet #AI"
        mock_tweet.author_id = "user_123"
        mock_tweet.created_at = MagicMock()
        mock_tweet.public_metrics = {
            "reply_count": 5,
            "retweet_count": 10,
            "like_count": 50,
        }
        mock_tweet.entities = {"hashtags": [{"tag": "AI"}]}

        # Configure mock response
        mock_response = MagicMock()
        mock_response.data = [mock_tweet]
        mock_client_instance = mock_client_cls.return_value
        mock_client_instance.search_recent_tweets.return_value = mock_response

        collector = TwitterCollector()
        tweets = collector.fetch_trending_tweets("AI news", max_results=1)

        assert len(tweets) == 1
        assert tweets[0]["id"] == "twitter_12345"
        assert tweets[0]["source"] == "twitter"
        assert tweets[0]["platform"] == "twitter"
        assert tweets[0]["text"] == "This is a test tweet #AI"
        assert tweets[0]["retweets"] == 10
        assert tweets[0]["likes"] == 50
        assert "AI" in tweets[0]["hashtags"]

    @patch("data_collection.twitter_collector.tweepy.Client")
    @patch("data_collection.twitter_collector.get_settings")
    def test_fetch_trending_tweets_handles_errors(self, mock_settings, mock_client_cls):
        """fetch_trending_tweets should handle API errors gracefully."""
        mock_settings.return_value = MagicMock(
            x_bearer_token="test_bearer",
            x_api_key="test_key",
            x_api_secret="test_secret",
            x_access_token="test_token",
            x_access_token_secret="test_token_secret",
        )
        from data_collection.twitter_collector import TwitterCollector

        mock_client_instance = mock_client_cls.return_value
        mock_client_instance.search_recent_tweets.side_effect = Exception("API error")

        collector = TwitterCollector()
        tweets = collector.fetch_trending_tweets("test")
        assert tweets == []

    @patch("data_collection.twitter_collector.tweepy.Client")
    @patch("data_collection.twitter_collector.get_settings")
    def test_fetch_trending_tweets_no_data(self, mock_settings, mock_client_cls):
        """fetch_trending_tweets should return empty list when no data."""
        mock_settings.return_value = MagicMock(
            x_bearer_token="test_bearer",
            x_api_key="test_key",
            x_api_secret="test_secret",
            x_access_token="test_token",
            x_access_token_secret="test_token_secret",
        )
        from data_collection.twitter_collector import TwitterCollector

        mock_response = MagicMock()
        mock_response.data = None
        mock_client_instance = mock_client_cls.return_value
        mock_client_instance.search_recent_tweets.return_value = mock_response

        collector = TwitterCollector()
        tweets = collector.fetch_trending_tweets("test")
        assert tweets == []

    @patch("data_collection.twitter_collector.tweepy.Client")
    @patch("data_collection.twitter_collector.get_settings")
    def test_get_post_hash(self, mock_settings, mock_client_cls):
        """get_post_hash should return a consistent MD5 hash."""
        mock_settings.return_value = MagicMock(
            x_bearer_token="test_bearer",
            x_api_key="test_key",
            x_api_secret="test_secret",
            x_access_token="test_token",
            x_access_token_secret="test_token_secret",
        )
        from data_collection.twitter_collector import TwitterCollector

        collector = TwitterCollector()
        hash1 = collector.get_post_hash("Test text")
        hash2 = collector.get_post_hash("test text")
        assert hash1 == hash2
        assert len(hash1) == 32