"""
Integration-style tests for the main orchestrator.
"""

import json
from unittest.mock import Mock, MagicMock, patch

from src.main import RedditSentimentBot
from src.database import DatabaseManager, AnalysisRun, Post


class FakeRedditClient:
    def __init__(self, posts_by_sub):
        self.posts_by_sub = posts_by_sub

    def fetch_subreddit_posts(self, name: str):
        return self.posts_by_sub.get(name, [])


def test_orchestrator_end_to_end_with_cost_tracking():
    # Prepare in-memory DB and components
    db = DatabaseManager(db_path=':memory:')

    # Fake Reddit posts
    posts = [
        {
            'id': 'p1', 'subreddit': 'ClaudeAI', 'title': 'Love Claude', 'selftext': 'great',
            'author': 'u1', 'url': 'http://x', 'created_utc': 123456, 'score': 10, 'num_comments': 0
        },
        {
            'id': 'p2', 'subreddit': 'ClaudeAI', 'title': 'Issues with API', 'selftext': 'rate limits',
            'author': 'u2', 'url': 'http://y', 'created_utc': 123457, 'score': 3, 'num_comments': 0
        },
    ]
    reddit = FakeRedditClient({'ClaudeAI': posts})

    # Build analyzer with mocked Anthropic client
    with patch('src.sentiment_analyzer.Anthropic'):
        from src.sentiment_analyzer import SentimentAnalyzer, CostManager
        cm = CostManager(daily_limit=5.0, db_manager=db)
        analyzer = SentimentAnalyzer(api_key='test', cost_manager=cm)
        # Mock API response
        mock_resp = Mock()
        mock_resp.content = [Mock(text=json.dumps([
            {'index': 0, 'sentiment_score': 0.8, 'sentiment_explanation': 'Positive', 'keywords': []},
            {'index': 1, 'sentiment_score': -0.4, 'sentiment_explanation': 'Negative', 'keywords': []},
        ]))]
        mock_resp.usage = Mock(input_tokens=1000, output_tokens=500)
        analyzer.client.messages.create = Mock(return_value=mock_resp)

    bot = RedditSentimentBot(
        reddit_client=reddit,
        subreddits=['ClaudeAI'],
        db_manager=db,
        analyzer=analyzer,
        cost_manager=cm,
    )

    run = bot.run_once()

    # Verify run completed and costs recorded
    with db.transaction() as session:
        db_run = session.query(AnalysisRun).filter_by(id=run.id).first()
        assert db_run is not None
        assert db_run.status == 'completed'
        # api_calls and cost estimate recorded via CostManager.record_usage
        assert (db_run.api_calls or 0) >= 1
        assert (db_run.api_cost_estimate or 0.0) > 0.0

        # Posts updated with sentiment
        p1 = session.query(Post).filter_by(id='p1').first()
        p2 = session.query(Post).filter_by(id='p2').first()
        assert p1.sentiment_score == 0.8
        assert p2.sentiment_score == -0.4

    db.close()

