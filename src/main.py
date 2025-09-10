"""
Main orchestration for the Reddit Sentiment Analysis Bot.

Coordinates fetching posts, analyzing sentiment via Claude,
and persisting results with summaries and cost tracking.
"""

import logging
from typing import List, Optional, Dict, Any

from src.database import DatabaseManager, AnalysisRun, Post
from src.sentiment_analyzer import SentimentAnalyzer, CostManager

logger = logging.getLogger(__name__)


class RedditSentimentBot:
    """High-level orchestrator for a single analysis run."""

    def __init__(
        self,
        *,
        reddit_client: Any,
        subreddits: List[str],
        db_manager: Optional[DatabaseManager] = None,
        analyzer: Optional[SentimentAnalyzer] = None,
        cost_manager: Optional[CostManager] = None,
        db_path: Optional[str] = None,
    ):
        self.reddit_client = reddit_client
        self.subreddits = subreddits
        self.db = db_manager or DatabaseManager(db_path=db_path)
        self.cost_manager = cost_manager or CostManager(daily_limit=5.0, db_manager=self.db)
        self.analyzer = analyzer or SentimentAnalyzer(api_key="", cost_manager=self.cost_manager)

    def run_once(self) -> AnalysisRun:
        """Execute a single end-to-end analysis run."""
        # Create analysis run
        run = self.db.create_analysis_run()
        logger.info("Created analysis run %s", run.id)

        # Attach run context to analyzer so costs are persisted to the run
        try:
            # type: ignore[attr-defined]
            self.analyzer.run_id = run.id  # set on analyzer for cost tracking
        except Exception:
            pass

        total_inserted = 0
        total_comments = 0

        # Fetch and insert posts per subreddit
        for name in self.subreddits:
            try:
                posts = self.reddit_client.fetch_subreddit_posts(name)
            except Exception as e:
                logger.error("Failed fetching posts for %s: %s", name, e)
                posts = []
            if not posts:
                continue
            inserted = self.db.bulk_insert_posts(posts, run.id)
            total_inserted += inserted

        # Analyze posts
        to_analyze = self.db.get_posts_for_analysis(batch_size=500, run_id=run.id)
        if to_analyze:
            items = self._build_items_from_posts(to_analyze)
            results = self.analyzer.process_items_in_batches(items, item_type='post')
            updates = [
                {
                    'id': it['id'],
                    'sentiment_score': it.get('sentiment_score'),
                    'sentiment_explanation': it.get('sentiment_explanation'),
                }
                for it in results
            ]
            self.db.update_sentiment_scores(updates, 'post')
            # Update keyword associations from analysis results
            self.db.update_keyword_associations_from_results(results, 'post')

        # Generate daily summaries
        self.db.generate_daily_summary(run.id)

        # Mark run as completed with stats (do not override api cost fields)
        stats = {
            'total_posts': total_inserted,
            'total_comments': total_comments,
        }
        self.db.complete_analysis_run(run.id, stats)

        return run

    def _build_items_from_posts(self, posts: List[Post]) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for p in posts:
            items.append({
                'id': p.id,
                'title': p.title or "",
                'selftext': p.selftext or "",
                'score': p.score or 0,
                # Avoid lazy-loading relationships on detached instances
                'subreddit': '',
            })
        return items
