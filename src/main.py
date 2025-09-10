"""
Main orchestration for the Reddit Sentiment Analysis Bot.

Coordinates fetching posts, analyzing sentiment via Claude,
and persisting results with summaries and cost tracking.
"""

import logging
from typing import List, Optional, Dict, Any

from src.database import DatabaseManager, AnalysisRun, Post, Comment
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
            
            # Fetch comments for each post
            for post in posts:
                try:
                    comments = self.reddit_client.fetch_post_comments(post['id'])
                    if comments:
                        # Add run_id to each comment before insertion
                        for comment in comments:
                            comment['analysis_run_id'] = run.id
                        comments_inserted = self.db.bulk_insert_comments(comments)
                        total_comments += comments_inserted
                except Exception as e:
                    logger.error("Failed fetching comments for post %s: %s", post.get('id'), e)

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
        
        # Analyze comments
        comments_to_analyze = self.db.get_comments_for_analysis(batch_size=500, run_id=run.id)
        if comments_to_analyze:
            comment_items = self._build_items_from_comments(comments_to_analyze)
            comment_results = self.analyzer.process_items_in_batches(comment_items, item_type='comment')
            comment_updates = [
                {
                    'id': it['id'],
                    'sentiment_score': it.get('sentiment_score'),
                    'sentiment_explanation': it.get('sentiment_explanation'),
                }
                for it in comment_results
            ]
            self.db.update_sentiment_scores(comment_updates, 'comment')
            # Update keyword associations from analysis results
            self.db.update_keyword_associations_from_results(comment_results, 'comment')

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
    
    def _build_items_from_comments(self, comments: List) -> List[Dict[str, Any]]:
        """Build items from comment objects for sentiment analysis."""
        items: List[Dict[str, Any]] = []
        for c in comments:
            items.append({
                'id': c.id,
                'body': c.body or "",
                'score': c.score or 0,
            })
        return items


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from src.secrets_manager import load_secrets
    from src.reddit_client import RedditClient
    from src.database import DatabaseManager
    from src.sentiment_analyzer import SentimentAnalyzer, CostManager
    from src.config import get_config
    
    # Load configuration
    config_manager = get_config()
    config = config_manager.config
    
    # Configure logging
    log_config = config_manager.get_logging_config()
    logging.basicConfig(
        level=getattr(logging, log_config.get('level', 'INFO')),
        format=log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_config.get('file', 'bot.log'))
        ]
    )
    
    try:
        # Load secrets
        logger.info("Loading secrets from keychain/environment...")
        secrets = load_secrets()
        
        # Initialize components
        logger.info("Initializing components...")
        
        # Database
        db_config = config_manager.get_database_config()
        db_path = secrets.get('DATABASE_PATH', db_config.get('path', './data/sentiment.db'))
        db_manager = DatabaseManager(db_path=db_path)
        
        # Reddit client
        reddit_config = config_manager.get_reddit_config()
        reddit_client = RedditClient(
            requests_per_minute=reddit_config.get('requests_per_minute', 60)
        )
        
        # Cost manager and sentiment analyzer
        cost_config = config_manager.get_cost_config()
        cost_manager = CostManager(
            daily_limit=cost_config.get('daily_limit', 5.0),
            db_manager=db_manager
        )
        
        claude_config = config_manager.get_claude_config()
        analyzer = SentimentAnalyzer(
            api_key=secrets['ANTHROPIC_API_KEY'],
            model=claude_config.get('model', 'claude-3-5-sonnet-20241022'),
            cost_manager=cost_manager
        )
        
        # Create bot instance
        bot = RedditSentimentBot(
            reddit_client=reddit_client,
            subreddits=config_manager.get_subreddits(),
            db_manager=db_manager,
            analyzer=analyzer,
            cost_manager=cost_manager
        )
        
        # Run analysis
        logger.info(f"Starting analysis for subreddits: {config_manager.get_subreddits()}")
        run = bot.run_once()
        
        logger.info(f"Analysis completed! Run ID: {run.id}")
        logger.info(f"Posts analyzed: {run.total_posts}")
        logger.info(f"Comments analyzed: {run.total_comments}")
        logger.info(f"Estimated cost: ${run.api_cost_estimate:.4f}" if run.api_cost_estimate else "Cost tracking not available")
        
    except Exception as e:
        logger.error(f"Bot execution failed: {e}")
        sys.exit(1)
