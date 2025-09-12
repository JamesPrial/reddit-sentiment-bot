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

    def run_once(self, output_file: Optional[str] = None) -> AnalysisRun:
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

        # Analyze posts (process all unanalyzed in batches)
        posts_output: List[Dict[str, Any]] = []
        while True:
            to_analyze = self.db.get_posts_for_analysis(batch_size=500, run_id=run.id)
            if not to_analyze:
                break
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
            if updates:
                self.db.update_sentiment_scores(updates, 'post')
                # Update keyword associations from analysis results
                self.db.update_keyword_associations_from_results(results, 'post')
                posts_output.extend(self._trim_output_fields(r, 'post') for r in results)
        
        # Analyze comments (process all unanalyzed in batches)
        comments_output: List[Dict[str, Any]] = []
        while True:
            comments_to_analyze = self.db.get_comments_for_analysis(batch_size=500, run_id=run.id)
            if not comments_to_analyze:
                break
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
            if comment_updates:
                self.db.update_sentiment_scores(comment_updates, 'comment')
                # Update keyword associations from analysis results
                self.db.update_keyword_associations_from_results(comment_results, 'comment')
                comments_output.extend(self._trim_output_fields(r, 'comment') for r in comment_results)

        # Generate daily summaries
        _summaries = self.db.generate_daily_summary(run.id)

        # Optionally write analysis output to file
        try:
            self._maybe_write_output_file(
                run_id=run.id,
                posts=posts_output,
                comments=comments_output,
                output_file=output_file,
                mode_label='run',
                summaries=self._serialize_summaries(_summaries),
            )
        except Exception as e:
            logger.warning(f"Failed writing analysis output file: {e}")

        # Mark run as completed with stats (do not override api cost fields)
        stats = {
            'total_posts': total_inserted,
            'total_comments': total_comments,
        }
        self.db.complete_analysis_run(run.id, stats)

        return run

    def analyze_existing_run(self, run_id: int, *, post_batch: int = 500, comment_batch: int = 500, output_file: Optional[str] = None) -> Dict[str, int]:
        """Analyze only unanalyzed posts/comments for an existing run.

        Args:
            run_id: Target analysis run ID
            post_batch: Batch size for posts
            comment_batch: Batch size for comments

        Returns:
            Stats with counts analyzed
        """
        # Attach run context to analyzer so costs are persisted to the run
        try:
            # type: ignore[attr-defined]
            self.analyzer.run_id = run_id
        except Exception:
            pass

        analyzed_posts = 0
        posts_output: List[Dict[str, Any]] = []
        while True:
            to_analyze = self.db.get_posts_for_analysis(batch_size=post_batch, run_id=run_id)
            if not to_analyze:
                break
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
            if updates:
                self.db.update_sentiment_scores(updates, 'post')
                self.db.update_keyword_associations_from_results(results, 'post')
                analyzed_posts += len(updates)
                posts_output.extend(self._trim_output_fields(r, 'post') for r in results)

        analyzed_comments = 0
        comments_output: List[Dict[str, Any]] = []
        while True:
            comments_to_analyze = self.db.get_comments_for_analysis(batch_size=comment_batch, run_id=run_id)
            if not comments_to_analyze:
                break
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
            if comment_updates:
                self.db.update_sentiment_scores(comment_updates, 'comment')
                self.db.update_keyword_associations_from_results(comment_results, 'comment')
                analyzed_comments += len(comment_updates)
                comments_output.extend(self._trim_output_fields(r, 'comment') for r in comment_results)

        # Regenerate daily summaries for this run (upsert by date+subreddit)
        summaries = self.db.generate_daily_summary(run_id)

        # Optionally write analysis output to file
        try:
            self._maybe_write_output_file(
                run_id=run_id,
                posts=posts_output,
                comments=comments_output,
                output_file=output_file or f"data/analysis_run_{run_id}_reanalyzed.json",
                mode_label='reanalyze',
                summaries=self._serialize_summaries(summaries),
            )
        except Exception as e:
            logger.warning(f"Failed writing analysis output file: {e}")

        logger.info(
            "Analyze-only mode completed: %s posts, %s comments analyzed; %s summaries updated",
            analyzed_posts,
            analyzed_comments,
            len(summaries),
        )
        return {"posts": analyzed_posts, "comments": analyzed_comments, "summaries": len(summaries)}

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

    def _trim_output_fields(self, item: Dict[str, Any], item_type: str) -> Dict[str, Any]:
        """Keep a concise subset of fields for file output."""
        allowed = {'id', 'sentiment_score', 'sentiment_explanation', 'keywords', 'score'}
        out = {k: v for k, v in item.items() if k in allowed}
        out['type'] = item_type
        if 'keywords' in out and not isinstance(out['keywords'], list):
            out['keywords'] = []
        return out

    def _maybe_write_output_file(
        self,
        *,
        run_id: int,
        posts: List[Dict[str, Any]],
        comments: List[Dict[str, Any]],
        output_file: Optional[str],
        mode_label: str,
        summaries: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        import json as _json
        import os as _os
        from datetime import datetime as _dt

        # Output only summary data (exclude item-level details and keywords)
        data = {
            'run_id': run_id,
            'mode': mode_label,
            'generated_at': _dt.utcnow().isoformat() + 'Z',
            'summaries': summaries or [],
        }
        path = output_file or f"data/analysis_run_{run_id}.json"
        _os.makedirs(_os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            _json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info("Wrote analysis output to %s", path)

    def _serialize_summaries(self, summaries: List[Any]) -> List[Dict[str, Any]]:
        """Convert DailySummary ORM objects into JSON-serializable dictionaries."""
        import json as _json
        out: List[Dict[str, Any]] = []
        for s in summaries or []:
            try:
                d: Dict[str, Any] = {
                    'id': getattr(s, 'id', None),
                    'analysis_run_id': getattr(s, 'analysis_run_id', None),
                    'subreddit_id': getattr(s, 'subreddit_id', None),
                    'date': getattr(s, 'date', None).isoformat() if getattr(s, 'date', None) else None,
                    'total_posts': getattr(s, 'total_posts', None),
                    'total_comments': getattr(s, 'total_comments', None),
                    'avg_post_sentiment': getattr(s, 'avg_post_sentiment', None),
                    'avg_comment_sentiment': getattr(s, 'avg_comment_sentiment', None),
                    'positive_posts': getattr(s, 'positive_posts', None),
                    'negative_posts': getattr(s, 'negative_posts', None),
                    'neutral_posts': getattr(s, 'neutral_posts', None),
                    'top_positive_post_id': getattr(s, 'top_positive_post_id', None),
                    'top_negative_post_id': getattr(s, 'top_negative_post_id', None),
                    'most_discussed_post_id': getattr(s, 'most_discussed_post_id', None),
                }
                out.append(d)
            except Exception:
                # Best-effort serialization fallback
                out.append({'error': 'failed to serialize summary'})
        return out


if __name__ == "__main__":
    import sys
    import os
    import argparse
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from src.secrets_manager import load_secrets
    from src.reddit_client import RedditClient
    from src.database import DatabaseManager
    from src.sentiment_analyzer import SentimentAnalyzer, CostManager
    from src.config import get_config
    
    # CLI arguments
    parser = argparse.ArgumentParser(description="Reddit Sentiment Analysis Bot")
    parser.add_argument(
        "--analyze-last-run",
        action="store_true",
        help="Analyze only the most recent run (no fetching)"
    )
    parser.add_argument(
        "--analyze-run-id",
        type=int,
        help="Analyze only the specified analysis run ID (no fetching)"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        help="Path to write JSON analysis output (default: data/analysis_run_<id>.json)"
    )
    parser.add_argument(
        "--reset-db",
        action="store_true",
        help="Drop and recreate the database schema (clears all data) and exit"
    )
    args = parser.parse_args()

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
        
        # If requested, reset the database before doing anything else
        if args.reset_db:
            logger.warning("--reset-db specified: clearing all data and exiting")
            db_manager.reset_database()
            logger.info("Database reset complete")
            sys.exit(0)

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
        
        if args.analyze_run_id is not None:
            # Analyze only the specified run ID
            target = db_manager.get_run_by_id(args.analyze_run_id)
            if not target:
                logger.error("Analysis run ID %s not found.", args.analyze_run_id)
                sys.exit(1)
            logger.info("Analyzing existing run ID %s", target.id)
            stats = bot.analyze_existing_run(target.id, output_file=args.output_file)
            logger.info(
                "Analyze-only finished: %s posts, %s comments, %s summaries",
                stats.get('posts', 0), stats.get('comments', 0), stats.get('summaries', 0)
            )
        elif args.analyze_last_run:
            # Analyze only the most recent run
            latest = db_manager.get_latest_run()
            if not latest:
                logger.error("No previous analysis runs found to analyze.")
                sys.exit(1)
            logger.info("Analyzing only last run (ID %s)", latest.id)
            stats = bot.analyze_existing_run(latest.id, output_file=args.output_file)
            logger.info(
                "Analyze-only finished: %s posts, %s comments, %s summaries",
                stats.get('posts', 0), stats.get('comments', 0), stats.get('summaries', 0)
            )
        else:
            # Full fetch + analyze run
            logger.info(f"Starting analysis for subreddits: {config_manager.get_subreddits()}")
            run = bot.run_once(output_file=args.output_file)
            
            logger.info(f"Analysis completed! Run ID: {run.id}")
            logger.info(f"Posts analyzed: {run.total_posts}")
            logger.info(f"Comments analyzed: {run.total_comments}")
            logger.info(f"Estimated cost: ${run.api_cost_estimate:.4f}" if run.api_cost_estimate else "Cost tracking not available")
        
    except Exception as e:
        logger.error(f"Bot execution failed: {e}")
        sys.exit(1)
