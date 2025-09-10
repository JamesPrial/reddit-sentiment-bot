"""
Comprehensive tests for the database module.

Tests include:
- Model relationships
- Transaction handling
- Bulk operations
- Query performance
- Migration execution
- Error handling
"""

import os
import tempfile
import pytest
from datetime import datetime, timedelta
from typing import List, Dict
from unittest.mock import patch, MagicMock

from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session

from src.database import (
    DatabaseManager, Base,
    Subreddit, AnalysisRun, Post, Comment, Keyword, DailySummary
)


class TestDatabaseModels:
    """Test SQLAlchemy model definitions and relationships."""
    
    @pytest.fixture
    def db_manager(self):
        """Create a temporary in-memory database for testing."""
        # Use in-memory database for tests
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_path = tmp.name
        
        manager = DatabaseManager(db_path=':memory:')
        yield manager
        manager.close()
    
    def test_subreddit_model(self, db_manager):
        """Test Subreddit model creation and attributes."""
        with db_manager.transaction() as session:
            subreddit = Subreddit(name='ClaudeAI')
            session.add(subreddit)
            session.flush()
            
            assert subreddit.id is not None
            assert subreddit.name == 'ClaudeAI'
            assert subreddit.created_at is not None
    
    def test_analysis_run_model(self, db_manager):
        """Test AnalysisRun model creation and defaults."""
        with db_manager.transaction() as session:
            run = AnalysisRun(
                run_date=datetime.utcnow().date(),
                started_at=datetime.utcnow()
            )
            session.add(run)
            session.flush()
            
            assert run.id is not None
            assert run.status == 'running'
            assert run.total_posts == 0
            assert run.api_cost_estimate == 0
    
    def test_post_model_with_relationships(self, db_manager):
        """Test Post model with foreign key relationships."""
        with db_manager.transaction() as session:
            # Create dependencies
            subreddit = Subreddit(name='ClaudeAI')
            run = AnalysisRun(
                run_date=datetime.utcnow().date(),
                started_at=datetime.utcnow()
            )
            session.add_all([subreddit, run])
            session.flush()
            
            # Create post
            post = Post(
                id='test123',
                subreddit_id=subreddit.id,
                analysis_run_id=run.id,
                title='Test Post',
                created_utc=int(datetime.utcnow().timestamp()),
                score=42
            )
            session.add(post)
            session.flush()
            
            # Test relationships
            assert post.subreddit.name == 'ClaudeAI'
            assert post.analysis_run.id == run.id
    
    def test_comment_model_with_post(self, db_manager):
        """Test Comment model with post relationship."""
        with db_manager.transaction() as session:
            # Create post first
            subreddit = Subreddit(name='ClaudeAI')
            post = Post(
                id='post123',
                subreddit_id=1,
                title='Test Post',
                created_utc=int(datetime.utcnow().timestamp())
            )
            session.add_all([subreddit, post])
            session.flush()
            
            # Create comment
            comment = Comment(
                id='comment456',
                post_id='post123',
                body='Test comment',
                created_utc=int(datetime.utcnow().timestamp())
            )
            session.add(comment)
            session.flush()
            
            # Test relationship
            assert comment.post.id == 'post123'
            assert len(post.comments) == 1
    
    def test_keyword_many_to_many_relationships(self, db_manager):
        """Test many-to-many relationships with keywords."""
        with db_manager.transaction() as session:
            # Create entities
            keyword = Keyword(term='Claude', category='product')
            subreddit = Subreddit(name='ClaudeAI')
            post = Post(
                id='post789',
                subreddit_id=1,
                title='About Claude',
                created_utc=int(datetime.utcnow().timestamp())
            )
            
            session.add_all([keyword, subreddit, post])
            session.flush()
            
            # Add keyword to post
            post.keywords.append(keyword)
            session.flush()
            
            # Test relationship both ways
            assert len(post.keywords) == 1
            assert post.keywords[0].term == 'Claude'
            assert len(keyword.posts) == 1
            assert keyword.posts[0].id == 'post789'


class TestDatabaseManager:
    """Test DatabaseManager operations."""
    
    @pytest.fixture
    def db_manager(self):
        """Create a temporary database for testing."""
        manager = DatabaseManager(db_path=':memory:')
        yield manager
        manager.close()
    
    @pytest.fixture
    def sample_posts(self):
        """Generate sample post data."""
        return [
            {
                'id': f'post_{i}',
                'subreddit': 'ClaudeAI',
                'title': f'Post Title {i}',
                'selftext': f'Post body {i}',
                'author': f'user_{i}',
                'url': f'https://reddit.com/post_{i}',
                'created_utc': int((datetime.utcnow() - timedelta(hours=i)).timestamp()),
                'score': i * 10,
                'num_comments': i * 2
            }
            for i in range(1, 6)
        ]
    
    @pytest.fixture
    def sample_comments(self):
        """Generate sample comment data."""
        return [
            {
                'id': f'comment_{i}',
                'post_id': 'post_1',
                'parent_id': 'post_1' if i <= 2 else f'comment_{i-1}',
                'author': f'commenter_{i}',
                'body': f'Comment text {i}',
                'score': i * 5,
                'created_utc': int(datetime.utcnow().timestamp())
            }
            for i in range(1, 6)
        ]
    
    def test_create_analysis_run(self, db_manager):
        """Test creating a new analysis run."""
        run = db_manager.create_analysis_run()
        
        assert run.id is not None
        assert run.status == 'running'
        assert run.started_at is not None
        assert run.completed_at is None
    
    def test_complete_analysis_run(self, db_manager):
        """Test completing an analysis run with stats."""
        run = db_manager.create_analysis_run()
        
        stats = {
            'total_posts': 100,
            'total_comments': 500,
            'api_calls': 25,
            'api_cost_estimate': 1.75
        }
        
        db_manager.complete_analysis_run(run.id, stats)
        
        # Verify updates
        with db_manager.transaction() as session:
            updated_run = session.query(AnalysisRun).filter_by(id=run.id).first()
            assert updated_run.status == 'completed'
            assert updated_run.completed_at is not None
            assert updated_run.total_posts == 100
            assert updated_run.api_cost_estimate == 1.75
    
    def test_fail_analysis_run(self, db_manager):
        """Test marking an analysis run as failed."""
        run = db_manager.create_analysis_run()
        
        error_msg = "API rate limit exceeded"
        db_manager.fail_analysis_run(run.id, error_msg)
        
        with db_manager.transaction() as session:
            failed_run = session.query(AnalysisRun).filter_by(id=run.id).first()
            assert failed_run.status == 'failed'
            assert failed_run.error_message == error_msg
            assert failed_run.completed_at is not None
    
    def test_get_or_create_subreddit_new(self, db_manager):
        """Test creating a new subreddit."""
        subreddit = db_manager.get_or_create_subreddit('ClaudeAI')
        
        assert subreddit.id is not None
        assert subreddit.name == 'ClaudeAI'
    
    def test_get_or_create_subreddit_existing(self, db_manager):
        """Test getting an existing subreddit."""
        # Create first
        sub1 = db_manager.get_or_create_subreddit('ClaudeAI')
        
        # Get existing
        sub2 = db_manager.get_or_create_subreddit('ClaudeAI')
        
        assert sub1.id == sub2.id
        assert sub1.name == sub2.name
    
    def test_bulk_insert_posts_success(self, db_manager, sample_posts):
        """Test successful bulk post insertion."""
        run = db_manager.create_analysis_run()
        
        inserted = db_manager.bulk_insert_posts(sample_posts, run.id)
        
        assert inserted == 5
        
        # Verify posts in database
        with db_manager.transaction() as session:
            posts = session.query(Post).all()
            assert len(posts) == 5
            assert posts[0].title == 'Post Title 1'
    
    def test_bulk_insert_posts_duplicate_handling(self, db_manager, sample_posts):
        """Test that duplicate posts are skipped."""
        run = db_manager.create_analysis_run()
        
        # Insert first time
        inserted1 = db_manager.bulk_insert_posts(sample_posts, run.id)
        assert inserted1 == 5
        
        # Try to insert again
        inserted2 = db_manager.bulk_insert_posts(sample_posts, run.id)
        assert inserted2 == 0  # All should be skipped
        
        # Verify still only 5 posts
        with db_manager.transaction() as session:
            count = session.query(Post).count()
            assert count == 5
    
    def test_bulk_insert_comments_success(self, db_manager, sample_posts, sample_comments):
        """Test successful bulk comment insertion."""
        run = db_manager.create_analysis_run()
        
        # Need posts first
        db_manager.bulk_insert_posts(sample_posts, run.id)
        
        # Insert comments
        inserted = db_manager.bulk_insert_comments(sample_comments)
        
        assert inserted == 5
        
        # Verify comments in database
        with db_manager.transaction() as session:
            comments = session.query(Comment).all()
            assert len(comments) == 5
    
    def test_bulk_insert_comments_missing_post(self, db_manager, sample_comments):
        """Test that comments without parent posts are skipped."""
        # Try to insert comments without posts
        inserted = db_manager.bulk_insert_comments(sample_comments)
        
        assert inserted == 0  # All should be skipped
    
    def test_update_sentiment_scores_posts(self, db_manager, sample_posts):
        """Test updating sentiment scores for posts."""
        run = db_manager.create_analysis_run()
        db_manager.bulk_insert_posts(sample_posts, run.id)
        
        # Prepare sentiment updates
        sentiment_data = [
            {'id': 'post_1', 'sentiment_score': 0.8, 'sentiment_explanation': 'Positive'},
            {'id': 'post_2', 'sentiment_score': -0.3, 'sentiment_explanation': 'Negative'},
            {'id': 'post_3', 'sentiment_score': 0.1, 'sentiment_explanation': 'Neutral'}
        ]
        
        db_manager.update_sentiment_scores(sentiment_data, 'post')
        
        # Verify updates
        with db_manager.transaction() as session:
            post1 = session.query(Post).filter_by(id='post_1').first()
            assert post1.sentiment_score == 0.8
            assert post1.sentiment_explanation == 'Positive'
            assert post1.analyzed_at is not None
    
    def test_get_posts_for_analysis(self, db_manager, sample_posts):
        """Test retrieving unanalyzed posts."""
        run = db_manager.create_analysis_run()
        db_manager.bulk_insert_posts(sample_posts, run.id)
        
        # Get unanalyzed posts
        posts = db_manager.get_posts_for_analysis(batch_size=3, run_id=run.id)
        
        assert len(posts) == 3
        assert all(p.sentiment_score is None for p in posts)
    
    def test_generate_daily_summary(self, db_manager, sample_posts):
        """Test generating daily summaries."""
        run = db_manager.create_analysis_run()
        db_manager.bulk_insert_posts(sample_posts, run.id)
        
        # Add sentiment scores
        sentiment_data = [
            {'id': f'post_{i}', 'sentiment_score': (i - 3) * 0.3}
            for i in range(1, 6)
        ]
        db_manager.update_sentiment_scores(sentiment_data, 'post')
        
        # Generate summaries
        summaries = db_manager.generate_daily_summary(run.id)
        
        assert len(summaries) == 1  # One subreddit
        summary = summaries[0]
        assert summary.total_posts == 5
        assert summary.positive_posts == 1  # post_5 with 0.6
        assert summary.negative_posts == 1  # post_1 with -0.6
        assert summary.neutral_posts == 3  # posts 2, 3, 4 between -0.3 and 0.3
    
    def test_cleanup_old_data(self, db_manager, sample_posts):
        """Test cleaning up old data."""
        run = db_manager.create_analysis_run()
        
        # Modify posts to be old
        for post in sample_posts:
            post['created_utc'] = int((datetime.utcnow() - timedelta(days=100)).timestamp())
        
        db_manager.bulk_insert_posts(sample_posts, run.id)
        
        # Verify posts exist
        with db_manager.transaction() as session:
            count_before = session.query(Post).count()
            assert count_before == 5
        
        # Clean up data older than 90 days
        db_manager.cleanup_old_data(days_to_keep=90)
        
        # Verify posts deleted
        with db_manager.transaction() as session:
            count_after = session.query(Post).count()
            assert count_after == 0

    def test_cleanup_old_data_deletes_comments(self, db_manager, sample_posts, sample_comments):
        """Cleanup should delete comments via ON DELETE CASCADE."""
        run = db_manager.create_analysis_run()
        # Make posts old
        for post in sample_posts:
            post['created_utc'] = int((datetime.utcnow() - timedelta(days=100)).timestamp())
        db_manager.bulk_insert_posts(sample_posts, run.id)
        # Insert comments for these posts
        inserted_comments = db_manager.bulk_insert_comments(sample_comments)
        assert inserted_comments == 5
        with db_manager.transaction() as session:
            assert session.query(Comment).count() == 5
        # Cleanup should remove posts and cascade to comments
        db_manager.cleanup_old_data(days_to_keep=90)
        with db_manager.transaction() as session:
            assert session.query(Post).count() == 0
            assert session.query(Comment).count() == 0
    
    def test_get_recent_posts(self, db_manager, sample_posts):
        """Test retrieving recent posts."""
        run = db_manager.create_analysis_run()
        db_manager.bulk_insert_posts(sample_posts, run.id)
        
        # Get posts from last 7 days
        recent = db_manager.get_recent_posts(days=7)
        
        assert len(recent) == 5
        assert recent[0].id == 'post_1'  # Most recent first
    
    def test_get_sentiment_trend(self, db_manager, sample_posts):
        """Test getting sentiment trend over time."""
        run = db_manager.create_analysis_run()
        
        # Spread posts across different days
        for i, post in enumerate(sample_posts):
            post['created_utc'] = int((datetime.utcnow() - timedelta(days=i)).timestamp())
        
        db_manager.bulk_insert_posts(sample_posts, run.id)
        
        # Add sentiment scores
        sentiment_data = [
            {'id': f'post_{i}', 'sentiment_score': (i - 3) * 0.2}
            for i in range(1, 6)
        ]
        db_manager.update_sentiment_scores(sentiment_data, 'post')
        
        # Get trend
        trend = db_manager.get_sentiment_trend(days=30)
        
        assert len(trend) > 0
        assert 'date' in trend[0]
        assert 'avg_sentiment' in trend[0]
        assert 'post_count' in trend[0]

    def test_generate_daily_summary_keyword_mentions_and_avg_comment_sentiment(self, db_manager, sample_posts, sample_comments):
        """Daily summary includes avg comment sentiment and keyword mentions."""
        import json as _json
        run = db_manager.create_analysis_run()
        # Insert posts and comments
        db_manager.bulk_insert_posts(sample_posts, run.id)
        db_manager.bulk_insert_comments(sample_comments)
        # Set comment sentiment scores
        comment_scores = [
            { 'id': f'comment_{i}', 'sentiment_score': score }
            for i, score in zip(range(1, 6), [0.5, -0.5, 0.0, 1.0, -1.0])
        ]
        db_manager.update_sentiment_scores(comment_scores, item_type='comment')
        # Add a keyword and associate to a post and a comment
        with db_manager.transaction() as session:
            kw = Keyword(term='Claude', category='product')
            session.add(kw)
            session.flush()
            p1 = session.query(Post).filter_by(id='post_1').first()
            c1 = session.query(Comment).filter_by(id='comment_1').first()
            p1.keywords.append(kw)
            c1.keywords.append(kw)
        # Generate summary
        summaries = db_manager.generate_daily_summary(run.id)
        assert len(summaries) == 1
        summary = summaries[0]
        # Avg comment sentiment should be present and close to 0.0
        assert summary.avg_comment_sentiment is not None
        assert abs(summary.avg_comment_sentiment - 0.0) < 1e-6
        # Keyword mentions should include Claude with count 2 (post + comment)
        mentions = _json.loads(summary.keyword_mentions)
        assert mentions.get('Claude') == 2


class TestTransactionManagement:
    """Test database transaction handling."""
    
    @pytest.fixture
    def db_manager(self):
        """Create a temporary database for testing."""
        manager = DatabaseManager(db_path=':memory:')
        yield manager
        manager.close()
    
    def test_transaction_commit(self, db_manager):
        """Test successful transaction commit."""
        with db_manager.transaction() as session:
            subreddit = Subreddit(name='test')
            session.add(subreddit)
        
        # Verify committed
        with db_manager.transaction() as session:
            count = session.query(Subreddit).count()
            assert count == 1
    
    def test_transaction_rollback_on_error(self, db_manager):
        """Test transaction rollback on error."""
        try:
            with db_manager.transaction() as session:
                subreddit = Subreddit(name='test')
                session.add(subreddit)
                raise ValueError("Test error")
        except ValueError:
            pass
        
        # Verify rolled back
        with db_manager.transaction() as session:
            count = session.query(Subreddit).count()
            assert count == 0
    
    def test_integrity_constraint_handling(self, db_manager):
        """Test handling of integrity constraint violations."""
        # Create a subreddit
        with db_manager.transaction() as session:
            sub1 = Subreddit(name='unique_name')
            session.add(sub1)
        
        # Try to create duplicate
        with pytest.raises(IntegrityError):
            with db_manager.transaction() as session:
                sub2 = Subreddit(name='unique_name')
                session.add(sub2)


class TestConnectionPooling:
    """Test database connection pooling."""
    
    def test_connection_pool_initialization(self):
        """Test that connection pool is properly initialized."""
        manager = DatabaseManager(db_path=':memory:')
        
        # Check that engine and session are initialized
        assert manager.engine is not None
        assert manager.Session is not None
        
        manager.close()
    
    def test_concurrent_connections(self, tmpdir):
        """Test handling multiple concurrent connections."""
        import threading
        
        db_path = str(tmpdir.join("test.db"))
        manager = DatabaseManager(db_path=db_path)
        
        results = []
        
        def worker():
            try:
                with manager.transaction() as session:
                    count = session.query(Subreddit).count()
                    results.append(True)
            except Exception as e:
                results.append(False)
        
        # Start multiple threads
        threads = []
        for _ in range(10):
            t = threading.Thread(target=worker)
            threads.append(t)
            t.start()
        
        # Wait for all threads
        for t in threads:
            t.join()
        
        # All should succeed
        assert all(results)
        
        manager.close()


class TestErrorHandling:
    """Test error handling scenarios."""
    
    @pytest.fixture
    def db_manager(self):
        """Create a temporary database for testing."""
        manager = DatabaseManager(db_path=':memory:')
        yield manager
        manager.close()
    
    def test_database_locked_retry(self, db_manager):
        """Test retry logic for database locked errors."""
        # This is hard to test directly with SQLite
        # Would need to mock the session to simulate locked database
        pass
    
    def test_invalid_data_handling(self, db_manager):
        """Test handling of invalid data during insertion."""
        # Try to insert post with missing required field
        invalid_post = [{
            'id': 'invalid',
            'subreddit': 'test',
            # Missing 'title' and 'created_utc'
        }]
        
        run = db_manager.create_analysis_run()
        
        # Should handle gracefully
        with pytest.raises(Exception):
            db_manager.bulk_insert_posts(invalid_post, run.id)


class TestPerformance:
    """Test performance benchmarks."""
    
    @pytest.fixture
    def db_manager(self):
        """Create a temporary database for testing."""
        manager = DatabaseManager(db_path=':memory:')
        yield manager
        manager.close()
    
    def test_bulk_insert_performance(self, db_manager):
        """Test performance of bulk insert operations."""
        import time
        
        # Generate large dataset
        large_dataset = [
            {
                'id': f'post_{i}',
                'subreddit': 'ClaudeAI',
                'title': f'Post Title {i}',
                'created_utc': int(datetime.utcnow().timestamp()),
                'score': i
            }
            for i in range(1000)
        ]
        
        run = db_manager.create_analysis_run()
        
        start = time.time()
        inserted = db_manager.bulk_insert_posts(large_dataset, run.id)
        duration = time.time() - start
        
        assert inserted == 1000
        assert duration < 5  # Should complete in under 5 seconds
        
        print(f"Bulk insert of 1000 posts took {duration:.2f} seconds")
    
    def test_query_performance(self, db_manager):
        """Test performance of common queries."""
        import time
        
        # Setup data
        posts = [
            {
                'id': f'post_{i}',
                'subreddit': 'ClaudeAI',
                'title': f'Post {i}',
                'created_utc': int((datetime.utcnow() - timedelta(days=i % 30)).timestamp()),
                'score': i
            }
            for i in range(500)
        ]
        
        run = db_manager.create_analysis_run()
        db_manager.bulk_insert_posts(posts, run.id)
        
        # Test query performance
        start = time.time()
        recent = db_manager.get_recent_posts(days=7)
        duration = time.time() - start
        
        assert duration < 1  # Should complete in under 1 second
        
        print(f"Query for recent posts took {duration:.2f} seconds")
