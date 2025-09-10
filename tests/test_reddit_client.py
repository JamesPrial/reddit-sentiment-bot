"""
Unit tests for Reddit client module.
"""

import os
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch, PropertyMock

import pytest
from prawcore.exceptions import ResponseException, RequestException

from src.reddit_client import RedditClient, RateLimitHandler


class TestRateLimitHandler:
    """Test rate limiting functionality."""
    
    def test_init(self):
        """Test rate limiter initialization."""
        handler = RateLimitHandler(requests_per_minute=30)
        assert handler.requests_per_minute == 30
        assert handler.request_times == []
        assert handler.min_interval == 2.0  # 60/30
    
    def test_record_request(self):
        """Test request tracking."""
        handler = RateLimitHandler()
        handler.record_request()
        handler.record_request()
        assert len(handler.request_times) == 2
        assert all(isinstance(t, float) for t in handler.request_times)
    
    def test_wait_if_needed_under_limit(self):
        """Test no waiting when under rate limit."""
        handler = RateLimitHandler(requests_per_minute=100)
        handler.record_request()
        
        # Should not wait
        start = time.time()
        handler.wait_if_needed()
        elapsed = time.time() - start
        assert elapsed < 0.1
    
    @patch('time.sleep')
    def test_wait_if_needed_at_limit(self, mock_sleep):
        """Test waiting when at rate limit."""
        handler = RateLimitHandler(requests_per_minute=2)
        
        # Fill up the limit
        now = time.time()
        handler.request_times = [now - 30, now - 1]  # 2 requests in last minute
        
        handler.wait_if_needed()
        
        # Should sleep for ~30 seconds
        mock_sleep.assert_called_once()
        sleep_time = mock_sleep.call_args[0][0]
        assert 29 < sleep_time < 31
    
    def test_cleanup_old_requests(self):
        """Test removal of requests older than 1 minute."""
        handler = RateLimitHandler()
        now = time.time()
        
        # Add mix of old and recent requests
        handler.request_times = [
            now - 120,  # 2 minutes ago
            now - 61,   # Just over 1 minute
            now - 30,   # 30 seconds ago
            now - 5     # 5 seconds ago
        ]
        
        handler.wait_if_needed()
        
        # Should only keep last 2 requests
        assert len(handler.request_times) == 2


class TestRedditClient:
    """Test Reddit client functionality."""
    
    def test_init_with_credentials(self):
        """Test initialization with provided credentials."""
        creds = {
            'client_id': 'test_id',
            'client_secret': 'test_secret',
            'user_agent': 'test_agent'
        }
        
        with patch('praw.Reddit') as mock_reddit:
            client = RedditClient(credentials=creds)
            
            mock_reddit.assert_called_once_with(
                client_id='test_id',
                client_secret='test_secret',
                user_agent='test_agent'
            )
            assert client.reddit is not None
            assert isinstance(client.rate_limiter, RateLimitHandler)
    
    def test_init_from_environment(self):
        """Test initialization from environment variables."""
        with patch.dict(os.environ, {
            'REDDIT_CLIENT_ID': 'env_id',
            'REDDIT_CLIENT_SECRET': 'env_secret',
            'REDDIT_USER_AGENT': 'env_agent'
        }):
            with patch('praw.Reddit') as mock_reddit:
                client = RedditClient()
                
                mock_reddit.assert_called_once_with(
                    client_id='env_id',
                    client_secret='env_secret',
                    user_agent='env_agent'
                )
    
    def test_init_missing_credentials(self):
        """Test error when credentials are missing."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Reddit credentials not found"):
                RedditClient()
    
    def test_fetch_subreddit_posts_success(self):
        """Test successful post fetching."""
        # Mock submission objects
        mock_posts = []
        cutoff = datetime.utcnow() - timedelta(hours=12)
        
        for i in range(3):
            post = Mock()
            post.id = f'post{i}'
            post.title = f'Title {i}'
            post.selftext = f'Content {i}'
            post.created_utc = cutoff.timestamp() + (i * 3600)  # Within 24h
            post.score = i * 10
            post.num_comments = i * 2
            post.upvote_ratio = 0.9
            post.url = f'http://reddit.com/post{i}'
            post.permalink = f'/r/test/post{i}'
            post.author = Mock()
            post.author.__str__ = Mock(return_value=f'user{i}')
            post.subreddit = Mock()
            post.subreddit.display_name = 'ClaudeAI'
            mock_posts.append(post)
        
        # Add old post that should be filtered
        old_post = Mock()
        old_post.created_utc = (datetime.utcnow() - timedelta(days=2)).timestamp()
        mock_posts.append(old_post)
        
        with patch('praw.Reddit') as mock_reddit_class:
            mock_reddit = Mock()
            mock_reddit_class.return_value = mock_reddit
            mock_subreddit = Mock()
            mock_reddit.subreddit.return_value = mock_subreddit
            mock_subreddit.new.return_value = mock_posts
            
            client = RedditClient(credentials={
                'client_id': 'test',
                'client_secret': 'test',
                'user_agent': 'test'
            })
            
            posts = client.fetch_subreddit_posts('ClaudeAI')
            
            # Should get 3 posts (not the old one)
            assert len(posts) == 3
            assert posts[0]['id'] == 'post0'
            assert posts[1]['title'] == 'Title 1'
            assert posts[2]['selftext'] == 'Content 2'
            assert all(p['subreddit'] == 'ClaudeAI' for p in posts)
    
    def test_fetch_subreddit_posts_not_found(self):
        """Test handling of non-existent subreddit."""
        with patch('praw.Reddit') as mock_reddit_class:
            mock_reddit = Mock()
            mock_reddit_class.return_value = mock_reddit
            
            # Simulate 404 error
            mock_response = Mock()
            mock_response.status_code = 404
            mock_reddit.subreddit.side_effect = ResponseException(mock_response)
            
            client = RedditClient(credentials={
                'client_id': 'test',
                'client_secret': 'test',
                'user_agent': 'test'
            })
            
            posts = client.fetch_subreddit_posts('NonExistent')
            assert posts == []
    
    def test_fetch_subreddit_posts_rate_limit(self):
        """Test handling of rate limit errors."""
        with patch('praw.Reddit') as mock_reddit_class:
            mock_reddit = Mock()
            mock_reddit_class.return_value = mock_reddit
            
            # Simulate 429 error
            mock_response = Mock()
            mock_response.status_code = 429
            mock_reddit.subreddit.side_effect = ResponseException(mock_response)
            
            client = RedditClient(credentials={
                'client_id': 'test',
                'client_secret': 'test',
                'user_agent': 'test'
            })
            
            posts = client.fetch_subreddit_posts('ClaudeAI')
            assert posts == []
    
    def test_fetch_subreddit_posts_network_error(self):
        """Test handling of network errors."""
        with patch('praw.Reddit') as mock_reddit_class:
            mock_reddit = Mock()
            mock_reddit_class.return_value = mock_reddit
            # RequestException requires request_args and request_kwargs
            mock_reddit.subreddit.side_effect = RequestException(
                request_args=(),
                request_kwargs={},
                original_exception=Exception("Network error")
            )
            
            client = RedditClient(credentials={
                'client_id': 'test',
                'client_secret': 'test',
                'user_agent': 'test'
            })
            
            posts = client.fetch_subreddit_posts('ClaudeAI')
            assert posts == []
    
    def test_fetch_post_comments_success(self):
        """Test successful comment fetching."""
        # Mock comment objects - need to be proper Comment type
        from praw.models import Comment
        mock_comments = []
        for i in range(3):
            comment = Mock(spec=Comment)
            comment.id = f'comment{i}'
            comment.body = f'Comment body {i}'
            comment.parent_id = f't3_post123' if i == 0 else f't1_comment{i-1}'
            comment.created_utc = 1234567890 + i
            comment.score = i * 5
            comment.is_submitter = (i == 0)
            comment.permalink = f'/r/test/comments/post123/comment{i}'
            comment.author = Mock()
            comment.author.__str__ = Mock(return_value=f'commenter{i}')
            mock_comments.append(comment)
        
        with patch('praw.Reddit') as mock_reddit_class:
            mock_reddit = Mock()
            mock_reddit_class.return_value = mock_reddit
            mock_submission = Mock()
            mock_reddit.submission.return_value = mock_submission
            mock_submission.comments = Mock()
            mock_submission.comments.replace_more = Mock()
            mock_submission.comments.list.return_value = mock_comments
            
            client = RedditClient(credentials={
                'client_id': 'test',
                'client_secret': 'test',
                'user_agent': 'test'
            })
            
            comments = client.fetch_post_comments('post123')
            
            assert len(comments) == 3
            assert comments[0]['id'] == 'comment0'
            assert comments[1]['body'] == 'Comment body 1'
            assert comments[2]['author'] == 'commenter2'
            assert all(c['post_id'] == 'post123' for c in comments)
    
    def test_fetch_post_comments_with_limit(self):
        """Test comment fetching with limit."""
        from praw.models import Comment
        mock_comments = [Mock(spec=Comment) for _ in range(10)]
        for i, comment in enumerate(mock_comments):
            comment.id = f'comment{i}'
            comment.body = f'Body {i}'
            comment.parent_id = 't3_post123'
            comment.created_utc = 1234567890
            comment.score = 0
            comment.is_submitter = False
            comment.permalink = '/test'
            comment.author = None
        
        with patch('praw.Reddit') as mock_reddit_class:
            mock_reddit = Mock()
            mock_reddit_class.return_value = mock_reddit
            mock_submission = Mock()
            mock_reddit.submission.return_value = mock_submission
            mock_submission.comments = Mock()
            mock_submission.comments.replace_more = Mock()
            mock_submission.comments.list.return_value = mock_comments
            
            client = RedditClient(credentials={
                'client_id': 'test',
                'client_secret': 'test',
                'user_agent': 'test'
            })
            
            comments = client.fetch_post_comments('post123', limit=5)
            
            assert len(comments) == 5
            assert comments[0]['id'] == 'comment0'
            assert comments[4]['id'] == 'comment4'
    
    def test_fetch_post_comments_error(self):
        """Test error handling in comment fetching."""
        with patch('praw.Reddit') as mock_reddit_class:
            mock_reddit = Mock()
            mock_reddit_class.return_value = mock_reddit
            mock_reddit.submission.side_effect = Exception("API Error")
            
            client = RedditClient(credentials={
                'client_id': 'test',
                'client_secret': 'test',
                'user_agent': 'test'
            })
            
            comments = client.fetch_post_comments('post123')
            assert comments == []
    
    def test_fetch_all_daily_data(self):
        """Test orchestrated fetching of posts and comments."""
        # Mock posts
        mock_posts = [
            {'id': 'post1', 'title': 'Post 1', 'subreddit': 'ClaudeAI'},
            {'id': 'post2', 'title': 'Post 2', 'subreddit': 'ClaudeAI'}
        ]
        
        # Mock comments
        mock_comments_1 = [
            {'id': 'c1', 'post_id': 'post1', 'body': 'Comment 1'},
            {'id': 'c2', 'post_id': 'post1', 'body': 'Comment 2'}
        ]
        mock_comments_2 = [
            {'id': 'c3', 'post_id': 'post2', 'body': 'Comment 3'}
        ]
        
        with patch('praw.Reddit'):
            client = RedditClient(credentials={
                'client_id': 'test',
                'client_secret': 'test',
                'user_agent': 'test'
            })
            
            # Mock the fetch methods
            with patch.object(client, 'fetch_subreddit_posts', return_value=mock_posts):
                with patch.object(client, 'fetch_post_comments') as mock_fetch_comments:
                    mock_fetch_comments.side_effect = [mock_comments_1, mock_comments_2]
                    
                    result = client.fetch_all_daily_data(['ClaudeAI'])
                    
                    assert len(result['posts']) == 2
                    assert len(result['comments']) == 3
                    assert result['posts'][0]['title'] == 'Post 1'
                    assert result['comments'][2]['body'] == 'Comment 3'
    
    def test_extract_post_data_normal(self):
        """Test post data extraction with normal author."""
        mock_submission = Mock()
        mock_submission.id = 'abc123'
        mock_submission.title = 'Test Post'
        mock_submission.selftext = 'Post content'
        mock_submission.created_utc = 1234567890
        mock_submission.score = 42
        mock_submission.num_comments = 10
        mock_submission.upvote_ratio = 0.85
        mock_submission.url = 'http://reddit.com/post'
        mock_submission.permalink = '/r/test/comments/abc123'
        mock_submission.author = Mock()
        mock_submission.author.__str__ = Mock(return_value='testuser')
        mock_submission.subreddit = Mock()
        mock_submission.subreddit.display_name = 'ClaudeAI'
        
        with patch('praw.Reddit'):
            client = RedditClient(credentials={
                'client_id': 'test',
                'client_secret': 'test',
                'user_agent': 'test'
            })
            
            data = client._extract_post_data(mock_submission)
            
            assert data['id'] == 'abc123'
            assert data['title'] == 'Test Post'
            assert data['selftext'] == 'Post content'
            assert data['author'] == 'testuser'
            assert data['score'] == 42
            assert data['subreddit'] == 'ClaudeAI'
    
    def test_extract_post_data_deleted_author(self):
        """Test post data extraction with deleted author."""
        mock_submission = Mock()
        mock_submission.id = 'xyz789'
        mock_submission.title = 'Deleted Author Post'
        mock_submission.selftext = None  # Also test None selftext
        mock_submission.created_utc = 1234567890
        mock_submission.score = 0
        mock_submission.num_comments = 0
        mock_submission.upvote_ratio = 0.5
        mock_submission.url = 'http://reddit.com/post'
        mock_submission.permalink = '/r/test/comments/xyz789'
        mock_submission.author = None  # Deleted author
        mock_submission.subreddit = Mock()
        mock_submission.subreddit.display_name = 'ClaudeAI'
        
        with patch('praw.Reddit'):
            client = RedditClient(credentials={
                'client_id': 'test',
                'client_secret': 'test',
                'user_agent': 'test'
            })
            
            data = client._extract_post_data(mock_submission)
            
            assert data['author'] == '[deleted]'
            assert data['selftext'] == ''  # None converted to empty string
    
    def test_extract_comment_data_normal(self):
        """Test comment data extraction."""
        mock_comment = Mock()
        mock_comment.id = 'com123'
        mock_comment.body = 'Great post!'
        mock_comment.parent_id = 't3_post123'
        mock_comment.created_utc = 1234567890
        mock_comment.score = 5
        mock_comment.is_submitter = False
        mock_comment.permalink = '/r/test/comments/post123/com123'
        mock_comment.author = Mock()
        mock_comment.author.__str__ = Mock(return_value='commenter')
        
        with patch('praw.Reddit'):
            client = RedditClient(credentials={
                'client_id': 'test',
                'client_secret': 'test',
                'user_agent': 'test'
            })
            
            data = client._extract_comment_data(mock_comment, 'post123')
            
            assert data['id'] == 'com123'
            assert data['post_id'] == 'post123'
            assert data['body'] == 'Great post!'
            assert data['author'] == 'commenter'
            assert data['score'] == 5
            assert not data['is_submitter']
    
    def test_extract_comment_data_deleted_author(self):
        """Test comment data extraction with deleted author."""
        mock_comment = Mock()
        mock_comment.id = 'com456'
        mock_comment.body = '[removed]'
        mock_comment.parent_id = 't1_com123'  # Reply to another comment
        mock_comment.created_utc = 1234567890
        mock_comment.score = -2
        mock_comment.is_submitter = True
        mock_comment.permalink = '/r/test/comments/post123/com456'
        mock_comment.author = None  # Deleted
        
        with patch('praw.Reddit'):
            client = RedditClient(credentials={
                'client_id': 'test',
                'client_secret': 'test',
                'user_agent': 'test'
            })
            
            data = client._extract_comment_data(mock_comment, 'post123')
            
            assert data['author'] == '[deleted]'
            assert data['parent_id'] == 't1_com123'
            assert data['is_submitter']