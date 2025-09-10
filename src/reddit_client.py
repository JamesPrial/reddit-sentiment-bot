"""
Reddit API client for fetching posts and comments from subreddits.

Uses PRAW (Python Reddit API Wrapper) to fetch all posts from the last 24 hours
and their associated comments. Includes rate limiting and comprehensive error handling.
"""

import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable

import praw
from praw.models import Subreddit, Submission, Comment, MoreComments
from prawcore.exceptions import ResponseException, RequestException
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

logger = logging.getLogger(__name__)


class RateLimitHandler:
    """Manages API rate limiting to avoid hitting Reddit's limits."""
    
    def __init__(self, requests_per_minute: int = 60):
        """
        Initialize rate limit handler.
        
        Args:
            requests_per_minute: Maximum requests allowed per minute
        """
        self.requests_per_minute = requests_per_minute
        self.request_times: List[float] = []
        self.min_interval = 60.0 / requests_per_minute
    
    def wait_if_needed(self):
        """Sleep if approaching rate limit."""
        now = time.time()
        
        # Remove requests older than 1 minute
        self.request_times = [t for t in self.request_times if now - t < 60]
        
        # If at limit, wait
        if len(self.request_times) >= self.requests_per_minute:
            sleep_time = 60 - (now - self.request_times[0]) + 0.1
            if sleep_time > 0:
                logger.info(f"Rate limit reached, sleeping for {sleep_time:.1f}s")
                time.sleep(sleep_time)
    
    def record_request(self):
        """Track API request for rate limiting."""
        self.request_times.append(time.time())


def _should_retry(exc: Exception) -> bool:
    """Retry on network errors and select HTTP statuses (429, 5xx)."""
    if isinstance(exc, RequestException):
        return True
    if isinstance(exc, ResponseException):
        code = getattr(exc.response, 'status_code', None)
        return code in {429, 500, 502, 503, 504}
    return False


class RedditClient:
    """Client for interacting with Reddit API via PRAW."""
    
    def __init__(self, credentials: Optional[Dict[str, str]] = None, requests_per_minute: int = 60):
        """
        Initialize Reddit client with credentials.
        
        Args:
            credentials: Dict with client_id, client_secret, user_agent
                        If not provided, uses environment variables
        """
        if credentials:
            client_id = credentials.get('client_id')
            client_secret = credentials.get('client_secret')
            user_agent = credentials.get('user_agent')
        else:
            client_id = os.environ.get('REDDIT_CLIENT_ID')
            client_secret = os.environ.get('REDDIT_CLIENT_SECRET')
            user_agent = os.environ.get('REDDIT_USER_AGENT')
        
        if not all([client_id, client_secret, user_agent]):
            raise ValueError("Reddit credentials not found. Ensure REDDIT_CLIENT_ID, "
                           "REDDIT_CLIENT_SECRET, and REDDIT_USER_AGENT are set")
        
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
        
        self.rate_limiter = RateLimitHandler(requests_per_minute=requests_per_minute)
        logger.info("RedditClient initialized successfully")

    # Retry helpers for transient failures
    @retry(
        retry=retry_if_exception(_should_retry),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.1, min=0.1, max=1.0),
        reraise=True,
    )
    def _get_subreddit(self, name: str):
        return self.reddit.subreddit(name)

    @retry(
        retry=retry_if_exception(_should_retry),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.1, min=0.1, max=1.0),
        reraise=True,
    )
    def _get_submission(self, post_id: str):
        return self.reddit.submission(id=post_id)
    
    def fetch_subreddit_posts(self, subreddit_name: str, 
                             time_filter: str = 'day') -> List[Dict[str, Any]]:
        """
        Fetch ALL posts from subreddit within time period.
        
        Args:
            subreddit_name: Name of subreddit (without r/)
            time_filter: Time period ('day', 'week', 'month')
        
        Returns:
            List of post dictionaries with metadata
        """
        posts = []
        
        try:
            self.rate_limiter.wait_if_needed()
            subreddit = self._get_subreddit(subreddit_name)
            self.rate_limiter.record_request()
            
            # Calculate cutoff time based on time_filter
            cutoff_timestamp: Optional[float]
            tf = (time_filter or 'day').lower()
            if tf == 'all':
                cutoff_timestamp = None
            else:
                days = {
                    'hour': 1.0 / 24.0,
                    'day': 1.0,
                    'week': 7.0,
                    'month': 30.0,
                    'year': 365.0,
                }.get(tf, 1.0)
                cutoff_time = datetime.utcnow() - timedelta(days=days)
                cutoff_timestamp = cutoff_time.timestamp()
            
            # Fetch new posts (most recent first)
            logger.info(f"Fetching new posts from r/{subreddit_name}")
            for submission in subreddit.new(limit=None):
                self.rate_limiter.wait_if_needed()
                
                # Stop when we hit posts older than cutoff (ignore stickied)
                if cutoff_timestamp is not None and submission.created_utc < cutoff_timestamp:
                    if getattr(submission, 'stickied', False):
                        # skip old stickied posts
                        continue
                    else:
                        break
                
                posts.append(self._extract_post_data(submission))
                self.rate_limiter.record_request()
            
            logger.info(f"Fetched {len(posts)} posts from r/{subreddit_name}")
            
        except ResponseException as e:
            if e.response.status_code == 404:
                logger.error(f"Subreddit r/{subreddit_name} not found")
            elif e.response.status_code == 403:
                logger.error(f"Access denied to r/{subreddit_name} (private/banned)")
            elif e.response.status_code == 429:
                logger.error("Rate limit exceeded, despite internal limiting")
            else:
                logger.error(f"Reddit API error for r/{subreddit_name}: {e}")
        except RequestException as e:
            logger.error(f"Network error fetching r/{subreddit_name}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error fetching r/{subreddit_name}: {e}")
        
        return posts
    
    def fetch_post_comments(self, post_id: str, 
                          limit: Optional[int] = None,
                          replace_more_limit: int = 0) -> List[Dict[str, Any]]:
        """
        Fetch ALL comments for a post (handles MoreComments).
        
        Args:
            post_id: Reddit post ID
            limit: Optional limit on number of comments
        
        Returns:
            Flattened list of comment dictionaries
        """
        comments = []
        
        try:
            self.rate_limiter.wait_if_needed()
            submission = self._get_submission(post_id)
            self.rate_limiter.record_request()
            
            # Expand all comments (replace MoreComments objects)
            self.rate_limiter.wait_if_needed()
            submission.comments.replace_more(limit=replace_more_limit)
            self.rate_limiter.record_request()
            
            # Extract all comments
            all_comments = submission.comments.list()
            
            for comment in all_comments:
                if limit and len(comments) >= limit:
                    break
                    
                if isinstance(comment, Comment):
                    comments.append(self._extract_comment_data(comment, post_id))
            
            logger.info(f"Fetched {len(comments)} comments for post {post_id}")
            
        except ResponseException as e:
            logger.error(f"Reddit API error fetching comments for {post_id}: {e}")
        except RequestException as e:
            logger.error(f"Network error fetching comments for {post_id}: {e}")
        except Exception as e:
            logger.error(f"Error fetching comments for post {post_id}: {e}")
        
        return comments
    
    def fetch_all_daily_data(self, subreddits: List[str]) -> Dict[str, Any]:
        """
        Orchestrate fetching all posts and comments from multiple subreddits.
        
        Args:
            subreddits: List of subreddit names
        
        Returns:
            Dictionary with 'posts' and 'comments' lists
        """
        all_posts = []
        all_comments = []
        
        for subreddit_name in subreddits:
            logger.info(f"Processing r/{subreddit_name}")
            
            # Fetch posts
            posts = self.fetch_subreddit_posts(subreddit_name)
            all_posts.extend(posts)
            
            # Fetch comments for each post
            for post in posts:
                comments = self.fetch_post_comments(post['id'])
                all_comments.extend(comments)
        
        logger.info(f"Total fetched: {len(all_posts)} posts, {len(all_comments)} comments")
        
        return {
            'posts': all_posts,
            'comments': all_comments
        }
    
    def _extract_post_data(self, submission: Submission) -> Dict[str, Any]:
        """
        Extract relevant data from PRAW Submission object.
        
        Args:
            submission: PRAW Submission object
        
        Returns:
            Dictionary with post data matching database schema
        """
        return {
            'id': submission.id,
            'subreddit': submission.subreddit.display_name,
            'title': submission.title,
            'selftext': submission.selftext or '',
            'author': str(submission.author) if submission.author else '[deleted]',
            'url': submission.url,
            'created_utc': int(submission.created_utc),
            'score': submission.score,
            'num_comments': submission.num_comments,
            'upvote_ratio': submission.upvote_ratio,
            'permalink': f"https://reddit.com{submission.permalink}"
        }
    
    def _extract_comment_data(self, comment: Comment, post_id: str) -> Dict[str, Any]:
        """
        Extract relevant data from PRAW Comment object.
        
        Args:
            comment: PRAW Comment object
            post_id: Parent post ID
        
        Returns:
            Dictionary with comment data matching database schema
        """
        return {
            'id': comment.id,
            'post_id': post_id,
            'parent_id': comment.parent_id,
            'body': comment.body,
            'author': str(comment.author) if comment.author else '[deleted]',
            'created_utc': int(comment.created_utc),
            'score': comment.score,
            'is_submitter': comment.is_submitter,
            'permalink': f"https://reddit.com{comment.permalink}"
        }
