"""
SQLAlchemy-based database operations with connection pooling and transaction management.

This module provides the database models and operations for the Reddit sentiment
analysis bot, including efficient bulk operations and transaction management.
"""

import os
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Any, Generator, TYPE_CHECKING, cast
from contextlib import contextmanager

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Text, DateTime, Date,
    ForeignKey, Table, Index, UniqueConstraint, func, and_, event
)
from sqlalchemy.orm import (
    relationship,
    sessionmaker,
    scoped_session,
    declarative_base,
    Session as SASession,
)
from sqlalchemy.pool import QueuePool, StaticPool
from sqlalchemy.exc import IntegrityError

logger = logging.getLogger(__name__)

Base = declarative_base()

from src.taxonomy import classify_term


# Association tables for many-to-many relationships
post_keywords = Table(
    'post_keywords',
    Base.metadata,
    Column('post_id', String, ForeignKey('posts.id', ondelete='CASCADE'), primary_key=True),
    Column('keyword_id', Integer, ForeignKey('keywords.id', ondelete='CASCADE'), primary_key=True)
)

comment_keywords = Table(
    'comment_keywords',
    Base.metadata,
    Column('comment_id', String, ForeignKey('comments.id', ondelete='CASCADE'), primary_key=True),
    Column('keyword_id', Integer, ForeignKey('keywords.id', ondelete='CASCADE'), primary_key=True)
)


class Subreddit(Base):
    """Model for subreddit information."""
    __tablename__ = 'subreddits'
    
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    # Relationships
    posts = relationship("Post", back_populates="subreddit", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Subreddit(name='{self.name}')>"


class AnalysisRun(Base):
    """Model for tracking each bot execution."""
    __tablename__ = 'analysis_runs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    run_date = Column(Date, nullable=False)
    started_at = Column(DateTime, nullable=False)
    completed_at = Column(DateTime)
    total_posts = Column(Integer, default=0)
    total_comments = Column(Integer, default=0)
    api_calls = Column(Integer, default=0)
    api_cost_estimate = Column(Float, default=0)
    status = Column(String, default='running')  # running, completed, failed
    error_message = Column(Text)
    
    # Relationships
    posts = relationship("Post", back_populates="analysis_run")
    summaries = relationship("DailySummary", back_populates="analysis_run")
    
    def __repr__(self):
        return f"<AnalysisRun(id={self.id}, status='{self.status}', date={self.run_date})>"


class Post(Base):
    """Model for Reddit posts with sentiment analysis."""
    __tablename__ = 'posts'
    
    id = Column(String, primary_key=True)  # Reddit post ID
    subreddit_id = Column(Integer, ForeignKey('subreddits.id'), nullable=False)
    analysis_run_id = Column(Integer, ForeignKey('analysis_runs.id'))
    title = Column(Text, nullable=False)
    selftext = Column(Text)
    author = Column(String)
    url = Column(String)
    created_utc = Column(Integer, nullable=False)
    score = Column(Integer, default=0)
    num_comments = Column(Integer, default=0)
    sentiment_score = Column(Float)
    sentiment_explanation = Column(Text)
    analyzed_at = Column(DateTime)
    
    # Relationships
    subreddit = relationship("Subreddit", back_populates="posts")
    analysis_run = relationship("AnalysisRun", back_populates="posts")
    comments = relationship("Comment", back_populates="post", cascade="all, delete-orphan")
    keywords = relationship("Keyword", secondary=post_keywords, back_populates="posts")
    
    # Indexes
    __table_args__ = (
        Index('idx_posts_created', 'created_utc'),
        Index('idx_posts_sentiment', 'sentiment_score'),
        Index('idx_posts_subreddit', 'subreddit_id'),
        Index('idx_posts_analysis_run', 'analysis_run_id'),
    )
    
    def __repr__(self):
        title_preview = (self.title or "")[:50]
        return f"<Post(id='{self.id}', title='{title_preview}...')>"


class Comment(Base):
    """Model for Reddit comments with sentiment analysis."""
    __tablename__ = 'comments'
    
    id = Column(String, primary_key=True)  # Reddit comment ID
    post_id = Column(String, ForeignKey('posts.id', ondelete='CASCADE'), nullable=False)
    parent_id = Column(String)  # Parent comment ID or post ID
    author = Column(String)
    body = Column(Text, nullable=False)
    score = Column(Integer, default=0)
    created_utc = Column(Integer, nullable=False)
    sentiment_score = Column(Float)
    analyzed_at = Column(DateTime)
    
    # Relationships
    post = relationship("Post", back_populates="comments")
    keywords = relationship("Keyword", secondary=comment_keywords, back_populates="comments")
    
    # Indexes
    __table_args__ = (
        Index('idx_comments_post', 'post_id'),
        Index('idx_comments_sentiment', 'sentiment_score'),
        Index('idx_comments_created', 'created_utc'),
    )
    
    def __repr__(self):
        return f"<Comment(id='{self.id}', post_id='{self.post_id}')>"


class Keyword(Base):
    """Model for tracked keywords and terms."""
    __tablename__ = 'keywords'
    
    id = Column(Integer, primary_key=True)
    term = Column(String, unique=True, nullable=False)
    category = Column(String)  # 'product', 'feature', 'company', 'competitor'
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    # Relationships
    posts = relationship("Post", secondary=post_keywords, back_populates="keywords")
    comments = relationship("Comment", secondary=comment_keywords, back_populates="keywords")
    
    def __repr__(self):
        return f"<Keyword(term='{self.term}', category='{self.category}')>"


class DailySummary(Base):
    """Model for pre-computed daily statistics."""
    __tablename__ = 'daily_summaries'
    
    id = Column(Integer, primary_key=True)
    analysis_run_id = Column(Integer, ForeignKey('analysis_runs.id'))
    subreddit_id = Column(Integer, ForeignKey('subreddits.id'))
    date = Column(Date, nullable=False)
    total_posts = Column(Integer, default=0)
    total_comments = Column(Integer, default=0)
    avg_post_sentiment = Column(Float)
    avg_comment_sentiment = Column(Float)
    positive_posts = Column(Integer, default=0)
    negative_posts = Column(Integer, default=0)
    neutral_posts = Column(Integer, default=0)
    top_positive_post_id = Column(String)
    top_negative_post_id = Column(String)
    most_discussed_post_id = Column(String)
    keyword_mentions = Column(Text)  # JSON string of keyword counts
    
    # Relationships
    analysis_run = relationship("AnalysisRun", back_populates="summaries")
    subreddit = relationship("Subreddit")
    
    # Indexes
    __table_args__ = (
        Index('idx_summary_date', 'date'),
        Index('idx_summary_subreddit', 'subreddit_id'),
        UniqueConstraint('date', 'subreddit_id', name='unique_daily_summary'),
    )
    
    def __repr__(self):
        return f"<DailySummary(date={self.date}, subreddit_id={self.subreddit_id})>"


class DatabaseManager:
    """Manager class for database operations with connection pooling and transactions."""
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize connection pool and session factory.
        
        Args:
            db_path: Path to SQLite database file
        """
        if db_path is None:
            db_path = os.environ.get('DATABASE_PATH', './data/sentiment.db')
        
        # Ensure directory exists for file-based databases
        if db_path != ':memory:':
            db_dir = os.path.dirname(db_path)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir)

        # Create engine with appropriate pooling for SQLite
        if db_path == ':memory:' or db_path.startswith('file::memory:'):
            # Shared in-memory DB for a single process/thread pool
            db_url = 'sqlite://'
            self.engine = create_engine(
                db_url,
                poolclass=StaticPool,
                connect_args={"check_same_thread": False},
                pool_pre_ping=True,
                echo=False,
            )
        else:
            db_url = f'sqlite:///{db_path}'
            self.engine = create_engine(
                db_url,
                poolclass=QueuePool,
                pool_size=5,
                max_overflow=10,
                pool_timeout=30,
                pool_pre_ping=True,
                connect_args={"check_same_thread": False},
                echo=False,
            )

        # Ensure SQLite enforces foreign key constraints
        if self.engine.dialect.name == 'sqlite':
            @event.listens_for(self.engine, "connect")
            def set_sqlite_pragma(dbapi_connection, connection_record):  # noqa: F811
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()
        
        # Create session factory
        self.Session = scoped_session(sessionmaker(bind=self.engine, expire_on_commit=False))
        
        # Create tables if they don't exist
        Base.metadata.create_all(self.engine)
        
        logger.info(f"Database initialized at {db_path}")
    
    @contextmanager
    def transaction(self) -> Generator[SASession, None, None]:
        """
        Context manager for database transactions.
        
        Yields:
            Session object for database operations
        """
        session: SASession = self.Session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Transaction rolled back: {e}")
            raise
        finally:
            session.close()
    
    def create_analysis_run(self) -> AnalysisRun:
        """
        Create a new analysis run entry.
        
        Returns:
            Created AnalysisRun object
        """
        with self.transaction() as session:
            run = AnalysisRun(
                run_date=datetime.now(timezone.utc).date(),
                started_at=datetime.now(timezone.utc),
                status='running'
            )
            session.add(run)
            session.flush()  # Get the ID
            # Create a detached copy with the ID
            run_id = run.id
            session.expunge(run)
            run.id = run_id  # type: ignore[assignment]
            return run
    
    def complete_analysis_run(self, run_id: int, stats: Optional[Dict[str, Any]] = None):
        """
        Mark an analysis run as completed and update statistics.
        
        Args:
            run_id: ID of the analysis run
            stats: Optional statistics to update
        """
        with self.transaction() as session:
            run = session.query(AnalysisRun).filter_by(id=run_id).first()
            if run:
                run.completed_at = datetime.now(timezone.utc)
                run.status = 'completed'
                
                if stats:
                    for key, value in stats.items():
                        if hasattr(run, key):
                            setattr(run, key, value)
    
    def fail_analysis_run(self, run_id: int, error_message: str):
        """
        Mark an analysis run as failed.
        
        Args:
            run_id: ID of the analysis run
            error_message: Error description
        """
        with self.transaction() as session:
            run = session.query(AnalysisRun).filter_by(id=run_id).first()
            if run:
                run.completed_at = datetime.now(timezone.utc)
                run.status = 'failed'
                run.error_message = error_message
    
    def get_or_create_subreddit(self, name: str, session: Optional[SASession] = None) -> Subreddit:
        """
        Get or create a subreddit entry.
        
        Args:
            name: Subreddit name
            session: Optional existing session to use
            
        Returns:
            Subreddit object
        """
        if session:
            # Use existing session (within a transaction)
            subreddit = session.query(Subreddit).filter_by(name=name).first()
            if not subreddit:
                subreddit = Subreddit(name=name)
                session.add(subreddit)
                session.flush()
            return subreddit
        else:
            # Create new transaction
            with self.transaction() as new_session:
                subreddit = new_session.query(Subreddit).filter_by(name=name).first()
                if not subreddit:
                    subreddit = Subreddit(name=name)
                    new_session.add(subreddit)
                    new_session.flush()
                # Create a detached copy
                sub_id = subreddit.id
                sub_name = subreddit.name
                new_session.expunge(subreddit)
                subreddit.id = sub_id  # type: ignore[assignment]
                subreddit.name = sub_name  # type: ignore[assignment]
                return subreddit
    
    def bulk_insert_posts(self, posts: List[Dict[str, Any]], run_id: int) -> int:
        """
        Bulk insert posts with transaction management.
        
        Args:
            posts: List of post dictionaries
            run_id: Analysis run ID
            
        Returns:
            Number of posts inserted
        """
        inserted = 0
        
        with self.transaction() as session:
            for post_data in posts:
                try:
                    # Get or create subreddit, passing the session
                    subreddit = self.get_or_create_subreddit(post_data['subreddit'], session)

                    # Check if post already exists
                    existing = session.query(Post).filter_by(id=post_data['id']).first()
                    if existing:
                        logger.debug(f"Post {post_data['id']} already exists, skipping")
                        continue

                    # Create new post
                    post = Post(
                        id=post_data['id'],
                        subreddit_id=subreddit.id,
                        analysis_run_id=run_id,
                        title=post_data['title'],
                        selftext=post_data.get('selftext'),
                        author=post_data.get('author'),
                        url=post_data.get('url'),
                        created_utc=post_data['created_utc'],
                        score=post_data.get('score', 0),
                        num_comments=post_data.get('num_comments', 0)
                    )
                    # Use a savepoint so a single failure doesn't rollback the whole batch
                    with session.begin_nested():
                        session.add(post)
                        session.flush()
                        inserted += 1

                except IntegrityError as e:
                    logger.warning(f"Integrity error inserting post {post_data.get('id')}: {e}")
                    # savepoint rolled back; continue with next
                    continue
        
        logger.info(f"Inserted {inserted} posts")
        return inserted
    
    def bulk_insert_comments(self, comments: List[Dict[str, Any]]) -> int:
        """
        Bulk insert comments with transaction management.
        
        Args:
            comments: List of comment dictionaries
            
        Returns:
            Number of comments inserted
        """
        inserted = 0
        
        with self.transaction() as session:
            for comment_data in comments:
                try:
                    # Check if comment already exists
                    existing = session.query(Comment).filter_by(id=comment_data['id']).first()
                    if existing:
                        logger.debug(f"Comment {comment_data['id']} already exists, skipping")
                        continue

                    # Verify post exists
                    post_exists = session.query(Post).filter_by(id=comment_data['post_id']).first()
                    if not post_exists:
                        logger.warning(f"Post {comment_data['post_id']} not found for comment {comment_data['id']}")
                        continue

                    # Create new comment
                    comment = Comment(
                        id=comment_data['id'],
                        post_id=comment_data['post_id'],
                        parent_id=comment_data.get('parent_id'),
                        author=comment_data.get('author'),
                        body=comment_data['body'],
                        score=comment_data.get('score', 0),
                        created_utc=comment_data['created_utc']
                    )
                    # Savepoint for per-row error isolation
                    with session.begin_nested():
                        session.add(comment)
                        session.flush()
                        inserted += 1

                except IntegrityError as e:
                    logger.warning(f"Integrity error inserting comment {comment_data.get('id')}: {e}")
                    continue
        
        logger.info(f"Inserted {inserted} comments")
        return inserted
    
    def update_sentiment_scores(self, items: List[Dict[str, Any]], item_type: str = 'post'):
        """
        Update sentiment scores for posts or comments.
        
        Args:
            items: List of dictionaries with 'id' and 'sentiment_score'
            item_type: 'post' or 'comment'
        """
        model = Post if item_type == 'post' else Comment
        
        with self.transaction() as session:
            for item_data in items:
                item = session.query(model).filter_by(id=item_data['id']).first()
                if item:
                    item.sentiment_score = item_data['sentiment_score']
                    item.analyzed_at = datetime.now(timezone.utc)
                    
                    if item_type == 'post' and 'sentiment_explanation' in item_data:
                        item.sentiment_explanation = item_data['sentiment_explanation']
        
        logger.info(f"Updated sentiment scores for {len(items)} {item_type}s")

    def update_keyword_associations_from_results(self, items: List[Dict[str, Any]], item_type: str = 'post') -> int:
        """
        Upsert keywords and associate them with posts or comments based on analysis results.

        Args:
            items: List of dicts with 'id' and 'keywords' (list[str]) from analyzer
            item_type: 'post' or 'comment'

        Returns:
            Number of associations created (approximate)
        """
        if not items:
            return 0

        Model = Post if item_type == 'post' else Comment

        created_links = 0
        with self.transaction() as session:
            # Build term set and load existing keywords in bulk
            term_set = set()
            for it in items:
                for kw in (it.get('keywords') or []):
                    if isinstance(kw, str) and kw.strip():
                        term_set.add(kw.strip().lower())

            if not term_set:
                return 0

            existing_keywords = session.query(Keyword).filter(
                Keyword.term.in_(list(term_set))
            ).all()
            kw_by_term = {k.term: k for k in existing_keywords}

            # Upsert missing
            for term in term_set:
                if term not in kw_by_term:
                    kw = Keyword(term=term, category=classify_term(term))
                    session.add(kw)
                    session.flush()
                    kw_by_term[term] = kw

            # Associate keywords to items
            for it in items:
                target = session.query(Model).filter_by(id=it.get('id')).first()
                if not target:
                    continue
                raw_terms = it.get('keywords') or []
                # Deduplicate per item
                seen = set()
                for term in raw_terms:
                    if not isinstance(term, str):
                        continue
                    norm = term.strip().lower()
                    if not norm or norm in seen:
                        continue
                    seen.add(norm)
                    kw = kw_by_term.get(norm)
                    if not kw:
                        continue
                    try:
                        target.keywords.append(kw)
                        session.flush()
                        created_links += 1
                    except IntegrityError:
                        session.rollback()
                        continue

        logger.info(f"Associated keywords for {len(items)} {item_type}s (links: {created_links})")
        return created_links
    
    def get_posts_for_analysis(self, batch_size: int = 20, run_id: Optional[int] = None) -> List[Post]:
        """
        Retrieve unanalyzed posts in batches.
        
        Args:
            batch_size: Number of posts to retrieve
            run_id: Optional analysis run ID to filter by
            
        Returns:
            List of Post objects
        """
        with self.transaction() as session:
            query = session.query(Post).filter(Post.sentiment_score.is_(None))

            if run_id is not None:
                query = query.filter_by(analysis_run_id=run_id)
            
            posts = query.limit(batch_size).all()
            
            # Detach from session to avoid lazy loading issues
            session.expunge_all()
            return posts
    
    def get_comments_for_analysis(self, batch_size: int = 50, run_id: Optional[int] = None) -> List[Comment]:
        """
        Retrieve unanalyzed comments in batches.
        
        Args:
            batch_size: Number of comments to retrieve
            run_id: Optional analysis run ID to filter by associated post
            
        Returns:
            List of Comment objects
        """
        with self.transaction() as session:
            query = session.query(Comment).filter(Comment.sentiment_score.is_(None))
            if run_id is not None:
                # Filter comments whose parent post belongs to the given analysis run
                query = query.join(Post, Comment.post_id == Post.id).filter(
                    Post.analysis_run_id == run_id
                )
            comments = query.limit(batch_size).all()
            
            session.expunge_all()
            return comments

    def get_latest_run(self, status: Optional[str] = None) -> Optional[AnalysisRun]:
        """Return the most recent analysis run, optionally filtered by status.

        Args:
            status: Optional status filter ('running', 'completed', 'failed')

        Returns:
            Latest AnalysisRun or None if none exist
        """
        with self.transaction() as session:
            query = session.query(AnalysisRun)
            if status:
                query = query.filter(AnalysisRun.status == status)
            run = query.order_by(AnalysisRun.started_at.desc()).first()
            if run:
                # Detach before returning
                session.expunge(run)
            return run

    def get_run_by_id(self, run_id: int) -> Optional[AnalysisRun]:
        """Return a single analysis run by ID, detached from session.

        Args:
            run_id: The analysis run ID

        Returns:
            AnalysisRun or None if not found
        """
        with self.transaction() as session:
            run = session.query(AnalysisRun).filter_by(id=run_id).first()
            if run:
                session.expunge(run)
            return run
    
    def generate_daily_summary(self, run_id: int) -> List[DailySummary]:
        """
        Generate and store daily summary statistics.
        
        Args:
            run_id: Analysis run ID
            
        Returns:
            List of created DailySummary objects
        """
        summaries = []
        
        with self.transaction() as session:
            # Get the run date
            run = session.query(AnalysisRun).filter_by(id=run_id).first()
            if not run:
                logger.error(f"Analysis run {run_id} not found")
                return summaries
            
            # Get all subreddits with posts from this run
            subreddits = session.query(Subreddit).join(Post).filter(
                Post.analysis_run_id == run_id
            ).distinct().all()
            
            for subreddit in subreddits:
                # Get posts for this subreddit and run
                posts = session.query(Post).filter(
                    and_(
                        Post.subreddit_id == subreddit.id,
                        Post.analysis_run_id == run_id
                    )
                ).all()
                
                if not posts:
                    continue
                
                # Calculate statistics
                total_posts = len(posts)
                total_comments = sum(len(p.comments) for p in posts)
                
                # Sentiment statistics for posts
                post_sentiments = [p.sentiment_score for p in posts if p.sentiment_score is not None]
                avg_post_sentiment = sum(post_sentiments) / len(post_sentiments) if post_sentiments else None

                # Average comment sentiment across comments for these posts
                post_ids = [p.id for p in posts]
                if post_ids:
                    avg_comment_sentiment = session.query(func.avg(Comment.sentiment_score)).filter(
                        Comment.post_id.in_(post_ids),
                        Comment.sentiment_score.isnot(None)
                    ).scalar()
                else:
                    avg_comment_sentiment = None
                
                positive_posts = sum(1 for s in post_sentiments if s > 0.3)
                negative_posts = sum(1 for s in post_sentiments if s < -0.3)
                neutral_posts = sum(1 for s in post_sentiments if -0.3 <= s <= 0.3)
                
                # Find top posts
                analyzed_posts = [p for p in posts if p.sentiment_score is not None]
                if analyzed_posts:
                    top_positive = max(analyzed_posts, key=lambda p: p.sentiment_score or 0)
                    top_negative = min(analyzed_posts, key=lambda p: p.sentiment_score or 0)
                    most_discussed = max(posts, key=lambda p: p.num_comments)
                else:
                    top_positive = None
                    top_negative = None
                    most_discussed = None

                # Keyword mentions across posts and their comments
                keyword_counts: Dict[str, int] = {}
                if post_ids:
                    # Post keyword counts
                    post_kw_rows = session.query(
                        Keyword.term,
                        func.count().label('cnt')
                    ).join(
                        post_keywords, Keyword.id == post_keywords.c.keyword_id
                    ).filter(
                        post_keywords.c.post_id.in_(post_ids)
                    ).group_by(Keyword.term).all()
                    for term, cnt in post_kw_rows:
                        keyword_counts[term] = keyword_counts.get(term, 0) + int(cnt or 0)

                    # Comment keyword counts for comments belonging to these posts
                    comment_kw_rows = session.query(
                        Keyword.term,
                        func.count().label('cnt')
                    ).join(
                        comment_keywords, Keyword.id == comment_keywords.c.keyword_id
                    ).join(
                        Comment, comment_keywords.c.comment_id == Comment.id
                    ).filter(
                        Comment.post_id.in_(post_ids)
                    ).group_by(Keyword.term).all()
                    for term, cnt in comment_kw_rows:
                        keyword_counts[term] = keyword_counts.get(term, 0) + int(cnt or 0)
                keyword_mentions_json = json.dumps(keyword_counts) if keyword_counts else None
                
                # Upsert summary (unique on date + subreddit)
                existing = session.query(DailySummary).filter(
                    and_(
                        DailySummary.date == run.run_date,
                        DailySummary.subreddit_id == subreddit.id,
                    )
                ).first()

                if existing:
                    # Update existing summary in-place
                    existing.analysis_run_id = run_id
                    existing.total_posts = total_posts
                    existing.total_comments = total_comments
                    existing.avg_post_sentiment = avg_post_sentiment
                    existing.avg_comment_sentiment = avg_comment_sentiment
                    existing.positive_posts = positive_posts
                    existing.negative_posts = negative_posts
                    existing.neutral_posts = neutral_posts
                    existing.top_positive_post_id = top_positive.id if top_positive else None
                    existing.top_negative_post_id = top_negative.id if top_negative else None
                    existing.most_discussed_post_id = most_discussed.id if most_discussed else None
                    existing.keyword_mentions = keyword_mentions_json
                    session.flush()
                    target = existing
                else:
                    # Create new summary
                    summary = DailySummary(
                        analysis_run_id=run_id,
                        subreddit_id=subreddit.id,
                        date=run.run_date,
                        total_posts=total_posts,
                        total_comments=total_comments,
                        avg_post_sentiment=avg_post_sentiment,
                        avg_comment_sentiment=avg_comment_sentiment,
                        positive_posts=positive_posts,
                        negative_posts=negative_posts,
                        neutral_posts=neutral_posts,
                        top_positive_post_id=top_positive.id if top_positive else None,
                        top_negative_post_id=top_negative.id if top_negative else None,
                        most_discussed_post_id=most_discussed.id if most_discussed else None,
                        keyword_mentions=keyword_mentions_json
                    )
                    session.add(summary)
                    session.flush()
                    target = summary

                # Create detached copy with all attributes
                summary_dict = {
                    'id': target.id,
                    'analysis_run_id': target.analysis_run_id,
                    'subreddit_id': target.subreddit_id,
                    'date': target.date,
                    'total_posts': target.total_posts,
                    'total_comments': target.total_comments,
                    'avg_post_sentiment': target.avg_post_sentiment,
                    'avg_comment_sentiment': target.avg_comment_sentiment,
                    'positive_posts': target.positive_posts,
                    'negative_posts': target.negative_posts,
                    'neutral_posts': target.neutral_posts,
                    'top_positive_post_id': target.top_positive_post_id,
                    'top_negative_post_id': target.top_negative_post_id,
                    'most_discussed_post_id': target.most_discussed_post_id,
                    'keyword_mentions': target.keyword_mentions
                }
                
                # Create new detached instance
                detached_summary = DailySummary(**summary_dict)
                summaries.append(detached_summary)
            
            logger.info(f"Generated {len(summaries)} daily summaries")
        
        return summaries
    
    def cleanup_old_data(self, days_to_keep: int = 90):
        """
        Remove data older than specified days.
        
        Args:
            days_to_keep: Number of days of data to keep
        """
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_to_keep)
        cutoff_timestamp = int(cutoff_date.timestamp())
        
        with self.transaction() as session:
            # Delete old posts (cascades to comments)
            old_posts = session.query(Post).filter(
                Post.created_utc < cutoff_timestamp
            ).count()
            
            if old_posts > 0:
                session.query(Post).filter(
                    Post.created_utc < cutoff_timestamp
                ).delete(synchronize_session=False)
                
                logger.info(f"Deleted {old_posts} posts older than {days_to_keep} days")
            
            # Delete old analysis runs
            old_runs = session.query(AnalysisRun).filter(
                AnalysisRun.run_date < cutoff_date.date()
            ).count()
            
            if old_runs > 0:
                session.query(AnalysisRun).filter(
                    AnalysisRun.run_date < cutoff_date.date()
                ).delete(synchronize_session=False)
                
                logger.info(f"Deleted {old_runs} analysis runs older than {days_to_keep} days")
    
    def get_recent_posts(self, days: int = 7, subreddit: Optional[str] = None) -> List[Post]:
        """
        Get recent posts for analysis or display.
        
        Args:
            days: Number of days to look back
            subreddit: Optional subreddit filter
            
        Returns:
            List of Post objects
        """
        cutoff = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp())
        
        with self.transaction() as session:
            query = session.query(Post).filter(Post.created_utc >= cutoff)
            
            if subreddit:
                sub = session.query(Subreddit).filter_by(name=subreddit).first()
                if sub:
                    query = query.filter_by(subreddit_id=sub.id)
            
            posts = query.order_by(Post.created_utc.desc()).all()
            session.expunge_all()
            return posts
    
    def get_sentiment_trend(self, days: int = 30, subreddit: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get sentiment trend over time.
        
        Args:
            days: Number of days to analyze
            subreddit: Optional subreddit filter
            
        Returns:
            List of dictionaries with date and average sentiment
        """
        cutoff = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp())
        
        with self.transaction() as session:
            # Build base query
            date_col = func.date(func.datetime(Post.created_utc, 'unixepoch')).label('date')
            query = session.query(
                date_col,
                func.avg(Post.sentiment_score).label('avg_sentiment'),
                func.count(Post.id).label('post_count')
            ).filter(
                and_(
                    Post.created_utc >= cutoff,
                    Post.sentiment_score.isnot(None)
                )
            )

            # Add subreddit filter if specified
            if subreddit:
                sub = session.query(Subreddit).filter_by(name=subreddit).first()
                if sub:
                    query = query.filter(Post.subreddit_id == sub.id)

            # Group by date and order
            results = query.group_by(date_col).order_by(date_col).all()

            return [
                {
                    'date': row.date,
                    'avg_sentiment': round(row.avg_sentiment, 3) if row.avg_sentiment is not None else None,
                    'post_count': row.post_count
                }
                for row in results
            ]
    
    def close(self):
        """Close database connections and clean up resources."""
        self.Session.remove()
        self.engine.dispose()
        logger.info("Database connections closed")

    def reset_database(self) -> None:
        """Drop and recreate all tables, clearing all data.

        Safe way to start with a clean schema without manually deleting files.
        """
        try:
            logger.warning("Resetting database: dropping all tables and recreating schema")
            # Ensure sessions are cleared before DDL
            self.Session.remove()
            Base.metadata.drop_all(self.engine)
            Base.metadata.create_all(self.engine)
            logger.info("Database schema recreated successfully")
        except Exception as e:
            logger.error(f"Failed to reset database: {e}")
            raise
