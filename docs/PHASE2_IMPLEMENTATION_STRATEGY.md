# Phase 2: Core Development - Comprehensive Implementation Strategy

## Overview
Phase 2 focuses on building the core components of the Reddit sentiment analysis bot. This strategy provides detailed implementation specifications, error handling patterns, testing requirements, and dependency management for each module.

## Implementation Timeline
- **Week 1**: Secrets Manager & Database Module (Foundation)
- **Week 2**: Reddit Client & Initial Data Collection
- **Week 3**: Sentiment Analyzer & Claude API Integration
- **Week 4**: Main Orchestration & Testing/Optimization

## Module Implementation Details

### 1. Secrets Manager (`src/secrets_manager.py`)

#### Purpose
Secure credential management using macOS Keychain with fallback to environment variables.

#### Core Functions

```python
def load_secrets() -> Dict[str, str]:
    """
    Load all required secrets into environment variables.
    Priority: macOS Keychain > Environment Variables
    Returns dict of loaded secrets for validation.
    """

def get_secret(service_name: str, account: str = None) -> Optional[str]:
    """
    Retrieve a single secret from Keychain.
    Falls back to environment variable if Keychain unavailable.
    """

def validate_secrets() -> Tuple[bool, List[str]]:
    """
    Validate all required secrets are present.
    Returns (is_valid, list_of_missing_secrets).
    """

def store_secret(service_name: str, value: str, account: str = None) -> bool:
    """
    Store/update a secret in macOS Keychain.
    Used for initial setup or credential rotation.
    """
```

#### Required Secrets
- `REDDIT_CLIENT_ID`
- `REDDIT_CLIENT_SECRET`
- `REDDIT_USER_AGENT`
- `ANTHROPIC_API_KEY`
- `DATABASE_PATH` (optional, defaults to `./data/sentiment.db`)

#### Error Handling
- Missing secrets: Raise `SecretsNotFoundError` with list of missing keys
- Keychain access failure: Fall back to environment variables with warning log
- Invalid secret format: Validate and raise `InvalidSecretFormatError`

#### Testing Strategy
- Mock subprocess calls to `security` command
- Test fallback to environment variables
- Test validation with missing/invalid secrets
- Test cross-platform compatibility

### 2. Database Module (`src/database.py`)

#### Purpose
SQLAlchemy-based database operations with connection pooling and transaction management.

#### SQLAlchemy Models

```python
class Subreddit(Base):
    __tablename__ = 'subreddits'
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    posts = relationship("Post", back_populates="subreddit")

class AnalysisRun(Base):
    __tablename__ = 'analysis_runs'
    id = Column(Integer, primary_key=True, autoincrement=True)
    run_date = Column(Date, nullable=False)
    started_at = Column(DateTime, nullable=False)
    completed_at = Column(DateTime)
    total_posts = Column(Integer, default=0)
    total_comments = Column(Integer, default=0)
    api_calls = Column(Integer, default=0)
    api_cost_estimate = Column(Float, default=0)
    status = Column(String, default='running')
    error_message = Column(Text)

class Post(Base):
    __tablename__ = 'posts'
    id = Column(String, primary_key=True)  # Reddit post ID
    subreddit_id = Column(Integer, ForeignKey('subreddits.id'))
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
    
    subreddit = relationship("Subreddit", back_populates="posts")
    comments = relationship("Comment", back_populates="post")
    keywords = relationship("Keyword", secondary="post_keywords")

class Comment(Base):
    __tablename__ = 'comments'
    id = Column(String, primary_key=True)  # Reddit comment ID
    post_id = Column(String, ForeignKey('posts.id'))
    parent_id = Column(String)
    author = Column(String)
    body = Column(Text, nullable=False)
    score = Column(Integer, default=0)
    created_utc = Column(Integer, nullable=False)
    sentiment_score = Column(Float)
    analyzed_at = Column(DateTime)
    
    post = relationship("Post", back_populates="comments")
    keywords = relationship("Keyword", secondary="comment_keywords")
```

#### Database Manager Class

```python
class DatabaseManager:
    def __init__(self, db_path: str):
        """Initialize connection pool and session factory."""
        
    def create_analysis_run(self) -> AnalysisRun:
        """Create new analysis run entry."""
        
    def bulk_insert_posts(self, posts: List[Dict]) -> int:
        """Bulk insert posts with transaction management."""
        
    def bulk_insert_comments(self, comments: List[Dict]) -> int:
        """Bulk insert comments with transaction management."""
        
    def update_sentiment_scores(self, items: List[Dict], item_type: str):
        """Update sentiment scores for posts or comments."""
        
    def generate_daily_summary(self, run_id: int) -> DailySummary:
        """Generate and store daily summary statistics."""
        
    def get_posts_for_analysis(self, batch_size: int = 20) -> List[Post]:
        """Retrieve unanalyzed posts in batches."""
        
    def cleanup_old_data(self, days_to_keep: int = 90):
        """Remove data older than specified days."""
```

#### Connection Management
- Connection pool size: 5 connections
- Pool overflow: 10 additional connections
- Connection timeout: 30 seconds
- Automatic reconnection on connection loss

#### Transaction Patterns
```python
@contextmanager
def transaction(self):
    """Context manager for database transactions."""
    session = self.Session()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        raise
    finally:
        session.close()
```

#### Error Handling
- Database locked: Retry with exponential backoff (max 3 attempts)
- Integrity constraint violation: Log and skip duplicate entries
- Connection pool exhausted: Queue requests or raise `DatabaseOverloadError`

#### Testing Strategy
- Use in-memory SQLite database (`:memory:`)
- Test all CRUD operations
- Test transaction rollback scenarios
- Test bulk operations performance
- Test connection pool behavior

### 3. Reddit Client (`src/reddit_client.py`)

#### Purpose
PRAW-based Reddit data collection with comprehensive error handling and rate limit management.

#### Core Classes

```python
class RedditClient:
    def __init__(self, credentials: Dict[str, str]):
        """Initialize PRAW with credentials."""
        
    def fetch_subreddit_posts(self, subreddit_name: str, 
                             time_filter: str = 'day') -> List[Dict]:
        """
        Fetch ALL posts from subreddit within time period.
        Returns list of post dictionaries with metadata.
        """
        
    def fetch_post_comments(self, post_id: str, 
                           limit: Optional[int] = None) -> List[Dict]:
        """
        Fetch ALL comments for a post (handles MoreComments).
        Returns flattened list of comment dictionaries.
        """
        
    def fetch_all_daily_data(self, subreddits: List[str]) -> Dict:
        """
        Orchestrate fetching all posts and comments.
        Returns structured data ready for database insertion.
        """

class RateLimitHandler:
    def __init__(self, requests_per_minute: int = 60):
        """Initialize rate limit tracking."""
        
    def wait_if_needed(self):
        """Sleep if approaching rate limit."""
        
    def record_request(self):
        """Track API request for rate limiting."""
```

#### Data Extraction Format
```python
# Post data structure
{
    'id': 'abc123',
    'subreddit': 'ClaudeAI',
    'title': 'Post title',
    'selftext': 'Post body text',
    'author': 'username',
    'url': 'https://reddit.com/...',
    'created_utc': 1234567890,
    'score': 42,
    'num_comments': 15,
    'fetched_at': datetime.utcnow()
}

# Comment data structure
{
    'id': 'def456',
    'post_id': 'abc123',
    'parent_id': 'abc123',  # or another comment ID
    'author': 'username',
    'body': 'Comment text',
    'score': 10,
    'created_utc': 1234567891,
    'fetched_at': datetime.utcnow()
}
```

#### Error Handling
- 403 Forbidden: Check credentials, retry with backoff
- 429 Too Many Requests: Implement exponential backoff
- 503 Service Unavailable: Retry with longer delays
- Invalid subreddit: Log warning and continue with others
- MoreComments expansion timeout: Log and return partial results

#### Performance Optimizations
- Parallel subreddit fetching using ThreadPoolExecutor
- Batch comment fetching for posts with >100 comments
- Cache PRAW instance for connection reuse
- Use `.list()` for efficient comment forest traversal

#### Testing Strategy
- Mock PRAW Reddit instance
- Mock submission and comment objects
- Test rate limit handling
- Test error recovery scenarios
- Test data extraction completeness

### 4. Sentiment Analyzer (`src/sentiment_analyzer.py`)

#### Purpose
Claude API integration for batch sentiment analysis with cost optimization.

#### Core Classes

```python
class SentimentAnalyzer:
    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022"):
        """Initialize Anthropic client with configuration."""
        
    def analyze_batch(self, items: List[Dict], item_type: str = 'post') -> List[Dict]:
        """
        Analyze sentiment for batch of posts/comments.
        Returns list of items with sentiment scores.
        """
        
    def prepare_batch_prompt(self, items: List[Dict], item_type: str) -> str:
        """
        Create optimized prompt for batch analysis.
        Includes context about Claude/AI discussion.
        """
        
    def parse_sentiment_response(self, response: str) -> List[Dict]:
        """
        Parse Claude's JSON response into sentiment scores.
        Handles malformed responses gracefully.
        """
        
    def calculate_api_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Calculate estimated API cost for the request.
        Claude 3.5 Sonnet: $3/1M input, $15/1M output.
        """

class CostManager:
    def __init__(self, daily_limit: float = 5.0):
        """Initialize cost tracking with daily limit."""
        
    def can_proceed(self, estimated_cost: float) -> bool:
        """Check if request would exceed daily limit."""
        
    def record_usage(self, actual_cost: float):
        """Record actual API usage for tracking."""
```

#### Prompt Engineering

```python
BATCH_ANALYSIS_PROMPT = """
Analyze the sentiment of these Reddit posts/comments about Claude and Anthropic.
Consider the technical nature of AI/LLM discussions where criticism can be constructive.

For each item, provide:
1. sentiment_score: float from -1.0 (very negative) to 1.0 (very positive)
2. explanation: brief reason for the score (max 50 words)
3. keywords: list of notable terms mentioned

Items to analyze:
{items_json}

Return a JSON array with analysis for each item in order.
"""
```

#### Batch Processing Strategy
- Optimal batch size: 15-20 posts per request
- Group by content length for consistent token usage
- Separate processing for posts vs comments
- Priority queue for high-engagement content

#### Response Validation
```python
def validate_sentiment_score(score: float) -> float:
    """Ensure score is within valid range [-1.0, 1.0]."""
    return max(-1.0, min(1.0, score))

def handle_parse_error(item: Dict, error: Exception) -> Dict:
    """Fallback for unparseable responses."""
    return {
        'sentiment_score': 0.0,  # Neutral fallback
        'explanation': f'Parse error: {str(error)}',
        'requires_reanalysis': True
    }
```

#### Cost Optimization
- Cache similar content patterns
- Skip very short comments (<10 words)
- Prioritize posts over comments for budget constraints
- Daily cost circuit breaker at $5.00

#### Error Handling
- API timeout: Retry with smaller batch size
- Rate limit: Implement exponential backoff
- Invalid API key: Fail fast with clear error
- Malformed response: Log and mark for reanalysis
- Cost limit exceeded: Gracefully stop and report

#### Testing Strategy
- Mock Anthropic client responses
- Test batch size optimization
- Test prompt generation variations
- Test response parsing edge cases
- Test cost calculation accuracy

### 5. Main Orchestration (`main.py`)

#### Purpose
Coordinate all components for complete analysis workflow with robust error recovery.

#### Workflow Stages

```python
class RedditSentimentBot:
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize all components with configuration."""
        
    def run_daily_analysis(self) -> AnalysisRun:
        """
        Execute complete daily analysis workflow.
        Returns completed AnalysisRun object.
        """
        
    def stage_1_initialize(self) -> AnalysisRun:
        """
        - Load secrets via secrets_manager
        - Validate all credentials
        - Initialize database connection
        - Create new analysis run
        """
        
    def stage_2_collect_data(self, run: AnalysisRun) -> Dict:
        """
        - Fetch posts from all configured subreddits
        - Fetch comments for all posts
        - Store raw data in database
        - Update run statistics
        """
        
    def stage_3_analyze_sentiment(self, run: AnalysisRun) -> Dict:
        """
        - Batch posts for analysis
        - Send to Claude API
        - Store sentiment scores
        - Track API costs
        """
        
    def stage_4_process_keywords(self, run: AnalysisRun):
        """
        - Extract keywords from content
        - Update keyword associations
        - Calculate keyword sentiment trends
        """
        
    def stage_5_generate_summaries(self, run: AnalysisRun):
        """
        - Generate daily summaries per subreddit
        - Calculate aggregate statistics
        - Identify top positive/negative posts
        - Store summary data
        """
        
    def stage_6_cleanup(self, run: AnalysisRun):
        """
        - Mark run as completed
        - Clean old data if configured
        - Backup database
        - Send notifications if configured
        """
```

#### Error Recovery Strategy

```python
class ErrorRecovery:
    def __init__(self, max_retries: int = 3):
        """Initialize recovery configuration."""
        
    @retry(max_attempts=3, backoff='exponential')
    def with_retry(self, func, *args, **kwargs):
        """Execute function with automatic retry."""
        
    def handle_stage_failure(self, stage: str, error: Exception, run: AnalysisRun):
        """
        Determine recovery action based on failure type.
        - Transient errors: Retry
        - Data errors: Skip and continue
        - Critical errors: Fail run
        """
        
    def resume_from_checkpoint(self, run_id: int):
        """Resume incomplete analysis from last successful stage."""
```

#### Logging Configuration

```python
import logging
from logging.handlers import RotatingFileHandler

def setup_logging(config: Dict):
    """Configure comprehensive logging."""
    
    # Main application logger
    app_logger = logging.getLogger('reddit_sentiment')
    app_logger.setLevel(config.get('level', 'INFO'))
    
    # File handler with rotation
    file_handler = RotatingFileHandler(
        config.get('file', 'bot.log'),
        maxBytes=config.get('max_size_mb', 10) * 1024 * 1024,
        backupCount=config.get('backup_count', 5)
    )
    
    # Console handler for immediate feedback
    console_handler = logging.StreamHandler()
    
    # Structured logging format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    app_logger.addHandler(file_handler)
    app_logger.addHandler(console_handler)
    
    return app_logger
```

#### Performance Monitoring

```python
class PerformanceMonitor:
    def __init__(self):
        """Initialize performance tracking."""
        
    def track_stage_duration(self, stage: str):
        """Decorator to track stage execution time."""
        
    def track_memory_usage(self):
        """Monitor memory consumption during execution."""
        
    def generate_performance_report(self, run: AnalysisRun) -> Dict:
        """
        Generate performance metrics:
        - Stage durations
        - API call counts
        - Database query times
        - Memory peak usage
        - Cost breakdown
        """
```

#### Testing Strategy
- Integration tests for complete workflow
- Test each stage in isolation
- Test error recovery scenarios
- Test configuration loading
- Test performance under load

## Testing Framework

### Test Structure
```
tests/
├── unit/
│   ├── test_secrets_manager.py
│   ├── test_database.py
│   ├── test_reddit_client.py
│   ├── test_sentiment_analyzer.py
│   └── test_main.py
├── integration/
│   ├── test_data_pipeline.py
│   ├── test_sentiment_pipeline.py
│   └── test_full_workflow.py
├── fixtures/
│   ├── reddit_responses.json
│   ├── claude_responses.json
│   └── sample_data.sql
└── conftest.py  # Shared pytest fixtures
```

### Testing Requirements
- Minimum 80% code coverage
- All external APIs must be mocked
- Use pytest fixtures for reusable test data
- Test both success and failure paths
- Performance tests for batch operations

### Key Test Scenarios

#### Unit Tests
1. **Secrets Manager**
   - Keychain access success/failure
   - Environment variable fallback
   - Secret validation
   - Cross-platform compatibility

2. **Database**
   - Model relationships
   - Transaction handling
   - Bulk operations
   - Query performance
   - Migration execution

3. **Reddit Client**
   - Post fetching
   - Comment expansion
   - Rate limit handling
   - Error recovery
   - Data extraction

4. **Sentiment Analyzer**
   - Batch processing
   - Prompt generation
   - Response parsing
   - Cost calculation
   - Error handling

5. **Main Orchestration**
   - Workflow execution
   - Stage transitions
   - Error recovery
   - Performance tracking

#### Integration Tests
1. **Data Pipeline**
   - Reddit → Database flow
   - Data integrity
   - Duplicate handling
   - Transaction boundaries

2. **Sentiment Pipeline**
   - Database → Claude → Database flow
   - Batch optimization
   - Cost tracking
   - Score validation

3. **Full Workflow**
   - End-to-end execution
   - Recovery from interruption
   - Performance benchmarks
   - Resource usage

## Dependency Management

### Core Dependencies
```txt
# requirements.txt
praw==7.7.1              # Reddit API wrapper
anthropic==0.26.0        # Claude API client
sqlalchemy==2.0.23       # Database ORM
alembic==1.13.0         # Database migrations
pyyaml==6.0.1           # Configuration files
python-dotenv==1.0.0    # Environment management
tenacity==8.2.3         # Retry logic
click==8.1.7            # CLI interface

# Development dependencies
pytest==7.4.3           # Testing framework
pytest-cov==4.1.0       # Coverage reporting
pytest-mock==3.12.0     # Mocking utilities
black==23.12.0          # Code formatting
pylint==3.0.3           # Code linting
mypy==1.7.1            # Type checking
```

### Version Pinning Strategy
- Pin major and minor versions for production dependencies
- Allow patch updates for security fixes
- Use lockfile for reproducible builds
- Regular dependency updates (monthly)

## Performance Benchmarks

### Target Metrics
- **Data Collection**: <5 minutes for all subreddits
- **Sentiment Analysis**: <10 minutes for 500 posts
- **Database Operations**: <1 second for bulk inserts
- **Total Runtime**: <20 minutes for complete analysis
- **Memory Usage**: <500MB peak
- **API Cost**: <$2.00 per run

### Optimization Strategies
1. **Parallel Processing**
   - Concurrent subreddit fetching
   - Parallel comment expansion
   - Batch database operations

2. **Caching**
   - PRAW instance reuse
   - Database connection pooling
   - Prepared statements

3. **Resource Management**
   - Lazy loading of large datasets
   - Streaming processing for comments
   - Periodic garbage collection

## Security Considerations

### Credential Security
- Never store secrets in code or config files
- Use Keychain/environment variables only
- Rotate API keys regularly
- Audit secret access logs

### Data Security
- Sanitize user-generated content
- Validate all external inputs
- Use parameterized queries
- Regular security updates

### API Security
- Rate limit compliance
- User-agent identification
- Respect robots.txt
- Handle PII appropriately

## Monitoring & Alerting

### Key Metrics to Monitor
- Analysis run completion status
- API costs per run
- Error rates by component
- Performance degradation
- Database growth rate

### Alert Conditions
- Run failure (immediate)
- Cost threshold exceeded (>$5)
- Performance degradation (>30 minutes)
- Database size warning (>1GB)
- API rate limit violations

### Notification Channels
- Log file entries (always)
- Console output (when interactive)
- Email notifications (optional)
- Slack webhook (optional)

## Rollout Strategy

### Phase 2.1: Foundation (Week 1)
1. Implement secrets_manager.py with full test coverage
2. Implement database.py with migrations
3. Validate database schema with test data
4. Performance benchmark database operations

### Phase 2.2: Data Collection (Week 2)
1. Implement reddit_client.py with mocked tests
2. Test with single subreddit (r/ClaudeAI)
3. Validate data completeness
4. Optimize for rate limits

### Phase 2.3: Sentiment Analysis (Week 3)
1. Implement sentiment_analyzer.py with cost tracking
2. Test with sample batch of 50 posts
3. Validate sentiment scoring accuracy
4. Optimize batch sizes for cost

### Phase 2.4: Orchestration (Week 4)
1. Implement main.py with full workflow
2. Run end-to-end tests
3. Performance optimization
4. Documentation and deployment prep

## Success Criteria

### Functional Requirements
- ✅ Fetches ALL posts from configured subreddits
- ✅ Fetches ALL comments for each post
- ✅ Analyzes sentiment for all content
- ✅ Stores data in SQLite database
- ✅ Generates daily summaries
- ✅ Handles errors gracefully

### Non-Functional Requirements
- ✅ Completes analysis in <20 minutes
- ✅ Costs <$2.00 per run
- ✅ 80% test coverage
- ✅ Handles 1000+ posts per day
- ✅ Recovers from transient failures
- ✅ Maintains data integrity

## Risk Mitigation

### Technical Risks
1. **Reddit API Changes**
   - Mitigation: Abstract API calls, version pin PRAW
   
2. **Claude API Rate Limits**
   - Mitigation: Implement backoff, batch optimization
   
3. **Database Corruption**
   - Mitigation: Daily backups, transaction management
   
4. **Cost Overruns**
   - Mitigation: Cost tracking, circuit breakers

### Operational Risks
1. **Credential Exposure**
   - Mitigation: Keychain storage, no hardcoding
   
2. **Data Loss**
   - Mitigation: Transaction boundaries, backups
   
3. **Performance Degradation**
   - Mitigation: Monitoring, optimization passes

## Documentation Requirements

### Code Documentation
- Docstrings for all classes and methods
- Type hints for all function signatures
- Inline comments for complex logic
- Example usage in module docstrings

### User Documentation
- Setup instructions
- Configuration guide
- Troubleshooting guide
- API documentation

### Developer Documentation
- Architecture overview
- Database schema reference
- Testing guide
- Deployment guide

## Conclusion

This comprehensive strategy ensures Phase 2 delivers a robust, scalable, and maintainable Reddit sentiment analysis bot. The modular architecture, extensive testing, and careful error handling provide a solid foundation for Phase 3 enhancements and long-term operation.

Key focus areas:
1. **Security-first** approach to credential management
2. **Cost-optimized** API usage patterns
3. **Robust error recovery** at every stage
4. **Comprehensive testing** for reliability
5. **Performance monitoring** for optimization

With this implementation strategy, the bot will reliably analyze Reddit sentiment daily while maintaining low operational costs and high data quality.