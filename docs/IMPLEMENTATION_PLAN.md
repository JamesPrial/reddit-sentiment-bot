# Reddit Sentiment Analysis Bot - Implementation Plan

## Overview
A Reddit sentiment analysis bot that runs every 24 hours to analyze ALL posts and comments from Claude-specific subreddits using Anthropic's Claude API for sentiment scoring, with data stored in SQLite for efficient querying and analysis.

## Target Subreddits
- r/ClaudeAI
- r/Anthropic  
- r/ClaudeCode

## Project Structure
```
reddit-sentiment-bot/
├── src/
│   ├── reddit_client.py      # Reddit API interaction
│   ├── sentiment_analyzer.py # Claude API sentiment analysis
│   ├── database.py           # SQLite database operations
│   └── scheduler.py          # Daily run orchestration
├── data/
│   └── sentiment.db          # SQLite database
├── config/
│   └── config.yaml           # Configuration file
├── migrations/
│   └── 001_initial_schema.sql # Database schema
├── .env                      # API credentials (Reddit & Anthropic)
├── requirements.txt          # Python dependencies
├── main.py                   # Entry point
└── IMPLEMENTATION_PLAN.md    # This file
```

## Core Components

### 1. Reddit API Integration
- **Library**: PRAW (Python Reddit API Wrapper) v7.7.1
- **Authentication**: OAuth2 with client credentials
- **Rate Limits**: 60 requests per minute (handled by PRAW)
- **Data Collection**:
  - ALL posts from last 24 hours per subreddit
  - ALL comments for each post (no limits)
  - Estimated volume: 100-500+ posts per subreddit daily
  - Use `.new()` with time filtering to get complete data

### 2. Sentiment Analysis with Claude API
- **API**: Anthropic Claude API (separate from Pro/Max subscription)
- **Model**: Claude 3.5 Sonnet ($3/million input tokens, $15/million output tokens)
- **Scoring System**: -1.0 (very negative) to 1.0 (very positive)
- **Batch Processing**: Group 10-20 posts per API call to reduce costs
- **Prompt Strategy**:
  ```
  Analyze sentiment for these Reddit posts about Claude/Anthropic.
  Return a JSON with sentiment scores from -1 to 1.
  Consider context of AI/LLM discussion.
  ```

### 3. SQLite Database Storage
- **Database**: SQLite3 (built into Python)
- **ORM**: SQLAlchemy for easier queries and migrations
- **Benefits**:
  - Efficient querying for trends and analytics
  - Better performance for large datasets
  - Built-in data integrity
  - Easy backup (single file)
  - Support for complex queries and aggregations

### 4. Database Schema

```sql
-- Main tables for Reddit sentiment analysis

-- Subreddits being tracked
CREATE TABLE subreddits (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Daily analysis runs
CREATE TABLE analysis_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_date DATE NOT NULL,
    started_at TIMESTAMP NOT NULL,
    completed_at TIMESTAMP,
    total_posts INTEGER DEFAULT 0,
    total_comments INTEGER DEFAULT 0,
    api_calls INTEGER DEFAULT 0,
    api_cost_estimate REAL DEFAULT 0,
    status TEXT DEFAULT 'running', -- running, completed, failed
    error_message TEXT
);

-- Reddit posts
CREATE TABLE posts (
    id TEXT PRIMARY KEY, -- Reddit post ID
    subreddit_id INTEGER NOT NULL,
    analysis_run_id INTEGER NOT NULL,
    title TEXT NOT NULL,
    selftext TEXT,
    author TEXT,
    url TEXT,
    created_utc INTEGER NOT NULL,
    score INTEGER DEFAULT 0,
    num_comments INTEGER DEFAULT 0,
    sentiment_score REAL,
    sentiment_explanation TEXT,
    analyzed_at TIMESTAMP,
    FOREIGN KEY (subreddit_id) REFERENCES subreddits(id),
    FOREIGN KEY (analysis_run_id) REFERENCES analysis_runs(id)
);

-- Reddit comments
CREATE TABLE comments (
    id TEXT PRIMARY KEY, -- Reddit comment ID
    post_id TEXT NOT NULL,
    parent_id TEXT,
    author TEXT,
    body TEXT NOT NULL,
    score INTEGER DEFAULT 0,
    created_utc INTEGER NOT NULL,
    sentiment_score REAL,
    analyzed_at TIMESTAMP,
    FOREIGN KEY (post_id) REFERENCES posts(id)
);

-- Keywords found in posts/comments
CREATE TABLE keywords (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    keyword TEXT UNIQUE NOT NULL,
    category TEXT -- 'product', 'feature', 'company', 'competitor'
);

-- Many-to-many relationship for post keywords
CREATE TABLE post_keywords (
    post_id TEXT NOT NULL,
    keyword_id INTEGER NOT NULL,
    frequency INTEGER DEFAULT 1,
    PRIMARY KEY (post_id, keyword_id),
    FOREIGN KEY (post_id) REFERENCES posts(id),
    FOREIGN KEY (keyword_id) REFERENCES keywords(id)
);

-- Many-to-many relationship for comment keywords
CREATE TABLE comment_keywords (
    comment_id TEXT NOT NULL,
    keyword_id INTEGER NOT NULL,
    frequency INTEGER DEFAULT 1,
    PRIMARY KEY (comment_id, keyword_id),
    FOREIGN KEY (comment_id) REFERENCES comments(id),
    FOREIGN KEY (keyword_id) REFERENCES keywords(id)
);

-- Daily summaries for quick access
CREATE TABLE daily_summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subreddit_id INTEGER NOT NULL,
    analysis_run_id INTEGER NOT NULL,
    summary_date DATE NOT NULL,
    total_posts INTEGER DEFAULT 0,
    total_comments INTEGER DEFAULT 0,
    avg_post_sentiment REAL,
    avg_comment_sentiment REAL,
    most_positive_post_id TEXT,
    most_negative_post_id TEXT,
    top_keywords TEXT, -- JSON array
    UNIQUE(subreddit_id, summary_date),
    FOREIGN KEY (subreddit_id) REFERENCES subreddits(id),
    FOREIGN KEY (analysis_run_id) REFERENCES analysis_runs(id)
);

-- Indexes for performance
CREATE INDEX idx_posts_subreddit ON posts(subreddit_id);
CREATE INDEX idx_posts_created ON posts(created_utc);
CREATE INDEX idx_posts_sentiment ON posts(sentiment_score);
CREATE INDEX idx_comments_post ON comments(post_id);
CREATE INDEX idx_comments_sentiment ON comments(sentiment_score);
CREATE INDEX idx_runs_date ON analysis_runs(run_date);
CREATE INDEX idx_summaries_date ON daily_summaries(summary_date);
```

### 5. Scheduling
- **Primary Method**: Cron job for Unix/Linux/Mac
- **Windows Alternative**: Task Scheduler
- **Schedule**: Daily at midnight local time
- **Cron Entry**: `0 0 * * * cd /path/to/bot && /usr/bin/python3 main.py`

## Implementation Steps

### Phase 1: Environment Setup
1. **Create Reddit App**
   - Go to reddit.com/prefs/apps
   - Create "script" type application
   - Note client_id and client_secret
   
2. **Get Anthropic API Key**
   - Sign up at console.anthropic.com
   - Create API key
   - Use $5 free credit for testing

3. **Install Dependencies**
   ```bash
   pip install praw==7.7.1
   pip install anthropic==0.26.0
   pip install python-dotenv==1.0.0
   pip install pyyaml==6.0.1
   pip install schedule==1.2.0
   pip install sqlalchemy==2.0.23
   pip install alembic==1.13.0  # for migrations
   ```

### Phase 2: Core Development
1. **Database Module** (`database.py`)
   - SQLAlchemy models for all tables
   - Connection pooling
   - Query helpers for common operations
   - Migration support with Alembic

2. **Reddit Client** (`reddit_client.py`)
   - PRAW initialization with credentials
   - Fetch all posts from last 24 hours
   - Retrieve all comments per post
   - Handle rate limiting and retries
   - Store raw data in database

3. **Sentiment Analyzer** (`sentiment_analyzer.py`)
   - Claude API client setup
   - Batch processing logic
   - Custom prompts for Claude context
   - Cost tracking per run
   - Update database with sentiment scores

4. **Main Script** (`main.py`)
   - Create analysis run entry
   - Orchestrate full workflow
   - Error handling and logging
   - Update run status and statistics
   - Generate daily summaries

### Phase 3: Claude-Specific Features
1. **Keyword Tracking**
   - Claude versions (3, 3.5, Opus, Sonnet, Haiku)
   - Features (Artifacts, Projects, Claude Code)
   - Company terms (Anthropic, Constitutional AI)
   - Competitors (GPT, Gemini, Copilot)
   - Store keyword associations in database

2. **Sentiment Context**
   - Adjust scoring for technical discussions
   - Recognize constructive criticism vs complaints
   - Weight based on engagement metrics
   - Store explanation with sentiment scores

### Phase 4: Deployment
1. **Initialize Database**
   ```bash
   # Run migrations to create schema
   alembic init alembic
   alembic revision --autogenerate -m "Initial schema"
   alembic upgrade head
   
   # Or direct SQL execution
   sqlite3 data/sentiment.db < migrations/001_initial_schema.sql
   ```

2. **Configure Environment**
   ```bash
   # .env file
   REDDIT_CLIENT_ID=your_client_id
   REDDIT_CLIENT_SECRET=your_secret
   REDDIT_USER_AGENT=sentiment-bot/1.0
   ANTHROPIC_API_KEY=your_api_key
   DATABASE_PATH=./data/sentiment.db
   ```

3. **Set Up Cron Job**
   ```bash
   crontab -e
   # Add: 0 0 * * * cd /path/to/reddit-sentiment-bot && python3 main.py
   ```

## Configuration File

### config.yaml
```yaml
reddit:
  subreddits:
    - ClaudeAI
    - Anthropic
    - ClaudeCode
  time_filter: "day"  # last 24 hours
  fetch_all: true     # get ALL posts, not just top
  
sentiment:
  api_key_env: "ANTHROPIC_API_KEY"
  model: "claude-3-5-sonnet-20241022"
  batch_size: 20      # posts per API call
  max_tokens: 4000    # per response
  temperature: 0.3    # lower for consistent scoring
  
keywords:
  products:
    - "Claude"
    - "Claude 3"
    - "Claude 3.5"
    - "Opus"
    - "Sonnet"
    - "Haiku"
  features:
    - "Artifacts"
    - "Projects"
    - "Claude Code"
    - "Computer Use"
    - "Vision"
  company:
    - "Anthropic"
    - "Constitutional AI"
    - "RLHF"
  competitors:
    - "ChatGPT"
    - "GPT-4"
    - "Gemini"
    - "Copilot"
    
database:
  path: "./data/sentiment.db"
  backup_enabled: true
  backup_dir: "./backups"
  vacuum_on_startup: true  # optimize database
  
logging:
  level: "INFO"
  file: "bot.log"
  max_size_mb: 10
  backup_count: 5
```

## Cost Estimation
- **Reddit API**: Free (within rate limits)
- **Anthropic API** (Claude 3.5 Sonnet):
  - Input: ~500 posts × 200 tokens avg = 100K tokens
  - Comments: ~5000 comments × 50 tokens = 250K tokens
  - Total input: ~350K tokens = ~$1.05/day
  - Output: ~50K tokens = ~$0.75/day
  - **Estimated daily cost: $1.80**
  - **Monthly cost: ~$54**

## Database Queries Examples

```python
# Get average sentiment by day for a subreddit
SELECT 
    date(created_utc, 'unixepoch') as post_date,
    AVG(sentiment_score) as avg_sentiment,
    COUNT(*) as post_count
FROM posts
WHERE subreddit_id = ?
GROUP BY post_date
ORDER BY post_date DESC
LIMIT 30;

# Find trending keywords over time
SELECT 
    k.keyword,
    COUNT(*) as mention_count,
    AVG(p.sentiment_score) as avg_sentiment
FROM keywords k
JOIN post_keywords pk ON k.id = pk.keyword_id
JOIN posts p ON pk.post_id = p.id
WHERE p.created_utc > ?
GROUP BY k.keyword
ORDER BY mention_count DESC
LIMIT 20;

# Get most controversial posts (high engagement, mixed sentiment)
SELECT 
    p.title,
    p.sentiment_score,
    p.num_comments,
    AVG(c.sentiment_score) as avg_comment_sentiment,
    ABS(p.sentiment_score - AVG(c.sentiment_score)) as controversy_score
FROM posts p
JOIN comments c ON p.id = c.post_id
GROUP BY p.id
HAVING COUNT(c.id) > 10
ORDER BY controversy_score DESC
LIMIT 10;
```

## Metrics & Reporting
- **Real-time Queries**:
  - Current day sentiment trends
  - Keyword frequency changes
  - Engagement vs sentiment correlation
  
- **Historical Analysis**:
  - 7/30/90 day trends
  - Feature announcement impact
  - Competitor mention analysis
  - User satisfaction over time

## Error Handling
- **Database Issues**:
  - Automatic backups before each run
  - Transaction rollback on errors
  - Integrity checks weekly
  
- **Reddit API Issues**:
  - Store partial results in transaction
  - Resume from last successful entry
  - Mark failed posts for retry
  
- **Claude API Issues**:
  - Queue failed items for reprocessing
  - Store raw data even if analysis fails
  - Cost limit circuit breaker

## Backup Strategy
- **Daily Backups**:
  ```bash
  sqlite3 sentiment.db ".backup backup_$(date +%Y%m%d).db"
  ```
- **Weekly Exports**: CSV dumps of key tables
- **Monthly Archives**: Compressed full backups

## Future Enhancements
- Web dashboard with SQL-powered analytics
- Real-time monitoring with database triggers
- Sentiment prediction models based on historical data
- A/B testing for sentiment prompts
- Multi-language support
- Export to data warehouse for advanced analytics