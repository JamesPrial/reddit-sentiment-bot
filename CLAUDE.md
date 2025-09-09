# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Reddit sentiment analysis bot that analyzes posts and comments from Claude/Anthropic-related subreddits (r/ClaudeAI, r/Anthropic, r/ClaudeCode) every 24 hours. Uses Anthropic's Claude API for sentiment scoring and stores data in SQLite for efficient querying and trend analysis.

## Development Commands

### Setup
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  

# Install dependencies
pip install -r requirements.txt

# Initialize database
python -m alembic upgrade head
# Or: sqlite3 data/sentiment.db < migrations/001_initial_schema.sql
```

### Running the Bot
```bash
# Run single analysis
python main.py

# Run tests - REQUIRED for all new code
pytest tests/
pytest tests/ --cov=src --cov-report=term-missing  # With coverage

# Linting and type checking
pylint src/
mypy src/
black src/  # Format code
```

### Testing Requirements
**IMPORTANT: All new implementations MUST have corresponding unit tests**
- Write tests for every new function/class
- Test files mirror source structure: `src/module.py` → `tests/test_module.py`
- Mock external APIs (Reddit, Anthropic) - never make real API calls in tests
- Use in-memory SQLite for database tests
- Maintain minimum 80% code coverage
- Run tests before any commit
- MINIMUM 1 failure condition must be tests for each function/class

## Architecture

refer to docs/IMPLEMENTATION_PLAN.md for more details

### Core Components
1. **reddit_client.py**: PRAW-based Reddit API client
   - Fetches ALL posts from last 24 hours per subreddit
   - Retrieves ALL comments for each post
   - Handles rate limiting automatically

2. **sentiment_analyzer.py**: Claude API integration
   - Batch processes 10-20 posts per API call
   - Uses Claude 3.5 Sonnet model
   - Returns sentiment scores from -1.0 to 1.0

3. **database.py**: SQLAlchemy-based database operations
   - Connection pooling
   - Transaction management
   - Query helpers for common operations

4. **scheduler.py**: Orchestrates daily runs
   - Managed via cron job (Unix/Mac) or Task Scheduler (Windows)

### Data Flow
```
Reddit API → Fetch posts/comments → Store raw data → 
Batch to Claude API → Get sentiment scores → 
Update database → Generate daily summaries
```

## Database Schema

refer to docs/DATABASE_SCHEMA.md for more details

### Key Tables
- **posts**: Reddit posts with sentiment scores
- **comments**: Reddit comments with sentiment analysis
- **keywords**: Tracked terms (Claude versions, features, competitors)
- **post_keywords/comment_keywords**: Many-to-many relationships
- **daily_summaries**: Pre-computed statistics for quick access
- **analysis_runs**: Tracks each bot execution
- **subreddits**: Monitored subreddit list

### Important Indexes
- Time-based queries: `idx_posts_created`
- Sentiment analysis: `idx_posts_sentiment`, `idx_comments_sentiment`
- Subreddit filtering: `idx_posts_subreddit`

### Common Query Patterns
```python
# Get sentiment trend
SELECT date(created_utc, 'unixepoch') as day, 
       AVG(sentiment_score) as avg_sentiment
FROM posts
WHERE created_utc > unixepoch('now', '-30 days')
GROUP BY day;

# Find controversial posts (high engagement, mixed sentiment)
SELECT p.title, p.sentiment_score,
       AVG(c.sentiment_score) as avg_comment_sentiment
FROM posts p
JOIN comments c ON p.id = c.post_id
GROUP BY p.id
HAVING COUNT(c.id) > 10;
```

## Configuration

### Required Environment Variables (.env)
```
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_secret
REDDIT_USER_AGENT=sentiment-bot/1.0
ANTHROPIC_API_KEY=your_api_key
DATABASE_PATH=./data/sentiment.db
```

### Config File (config/config.yaml)
- Subreddit list
- Claude API settings (model, batch size, temperature)
- Keyword categories (products, features, company, competitors)
- Database settings
- Logging configuration

## API Cost Management
- Batch processing: 10-20 posts per Claude API call
- Estimated daily cost: ~$1.80 (350K input + 50K output tokens)
- Cost tracking in `analysis_runs` table
- Circuit breaker for cost limits

## Error Handling Strategy
- Database transactions with automatic rollback
- Partial result storage on API failures
- Failed items queued for retry
- Comprehensive logging to `bot.log`
- Daily backups before each run

## Key Design Decisions
1. **SQLite over flat files**: Enables efficient querying, data integrity, and trend analysis
2. **Batch API calls**: Reduces costs by 10-20x compared to individual calls
3. **Complete data collection**: Fetches ALL posts/comments, not just top/hot
4. **Keyword tracking**: Monitors mentions of Claude versions, features, and competitors
5. **Pre-computed summaries**: Daily aggregations for dashboard performance

## Development Workflow
1. Check existing code patterns before implementing new features
2. Write tests FIRST (TDD approach recommended)
3. Use SQLAlchemy models, don't write raw SQL
4. Mock all external services in tests
5. Run linting and tests before commits
6. Update migration files for any schema changes
7. Document API response formats for mocking