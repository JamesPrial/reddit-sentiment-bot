# Reddit Sentiment Analysis Bot

A bot that analyzes sentiment in Reddit posts and comments about Claude, Anthropic, and AI assistants using the Claude API.

## Overview

This bot:
- Fetches all posts from the last 24 hours from r/ClaudeAI, r/Anthropic, and r/ClaudeCode
- Retrieves all comments for each post
- Analyzes sentiment using Claude 3.5 Sonnet
- Stores results in SQLite database for trend analysis
- Tracks keywords and product mentions
- Manages API costs with configurable daily limits

## Setup

### Prerequisites

- Python 3.8+
- macOS (for Keychain support) or Linux/Windows (uses environment variables)
- Reddit API credentials
- Anthropic API key

### 1. Install Dependencies

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. Configure API Credentials

#### Option A: macOS Keychain (Recommended for Mac users)

```bash
# Add Reddit credentials
security add-generic-password -s "reddit-sentiment-bot" -l "REDDIT_CLIENT_ID" -w "your_client_id"
security add-generic-password -s "reddit-sentiment-bot" -l "REDDIT_CLIENT_SECRET" -w "your_client_secret"
security add-generic-password -s "reddit-sentiment-bot" -l "REDDIT_USER_AGENT" -w "sentiment-bot/1.0 by /u/yourusername"

# Add Anthropic API key
security add-generic-password -s "reddit-sentiment-bot" -l "ANTHROPIC_API_KEY" -w "your_api_key"

# Optional: Set custom database path
security add-generic-password -s "reddit-sentiment-bot" -l "DATABASE_PATH" -w "/path/to/sentiment.db"
```

#### Option B: Environment Variables

```bash
export REDDIT_CLIENT_ID="your_client_id"
export REDDIT_CLIENT_SECRET="your_client_secret"
export REDDIT_USER_AGENT="sentiment-bot/1.0 by /u/yourusername"
export ANTHROPIC_API_KEY="your_api_key"
export DATABASE_PATH="./data/sentiment.db"  # Optional
```

Or create a `.env` file (not recommended for production):
```
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=sentiment-bot/1.0 by /u/yourusername
ANTHROPIC_API_KEY=your_api_key
```

### 3. Getting API Credentials

#### Reddit API
1. Go to https://www.reddit.com/prefs/apps
2. Click "Create App" or "Create Another App"
3. Fill in the form:
   - Name: Your bot name
   - App type: Select "script"
   - Description: Brief description
   - About URL: Can be blank
   - Redirect URI: http://localhost:8080 (not used but required)
4. Click "Create app"
5. Your client ID is the string under "personal use script"
6. Your client secret is the "secret" string

#### Anthropic API
1. Go to https://console.anthropic.com/
2. Sign up or log in
3. Navigate to API Keys section
4. Create a new API key
5. Copy the key immediately (it won't be shown again)

### 4. Initialize Database

```bash
# Run migrations to create database schema
python -m alembic upgrade head

# Or manually create schema
sqlite3 data/sentiment.db < migrations/001_initial_schema.sql
```

## Configuration

Edit `config/config.yaml` to customize:
- Target subreddits
- Claude model and batch size
- Rate limits
- Cost limits
- Database settings

Edit `config/keywords.yaml` to track specific terms:
- Products (Claude versions, etc.)
- Features (Artifacts, Projects, etc.)
- Company terms
- Competitors

## Running the Bot

### Manual Execution

```bash
# Activate virtual environment
source venv/bin/activate

# Run single analysis
python -m src.main

# Or
python src/main.py
```

The bot will:
1. Fetch posts from configured subreddits
2. Fetch all comments for each post
3. Analyze sentiment in batches
4. Store results in database
5. Generate daily summaries
6. Log progress to console and bot.log

### Output

- **Database**: `data/sentiment.db` - SQLite database with all results
- **Logs**: `bot.log` - Detailed execution logs
- **Console**: Real-time progress updates

## Database Schema

Key tables:
- `posts` - Reddit posts with sentiment scores
- `comments` - Reddit comments with sentiment scores
- `keywords` - Tracked terms found in content
- `daily_summaries` - Aggregated statistics
- `analysis_runs` - Execution history with costs

## Cost Management

- Default daily limit: $5.00
- Claude 3.5 Sonnet pricing:
  - Input: $3 per million tokens
  - Output: $15 per million tokens
- Estimated daily cost: ~$1.50-2.00 for 3 subreddits
- Batch processing (15 posts/comments per API call) reduces costs by 10-15x

## Troubleshooting

### Missing Secrets Error
```
SecretsNotFoundError: Missing required secrets: REDDIT_CLIENT_ID, ...
```
Solution: Ensure all credentials are added to Keychain or environment variables

### Reddit API Errors
- 403 Forbidden: Check if subreddit is private/banned
- 429 Rate Limited: Bot automatically handles rate limiting
- 404 Not Found: Verify subreddit name (without r/ prefix)

### Claude API Errors
- Rate limits: Bot includes exponential backoff
- Cost limits: Adjust `daily_limit` in config.yaml
- Invalid API key: Verify key in Keychain/environment

### Database Errors
- Ensure data/ directory exists
- Check write permissions
- Run migrations if schema is missing

## Development

### Running Tests
```bash
pytest tests/
pytest tests/ --cov=src --cov-report=term-missing
```

### Adding New Features
1. Write tests first (TDD approach)
2. Follow existing code patterns
3. Update database schema if needed
4. Run linting: `pylint src/`
5. Type checking: `mypy src/`

## Architecture

- **secrets_manager.py**: Secure credential management
- **database.py**: SQLAlchemy models and operations
- **reddit_client.py**: PRAW-based Reddit API client
- **sentiment_analyzer.py**: Claude API integration
- **taxonomy.py**: Keyword extraction and categorization
- **main.py**: Orchestration and entry point
- **config.py**: Configuration management

## License

[Your License Here]

## Contributing

[Your Contributing Guidelines Here]