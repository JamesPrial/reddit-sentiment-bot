# Database Schema Documentation

## Overview
SQLite database schema for storing Reddit sentiment analysis data with efficient querying capabilities.

## Entity Relationship Diagram

```
subreddits (1) ─────< (∞) posts
     │                      │
     │                      ├──< comments
     │                      │
     └──< daily_summaries   └──< post_keywords ──> keywords
                                                      │
                                 comment_keywords <───┘

analysis_runs (1) ────< (∞) posts
     │
     └──< daily_summaries
```

## Tables

### 1. `subreddits`
Stores the subreddits being monitored.

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PRIMARY KEY | Unique identifier |
| name | TEXT UNIQUE NOT NULL | Subreddit name (without r/) |
| created_at | TIMESTAMP | When added to monitoring |

### 2. `analysis_runs`
Tracks each daily analysis execution.

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PRIMARY KEY | Unique run identifier |
| run_date | DATE | Date of analysis |
| started_at | TIMESTAMP | Start time |
| completed_at | TIMESTAMP | Completion time |
| total_posts | INTEGER | Number of posts processed |
| total_comments | INTEGER | Number of comments processed |
| api_calls | INTEGER | Claude API calls made |
| api_cost_estimate | REAL | Estimated cost in USD |
| status | TEXT | running/completed/failed |
| error_message | TEXT | Error details if failed |

### 3. `posts`
Reddit posts with sentiment analysis.

| Column | Type | Description |
|--------|------|-------------|
| id | TEXT PRIMARY KEY | Reddit post ID |
| subreddit_id | INTEGER FK | Link to subreddit |
| analysis_run_id | INTEGER FK | Link to analysis run |
| title | TEXT | Post title |
| selftext | TEXT | Post body (if text post) |
| author | TEXT | Reddit username |
| url | TEXT | Post URL |
| created_utc | INTEGER | Unix timestamp |
| score | INTEGER | Reddit score |
| num_comments | INTEGER | Comment count |
| sentiment_score | REAL | -1.0 to 1.0 |
| sentiment_explanation | TEXT | Claude's reasoning |
| analyzed_at | TIMESTAMP | When analyzed |

### 4. `comments`
Reddit comments with sentiment analysis.

| Column | Type | Description |
|--------|------|-------------|
| id | TEXT PRIMARY KEY | Reddit comment ID |
| post_id | TEXT FK | Parent post |
| parent_id | TEXT | Parent comment ID (if nested) |
| author | TEXT | Reddit username |
| body | TEXT | Comment text |
| score | INTEGER | Reddit score |
| created_utc | INTEGER | Unix timestamp |
| sentiment_score | REAL | -1.0 to 1.0 |
| analyzed_at | TIMESTAMP | When analyzed |

### 5. `keywords`
Master list of tracked keywords.

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PRIMARY KEY | Unique identifier |
| term | TEXT UNIQUE | Keyword/phrase |
| category | TEXT | product/feature/company/competitor |
| created_at | TIMESTAMP | When keyword was added |

### 6. `post_keywords`
Many-to-many relationship between posts and keywords.

| Column | Type | Description |
|--------|------|-------------|
| post_id | TEXT FK | Post ID |
| keyword_id | INTEGER FK | Keyword ID |
| frequency | INTEGER | Times mentioned |

### 7. `comment_keywords`
Many-to-many relationship between comments and keywords.

| Column | Type | Description |
|--------|------|-------------|
| comment_id | TEXT FK | Comment ID |
| keyword_id | INTEGER FK | Keyword ID |
| frequency | INTEGER | Times mentioned |

### 8. `daily_summaries`
Pre-computed daily statistics for quick access.

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PRIMARY KEY | Unique identifier |
| subreddit_id | INTEGER FK | Subreddit |
| analysis_run_id | INTEGER FK | Analysis run |
| date | DATE | Date summarized |
| total_posts | INTEGER | Post count |
| total_comments | INTEGER | Comment count |
| avg_post_sentiment | REAL | Average post sentiment |
| avg_comment_sentiment | REAL | Average comment sentiment |
| top_positive_post_id | TEXT | Most positive post |
| top_negative_post_id | TEXT | Most negative post |
| most_discussed_post_id | TEXT | Post with most comments |
| keyword_mentions | TEXT | JSON object of keyword counts |

## Indexes

Performance-optimized indexes:

- `idx_posts_subreddit` - Quick subreddit filtering
- `idx_posts_created` - Time-based queries
- `idx_posts_sentiment` - Sentiment analysis queries
- `idx_comments_post` - Comment retrieval by post
- `idx_comments_sentiment` - Comment sentiment queries
- `idx_runs_date` - Run history queries
- `idx_summary_date` - Summary timeline queries

## Sample Queries

### Get sentiment trend for last 30 days
```sql
SELECT 
    date(p.created_utc, 'unixepoch') as day,
    s.name as subreddit,
    AVG(p.sentiment_score) as avg_sentiment,
    COUNT(p.id) as post_count
FROM posts p
JOIN subreddits s ON p.subreddit_id = s.id
WHERE p.created_utc > unixepoch('now', '-30 days')
GROUP BY day, s.name
ORDER BY day DESC, s.name;
```

### Find most mentioned features this week
```sql
SELECT 
    k.keyword,
    k.category,
    COUNT(DISTINCT pk.post_id) as post_mentions,
    COUNT(DISTINCT ck.comment_id) as comment_mentions,
    AVG(p.sentiment_score) as avg_post_sentiment
FROM keywords k
LEFT JOIN post_keywords pk ON k.id = pk.keyword_id
LEFT JOIN posts p ON pk.post_id = p.id
LEFT JOIN comment_keywords ck ON k.id = ck.keyword_id
WHERE k.category = 'feature'
    AND p.created_utc > unixepoch('now', '-7 days')
GROUP BY k.keyword
ORDER BY (post_mentions + comment_mentions) DESC
LIMIT 10;
```

### Get posts with biggest sentiment difference from comments
```sql
SELECT 
    p.title,
    p.sentiment_score as post_sentiment,
    AVG(c.sentiment_score) as avg_comment_sentiment,
    COUNT(c.id) as num_comments,
    p.sentiment_score - AVG(c.sentiment_score) as sentiment_gap
FROM posts p
JOIN comments c ON p.id = c.post_id
WHERE p.created_utc > unixepoch('now', '-1 day')
GROUP BY p.id
HAVING num_comments >= 5
ORDER BY ABS(sentiment_gap) DESC
LIMIT 20;
```

### Daily summary for dashboard
```sql
SELECT 
    s.name as subreddit,
    ds.summary_date,
    ds.total_posts,
    ds.total_comments,
    ds.avg_post_sentiment,
    ds.avg_comment_sentiment,
    p1.title as most_positive_title,
    p2.title as most_negative_title,
    ds.top_keywords
FROM daily_summaries ds
JOIN subreddits s ON ds.subreddit_id = s.id
LEFT JOIN posts p1 ON ds.top_positive_post_id = p1.id
LEFT JOIN posts p2 ON ds.top_negative_post_id = p2.id
WHERE ds.date >= date('now', '-7 days')
ORDER BY ds.date DESC, s.name;
```

## Data Retention

- **Posts & Comments**: Keep indefinitely (low storage requirement)
- **Analysis Runs**: Keep for 1 year, then archive
- **Daily Summaries**: Keep indefinitely
- **Keywords**: Permanent reference table

## Backup Procedures

### Daily Backup Script
```bash
#!/bin/bash
DB_PATH="/path/to/sentiment.db"
BACKUP_DIR="/path/to/backups"
DATE=$(date +%Y%m%d)

# Create backup
sqlite3 $DB_PATH ".backup $BACKUP_DIR/sentiment_$DATE.db"

# Compress older backups (>7 days)
find $BACKUP_DIR -name "*.db" -mtime +7 -exec gzip {} \;

# Delete very old backups (>90 days)
find $BACKUP_DIR -name "*.gz" -mtime +90 -delete
```

### Export to CSV
```python
import sqlite3
import pandas as pd
from datetime import datetime

conn = sqlite3.connect('sentiment.db')

# Export posts for analysis
query = """
SELECT p.*, s.name as subreddit_name
FROM posts p
JOIN subreddits s ON p.subreddit_id = s.id
WHERE date(p.created_utc, 'unixepoch') = date('now', '-1 day')
"""

df = pd.read_sql_query(query, conn)
df.to_csv(f'posts_export_{datetime.now():%Y%m%d}.csv', index=False)
```

## Performance Considerations

1. **Vacuum regularly**: `VACUUM;` to reclaim space and optimize
2. **Analyze after bulk inserts**: `ANALYZE;` to update query planner
3. **Monitor size**: Database should stay under 1GB for first year
4. **Consider partitioning**: If >1M posts, consider monthly tables

## Migration Strategy

When schema changes are needed:

1. Use Alembic for version control
2. Test migrations on backup first
3. Run during off-hours
4. Keep rollback script ready
5. Document all changes in CHANGELOG.md
