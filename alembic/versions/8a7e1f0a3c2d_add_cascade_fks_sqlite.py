"""Add ON DELETE CASCADE to relevant foreign keys (SQLite)

Revision ID: 8a7e1f0a3c2d
Revises: 5d81289aef04
Create Date: 2025-09-10 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '8a7e1f0a3c2d'
down_revision: Union[str, Sequence[str], None] = '5d81289aef04'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    conn = op.get_bind()
    if conn.dialect.name == 'sqlite':
        # Recreate tables to add ON DELETE CASCADE constraints
        op.execute("PRAGMA foreign_keys=OFF")

        # comments table with ON DELETE CASCADE on post_id
        op.execute(
            """
            CREATE TABLE comments_new (
                id TEXT PRIMARY KEY,
                post_id TEXT NOT NULL,
                parent_id TEXT,
                author TEXT,
                body TEXT NOT NULL,
                score INTEGER,
                created_utc INTEGER NOT NULL,
                sentiment_score FLOAT,
                analyzed_at DATETIME,
                FOREIGN KEY(post_id) REFERENCES posts(id) ON DELETE CASCADE
            );
            """
        )
        op.execute(
            """
            INSERT INTO comments_new (id, post_id, parent_id, author, body, score, created_utc, sentiment_score, analyzed_at)
            SELECT id, post_id, parent_id, author, body, score, created_utc, sentiment_score, analyzed_at FROM comments;
            """
        )
        op.execute("DROP INDEX IF EXISTS idx_comments_sentiment")
        op.execute("DROP INDEX IF EXISTS idx_comments_post")
        op.execute("DROP INDEX IF EXISTS idx_comments_created")
        op.execute("DROP TABLE comments")
        op.execute("ALTER TABLE comments_new RENAME TO comments")
        op.execute("CREATE INDEX idx_comments_post ON comments(post_id)")
        op.execute("CREATE INDEX idx_comments_sentiment ON comments(sentiment_score)")
        op.execute("CREATE INDEX idx_comments_created ON comments(created_utc)")

        # post_keywords with CASCADE on both FKs
        op.execute(
            """
            CREATE TABLE post_keywords_new (
                post_id TEXT NOT NULL,
                keyword_id INTEGER NOT NULL,
                PRIMARY KEY (post_id, keyword_id),
                FOREIGN KEY(post_id) REFERENCES posts(id) ON DELETE CASCADE,
                FOREIGN KEY(keyword_id) REFERENCES keywords(id) ON DELETE CASCADE
            );
            """
        )
        op.execute(
            """
            INSERT INTO post_keywords_new (post_id, keyword_id)
            SELECT post_id, keyword_id FROM post_keywords;
            """
        )
        op.execute("DROP TABLE post_keywords")
        op.execute("ALTER TABLE post_keywords_new RENAME TO post_keywords")

        # comment_keywords with CASCADE on both FKs
        op.execute(
            """
            CREATE TABLE comment_keywords_new (
                comment_id TEXT NOT NULL,
                keyword_id INTEGER NOT NULL,
                PRIMARY KEY (comment_id, keyword_id),
                FOREIGN KEY(comment_id) REFERENCES comments(id) ON DELETE CASCADE,
                FOREIGN KEY(keyword_id) REFERENCES keywords(id) ON DELETE CASCADE
            );
            """
        )
        op.execute(
            """
            INSERT INTO comment_keywords_new (comment_id, keyword_id)
            SELECT comment_id, keyword_id FROM comment_keywords;
            """
        )
        op.execute("DROP TABLE comment_keywords")
        op.execute("ALTER TABLE comment_keywords_new RENAME TO comment_keywords")

        op.execute("PRAGMA foreign_keys=ON")
    else:
        # For non-SQLite, adjust constraints with database-appropriate ALTER TABLE
        # (left as no-op; models enforce correct behavior for new deployments)
        pass


def downgrade() -> None:
    conn = op.get_bind()
    if conn.dialect.name == 'sqlite':
        op.execute("PRAGMA foreign_keys=OFF")

        # Recreate without ON DELETE CASCADE (original state)
        op.execute(
            """
            CREATE TABLE comments_old (
                id TEXT PRIMARY KEY,
                post_id TEXT NOT NULL,
                parent_id TEXT,
                author TEXT,
                body TEXT NOT NULL,
                score INTEGER,
                created_utc INTEGER NOT NULL,
                sentiment_score FLOAT,
                analyzed_at DATETIME,
                FOREIGN KEY(post_id) REFERENCES posts(id)
            );
            """
        )
        op.execute(
            """
            INSERT INTO comments_old (id, post_id, parent_id, author, body, score, created_utc, sentiment_score, analyzed_at)
            SELECT id, post_id, parent_id, author, body, score, created_utc, sentiment_score, analyzed_at FROM comments;
            """
        )
        op.execute("DROP INDEX IF EXISTS idx_comments_sentiment")
        op.execute("DROP INDEX IF EXISTS idx_comments_post")
        op.execute("DROP INDEX IF EXISTS idx_comments_created")
        op.execute("DROP TABLE comments")
        op.execute("ALTER TABLE comments_old RENAME TO comments")
        op.execute("CREATE INDEX idx_comments_post ON comments(post_id)")
        op.execute("CREATE INDEX idx_comments_sentiment ON comments(sentiment_score)")
        op.execute("CREATE INDEX idx_comments_created ON comments(created_utc)")

        op.execute(
            """
            CREATE TABLE post_keywords_old (
                post_id TEXT NOT NULL,
                keyword_id INTEGER NOT NULL,
                PRIMARY KEY (post_id, keyword_id),
                FOREIGN KEY(post_id) REFERENCES posts(id),
                FOREIGN KEY(keyword_id) REFERENCES keywords(id)
            );
            """
        )
        op.execute(
            """
            INSERT INTO post_keywords_old (post_id, keyword_id)
            SELECT post_id, keyword_id FROM post_keywords;
            """
        )
        op.execute("DROP TABLE post_keywords")
        op.execute("ALTER TABLE post_keywords_old RENAME TO post_keywords")

        op.execute(
            """
            CREATE TABLE comment_keywords_old (
                comment_id TEXT NOT NULL,
                keyword_id INTEGER NOT NULL,
                PRIMARY KEY (comment_id, keyword_id),
                FOREIGN KEY(comment_id) REFERENCES comments(id),
                FOREIGN KEY(keyword_id) REFERENCES keywords(id)
            );
            """
        )
        op.execute(
            """
            INSERT INTO comment_keywords_old (comment_id, keyword_id)
            SELECT comment_id, keyword_id FROM comment_keywords;
            """
        )
        op.execute("DROP TABLE comment_keywords")
        op.execute("ALTER TABLE comment_keywords_old RENAME TO comment_keywords")

        op.execute("PRAGMA foreign_keys=ON")
    else:
        pass

