"""
Configuration loader for the Reddit Sentiment Bot.

Loads settings from config files and provides default values.
"""

import os
import yaml
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class Config:
    """Configuration manager for the bot."""
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize configuration loader.
        
        Args:
            config_dir: Path to config directory, defaults to ../config from this file
        """
        if config_dir is None:
            config_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'config'
            )
        self.config_dir = config_dir
        self._config = None
        self._keywords = None
    
    @property
    def config(self) -> Dict[str, Any]:
        """Load main configuration file."""
        if self._config is None:
            config_path = os.path.join(self.config_dir, 'config.yaml')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    self._config = yaml.safe_load(f) or {}
                logger.info(f"Loaded configuration from {config_path}")
            else:
                self._config = self.get_default_config()
                logger.warning(f"Config file not found at {config_path}, using defaults")
        return self._config
    
    @property
    def keywords(self) -> Dict[str, Any]:
        """Load keywords configuration."""
        if self._keywords is None:
            keywords_path = os.path.join(self.config_dir, 'keywords.yaml')
            if os.path.exists(keywords_path):
                with open(keywords_path, 'r') as f:
                    self._keywords = yaml.safe_load(f) or {}
                logger.info(f"Loaded keywords from {keywords_path}")
            else:
                self._keywords = self.get_default_keywords()
                logger.warning(f"Keywords file not found at {keywords_path}, using defaults")
        return self._keywords
    
    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """Return default configuration."""
        return {
            'subreddits': ['ClaudeAI', 'Anthropic', 'ClaudeCode'],
            'claude': {
                'model': 'claude-3-5-sonnet-20241022',
                'batch_size': 15,
                'temperature': 0.3,
                'max_tokens': 2000
            },
            'reddit': {
                'requests_per_minute': 60,
                'time_filter': 'day',  # Fetch posts from last day
                'fetch_comments': True,
                'comment_limit': None  # None = fetch all comments
            },
            'cost': {
                'daily_limit': 5.0,
                'warn_threshold': 4.0
            },
            'database': {
                'path': './data/sentiment.db',
                'backup_before_run': True
            },
            'logging': {
                'level': 'INFO',
                'file': 'bot.log',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            }
        }
    
    @staticmethod
    def get_default_keywords() -> Dict[str, Any]:
        """Return default keywords configuration."""
        return {
            'keywords': {
                'products': [
                    'Claude', 'Claude 3', 'Claude 3.5', 
                    'Opus', 'Opus 4.1', 'Sonnet', 'Sonnet 4', 'Haiku'
                ],
                'features': [
                    'Artifacts', 'Projects', 'Claude Code', 
                    'Computer Use', 'Vision'
                ],
                'company': [
                    'Anthropic', 'Constitutional AI', 'RLHF'
                ],
                'competitors': [
                    'ChatGPT', 'GPT-4', 'GPT4', 'Gemini', 
                    'Copilot', 'Codex'
                ]
            }
        }
    
    def get_subreddits(self) -> list:
        """Get list of subreddits to monitor."""
        return self.config.get('subreddits', ['ClaudeAI', 'Anthropic', 'ClaudeCode'])
    
    def get_claude_config(self) -> Dict[str, Any]:
        """Get Claude API configuration."""
        return self.config.get('claude', {})
    
    def get_reddit_config(self) -> Dict[str, Any]:
        """Get Reddit API configuration."""
        return self.config.get('reddit', {})
    
    def get_cost_config(self) -> Dict[str, Any]:
        """Get cost management configuration."""
        return self.config.get('cost', {})
    
    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration."""
        return self.config.get('database', {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self.config.get('logging', {})
    
    def save_default_configs(self):
        """Save default configuration files for user reference."""
        os.makedirs(self.config_dir, exist_ok=True)
        
        # Save default config.yaml
        config_path = os.path.join(self.config_dir, 'config.yaml')
        if not os.path.exists(config_path):
            with open(config_path, 'w') as f:
                yaml.dump(self.get_default_config(), f, default_flow_style=False, sort_keys=False)
            logger.info(f"Created default config at {config_path}")
        
        # Save default keywords.yaml  
        keywords_path = os.path.join(self.config_dir, 'keywords.yaml')
        if not os.path.exists(keywords_path):
            with open(keywords_path, 'w') as f:
                yaml.dump(self.get_default_keywords(), f, default_flow_style=False, sort_keys=False)
            logger.info(f"Created default keywords at {keywords_path}")


# Singleton instance
_config = None


def get_config(config_dir: Optional[str] = None) -> Config:
    """Get singleton configuration instance."""
    global _config
    if _config is None:
        _config = Config(config_dir)
    return _config