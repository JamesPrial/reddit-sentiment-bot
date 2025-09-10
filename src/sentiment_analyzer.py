"""
Sentiment analysis module using Claude API for Reddit content analysis.

This module provides batch sentiment analysis capabilities for Reddit posts
and comments, with cost optimization and comprehensive error handling.
"""

import json
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, date
from dataclasses import dataclass
import time

from anthropic import Anthropic, RateLimitError, APIError
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


@dataclass
class CostTracking:
    """Track API costs for the current session."""
    daily_limit: float
    current_daily_cost: float
    session_cost: float
    input_tokens: int
    output_tokens: int
    
    @property
    def remaining_budget(self) -> float:
        """Calculate remaining daily budget."""
        return self.daily_limit - self.current_daily_cost


class SentimentAnalyzer:
    """Claude API integration for batch sentiment analysis."""
    
    # Claude 3.5 Sonnet pricing (per 1M tokens)
    INPUT_COST_PER_1M = 3.0  # $3 per million input tokens
    OUTPUT_COST_PER_1M = 15.0  # $15 per million output tokens
    
    # Batch processing configuration
    DEFAULT_BATCH_SIZE = 15
    MAX_BATCH_SIZE = 20
    MIN_BATCH_SIZE = 5
    
    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022", cost_manager: Optional["CostManager"] = None):
        """
        Initialize Anthropic client with configuration.
        
        Args:
            api_key: Anthropic API key
            model: Claude model to use for analysis
        """
        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.cost_manager = cost_manager
        self.cost_tracker = CostTracking(
            daily_limit=5.0,
            current_daily_cost=0.0,
            session_cost=0.0,
            input_tokens=0,
            output_tokens=0
        )
        
    def analyze_batch(self, items: List[Dict], item_type: str = 'post') -> List[Dict]:
        """
        Analyze sentiment for batch of posts/comments.
        
        Args:
            items: List of items to analyze (posts or comments)
            item_type: Type of items ('post' or 'comment')
            
        Returns:
            List of items with sentiment scores added
        """
        if not items:
            return []
            
        # Check budget before proceeding
        estimated_cost = self._estimate_batch_cost(items)
        if not self._can_proceed_with_cost(estimated_cost):
            logger.warning(f"Cost limit would be exceeded. Estimated: ${estimated_cost:.2f}, "
                         f"Remaining: ${self.cost_tracker.remaining_budget:.2f}")
            raise ValueError("Daily cost limit would be exceeded")
        
        # Prepare batch prompt
        prompt = self.prepare_batch_prompt(items, item_type)
        
        try:
            # Call Claude API with retry logic
            response = self._call_claude_api(prompt)
            
            # Parse response and add sentiment scores
            sentiment_results = self.parse_sentiment_response(response.content[0].text)
            
            # Update cost tracking
            self._update_cost_tracking(response.usage)
            
            # Merge sentiment scores with original items
            # Prefer explicit index mapping returned by the model if present
            index_map = {}
            if sentiment_results and isinstance(sentiment_results, list) and isinstance(sentiment_results[0], dict) and 'index' in sentiment_results[0]:
                try:
                    index_map = {int(r['index']): r for r in sentiment_results if 'index' in r}
                except Exception:
                    index_map = {}
            if index_map:
                for i, item in enumerate(items):
                    r = index_map.get(i)
                    if r:
                        # Do not propagate the helper index key to items
                        r = {k: v for k, v in r.items() if k != 'index'}
                        item.update(r)
                    else:
                        item.update({
                            'sentiment_score': 0.0,
                            'sentiment_explanation': 'Analysis missing for this item',
                            'requires_reanalysis': True
                        })
            else:
                for i, item in enumerate(items):
                    if i < len(sentiment_results):
                        item.update(sentiment_results[i])
                    else:
                        # Fallback for missing results
                        item.update({
                            'sentiment_score': 0.0,
                            'sentiment_explanation': 'Analysis failed',
                            'requires_reanalysis': True
                        })
            
            return items
            
        except Exception as e:
            logger.error(f"Batch analysis failed: {str(e)}")
            # Mark all items for reanalysis
            for item in items:
                item.update({
                    'sentiment_score': 0.0,
                    'sentiment_explanation': f'Error: {str(e)}',
                    'requires_reanalysis': True
                })
            return items
    
    def prepare_batch_prompt(self, items: List[Dict], item_type: str) -> str:
        """
        Create optimized prompt for batch analysis.
        
        Args:
            items: List of items to analyze
            item_type: Type of items ('post' or 'comment')
            
        Returns:
            Formatted prompt string
        """
        # Prepare items for analysis
        items_for_analysis = []
        for i, item in enumerate(items):
            if item_type == 'post':
                content = {
                    'index': i,
                    'title': item.get('title', ''),
                    'text': item.get('selftext', '')[:1000],  # Limit text length
                    'subreddit': item.get('subreddit', ''),
                    'score': item.get('score', 0)
                }
            else:  # comment
                content = {
                    'index': i,
                    'text': item.get('body', '')[:500],  # Shorter limit for comments
                    'score': item.get('score', 0)
                }
            items_for_analysis.append(content)
        
        items_json = json.dumps(items_for_analysis, indent=2)
        
        prompt = f"""Analyze the sentiment of these Reddit {item_type}s about Claude, Anthropic, and AI assistants.
Consider the technical nature of AI/LLM discussions where criticism can be constructive.

For each item, provide:
1. sentiment_score: float from -1.0 (very negative) to 1.0 (very positive)
   - -1.0 to -0.6: Very negative (hostile, angry, deeply frustrated)
   - -0.6 to -0.2: Negative (critical, disappointed, problematic)
   - -0.2 to 0.2: Neutral (balanced, factual, mixed feelings)
   - 0.2 to 0.6: Positive (satisfied, helpful, constructive)
   - 0.6 to 1.0: Very positive (enthusiastic, praising, highly satisfied)
2. sentiment_explanation: brief reason for the score (max 50 words)
3. keywords: list of notable terms mentioned (products, features, issues)

Consider context clues:
- Technical criticism with solutions → less negative
- Frustration without hostility → mildly negative
- Factual comparisons → neutral
- Enthusiastic recommendations → positive

Items to analyze:
{items_json}

Return ONLY a JSON array with analysis for each item in order, maintaining the same index values.
Example format:
[
  {{
    "index": 0,
    "sentiment_score": 0.5,
    "sentiment_explanation": "User appreciates Claude's capabilities",
    "keywords": ["Claude", "helpful", "coding"]
  }}
]"""
        
        return prompt
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60)
    )
    def _call_claude_api(self, prompt: str):
        """
        Call Claude API with retry logic.
        
        Args:
            prompt: The prompt to send to Claude
            
        Returns:
            API response object
        """
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                temperature=0.3,  # Lower temperature for more consistent scoring
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            return response
            
        except RateLimitError as e:
            logger.warning(f"Rate limit hit, retrying: {str(e)}")
            # Rely on tenacity backoff; do not add extra sleep here
            raise
            
        except APIError as e:
            logger.error(f"Claude API error: {str(e)}")
            raise
    
    def parse_sentiment_response(self, response: str) -> List[Dict]:
        """
        Parse Claude's JSON response into sentiment scores.
        
        Args:
            response: Raw response string from Claude
            
        Returns:
            List of parsed sentiment dictionaries
        """
        try:
            # Clean response if needed
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.endswith("```"):
                response = response[:-3]
            
            # Parse JSON
            results = json.loads(response.strip())
            
            # Validate and clean results
            cleaned_results = []
            for item in results:
                # Validate sentiment score
                score = item.get('sentiment_score', 0.0)
                score = self._validate_sentiment_score(score)
                
                cleaned_item = {
                    'sentiment_score': score,
                    'sentiment_explanation': item.get('sentiment_explanation', '')[:200],
                    'keywords': item.get('keywords', [])[:10]  # Limit keywords
                }
                # Preserve index if provided for alignment
                if isinstance(item, dict) and 'index' in item:
                    try:
                        cleaned_item['index'] = int(item['index'])
                    except Exception:
                        pass
                cleaned_results.append(cleaned_item)
            
            return cleaned_results
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Claude response: {str(e)}")
            logger.debug(f"Response was: {response[:500]}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error parsing response: {str(e)}")
            return []
    
    def _validate_sentiment_score(self, score: float) -> float:
        """
        Ensure score is within valid range [-1.0, 1.0].
        
        Args:
            score: Raw sentiment score
            
        Returns:
            Validated score within range
        """
        try:
            score = float(score)
            return max(-1.0, min(1.0, score))
        except (TypeError, ValueError):
            logger.warning(f"Invalid sentiment score: {score}, defaulting to 0.0")
            return 0.0
    
    def calculate_api_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Calculate estimated API cost for the request.
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            
        Returns:
            Estimated cost in dollars
        """
        input_cost = (input_tokens / 1_000_000) * self.INPUT_COST_PER_1M
        output_cost = (output_tokens / 1_000_000) * self.OUTPUT_COST_PER_1M
        return input_cost + output_cost
    
    def _estimate_batch_cost(self, items: List[Dict]) -> float:
        """
        Estimate cost for analyzing a batch of items.
        
        Args:
            items: Items to analyze
            
        Returns:
            Estimated cost in dollars
        """
        # Rough estimation: ~100 tokens per item input, ~50 tokens per item output
        estimated_input_tokens = len(items) * 150 + 500  # Include prompt overhead
        estimated_output_tokens = len(items) * 75
        return self.calculate_api_cost(estimated_input_tokens, estimated_output_tokens)
    
    def _can_proceed_with_cost(self, estimated_cost: float) -> bool:
        """
        Check if request would exceed daily limit.
        
        Args:
            estimated_cost: Estimated cost for the request
            
        Returns:
            True if within budget, False otherwise
        """
        # Prefer centralized CostManager if provided
        if self.cost_manager is not None:
            try:
                return self.cost_manager.can_proceed(estimated_cost)
            except Exception:
                logger.warning("CostManager.can_proceed failed; falling back to local tracker")
        return (self.cost_tracker.current_daily_cost + estimated_cost) <= self.cost_tracker.daily_limit
    
    def _update_cost_tracking(self, usage):
        """
        Update cost tracking with actual usage.
        
        Args:
            usage: Usage object from API response
        """
        if usage:
            input_tokens = getattr(usage, 'input_tokens', 0)
            output_tokens = getattr(usage, 'output_tokens', 0)
            
            cost = self.calculate_api_cost(input_tokens, output_tokens)
            
            self.cost_tracker.input_tokens += input_tokens
            self.cost_tracker.output_tokens += output_tokens
            self.cost_tracker.session_cost += cost
            self.cost_tracker.current_daily_cost += cost
            # Also record via CostManager if present (pass run_id if available)
            if self.cost_manager is not None:
                try:
                    run_id = getattr(self, 'run_id', None)
                    self.cost_manager.record_usage(cost, run_id=run_id)
                except Exception:
                    logger.warning("CostManager.record_usage failed; continuing with local tracking")
            
            logger.info(f"API call cost: ${cost:.4f} "
                       f"(Session total: ${self.cost_tracker.session_cost:.2f}, "
                       f"Daily total: ${self.cost_tracker.current_daily_cost:.2f})")
    
    def get_optimal_batch_size(self, items: List[Dict]) -> int:
        """
        Determine optimal batch size based on content length.
        
        Args:
            items: Items to process
            
        Returns:
            Optimal batch size
        """
        if not items:
            return self.DEFAULT_BATCH_SIZE
            
        # Calculate average content length
        total_length = sum(
            len(item.get('title', '') + item.get('selftext', '') + item.get('body', ''))
            for item in items
        )
        avg_length = total_length / len(items) if items else 0
        
        # Adjust batch size based on content length
        if avg_length < 100:
            return self.MAX_BATCH_SIZE
        elif avg_length < 500:
            return self.DEFAULT_BATCH_SIZE
        else:
            return self.MIN_BATCH_SIZE
    
    def process_items_in_batches(self, items: List[Dict], item_type: str = 'post') -> List[Dict]:
        """
        Process multiple items in optimal batches.
        
        Args:
            items: All items to process
            item_type: Type of items ('post' or 'comment')
            
        Returns:
            All items with sentiment scores
        """
        if not items:
            return []
        
        batch_size = self.get_optimal_batch_size(items)
        all_results = []
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1} "
                       f"({len(batch)} {item_type}s)")
            
            try:
                results = self.analyze_batch(batch, item_type)
                all_results.extend(results)
            except ValueError as e:
                # Cost limit exceeded
                logger.error(f"Stopping batch processing: {str(e)}")
                # Add remaining items as unanalyzed
                for item in items[i:]:
                    item.update({
                        'sentiment_score': 0.0,
                        'sentiment_explanation': 'Not analyzed - cost limit',
                        'requires_reanalysis': True
                    })
                    all_results.append(item)
                break
            except Exception as e:
                logger.error(f"Batch {i//batch_size + 1} failed: {str(e)}")
                # Add failed batch as unanalyzed
                for item in batch:
                    item.update({
                        'sentiment_score': 0.0,
                        'sentiment_explanation': f'Batch failed: {str(e)}',
                        'requires_reanalysis': True
                    })
                    all_results.append(item)
        
        return all_results


class CostManager:
    """Manage API costs and enforce daily limits."""
    
    def __init__(self, daily_limit: float = 5.0, db_manager=None):
        """
        Initialize cost tracking with daily limit.
        
        Args:
            daily_limit: Maximum daily spend in dollars
            db_manager: Optional database manager for persistence
        """
        self.daily_limit = daily_limit
        self.db_manager = db_manager
        self.today = date.today()
        self._daily_cost = None
    
    @property
    def daily_cost(self) -> float:
        """Get current daily cost from database or cache."""
        # Reset if new day
        if date.today() != self.today:
            self.today = date.today()
            self._daily_cost = None
        
        if self._daily_cost is None:
            self._daily_cost = self._load_daily_cost()
        
        return self._daily_cost
    
    def _load_daily_cost(self) -> float:
        """Load today's cost from database."""
        if not self.db_manager:
            return 0.0
        
        try:
            # Query today's analysis runs for total cost
            from datetime import datetime
            today_start = datetime.combine(self.today, datetime.min.time())
            
            with self.db_manager.transaction() as session:
                from src.database import AnalysisRun
                from sqlalchemy import func
                
                total = session.query(func.sum(AnalysisRun.api_cost_estimate)).filter(
                    AnalysisRun.started_at >= today_start
                ).scalar()
                
                return float(total or 0.0)
                
        except Exception as e:
            logger.error(f"Failed to load daily cost: {str(e)}")
            return 0.0
    
    def can_proceed(self, estimated_cost: float) -> bool:
        """
        Check if request would exceed daily limit.
        
        Args:
            estimated_cost: Estimated cost for the request
            
        Returns:
            True if within budget, False otherwise
        """
        return (self.daily_cost + estimated_cost) <= self.daily_limit
    
    def record_usage(self, actual_cost: float, run_id: Optional[int] = None):
        """
        Record actual API usage for tracking.
        
        Args:
            actual_cost: Actual cost incurred
            run_id: Optional analysis run ID to update
        """
        self._daily_cost = self.daily_cost + actual_cost
        
        if self.db_manager and run_id:
            try:
                with self.db_manager.transaction() as session:
                    from src.database import AnalysisRun
                    
                    run = session.query(AnalysisRun).filter_by(id=run_id).first()
                    if run:
                        run.api_cost_estimate = (run.api_cost_estimate or 0) + actual_cost
                        run.api_calls = (run.api_calls or 0) + 1
                        
            except Exception as e:
                logger.error(f"Failed to record usage in database: {str(e)}")
    
    def get_remaining_budget(self) -> float:
        """Get remaining budget for today."""
        return max(0, self.daily_limit - self.daily_cost)
    
    def reset_daily_tracking(self):
        """Reset daily cost tracking (for testing)."""
        self.today = date.today()
        self._daily_cost = None
