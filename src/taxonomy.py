"""
Keyword taxonomy loader and classifier.

Loads Claude-specific keyword categories from a YAML config and
provides helper functions to classify terms.
"""

from __future__ import annotations

import os
import logging
from typing import Dict, Set, Optional

import yaml

logger = logging.getLogger(__name__)


def _norm(term: str) -> str:
    return term.strip().lower()

def _compact(term: str) -> str:
    """Normalize and remove non-alphanumeric characters for fuzzy matching."""
    t = _norm(term)
    return "".join(ch for ch in t if ch.isalnum())


# Default taxonomy (normalized, lower-case)
DEFAULT_TAXONOMY: Dict[str, Set[str]] = {
    'product': {
        'claude', 'claude 3', 'claude 3.5', 'opus', 'opus 4.1', 'sonnet', 'sonnet 4', 'haiku'
    },
    'feature': {
        'artifacts', 'projects', 'claude code', 'computer use', 'vision'
    },
    'company': {
        'anthropic', 'constitutional ai', 'rlhf'
    },
    'competitor': {
        'chatgpt', 'gpt-4', 'gpt4', 'gemini', 'copilot', 'codex'
    },
}

_CACHED_PATH: Optional[str] = None
_CACHED_TAXONOMY: Optional[Dict[str, Set[str]]] = None
_CACHED_ALIASES: Optional[Dict[str, str]] = None  # compact_term -> category


def load_keywords_config(path: Optional[str] = None) -> Dict[str, Set[str]]:
    """Load taxonomy from YAML and merge with defaults.

    YAML structure:
    keywords:
      products: ["Claude", "Sonnet 4", ...]
      features: ["Artifacts", "Claude Code", ...]
      company: ["Anthropic", ...]
      competitors: ["Codex", "GPT-4", ...]
    """
    data: Dict[str, Set[str]] = {
        'product': set(DEFAULT_TAXONOMY['product']),
        'feature': set(DEFAULT_TAXONOMY['feature']),
        'company': set(DEFAULT_TAXONOMY['company']),
        'competitor': set(DEFAULT_TAXONOMY['competitor']),
    }

    if not path:
        return data

    try:
        with open(path, 'r', encoding='utf-8') as f:
            raw = yaml.safe_load(f) or {}
        kw = (raw.get('keywords') or {}) if isinstance(raw, dict) else {}

        # Merge with defaults (normalize terms)
        for cat_key, dst_cat in (
            ('products', 'product'),
            ('features', 'feature'),
            ('company', 'company'),
            ('competitors', 'competitor'),
        ):
            terms = kw.get(cat_key) or []
            if isinstance(terms, list):
                for t in terms:
                    if isinstance(t, str) and t.strip():
                        data[dst_cat].add(_norm(t))
    except FileNotFoundError:
        logger.info("Keyword config not found at %s; using defaults", path)
    except Exception as e:
        logger.warning("Failed to load keyword config %s: %s", path, e)

    return data


def get_taxonomy() -> Dict[str, Set[str]]:
    """Get cached taxonomy based on KEYWORDS_CONFIG_PATH or default file.

    - Uses env KEYWORDS_CONFIG_PATH if set
    - Else uses ./config/keywords.yaml if it exists
    - Falls back to defaults
    """
    global _CACHED_PATH, _CACHED_TAXONOMY
    env_path = os.environ.get('KEYWORDS_CONFIG_PATH')
    default_path = os.path.join('config', 'keywords.yaml')
    chosen = env_path or (default_path if os.path.exists(default_path) else None)

    if _CACHED_TAXONOMY is not None and _CACHED_PATH == chosen:
        return _CACHED_TAXONOMY

    taxonomy = load_keywords_config(chosen)
    _CACHED_TAXONOMY = taxonomy
    _CACHED_PATH = chosen
    # rebuild alias map
    _build_aliases_cache(taxonomy)
    return taxonomy


def _build_aliases_cache(taxonomy: Dict[str, Set[str]]) -> None:
    """Build a compact-term alias map for fast fuzzy matching."""
    global _CACHED_ALIASES
    alias: Dict[str, str] = {}
    for cat, terms in taxonomy.items():
        for term in terms:
            alias[_compact(term)] = cat
    _CACHED_ALIASES = alias


def classify_term(term: str, taxonomy: Optional[Dict[str, Set[str]]] = None) -> Optional[str]:
    """Classify a term into a keyword category or return None.

    Normalizes term to lower-case and matches exact tokens.
    """
    if not term or not isinstance(term, str):
        return None
    from difflib import SequenceMatcher

    t = _norm(term)
    if not t:
        return None

    # Exact match first
    tax = taxonomy or get_taxonomy()
    for cat, terms in tax.items():
        if t in terms:
            return cat

    # Use compact alias map for fuzzy/substring matching
    compact = _compact(term)
    alias = _CACHED_ALIASES
    if alias is None:
        _build_aliases_cache(tax)
        alias = _CACHED_ALIASES or {}

    # 1) Exact compact match
    if compact in alias:
        return alias[compact]

    # 2) Substring containment (prefer longest taxonomy term contained)
    best_cat = None
    best_len = 0
    for k, cat in alias.items():
        if k and (k in compact or compact in k):
            if len(k) > best_len:
                best_len = len(k)
                best_cat = cat
    if best_cat:
        return best_cat

    # 3) Fuzzy ratio threshold
    best_ratio = 0.0
    best_cat = None
    for k, cat in alias.items():
        ratio = SequenceMatcher(None, compact, k).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_cat = cat
    if best_ratio >= 0.85:
        return best_cat

    return None
