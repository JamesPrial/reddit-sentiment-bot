"""
Tests for taxonomy fuzzy classification.
"""

from src.taxonomy import classify_term, get_taxonomy


def test_fuzzy_matches_variants():
    # Ensure taxonomy is loaded
    _ = get_taxonomy()

    # Hyphen/space variants
    assert classify_term('Claude Code') == 'feature'
    assert classify_term('claude-code') == 'feature'
    assert classify_term('ClaudeCode') == 'feature'

    # Numeric variants
    assert classify_term('Opus 4.1') == 'product'
    assert classify_term('Opus4.1') == 'product'
    assert classify_term('sonnet-4') == 'product'
    assert classify_term('SONNET4') == 'product'

    # Competitors
    assert classify_term('GPT4') == 'competitor'
    assert classify_term('gpt-4') == 'competitor'
    assert classify_term('codex') == 'competitor'

