"""
Comprehensive tests for the secrets_manager module.

Tests include:
- Keychain access success/failure scenarios
- Environment variable fallback
- Secret validation
- Cross-platform compatibility
- Error handling
"""

import os
import subprocess
import pytest
from unittest.mock import patch, MagicMock, call
from typing import Dict, List

from src.secrets_manager import (
    SecretsManager,
    SecretsNotFoundError,
    InvalidSecretFormatError,
    load_secrets,
    get_secret,
    store_secret,
    validate_secrets
)


class TestSecretsManager:
    """Test suite for SecretsManager class."""
    
    @pytest.fixture
    def manager(self):
        """Create a fresh SecretsManager instance for each test."""
        return SecretsManager()
    
    @pytest.fixture
    def mock_env(self):
        """Mock environment with test secrets."""
        test_env = {
            'REDDIT_CLIENT_ID': 'test_client_id',
            'REDDIT_CLIENT_SECRET': 'test_client_secret',
            'REDDIT_USER_AGENT': 'test-bot/1.0 by /u/testuser',
            'ANTHROPIC_API_KEY': 'sk-test-key-123',
            'DATABASE_PATH': '/test/path/db.sqlite'
        }
        with patch.dict(os.environ, test_env, clear=True):
            yield test_env
    
    def test_keychain_availability_check_success(self, manager):
        """Test successful Keychain availability check."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            
            result = manager._check_keychain_availability()
            
            assert result is True
            mock_run.assert_called_once_with(
                ['security', 'list-keychains'],
                capture_output=True,
                text=True,
                check=False
            )
    
    def test_keychain_availability_check_failure(self, manager):
        """Test Keychain not available (non-macOS system)."""
        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = FileNotFoundError()
            
            result = manager._check_keychain_availability()
            
            assert result is False
    
    def test_get_secret_from_keychain_success(self, manager):
        """Test successful secret retrieval from Keychain."""
        manager.keychain_available = True
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout='test_secret_value\n'
            )
            
            result = manager.get_secret('TEST_SECRET')
            
            assert result == 'test_secret_value'
            mock_run.assert_called_once()
            
            # Verify the command structure
            cmd = mock_run.call_args[0][0]
            assert 'security' in cmd
            assert 'find-generic-password' in cmd
            assert 'TEST_SECRET' in cmd
    
    def test_get_secret_keychain_fallback_to_env(self, manager, mock_env):
        """Test fallback to environment variable when Keychain fails."""
        manager.keychain_available = True
        
        with patch('subprocess.run') as mock_run:
            # Simulate Keychain not having the secret
            mock_run.return_value = MagicMock(returncode=1)
            
            result = manager.get_secret('REDDIT_CLIENT_ID')
            
            assert result == 'test_client_id'
    
    def test_get_secret_no_keychain_uses_env(self, manager, mock_env):
        """Test direct environment variable use when Keychain unavailable."""
        manager.keychain_available = False
        
        result = manager.get_secret('REDDIT_CLIENT_ID')
        
        assert result == 'test_client_id'
    
    def test_get_secret_not_found(self, manager):
        """Test behavior when secret is not found anywhere."""
        manager.keychain_available = False
        
        with patch.dict(os.environ, {}, clear=True):
            result = manager.get_secret('NONEXISTENT_SECRET')
            
            assert result is None
    
    def test_store_secret_success(self, manager):
        """Test successful secret storage in Keychain."""
        manager.keychain_available = True
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            
            result = manager.store_secret('TEST_SECRET', 'test_value')
            
            assert result is True
            # Should be called twice: delete then add
            assert mock_run.call_count == 2
    
    def test_store_secret_keychain_unavailable(self, manager):
        """Test store secret fails gracefully when Keychain unavailable."""
        manager.keychain_available = False
        
        result = manager.store_secret('TEST_SECRET', 'test_value')
        
        assert result is False
    
    def test_validate_secrets_all_present(self, manager, mock_env):
        """Test validation when all required secrets are present."""
        manager.keychain_available = False
        
        is_valid, missing = manager.validate_secrets()
        
        assert is_valid is True
        assert missing == []
    
    def test_validate_secrets_some_missing(self, manager):
        """Test validation when some secrets are missing."""
        manager.keychain_available = False
        
        # Only provide some secrets
        partial_env = {
            'REDDIT_CLIENT_ID': 'test_id',
            'REDDIT_CLIENT_SECRET': 'test_secret'
        }
        
        with patch.dict(os.environ, partial_env, clear=True):
            is_valid, missing = manager.validate_secrets()
            
            assert is_valid is False
            assert 'REDDIT_USER_AGENT' in missing
            assert 'ANTHROPIC_API_KEY' in missing
    
    def test_validate_secret_format_reddit_user_agent_valid(self, manager):
        """Test valid Reddit user agent format."""
        valid_agent = 'sentiment-bot/1.0 by /u/testuser'
        
        result = manager.validate_secret_format('REDDIT_USER_AGENT', valid_agent)
        
        assert result is True
    
    def test_validate_secret_format_reddit_user_agent_invalid(self, manager):
        """Test invalid Reddit user agent format."""
        invalid_agent = 'just a string'
        
        with pytest.raises(InvalidSecretFormatError) as exc_info:
            manager.validate_secret_format('REDDIT_USER_AGENT', invalid_agent)
        
        assert 'format' in str(exc_info.value)
    
    def test_validate_secret_format_anthropic_key_valid(self, manager):
        """Test valid Anthropic API key format."""
        valid_key = 'sk-ant-api-key-123'
        
        result = manager.validate_secret_format('ANTHROPIC_API_KEY', valid_key)
        
        assert result is True
    
    def test_validate_secret_format_anthropic_key_invalid(self, manager):
        """Test invalid Anthropic API key format."""
        invalid_key = 'not-an-api-key'
        
        with pytest.raises(InvalidSecretFormatError) as exc_info:
            manager.validate_secret_format('ANTHROPIC_API_KEY', invalid_key)
        
        assert 'sk-' in str(exc_info.value)
    
    def test_validate_secret_format_empty_value(self, manager):
        """Test validation fails for empty values."""
        with pytest.raises(InvalidSecretFormatError) as exc_info:
            manager.validate_secret_format('ANY_SECRET', '')
        
        assert 'empty' in str(exc_info.value)
    
    def test_load_secrets_success(self, manager, mock_env):
        """Test successful loading of all secrets."""
        manager.keychain_available = False
        
        loaded = manager.load_secrets()
        
        assert len(loaded) == 5  # 4 required + 1 optional
        assert loaded['REDDIT_CLIENT_ID'] == 'test_client_id'
        assert loaded['ANTHROPIC_API_KEY'] == 'sk-test-key-123'
    
    def test_load_secrets_missing_required(self, manager):
        """Test load_secrets raises error when required secrets missing."""
        manager.keychain_available = False
        
        with patch.dict(os.environ, {'REDDIT_CLIENT_ID': 'test'}, clear=True):
            with pytest.raises(SecretsNotFoundError) as exc_info:
                manager.load_secrets()
            
            assert 'Missing required secrets' in str(exc_info.value)
    
    def test_load_secrets_invalid_format(self, manager):
        """Test load_secrets raises error for invalid secret format."""
        manager.keychain_available = False
        
        bad_env = {
            'REDDIT_CLIENT_ID': 'test_id',
            'REDDIT_CLIENT_SECRET': 'test_secret',
            'REDDIT_USER_AGENT': 'invalid-format',  # Bad format
            'ANTHROPIC_API_KEY': 'sk-test'
        }
        
        with patch.dict(os.environ, bad_env, clear=True):
            with pytest.raises(InvalidSecretFormatError):
                manager.load_secrets()
    
    def test_load_secrets_default_database_path(self, manager):
        """Test that DATABASE_PATH gets default value if not provided."""
        manager.keychain_available = False
        
        required_env = {
            'REDDIT_CLIENT_ID': 'test_id',
            'REDDIT_CLIENT_SECRET': 'test_secret',
            'REDDIT_USER_AGENT': 'bot/1.0 by /u/test',
            'ANTHROPIC_API_KEY': 'sk-test'
        }
        
        with patch.dict(os.environ, required_env, clear=True):
            loaded = manager.load_secrets()
            
            assert loaded['DATABASE_PATH'] == './data/sentiment.db'
            assert os.environ['DATABASE_PATH'] == './data/sentiment.db'
    
    def test_keychain_with_account_parameter(self, manager):
        """Test using account parameter for Keychain operations."""
        manager.keychain_available = True
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout='secret_value\n')
            
            result = manager.get_secret('TEST_SECRET', account='test_account')
            
            assert result == 'secret_value'
            
            # Verify account parameter was included
            cmd = mock_run.call_args[0][0]
            assert '-a' in cmd
            assert 'test_account' in cmd


class TestModuleFunctions:
    """Test module-level convenience functions."""
    
    def test_load_secrets_function(self):
        """Test the module-level load_secrets function."""
        with patch('src.secrets_manager.SecretsManager.load_secrets') as mock_load:
            mock_load.return_value = {'TEST': 'value'}
            
            result = load_secrets()
            
            assert result == {'TEST': 'value'}
            mock_load.assert_called_once()
    
    def test_get_secret_function(self):
        """Test the module-level get_secret function."""
        with patch('src.secrets_manager.SecretsManager.get_secret') as mock_get:
            mock_get.return_value = 'secret_value'
            
            result = get_secret('TEST_SECRET')
            
            assert result == 'secret_value'
            mock_get.assert_called_once_with('TEST_SECRET', None)
    
    def test_store_secret_function(self):
        """Test the module-level store_secret function."""
        with patch('src.secrets_manager.SecretsManager.store_secret') as mock_store:
            mock_store.return_value = True
            
            result = store_secret('TEST_SECRET', 'test_value')
            
            assert result is True
            mock_store.assert_called_once_with('TEST_SECRET', 'test_value', None)
    
    def test_validate_secrets_function(self):
        """Test the module-level validate_secrets function."""
        with patch('src.secrets_manager.SecretsManager.validate_secrets') as mock_validate:
            mock_validate.return_value = (True, [])
            
            is_valid, missing = validate_secrets()
            
            assert is_valid is True
            assert missing == []
            mock_validate.assert_called_once()


class TestErrorConditions:
    """Test error handling and edge cases."""
    
    def test_subprocess_exception_handling(self):
        """Test handling of subprocess exceptions."""
        manager = SecretsManager()
        manager.keychain_available = True
        
        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = Exception("Subprocess error")
            
            result = manager._get_from_keychain('TEST_SECRET')
            
            assert result is None
    
    def test_store_secret_with_subprocess_error(self):
        """Test store_secret handles subprocess errors gracefully."""
        manager = SecretsManager()
        manager.keychain_available = True
        
        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = Exception("Subprocess error")
            
            result = manager.store_secret('TEST_SECRET', 'value')
            
            assert result is False
    
    def test_validate_optional_secret_invalid_format(self):
        """Test that invalid optional secrets don't cause failure."""
        manager = SecretsManager()
        manager.keychain_available = False
        
        env = {
            'REDDIT_CLIENT_ID': 'test_id',
            'REDDIT_CLIENT_SECRET': 'test_secret',
            'REDDIT_USER_AGENT': 'bot/1.0 by /u/test',
            'ANTHROPIC_API_KEY': 'sk-test',
            'DATABASE_PATH': ''  # Invalid but optional
        }
        
        with patch.dict(os.environ, env, clear=True):
            loaded = manager.load_secrets()
            
            # Should succeed even with invalid optional secret
            assert 'REDDIT_CLIENT_ID' in loaded
            # DATABASE_PATH should get default value
            assert loaded['DATABASE_PATH'] == './data/sentiment.db'


class TestCrossPlatformCompatibility:
    """Test cross-platform compatibility."""
    
    def test_non_macos_system(self):
        """Test behavior on non-macOS systems."""
        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = FileNotFoundError("security command not found")
            
            manager = SecretsManager()
            
            assert manager.keychain_available is False
            
            # Should still work with environment variables
            with patch.dict(os.environ, {'TEST_SECRET': 'env_value'}):
                result = manager.get_secret('TEST_SECRET')
                assert result == 'env_value'
    
    def test_keychain_command_failure(self):
        """Test handling of Keychain command failures."""
        manager = SecretsManager()
        
        with patch('subprocess.run') as mock_run:
            # First call succeeds (availability check)
            # Second call fails (actual secret retrieval)
            mock_run.side_effect = [
                MagicMock(returncode=0),  # list-keychains succeeds
                MagicMock(returncode=1, stderr='The specified item could not be found')
            ]
            
            manager = SecretsManager()
            result = manager.get_secret('NONEXISTENT')
            
            assert result is None