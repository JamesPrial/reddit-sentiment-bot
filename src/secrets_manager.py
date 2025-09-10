"""
Secure credential management using macOS Keychain with fallback to environment variables.

This module provides a secure way to manage API keys and other sensitive credentials,
prioritizing macOS Keychain storage over environment variables for enhanced security.
"""

import os
import subprocess
import json
import logging
from typing import Dict, Optional, List, Tuple

logger = logging.getLogger(__name__)


class SecretsNotFoundError(Exception):
    """Raised when required secrets are not found."""
    pass


class InvalidSecretFormatError(Exception):
    """Raised when a secret has an invalid format."""
    pass


class SecretsManager:
    """Manages secure credential storage and retrieval."""
    
    REQUIRED_SECRETS = [
        'REDDIT_CLIENT_ID',
        'REDDIT_CLIENT_SECRET',
        'REDDIT_USER_AGENT',
        'ANTHROPIC_API_KEY'
    ]
    
    OPTIONAL_SECRETS = [
        'DATABASE_PATH'
    ]
    
    SERVICE_NAME = 'reddit-sentiment-bot'
    
    def __init__(self):
        """Initialize the secrets manager."""
        self.keychain_available = self._check_keychain_availability()
        if not self.keychain_available:
            logger.warning("macOS Keychain not available, falling back to environment variables")
    
    def _check_keychain_availability(self) -> bool:
        """Check if macOS Keychain is available on this system."""
        try:
            result = subprocess.run(
                ['security', 'list-keychains'],
                capture_output=True,
                text=True,
                check=False
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False
    
    def get_secret(self, secret_name: str, account: Optional[str] = None) -> Optional[str]:
        """
        Retrieve a single secret from Keychain or environment variables.
        
        Args:
            secret_name: Name of the secret to retrieve
            account: Optional account name for Keychain lookup
            
        Returns:
            Secret value or None if not found
        """
        # Try Keychain first if available
        if self.keychain_available:
            value = self._get_from_keychain(secret_name, account)
            if value:
                return value
        
        # Fall back to environment variable
        env_value = os.environ.get(secret_name)
        if env_value:
            logger.debug(f"Retrieved {secret_name} from environment variable")
            return env_value
        
        return None
    
    def _get_from_keychain(self, secret_name: str, account: Optional[str] = None) -> Optional[str]:
        """
        Retrieve a secret from macOS Keychain.
        
        Args:
            secret_name: Name of the secret to retrieve
            account: Optional account name
            
        Returns:
            Secret value or None if not found
        """
        try:
            cmd = [
                'security', 'find-generic-password',
                '-s', self.SERVICE_NAME,
                '-l', secret_name,
                '-w'
            ]
            
            if account:
                cmd.extend(['-a', account])
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode == 0:
                logger.debug(f"Retrieved {secret_name} from Keychain")
                return result.stdout.strip()
            
        except Exception as e:
            logger.warning(f"Error accessing Keychain for {secret_name}: {e}")
        
        return None
    
    def store_secret(self, secret_name: str, value: str, account: Optional[str] = None) -> bool:
        """
        Store or update a secret in macOS Keychain.
        
        Args:
            secret_name: Name of the secret to store
            value: Secret value
            account: Optional account name
            
        Returns:
            True if successful, False otherwise
        """
        if not self.keychain_available:
            logger.error("Cannot store secret: Keychain not available")
            return False
        
        try:
            # First try to delete existing entry (ignore errors)
            delete_cmd = [
                'security', 'delete-generic-password',
                '-s', self.SERVICE_NAME,
                '-l', secret_name
            ]
            if account:
                delete_cmd.extend(['-a', account])
            
            subprocess.run(delete_cmd, capture_output=True, check=False)
            
            # Add new entry
            add_cmd = [
                'security', 'add-generic-password',
                '-s', self.SERVICE_NAME,
                '-l', secret_name,
                '-a', account or '',  # Account is required, use empty string if not provided
                '-w', value
            ]
            
            result = subprocess.run(
                add_cmd,
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode == 0:
                logger.info(f"Successfully stored {secret_name} in Keychain")
                return True
            else:
                logger.error(f"Failed to store {secret_name} in Keychain: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error storing secret in Keychain: {e}")
            return False
    
    def validate_secrets(self) -> Tuple[bool, List[str]]:
        """
        Validate that all required secrets are present.
        
        Returns:
            Tuple of (is_valid, list_of_missing_secrets)
        """
        missing_secrets = []
        
        for secret_name in self.REQUIRED_SECRETS:
            value = self.get_secret(secret_name)
            if not value:
                missing_secrets.append(secret_name)
        
        is_valid = len(missing_secrets) == 0
        
        if not is_valid:
            logger.error(f"Missing required secrets: {missing_secrets}")
        else:
            logger.info("All required secrets validated successfully")
        
        return is_valid, missing_secrets
    
    def validate_secret_format(self, secret_name: str, value: str) -> bool:
        """
        Validate the format of a specific secret.
        
        Args:
            secret_name: Name of the secret
            value: Secret value to validate
            
        Returns:
            True if format is valid
            
        Raises:
            InvalidSecretFormatError: If format is invalid
        """
        if not value or not value.strip():
            raise InvalidSecretFormatError(f"{secret_name} cannot be empty")
        
        # Specific validation rules
        if secret_name == 'REDDIT_USER_AGENT':
            if '/' not in value or 'by' not in value.lower():
                raise InvalidSecretFormatError(
                    f"{secret_name} must be in format: 'bot-name/version by /u/username'"
                )
        
        elif secret_name == 'ANTHROPIC_API_KEY':
            if not value.startswith('sk-'):
                raise InvalidSecretFormatError(
                    f"{secret_name} must start with 'sk-'"
                )
        
        elif secret_name == 'DATABASE_PATH':
            # Just check it's a valid path format
            if not value or value.startswith(' '):
                raise InvalidSecretFormatError(
                    f"{secret_name} must be a valid path"
                )
        
        return True
    
    def load_secrets(self) -> Dict[str, str]:
        """
        Load all required secrets into environment variables.
        
        Priority: macOS Keychain > Environment Variables
        
        Returns:
            Dictionary of loaded secrets for validation
            
        Raises:
            SecretsNotFoundError: If required secrets are missing
        """
        loaded_secrets = {}
        
        # Load all required and optional secrets
        all_secrets = self.REQUIRED_SECRETS + self.OPTIONAL_SECRETS
        
        for secret_name in all_secrets:
            value = self.get_secret(secret_name)
            
            if value:
                # Validate format before setting
                try:
                    self.validate_secret_format(secret_name, value)
                except InvalidSecretFormatError as e:
                    if secret_name in self.REQUIRED_SECRETS:
                        raise
                    else:
                        logger.warning(f"Optional secret validation failed: {e}")
                        continue
                
                # Set in environment for application use
                os.environ[secret_name] = value
                loaded_secrets[secret_name] = value
                logger.debug(f"Loaded {secret_name}")
        
        # Validate all required secrets are present
        is_valid, missing = self.validate_secrets()
        if not is_valid:
            raise SecretsNotFoundError(
                f"Missing required secrets: {', '.join(missing)}"
            )
        
        # Set default for optional DATABASE_PATH if not provided
        if 'DATABASE_PATH' not in loaded_secrets:
            default_path = './data/sentiment.db'
            os.environ['DATABASE_PATH'] = default_path
            loaded_secrets['DATABASE_PATH'] = default_path
            logger.info(f"Using default DATABASE_PATH: {default_path}")
        
        logger.info(f"Successfully loaded {len(loaded_secrets)} secrets")
        return loaded_secrets


# Module-level convenience functions
_manager = None

def get_manager() -> SecretsManager:
    """Get or create the singleton SecretsManager instance."""
    global _manager
    if _manager is None:
        _manager = SecretsManager()
    return _manager


def load_secrets() -> Dict[str, str]:
    """
    Load all required secrets into environment variables.
    
    This is the main entry point for applications to load secrets.
    
    Returns:
        Dictionary of loaded secrets
        
    Raises:
        SecretsNotFoundError: If required secrets are missing
    """
    return get_manager().load_secrets()


def get_secret(secret_name: str, account: Optional[str] = None) -> Optional[str]:
    """
    Retrieve a single secret from Keychain or environment.
    
    Args:
        secret_name: Name of the secret to retrieve
        account: Optional account name for Keychain lookup
        
    Returns:
        Secret value or None if not found
    """
    return get_manager().get_secret(secret_name, account)


def store_secret(secret_name: str, value: str, account: Optional[str] = None) -> bool:
    """
    Store or update a secret in macOS Keychain.
    
    Args:
        secret_name: Name of the secret to store
        value: Secret value
        account: Optional account name
        
    Returns:
        True if successful, False otherwise
    """
    return get_manager().store_secret(secret_name, value, account)


def validate_secrets() -> Tuple[bool, List[str]]:
    """
    Validate that all required secrets are present.
    
    Returns:
        Tuple of (is_valid, list_of_missing_secrets)
    """
    return get_manager().validate_secrets()