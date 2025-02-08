"""
Configuration Manager

Centralized configuration management system that handles:
1. Environment variables (.env)
2. API credentials (secure storage)
3. Trading parameters (YAML configs)
4. System settings

Features:
- Secure credential management
- Configuration validation
- Environment variable processing
- YAML configuration loading
"""

import os
from typing import Dict, Any, Optional
from pathlib import Path
import yaml
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

class ConfigManager:
    """
    Single source of truth for all system configuration.
    
    Responsibilities:
        - Load and validate environment variables
        - Manage API credentials
        - Load YAML configurations
        - Provide access to system settings
    """

    def __init__(self):
        # Load environment first
        load_dotenv(verbose=True)
        
        # Initialize storage
        self.credentials = {}  # Secure credential storage
        self.configs = {}      # Configuration storage
        
        # Set up paths
        self.config_dir = Path(__file__).parent.parent / "config"
        
        # Load configurations
        self._load_credentials()  # Load API keys first
        self._load_configs()      # Load trading configs
        self._validate_env()      # Validate environment

    def _load_credentials(self) -> None:
        """Load and validate exchange credentials from environment"""
        binance_creds = {
            'api_key': os.getenv('BINANCE_API_KEY'),
            'api_secret': os.getenv('BINANCE_API_SECRET')
        }
        
        # Validate credentials exist and have correct format
        if not all(binance_creds.values()):
            raise ValueError("Missing Binance API credentials")
        
        # Validate credential lengths (Binance keys are 64 chars)
        if not all(len(str(v)) >= 64 for v in binance_creds.values()):
            raise ValueError("Invalid credential format")
            
        self.credentials['binance'] = binance_creds
        logger.info("Exchange credentials loaded and validated")

    def _validate_env(self) -> None:
        """Validate required environment variables"""
        required = {
            'BINANCE_API_KEY': 'API Key',
            'BINANCE_API_SECRET': 'API Secret'
        }
        
        missing = []
        for var, name in required.items():
            value = os.getenv(var)
            if value and len(value.strip()) >= 64:  # Basic validation for key length
                logger.info(f"{name} present: True (length: {len(value)})")
            else:
                missing.append(var)
                
        if missing:
            raise ValueError(f"Missing or invalid environment variables: {', '.join(missing)}")

    def _load_configs(self) -> None:
        """Load all configuration files"""
        if not self.config_dir.exists():
            raise FileNotFoundError(f"Config directory not found: {self.config_dir}")
            
        for yaml_file in self.config_dir.glob("*.yaml"):
            try:
                with yaml_file.open() as f:
                    config_name = yaml_file.stem
                    self.configs[config_name] = yaml.safe_load(f)
                logger.debug(f"Loaded configuration: {config_name}")
            except Exception as e:
                logger.error(f"Error loading {yaml_file}: {e}")

    def get_config(self, name: str) -> Optional[Dict[str, Any]]:
        """Get configuration by name"""
        return self.configs.get(name)

    def get_exchange_credentials(self, exchange: str) -> dict:
        """Get API credentials for specified exchange"""
        if exchange.lower() not in self.credentials:
            raise ValueError(f"No credentials found for {exchange}")
        return self.credentials[exchange.lower()]

def substitute_env_vars(config: Dict[str, Any]) -> Dict[str, Any]:
    """Substitute environment variables in config values"""
    if isinstance(config, dict):
        return {k: substitute_env_vars(v) for k, v in config.items()}
    return config  # Return config as is - no substitution

def load_config() -> Dict[str, Dict[str, Any]]:
    """Load all configuration files without environment variable substitution"""
    config_dir = Path(__file__).parent.parent / "config"
    
    if not config_dir.exists():
        raise FileNotFoundError(f"Config directory not found: {config_dir}")
        
    configs = {}
    for yaml_file in config_dir.glob("*.yaml"):
        with yaml_file.open() as f:
            configs[yaml_file.stem] = yaml.safe_load(f)
            
    return configs
