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
- Hierarchical access with dot notation
- Environment-specific configuration
"""

import os
import re
from typing import Dict, Any, Optional, Union, List
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
        - Support hierarchical config access
        - Handle environment-specific configurations
    """
    _instance = None  # Singleton pattern
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = ConfigManager()
        return cls._instance

    def __init__(self, env: str = None):
        # Skip if already initialized (singleton)
        if ConfigManager._instance is not None:
            return
            
        # Load environment first
        load_dotenv(verbose=True)
        
        # Set environment (dev/prod)
        self.env = env or os.getenv('TRADING_ENV', 'development')
        logger.info(f"Initializing ConfigManager for environment: {self.env}")
        
        # Initialize storage
        self.credentials = {}  # Secure credential storage
        self.configs = {}      # Configuration storage
        
        # Set up paths
        self.config_dir = Path(__file__).parent.parent / "config"
        
        # Load configurations
        self._load_credentials()  # Load API keys first
        self._load_configs()      # Load trading configs
        self._process_config_references()  # Handle cross-file references
        self._validate_env()      # Validate environment
        
        # Set as singleton instance
        ConfigManager._instance = self
        
        logger.info(f"ConfigManager initialized with {len(self.configs)} configuration files")

    def _load_credentials(self) -> None:
        """Load and validate exchange credentials from environment"""
        # Primary exchange
        self._load_exchange_credentials('binance')
        
        # Additional exchanges (if configured)
        for exchange in ['kucoin', 'gate']:
            try:
                self._load_exchange_credentials(exchange, required=False)
            except ValueError:
                # Skip if not configured
                pass
                
        # Load news API credentials
        self._load_api_credentials('cryptopanic', required=True)
        self._load_api_credentials('newsapi', required=False)
    
    def _load_exchange_credentials(self, exchange: str, required: bool = True) -> None:
        """Load credentials for a specific exchange"""
        key_var = f"{exchange.upper()}_API_KEY"
        secret_var = f"{exchange.upper()}_API_SECRET"
        
        api_key = os.getenv(key_var)
        api_secret = os.getenv(secret_var)
        
        if required and (not api_key or not api_secret):
            logger.error(f"Missing {exchange} credentials in environment")
            raise ValueError(f"{exchange} API credentials not set")
        
        if api_key and api_secret:
            # Store validated credentials
            self.credentials[exchange.lower()] = {
                'api_key': api_key,
                'api_secret': api_secret
            }
            
            # Add passphrase for exchanges that need it
            if exchange.lower() in ['kucoin']:
                passphrase = os.getenv(f"{exchange.upper()}_PASSPHRASE")
                if passphrase:
                    self.credentials[exchange.lower()]['passphrase'] = passphrase
            
            logger.info(f"{exchange.capitalize()} credentials loaded")
    
    def _load_api_credentials(self, service: str, required: bool = False) -> None:
        """Load credentials for a specific API service"""
        key_var = f"{service.upper()}_API_KEY"
        api_key = os.getenv(key_var)
        
        if required and not api_key:
            logger.error(f"Missing {service} API key in environment")
            raise ValueError(f"{service} API key not set")
            
        if api_key:
            self.credentials[service.lower()] = {'api_key': api_key}
            logger.info(f"{service.capitalize()} API credentials loaded")

    def _validate_env(self) -> None:
        """Validate required environment variables"""
        # Core required variables
        required = {
            'BINANCE_API_KEY': 'Binance API Key',
            'BINANCE_API_SECRET': 'Binance API Secret',
            'CRYPTOPANIC_API_KEY': 'CryptoPanic API Key'
        }
        
        missing = []
        for var, name in required.items():
            value = os.getenv(var)
            if value and len(value.strip()) > 10:  # Basic validation
                logger.info(f"{name} present: True")
            else:
                missing.append(name)
                
        if missing:
            raise ValueError(f"Missing or invalid environment variables: {', '.join(missing)}")
        
        logger.info("All required environment variables validated")

    def _load_configs(self) -> None:
        """Load all configuration files"""
        if not self.config_dir.exists():
            raise FileNotFoundError(f"Config directory not found: {self.config_dir}")
            
        # Load all base configs first
        for yaml_file in self.config_dir.glob("*.yaml"):
            try:
                with yaml_file.open() as f:
                    config_name = yaml_file.stem
                    self.configs[config_name] = yaml.safe_load(f)
                logger.debug(f"Loaded configuration: {config_name}")
            except Exception as e:
                logger.error(f"Error loading {yaml_file}: {e}")
        
        # Load environment-specific configs to override base configs
        env_config_dir = self.config_dir / self.env
        if env_config_dir.exists():
            for yaml_file in env_config_dir.glob("*.yaml"):
                try:
                    with yaml_file.open() as f:
                        config_name = yaml_file.stem
                        env_config = yaml.safe_load(f)
                        
                        # Merge with base config
                        if config_name in self.configs:
                            self._deep_update(self.configs[config_name], env_config)
                        else:
                            self.configs[config_name] = env_config
                            
                    logger.debug(f"Loaded {self.env} environment config: {config_name}")
                except Exception as e:
                    logger.error(f"Error loading environment config {yaml_file}: {e}")
        
        # Process environment variable substitution
        for config_name, config in self.configs.items():
            self.configs[config_name] = self._substitute_env_vars(config)

    def _process_config_references(self) -> None:
        """Process references between configuration files"""
        # Ensure trading.yaml has references to other config files
        if 'trading' in self.configs:
            trading_config = self.configs['trading']
            if 'configs' not in trading_config:
                trading_config['configs'] = {}
                
            # Add references to other config files
            config_files = ['strategies', 'model', 'services', 'logging']
            for config_file in config_files:
                if config_file in self.configs and config_file not in trading_config['configs']:
                    trading_config['configs'][config_file] = f"config/{config_file}.yaml"

    def _substitute_env_vars(self, config: Any) -> Any:
        """Substitute environment variables in config values"""
        if isinstance(config, dict):
            return {k: self._substitute_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._substitute_env_vars(item) for item in config]
        elif isinstance(config, str):
            # Replace ${VAR} or $VAR with environment variable
            pattern = r'\${([^}]+)}|\$([a-zA-Z0-9_]+)'
            
            def replace_var(match):
                var_name = match.group(1) or match.group(2)
                return os.getenv(var_name, '')
                
            return re.sub(pattern, replace_var, config)
        else:
            return config

    def _deep_update(self, original: Dict, update: Dict) -> None:
        """Recursively update a dictionary in place"""
        for key, value in update.items():
            if key in original and isinstance(original[key], dict) and isinstance(value, dict):
                self._deep_update(original[key], value)
            else:
                original[key] = value

    def get_config(self, name: str) -> Optional[Dict[str, Any]]:
        """Get configuration by name"""
        return self.configs.get(name)
        
    def get(self, path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation path
        
        Args:
            path: Path to config value using dot notation (e.g., 'trading.signals.threshold')
            default: Default value to return if path doesn't exist
            
        Returns:
            Config value or default
        """
        parts = path.split('.')
        
        # Get the first config file
        if not parts:
            return default
            
        config = self.configs.get(parts[0])
        if config is None:
            return default
            
        # Navigate the nested structure
        for part in parts[1:]:
            if not isinstance(config, dict):
                return default
                
            config = config.get(part)
            if config is None:
                return default
                
        return config

    def get_exchange_credentials(self, exchange: str) -> dict:
        """Get API credentials for specified exchange"""
        if exchange.lower() not in self.credentials:
            raise ValueError(f"No credentials found for {exchange}")
        return self.credentials[exchange.lower()]
        
    def get_api_key(self, service: str) -> str:
        """Get API key for specified service"""
        if service.lower() not in self.credentials:
            raise ValueError(f"No API key found for {service}")
        return self.credentials[service.lower()]['api_key']
        
    def get_paths(self) -> Dict[str, Path]:
        """Get system paths"""
        root_dir = self.config_dir.parent
        return {
            'root': root_dir,
            'config': self.config_dir,
            'logs': root_dir / 'logs',
            'data': root_dir / 'data',
            'models': root_dir / 'models',
        }
