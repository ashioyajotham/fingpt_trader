import os
from typing import Dict, Any, Optional
from pathlib import Path
import yaml
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

class ConfigManager:
    def __init__(self):
        # Load .env file first
        env_path = Path(__file__).parent.parent / '.env'
        load_dotenv(env_path)
        
        self.config_dir = Path(__file__).parent.parent / "config"
        self.configs = {}
        self._load_configs()
        self._validate_required_env_vars()

    def _validate_required_env_vars(self) -> None:
        """Validate required environment variables exist"""
        required_vars = {
            'BINANCE_API_KEY': 'Binance API key',
            'BINANCE_API_SECRET': 'Binance API secret'
        }
        
        missing = []
        for var, name in required_vars.items():
            value = os.getenv(var)
            if value:
                logger.info(f"{name} present: True (length: {len(value)})")
            else:
                missing.append(name)
                
        if missing:
            raise ValueError(f"Missing environment variables: {', '.join(missing)}")
        
        logger.info("Environment variables validated successfully")

    def _load_configs(self) -> None:
        """Load all YAML config files"""
        if not self.config_dir.exists():
            raise FileNotFoundError(f"Config directory not found: {self.config_dir}")

        for config_file in self.config_dir.glob("*.yaml"):
            if config_file.exists():
                with open(config_file) as f:
                    self.configs[config_file.stem] = yaml.safe_load(f)

    def get_config(self, name: str) -> Optional[Dict]:
        """Get config by name with environment variable substitution"""
        if name not in self.configs:
            return None

        config = self.configs[name]
        return self._substitute_env_vars(config)

    def _substitute_env_vars(self, config: Dict) -> Dict:
        """Replace ${ENV_VAR} with environment variable values"""
        if isinstance(config, dict):
            result = {}
            for k, v in config.items():
                if isinstance(v, dict):
                    result[k] = self._substitute_env_vars(v)
                elif isinstance(v, list):
                    result[k] = [self._substitute_env_vars(item) if isinstance(item, dict) else item for item in v]
                elif isinstance(v, str) and v.startswith('${') and v.endswith('}'):
                    env_var = v[2:-1]
                    env_value = os.environ.get(env_var)
                    if not env_value:
                        raise ValueError(f"Environment variable {env_var} not set")
                    result[k] = env_value
                else:
                    result[k] = v
            return result
        return config

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
