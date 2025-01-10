import yaml
import os
from typing import Any, Dict, Optional
from pathlib import Path

class ConfigManager:
    def __init__(self, config_dir: str = "../config"):
        self.config_dir = Path(config_dir)
        self.configs = {}
        self._load_configs()
        
    def _load_configs(self) -> None:
        """Load all YAML configs"""
        for file in self.config_dir.glob("*.yaml"):
            with open(file) as f:
                self.configs[file.stem] = yaml.safe_load(f)
                
    def get_config(self, name: str) -> Dict:
        """Get config by name with environment overrides"""
        config = self.configs.get(name, {})
        return self._apply_env_overrides(config)
        
    def _apply_env_overrides(self, config: Dict) -> Dict:
        """Override config values from environment variables"""
        for key, value in config.items():
            env_key = f"FINGPT_{key.upper()}"
            if env_key in os.environ:
                config[key] = os.environ[env_key]
        return config