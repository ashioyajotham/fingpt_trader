import os
from pathlib import Path

from typing import Dict, Any, Optional

import yaml


class ConfigManager:
    def __init__(self):
        self.config_dir = Path(__file__).parent.parent / "config"
        self.configs = {}
        self._load_configs()

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
            return {k: self._substitute_env_vars(v) for k, v in config.items()}
        elif (
            isinstance(config, str) and config.startswith("${") and config.endswith("}")
        ):
            env_var = config[2:-1]
            return os.getenv(env_var, config)
        return config

def substitute_env_vars(config: Dict[str, Any]) -> Dict[str, Any]:
    """Substitute environment variables in config values"""
    if isinstance(config, dict):
        return {k: substitute_env_vars(v) for k, v in config.items()}
    elif isinstance(config, str) and config.startswith("${") and config.endswith("}"):
        env_var = config[2:-1]
        return os.getenv(env_var, config)
    return config

def load_config() -> Dict[str, Dict[str, Any]]:
    """Load all configuration files with environment variable substitution"""
    config_dir = Path(__file__).parent.parent / "config"
    
    if not config_dir.exists():
        raise FileNotFoundError(f"Config directory not found: {config_dir}")
        
    configs = {}
    for yaml_file in config_dir.glob("*.yaml"):
        with yaml_file.open() as f:
            configs[yaml_file.stem] = substitute_env_vars(yaml.safe_load(f))
            
    return configs
