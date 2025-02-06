import logging
import logging.handlers
from pathlib import Path
from typing import Dict

class LogManager:
    """Simple rotating file logger with optional YAML config support"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
    def setup_basic_logging(self) -> None:
        """Setup basic rotating file logger"""
        log_dir = Path(self.config.get("log_dir", "logs"))
        log_dir.mkdir(exist_ok=True)

        root_logger = logging.getLogger()
        root_logger.setLevel(self.config.get("level", "INFO"))

        # File handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / "fingpt.log", 
            maxBytes=10_000_000, 
            backupCount=5
        )
        file_handler.setFormatter(self._get_formatter())
        root_logger.addHandler(file_handler)
        
    @staticmethod
    def setup_from_yaml(yaml_path: str = None):
        """Use YAML config if available, else fall back to basic"""
        try:
            from utils.logging import setup_logging
            setup_logging(config_path=yaml_path)
        except Exception as e:
            print(f"Failed to load YAML config: {e}, falling back to basic logging")
            LogManager().setup_basic_logging()

    def _get_formatter(self) -> logging.Formatter:
        return logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )
