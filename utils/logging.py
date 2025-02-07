import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict

class LogManager:
    """Simple rotating file logger with optional YAML config support"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.log_dir = Path(self.config.get("log_dir", "logs"))
        self.level = getattr(logging, self.config.get("level", "INFO"))
        
    def setup_basic_logging(self) -> None:
        """Setup basic logging with console and file output"""
        self.log_dir.mkdir(exist_ok=True)
        
        # Create formatters
        console_formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%H:%M:%S'
        )
        
        file_formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s - %(message)s'
        )

        # Setup console handler
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(self.level)
        console.setFormatter(console_formatter)

        # Setup file handler
        log_file = self.log_dir / f"trading_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(self.level)
        file_handler.setFormatter(file_formatter)

        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(self.level)
        root_logger.addHandler(console)
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
