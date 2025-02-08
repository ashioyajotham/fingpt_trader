import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict

class LogManager:
    """Manages logging with different verbosity levels and file separation"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.log_dir = Path(self.config.get("log_dir", "logs"))
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
    def setup_basic_logging(self, verbosity: str = "INFO") -> None:
        """
        Setup logging with configurable verbosity
        
        Args:
            verbosity: Logging level ("DEBUG", "INFO", "WARNING", "ERROR")
        """
        self.log_dir.mkdir(exist_ok=True)
        
        # Map verbosity to logging level
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR
        }
        level = level_map.get(verbosity.upper(), logging.WARNING)
        
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
        console.setLevel(level)
        console.setFormatter(console_formatter)

        # Setup main log file
        main_log = self.log_dir / f"trading_{self.timestamp}.log"
        file_handler = logging.FileHandler(main_log)
        file_handler.setLevel(level)
        file_handler.setFormatter(file_formatter)

        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(level)
        
        # Remove any existing handlers
        root_logger.handlers.clear()
        
        # Add handlers
        root_logger.addHandler(console)
        root_logger.addHandler(file_handler)
        
        # Set third-party loggers to WARNING unless in DEBUG mode
        if verbosity.upper() != "DEBUG":
            for logger_name in ["urllib3", "binance", "asyncio"]:
                logging.getLogger(logger_name).setLevel(logging.WARNING)
        
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
