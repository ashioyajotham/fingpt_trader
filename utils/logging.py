import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

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

        # Setup main log file - use specific file if provided in config
        if "log_file" in self.config:
            main_log = Path(self.config["log_file"])
        else:
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
        
    def setup_rich_logging(self, level=logging.INFO):
        """Set up Rich-compatible logging"""
        from rich.logging import RichHandler
        
        # Create logs directory if needed
        self.log_dir.mkdir(exist_ok=True)
        
        # Configure rich handler for console output
        rich_handler = RichHandler(
            rich_tracebacks=True,
            markup=True,  # Enable rich markup in log messages
            show_time=False  # Rich adds its own timestamp
        )
        rich_handler.setLevel(level)
        
        # Configure file handler for log files
        main_log = self.log_dir / f"trading_{self.timestamp}.log"
        file_handler = logging.FileHandler(main_log, encoding='utf-8')  # Specify UTF-8 encoding
        file_formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(level)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(level)
        
        # Remove existing handlers and add new ones
        root_logger.handlers.clear()
        root_logger.addHandler(rich_handler)
        root_logger.addHandler(file_handler)
        
        # Set third-party loggers to WARNING
        for logger_name in ["urllib3", "binance", "asyncio"]:
            logging.getLogger(logger_name).setLevel(logging.WARNING)
        
    @staticmethod
    def setup_from_yaml(yaml_path: str = None):
        """
        This is now just a wrapper around setup_basic_logging for backwards compatibility.
        We no longer attempt to load YAML config but keep the method signature for compatibility.
        """
        # Just use basic logging without trying to load YAML config
        LogManager().setup_basic_logging()

    def _get_formatter(self) -> logging.Formatter:
        return logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )

def log_with_context(level: int, message: str, context: Dict[str, Any] = None):
    """Log a message with additional context data if DEBUG level is enabled."""
    logger = logging.getLogger()
    
    # Always log the main message at the specified level
    logger.log(level, message)
    
    # If we're in DEBUG mode and there's context, add it as a separate indented entry
    if context and logger.level <= logging.DEBUG:
        context_str = "\n  ".join(f"{k}: {v}" for k, v in context.items())
        logger.debug(f"Context:\n  {context_str}")

def debug(message: str, context: Dict[str, Any] = None):
    """Log a debug message with optional context."""
    log_with_context(logging.DEBUG, message, context)

def info(message: str, context: Dict[str, Any] = None):
    """Log an info message with optional context."""
    log_with_context(logging.INFO, message, context)

def warning(message: str, context: Dict[str, Any] = None):
    """Log a warning message with optional context."""
    log_with_context(logging.WARNING, message, context)

def error(message: str, context: Dict[str, Any] = None):
    """Log an error message with optional context."""
    log_with_context(logging.ERROR, message, context)
