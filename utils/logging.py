import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
from rich.logging import RichHandler

class LogManager:
    """Manages logging with different verbosity levels and file separation"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.log_dir = Path(self.config.get("log_dir", "logs"))
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
    def setup_basic_logging(self, verbosity: str = "INFO") -> None:
        """
        Configure basic logging with console and file handlers
        """
        level = getattr(logging, verbosity.upper(), logging.INFO)
        
        # Configure root logger
        logger = logging.getLogger()
        logger.setLevel(level)
        
        # Clear any existing handlers to prevent duplicates
        logger.handlers.clear()
        
        # Create formatter for file logs (detailed)
        file_formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)-8s %(message)-80s %(name)s:%(lineno)d",
            "%Y-%m-%d %H:%M:%S"
        )
        
        log_dir = self.log_dir  # Changed from self.base_dir
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(file_formatter)
        
        # Add Rich handler for console (with rich formatting support)
        from rich.logging import RichHandler
        console_handler = RichHandler(
            rich_tracebacks=True, 
            markup=True,  # Enable Rich markup parsing
            show_path=True
        )
        console_handler.setFormatter(logging.Formatter("%(message)s"))
        console_handler.setLevel(level)
        
        # Add handlers
        logger.addHandler(console_handler)  # Only one console handler with Rich support
        logger.addHandler(file_handler)
        
        # Log initial setup information
        logger.info(f"Logging initialized at level {verbosity}")
        logger.info(f"Logs will be saved to: {log_file}")
        
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

def setup_rich_logging():
    # Remove default handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add Rich handler
    rich_handler = RichHandler(markup=True)  # Enable markup parsing
    rich_handler.setFormatter(logging.Formatter("%(message)s"))
    root_logger.addHandler(rich_handler)
    root_logger.setLevel(logging.INFO)

def setup_logging():
    # Remove or reconfigure one of these handlers
    # Option 1: Keep only the Rich handler
    logging.basicConfig(handlers=[], level=logging.INFO)  # Empty the default handlers
    
    # Option 2: Make the handlers handle different log levels
    # console_handler.setLevel(logging.WARNING)  # Only show warnings and above in console
    # rich_handler.setLevel(logging.INFO)  # Show all info in Rich format
