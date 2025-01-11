import logging
import logging.handlers
from pathlib import Path
from typing import Dict


class LogManager:
    def __init__(self, config: Dict):
        self.config = config
        self.setup_logging()

    def setup_logging(self) -> None:
        """Configure logging system"""
        log_dir = Path(self.config.get("log_dir", "logs"))
        log_dir.mkdir(exist_ok=True)

        root_logger = logging.getLogger()
        root_logger.setLevel(self.config.get("level", "INFO"))

        # File handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / "fingpt.log", maxBytes=10_000_000, backupCount=5
        )
        file_handler.setFormatter(self._get_formatter())
        root_logger.addHandler(file_handler)

    def _get_formatter(self) -> logging.Formatter:
        """Create custom log formatter"""
        return logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
