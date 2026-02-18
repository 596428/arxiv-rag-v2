"""
arXiv RAG v1 - Logging Configuration

구조화된 로깅 설정
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_logging(
    name: str = "arxiv-rag",
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    log_to_console: bool = True,
) -> logging.Logger:
    """
    Set up a logger with consistent formatting.

    Args:
        name: Logger name
        level: Logging level (default: INFO)
        log_file: Optional file path for logging
        log_to_console: Whether to log to console

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers.clear()

    # Format: timestamp - level - module - message
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a child logger with the given name.

    Args:
        name: Module name (e.g., "collection", "parsing")

    Returns:
        Logger instance
    """
    return logging.getLogger(f"arxiv-rag.{name}")


class ProgressLogger:
    """
    Helper class for logging progress of batch operations.

    Usage:
        progress = ProgressLogger(total=1000, name="papers")
        for paper in papers:
            process(paper)
            progress.update()
    """

    def __init__(
        self,
        total: int,
        name: str = "items",
        logger: Optional[logging.Logger] = None,
        log_every: int = 100,
    ):
        self.total = total
        self.name = name
        self.logger = logger or get_logger("progress")
        self.log_every = log_every
        self.current = 0
        self.start_time = datetime.now()

    def update(self, count: int = 1) -> None:
        """Update progress counter."""
        self.current += count
        if self.current % self.log_every == 0 or self.current == self.total:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            rate = self.current / elapsed if elapsed > 0 else 0
            eta = (self.total - self.current) / rate if rate > 0 else 0
            self.logger.info(
                f"Progress: {self.current}/{self.total} {self.name} "
                f"({self.current/self.total*100:.1f}%) | "
                f"Rate: {rate:.1f}/s | ETA: {eta:.0f}s"
            )

    def done(self) -> None:
        """Log completion."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        self.logger.info(
            f"Completed: {self.current} {self.name} in {elapsed:.1f}s "
            f"({self.current/elapsed:.1f}/s)"
        )


# Initialize root logger on import
_root_logger = setup_logging()
