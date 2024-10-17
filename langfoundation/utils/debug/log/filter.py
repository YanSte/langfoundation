import logging
from typing import Set


class LogFilter(logging.Filter):
    def __init__(
        self,
        allowed_levels: Set[int] = {
            logging.CRITICAL,
            logging.ERROR,
            logging.WARNING,
            logging.INFO,
        },
    ):
        super().__init__()
        self.allowed_levels = allowed_levels

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno in self.allowed_levels
