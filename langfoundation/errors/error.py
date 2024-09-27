from __future__ import annotations


class PydanticChainError(Exception):
    """Custom exception class for errors that occur within Chain."""

    origin: str
    error: Exception

    def __init__(self, origin: str, error: Exception):
        super().__init__([origin, error])
        self.origin = origin
        self.error = error
