from __future__ import annotations

from typing import Optional


class PydanticChainError(Exception):
    """Custom exception class for errors that occur within Chain."""

    origin: str
    error: Exception

    def __init__(self, origin: str, error: Exception, custom_message: Optional[str] = None):
        custom_message = "" if custom_message is None else "\n >>>\n\nMessage: " + custom_message + "\n<<<"
        super().__init__(f"Error occurred in {origin.upper()}.{custom_message}\n- Original error:\n{str(error)}\n")
        self.origin = origin.upper()
        self.error = error
