from __future__ import annotations

from enum import auto
from typing import Optional

from strenum import LowercaseStrEnum

from langfoundation.callback.base.records.base import BaseRecord


# Stream Token
# ---


class TokenStreamState(LowercaseStrEnum):
    # Start Token
    START = auto()
    # Stream
    STREAM = auto()
    # End Token
    END = auto()
    # Error
    ERROR = auto()


class TokenOrigin(LowercaseStrEnum):
    LLM = auto()
    AGENT = auto()
    RETRIEVER = auto()


class TokenStream(BaseRecord):
    """
    Represents a token stream record.

    Attributes:
        id (int): The identifier for the token stream.
        token (str): The current token in the stream.
        cumulate_tokens (str): The accumulated tokens in the stream.
        state (TokenStreamState): The current state of the token stream.
        origin (TokenOrigin): The origin of the token stream (LLM or Agent).
    """

    id: int
    token: str
    cumulate_tokens: str
    state: TokenStreamState
    origin: TokenOrigin
    error: Optional[BaseException] = None

    class Config:
        arbitrary_types_allowed = True
