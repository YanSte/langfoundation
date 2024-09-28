from __future__ import annotations

from enum import auto

from strenum import LowercaseStrEnum


class AsyncLogHandlerType(LowercaseStrEnum):
    """
    Enumeration for Async Log Handler Types.

    Attributes:
        ON_CHAIN_START: Event logged when a chain process starts.
        ON_CHAIN_END: Event logged when a chain process ends.
        ON_CHAIN_ERROR: Event logged when an error occurs in a chain process.
        ON_LLM_START: Event logged when an LLM process starts.
        ON_LLM_ERROR: Event logged when an error occurs in an LLM process.
        ON_LLM_NEW_TOKEN: Event logged for each new token generated during an LLM process.
        ON_LLM_END: Event logged when an LLM process ends.
        ON_TOOL_START: Event logged when a tool process starts.
        ON_TOOL_END: Event logged when a tool process ends.
        ON_TOOL_ERROR: Event logged when an error occurs in a tool process.
        ON_RETRIEVER_START: Event logged when a retriever process starts.
        ON_RETRIEVER_END: Event logged when a retriever process ends.
        ON_AGENT_ACTION: Event logged for each action taken by an agent.
        ON_AGENT_FINISH: Event logged when an agent completes its task.
    """

    ON_CHAIN_START = auto()
    ON_CHAIN_END = auto()
    ON_CHAIN_ERROR = auto()
    ON_LLM_START = auto()
    ON_LLM_ERROR = auto()
    ON_LLM_NEW_TOKEN = auto()
    ON_LLM_END = auto()
    ON_TOOL_START = auto()
    ON_TOOL_END = auto()
    ON_TOOL_ERROR = auto()
    ON_RETRIEVER_START = auto()
    ON_RETRIEVER_END = auto()
    ON_AGENT_ACTION = auto()
    ON_AGENT_FINISH = auto()
