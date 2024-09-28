from __future__ import annotations

import logging
from typing import Any

from langfoundation.callback.base.base import BaseAsyncDisplayCallbackHandler
from langfoundation.callback.base.records.agent import AgentRecord
from langfoundation.callback.base.records.retriever import RetrieverRecord
from langfoundation.callback.base.records.token import TokenStream, TokenStreamState
from langfoundation.callback.base.records.tool import ToolRecord


logger = logging.getLogger(None)


class AsyncStreamCallbackHandler(BaseAsyncDisplayCallbackHandler):
    """
    A callback handler for stream events.

    This class provides methods to handle various events related to stream actions.
    """

    def __init__(
        self,
        should_cumulate_token: bool = False,
    ) -> None:
        """
        Initialize the callback handler.

        Args:
            should_cumulate_token: Whether to cumulate tokens.
        """
        self.should_cumulate_token = should_cumulate_token

    # Token
    # ----

    async def on_token_stream(
        self,
        token: TokenStream,
        **kwargs: Any,
    ) -> None:
        """
        Handle a stream token event.

        Args:
            token: The token to handle.
        """
        if token.state == TokenStreamState.START:
            logger.info("[START]")

        logger.info(token.token)

        if token.state == TokenStreamState.END:
            logger.info("[END]")

    # Reriever
    # ----

    async def on_retriever(self, retriever: RetrieverRecord, **kwargs: Any) -> None:
        """
        Abstract method to handle a retriever event.
        """

    def non_async_on_retriever(self, retriever: RetrieverRecord, **kwargs: Any) -> None:
        """
        Abstract method to handle a retriever event.
        """

    # Tool
    # ----

    async def on_tool(self, tool: ToolRecord, **kwargs: Any) -> None:
        """
        Abstract method to handle a tool event.
        """

    # Agent
    # ----

    async def on_agent(self, agent: AgentRecord, **kwargs: Any) -> None:
        """
        Abstract method to handle an agent event.
        """

    async def on_feedback(self, feedback: str, **kwargs: Any) -> None:
        """
        Abstract method to handle a feedback event.
        """
