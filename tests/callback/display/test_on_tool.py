from typing import Any, List
from unittest.mock import AsyncMock

import pytest
from langchain_core.agents import AgentAction

from langfoundation.callback.base.base import (
    BaseAsyncDisplayCallbackHandler,
)
from langfoundation.callback.base.records.agent import AgentRecord
from langfoundation.callback.base.records.retriever import (
    RetrieverRecord,
)
from langfoundation.callback.base.records.token import TokenStream
from langfoundation.callback.base.records.tool import (
    ToolRecord,
    ToolsState,
)
from langfoundation.callback.base.tags import Tags


@pytest.mark.asyncio
async def test_on_agent_action_should_on_agent() -> None:
    # Arrange
    # ---
    tags = [Tags.AGENT_FEEDBACK.value]

    tool = "tool name"
    tool_input = "tool input"
    log = "new log"

    callback = InspectorAsyncBaseDisplayCallbackHandler()
    callback.on_tool = AsyncMock()
    agent_action = AgentAction(tool=tool, tool_input=tool_input, log=log)

    # Act
    # ---
    await callback.on_agent_action(action=agent_action, tags=tags)

    # Assert
    # ---
    expected_tool_record = ToolRecord(
        name=tool,
        input_str=tool_input,
        state=ToolsState.REQUEST,
    )

    expected_tags = [Tags.AGENT_FEEDBACK.value]

    callback.on_tool.assert_awaited_once_with(
        tool=expected_tool_record,
        tags=expected_tags,
    )


##################################################
# AsyncBaseDisplayCallbackHandler Implementated
##################################################


class InspectorAsyncBaseDisplayCallbackHandler(BaseAsyncDisplayCallbackHandler):
    def __init__(
        self,
        view_tags: List[str] = [],
        feedback_tags: List[str] = [],
        display_agent_tags: List[str] = [],
        should_cumulate_token: bool = False,
        verbose: bool = True,
    ) -> None:
        self.view_tags = view_tags
        self.feedback_tags = feedback_tags
        self.display_agent_tags = display_agent_tags
        self.should_cumulate_token = should_cumulate_token
        self.verbose = verbose

    async def on_token_stream(self, token: TokenStream, **kwargs: Any) -> None:
        """
        Abstract method to handle a stream token event.
        """
        pass

    async def on_retriever(self, retriever: RetrieverRecord, **kwargs: Any) -> None:
        """
        Abstract method to handle a retriever event.
        """
        pass

    def non_async_on_retriever(self, retriever: RetrieverRecord, **kwargs: Any) -> None:
        """
        Abstract method to handle a retriever event.
        """
        pass

    async def on_tool(self, tool: ToolRecord, **kwargs: Any) -> None:
        """
        Abstract method to handle a tool event.
        """
        pass

    async def on_agent(self, agent: AgentRecord, **kwargs: Any) -> None:
        """
        Abstract method to handle an agent event.
        """
        pass

    async def on_feedback(self, feedback: str, **kwargs: Any) -> None:
        """
        Abstract method to handle a feedback event.
        """
