from typing import Any
from unittest.mock import AsyncMock

from langchain_core.agents import AgentAction, AgentFinish
import pytest

from langfoundation.callback.base.base import (
    BaseAsyncDisplayCallbackHandler,
)
from langfoundation.callback.base.records.agent import (
    AgentRecord,
    AgentState,
)
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
    callback.on_agent = AsyncMock()
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

    expected_agent_record = AgentRecord(
        state=AgentState.THOUGHT,
        tool=expected_tool_record,
        log=log,
    )

    expected_tags = [Tags.AGENT_FEEDBACK.value]

    callback.on_agent.assert_called_once_with(agent=expected_agent_record, tags=expected_tags)


@pytest.mark.asyncio
async def test_on_agent_action_when_without_agent_tags_should_not_on_agent() -> None:
    # Arrange
    # ---
    callback = InspectorAsyncBaseDisplayCallbackHandler()
    callback.on_agent = AsyncMock()

    # Act
    # ---
    await callback.on_agent_action(action=AsyncMock(), tags=["wrong_tags"])

    # Assert
    # ---
    callback.on_agent.assert_not_called()


@pytest.mark.asyncio
async def test_on_agent_finish_when_agent_answer_directly_should_on_agent() -> None:
    """
    When answer directly
    - No tool are used
    - Skip on_tool use
    - Skip on_agent_start
    - Callback _agent_record Is null

    Example:
    - Missing some context
    - Has already some context in history
    - Base on the model knowledge
    """
    # Arrange
    # ---
    tags = [Tags.AGENT_FEEDBACK.value]

    return_values = {"output": "return value"}
    log = "new log"
    callback = InspectorAsyncBaseDisplayCallbackHandler()
    callback.on_agent = AsyncMock()

    agent_action = AgentFinish(return_values=return_values, log=log)

    # Act
    # ---
    assert callback._agent_record is None
    await callback.on_agent_finish(finish=agent_action, tags=tags)

    # Assert
    # ---
    expected_agent_record = AgentRecord(
        state=AgentState.END,
        result=return_values,
        log=log,
    )

    expected_tags = [Tags.AGENT_FEEDBACK.value]

    callback.on_agent.assert_called_once_with(
        agent=expected_agent_record,
        tags=expected_tags,
    )


@pytest.mark.asyncio
async def test_on_agent_finish_when_tool_used_should_on_agent() -> None:
    # Arrange
    # ---
    tags = [Tags.AGENT_FEEDBACK.value]

    log = "old log"
    new_log = "new log"
    return_values = {"output": "return value"}
    tool_record = ToolRecord(
        name="action.tool",
        input_str="action.tool_input",
        state=ToolsState.END,
    )

    callback = InspectorAsyncBaseDisplayCallbackHandler()
    callback.on_agent = AsyncMock()
    callback._agent_record = AgentRecord(
        state=AgentState.THOUGHT,
        log=log,  # finish.log,
        tool=tool_record,
    )

    # TODO: See the format of Output of all app
    agent_action = AgentFinish(return_values=return_values, log=new_log)

    # Act
    # ---
    await callback.on_agent_finish(finish=agent_action, tags=tags)

    # Assert
    # ---
    expected_agent_record = AgentRecord(
        state=AgentState.END,
        result=return_values,
        log=new_log,
        tool=tool_record,
    )

    expected_tags = [Tags.AGENT_FEEDBACK.value]

    callback.on_agent.assert_called_once_with(
        agent=expected_agent_record,
        tags=expected_tags,
    )


@pytest.mark.asyncio
async def test_on_agent_finish_when_without_agent_tags_should_not_on_agent() -> None:
    # Arrange
    # ---
    callback = InspectorAsyncBaseDisplayCallbackHandler()
    callback.on_agent = AsyncMock()

    # Act
    # ---

    await callback.on_agent_finish(finish=AsyncMock(), tags=["wrong_tags"])

    # Assert
    # ---
    callback.on_agent.assert_not_called()


##################################################
# AsyncBaseDisplayCallbackHandler Implementated
##################################################


class InspectorAsyncBaseDisplayCallbackHandler(BaseAsyncDisplayCallbackHandler):
    def __init__(
        self,
        should_cumulate_token: bool = False,
        verbose: bool = True,
    ) -> None:
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
        pass
