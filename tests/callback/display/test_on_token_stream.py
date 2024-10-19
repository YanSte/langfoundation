from typing import Any, List
from unittest.mock import AsyncMock

import pytest
from langchain_core.outputs import LLMResult
from langchain_core.outputs.generation import Generation
from pydantic import BaseModel

from langfoundation.callback.base.base import (
    BaseAsyncDisplayCallbackHandler,
)
from langfoundation.callback.base.records.agent import AgentRecord
from langfoundation.callback.base.records.retriever import (
    RetrieverRecord,
)
from langfoundation.callback.base.records.token import (
    TokenOrigin,
    TokenStream,
    TokenStreamState,
)
from langfoundation.callback.base.records.tool import ToolRecord
from langfoundation.callback.base.tags import Tags
from langfoundation.parser.display.parser import DisplayOutputParser

####################################################################################
# State
####################################################################################


@pytest.mark.asyncio
async def test_on_stream_when_on_llm_should_start_stream_end() -> None:
    # Arrange
    # ---
    tags = [Tags.DISPLAY.value]
    callback = InspectorAsyncBaseDisplayCallbackHandler(should_cumulate_token=False)
    callback.on_token_stream = AsyncMock()  # type: ignore[assignment]
    res = LLMResult(generations=[[Generation(text="Hello end")]])

    # Act & Assert
    # ---
    await callback.on_llm_new_token("abs", tags=tags)
    assert callback._token_stream_record.state == TokenStreamState.START  # type: ignore

    # Act & Assert
    # ---
    await callback.on_llm_new_token("abs", tags=tags)
    assert callback._token_stream_record.state == TokenStreamState.STREAM  # type: ignore

    # Act & Assert
    # ---
    await callback.on_llm_new_token("abs", tags=tags)
    assert callback._token_stream_record.state == TokenStreamState.STREAM  # type: ignore

    # Act & Assert
    # ---
    await callback.on_llm_new_token("abs", tags=tags)
    assert callback._token_stream_record.state == TokenStreamState.STREAM  # type: ignore

    # Act & Assert
    # ---
    await callback.on_llm_new_token("", tags=tags)
    assert callback._token_stream_record.state == TokenStreamState.END  # type: ignore

    # Act & Assert
    # ---
    await callback.on_llm_end(response=res, tags=tags, kwargs={})
    assert callback._token_stream_record is None


@pytest.mark.asyncio
async def test_on_non_stream_when_on_llm_start_and_end_should_stream_end() -> None:
    # Arrange
    # ---
    tags = [Tags.DISPLAY.value]
    callback = InspectorAsyncBaseDisplayCallbackHandler(should_cumulate_token=False)
    callback.on_token_stream = AsyncMock()  # type: ignore[assignment]
    res = LLMResult(generations=[[Generation(text="Hello end")]])

    # Act
    # ---
    await callback.on_llm_start(serialized={}, prompts=[], tags=tags)
    await callback.on_llm_end(response=res, tags=tags, kwargs={})

    # Assert
    # ---
    expected_tags = [Tags.DISPLAY.value]
    expected_token = "Hello end"
    expected_token_stream = TokenStream(
        id=1,
        token="",
        cumulate_tokens=expected_token,
        state=TokenStreamState.END,
        origin=TokenOrigin.LLM,
    )
    callback.on_token_stream.assert_called_with(token=expected_token_stream, kwargs={}, tags=expected_tags)
    assert callback._token_stream_record is None


####################################################################################
# On LLM Stream
####################################################################################


@pytest.mark.asyncio
async def test_on_llm_new_token_start_should_on_token_start() -> None:
    """Display when tag view and LLM new Token and START"""
    # Arrange
    # ---
    tags = [Tags.DISPLAY.value]
    callback = InspectorAsyncBaseDisplayCallbackHandler(should_cumulate_token=False)
    callback.on_token_stream = AsyncMock()  # type: ignore[assignment]

    token = "test token"

    # Act
    # ---
    await callback.on_llm_new_token(token, tags=tags)

    # Assert
    # ---
    expected_tags = tags = [Tags.DISPLAY.value]
    expected_token = TokenStream(
        id=0,
        token=token,
        cumulate_tokens=token,
        state=TokenStreamState.START,
        origin=TokenOrigin.LLM,
    )

    callback.on_token_stream.assert_called_once_with(
        token=expected_token,
        tags=expected_tags,
    )


@pytest.mark.asyncio
async def test_on_llm_new_token_stream_should_on_token_stream() -> None:
    """Display when tag view and LLM new Token and STREAM"""

    tags = [Tags.DISPLAY.value]

    callback = InspectorAsyncBaseDisplayCallbackHandler(should_cumulate_token=False)

    callback.on_token_stream = AsyncMock()  # type: ignore[assignment]

    start_token = "Start Token"
    callback._token_stream_record = TokenStream(
        id=1,
        token=start_token,
        cumulate_tokens=start_token,
        state=TokenStreamState.START,
        origin=TokenOrigin.LLM,
    )

    token = " Stream"
    cumulate_tokens = start_token + token

    # Act
    # ---
    await callback.on_llm_new_token(token, tags=tags)

    # Assert
    # ---
    expected_tags = [Tags.DISPLAY.value]
    expected_token_stream = TokenStream(
        id=2,
        token=token,
        cumulate_tokens=cumulate_tokens,
        state=TokenStreamState.STREAM,
        origin=TokenOrigin.LLM,
    )
    callback.on_token_stream.assert_called_once_with(
        token=expected_token_stream,
        tags=expected_tags,
    )
    assert callback._token_stream_record == expected_token_stream


@pytest.mark.asyncio
async def test_on_llm_new_token_end_should_on_token_end() -> None:
    """Display when tag view and LLM new Token and token is empty END"""
    # Arrange
    # ---
    tags = [Tags.DISPLAY.value]

    callback = InspectorAsyncBaseDisplayCallbackHandler(should_cumulate_token=False)
    callback.on_token_stream = AsyncMock()  # type: ignore[assignment]

    cumulate_stream_token = "Stream Token"
    callback._token_stream_record = TokenStream(
        id=1,
        token=cumulate_stream_token,
        cumulate_tokens=cumulate_stream_token,
        state=TokenStreamState.STREAM,
        origin=TokenOrigin.LLM,
    )

    empty_token = ""
    cumulate_tokens = cumulate_stream_token

    # Act
    # ---
    await callback.on_llm_new_token(empty_token, tags=tags)

    # Assert
    # ---
    expected_token_stream = TokenStream(
        id=2,
        token="",
        cumulate_tokens=cumulate_tokens,
        state=TokenStreamState.END,
        origin=TokenOrigin.LLM,
    )
    expected_tags = [Tags.DISPLAY.value]
    callback.on_token_stream.assert_called_once_with(
        token=expected_token_stream,
        tags=expected_tags,
    )


####################################################################################
# Display with Parser
####################################################################################


@pytest.mark.asyncio
async def test_on_llm_new_token_start_parser_should_on_token_start() -> None:
    """Display when tag view and LLM new Token and START"""
    # Arrange
    # ---
    display_parser = DisplayOutputParser(
        pydantic_object=TestModel,
        key="test",
    )

    tags = [Tags.DISPLAY.value, Tags.get_display_tag_with_parser(display_parser)]

    callback = InspectorAsyncBaseDisplayCallbackHandler(should_cumulate_token=False)
    callback.add_display_parser(display_parser)

    callback.on_token_stream = AsyncMock()  # type: ignore[assignment]

    # Token
    token = '{"test": "Test"}'

    # Act
    # ---
    await callback.on_llm_new_token(token=token, tags=tags)

    # Assert
    # ---
    expected_cumulate_tokens = token
    expected_parsed_token = "Test"

    expected_token = TokenStream(
        id=0,
        token=expected_parsed_token,  # Parse on display_mapper
        cumulate_tokens=expected_cumulate_tokens,
        state=TokenStreamState.START,
        origin=TokenOrigin.LLM,
    )

    expected_tags = [
        Tags.DISPLAY.value,
        Tags.get_display_tag_with_parser(display_parser),
    ]

    callback.on_token_stream.assert_called_once_with(
        token=expected_token,
        tags=expected_tags,
    )


@pytest.mark.asyncio
async def test_on_llm_new_token_after_start_with_parser_should_on_token_stream() -> None:
    """Display when tag view and LLM new Token and START"""
    # Arrange
    # ---
    display_parser = DisplayOutputParser(
        pydantic_object=TestModel,
        key="test",
    )

    tags = [Tags.DISPLAY.value, Tags.get_display_tag_with_parser(display_parser)]

    callback = InspectorAsyncBaseDisplayCallbackHandler(should_cumulate_token=False)
    callback.add_display_parser(display_parser)
    callback.on_token_stream = AsyncMock()  # type: ignore[assignment]

    # Act
    # --
    await callback.on_llm_new_token(token='{"test": "T', tags=tags)
    await callback.on_llm_new_token(token="es", tags=tags)
    await callback.on_llm_new_token(token='t"}', tags=tags)

    # Assert
    # ---
    expected_cumulate_tokens = '{"test": "Test"}'
    expected_parsed_token = "t"

    expected_token = TokenStream(
        id=2,
        token=expected_parsed_token,  # Parse on display_mapper
        cumulate_tokens=expected_cumulate_tokens,
        state=TokenStreamState.STREAM,
        origin=TokenOrigin.LLM,
    )

    expected_tags = [
        Tags.DISPLAY.value,
        Tags.get_display_tag_with_parser(display_parser),
    ]

    callback.on_token_stream.assert_called_with(
        token=expected_token,
        tags=expected_tags,
    )


@pytest.mark.asyncio
async def test_on_llm_new_token_start_with_parser_should_on_token_end() -> None:
    # Arrange
    # ---
    display_parser = DisplayOutputParser(
        pydantic_object=TestModel,
        key="test",
    )

    tags = [Tags.DISPLAY.value, Tags.get_display_tag_with_parser(display_parser)]

    callback = InspectorAsyncBaseDisplayCallbackHandler(should_cumulate_token=False)
    callback.add_display_parser(display_parser)
    callback.on_token_stream = AsyncMock()  # type: ignore[assignment]

    # Act
    # --
    await callback.on_llm_new_token(token='{"test": "T', tags=tags)
    await callback.on_llm_new_token(token="es", tags=tags)
    await callback.on_llm_new_token(token='t"}', tags=tags)
    await callback.on_llm_new_token(token="", tags=tags)

    # Assert
    # ---
    expected_cumulate_tokens = '{"test": "Test"}'
    expected_parsed_token = ""

    expected_token = TokenStream(
        id=3,
        token=expected_parsed_token,
        cumulate_tokens=expected_cumulate_tokens,
        state=TokenStreamState.END,
        origin=TokenOrigin.LLM,
    )

    expected_tags = [
        Tags.DISPLAY.value,
        Tags.get_display_tag_with_parser(display_parser),
    ]

    callback.on_token_stream.assert_called_with(
        token=expected_token,
        tags=expected_tags,
    )


####################################################################################
# Error
####################################################################################


@pytest.mark.asyncio
async def test_on_llm_error_when_start_should_on_token_error() -> None:
    # Arrange
    # ---
    tags = [Tags.DISPLAY.value]
    callback = InspectorAsyncBaseDisplayCallbackHandler(should_cumulate_token=False)
    callback.on_token_stream = AsyncMock()  # type: ignore[assignment]
    error = ValueError("My error")

    # Act
    # ---
    await callback.on_llm_error(error=error, tags=tags)

    # Assert
    # ---
    expected_error = error
    expected_tags = [Tags.DISPLAY.value]
    expected_token_stream = TokenStream(
        id=0,
        token="",
        cumulate_tokens="",
        state=TokenStreamState.ERROR,
        origin=TokenOrigin.LLM,
        error=expected_error,
    )
    callback.on_token_stream.assert_called_once_with(
        token=expected_token_stream,
        tags=expected_tags,
    )
    assert callback._token_stream_record is None


@pytest.mark.asyncio
async def test_on_llm_error_when_working_should_on_token_error() -> None:
    # Arrange
    # ---
    tags = [Tags.DISPLAY.value]
    callback = InspectorAsyncBaseDisplayCallbackHandler(should_cumulate_token=False)
    callback.on_token_stream = AsyncMock()  # type: ignore[assignment]
    error = ValueError("My error")

    # Act
    # ---
    await callback.on_llm_new_token("abs", tags=tags)
    await callback.on_llm_new_token("abs", tags=tags)
    await callback.on_llm_new_token("abs", tags=tags)
    await callback.on_llm_new_token("abs", tags=tags)
    await callback.on_llm_error(error=error, tags=tags)

    # Assert
    # ---
    expected_error = error
    expected_tags = [Tags.DISPLAY.value]
    expected_token_stream = TokenStream(
        id=4,
        token="",
        cumulate_tokens="absabsabsabs",
        state=TokenStreamState.ERROR,
        origin=TokenOrigin.LLM,
        error=expected_error,
    )
    callback.on_token_stream.assert_called_with(token=expected_token_stream, tags=expected_tags)
    assert callback._token_stream_record is None


####################################################################################
# Not Display
####################################################################################


@pytest.mark.asyncio
async def test_on_stream_when_not_valid_tags_should_not_stream_token() -> None:
    """Display when tag view and LLM new Token and START"""
    # Arrange
    # ---
    tags: List[str] = []
    callback = InspectorAsyncBaseDisplayCallbackHandler(should_cumulate_token=False)
    callback.on_token_stream = AsyncMock()  # type: ignore[assignment]
    res = LLMResult(generations=[[Generation(text="Hello end")]])

    # Act & Assert
    # ---
    await callback.on_llm_new_token("abs", tags=tags)
    callback.on_token_stream.assert_not_called()

    # Act & Assert
    # ---
    await callback.on_llm_new_token("abs", tags=tags)
    callback.on_token_stream.assert_not_called()

    # Act & Assert
    # ---
    await callback.on_llm_end(response=res, tags=tags, kwargs={})
    callback.on_token_stream.assert_not_called()

    assert callback._token_stream_record is None


@pytest.mark.asyncio
async def test_on_non_stream_when_not_valid_tags_should_not_stream_token() -> None:
    # Arrange
    # ---
    tags: List[str] = []
    callback = InspectorAsyncBaseDisplayCallbackHandler(should_cumulate_token=False)
    callback.on_token_stream = AsyncMock()  # type: ignore[assignment]
    res = LLMResult(generations=[[Generation(text="Hello end")]])

    # Act
    # ---
    await callback.on_llm_end(response=res, tags=tags, kwargs={})

    # Assert
    # ---
    callback.on_token_stream.assert_not_called()
    assert callback._token_stream_record is None


####################################################################################
# AsyncBaseDisplayCallbackHandler Implementated
####################################################################################


class TestModel(BaseModel):
    test: str
    test_value: str


class InspectorAsyncBaseDisplayCallbackHandler(BaseAsyncDisplayCallbackHandler):
    def __init__(
        self,
        should_cumulate_token: bool = False,
        verbose: bool = True,
    ) -> None:
        self.should_cumulate_token = should_cumulate_token
        self.verbose = verbose

    # Token
    # ----
    async def on_token_stream(self, token: TokenStream, **kwargs: Any) -> None:
        """
        Abstract method to handle a stream token event.
        """
        pass

    # Reriever
    # ----

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

    # Tool
    # ----

    async def on_tool(self, tool: ToolRecord, **kwargs: Any) -> None:
        """
        Abstract method to handle a tool event.
        """
        pass

    # Agent
    # ----

    async def on_agent(self, agent: AgentRecord, **kwargs: Any) -> None:
        """
        Abstract method to handle an agent event.
        """
        pass

    async def on_feedback(self, feedback: str, **kwargs: Any) -> None:
        """
        Abstract method to handle a feedback event.
        """
