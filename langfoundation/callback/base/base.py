from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence
from uuid import UUID

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult

from langfoundation.callback.base.records.agent import AgentRecord, AgentState
from langfoundation.callback.base.records.retriever import RetrieverRecord, RetrieverState
from langfoundation.callback.base.records.token import TokenOrigin, TokenStream, TokenStreamState
from langfoundation.callback.base.records.tool import ToolRecord, ToolsState
from langfoundation.callback.base.tags import Tags
from langfoundation.parser.display.parser import DisplayOutputParser
from langfoundation.utils.debug.formatter import Formatter


class BaseAsyncDisplayCallbackHandler(AsyncCallbackHandler, ABC):
    """
    A callback handler for asynchronous display events.

    This class provides methods to handle various events related to display actions,
    such as LLM (Language Model) events, tool events, retriever events, and agent events.
    """

    display_tags_parsers: Dict[str, DisplayOutputParser] = {}
    should_cumulate_token: bool = True

    _agent_record: Optional[AgentRecord] = None
    _retriever_record: Optional[RetrieverRecord] = None
    _token_stream_record: Optional[TokenStream] = None

    verbose: bool = False

    ############################
    # Abstract
    ############################

    @abstractmethod
    async def on_token_stream(self, token: TokenStream, **kwargs: Any) -> None:
        """
        Abstract method to handle a stream token event.
        """

    @abstractmethod
    async def on_retriever(self, retriever: RetrieverRecord, **kwargs: Any) -> None:
        """
        Abstract method to handle a retriever event.
        """

    @abstractmethod
    def non_async_on_retriever(self, retriever: RetrieverRecord, **kwargs: Any) -> None:
        """
        Abstract method to handle a retriever event.
        """

    @abstractmethod
    async def on_tool(self, tool: ToolRecord, **kwargs: Any) -> None:
        """
        Abstract method to handle a tool event.
        """

    @abstractmethod
    async def on_agent(self, agent: AgentRecord, **kwargs: Any) -> None:
        """
        Abstract method to handle an agent event.
        """

    @abstractmethod
    async def on_feedback(self, feedback: str, **kwargs: Any) -> None:
        """
        Abstract method to handle a feedback event.
        """

    ############################
    # Start
    ############################
    def _display_encapsule(self, msg: str) -> str:
        return "```\n" + msg.replace("```", "'''") + "\n```"

    async def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Any:
        if Tags.FEEDBACK.value in tags:
            if messages and messages[0]:
                messages_str = "".join([message.pretty_repr() for message in messages[0]])
                feedback = (
                    "Chat Start:\n\n"
                    + self._display_encapsule(serialized["name"])
                    + "\n\nPrompt:\n\n"
                    + self._display_encapsule(messages_str)
                )
                await self.on_feedback(feedback=feedback, **kwargs)

    async def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Handle the start of an LLM event.
        """
        if Tags.FEEDBACK.value in tags:  # type: ignore
            if prompts:
                repr = serialized["repr"]
                prompts_str = "".join(prompts)
                feedback = "\n\nLLM Start:\n\n" + self._display_encapsule(repr) + "\n\nPrompt:\n\n" + self._display_encapsule(prompts_str)
                await self.on_feedback(feedback=feedback, **kwargs)

    ############################
    # LLM
    ############################

    async def on_llm_new_token(
        self,
        token: str,
        tags: List[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Handle a new token in the LLM event.
        """
        tags = tags or []
        if Tags.is_not_display_or_with_parser_tag(tags):
            return

        kwargs["tags"] = tags

        # STATE: START
        if self._token_stream_record is None:
            cleaned_token = token.strip()
            self._token_stream_record = TokenStream(
                id=0,
                token=cleaned_token,
                cumulate_tokens=cleaned_token,
                state=TokenStreamState.START,
                origin=TokenOrigin.LLM,
            )

        # STATE: END (Empty token)
        elif not token:
            self._token_stream_record.update(
                id=self._token_stream_record.id + 1,
                token="",
                cumulate_tokens=self._token_stream_record.cumulate_tokens,
                state=TokenStreamState.END,
            )

        # STATE: STREAM
        else:
            self._token_stream_record.update(
                id=self._token_stream_record.id + 1,
                token=token,
                cumulate_tokens=self._token_stream_record.cumulate_tokens + token,
                state=TokenStreamState.STREAM,
            )

        # CASE: PARSER
        token_stream: Optional[TokenStream] = self._token_stream_record

        if display_parser := self._find_display_parser_by_tags(tags):
            token_stream = display_parser.display_parse(self._token_stream_record)

        # CASE: OUPUT
        if token_stream:
            await self.on_token_stream(
                token=token_stream,
                **kwargs,
            )

    async def on_llm_end(
        self,
        response: LLMResult,
        tags: List[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Handle the end of an LLM event.
        """
        tags = tags or []
        if not response.generations or not response.generations[0]:
            self._token_stream_record = None
            return

        final_text = response.generations[0][0].text
        kwargs["tags"] = tags

        # Feedback
        # ---
        if Tags.FEEDBACK.value in tags:
            final_text_feedback = ""
            if final_text:
                final_text_feedback = "\n\nText:\n\n" + self._display_encapsule(final_text)

            function_calling_tools_display = ""
            function_calling_tools = response.generations[0][0].message.tool_calls  # type: ignore
            if function_calling_tools:
                function_calling_tools_display = "\n\nFunction Calling:\n\n" + self._display_encapsule(
                    Formatter.format(function_calling_tools)
                )

            feedback = "LLM End:\n\n" + final_text_feedback + function_calling_tools_display
            await self.on_feedback(feedback, **kwargs)

        # Case of no Streaming
        # ---
        if Tags.is_display_or_with_parser_tag(tags) and self._token_stream_record is None:
            # State Start
            await self.on_llm_new_token(token=final_text, **kwargs)
            # State END
            await self.on_llm_new_token(token="", **kwargs)

        # Reset
        # ---
        self._token_stream_record = None

    async def on_llm_error(
        self,
        error: BaseException,
        tags: List[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        tags = tags or []

        kwargs["tags"] = tags

        if Tags.FEEDBACK.value in tags:
            description = str(error.args) if error.args else "None"
            feedback = self._display_encapsule(description)
            await self.on_feedback(feedback=feedback, **kwargs)

        if Tags.is_not_display_or_with_parser_tag(tags):
            return
        # During a Stream
        if not self._token_stream_record:
            self._token_stream_record = TokenStream(
                id=0,
                token="",
                cumulate_tokens="",
                state=TokenStreamState.ERROR,
                origin=TokenOrigin.LLM,
                error=error,
            )
        # During at Start
        else:
            self._token_stream_record.update(
                id=self._token_stream_record.id + 1,
                token="",
                error=error,
                state=TokenStreamState.ERROR,
            )

        await self.on_token_stream(
            token=self._token_stream_record,
            **kwargs,
        )

        self._token_stream_record = None

    ############################
    # Chain
    ############################

    async def on_chain_error(
        self,
        error: BaseException,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when chain errors."""
        tags = tags or []

        kwargs["tags"] = tags

        if Tags.FEEDBACK.value in tags:
            description = str(error.args) if error.args else "None"
            feedback = self._display_encapsule(description)
            await self.on_feedback(feedback=feedback, **kwargs)

        if Tags.is_not_display_or_with_parser_tag(tags):
            return

        # During a Stream
        if not self._token_stream_record:
            self._token_stream_record = TokenStream(
                id=0,
                token="",
                cumulate_tokens="",
                state=TokenStreamState.ERROR,
                origin=TokenOrigin.LLM,
                error=error,
            )
        # During at Start
        else:
            self._token_stream_record.update(
                id=self._token_stream_record.id + 1,
                token="",
                error=error,
                state=TokenStreamState.ERROR,
            )

        await self.on_token_stream(
            token=self._token_stream_record,
            **kwargs,
        )

        self._token_stream_record = None

    ############################
    # Tools
    ############################

    async def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        tags: List[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Handle the start of a tool event.
        """
        if Tags.TOOL_FEEDBACK.value not in tags:  # type: ignore
            return

        if self._agent_record is None or self._agent_record.tool is None:
            raise ValueError("agent record must not be None with the tool.")

        kwargs["serialized"] = serialized
        kwargs["tags"] = tags

        if self._agent_record.tool.state is not ToolsState.ERROR:
            self._agent_record.tool.update(state=ToolsState.START)

        await self.on_tool(self._agent_record.tool, **kwargs)

    async def on_tool_end(
        self,
        output: str,
        tags: List[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Handle the end of a tool event.
        """
        if Tags.TOOL_FEEDBACK.value not in tags:  # type: ignore
            return

        if self._agent_record is None or self._agent_record.tool is None:
            raise ValueError("agent record must not be None with the tool.")

        kwargs["tags"] = tags

        self._agent_record.tool.update(
            output=output,
            state=(ToolsState.ERROR if self._agent_record.tool.state == ToolsState.ERROR else ToolsState.END),
        )

        await self.on_tool(self._agent_record.tool, **kwargs)

        self._agent_record.tool = None

    async def on_tool_error(
        self,
        error: BaseException,
        tags: List[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Handle an error in a tool event.
        """
        if Tags.TOOL_FEEDBACK.value not in tags:  # type: ignore
            return

        if self._agent_record is None or self._agent_record.tool is None:
            raise ValueError("agent record must not be None with the tool.")

        kwargs["tags"] = tags

        self._agent_record.tool.update(state=ToolsState.ERROR, error=error)

        await self.on_tool(self._agent_record.tool, **kwargs)

    ############################
    # Agent
    ############################

    async def on_agent_action(
        self,
        action: AgentAction,
        tags: List[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Handle an agent action event.
        """
        if Tags.AGENT_FEEDBACK.value not in tags:  # type: ignore
            return

        kwargs["tags"] = tags

        tool_record = ToolRecord(
            name=action.tool,
            input_str=action.tool_input,
            state=ToolsState.REQUEST,
        )
        self._agent_record = AgentRecord(
            state=AgentState.THOUGHT,
            tool=tool_record,
            log=action.log,
        )
        # Required: Agent first
        await self.on_agent(agent=self._agent_record, **kwargs)
        # Required: Tool seconde
        await self.on_tool(tool=tool_record, **kwargs)

    async def on_agent_finish(
        self,
        finish: AgentFinish,
        tags: List[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Handle the finish of an agent event.
        """
        if Tags.AGENT_FEEDBACK.value not in tags:  # type: ignore
            return

        kwargs["tags"] = tags

        if self._agent_record is None:
            self._agent_record = AgentRecord(state=AgentState.END, log=finish.log, result=finish.return_values)
        else:
            self._agent_record.update(state=AgentState.END, log=finish.log, result=finish.return_values)

        await self.on_agent(agent=self._agent_record, **kwargs)

        self._agent_record = None

    ############################
    # Retriever
    ############################

    # Async
    # ---
    async def on_retriever_start(
        self,
        serialized: Dict[str, Any],
        query: str,
        tags: List[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Handle the start of a retriever event.
        """
        if Tags.RETRIEVER_FEEDBACK.value not in tags:  # type: ignore
            return

        kwargs["tags"] = tags

        self._retriever_record = RetrieverRecord(
            query=query,
            state=RetrieverState.START,
            documents=None,
        )

        await self.on_retriever(self._retriever_record, **kwargs)

    async def on_retriever_end(
        self,
        documents: Sequence[Document],
        tags: List[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Handle the end of a retriever event.
        """
        if Tags.RETRIEVER_FEEDBACK.value not in tags:  # type: ignore
            return

        if self._retriever_record is None:
            raise ValueError("retriever record must not be None.")

        kwargs["tags"] = tags

        self._retriever_record.update(documents=documents, state=RetrieverState.END)

        await self.on_retriever(self._retriever_record, **kwargs)

        self._retriever_record = None

    # Non Async
    # ---
    def non_async_on_retriever_start(
        self,
        query: str,
        *,
        tags: List[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Handle the start of a retriever event.
        """
        if Tags.RETRIEVER_FEEDBACK.value not in tags:  # type: ignore
            return

        kwargs["tags"] = tags

        self._retriever_record = RetrieverRecord(
            query=query,
            state=RetrieverState.START,
            documents=None,
        )

        self.non_async_on_retriever(self._retriever_record, **kwargs)

    def non_async_on_retriever_end(
        self,
        documents: Sequence[Document],
        *,
        tags: List[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Handle the end of a retriever event.
        """

        if Tags.RETRIEVER_FEEDBACK.value not in tags:  # type: ignore
            return

        if self._retriever_record is None:
            raise ValueError("retriever record must not be None.")

        kwargs["documents"] = documents
        kwargs["tags"] = tags

        self._retriever_record.update(documents=documents, state=RetrieverState.END)

        self.non_async_on_retriever(self._retriever_record, **kwargs)

        self._retriever_record = None

    # Display Parser
    # ---

    def add_display_parser(self, parser: DisplayOutputParser) -> None:
        tag_name = parser.tag
        self.display_tags_parsers[tag_name] = parser

    def add_display_parsers(self, parsers: List[DisplayOutputParser]) -> None:
        for parser in parsers:
            self.add_display_parser(parser)

    # Helpers
    # ---

    def _find_display_parser_by_tags(self, tags: List[str]) -> Optional[DisplayOutputParser]:
        """Get the first occurrence of BaseDisplayOutParser for a tags."""
        for tag in tags:
            if tag in self.display_tags_parsers:
                return self.display_tags_parsers[tag]
        return None
