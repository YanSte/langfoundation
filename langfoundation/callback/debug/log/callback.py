from __future__ import annotations

import logging
from typing import Any, Dict, List, Sequence, Set

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
from langfoundation.utils.debug.formatter import Formatter

from langfoundation.callback.debug.log.type import AsyncLogHandlerType


logger = logging.getLogger(None)


class AsyncLogEventHandler(AsyncCallbackHandler):
    """A class to handle events and log them."""

    _pretty = Formatter()
    _separator: str
    _semi_separator: str
    _included_log: Set[AsyncLogHandlerType]
    _included_kwarg: bool

    def __init__(
        self,
        included_log: Set[AsyncLogHandlerType] = set(),
        included_kwarg: bool = True,
        separator: str = "- - - " * 20,
        semi_separator: str = "-" * 10,
    ):
        self._included_log = included_log
        self._separator = separator
        self._included_kwarg = included_kwarg
        self._semi_separator = semi_separator

    # Chain
    # ----
    async def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        **kwargs: Any,
    ) -> None:
        """Run when the chat model starts."""
        if AsyncLogHandlerType.ON_LLM_START not in self._included_log:
            return
        kwargs.update(serialized)
        kwargs.update(messages=serialized)

        representation = ""
        for message_list in messages:
            message_reprs = [f"{msg.type.title()}: {msg.content}" for msg in message_list]
            representation += "\n".join(message_reprs)

        self._log_message(
            title="On Chat model Start",
            subtitle="Prompts Representation",
            data=representation,
            **kwargs,
        )

    async def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        **kwargs: Any,
    ) -> None:
        if AsyncLogHandlerType.ON_LLM_START not in self._included_log:
            return
        kwargs.update(serialized)
        self._log_message(
            title="On LLM Start",
            subtitle="Prompts Representation",
            data=prompts,
            **kwargs,
        )

    async def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> None:
        if AsyncLogHandlerType.ON_CHAIN_START not in self._included_log:
            return
        kwargs.update(serialized)
        self._log_message(title="On Chain Start", subtitle="Inputs", data=inputs, **kwargs)

    async def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        if AsyncLogHandlerType.ON_CHAIN_END not in self._included_log:
            return
        self._log_message(title="On Chain End", subtitle="Outputs", data=outputs, **kwargs)

    async def on_chain_error(self, error: BaseException, **kwargs: Any) -> Any:
        if AsyncLogHandlerType.ON_CHAIN_ERROR not in self._included_log:
            return
        self._log_message(
            title="On Chain Error",
            subtitle="Error",
            data=error,
            **kwargs,
        )

    async def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        if AsyncLogHandlerType.ON_LLM_ERROR not in self._included_log:
            return
        self._log_message(title="On LLM Error", subtitle="Error", data=error, **kwargs)

    async def on_llm_new_token(
        self,
        token: str,
        **kwargs: Any,
    ) -> None:
        if AsyncLogHandlerType.ON_LLM_NEW_TOKEN not in self._included_log:
            return
        self._log_message(title="On LLM New Token", subtitle="Token", data=token, **kwargs)

    async def on_llm_end(
        self,
        response: LLMResult,
        **kwargs: Any,
    ) -> None:
        if AsyncLogHandlerType.ON_LLM_END not in self._included_log:
            return
        self._log_message(title="On LLM End", subtitle="Response", data=response, **kwargs)

    async def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        if AsyncLogHandlerType.ON_TOOL_START not in self._included_log:
            return
        kwargs["serialized"] = serialized
        self._log_message(title="On Tool Start", subtitle="Input", data=input_str, **kwargs)

    async def on_tool_end(
        self,
        output: Any,
        **kwargs: Any,
    ) -> None:
        if AsyncLogHandlerType.ON_TOOL_END not in self._included_log:
            return
        self._log_message(title="On Tool End", subtitle="Ouput", data=output, **kwargs)

    async def on_tool_error(
        self,
        error: BaseException,
        **kwargs: Any,
    ) -> None:
        if AsyncLogHandlerType.ON_TOOL_ERROR not in self._included_log:
            return
        self._log_message(title="On Tool Error", subtitle="Error", data=error, **kwargs)

    async def on_retriever_start(
        self,
        serialized: Dict[str, Any],
        query: str,
        **kwargs: Any,
    ) -> None:
        if AsyncLogHandlerType.ON_RETRIEVER_START not in self._included_log:
            return
        kwargs["serialized"] = serialized
        self._log_message(title="On Retriever Start", subtitle="Query", data=query, **kwargs)

    async def on_retriever_end(
        self,
        documents: Sequence[Document],
        **kwargs: Any,
    ) -> None:
        if AsyncLogHandlerType.ON_RETRIEVER_END not in self._included_log:
            return
        self._log_message(title="On Retriever End", subtitle="Documents", data=documents, **kwargs)

    async def on_agent_action(
        self,
        action: AgentAction,
        **kwargs: Any,
    ) -> None:
        if AsyncLogHandlerType.ON_AGENT_ACTION not in self._included_log:
            return
        self._log_message(title="On Agent Action", subtitle="Action", data=action, **kwargs)

    async def on_agent_finish(
        self,
        finish: AgentFinish,
        **kwargs: Any,
    ) -> None:
        if AsyncLogHandlerType.ON_AGENT_FINISH not in self._included_log:
            return
        self._log_message(title="On Agent Finish", subtitle="Finish", data=finish, **kwargs)

    def _log_message(
        self,
        title: str,
        subtitle: str,
        data: Any = None,
        **kwargs: Any,
    ) -> None:
        if isinstance(data, Exception):
            logger.error(
                [data, kwargs],
                extra={
                    "title": "[ERROR] " + title + " : " + subtitle,
                },
            )

        else:
            logger.info(
                [data, kwargs],
                extra={
                    "title": title + " : " + subtitle,
                    "verbose": True,
                },
            )
