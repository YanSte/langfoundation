from __future__ import annotations

import logging
from typing import Callable, Optional, Type, Union

from langchain.agents.agent import AgentOutputParser as LangchainAgentOutputParser
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.exceptions import OutputParserException
from langchain_core.utils.json import parse_json_markdown

from langfoundation.parser.agent.model import AgentState

logger = logging.getLogger(__name__)


class AgentOutputParser(LangchainAgentOutputParser):
    """Output parser for the structured chat agent."""

    name = "AgentOutputParser"
    pydantic_object: Type[AgentState] = AgentState
    pre_parse_cleaning: Optional[Callable[[str], str]] = None
    return_values_key = "output"
    verbose: bool = False

    @property
    def _type(self) -> str:
        return __class__.__name__

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        if self.verbose:
            logger.info(
                text,
                extra={"title": "[Parse] Start" + " : " + self.__class__.__name__},
            )

        try:
            text = text.strip()

            if self.pre_parse_cleaning:
                text = self.pre_parse_cleaning(text)

            json_object = parse_json_markdown(text)
            state = self.pydantic_object(**json_object)
            if self.verbose:
                logger.info(
                    state,
                    extra={"title": "[Parse] Parsed Json" + " : " + self.__class__.__name__},
                )

            if not state.action:
                return AgentFinish(
                    {self.return_values_key: state.action_input},
                    state.thought,
                )
            else:
                return AgentAction(
                    state.action,
                    state.action_input,
                    state.thought,
                )

        except Exception as e:
            raise OutputParserException(f"Could not parse LLM output: {text} error:{e}")

    def display_parse(self, text: str) -> Optional[str]:
        if self.verbose:
            logger.info(
                text,
                extra={"title": "[Display] text" + " : " + self.__class__.__name__},
            )

        try:
            agent_state = self.parse(text)

            if isinstance(agent_state, AgentFinish):
                agent_finish = agent_state
                return agent_finish.return_values[self.return_values_key]
            else:
                return None

        except Exception:
            return None
