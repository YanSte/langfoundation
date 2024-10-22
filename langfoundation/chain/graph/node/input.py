from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from langchain_core.prompts.chat import BaseMessagePromptTemplate, ChatPromptTemplate
from pydantic import BaseModel

from langfoundation.chain.graph.node.partial import Partial
from langfoundation.chain.graph.node.template import (
    BASE_THINK_STEP_BY_STEP_HUMAIM_PROMPT_TEMPLATE,
)


class BaseInput(BaseModel, ABC):
    """
    Base input for node.
    """

    @property
    @abstractmethod
    def prompt_template(self) -> ChatPromptTemplate:
        """
        Prompt template for call.

        This method should return a prompt template for a call.
        """
        raise NotImplementedError("Prompt template not implemented")

    @property
    @abstractmethod
    def retry_prompt_template(self) -> ChatPromptTemplate:
        """
        Prompt template for call for retry.

        This method should return a prompt template for retrying a call.
        """
        raise NotImplementedError("Retry Prompt template not implemented")

    @abstractmethod
    def prompt_arg(self, errors: Optional[List[Exception]]) -> Partial:
        """
        This method should return a partial object that contains all the data
        needed to generate a prompt for a call or retrying a call.

        - `errors`: When a retry, the `errors` parameter should contain the list of errors that have occurred.
        """

    @property
    def step_by_step_prompt_template(self) -> BaseMessagePromptTemplate:
        return BASE_THINK_STEP_BY_STEP_HUMAIM_PROMPT_TEMPLATE

    extra: Dict[str, Any] = {}

    class Config:
        arbitrary_types_allowed = True
        extra = "forbid"
