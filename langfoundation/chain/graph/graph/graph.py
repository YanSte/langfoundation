from __future__ import annotations

from abc import ABC
from typing import TypeVar

from pydantic import BaseModel

from langfoundation.callback.display.tags import Tags
from langfoundation.chain.pydantic.graph import BasePydanticGraphChain


# "GraphState" must be a subclass of BaseModel.
GraphState = TypeVar("GraphState", bound=BaseModel)

# "Input" must be a subclass of BaseModel.
Input = TypeVar("Input", bound=BaseModel)

# "Input" must be a subclass of BaseModel.
Ouput = TypeVar("Ouput", bound=BaseModel)


class BaseGraphChain(
    BasePydanticGraphChain[GraphState, Input, Ouput],
    ABC,
):
    @property
    def is_display(self) -> bool:
        if not self.tags:
            return False

        return Tags.is_display_or_with_parser_tag(self.tags)

    @property
    def is_feedback(self) -> bool:
        if not self.tags:
            return False
        return Tags.FEEDBACK in self.tags
