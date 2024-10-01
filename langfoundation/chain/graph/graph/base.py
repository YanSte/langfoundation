from __future__ import annotations

from abc import ABC
from typing import List, Optional, TypeVar

from langfoundation.callback.base.tags import Tags
from langfoundation.chain.pydantic.graph import BasePydanticGraphChain
from pydantic import BaseModel

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

    def get_non_display_tags(self) -> Optional[List[str]]:
        if not self.tags:
            return None
        return Tags.get_non_display_tag(self.tags)
