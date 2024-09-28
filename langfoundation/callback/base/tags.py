from __future__ import annotations

from enum import auto
import re
from typing import List, Optional

from strenum import UppercaseStrEnum

from langfoundation.parser.display.parser import DisplayOutputParser


class Tags(UppercaseStrEnum):
    """Enum class for various tag display"""

    DISPLAY = auto()
    """DISPLAY is for displaying the chain returns to user"""

    FEEDBACK = auto()
    """FEEDBACK is a general tag for all returns as feedback"""

    TOOL_FEEDBACK = auto()

    AGENT_FEEDBACK = auto()
    """AGENT_FEEDBACK is for returns specific to agent feedback, for example, reasons provided by the agent"""

    RETRIEVER_FEEDBACK = auto()
    """RETRIEVER_FEEDBACK is for returns specific to retrieval feedback, such as retrieved chunks"""

    @staticmethod
    def get_display_tag_with_parser(display_parser: DisplayOutputParser) -> str:
        """
        Get Display tag with a spefic parser.
        """
        pydantic_object = display_parser.pydantic_object
        if not pydantic_object:
            raise ValueError()
        name = pydantic_object.__name__  # type: ignore
        return (_Tags_Constants._DISPLAY + name + _Tags_Constants._WITH_PARSER).upper()

    @staticmethod
    def is_not_display_or_with_parser_tag(tags: List[str]) -> bool:
        """Check if tags do not contain display or display with parser tags."""

        return not Tags.is_display_or_with_parser_tag(tags)

    @staticmethod
    def is_display_or_with_parser_tag(tags: List[str]) -> bool:
        """Check if tags contain display or display with parser tags."""
        for tag in tags:
            if tag == Tags.DISPLAY or re.match(_Tags_Constants._PATTERN, tag):
                return True
        return False

    @staticmethod
    def get_non_display_tag(tags: List[str]) -> Optional[List[str]]:
        tags = tags.copy()

        for tag in tags:
            if tag == Tags.DISPLAY or re.match(_Tags_Constants._PATTERN, tag):
                tags.remove(tag)

        if len(tags) == 0:
            return None
        else:
            return tags


class _Tags_Constants:
    """Constants used for tag generation and pattern matching."""

    _WITH_PARSER = "_WITH_PARSER_"
    _DISPLAY = "DISPLAY_"
    _PATTERN = "^" + re.escape(_DISPLAY) + ".*" + re.escape(_WITH_PARSER) + ".*$"
