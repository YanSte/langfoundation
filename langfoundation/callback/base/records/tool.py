from __future__ import annotations

from enum import auto
from typing import Optional, Union

from strenum import LowercaseStrEnum

from langfoundation.callback.base.records.base import BaseRecord


class ToolsState(LowercaseStrEnum):
    REQUEST = auto()
    # The LLM is thinking about what to do next. We don't know which tool we'll run.
    START = auto()
    # The LLM has decided to run a tool. We don't have results from the tool yet.
    END = auto()
    # The LLM completed with an error.
    ERROR = auto()


class ToolRecord(BaseRecord):
    """
    Represents a tool record.

    Attributes:
        name (str): The name of the tool.
        input_str (Union[str, dict]): The input given to the tool.
        state (ToolsState): The current state of the tool.
        output (Optional[str]): The output of the tool, if any.
        error (Optional[BaseException]): Any error that occurred while running the tool, if any.
    """

    name: str
    input_str: Union[str, dict]
    state: ToolsState
    output: Optional[str] = None
    error: Optional[BaseException] = None

    class Config:
        arbitrary_types_allowed = True

    def get_tool_label(self, max_tool_input_length: int = 100) -> str:
        """
        Return the label for an LLMThought that has an associated tool.

        Args:
            max_tool_input_length (int, optional): Maximum length of the tool input string to include in the label. Defaults to 100.

        Returns:
            str: The tool label.
        """
        name_output = self.name
        input_output = self.input_str

        if self.name == "_Exception":
            name_output = "Parsing error"

        input_str_len = min(
            max_tool_input_length,
            len(self.input_str),
        )

        input_output = self.input_str[:input_str_len]
        if len(self.input_str) > input_str_len:
            input_output = input_output + "..."

        input_output = input_output.replace("\n", " ")

        return f"**{name_output}:** {input_output}"
