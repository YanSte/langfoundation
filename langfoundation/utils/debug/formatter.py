from __future__ import annotations

import json
import logging
from typing import Any, Dict

from pydantic import BaseModel


logger = logging.getLogger(__name__)


class Formatter:
    @classmethod
    def format(cls, value: Any, indent: int = 0) -> str:
        return Formatter().format_any(value, indent)

    @classmethod
    def dump(cls, value: Any, indent: int = 0) -> None:
        logger.debug(Formatter().format_any(value, indent))

    def __init__(self) -> None:
        self.types: Dict = {}
        self.indent_char = "  "
        self.line_ending_char = "\n"
        self.set_formatter(BaseModel, self._format_base_model)
        self.set_formatter(str, self._format_json_string)
        self.set_formatter(dict, self._format_dict)
        self.set_formatter(list, self._format_list)
        self.set_formatter(tuple, self._format_tuple)
        self.set_formatter(object, self._format_object)

    def set_formatter(self, obj: Any, callback: Any) -> None:
        self.types[obj] = callback

    def format_any(self, value: Any, indent=0) -> str:
        for type_formatter, formatter in self.types.items():
            if isinstance(value, type_formatter):
                return formatter(value, indent)
        return self._format_object(value, indent)

    def _format_object(self, value: Any, indent: int) -> str:
        return repr(value)

    def _format_dict(self, value: Any, indent: int) -> str:
        items = [
            f"{self.line_ending_char}{self.indent_char * (indent + 1)}"
            f"{repr(key)}: {self.types.get(type(value[key]), self.format_any)(value[key], indent + 1)}"
            for key in value
        ]
        return "{" + ",".join(items) + f"{self.line_ending_char}{self.indent_char * indent}" + "}"

    def _format_list(self, value: Any, indent: int) -> str:
        items = [
            f"{self.line_ending_char}{self.indent_char * (indent + 1)}" f"{self.types.get(type(item), self.format_any)(item, indent + 1)}"
            for item in value
        ]
        return "[" + ",".join(items) + f"{self.line_ending_char}{self.indent_char * indent}" + "]"

    def _format_tuple(self, value: Any, indent: int) -> str:
        items = [
            f"{self.line_ending_char}{self.indent_char * (indent + 1)}" f"{self.types.get(type(item), self.format_any)(item, indent + 1)}"
            for item in value
        ]
        return "(" + ",".join(items) + f"{self.line_ending_char}{self.indent_char * indent}" + ")"

    def _format_json_string(self, value: str, indent: int) -> str:
        try:
            json_dict = json.loads(value)
            return self._format_dict(json_dict, indent)
        except Exception:
            return self._format_object(value, indent)

    def _format_base_model(self, value: BaseModel, indent: int) -> str:
        _dict = self._format_dict(dict(value), indent)
        _dict = value.__class__.__name__ + _dict.replace('"', "").replace(":", "=").replace("{", "(").replace("}", ")")
        return _dict
