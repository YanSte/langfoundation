from __future__ import annotations

from typing import Any, Optional, TypeVar

from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field, PrivateAttr

from langfoundation.callback.base.records.token import (
    TokenStream,
    TokenStreamState,
)


TBaseModel = TypeVar("TBaseModel", bound=BaseModel)


class DisplayOutputParser(JsonOutputParser):
    """
    A display output parser that parse a JsonOutput based on a key and
    return the value of the key.

    The parser also store the previous value of the key and only return
    the updated value if the value has changed.

    The parser will return None if the value is empty or if there is an
    error during the parsing.
    """

    key: str = Field("Mapper of Pydantic object, use ")
    previous: Optional[str] = None
    _id: int = PrivateAttr(default=0)

    def parse(self, text: str) -> Any:
        """
        Parse the text and return the value of the key.
        """
        json_object = super().parse(text)
        return json_object.get(self.key)

    def display_parse(self, token_stream: TokenStream) -> Optional[TokenStream]:
        """
        Parse the token stream and return a new token stream with the updated value.
        """
        try:
            display_output = self.parse(token_stream.cumulate_tokens)
            if not display_output:
                return None

            display_output = str(display_output)

            if self.previous is None:
                self.previous = display_output
                token_stream.update(
                    id=0,
                    token=display_output,
                    state=TokenStreamState.START,
                )
                return token_stream

            else:
                diff = display_output.replace(self.previous, "")
                if diff:
                    self.previous = display_output
                    self._id += 1
                    token_stream.update(
                        id=self._id,
                        token=diff,
                    )
                    return token_stream
                else:
                    if token_stream.state == TokenStreamState.END:
                        return token_stream
                    else:
                        return None
        except Exception:
            return None

    @property
    def tag(self) -> str:
        """
        Get the tag of the parser.
        """
        from langfoundation.callback.base.tags import Tags

        return Tags.get_display_tag_with_parser(self)
