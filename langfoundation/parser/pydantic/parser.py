# from typing import Any, List


from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, cast

import json_repair
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers.transform import BaseTransformOutputParser
from langchain_core.outputs import Generation
from pydantic import BaseModel, Field

from langfoundation.parser.pydantic.template import PYDANTIC_STREAM_FORMAT_INSTRUCTIONS
from langfoundation.utils.pydantic.base_model import (
    FieldType,
    render_json_schema_with_field_value,
)

Output = TypeVar("Output", bound=BaseModel)


class PydanticOutputParser(BaseTransformOutputParser[Output]):
    """
    Pydantic output parser that can handle streaming input.

    # Note:
    - In your prompt pay attention to the format of the output.
    - Parser accepte only Json with "Key": Value
    - Parser accepte only Json not with '' (example: 'Key': 'Value')
    """

    pydantic_object: Type[Output] = Field(description="The type of the pydantic object that will be used to parse the output.")
    name: str = Field(
        default="PydanticOutputParser",
        description="The name of the parser.",
    )
    """
    A list of types for linked references on the Basemodel `pydantic_object`.
    Make an auto the Json generation of this one.
    """
    basemodel_linked_refs_types: List[Type[BaseModel]] = Field(
        default=[],
        description="""
        A list of types for linked references on the Basemodel `pydantic_object`.
        Make an auto the Json generation of this one.
        """,
    )
    format_instructions: str = Field(
        default=PYDANTIC_STREAM_FORMAT_INSTRUCTIONS,
        description="Format instructions to be displayed in the prompt.",
    )
    pre_transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = Field(
        default=None,
        description="""
        A function to be called before the parsing start to cast the output.
        It should take the output and return a string.
        """,
    )
    post_validation: Optional[Callable[[Output], bool]] = Field(
        default=None,
        description="""
        A function to be called after validation.
        It should take the output and return a boolean.
        """,
    )

    @property
    def _type(self) -> str:
        return "pydantic_stream_output_parser"

    @property
    def OutputType(self) -> Type[Output]:
        """Return the pydantic model."""
        return self.pydantic_object

    def parse(self, text: str) -> Output:
        """
        Parse a pydantic object from a JSON string.

        The string is first cleaned with `json_repair.repair_json` to fix any invalid JSON.
        """
        object = json_repair.repair_json(
            text,
            skip_json_loads=True,
            return_objects=True,
        )
        _dict = cast(Dict, object)

        if self.pre_transform:
            _dict = self.pre_transform(_dict)

        return self._parse_obj(_dict)

    def parse_result(self, result: List[Generation], *, partial: bool = False) -> Output:
        """
        Parse a list of Generation objects into a pydantic model of type `OutputType`.
        """
        text = result[0].text
        result_object = self.parse(text)
        if self.post_validation and not self.post_validation(result_object):
            msg = f"Failed to parse {self.pydantic_object.__name__} doesn't pass the post validation"
            e = ValueError(msg)
            self._parser_exception(e, dict(result_object))

        return result_object

    def get_format_instructions(self) -> str:
        """
        Provides format instructions for the output of the model base on the Field description.

        Model Definition:
        ```python
        class MockModel(BaseModel):
            property1: str = Field(description="Property 1")
            property2: List[Dict[str, str]] = Field(
                description="A dict of str key and value"
            )
        ```python

        Returns a string describing output format:
        ```
        ## Output Format
        The output require to be formatted as a JSON Object and must conform to the JSON schema below:
        '''json
        {
            "property1": "Property 1",
            "property2": "A dict of str key and value",
        }
        '''
        ```
        """
        formatted_schema = render_json_schema_with_field_value(
            render_type=FieldType.DESCRIPTION,
            basemodel_type=self.pydantic_object,
            basemodel_linked_refs_types=self.basemodel_linked_refs_types,
        )
        return self.format_instructions.format(schema=formatted_schema)

    def get_format_value_instructions(self) -> str:
        """
        Provides format instructions for the output of the model base on the Field default.

        Model Definition:
        ```python
        class MockModel(BaseModel):
            property1: str = Field(default="Property 1")
            property2: List[Dict[str, str]] = Field(
                default={
                    {
                        "subproperty1": "Subproperty 1",
                        "subproperty2": "Subproperty 2"
                    }
                }
            )
        ```python

        Returns a string describing output format:
        ```
        ## Output Format
        The output require to be formatted as a JSON Object and must conform to the JSON schema below:
        '''json
        {
            "property1": "Property 1",
            "property2": {
                "subproperty1": "Subproperty 1",
                "subproperty2": "Subproperty 2",
            },
        }
        '''
        ```
        """
        formatted_schema = render_json_schema_with_field_value(
            render_type=FieldType.DESCRIPTION,
            basemodel_type=self.pydantic_object,
            basemodel_linked_refs_types=self.basemodel_linked_refs_types,
        )
        return self.format_instructions.format(schema=formatted_schema)

    def _parser_exception(self, e: Exception, json_object: dict) -> OutputParserException:
        json_string = json.dumps(json_object)
        name = self.pydantic_object.__name__
        msg = f"Failed to parse {name} from completion {json_string}. Got: {e}"
        return OutputParserException(msg, llm_output=json_string)

    def _parse_obj(self, obj: dict) -> Output:
        try:
            return self.pydantic_object.model_validate(obj)
        except Exception as e:
            raise self._parser_exception(e, obj)
