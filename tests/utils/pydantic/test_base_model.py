from __future__ import annotations

import json
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from langfoundation.utils.pydantic.base_model import FieldType, render_json_schema_with_field_value


class SelectedTool(BaseModel):
    raison: str = Field(
        description="In short explain, step-by-step, your thinking for " "this chosen N tool you choose.",
        default="value",
    )
    name: str = Field(
        description="Name of the N tool.",
        default="value",
    )
    arguments: Dict[str, Any] = Field(
        description="The arguments for the N tool, a dictionnary type of Dict[str, Any].",
        default={"value": "value"},
    )


class UseToolsWithDesriptionAndType(BaseModel):
    tools: List[SelectedTool] = Field(description="The tools used.")
    value: int = Field(description="this int")
    value2: int = Field(description="this int3")


class UseToolsWithCustomKeyField(BaseModel):
    tools: List[SelectedTool] = Field(
        description="The tools used.",
    )
    value: int = Field(
        description="this int",
    )
    value2: int = Field(
        description="this int3",
    )


class UseToolsWithDesriptionAndValue(BaseModel):
    tools: List[SelectedTool] = Field(description="The tools used.", default=[SelectedTool(name=">>custom<<")])
    value: int = Field(description="this int")
    value2: int = Field(description="this int3")


class UseToolsWithDesription(BaseModel):
    tools: List[SelectedTool] = Field(description="The tools used.")


class UseToolsWithValue(BaseModel):
    tools: List[SelectedTool] = Field(description="The tools used.")


class UseToolsWithNoExample(BaseModel):
    tools: List[SelectedTool] = Field(description="The tools used.")


def test_formatting_when_render_description_should_provide_format_on_description() -> None:
    # Arrange
    formatted_schema = render_json_schema_with_field_value(
        render_type=FieldType.DESCRIPTION,
        basemodel_type=UseToolsWithDesription,
        basemodel_linked_refs_types=[SelectedTool],
    )

    expected_format = """
    {
        "tools": [
            {
                "raison": "In short explain, step-by-step, your thinking for this chosen N tool you choose.",
                "name": "Name of the N tool.",
                "arguments": "The arguments for the N tool, a dictionnary type of Dict[str, Any]."
            }
        ]
    }
    """
    expected_format = json.loads(expected_format)

    # Act & Assert
    format = json.loads(formatted_schema)
    assert format == expected_format


def test_formatting_when_render_default_value_should_provide_format_on_default_value() -> None:
    # Arrange
    formatted_schema = render_json_schema_with_field_value(
        render_type=FieldType.DEFAULT,
        basemodel_type=UseToolsWithValue,
        basemodel_linked_refs_types=[SelectedTool],
    )

    expected_format = """
    {
        "tools": [
            {
                "raison": "value",
                "name": "value",
                "arguments": {"value":"value"}
            }
        ]
    }
    """
    expected_format = json.loads(expected_format)

    # Act & Assert
    format = json.loads(formatted_schema)
    assert format == expected_format


def test_formatting_when_render_default_with_not_defined_default_should_provide_default_and_empty() -> None:
    # Arrange
    formatted_schema = render_json_schema_with_field_value(
        render_type=FieldType.DEFAULT,
        basemodel_type=UseToolsWithDesriptionAndValue,
        basemodel_linked_refs_types=[SelectedTool],
    )

    expected_format = """
    {
        "tools": [
            {
                "raison": "value",
                "name": ">>custom<<",
                "arguments": {"value":"value"}
            }
        ],
        "value": "",
        "value2": ""
    }
    """
    expected_format = json.loads(expected_format)

    # Act & Assert
    format = json.loads(formatted_schema)
    assert format == expected_format


def test_formatting_when_render_type_description_should_provide_format_on_description() -> None:
    # Arrange
    formatted_schema = render_json_schema_with_field_value(
        render_type=FieldType.DESCRIPTION,
        basemodel_type=UseToolsWithDesriptionAndType,
        basemodel_linked_refs_types=[SelectedTool],
    )

    expected_format = """
    {
        "tools": [
            {
                "raison": "In short explain, step-by-step, your thinking for this chosen N tool you choose.",
                "name": "Name of the N tool.",
                "arguments": "The arguments for the N tool, a dictionnary type of Dict[str, Any]."
            }
        ],
        "value": "this int",
        "value2": "this int3"
    }
    """
    expected_format = json.loads(expected_format)

    # Act & Assert
    format = json.loads(formatted_schema)
    assert format == expected_format


def test_formatting_when_custom_field_key_should_provide_on_custom_field_key() -> None:
    # Arrange
    formatted_schema = render_json_schema_with_field_value(
        render_type=FieldType.DESCRIPTION,
        basemodel_type=UseToolsWithCustomKeyField,
        basemodel_linked_refs_types=[SelectedTool],
    )

    expected_format = """
        {
        "tools":[
            {
                "raison":"In short explain, step-by-step, your thinking for this chosen N tool you choose.",
                "name":"Name of the N tool.",
                "arguments":"The arguments for the N tool, a dictionnary type of Dict[str, Any]."
            }
        ],
        "value":"this int",
        "value2":"this int3"
        }
    """
    expected_format = json.loads(expected_format)

    # Act & Assert
    format = json.loads(formatted_schema)
    assert format == expected_format
