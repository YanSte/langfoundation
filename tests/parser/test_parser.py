import json
from typing import Dict, List

import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.outputs import Generation
from pydantic import BaseModel, Field

from langfoundation.parser.pydantic.parser import PydanticOutputParser


class TestModel(BaseModel):
    property1: str
    property2: int


def test_parse() -> None:
    """
    Tests that the parse method correctly parses a given JSON string into
    a pydantic model.

    GIVEN:
        - A PydanticOutputParser with a given pydantic object
        - A JSON string matching the pydantic object

    WHEN:
        - The parse method is called with the given string

    THEN:
        - The method returns an instance of the pydantic object with the
          correct values
    """

    parser = PydanticOutputParser(pydantic_object=TestModel)
    text = '{"property1": "test", "property2": 123}'
    result = parser.parse(text)
    assert result.property1 == "test"
    assert result.property2 == 123


def test_parse_result() -> None:
    """
    Tests that the parse_result method correctly parses a given list of Generation
    objects into an instance of the pydantic object.

    GIVEN:
        - A PydanticOutputParser with a given pydantic object
        - A list of Generation objects matching the pydantic object

    WHEN:
        - The parse_result method is called with the given list

    THEN:
        - The method returns an instance of the pydantic object with the
          correct values
    """
    parser = PydanticOutputParser(pydantic_object=TestModel)
    result = [Generation(text='{"property1": "test", "property2": 123}')]
    output = parser.parse_result(result)
    assert output.property1 == "test"
    assert output.property2 == 123


def test_get_format_instructions() -> None:
    """
    Tests that the get_format_instructions method correctly returns a string
    describing the JSON schema expected of the output.

    GIVEN:
        - A PydanticOutputParser with a given pydantic object

    WHEN:
        - The get_format_instructions method is called

    THEN:
        - The method returns a string describing the JSON schema expected
          of the output, which contains the names of the properties in the
          pydantic object
    """
    parser = PydanticOutputParser(pydantic_object=TestModel)
    instructions = parser.get_format_instructions()
    assert "property1" in instructions
    assert "property2" in instructions


def test_get_format_value_instructions() -> None:
    """
    Tests that the get_format_value_instructions method correctly returns a string
    describing the JSON schema expected of the output, including default values
    for each property.

    GIVEN:
        - A PydanticOutputParser with a given pydantic object

    WHEN:
        - The get_format_value_instructions method is called

    THEN:
        - The method returns a string describing the JSON schema expected
          of the output, which contains the names of the properties in the
          pydantic object, as well as their default values
    """
    parser = PydanticOutputParser(pydantic_object=TestModel)
    instructions = parser.get_format_value_instructions()
    assert "property1" in instructions
    assert "property2" in instructions


def test_parser_exception() -> None:
    """
    Tests that the parse method correctly raises an OutputParserException when
    given invalid JSON.

    GIVEN:
        - A PydanticOutputParser with a given pydantic object
        - A string that is not valid JSON

    WHEN:
        - The parse method is called with the given string

    THEN:
        - The method raises an OutputParserException
    """
    parser = PydanticOutputParser(pydantic_object=TestModel)
    with pytest.raises(OutputParserException):
        parser.parse('{"property1": "test"}')


def test_get_output_format() -> None:
    """
    Tests that the get_format_instructions method correctly returns a string
    describing the JSON schema expected of the output.

    GIVEN:
        - A PydanticOutputParser with a given pydantic object
        - The given pydantic object has a property with a description

    WHEN:
        - The get_format_instructions method is called

    THEN:
        - The method returns a string describing the JSON schema expected
          of the output, which contains the names of the properties in the
          pydantic object, as well as their description
    """

    # Arrange
    class MockModel(BaseModel):
        property1: str = Field(description="Property 1")
        property2: List[Dict[str, str]] = Field(
            description='In format of: {"subproperty1": "Subproperty 1", "subproperty2": "Subproperty 2"}'
        )

    parser = PydanticOutputParser(pydantic_object=MockModel)

    expected_schema = (
        '{"property1": "Property 1", "property2": "In format of: {"subproperty1": "Subproperty 1", "subproperty2": "Subproperty 2"}"}'  # noqa
    )

    expected_output = parser.format_instructions.format(schema=expected_schema)

    # Act
    result = parser.get_format_instructions()

    # Assert
    assert result == expected_output


def test_get_output_format_with_optional_property() -> None:
    """
    Tests that the get_format_instructions method correctly returns a string
    describing the JSON schema expected of the output, including default values
    for each property, when the Field has a default value.

    GIVEN:
        - A PydanticOutputParser with a given pydantic object

    WHEN:
        - The get_format_value_instructions method is called

    THEN:
        - The method returns a string describing the JSON schema expected
          of the output, which contains the names of the properties in the
          pydantic object, as well as their default values
    """

    # Arrange
    class MockModel(BaseModel):
        # A string field with a description
        property1: str = Field(description="Property 1")
        # A list of dictionaries with a description
        property2: List[Dict[str, str]] = Field(
            description="In format of: {'subproperty1': 'Subproperty 1', 'subproperty2': 'Subproperty 2'}"
        )

    parser = PydanticOutputParser(pydantic_object=MockModel)

    expected_schema = "{\"property1\": \"Property 1\", \"property2\": \"In format of: {'subproperty1': 'Subproperty 1', 'subproperty2': 'Subproperty 2'}\"}"  # noqa

    expected_output = parser.format_instructions.format(schema=expected_schema)

    # Act
    result = parser.get_format_instructions()

    # Assert
    assert result == expected_output


def test_get_output_format_when_value_should_select_value() -> None:
    """
    Tests that the get_format_value_instructions method correctly returns a string
    describing the JSON schema expected of the output, including default values
    for each property, when the Field has a default value.

    GIVEN:
        - A PydanticOutputParser with a given pydantic object

    WHEN:
        - The get_format_value_instructions method is called

    THEN:
        - The method returns a string describing the JSON schema expected
          of the output, which contains the names of the properties in the
          pydantic object, as well as their default values
    """

    # Arrange
    class MockModel(BaseModel):
        property1: str = Field(
            description="Property 1",
            default="hello",
        )
        property2: Dict[str, str] = Field(
            repr=True,
            description="Property 2",
            default={"hello": "hello"},
        )

    parser = PydanticOutputParser(pydantic_object=MockModel)

    expected_schema = json.dumps(
        {
            "property1": "Property 1",
            "property2": "Property 2",
        },
    )

    expected_output = parser.format_instructions.format(schema=expected_schema)

    # Act
    result = parser.get_format_value_instructions()

    # Assert
    assert result == expected_output


def test_get_output_format_when_complex_basemodel_with_ref_should_return_complex_with_ref_format() -> None:
    """
    GIVEN:
        - A Pydantic model with a property referencing another Pydantic model
    WHEN:
        - The get_format_instructions method is called
    THEN:
        - The method returns a string describing the JSON schema expected
          of the output, which contains the names of the properties in the
          pydantic object, as well as their default values
    """

    class SliceOutputTopicQuestion(BaseModel):
        topic: str = Field(
            description="N topic should focus.",
        )
        question: str = Field(
            description="N segmented query for processing.",
        )
        topics: List[str] = Field(
            description="Segmented query for processing.",
            default=["test", "test"],
        )

    class SliceQuestionsOutput(BaseModel):
        """Route Path"""

        question: str = Field(
            description="The input question to answer.",
        )
        thought: str = Field(
            description="Explain step-by-step your thinking for the chosen segments you will create.",
        )
        segments: List[SliceOutputTopicQuestion] = Field(
            description="List of segmented use question into different pieces topic and question."
        )

    # Arrange
    parser = PydanticOutputParser(
        pydantic_object=SliceQuestionsOutput,
        basemodel_linked_refs_types=[SliceOutputTopicQuestion],
    )

    expected_schema = json.dumps(
        {
            "question": "The input question to answer.",
            "thought": "Explain step-by-step your thinking for the chosen segments you will create.",
            "segments": [
                {
                    "topic": "N topic should focus.",
                    "question": "N segmented query for processing.",
                    "topics": "Segmented query for processing.",
                }
            ],
        },
    )

    expected_output = parser.format_instructions.format(schema=expected_schema)

    # Act
    result = parser.get_format_instructions()

    # Assert
    assert result == expected_output
