from typing import Optional

from langchain_core.callbacks import AsyncCallbackManagerForChainRun, CallbackManagerForChainRun
from pydantic import BaseModel
import pytest

from langfoundation.chain.pydantic.chain import BasePydanticChain


class InputModel(BaseModel):
    field1: str


class OutputModel(BaseModel):
    field2: str


class TestChain(BasePydanticChain[InputModel, OutputModel]):
    async def acall(self, input: InputModel, run_manager: Optional[AsyncCallbackManagerForChainRun]) -> OutputModel:  # type: ignore
        return OutputModel(field2=input.field1)

    def call(self, input: InputModel, run_manager: Optional[CallbackManagerForChainRun]) -> OutputModel:
        return OutputModel(field2=input.field1)


@pytest.fixture
def test_chain() -> TestChain:
    """
    Fixture returning an instance of TestChain.

    This fixture is used in tests requiring a full TestChain instance.
    """
    return TestChain(max_retries=0)


def test_chain_type(test_chain: TestChain) -> None:
    """
    Tests that the _chain_type property is correctly set to "testchain".

    The _chain_type property should return a string that identifies the type of the chain.
    """
    assert test_chain._chain_type == "testchain"


def test_input_keys(test_chain: TestChain) -> None:
    """
    Tests that the input_keys property is correctly set to ["field1"].

    The input_keys property should return a list of the keys of the fields Input BaseModel.
    """
    assert test_chain.input_keys == ["field1"]


def test_output_keys(test_chain: TestChain) -> None:
    """
    Tests that the output_keys property is correctly set to ["field2"].

    The output_keys property should return a list of the keys of the fields Output BaseModel.
    """
    assert test_chain.output_keys == ["field2"]


def test_convert_input_model_to_dict(test_chain: TestChain) -> None:
    """
    Tests that the _convert_input_model_to_dict method correctly converts an input model into a dictionary.

    The _convert_input_model_to_dict method should return a dictionary with the same keys and values as the input model.
    """
    input_model = InputModel(field1="test")
    assert test_chain._convert_input_model_to_dict(input_model) == {"field1": "test"}


def test_convert_output_model_to_dict(test_chain: TestChain) -> None:
    """
    Tests that the _convert_output_model_to_dict method correctly converts an output model into a dictionary.

    The _convert_output_model_to_dict method should return a dictionary with the same keys and values as the output model.
    """
    output_model = OutputModel(field2="test")
    assert test_chain._convert_output_model_to_dict(output_model) == {"field2": "test"}


def test_invoke_as_model(test_chain: TestChain) -> None:
    """
    Tests that the invoke_as_model method correctly calls the graph as a sync method with an input model.

    The invoke_as_model method should return an OutputModel instance with the same attributes as the output of the graph.
    """
    input_model = InputModel(field1="test")
    output_model = test_chain.invoke_as_model(input_model)
    assert output_model.field2 == "test"


@pytest.mark.asyncio
async def test_ainvoke_as_model(test_chain: TestChain) -> None:
    """
    Tests that the ainvoke_as_model method correctly calls the graph as an async method with an input model.

    The ainvoke_as_model method should return an OutputModel instance with the same attributes as the output of the graph.
    """
    input_model = InputModel(field1="test")
    output_model = await test_chain.ainvoke_as_model(input_model)
    assert output_model.field2 == "test"
