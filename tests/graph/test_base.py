from typing import List

from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel
import pytest

from langfoundation.chain.pydantic.graph import BasePydanticGraphChain


class StateModel(BaseModel):
    field1: str
    field2: int


class InputModel(StateModel):
    pass


class OutputModel(BaseModel):
    result: str


class TestGraphChain(BasePydanticGraphChain[StateModel, InputModel, OutputModel]):
    @property
    def entry_point(self) -> str:
        return "entry"

    @property
    def finish_points(self) -> List[str]:
        return ["finish"]

    def _workflow(self, workflow: StateGraph) -> None:
        workflow.add_node("entry", lambda state: {"result": f"{state.field1} - {state.field2}"})
        workflow.add_node("finish", lambda state: state)
        workflow.add_edge("entry", "finish")


@pytest.fixture
def test_chain() -> TestGraphChain:
    """
    Fixture returning an instance of TestGraphChain.

    This fixture is used in tests requiring a full TestGraphChain instance.
    """
    return TestGraphChain()


def test_chain_type(test_chain: TestGraphChain) -> None:
    """
    Tests that the _chain_type property is correctly set to "testgraphchain".
    """
    assert test_chain._chain_type == "testgraphchain"


def test_input_keys(test_chain: TestGraphChain) -> None:
    """
    Tests that the input_keys property is correctly set to ["field1", "field2"].

    The input_keys property should return a list of the keys of the fields Input BaseModel.
    """
    assert test_chain.input_keys == ["field1", "field2"]


def test_output_keys(test_chain: TestGraphChain) -> None:
    """
    Tests that the output_keys property is correctly set to ["result"].

    The output_keys property should return a list of the keys of the fields Output BaseModel.
    """
    assert test_chain.output_keys == ["result"]


def test_graph(test_chain: TestGraphChain) -> None:
    """
    Tests that the _graph property is correctly set to a CompiledStateGraph instance.

    The _graph property should return a CompiledStateGraph instance that represents the workflow graph.
    """
    assert isinstance(test_chain._graph, CompiledStateGraph)


def test_convert_input_model_to_dict(test_chain: TestGraphChain) -> None:
    """
    Tests that the _convert_input_model_to_dict method correctly converts an input model into a dictionary.

    The _convert_input_model_to_dict method should return a dictionary with the same keys and values as the input model.
    """
    input_model = InputModel(field1="test", field2=123)
    expected_output = {"field1": "test", "field2": 123}
    assert test_chain._convert_input_model_to_dict(input_model) == expected_output


@pytest.mark.asyncio
async def test_ainvoke_as_model(test_chain: TestGraphChain) -> None:
    """
    Tests that the ainvoke_as_model method correctly calls the graph as an async method with an input model.

    The ainvoke_as_model method should return an OutputModel instance with the same attributes as the output of the graph.
    """
    input_model = InputModel(field1="test", field2=123)
    output_model = await test_chain.ainvoke_as_model(input_model)
    assert output_model.result == "test - 123"


@pytest.mark.asyncio
async def test_invoke_as_model(test_chain: TestGraphChain) -> None:
    """
    Tests that the invoke_as_model method correctly calls the graph as a sync method with an input model.

    The invoke_as_model method should return an OutputModel instance with the same attributes as the output of the graph.
    """
    input_model = InputModel(field1="test", field2=123)
    output_model = test_chain.invoke_as_model(input_model)
    assert output_model.result == "test - 123"


@pytest.mark.asyncio
async def test_ainvoke(test_chain: TestGraphChain) -> None:
    """
    Tests that the ainvoke method correctly calls the graph as an async method with a dictionary input.

    The ainvoke method should return a dictionary with the same keys and values as the output of the graph.
    """
    input_dict = {"field1": "test", "field2": 123}
    output_dict = await test_chain.ainvoke(input_dict)
    assert output_dict["result"] == "test - 123"


def test_invoke(test_chain: TestGraphChain) -> None:
    """
    Tests that the invoke method correctly calls the graph as a sync method with a dictionary input.

    The invoke method should return a dictionary with the same keys and values as the output of the graph.
    """
    input_dict = {"field1": "test", "field2": 123}
    output_dict = test_chain.invoke(input_dict)
    assert output_dict["result"] == "test - 123"
