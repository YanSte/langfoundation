from typing import List, Optional

import pytest
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from langfoundation.chain.graph.graph.base import BaseGraphChain
from langfoundation.chain.pydantic.errors.error import PydanticChainError


class StateModel(BaseModel):
    field1: str = Field(description="Field 1")
    field2: Optional[str] = Field(description="Field 2", default=None)
    field3: Optional[int] = Field(description="Field 3", default=None)
    field4: Optional[str] = Field(description="Field 4", default=None)


class InputModel(BaseModel):
    field1: str = Field(description="Field 1")


class OutputModel(BaseModel):
    field3: int = Field(description="Field 3")
    field4: str = Field(description="Field 4")


class TestGraphChain(BaseGraphChain[StateModel, InputModel, OutputModel]):
    @property
    def entry_point(self) -> str:
        return "entry"

    @property
    def finish_points(self) -> List[str]:
        return ["finish"]

    def _workflow(self, workflow: StateGraph) -> None:
        def define(state: StateModel) -> StateModel:
            state.field4 = state.field1

            state.field1 = "field1"
            state.field2 = "field3"
            state.field3 = 3
            return state

        workflow.add_node("entry", define)
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
    assert test_chain.input_keys == ["field1"]


def test_output_keys(test_chain: TestGraphChain) -> None:
    """
    Tests that the output_keys property is correctly set to ["result"].

    The output_keys property should return a list of the keys of the fields Output BaseModel.
    """
    assert test_chain.output_keys == ["field3", "field4"]


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
    input_model = InputModel(field1="test")
    expected_output = {"field1": "test"}
    assert test_chain._convert_input_model_to_dict(input_model) == expected_output


@pytest.mark.asyncio
async def test_ainvoke_as_model(test_chain: TestGraphChain) -> None:
    """
    Tests that the ainvoke_as_model method correctly calls the graph as an async method with an input model.

    The ainvoke_as_model method should return an OutputModel instance with the same attributes as the output of the graph.
    """
    input_model = InputModel(field1="test")
    output_model = await test_chain.ainvoke_as_model(input_model)
    assert output_model.field3 == 3


@pytest.mark.asyncio
async def test_invoke_as_model(test_chain: TestGraphChain) -> None:
    """
    Tests that the invoke_as_model method correctly calls the graph as a sync method with an input model.

    The invoke_as_model method should return an OutputModel instance with the same attributes as the output of the graph.
    """
    input_model = InputModel(field1="test")
    output_model = test_chain.invoke_as_model(input_model)
    assert output_model.field3 == 3


@pytest.mark.asyncio
async def test_invoke_as_model_pass_value(test_chain: TestGraphChain) -> None:
    """
    Tests that the invoke_as_model method correctly calls the graph as a sync method with an input model.

    The invoke_as_model method should return an OutputModel instance with the same attributes as the output of the graph.
    """
    input_model = InputModel(field1="test")
    output_model = test_chain.invoke_as_model(input_model)
    assert output_model.field4 == input_model.field1


@pytest.mark.asyncio
async def test_ainvoke(test_chain: TestGraphChain) -> None:
    """
    Tests that the ainvoke method correctly calls the graph as an async method with a dictionary input.

    The ainvoke method should return a dictionary with the same keys and values as the output of the graph.
    """
    input_dict = {"field1": "test", "field2": 123}
    output_dict = await test_chain.ainvoke(input_dict)
    assert output_dict["field3"] == 3


def test_invoke(test_chain: TestGraphChain) -> None:
    """
    Tests that the invoke method correctly calls the graph as a sync method with a dictionary input.

    The invoke method should return a dictionary with the same keys and values as the output of the graph.
    """
    input_dict = {"field1": "test", "field2": 123}
    output_dict = test_chain.invoke(input_dict)
    assert output_dict["field3"] == 3


def test_invoke_declara_state_model_property_with_default_value_no_optional(
    test_chain: TestGraphChain,
) -> None:
    """
    Tests that the invoke method correctly raises a PydanticChainError when the StateModel is not valid.

    The invoke method should raise a PydanticChainError with a message that includes the required properties
    from OutputModel that need to be type `Optional` with default value or `None`.
    """

    class StateModel(BaseModel):
        field1: str = Field(description="Field 1")
        field2: str = Field(description="Field 2", default="Hello set")  # NOTE: Set by default

    class InputModel(BaseModel):
        field1: str = Field(description="Field 1")

    class OutputModel(BaseModel):
        field2: str = Field(description="Field 2", default=None)

    class TestGraphChain(BaseGraphChain[StateModel, InputModel, OutputModel]):
        @property
        def entry_point(self) -> str:
            return "entry"

        @property
        def finish_points(self) -> List[str]:
            return ["finish"]

        def _workflow(self, workflow: StateGraph) -> None:
            def define(state: StateModel) -> StateModel:
                state.field1 = "field1"
                state.field2 = "updated_fied2"

                return state

            workflow.add_node("entry", define)
            workflow.add_node("finish", lambda state: state)
            workflow.add_edge("entry", "finish")

    input_dict = {"field1": "test"}
    result = TestGraphChain().invoke_as_model(input_dict)

    assert result.field2 == "updated_fied2"


def test_invoke_raises_pydantic_chain_error_with_invalid_state_model_missing_default_value(
    test_chain: TestGraphChain,
) -> None:
    """
    Tests that the invoke method correctly raises a PydanticChainError when the StateModel is not valid.

    The invoke method should raise a PydanticChainError with a message that includes the required properties
    from OutputModel that need to be type `Optional` with default value or `None`.
    """

    class StateModel(BaseModel):
        field1: str = Field(description="Field 1")
        field2: Optional[str] = Field(description="Field 2")  # Note: Missing default value

    class InputModel(BaseModel):
        field1: str = Field(description="Field 1")

    class OutputModel(BaseModel):
        field2: str = Field(description="Field 2", default=None)

    class TestGraphChain(BaseGraphChain[StateModel, InputModel, OutputModel]):
        @property
        def entry_point(self) -> str:
            return "entry"

        @property
        def finish_points(self) -> List[str]:
            return ["finish"]

        def _workflow(self, workflow: StateGraph) -> None:
            def define(state: StateModel) -> StateModel:
                state.field1 = "field1"
                state.field2 = "field2"

                return state

            workflow.add_node("entry", define)
            workflow.add_node("finish", lambda state: state)
            workflow.add_edge("entry", "finish")

    with pytest.raises(PydanticChainError) as exc_info:
        input_dict = {"field1": "test"}
        TestGraphChain().invoke_as_model(input_dict)

    assert "OUTPUTMODEL" in str(exc_info.value)
    assert "StateModel" in str(exc_info.value)
    assert "field2" in str(exc_info.value)
    assert "default" in str(exc_info.value)
    assert "Optional" in str(exc_info.value)


def test_invoke_raises_pydantic_chain_error_with_missing_property_in_state_model_from_input(
    test_chain: TestGraphChain,
) -> None:
    """
    Tests that the invoke method correctly raises a PydanticChainError when the StateModel is not valid.

    The invoke method should raise a PydanticChainError with a message that includes the required properties
    from OutputModel that need to be type `Optional` with default value or `None`.
    """

    class StateModel(BaseModel):
        # Note: Not declared field1 from input
        field2: Optional[str] = Field(description="Field 2", default=None)

    class InputModel(BaseModel):
        field1: str = Field(description="Field 1")

    class OutputModel(BaseModel):
        field2: str = Field(description="Field 2", default=None)

    class TestGraphChain(BaseGraphChain[StateModel, InputModel, OutputModel]):
        @property
        def entry_point(self) -> str:
            return "entry"

        @property
        def finish_points(self) -> List[str]:
            return ["finish"]

        def _workflow(self, workflow: StateGraph) -> None:
            def define(state: StateModel) -> StateModel:
                # state.field1 = "field1"
                state.field2 = "field2"

                return state

            workflow.add_node("entry", define)
            workflow.add_node("finish", lambda state: state)
            workflow.add_edge("entry", "finish")

    input_dict = {"field1": "test"}
    with pytest.raises(PydanticChainError) as exc_info:
        TestGraphChain().invoke_as_model(input_dict)

    assert "INPUTMODEL" in str(exc_info.value)
    assert "StateModel" in str(exc_info.value)
    assert "field1" in str(exc_info.value)
    assert "required" in str(exc_info.value)


def test_invoke_raises_pydantic_chain_error_with_missing_property_in_state_model_from_output(
    test_chain: TestGraphChain,
) -> None:
    """
    Tests that the invoke method correctly raises a PydanticChainError when the StateModel is not valid.

    The invoke method should raise a PydanticChainError with a message that includes the required properties
    from OutputModel that are missing in StateModel.
    """

    class StateModel(BaseModel):
        field1: str = Field(description="Field 1")
        # Note: Not declared field2 from output

    class InputModel(BaseModel):
        field1: str = Field(description="Field 1")

    class OutputModel(BaseModel):
        field2: str = Field(description="Field 2", default=None)

    class TestGraphChain(BaseGraphChain[StateModel, InputModel, OutputModel]):
        @property
        def entry_point(self) -> str:
            return "entry"

        @property
        def finish_points(self) -> List[str]:
            return ["finish"]

        def _workflow(self, workflow: StateGraph) -> None:
            def define(state: StateModel) -> StateModel:
                state.field1 = "field1"
                # state.field2 = "field2"

                return state

            workflow.add_node("entry", define)
            workflow.add_node("finish", lambda state: state)
            workflow.add_edge("entry", "finish")

    input_dict = {"field1": "test"}
    with pytest.raises(PydanticChainError) as exc_info:
        TestGraphChain().invoke_as_model(input_dict)

    assert "OUTPUTMODEL" in str(exc_info.value)
    assert "StateModel" in str(exc_info.value)
    assert "field2" in str(exc_info.value)
    assert "required" in str(exc_info.value)
