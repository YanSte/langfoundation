from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cached_property
import logging
from typing import Any, cast, Dict, Generic, List, Optional, Type, TypeVar, Union

from langchain.chains.base import Chain as BaseChain
from langchain_core.callbacks import AsyncCallbackManagerForChainRun, CallbackManagerForChainRun
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.graph import Graph
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from langfoundation.errors.error import PydanticChainError
from langfoundation.utils.pydantic.base_model import get_type_base_generic, populate_missing_optional_fields, required_fields
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)

# "State" must be a subclass of BaseModel.
State = TypeVar("State", bound=BaseModel)

# "Input" must be a subclass of BaseModel.
Input = TypeVar("Input", bound=BaseModel)

# "Ouput" must be a subclass of BaseModel.
Ouput = TypeVar("Ouput", bound=BaseModel)


class BasePydanticGraphChain(
    BaseChain,
    Generic[State, Input, Ouput],
    ABC,
):
    """
    The BasePydanticGraphChain class is a generic abstract base class for Graph that extends from the Chain class of Langchain.
    This class is designed to work with the Runnable Dict Chain of Lanchain, and allows for casting into Pydantic models without
    any disruptions to the updates of Langchain.

    The BasePydanticGraphChain class can work with any types of State, Input, Output that are subclasses of the BaseModel from the pydantic.

    - To use this class, subclass it and override the 'entry_point', 'finish_points', 'workflow' methods.
    - Then, use the 'ainvoke_as_model' or 'invoke_as_model' methods to call the chain as Input and get Output BaseModel Pydantic.
    """

    # General
    # ---

    name: str = Field(
        description="The name of the chain. Default is class name",
        default_factory=lambda: __name__,
    )

    @property
    def _chain_type(self) -> str:
        return self.__class__.__name__.lower()

    # Input / Output / State
    # ---

    @property
    def StateModelType(self) -> Type[State]:
        "State Model type of BaseModel."
        return cast(
            Type[State],
            get_type_base_generic(
                self,
                type_generic_index=1,
                type_generic_inner_index=0,
            ),
        )

    @property
    def InputModelType(self) -> Type[Input]:
        "Input Model type of BaseModel."
        return cast(
            Type[Input],
            get_type_base_generic(
                self,
                type_generic_index=1,
                type_generic_inner_index=1,
            ),
        )

    @property
    def OutputModelType(self) -> Type[Ouput]:
        "Output Model type of BaseModel."
        return cast(
            Type[Ouput],
            get_type_base_generic(
                self,
                type_generic_index=1,
                type_generic_inner_index=2,
            ),
        )

    @property
    def input_keys(self) -> List[str]:
        """
        This property returns a list of the keys of the fields State BaseModel.

        Only for required keys.
        Avoid the State in State object and required heritance.
        """
        return list(required_fields(self.InputModelType))

    @property
    def output_keys(self) -> List[str]:
        """
        This property returns a list of the keys of the fields State BaseModel.
        """
        return list(self.OutputModelType.model_fields.keys())

    # Graph
    # ---

    @cached_property
    def _graph(self) -> CompiledStateGraph:
        """
        Graph workflow
        """
        workflow = StateGraph(
            state_schema=self.StateModelType,
            # Note: The input is the same as the state type,
            # which allows for the creation of public inputs and internal with default value for StateGraph.
            input=self.StateModelType,
            output=self.OutputModelType,
        )
        self._workflow(workflow)

        workflow.set_entry_point(self.entry_point)

        for finish in self.finish_points:
            workflow.set_finish_point(finish)

        return workflow.compile()

    # Abstract
    # ---

    @property
    @abstractmethod
    def entry_point(self) -> str:
        """
        Abstract to be overridden by subclasses to define entry points
        for the workflow.
        """

        raise NotImplementedError()

    @property
    @abstractmethod
    def finish_points(self) -> List[str]:
        """
        Abstract to be overridden by subclasses to define finish points
        for the workflow.
        """
        raise NotImplementedError()

    @abstractmethod
    def _workflow(self, workflow: StateGraph) -> None:
        """
        Abstract method to be implemented by subclasses to set up the workflow
        within the state graph.
        """
        raise NotImplementedError()

    # Config
    # ---

    class Config:
        arbitrary_types_allowed = True
        keep_untouched = (cached_property,)

    # Invoke
    # ---
    async def ainvoke_as_model(
        self,
        input: Union[Input, Dict[str, Any]],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Ouput:
        input = self._convert_input_model_to_dict(input)

        logger.info(
            input,
            extra={"title": "[Start] Invoke" + " : " + self._chain_type, "verbose": self.verbose},
        )

        output = await super().ainvoke(input=input, config=config, kwargs=kwargs)
        output_model = self.OutputModelType(**output)

        logger.info(
            output,
            extra={"title": "[End] Invoke" + " : " + self._chain_type, "verbose": self.verbose},
        )
        return output_model

    async def invoke_as_model(
        self,
        input: Union[Input, Dict[str, Any]],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Ouput:
        input = self._convert_input_model_to_dict(input)

        logger.info(
            input,
            extra={"title": "[Start] Invoke" + " : " + self._chain_type, "verbose": self.verbose},
        )

        output = super().invoke(input=input, config=config, kwargs=kwargs)
        output_model = self.OutputModelType(**output)

        logger.info(
            output,
            extra={"title": "[End] Invoke" + " : " + self._chain_type, "verbose": self.verbose},
        )
        return output_model

    async def ainvoke(
        self,
        input: Dict[str, Any],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        logger.info(
            input,
            extra={"title": "[Start] Invoke" + " : " + self._chain_type, "verbose": self.verbose},
        )

        output = await super().ainvoke(input, config, **kwargs)

        logger.info(
            output,
            extra={"title": "[End] Invoke" + " : " + self._chain_type, "verbose": self.verbose},
        )
        return output

    def invoke(
        self,
        input: Dict[str, Any],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        logger.info(
            input,
            extra={"title": "[Start] Invoke" + " : " + self._chain_type, "verbose": self.verbose},
        )

        output = super().invoke(input, config, **kwargs)

        logger.info(
            output,
            extra={"title": "[End] Invoke" + " : " + self._chain_type, "verbose": self.verbose},
        )
        return output

    # Call
    # ---

    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        inputs = populate_missing_optional_fields(self.StateModelType, inputs)
        state_model = self.StateModelType(**inputs)
        callbacks = run_manager.get_child() if run_manager else None
        try:
            new_state = await self._graph.ainvoke(
                input=state_model,
                config=RunnableConfig(callbacks=callbacks),
            )
            ouput_model = self.OutputModelType(**new_state)  # Valide representation
            return dict(ouput_model)

        except PydanticChainError as e:
            logger.error(
                e,
                stack_info=True,
                extra={"title": "[ERROR] _acall" + " : " + self._chain_type},
            )
            raise e

        except Exception as e:
            error = PydanticChainError(origin=self._chain_type, error=e)
            logger.error(
                e,
                stack_info=True,
                extra={"title": "[ERROR] _acall" + " : " + self._chain_type},
            )
            raise error

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        inputs = populate_missing_optional_fields(self.StateModelType, inputs)
        state_model = self.StateModelType(**inputs)
        callbacks = run_manager.get_child() if run_manager else None
        try:
            new_state = self._graph.invoke(
                input=state_model,
                config=RunnableConfig(callbacks=callbacks),
            )
            ouput_model = self.OutputModelType(**new_state)  # Valide representation
            return dict(ouput_model)

        except PydanticChainError as e:
            logger.error(
                e,
                stack_info=True,
                extra={"title": "[ERROR] _acall" + " : " + self._chain_type},
            )
            raise e

        except Exception as e:
            error = PydanticChainError(origin=self._chain_type, error=e)
            logger.error(
                e,
                stack_info=True,
                extra={"title": "[ERROR] _acall" + " : " + self._chain_type},
            )
            raise error

    # Graph
    # ---

    def get_graph(self, config: Optional[RunnableConfig] = None) -> Graph:
        return self._graph.get_graph(config=config)

    # Private Convert
    # ---

    def _convert_input_model_to_dict(self, input: Union[Input, Dict[str, Any], BaseModel]) -> Dict[str, Any]:
        if isinstance(input, BaseModel):
            return input.dict()
        return input
