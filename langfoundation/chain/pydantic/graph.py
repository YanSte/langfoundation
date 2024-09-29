from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cached_property
import logging
from typing import (
    Any,
    cast,
    Dict,
    Generic,
    List,
    Optional,
    Type,
    TypeVar,
    Union,
)

from langchain.chains.base import Chain as BaseChain
from langchain_core.callbacks import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.graph import Graph
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic_core import InitErrorDetails, PydanticCustomError, PydanticUndefined

from langfoundation.chain.pydantic.errors.error import PydanticChainError
from langfoundation.utils.pydantic.base_model import (
    get_type_base_generic,
    required_fields,
)
from pydantic import BaseModel, Field, ValidationError


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
            input=self.InputModelType,
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
        config: Union[RunnableConfig, None] = None,
        **kwargs: Any,
    ) -> Ouput:
        """Call the computation graph asynchronously and return model in pydantic basemodel."""
        try:
            self._validate_input_state_output_type()
            input = self._convert_input_model_to_dict(input)

            logger.info(
                input,
                extra={
                    "title": "[Start] Invoke" + " : " + self._chain_type,
                    "verbose": self.verbose,
                },
            )

            output = await super().ainvoke(input=input, config=config, kwargs=kwargs)
            output_model = self.OutputModelType(**output)

            logger.info(
                output,
                extra={
                    "title": "[End] Invoke" + " : " + self._chain_type,
                    "verbose": self.verbose,
                },
            )
            return output_model

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

    def invoke_as_model(
        self,
        input: Union[Input, Dict[str, Any]],
        config: Union[RunnableConfig, None] = None,
        **kwargs: Any,
    ) -> Ouput:
        """Call the computation graph synchronously and return model in pydantic basemodel."""
        try:
            self._validate_input_state_output_type()
            input = self._convert_input_model_to_dict(input)

            logger.info(
                input,
                extra={
                    "title": "[Start] Invoke" + " : " + self._chain_type,
                    "verbose": self.verbose,
                },
            )

            output = super().invoke(input=input, config=config, kwargs=kwargs)
            output_model = self.OutputModelType(**output)

            logger.info(
                output,
                extra={
                    "title": "[End] Invoke" + " : " + self._chain_type,
                    "verbose": self.verbose,
                },
            )
            return output_model

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

    async def ainvoke(
        self,
        input: Dict[str, Any],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Call the computation graph asynchronously."""
        try:
            self._validate_input_state_output_type()
            input = self._convert_input_model_to_dict(input)

            logger.info(
                input,
                extra={
                    "title": "[Start] Invoke" + " : " + self._chain_type,
                    "verbose": self.verbose,
                },
            )

            output = await super().ainvoke(input, config, **kwargs)

            logger.info(
                output,
                extra={
                    "title": "[End] Invoke" + " : " + self._chain_type,
                    "verbose": self.verbose,
                },
            )
            return output

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

    def invoke(
        self,
        input: Dict[str, Any],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Call the computation graph synchronously."""
        try:
            self._validate_input_state_output_type()
            input = self._convert_input_model_to_dict(input)

            logger.info(
                input,
                extra={
                    "title": "[Start] Invoke" + " : " + self._chain_type,
                    "verbose": self.verbose,
                },
            )

            output = super().invoke(input, config, **kwargs)

            logger.info(
                output,
                extra={
                    "title": "[End] Invoke" + " : " + self._chain_type,
                    "verbose": self.verbose,
                },
            )
            return output

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

    # Call
    # ---

    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """
        Call the computation graph asynchronously.
        """
        try:
            callbacks = run_manager.get_child() if run_manager else None

            new_state = await self._graph.ainvoke(
                input=inputs,
                config=RunnableConfig(callbacks=callbacks),
            )
            ouput_model = self.OutputModelType(**new_state)
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
        """
        Call the computation graph synchronously.
        """
        try:
            callbacks = run_manager.get_child() if run_manager else None

            new_state = self._graph.invoke(
                input=inputs,
                config=RunnableConfig(callbacks=callbacks),
            )
            ouput_model = self.OutputModelType(**new_state)
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
        """Returns a drawable representation of the computation graph."""
        return self._graph.get_graph(config=config)

    # Private Convert
    # ---

    def _convert_input_model_to_dict(self, input: Union[Input, Dict[str, Any], BaseModel]) -> Dict[str, Any]:
        if isinstance(input, BaseModel):
            return input.model_dump()
        return input

    def _validate_input_state_output_type(self) -> None:
        """
        Check if the state model contains all the properties from the input model.
        Also, check if the output model contains all the properties from the state model.
        """
        self._valide_input_properties_include_in_state_models(
            self.InputModelType,
            self.StateModelType,
        )

        self._valide_output_properties_include_in_state_models_as_optional(
            self.OutputModelType,
            self.StateModelType,
        )

        self._validate_other_properties_in_state_model(
            self.InputModelType,
            self.OutputModelType,
            self.StateModelType,
        )

    def _valide_input_properties_include_in_state_models(
        self,
        input_model: type[BaseModel],
        state_model: type[BaseModel],
    ) -> None:
        """
        Check if the state model contains all the properties from the input model.
        """
        input_fields = input_model.model_fields
        state_fields = state_model.model_fields

        missing_properties = []
        incorrect_type_properties = []

        for prop_name, prop_field in input_fields.items():
            if prop_name not in state_fields:
                missing_properties.append(prop_name)
            elif prop_field.annotation != state_fields[prop_name].annotation:
                incorrect_type_properties.append(prop_name)

        if missing_properties or incorrect_type_properties:
            line_errors = []
            for prop_name in missing_properties:
                line_errors.append(
                    InitErrorDetails(
                        type=PydanticCustomError(
                            "missing",
                            "In State, a property is required from Input in State.",
                        ),
                        loc=(prop_name,),
                        input="Field required",
                    )
                )
            for prop_name in incorrect_type_properties:
                line_errors.append(
                    InitErrorDetails(
                        type=PydanticCustomError(
                            "type_error",
                            "In State, a property is required to be thesame type from Input in State.",
                        ),
                        loc=(prop_name,),
                        input="Field type mismatch",
                    )
                )
            custom_message = f"Please have a look at `{state_model.__name__}`, Model State is not valid.\n"

            if missing_properties:
                custom_message += "Required all properties from Input in State:\n"
                custom_message += f"- {'\n- '.join(missing_properties)}\n"

            if incorrect_type_properties:
                custom_message += "Required properties to be same type from Input in State:\n"
                custom_message += f"- {'\n- '.join(incorrect_type_properties)}\n"

            e = ValidationError.from_exception_data(f"{state_model.__name__}", line_errors)

            error = PydanticChainError(
                origin=input_model.__name__,
                error=e,
                custom_message=custom_message,
            )
            logger.error(
                error,
                stack_info=True,
                extra={"title": "[ERROR] _acall" + " : " + input_model.__name__},
            )
            raise error

    def _valide_output_properties_include_in_state_models_as_optional(
        self,
        output_model: type[BaseModel],
        state_model: type[BaseModel],
    ) -> None:
        """
        Check if the state model contains all the properties from the output model.
        Also, check if the properties in the state model are set as Optional with default value set to None.
        """
        output_fields = output_model.model_fields
        state_fields = state_model.model_fields

        missing_properties = []

        non_optional_properties = []

        for prop_name, _ in output_fields.items():
            if prop_name not in state_fields:
                missing_properties.append(prop_name)
            else:
                state_prop_field = state_fields[prop_name]
                # Check if the property in the state model is set as Optional with default value set to None.
                if state_prop_field.default is PydanticUndefined or state_prop_field.is_required():
                    non_optional_properties.append(prop_name)

        if missing_properties or non_optional_properties:
            line_errors = []
            for prop_name in missing_properties:
                line_errors.append(
                    InitErrorDetails(
                        type=PydanticCustomError(
                            "missing",
                            "In State, a property is required from Output in State.",
                        ),
                        loc=(prop_name,),
                        input="Property required",
                    )
                )
            for prop_name in non_optional_properties:
                line_errors.append(
                    InitErrorDetails(
                        type=PydanticCustomError(
                            "value_error",
                            "In State, a property from Output is required to be set as type Optional with default `None` or type Non-Optional with default value.",  # noqa
                        ),
                        loc=(prop_name,),
                        input="Field required to be with value or None",
                    )
                )
            e = ValidationError.from_exception_data(
                f"{state_model.__name__}",
                line_errors,
            )
            custom_message = f"Please have a look at `{state_model.__name__}`, Model State is not valid.\n"

            if missing_properties:
                custom_message += "Required all properties from Output in State:\n"
                custom_message += f"- {'\n- '.join(missing_properties)}\n"

            if non_optional_properties:
                custom_message += "Required properties from Output in State to be type Optional with default value `None` or type Non-Optional with default value:\n"  # noqa
                custom_message += f"- {'\n- '.join(non_optional_properties)}\n"

            error = PydanticChainError(
                origin=output_model.__name__,
                error=e,
                custom_message=custom_message,
            )
            logger.error(
                error,
                stack_info=True,
                extra={"title": "[ERROR] _acall" + " : " + output_model.__name__},
            )
            raise error

    def _validate_other_properties_in_state_model(
        self,
        input_model: type[BaseModel],
        output_model: type[BaseModel],
        state_model: type[BaseModel],
    ) -> None:
        """
        Check if all other properties in the state model (excluding the ones from input and output) are optional or have a default value.
        """
        input_fields = input_model.model_fields
        output_fields = output_model.model_fields
        state_fields = state_model.model_fields

        invalid_properties = []

        for prop_name, prop_field in state_fields.items():
            if prop_name not in input_fields and prop_name not in output_fields:
                if prop_field.default is PydanticUndefined or prop_field.is_required():
                    invalid_properties.append(prop_name)

        if invalid_properties:
            line_errors = []
            for prop_name in invalid_properties:
                line_errors.append(
                    InitErrorDetails(
                        type=PydanticCustomError(
                            "value_error",
                            "In State, a property (Not from Input and Output) is required to be set as type Optional with default `None` or type Non-Optional with default value.",  # noqa
                        ),
                        loc=(prop_name,),
                        input="Field required to be with value or None",
                    )
                )
            e = ValidationError.from_exception_data(
                f"{state_model.__name__}",
                line_errors,
            )
            custom_message = f"Please have a look at `{state_model.__name__}`, Model State is not valid.\n"
            custom_message += "Required other properties (Not from Input and Output) in State to be type Optional with default `None` or type Non-Optional with default value:\n"  # noqa
            custom_message += f"- {'\n- '.join(invalid_properties)}\n"

            error = PydanticChainError(
                origin=state_model.__name__,
                error=e,
                custom_message=custom_message,
            )
            logger.error(
                error,
                stack_info=True,
                extra={"title": "[ERROR] _acall" + " : " + state_model.__name__},
            )
            raise error
