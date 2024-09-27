from __future__ import annotations

from abc import ABC, abstractmethod
import logging
from typing import Any, Callable, cast, Dict, Generic, List, Optional, Type, TypeVar, Union

from langchain.chains.base import Chain as BaseChain
from langchain_core.callbacks import AsyncCallbackManagerForChainRun, CallbackManagerForChainRun
from langchain_core.runnables import RunnableConfig

from langfoundation.errors.error import PydanticChainError
from langfoundation.errors.max_retry import MaxRetryError
from langfoundation.parser.pydantic.parser import PydanticOutputParser
from langfoundation.utils.pydantic.base_model import get_type_base_generic, required_fields
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)

# "Input" must be a subclass of BaseModel.
Input = TypeVar("Input", bound=BaseModel)

# "Output" must be a subclass of BaseModel.
Output = TypeVar("Output", bound=BaseModel)


class BasePydanticChain(
    BaseChain,
    Generic[Input, Output],
    ABC,
):
    """
    The BasePydanticChain class is a generic abstract base class that extends from the Chain class of Langchain.
    This class is designed to work with the Runnable Dict Chain of Lanchain, and allows for casting
    into Pydantic models without any disruptions to the updates of Langchain.

    The BasePydanticChain class can work with any types of Input and Output that are subclasses of
    the BaseModel from the pydantic library.

    - To use this class, subclass it and override the 'acall' or 'call' methods.
    - If you use 'fallback_max_retries' override 'afallback' or 'fallback' methods.
    - Then, use the 'ainvoke_as_model' or 'invoke_as_model' methods to call the chain as
    Input Output BaseModel Pydantic.
    """

    # General
    # ---

    @property
    def _chain_type(self) -> str:
        """
        The type of the chain (e.g. "basepydanticchain", "basepydanticgraphchain", etc.).
        """
        return self.__class__.__name__.lower()

    name: str = Field(
        description="The name of the chain.",
        default=__name__,
    )

    fallback_max_retries: Optional[int] = Field(
        description="Maximum of retries allowed for the fallback.",
    )

    # Input / Output
    # ---

    @property
    def InputModelType(self) -> Type[Input]:
        "Input Model type of BaseModel."
        return cast(
            Type[Input],
            get_type_base_generic(
                self,
                type_generic_index=1,
                type_generic_inner_index=0,
            ),
        )

    @property
    def OutputModelType(self) -> Type[Output]:
        "Output Model type of BaseModel."
        return cast(
            Type[Output],
            get_type_base_generic(
                self,
                type_generic_index=1,
                type_generic_inner_index=1,
            ),
        )

    @property
    def input_keys(self) -> List[str]:
        """
        This property returns a list of the keys of the fields Input BaseModel.

        Only for required keys.
        """
        return list(required_fields(self.InputModelType))

    @property
    def output_keys(self) -> List[str]:
        """
        This property returns a list of the keys of the fields Output BaseModel.
        """
        return list(self.OutputModelType.model_fields.keys())

    # Async
    # ---

    @abstractmethod
    async def acall(self, input: Input, run_manager: Optional[AsyncCallbackManagerForChainRun]) -> Output:  # type: ignore
        """
        Abstract method that must be implemented by subclasses.

        This method is designed to handle async calls with pydantic input.
        """
        raise NotImplementedError("To implement, Does not support async call")

    async def afallback(
        self,
        input: Input,
        run_manager: Optional[AsyncCallbackManagerForChainRun],
        errors: List[Exception],
    ) -> Output:
        """
        Needs to be implemented by subclasses if `fallback_max_retries` is set.

        This asynchronous method is designed to handle any fallback operations in the event of an error during the call.
        """
        raise NotImplementedError("Not implemented, `fallback_max_retries` with True but `afallback` is not implemented.")

    # Sync
    # ---

    def call(self, input: Input, run_manager: Optional[CallbackManagerForChainRun]) -> Output:
        """
        Abstract method that must be implemented by subclasses.

        This method is designed to handle calls with pydantic input.
        """
        raise NotImplementedError("To implement, Does not support non async call")

    def fallback(
        self,
        input: Input,
        run_manager: Optional[CallbackManagerForChainRun],
        errors: List[Exception],
    ) -> Output:
        """
        Needs to be implemented by subclasses if `fallback_max_retries` is set.

        This method is designed to handle any fallback operations in the event of an error during the call.
        """
        raise NotImplementedError("Not implemented, `fallback_max_retries` with True but `fallback` is not implemented.")

    # Invoke
    # ---
    async def ainvoke_as_model(
        self,
        input: Union[Input, Dict[str, Any]],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Output:
        """
        Asynchronous method that invokes the chain with the given input and configuration.

        Returns:
            The output of the chain as a pydantic model.
        """
        input = self._convert_input_model_to_dict(input)

        logger.info(
            input,
            extra={"title": "[Start] Invoke" + " : " + self._chain_type, "verbose": self.verbose},
        )

        output = await super().ainvoke(input=input, config=config, kwargs=kwargs)

        logger.info(
            output,
            extra={"title": "[End] Invoke" + " : " + self._chain_type, "verbose": self.verbose},
        )

        return self.OutputModelType(**output)

    def invoke_as_model(
        self,
        input: Union[Input, Dict[str, Any]],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Output:
        """
        Synchronous method that invokes the chain with the given input and configuration.

        Returns the output of the chain as a pydantic model.
        """
        input = self._convert_input_model_to_dict(input)

        logger.info(
            input,
            extra={"title": "[Start] Invoke" + " : " + self._chain_type, "verbose": self.verbose},
        )

        output = super().invoke(input=input, config=config, kwargs=kwargs)

        logger.info(
            output,
            extra={"title": "[End] Invoke" + " : " + self._chain_type, "verbose": self.verbose},
        )
        # Convert the output to a pydantic model
        return self.OutputModelType(**output)

    async def ainvoke(
        self,
        input: Dict[str, Any],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Asynchronous method that invokes the chain with the given input and configuration.

        Returns the output of the chain as a dictionary.
        """
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
        """
        Synchronous method that invokes the chain with the given input and configuration.

        Returns the output of the chain as a dictionary.
        """
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

    def _fail_fallback_output(
        self,
        input: Input,
        retries: int,
        previous_errors: List[Exception],
    ) -> Output:
        """
        Raises a MaxRetryError if the fallback has reached the maximum number of retries.
        """
        raise MaxRetryError(
            f"Retry Cycle {retries} without success, errors: {previous_errors}",
        )

    # Call
    # ---
    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """
        Asynchronous method that invokes the chain with the given input and configuration.

        If the chain fails, it will retry the fallback until it reaches the maximum number
        of retries. If the fallback fails, it will raise a MaxRetryError.
        """
        input_model = self.InputModelType(**inputs)
        try:
            ouput_model = await self.acall(input=input_model, run_manager=run_manager)
            return self._convert_output_model_to_dict(ouput_model)

        except Exception as origin_error:
            if self.fallback_max_retries:
                # Retry the fallback with the input model until it reaches the maximum number of retries
                retries = 0
                retry_errors: List[Exception] = [origin_error]
                while retries < self.fallback_max_retries:
                    try:
                        ouput_model = await self.afallback(
                            input_model,
                            run_manager,
                            retry_errors,
                        )
                        return self._convert_output_model_to_dict(ouput_model)

                    except Exception as error:
                        retry_errors.append(error)
                        retries += 1
                        logger.warning(
                            retry_errors,
                            extra={"title": "[RETRY] _acall" + " : " + self._chain_type},
                        )

            # If the fallback failed, raise a PydanticChainError
            chain_error: PydanticChainError
            if isinstance(origin_error, PydanticChainError):
                track = self._chain_type + " -> " + origin_error.origin
                chain_error = PydanticChainError(track, origin_error.error)
            else:
                chain_error = PydanticChainError(self._chain_type, origin_error)

            logger.error(
                chain_error,
                stack_info=True,
                extra={"title": "[ERROR] _acall" + " : " + self._chain_type},
            )

            raise chain_error

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """
        Synchronous method that invokes the chain with the given input and configuration.

        If the chain fails, it will retry the fallback until it reaches the maximum number
        of retries. If the fallback fails, it will raise a MaxRetryError.
        """
        input_model = self.InputModelType(**inputs)

        try:
            ouput_model = self.call(input=input_model, run_manager=run_manager)
            return self._convert_output_model_to_dict(ouput_model)

        except Exception as origin_error:
            if self.fallback_max_retries:
                retries = 0
                retry_errors: List[Exception] = [origin_error]
                while retries < self.fallback_max_retries:
                    try:
                        ouput_model = self.fallback(
                            input_model,
                            run_manager,
                            retry_errors,
                        )
                        return self._convert_output_model_to_dict(ouput_model)

                    except Exception as error:
                        retry_errors.append(error)
                        retries += 1
                        logger.warning(
                            retry_errors,
                            extra={"title": "[RETRY] _acall" + " : " + self._chain_type},
                        )

            chain_error: PydanticChainError
            if isinstance(origin_error, PydanticChainError):
                track = self._chain_type + " -> " + origin_error.origin
                chain_error = PydanticChainError(track, origin_error.error)
            else:
                chain_error = PydanticChainError(self._chain_type, origin_error)

            logger.error(
                chain_error,
                stack_info=True,
                extra={"title": "[ERROR] _acall" + " : " + self._chain_type},
            )

            raise chain_error

    # Helpers
    # ---

    def get_custom_pydantic_parser(
        self,
        basemodel_type: Type[BaseModel],
        basemodel_linked_refs_types: List[Type[BaseModel]] = [],
        post_validation: Optional[Callable[[BaseModel], bool]] = None,
    ) -> PydanticOutputParser[BaseModel]:
        """
        This function is used to get a PydanticOutputParser instance.

        Parameters:

        - basemodel_type (Optional[Type[BaseModel]], optional):
            The type of the base model. If not provided, Output Model of the class is used.

        - basemodel_linked_refs_types (List[Type[BaseModel]], optional):
            A list of types for linked references on the Basemodel `basemodel_type`.
            Make an auto the Json generation of this one.

        - post_validation (Optional[Callable[[BaseModel], bool]], optional):
            A function to be called after validation. If not provided, no function is called.

        """

        return PydanticOutputParser(
            pydantic_object=basemodel_type,
            basemodel_linked_refs_types=basemodel_linked_refs_types,
            post_validation=post_validation,
        )

    def get_output_pydantic_parser(
        self,
        basemodel_linked_refs_types: List[Type[BaseModel]] = [],
        post_validation: Optional[Callable[[Output], bool]] = None,
    ) -> PydanticOutputParser[Output]:
        """
        This function is used to get a PydanticOutputParser instance.

        Parameters:

        - basemodel_type (Optional[Type[BaseModel]], optional):
            The type of the base model. If not provided, Output Model of the class is used.

        - basemodel_linked_refs_types (List[Type[BaseModel]], optional):
            A list of types for linked references on the Basemodel `basemodel_type`.
            Make an auto the Json generation of this one.

        - post_validation (Optional[Callable[[BaseModel], bool]], optional):
            A function to be called after validation. If not provided, no function is called.

        """
        return PydanticOutputParser(
            pydantic_object=self.OutputModelType,
            basemodel_linked_refs_types=basemodel_linked_refs_types,
            post_validation=post_validation,
        )

    # Private Convert
    # ---

    def _convert_input_model_to_dict(self, input: Union[Input, Dict[str, Any], BaseModel]) -> Dict[str, Any]:
        if isinstance(input, BaseModel):
            return input.dict()

        return input

    def _convert_output_model_to_dict(self, output: Output) -> Dict[str, Any]:
        if not isinstance(output, self.OutputModelType):
            raise TypeError(f"Error Output not typeof {self.OutputModelType}, Got: {type(output)}")
        return output.dict()
