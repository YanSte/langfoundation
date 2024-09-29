from __future__ import annotations

from abc import ABC
import logging
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from langchain_core.callbacks import (
    AsyncCallbackManagerForChainRun,
)
from langchain_core.language_models import BaseLanguageModel
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers.base import BaseOutputParser
from langchain_core.prompts.base import BasePromptTemplate
from langchain_core.prompts.chat import SystemMessagePromptTemplate
from langchain_core.runnables import (
    RunnableConfig,
    RunnableLambda,
    RunnableSerializable,
)
from pydantic import BaseModel, Field

from langfoundation.callback.base.tags import Tags
from langfoundation.chain.graph.node.input import BaseInput
from langfoundation.chain.pydantic.chain import BasePydanticChain
from langfoundation.errors.max_retry import MaxRetryError
from langfoundation.modelhub.chat.config import ChatModelConfig
from langfoundation.parser.pydantic.parser import PydanticOutputParser
from langfoundation.utils.py.py_class import has_method_implementation


logger = logging.getLogger(__name__)

# "Input" must be a subclass of BaseModel.
Input = TypeVar("Input", bound=BaseInput)

# "Output" must be a subclass of BaseModel.
Output = TypeVar("Output", bound=BaseModel)


class BaseNodeChain(
    BasePydanticChain[Input, Output],
    ABC,
):
    """
    BaseNodeChain is an abstract class that extends from BasePydanticChain.
    It is designed to work with the Runnable Dict Chain of Lanchain, and allows for casting
    into Pydantic models without any disruptions to the updates of Langchain.

    The BasePydanticChain class can work with any types of Input and Output that are subclasses of
    the BaseModel from the pydantic library.

    To use this class, subclass it and override the 'acall' or 'call' methods.
    If you use 'fallback_max_retries' override 'afallback' or 'fallback' methods.
    Then, use the 'ainvoke_as_model' or 'invoke_as_model' methods to call the chain as
    Input Output BaseModel Pydantic.
    """

    # Model
    # ---

    llm_config: ChatModelConfig = Field(
        description=" Specifies the size of the model to be used for invocation.",
    )

    # Fallback
    # ---

    fallback_max_retries: int = Field(
        description="Maximum of retries allowed for the fallback.",
    )

    fallback_llm_config: ChatModelConfig = Field(
        description="The model size to be used for fallback.",
    )

    @property
    def is_display(self) -> bool:
        """
        Returns True if the node is a display node.
        """
        if not self.tags:
            return False
        return Tags.is_display_or_with_parser_tag(self.tags)

    @property
    def is_feedback(self) -> bool:
        """
        Returns True if the node is a feedback node.
        """
        if not self.tags:
            return False
        return Tags.FEEDBACK.value in self.tags

    @property
    def with_output_format(self) -> bool:
        """
        Whether or not to add the output format in the prompt.
        """
        return True

    # Prompt
    # ---

    @property
    def output_basemodel_linked_refs_types(self) -> List[Type[BaseModel]]:
        """
        List of types linked reference models in the output.
        When you Output with inner models, you have to provide the class to get the types of the linked reference models.
        For auto generation of the Output by Models.

        This property is useful in mapping the structure of the output data.

        Example:

        ```python
        class PlannerQuestion(BaseModel):
            raison: str = Field(
                prompt="blabla",
            )
            question: str = Field(
                prompt="blabla",
            )

        class PlannerOutput(BaseModel):
        # Here we have list of PlannerQuestion
        >>>    segmented_questions: List[PlannerQuestion] = Field(...)

        >>> Add PlannerQuestion
        @property
        def output_basemodel_linked_refs_types(self) -> List[Type[BaseModel]]:
            return [PlannerQuestion]
        ```
        """
        return []

    # Public
    # ---

    def get_config(
        self,
        run_manager: Optional[AsyncCallbackManagerForChainRun],
        exclude: List[Tags] = [],
    ) -> RunnableConfig:
        """
        Returns a RunnableConfig with the callbacks and tags of the node.
        """
        callbacks = run_manager.get_child() if run_manager else None

        return RunnableConfig(callbacks=callbacks, tags=list(self.tags or []))

    # Private
    # ---

    # Call
    # ---

    async def acall(  # type: ignore[override]
        self,
        input: Input,
        run_manager: Optional[AsyncCallbackManagerForChainRun],
    ) -> Output:
        """
        Calls the chain with the given input and run manager.
        """
        config = self.get_config(run_manager)

        llm = self._llm()

        parser = self._parser(
            input,
            config,
        )

        prompt_template, input_data = self._prompt(
            input,
            parser,
            config,
        )

        chain = self._chain(
            input,
            llm,
            parser,
            prompt_template,
            config,
        )

        return await chain.ainvoke(
            input=input_data,
            config=config,
            return_only_outputs=True,
        )

    async def afallback(
        self,
        input: Input,
        run_manager: Optional[AsyncCallbackManagerForChainRun],
        errors: List[Exception],
    ) -> Output:
        """
        The fallback method is a special case of the call method. It is used when the call method fails
        and there is a fallback defined in the node.
        """
        config = self.get_config(run_manager)

        llm = self._fallback_llm()

        parser = self._parser(
            input,
            config,
        )

        prompt_template, input_data = self._retry_prompt(
            input,
            parser,
            config,
            errors,
        )

        chain = self._chain(
            input,
            llm,
            parser,
            prompt_template,
            config,
        )

        return await chain.ainvoke(
            input=input_data,
            config=config,
            return_only_outputs=True,
        )

    # Chain
    # ---

    def _llm(
        self,
    ) -> BaseChatModel:
        """
        Returns the output parser.
        """
        llm = self.llm_config.model

        if hasattr(llm, "streaming"):
            llm.streaming = self.is_display  # type: ignore

        # We assum if not display, then it is not streaming
        llm.disable_streaming = not self.is_display

        return llm

    def _fallback_llm(
        self,
    ) -> BaseChatModel:
        """
        Returns the output parser.
        """
        llm = self.fallback_llm_config.model

        if hasattr(llm, "streaming"):
            llm.streaming = self.is_display  # type: ignore

        # We assum if not display, then it is not streaming
        llm.disable_streaming = not self.is_display

        logger.info(
            {"Model Config:": llm._identifying_params},
            extra={
                "title": "[Invoke] " + " : " + self._chain_type,
                "verbose": True,
            },
        )
        return llm

    def _prompt(
        self,
        input: Input,
        parser: BaseOutputParser,
        config: RunnableConfig,
        previous_errors: List[Exception] = [],
    ) -> Tuple[BasePromptTemplate, Dict]:
        """
        Prompt template and input data.
        """
        input_data = input.prompt_arg(None).model_dump()
        prompt_template = input.prompt_template

        if self.with_output_format:
            if has_method_implementation(
                BaseOutputParser.get_format_instructions.__name__,
                parser.__class__,
                excluded=[BaseOutputParser],
            ):
                output_format = parser.get_format_instructions()
                # Add the output format instructions to the input data
                input_data["output_format"] = output_format

            prompt_template = input.prompt_template + SystemMessagePromptTemplate.from_template("\n\n{output_format}")

        return (prompt_template, input_data)

    def _retry_prompt(
        self,
        input: Input,
        parser: BaseOutputParser,
        config: RunnableConfig,
        previous_errors: List[Exception] = [],
    ) -> Tuple[BasePromptTemplate, Dict]:
        """
        Returns the retry prompt template and input data.
        """
        input_data = input.prompt_arg(previous_errors).dict()

        if has_method_implementation(
            BaseOutputParser.get_format_instructions.__name__,
            parser.__class__,
            excluded=[BaseOutputParser],
        ):
            output_format = parser.get_format_instructions()
            input_data["output_format"] = output_format

        return (input.retry_prompt_template, input_data)

    def _parser(
        self,
        input: Input,
        config: RunnableConfig,
    ) -> BaseOutputParser:
        """
        Returns the output parser.
        """
        return self.get_output_pydantic_parser()

    def _chain(
        self,
        input: Input,
        llm: BaseLanguageModel,
        parser: BaseOutputParser,
        prompt: BasePromptTemplate,
        config: RunnableConfig,
    ) -> RunnableSerializable:
        """
        Returns the chain.
        """
        chain: RunnableSerializable

        if not self.is_display and self.llm_config.has_structured_output:
            # If not display (streaming) use func calling (func calling can't streaming output)
            chain = prompt | llm.with_structured_output(
                schema=parser.OutputType,
            )
            logger.info(
                "Use Function Calling",
                extra={
                    "title": "[Invoke] " + " : " + self._chain_type,
                    "verbose": self.verbose,
                },
            )
        elif self.is_display and self.llm_config.has_json_mode:
            # If display (streaming) use json mode (json mode give streaming output)
            chain = prompt | llm.with_structured_output(
                schema=parser.OutputType,
                method="json_mode",
            )
            logger.info(
                "Use Json Mode",
                extra={
                    "title": "[Invoke] " + " : " + self._chain_type,
                    "verbose": self.verbose,
                },
            )
        else:
            logger.info(
                "Use without stuctured output",
                extra={
                    "title": "[Invoke] " + " : " + self._chain_type,
                    "verbose": self.verbose,
                },
            )
            chain = prompt | llm | parser

        if has_method_implementation(
            BaseNodeChain._output.__name__,
            self.__class__,
            excluded=[BaseNodeChain],
        ):

            def ouput(output: BaseModel) -> Output:
                return self._output(input, output, config)

            return chain | RunnableLambda(ouput)

        else:
            return chain

    def _fail_fallback_output(
        self,
        input: Input,
        retries: int,
        previous_errors: List[Exception],
    ) -> Output:
        """
        Raises a MaxRetryError if the fallback has reached the maximum number of retries.
        """
        raise MaxRetryError(f"Retry Cycle {retries} without success, errors: {previous_errors}")

    def _output(
        self,
        input: Input,
        output: Union[Output, BaseModel, Any],
        config: RunnableConfig,
    ) -> Output:
        """
        Returns the output of the node.
        """
        raise NotImplementedError()

    # Parser
    # ---

    def get_custom_pydantic_parser(
        self,
        basemodel_type: Type[BaseModel],
        basemodel_linked_refs_types: List[Type[BaseModel]] = [],
        pre_transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        post_validation: Optional[Callable[[BaseModel], bool]] = None,
    ) -> PydanticOutputParser[BaseModel]:
        """
        Get a custom PydanticOutputParser for a given BaseModel.

        Args:
            basemodel_type: The type of the BaseModel to parse.
            basemodel_linked_refs_types: A list of types linked reference models in the output.
            pre_transform: A function to pretransform the output before cast in object.
            post_validation: A function given a BaseModel Output and validate it.

        Returns:
            A PydanticOutputParser for the given BaseModel.
        """
        return PydanticOutputParser(
            pydantic_object=basemodel_type,
            basemodel_linked_refs_types=basemodel_linked_refs_types,
            pre_transform=pre_transform,
            post_validation=post_validation,
        )

    def get_output_pydantic_parser(
        self,
        pre_transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        post_validation: Optional[Callable[[Output], bool]] = None,
    ) -> PydanticOutputParser[Output]:
        """
        This function is used to get a PydanticOutputParser with defined Output of the class.

        Args:
            basemodel_type: The type of the BaseModel to parse.
            basemodel_linked_refs_types: A list of types linked reference models in the output.
            pre_transform: A function to pretransform the output before cast in object.
            post_validation: A function given a BaseModel Output and validate it.

        Returns:
            A PydanticOutputParser for the given BaseModel.
        """
        return PydanticOutputParser(
            pydantic_object=self.OutputModelType,
            basemodel_linked_refs_types=self.output_basemodel_linked_refs_types,
            pre_transform=pre_transform,
            post_validation=post_validation,
        )
