import logging
from functools import cache

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field

from langfoundation.modelhub.chat.params import ChatModelParams

logger = logging.getLogger(__name__)


class ChatModelConfiguration(BaseModel):
    """
    Configuration for the chat model.

    Difference between LLM Models and LLM Chat Models:

    - LLMs Models: These are simpler models that take a single string as input and return a string as output.

    - LLMs Chat Models: Models that can handle a sequence of messages as input and return a “chat message object” as output.
    This object includes the message content and the role of the sender.

    That use Chat Markup Language (ChatML for short)
    [See here](https://www.promptingguide.ai/models/chatgpt.en#conversations-with-chatgpt)
    """

    model: BaseChatModel = Field(
        description="The provider for the chat model.",
    )
    has_structured_output: bool = Field(
        default=True,
        description="Whether the provider supports function calling or structured output.",
    )
    has_json_mode: bool = Field(
        default=True,
        description="Whether the provider supports json mode.",
    )
    keep_model_params: bool = Field(
        default=False,
        description="""
        Whether to keep the defined model parameters.

        Else if ChatModelParams the params is override.
        """,
    )

    @cache
    def get_model(self, params: ChatModelParams) -> BaseChatModel:
        model_instance = self.model.model_copy(deep=True)
        self._update_model_in_place(model=model_instance, params=params)
        return model_instance

    def _update_model_in_place(self, model: BaseChatModel, params: ChatModelParams) -> None:
        try:
            new_values = params.model_dump(exclude_unset=True)

            for key, value in new_values.items():
                if self.keep_model_params and hasattr(model, key):
                    continue  # Skip updating this parameter
                setattr(model, key, value)

        except Exception as e:
            logger.error(f"Failed to update model in place: {e}", exc_info=True)
            raise
