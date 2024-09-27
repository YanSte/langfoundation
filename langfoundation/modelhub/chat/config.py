from abc import ABC
from typing import Any, Dict, Optional, Union

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field

from langfoundation.modelhub.chat.base import BaseChatModelProvider


class ChatModelConfig(BaseModel, ABC):
    """
    Configuration for the chat model.

    Difference between LLM Models and LLM Chat Models:

    - LLMs Models: These are simpler models that take a single string as input and return a string as output.

    - LLMs Chat Models: Models that can handle a sequence of messages as input and return a “chat message object” as output.
    This object includes the message content and the role of the sender.

    That use Chat Markup Language (ChatML for short)
    [See here](https://www.promptingguide.ai/models/chatgpt.en#conversations-with-chatgpt)
    """

    model_provider: Union[Any, BaseChatModelProvider] = Field(
        description="The provider for the chat model.",
    )
    temperature: float = Field(
        default=0.7,
        description="The temperature for the chat model.",
    )
    top_p: float = Field(
        default=1,
        description="The top p value for the chat model.",
    )
    max_tokens: Optional[int] = Field(
        default=None,
        description="The maximum number of tokens for the chat model.",
    )
    extra_config: Optional[Dict[str, Any]] = None

    def model(self, streaming: bool) -> BaseChatModel:
        """
        Get the underlying chat model.

        Args:
        - streaming: Whether to stream the model or not.

        Returns:
        - The underlying chat model.
        """
        return self.model_provider.model(
            self.temperature,
            self.top_p,
            self.max_tokens,
            streaming,
        )

    @property
    def has_structured_output(self) -> bool:
        """
        Whether the provider supports function calling.
        """
        return self.model_provider.has_structured_output

    @property
    def has_json_mode(self) -> bool:
        """
        Whether the provider supports json mode.
        """
        return self.model_provider.has_json_mode

    class Config:
        arbitrary_types_allowed = True
