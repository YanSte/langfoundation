from abc import ABC
import logging
from typing import Any, Dict, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


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
    extra: Optional[Dict[str, Any]] = None
