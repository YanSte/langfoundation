from abc import abstractmethod
from enum import StrEnum
from functools import lru_cache
import logging
from typing import Any, Dict, Optional, Type

from langchain_core.language_models.chat_models import BaseChatModel
from langfoundation.utils.py.abc_enum import ABCEnumMeta
from langfoundation.utils.pydantic.frozen_dict import FrozenConfig


logger = logging.getLogger(__name__)


@lru_cache(maxsize=None)
def _get_llm_instance(
    provider: Type[BaseChatModel],
    config: FrozenConfig,
) -> BaseChatModel:
    """
    Returns an instance of the LLMs Chat Models with the given configuration.

    Note: FrozenConfig is to pass a dict into LRU Cache. (LRU Cache doesn't accept dict)
    """
    global_config = config.value
    logger.info(
        global_config,
        extra={
            "title": "[Cache LLM ChatModel]",
            "verbose": True,
        },
    )
    return provider(**global_config)


class BaseChatModelProvider(StrEnum, metaclass=ABCEnumMeta):
    """
    Base class for all langchain chat model providers.

    The difference between two types of language models available in Langchain:

    - LLMs Models: Simpler models that take a single string as input and return a string as output.
    Example: Prompt with full text.

    - LLMs Chat Models: Models that can handle a sequence of messages as input and return a “chat message object” as output.
    This object includes the message content and the role of the sender.
    Example:
    ```
    {"role": "system", "content": "Your are..", "role": "user", "content": "Hello .."}
    ```
    [See here](https://www.promptingguide.ai/models/chatgpt.en#conversations-with-chatgpt)
    """

    @property
    @abstractmethod
    def provider(self) -> Type[BaseChatModel]:
        """
        The type of the chat model provider.
        """

    @property
    def has_structured_output(self) -> bool:
        """
        Whether the provider supports function calling.

        [Docs](https://python.langchain.com/docs/integrations/chat/)
        """
        return False

    @property
    def has_json_mode(self) -> bool:
        """
        Whether the provider supports json mode.

        [Docs](https://python.langchain.com/docs/integrations/chat/)
        """
        return False

    def _config(
        self,
        temperature: float = 0.7,
        top_p: float = 1,
        max_tokens: Optional[int] = None,
        streaming: bool = True,
        extra_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        A dictionary of configuration options for the provider.
        """
        global_config: Dict[str, Any] = {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "model": self.value,
            "model_name": self.value,
            "streaming": streaming,
        }
        global_config.update(extra_config or {})
        return global_config

    def model(
        self,
        temperature: float = 0.7,
        top_p: float = 1,
        max_tokens: Optional[int] = None,
        streaming: bool = True,
        cache: bool = True,
        extra_config: Optional[Dict[str, Any]] = None,
    ) -> BaseChatModel:
        """
        Get a chat model based on the provider.

        Args:
        - temperature: The temperature for the chat model.
        - top_p: The top p value for the chat model.
        - max_tokens: The maximum number of tokens for the chat model.
        - streaming: Whether to stream the model or not.
        - cache: Whether to cache the model or not.
        - extra_config: Extra configuration options for the provider.
        """
        # TODO: Base value in provider to map the dict
        global_config = self._config(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            streaming=streaming,
            extra_config=extra_config,
        )

        # If cache is True, use the cached model
        if cache:
            return _get_llm_instance(
                provider=self.provider,
                config=FrozenConfig(value=global_config),
            )
        # If cache is False, create a new model
        else:
            return self.provider(**global_config)
