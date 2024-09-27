from abc import abstractmethod
from enum import StrEnum
from functools import lru_cache
import logging
from typing import Any, Dict, Type

from langchain_core.embeddings import Embeddings
from toolkit.py.abc_enum import ABCEnumMeta
from toolkit.pydantic.frozen_dict import FrozenConfig


logger = logging.getLogger(__name__)


@lru_cache(maxsize=None)
def _get_embedding_instance(
    provider: Type[Embeddings],
    config: FrozenConfig,
) -> Embeddings:
    """
    Returns an instance of the embedding model with the given configuration.

    Args:
    - provider: The class of the embedding model to instantiate.
    - config: The configuration to use for the embedding model.

    Note: FrozenConfig is to pass a dict into LRU Cache. (LRU Cache doesn't accept dict)
    """
    model = provider(**config.value)
    logger.info(
        config.value,
        extra={
            "title": "[Cache Model] " + " : " + str(type(model).__name__) + " : " + str(config),
            "verbose": True,
        },
    )
    return model


class BaseEmbeddingProvider(StrEnum, metaclass=ABCEnumMeta):
    """
    Base class for all embedding providers.
    """

    @property
    @abstractmethod
    def provider(self) -> Type[Embeddings]:
        """
        The class of the embedding model to instantiate.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def config(self) -> Dict[str, Any]:
        """
        The configuration to use for the embedding model.
        """
        raise NotImplementedError()

    def model(self, cache: bool = True) -> Embeddings:
        """
        Returns an instance of the embedding model with the given configuration.

        Args:
        - cache: Whether to use a cached version of the model. Defaults to True.

        Returns:
        - The instance of the embedding model.
        """
        if cache:
            return _get_embedding_instance(
                provider=self.provider,
                config=FrozenConfig(value=self.config),
            )
        else:
            return self.provider(**self.config)
