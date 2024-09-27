from abc import abstractmethod
from enum import StrEnum
import logging
from typing import Any, Type, Union

from langchain_community.cross_encoders.base import BaseCrossEncoder
from toolkit.py.abc_enum import ABCEnumMeta

from langfoundation.modelhub.rerank.rank.base import BaseRerankModel


logger = logging.getLogger(__name__)


class BaseRerankProvider(StrEnum, metaclass=ABCEnumMeta):
    @property
    @abstractmethod
    def provider(
        self,
    ) -> Union[
        Type[BaseCrossEncoder],
        Type[Any],
    ]:
        raise NotImplementedError()

    @abstractmethod
    def model(
        self,
        top_n: int,
    ) -> BaseRerankModel:
        raise NotImplementedError()
