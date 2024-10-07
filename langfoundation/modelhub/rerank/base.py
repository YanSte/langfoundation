from abc import ABC, abstractmethod
from typing import List, Tuple

from pydantic import BaseModel


class BaseRerankModel(BaseModel, ABC):
    """
    Interface for Rerank models.

    Cross encoders are models that take in a query and a list of documents, and output a score for each document in the list.
    """

    @abstractmethod
    def rerank(
        self,
        query: str,
        docs: List[str],
        top_n: int,
    ) -> List[Tuple[int, float]]:
        """
        Score pairs' similarity.

        Args:
            query: The query string.
            docs: The list of document strings.
            top_n: The number of top results to return from the reranker.

        Returns:
            A list of tuples of index of the document in the `docs` list and the similarity score.
        """
        raise NotImplementedError()

    @abstractmethod
    async def arerank(
        self,
        query: str,
        docs: List[str],
        top_n: int,
    ) -> List[Tuple[int, float]]:
        """
        Score pairs' similarity.

        Args:
            query: The query string.
            docs: The list of document strings.
            top_n: The number of top results to return from the reranker.

        Returns:
            A list of tuples of index of the document in the `docs` list and the similarity score.
        """
        raise NotImplementedError()
