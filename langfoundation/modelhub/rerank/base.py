from abc import ABC, abstractmethod
from typing import List, Tuple

from pydantic import BaseModel, Field


class BaseRerankModel(BaseModel, ABC):
    """
    Interface for Rerank models.

    Cross encoders are models that take in a query and a list of documents, and output a score for each document in the list.
    """

    top_n: int = Field(
        description="The number of top results to return from the reranker.",
    )

    @abstractmethod
    def rerank(
        self,
        query: str,
        docs: List[str],
    ) -> List[Tuple[int, float]]:
        """Score pairs' similarity.

        Args:
            query: The query string.
            docs: The list of document strings.

        Returns:
            A list of tuples of index of the document in the `docs` list and the similarity score.
        """
        raise NotImplementedError()

    @abstractmethod
    async def arerank(
        self,
        query: str,
        docs: List[str],
    ) -> List[Tuple[int, float]]:
        """Score pairs' similarity.

        Args:
            query: The query string.
            docs: The list of document strings.

        Returns:
            A list of tuples of index of the document in the `docs` list and the similarity score.
        """
        raise NotImplementedError()
