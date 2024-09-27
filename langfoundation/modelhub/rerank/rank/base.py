from abc import ABC, abstractmethod
import logging
from typing import List, Tuple

from pydantic import BaseModel


logger = logging.getLogger(__name__)


class BaseRerankModel(BaseModel, ABC):
    """Interface for cross encoder models."""

    top_n: int

    @abstractmethod
    def rerank(
        self,
        query: str,
        docs: List[str],
    ) -> List[Tuple[int, float]]:
        """Score pairs' similarity.

        Args:
            text_pairs: List of pairs of texts.

        Returns:
            List of scores.
        """
