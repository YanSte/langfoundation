from typing import Any, Union

from langchain_core.embeddings import Embeddings
from pydantic import BaseModel

from langfoundation.modelhub.embedding.base import BaseEmbeddingProvider


class EmbeddingConfig(BaseModel):
    model_provider: Union[Any, BaseEmbeddingProvider]

    def model(self, cache: bool = True) -> Embeddings:
        return self.model_provider.model(cache)

    class Config:
        arbitrary_types_allowed = True
