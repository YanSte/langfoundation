from typing import Any, Union

from pydantic import BaseModel

from langfoundation.modelhub.rerank.base import BaseRerankProvider
from langfoundation.modelhub.rerank.rank.base import BaseRerankModel


class RerankConfig(BaseModel):
    model_provider: Union[Any, BaseRerankProvider]

    top_n: int

    def model(self) -> BaseRerankModel:
        return self.model_provider.model(top_n=self.top_n)

    class Config:
        arbitrary_types_allowed = True
