import logging
from typing import Optional

from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


class ChatModelParams(BaseModel):
    """
    Configuration for the chat model.
    """

    temperature: float = Field(
        default=0.7,
        description="The temperature to use when generating text.",
    )
    top_p: float = Field(
        default=1,
        description="The top-p value to use when generating text.",
    )
    max_tokens: Optional[int] = Field(
        default=None,
        description="The maximum number of tokens to generate.",
    )
