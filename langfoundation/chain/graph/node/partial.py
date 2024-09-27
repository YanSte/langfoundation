from __future__ import annotations

from typing import Any, Callable, Union

from pydantic import BaseModel, Extra


class Partial(BaseModel):
    """
    A partial model is a model that can be used to create a full model by filling in the missing fields.

    This model can be used to create a full model by filling in the missing fields.
    Fields that are not present in the partial model will be filled in with the default values specified in the full model.

    Example:
    >>> from langpydantic import Partial
    >>> class MyModel(BaseModel):
    ...     a: str
    ...     b: str
    >>> partial_model = Partial(a="a")
    >>> full_model = MyModel(**partial_model.dict(exclude_none=True))
    >>> full_model.b
    ''
    """

    def __init__(
        self,
        **kwargs: Union[str, Callable[[], str], Any],
    ):
        super().__init__(**kwargs)

    class Config:
        arbitrary_types_allowed = True
        extra = Extra.allow
