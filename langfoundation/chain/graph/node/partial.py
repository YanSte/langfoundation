from __future__ import annotations

from typing import Callable, Union

from pydantic import BaseModel


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
        **kwargs: Union[str, int, float, bool, Callable[[], str]],
    ):
        super().__init__(**kwargs)

    def get_value(self) -> dict:
        result = {}
        for key, value in self.__dict__.items():
            if callable(value):
                value = value()

            if value is not None:
                result[key] = str(value)

        return result

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"
