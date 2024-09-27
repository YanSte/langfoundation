from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class BaseRecord(BaseModel):
    def update(self, **kwargs: Any) -> None:
        """
        Update the properties of the RetrieverRecord.

        Args:
            **kwargs: The properties to update.

        Returns:
            None
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"{self.__class__.__name__} does not have attribute '{key}'")
