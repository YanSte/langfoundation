from typing import Any, Dict

from pydantic import BaseModel


class FrozenConfig(BaseModel):
    value: Dict[str, Any]

    def __hash__(self) -> int:
        def _hash_dict(d: Dict) -> frozenset:
            return frozenset((k, _hash_dict(v) if isinstance(v, dict) else v) for k, v in d.items())

        return hash(_hash_dict(self.value))
