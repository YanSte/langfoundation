from abc import ABCMeta
from enum import EnumMeta
from typing import Any, Type


class ABCEnumMeta(ABCMeta, EnumMeta):
    """
    Metaclass that combines the functionality of ABCMeta and EnumMeta.
    Ensures that abstract methods in the enum classes are implemented.

    Args:
        mcls (Type['ABCEnumMeta']): The metaclass.
        *args: Variable length argument list.
        **kw: Arbitrary keyword arguments.

    Returns:
        Type[Any]: A new instance of an enum class that cannot be instantiated if it has abstract methods.
    """

    def __new__(mcls: Type["ABCEnumMeta"], *args: Any, **kw: Any) -> Type[Any]:
        abstract_enum_cls = super().__new__(mcls, *args, **kw)
        # Only check abstractions if members were defined.
        if abstract_enum_cls._member_map_:
            try:  # Handle existence of undefined abstract methods.
                absmethods = list(abstract_enum_cls.__abstractmethods__)
                if absmethods:
                    missing = ", ".join(f"{method!r}" for method in absmethods)
                    plural = "s" if len(absmethods) > 1 else ""
                    raise TypeError(
                        f"cannot instantiate abstract class {abstract_enum_cls.__name__!r}" f" with abstract method{plural} {missing}"
                    )
            except AttributeError:
                pass
        return abstract_enum_cls
