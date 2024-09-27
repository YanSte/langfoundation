import inspect
from typing import List


def has_method_implementation(
    method: str,
    cls: object,
    excluded: List[type] = [],
) -> bool:
    """
    Check if a method is implemented within a given class or its ancestors.

    Examples:
        Consider a class hierarchy where `BaseClass` defines a method `output` that raises
        `NotImplementedError`, and `SubClass` overrides `output` with its own implementation.

        ```python
        class Output:
            pass

        class BaseClass:
            def method(self, output: Any) -> Output:
                raise NotImplementedError()

        class SubClass(BaseClass):
            def method(self, output: Any) -> Output:
                # Implementation here
                return Output()

        # Check if `output` method is implemented in `BaseClass` and `SubClass`
        print(has_method_implementation('method', BaseClass))  # False
        print(has_method_implementation('method', SubClass))   # True

        # In your mother class, if the child has implemented the method.
        has_method_implementation('method', self.__class__) # True
        ```
    """
    excluded_class_names = [excluded_class.__name__ for excluded_class in excluded]
    # Get all the methods of the class
    object = inspect.getmembers(cls, inspect.isfunction)
    for name, member in object:
        class_name = member.__qualname__.split(".")[0]
        if method == name and class_name not in excluded_class_names:
            return True
    else:
        return False
