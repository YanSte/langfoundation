import codecs
from enum import auto, StrEnum
import json
from typing import Any, cast, Dict, get_args, Iterator, List, Type, Union

from langchain_core.tools import StructuredTool

from pydantic import BaseModel


def required_fields(model: Type[BaseModel], recursive: bool = False) -> Iterator[str]:
    """
    Retrieve the names of required fields in a Pydantic model.

    This function takes a Pydantic model and returns the names of all required fields
    in that model. If the `recursive` argument is True, it will also return required
    fields for nested models.
    """

    # Iterate through all fields in the model
    for name, field in model.model_fields.items():
        # If the field is not required, skip it
        if not field.is_required():
            continue

        # Get the type of the field
        t = field.annotation

        # If the field is a nested model and we want to recurse into it
        if recursive and isinstance(t, type) and issubclass(t, BaseModel):
            # Yield all required fields in the nested model
            yield from required_fields(t, recursive=True)
        else:
            # Otherwise, just yield the name of the field
            yield name


def optional_fields(model: Type[BaseModel], recursive: bool = False) -> Iterator[str]:
    """
    Retrieve the names of optional fields in a Pydantic model.

    This function takes a Pydantic model and returns the names of all optional fields
    in that model. If the `recursive` argument is True, it will also return optional
    fields for nested models.

    Args:
        model (Type[BaseModel]): The Pydantic model to check.
        recursive (bool, optional): Whether to check nested models. Defaults to False.

    Yields:
        Iterator[str]: The names of the optional fields.
    """
    # Iterate through all fields in the model
    for name, field in model.model_fields.items():
        # If the field is not required, skip it
        if not field.is_required():
            continue

        # Get the type of the field
        t = field.annotation

        # If the field is a nested model and we want to recurse into it
        if recursive and isinstance(t, type) and issubclass(t, BaseModel):
            # Yield all required fields in the nested model
            yield from required_fields(t, recursive=True)
        else:
            # Otherwise, just yield the name of the field
            yield name


def populate_missing_optional_fields(model: Type[BaseModel], inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Populate missing optional fields in the input dictionary with None.

    This function takes a Pydantic model and an input dictionary, and populates any
    missing optional fields in the input dictionary with None.

    Args:
        model (Type[BaseModel]): The Pydantic model to check.
        inputs (Dict[str, Any]): The input dictionary to populate.

    Returns:
        Dict[str, Any]: The populated input dictionary.
    """
    # Get the set of optional fields for the model
    optional_fields_set = set(optional_fields(model))

    # Iterate over the optional fields that are not present in the input dictionary
    for key in optional_fields_set - set(inputs.keys()):
        # Populate the missing optional field with None
        inputs[key] = None

    # Return the populated input dictionary
    return inputs


def _get_class_from_ref(ref: str, basemodel_linked_refs_types: List[Type[BaseModel]]) -> Type[BaseModel]:
    """Retrieve the Pydantic model class from a reference string."""

    # Extract the class name from the reference string
    class_name = ref.split("/")[-1]

    for type_ in basemodel_linked_refs_types:
        if type_.__name__ == class_name:
            return type_
    else:
        # Retrieve the class from the globals() dictionary
        cls = globals().get(class_name)
        if cls is None:
            raise ValueError(
                f"""Error with '{class_name}' not found in the current globals.
                Maybe you forgot that internal class in Basemodel class needs to be pass as refs with 'basemodel_linked_refs_types'.
                """
            )
        return cls


class FieldType(StrEnum):
    """
    A FieldType is a field type that can be rendered.

    The following types are supported:
        - DEFAULT: The default value of the field.
        - DESCRIPTION: The description of the field.
    """

    DEFAULT = auto()
    DESCRIPTION = auto()

    @staticmethod
    def custom(field_key: str) -> str:
        """
        Create a custom FieldType.

        Args:
            field_key (str): The value of the custom FieldType.

        Returns:
            str: The custom FieldType.
        """
        return field_key


def _get_description_or_value(render_type: Union[FieldType, str], _dict: Dict[str, str]) -> str:
    """
    Retrieve the 'default' value or 'description' from a dictionary.
    """
    field_key: str = render_type if isinstance(render_type, str) else cast(FieldType, render_type).value

    return _dict.get(field_key, "")


def get_dict_schema_with_field_value(
    render_type: Union[FieldType, str],
    model: Type[BaseModel],
    basemodel_linked_refs_types: List[Type[BaseModel]] = [],
) -> Dict[str, Any]:
    def get_descriptions(properties: Dict[str, Any], render_type: Union[FieldType, str]) -> Dict[str, Any]:
        descriptions: Dict[str, Any] = {}

        for field, info in properties.items():
            descriptions[field] = _get_description_or_value(render_type, info)
            is_use_default = True if render_type is FieldType.DEFAULT and "default" in info else False
            if "type" in info and not is_use_default:
                if info["type"] == "array" and "items" in info and "$ref" in info["items"]:
                    base_model_type = _get_class_from_ref(
                        info["items"]["$ref"],
                        basemodel_linked_refs_types,
                    )
                    descriptions[field] = [
                        get_dict_schema_with_field_value(
                            render_type,
                            base_model_type,
                            basemodel_linked_refs_types,
                        )
                    ]

        return descriptions

    try:
        schema = model.model_json_schema()
    except Exception as e:
        raise ImportError(
            f"""Error occurs with {model} shema.
            Possibly due to definition not from Basemodel

            Or

            Possibly due to definition of class A with dependency of class B.
            Make sure to declare you class {model}:
            ```
            class B

            class A
               b: b
            ```

            Got {e}
            """
        )

    properties = schema.get("properties", {})
    return get_descriptions(properties, render_type)


def render_json_schema_with_field_value(
    render_type: Union[FieldType, str],
    basemodel_type: Type[BaseModel],
    basemodel_linked_refs_types: List[Type[BaseModel]] = [],
) -> str:
    formatted_schema = get_dict_schema_with_field_value(
        render_type,
        basemodel_type,
        basemodel_linked_refs_types,
    )

    output_format = json.dumps(formatted_schema)

    # Unicode escape sequences non-ASCII characters using their Unicode code points. Replace \u0022.
    output_format = codecs.decode(output_format, "unicode_escape")

    # Remove the \ characters
    output_format = output_format.replace("\\", "")

    return output_format


def get_type_base_generic(
    class_object: Any,
    type_generic_index: int,
    type_generic_inner_index: int,
) -> Type[BaseModel]:
    """
    Retrieves the base type origin from the given class object based on its original bases.

    This method is used to extract the base type of a generic class. It takes a class object,
    an index of the base class, and an index of the inner type argument.

    Examples:
        >>> class MyClass(Generic[T, U, K], BaseModel):
        ...     pass
        >>> get_type_base_generic(MyClass, 0, 0)
        <class 'T'>
        >>> get_type_base_generic(MyClass, 0, 1)
        <class 'T'>

        >>> class MyClass2(Other, Generic[T, U, K], BaseModel):
        ...     pass
        >>> get_type_base_generic(MyClass, 1, 0)
        <class 'T'>
        >>> get_type_base_generic(MyClass, 1, 1)
        <class 'T'>
    """
    # First loop through all parent classes and if any of them is
    # a pydantic model, we will pick up the generic parameterization
    # from that model via the __pydantic_generic_metadata__ attribute.
    base = class_object.__class__.mro()[type_generic_index]
    if hasattr(base, "__pydantic_generic_metadata__"):
        metadata = base.__pydantic_generic_metadata__
        if "args" in metadata:
            return metadata["args"][type_generic_inner_index]

    # If we didn't find a pydantic model in the parent classes,
    # then loop through __orig_bases__. This corresponds to
    # Runnables that are not pydantic models.
    cls = class_object.__class__.__orig_bases__[type_generic_index]
    type_args = get_args(cls)
    if type_args:
        return type_args[type_generic_inner_index]

    raise TypeError(
        f"Runnable {class_object.get_name()} doesn't have an inferable InputType. "
        "Override the InputType property to specify the input type."
    )


def render_tools_text_names(tools: List[StructuredTool], separator: str = "\n") -> str:
    """Render the tool name and description in plain text."""
    return separator.join([f"{tool.name}" for tool in tools])


def render_tools_names_descriptions(tools: List[StructuredTool], separator: str = "\n") -> str:
    """Render the tool name and description in plain text."""
    return separator.join([f"{tool.name}: {tool.description}" for tool in tools])


def render_tools_names_descriptions_args(
    render_type: Union[FieldType, str],
    tools: List[StructuredTool],
    separator: str = "\n",
) -> str:
    """Render the tool name, description, and args in plain text."""
    tool_strings = []
    for tool in tools:
        if not tool.args_schema:
            raise ValueError(f"Define the pydantic 'args_schema' to the tool {tool.name} to make outputformat avaliable.")
        rendered_tools = render_json_schema_with_field_value(
            render_type=render_type,
            basemodel_type=tool.args_schema,
        )
        tool_strings.append(f"- '{tool.name}': {tool.description} with args:\n```json\n{rendered_tools}```")
    return separator.join(tool_strings)
