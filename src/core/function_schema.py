"""
Function Schema Converter

Converts Python functions with type hints to tool schemas for LLM function calling.
"""

import inspect
from typing import Dict, List, Optional, Any, Union, get_type_hints, get_origin, get_args


def python_type_to_json_schema(python_type) -> Dict[str, Any]:
    """
    Convert Python type hint to JSON schema type.

    Handles:
    - Basic types: str, int, float, bool
    - Container types: dict, list, Dict, List
    - Optional types: Optional[T] → ["T", "null"]
    - Complex generics: Dict[str, str], List[Dict]
    """
    # Handle None type
    if python_type is type(None):
        return {"type": "null"}

    # Handle Optional[T] - unwrap and add null
    # Note: Optional[T] is actually Union[T, None]
    origin = get_origin(python_type)
    if origin is Union:
        args = get_args(python_type)
        # Check if it's Optional (Union with None)
        if type(None) in args:
            inner_types = [arg for arg in args if arg is not type(None)]
            if len(inner_types) == 1:
                inner_schema = python_type_to_json_schema(inner_types[0])
                # Add null to type
                if isinstance(inner_schema.get("type"), str):
                    inner_schema["type"] = [inner_schema["type"], "null"]
                return inner_schema

    # Handle Dict[K, V]
    # For Dict types, we allow additional properties (can't use strict mode with these)
    if origin in (dict, Dict):
        args = get_args(python_type)
        value_schema = python_type_to_json_schema(args[1]) if len(args) > 1 else {}
        return {
            "type": "object",
            "additionalProperties": value_schema or True
        }

    # Handle List[T]
    if origin in (list, List):
        args = get_args(python_type)
        item_schema = python_type_to_json_schema(args[0]) if args else {}
        return {
            "type": "array",
            "items": item_schema
        }

    # Basic type mapping
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        dict: "object",
        list: "array",
    }

    json_type = type_map.get(python_type)
    if json_type:
        # For lowercase dict (flexible object), allow any properties
        if json_type == "object":
            return {"type": json_type}  # No additionalProperties constraint
        return {"type": json_type}

    # Fallback - treat as flexible object
    return {"type": "object"}


def extract_param_description(docstring: str, param_name: str) -> str:
    """Extract parameter description from docstring (Google/NumPy style)"""
    if not docstring:
        return ""

    # Look for Args: section
    lines = docstring.split('\n')
    in_args = False

    for line in lines:
        stripped = line.strip()

        # Detect Args section
        if stripped.lower().startswith('args:'):
            in_args = True
            continue

        # Exit Args section
        if in_args and stripped and not stripped.startswith(param_name) and ':' in stripped:
            # Hit next section or param
            if not stripped.split(':')[0].strip().replace('_', '').isalnum():
                break

        # Extract description
        if in_args and stripped.startswith(f"{param_name}:"):
            desc = stripped.split(':', 1)[1].strip()
            return desc

    return ""


def extract_function_description(docstring: str) -> str:
    """Extract function description (first line before Args:)"""
    if not docstring:
        return ""

    lines = docstring.strip().split('\n')
    description_lines = []

    for line in lines:
        stripped = line.strip()
        if stripped.lower().startswith('args:'):
            break
        if stripped:
            description_lines.append(stripped)

    return ' '.join(description_lines)


def python_function_to_tool_schema(func: callable) -> Dict[str, Any]:
    """
    Convert Python function to LLM tool schema.

    Uses:
    - inspect.signature() for parameters
    - typing.get_type_hints() for types
    - Docstring parsing for descriptions

    Returns tool schema format:
    {
        "type": "function",
        "name": "...",
        "description": "...",
        "parameters": {...},
        "strict": True
    }
    """
    sig = inspect.signature(func)
    type_hints = get_type_hints(func)

    properties = {}
    required = []

    for param_name, param in sig.parameters.items():
        # Skip internal params
        if param_name.startswith('_'):
            continue

        # Get type hint
        param_type = type_hints.get(param_name, str)
        param_schema = python_type_to_json_schema(param_type)

        # Add description
        param_schema["description"] = extract_param_description(func.__doc__, param_name)

        # If parameter has default value, make it optional by adding null type
        if param.default != inspect.Parameter.empty:
            # Make type array if not already
            if isinstance(param_schema.get("type"), str):
                param_schema["type"] = [param_schema["type"], "null"]
            elif isinstance(param_schema.get("type"), list):
                if "null" not in param_schema["type"]:
                    param_schema["type"].append("null")

        properties[param_name] = param_schema

        # In strict mode, ALL parameters must be in required array
        required.append(param_name)

    return {
        "type": "function",
        "name": func.__name__,
        "description": extract_function_description(func.__doc__),
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False
        },
        "strict": False  # Disable strict mode - match Gemini's best-effort behavior
    }
