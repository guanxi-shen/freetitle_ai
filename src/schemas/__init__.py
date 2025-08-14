"""
Schema registry for structured output.
Auto-selects Gemini or GPT schemas based on model type.
"""

import importlib
from typing import Dict, Any


def get_schema(agent_name: str, model_type: str) -> Dict[str, Any]:
    """
    Load the appropriate schema for an agent based on model type.

    Args:
        agent_name: Name of the agent (e.g., "script_agent", "orchestrator_agent")
        model_type: Model type ("gemini" or "gpt")

    Returns:
        Schema dictionary compatible with the specified model

    Raises:
        ValueError: If agent_name or model_type is invalid
        ImportError: If schema module doesn't exist
    """
    # Validate model type
    if model_type not in ["gemini", "gpt"]:
        raise ValueError(f"Invalid model_type: {model_type}. Must be 'gemini' or 'gpt'.")

    # Available schemas
    valid_agents = [
        "script_agent",
        "orchestrator_agent",
        "answer_parser_agent",
        "video_editor_agent"
    ]

    if agent_name not in valid_agents:
        raise ValueError(f"Invalid agent_name: {agent_name}. Must be one of {valid_agents}")

    try:
        # Dynamically import the schema module
        module = importlib.import_module(f"src.schemas.{agent_name}")

        # Select the appropriate schema
        if model_type == "gemini":
            schema = module.GEMINI_SCHEMA
        else:  # gpt
            schema = module.GPT_SCHEMA

        return schema

    except ImportError as e:
        raise ImportError(f"Schema module not found for agent '{agent_name}': {e}")
    except AttributeError as e:
        raise AttributeError(f"Schema not found in module '{agent_name}' for model '{model_type}': {e}")


# For backward compatibility - agents can still reference by name
AVAILABLE_SCHEMAS = [
    "script_agent",
    "orchestrator_agent",
    "answer_parser_agent",
    "video_editor_agent"
]
