"""
Central registry for video generation tools

Single source of truth for:
- Tool metadata and naming
- Dynamic prompt templates
- Parameter schemas
- Provider routing
"""

import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


# Import provider classes
def _get_provider_classes():
    """
    Lazy import of provider classes
    Returns dict of provider class references
    """
    from .client_veo_google import GoogleVeoGenerator

    return {
        "GoogleVeoGenerator": GoogleVeoGenerator
    }


# ==============================================================================
# VIDEO TOOL REGISTRY
# ==============================================================================

VIDEO_TOOL_REGISTRY = {
    "google_veo_i2v": {
        # Core metadata
        "provider_class": "GoogleVeoGenerator",
        "task_id_type": "string",  # Operation names are strings

        # Naming breakdown
        "provider": "google",
        "model": "veo",
        "version": "3.1",
        "type": "i2v",

        # Dynamic prompt strings (for LLM injection)
        "tool_description": """- google_veo_i2v: Google Veo 3.1 single/dual-frame (PRIMARY)
  * Input: 1 frame (single-frame) OR 2 frames (dual-frame) + generation_prompt
  * Output: HD video, 8s duration
  * Best for: Highest quality via official Google API
  * Auto-detects mode: provide end_frame_path for dual-frame, omit for single-frame""",

        "selection_guidance": """Choose google_veo_i2v when:
  * Single-frame: Only need to match one storyboard frame (default choice)
  * Dual-frame: Storyboard provides 2 frames for smooth interpolation
  * Want highest quality via official Google Veo API
  * Using Google Cloud infrastructure""",

        # Script planning guidance (for script agent)
        "script_planning_guidance": """google_veo_i2v - Flexible single/dual-frame workflow (8s):
  * Single-frame mode: ONLY start_frame field (no end_frame)
    - Plan motion filling full 8-second duration
    - Avoid minimal movements that complete too quickly
  * Dual-frame mode: BOTH start_frame AND end_frame fields
    - Plan clear beginning and ending states with interpolation between
    - Best for: Transformation shots, controlled motion between states
  * Product shots: Avoid straight-to-camera angles with camera movement
  * Best for: Most standard shots, highest quality via official Google API""",

        # Parameter definition
        "required_params": ["generation_prompt", "start_frame_path"],
        "optional_params": {
            "end_frame_path": None,  # Optional - triggers dual-frame when provided
            "aspect_ratio": "vertical",
            "duration": 8,
            "scene": None,
            "shot": None
        },

        # Function schema (for LLM function calling)
        "function_schema": {
            "name": "google_veo_i2v",
            "description": "Generate video from storyboard frame(s) using Google Veo (supports single or dual-frame)",
            "parameters": {
                "type": "object",
                "properties": {
                    "generation_prompt": {
                        "type": "string",
                        "description": "Video generation prompt with motion description"
                    },
                    "start_frame_path": {
                        "type": "string",
                        "description": "GCS path (gs://) to starting frame"
                    },
                    "end_frame_path": {
                        "type": "string",
                        "description": "Optional GCS path (gs://) to ending frame (for dual-frame interpolation)"
                    },
                    "aspect_ratio": {
                        "type": "string",
                        "enum": ["vertical", "horizontal"],
                        "description": "Video aspect ratio"
                    },
                    "duration": {
                        "type": "integer",
                        "enum": [4, 6, 8],
                        "description": "Video duration in seconds (default: 8)"
                    }
                },
                "required": ["generation_prompt", "start_frame_path"]
            }
        }
    }
}


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def get_tool_info(tool_name: str) -> Optional[Dict[str, Any]]:
    """Get complete info for a tool"""
    return VIDEO_TOOL_REGISTRY.get(tool_name)


def get_available_tools() -> List[str]:
    """Get list of available tool names"""
    return list(VIDEO_TOOL_REGISTRY.keys())


def get_tools_description() -> str:
    """Get formatted description of all tools for prompts"""
    descriptions = []
    for tool_name, info in VIDEO_TOOL_REGISTRY.items():
        descriptions.append(info.get("tool_description", f"- {tool_name}: No description available"))
    return "\n\n".join(descriptions)


def get_selection_guidance() -> str:
    """Get combined selection guidance for all tools"""
    guidance = []
    for tool_name, info in VIDEO_TOOL_REGISTRY.items():
        if "selection_guidance" in info:
            guidance.append(f"### {tool_name}\n{info['selection_guidance']}")
    return "\n\n".join(guidance)


def get_script_planning_guidance() -> str:
    """Get combined script planning guidance for script agent"""
    guidance = []
    for tool_name, info in VIDEO_TOOL_REGISTRY.items():
        if "script_planning_guidance" in info:
            guidance.append(info["script_planning_guidance"])
    return "\n\n".join(guidance)


def get_function_schemas(tool_names: Optional[List[str]] = None) -> List[Dict]:
    """Get function schemas for specified tools (or all if none specified)"""
    if tool_names is None:
        tool_names = get_available_tools()

    schemas = []
    for name in tool_names:
        info = get_tool_info(name)
        if info and "function_schema" in info:
            schemas.append(info["function_schema"])

    return schemas


def get_provider_class(tool_name: str):
    """Get provider class for a tool"""
    info = VIDEO_TOOL_REGISTRY.get(tool_name)
    if not info:
        logger.warning(f"[Registry] Unknown tool '{tool_name}', falling back to google_veo_i2v")
        tool_name = "google_veo_i2v"
        info = VIDEO_TOOL_REGISTRY[tool_name]

    provider_classes = _get_provider_classes()
    return provider_classes.get(info["provider_class"])
