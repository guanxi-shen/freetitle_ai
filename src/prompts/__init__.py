"""Prompt templates for video generation agents

This module re-exports all prompts from their individual modules
to maintain backward compatibility with existing imports.
"""

# Import orchestrator prompts
from .orchestrator import (
    ORCHESTRATOR_PROMPT_TEMPLATE,
    ORCHESTRATOR_RETRY_PROMPT_TEMPLATE
)

# Import system prompts (answer parser and memory)
from .system import (
    ANSWER_SYNTHESIS_WITH_SUFFICIENCY_TEMPLATE,
    ANSWER_SYNTHESIS_BASE_PROMPT_TEMPLATE,
    MEMORY_SUMMARY_PROMPT_TEMPLATE,
    MEMORY_UPDATE_SUMMARY_PROMPT_TEMPLATE,
    MEMORY_UPDATE_CONVERSATION_PROMPT_TEMPLATE
)

# Import agent-specific prompts
from .script import SCRIPT_AGENT_PROMPT_TEMPLATE
from .character import CHARACTER_AGENT_PROMPT_TEMPLATE
# STORYBOARD_AGENT_PROMPT_TEMPLATE removed - storyboard uses orchestrator pattern (STORYBOARD_ORCHESTRATOR_PROMPT + STORYBOARD_SUB_AGENT_PROMPT)
from .supplementary import SUPPLEMENTARY_AGENT_PROMPT_TEMPLATE
from .video import VIDEO_AGENT_PROMPT_TEMPLATE, VIDEO_GENERATION_PROMPT_TEMPLATE
from .video_editor import VIDEO_EDITOR_PROMPT_TEMPLATE
from .audio import AUDIO_AGENT_PROMPT_TEMPLATE

# Import agent registry
from .registry import AGENT_REGISTRY

# Export all prompts (maintains backward compatibility)
__all__ = [
    # Orchestrator
    'ORCHESTRATOR_PROMPT_TEMPLATE',
    'ORCHESTRATOR_RETRY_PROMPT_TEMPLATE',
    # System
    'ANSWER_SYNTHESIS_WITH_SUFFICIENCY_TEMPLATE',
    'ANSWER_SYNTHESIS_BASE_PROMPT_TEMPLATE',
    'MEMORY_SUMMARY_PROMPT_TEMPLATE',
    'MEMORY_UPDATE_SUMMARY_PROMPT_TEMPLATE',
    'MEMORY_UPDATE_CONVERSATION_PROMPT_TEMPLATE',
    # Agents
    'SCRIPT_AGENT_PROMPT_TEMPLATE',
    'CHARACTER_AGENT_PROMPT_TEMPLATE',
    # 'STORYBOARD_AGENT_PROMPT_TEMPLATE',  # Removed - deprecated
    'SUPPLEMENTARY_AGENT_PROMPT_TEMPLATE',
    'VIDEO_AGENT_PROMPT_TEMPLATE',
    'VIDEO_GENERATION_PROMPT_TEMPLATE',
    'VIDEO_EDITOR_PROMPT_TEMPLATE',
    'AUDIO_AGENT_PROMPT_TEMPLATE',
    # Registry
    'AGENT_REGISTRY'
]
