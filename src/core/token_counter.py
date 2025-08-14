"""
Token counting using Gemini API
Uses free Gemini token counting API, falls back to estimation
"""

from typing import Union, List, Dict, Any, Optional
from enum import Enum


class CountingMethod(Enum):
    """Token counting method used"""
    GEMINI_API = "gemini_api"
    CHARACTER_ESTIMATION = "character_estimation"


def count_tokens(
    model_name: str,
    content: Union[str, List[Dict], Dict],
    system: Optional[str] = None,
    **kwargs
) -> int:
    """
    Count tokens using Gemini API or estimation fallback.

    Args:
        model_name: Model identifier (e.g., "gemini-2.5-pro", "gemini-3-pro-preview")
        content: Text string or structured message array
        system: Optional system instruction
        **kwargs: Additional provider-specific options

    Returns:
        Token count (integer)

    Examples:
        >>> count_tokens("gemini-2.5-pro", "Hello world")
        2
    """
    # Use Gemini API for Gemini models, estimation for others
    model_lower = model_name.lower()

    if "gemini" in model_lower:
        return _count_gemini_tokens(model_name, content, **kwargs)
    else:
        return _count_estimation(content, system)


def _count_gemini_tokens(
    model_name: str,
    content: Union[str, Any],
    **kwargs
) -> int:
    """
    Count tokens using official Gemini API (FREE).

    Reuses existing GeminiVertexLLM implementation.
    Falls back to estimation on any error.
    """
    try:
        from .llm import GeminiVertexLLM

        # Create instance with specified model
        gemini = GeminiVertexLLM(model_name=model_name)

        # Use existing count_tokens method
        return gemini.count_tokens(content)

    except Exception as e:
        print(f"[Token Counter] Gemini API error, using estimation: {e}")
        return _count_estimation(content, None)


def _count_estimation(
    content: Union[str, List, Dict],
    system: Optional[str] = None
) -> int:
    """
    Fallback character-based token estimation.

    Uses the common heuristic: 1 token ~ 4 characters
    """
    total_chars = 0

    # Content
    if isinstance(content, str):
        total_chars += len(content)
    elif isinstance(content, (list, dict)):
        total_chars += len(str(content))

    # System instruction
    if system:
        total_chars += len(system)

    return total_chars // 4
