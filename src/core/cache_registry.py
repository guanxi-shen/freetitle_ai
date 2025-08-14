"""
Gemini Context Caching Registry

Manages cache lifecycle for Gemini LLM calls with timestamp-based invalidation.
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, List
from google import genai
from google.genai.types import Content, Part, CreateCachedContentConfig
from .config import PROJECT_ID, LOCATION, CREDENTIALS, ENABLE_CACHE
from ..core.state import RAGState

logger = logging.getLogger(__name__)

# In-memory cache registry
_cache_registry: Dict[str, Dict] = {}


def get_or_create_cache(
    agent_name: str,
    system_instruction: str,
    state: RAGState = None,
    prompt_variation: str = None,
    include_enterprise: bool = False,
    include_references: bool = False,
    include_documents: bool = False
) -> Optional[str]:
    """
    Get existing cache or create new one with timestamp-based invalidation.

    Returns Gemini cache resource name (e.g., projects/.../cachedContents/xyz) or None.
    """
    # Check global flag
    if not ENABLE_CACHE:
        return None

    cache_key = _build_cache_key(
        agent_name,
        prompt_variation,
        state,
        include_enterprise,
        include_references,
        include_documents
    )

    if cache_key in _cache_registry:
        if _is_cache_valid(cache_key, state, include_enterprise, include_references, include_documents):
            resource_name = _cache_registry[cache_key]["resource_name"]
            logger.info(f"[Cache] {agent_name} - Reusing cache: {cache_key[:50]}...")
            logger.info(f"[Cache] {agent_name} - Full resource name: {resource_name}")
            return resource_name
        else:
            logger.info(f"[Cache] {agent_name} - Cache invalidated (timestamp changed)")
            del _cache_registry[cache_key]

    try:
        cache_contents = _build_cache_contents(state, include_enterprise, include_references, include_documents)

        # Use official Gemini token counter for accurate validation
        actual_tokens = _count_cache_tokens(system_instruction, cache_contents)

        # Gemini 2.5 Pro minimum: 1024 tokens (verified via testing)
        if actual_tokens < 1024:
            logger.warning(f"[Cache] {agent_name} - Below minimum ({actual_tokens} tokens < 1024), skipping cache")
            return None

        logger.info(f"[Cache] {agent_name} - Creating cache with {actual_tokens} tokens")

        resource_name = _create_gemini_cache(
            system_instruction=system_instruction,
            contents=cache_contents,
            ttl="3600s",  # 1 hour
            display_name=f"{agent_name}-{prompt_variation or 'default'}"
        )

        content_timestamps = _get_content_timestamps(state, include_enterprise, include_references, include_documents)

        _cache_registry[cache_key] = {
            "resource_name": resource_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "content_timestamps": content_timestamps
        }

        logger.info(f"[Cache] {agent_name} - Created cache: {cache_key[:50]}... ({actual_tokens} tokens)")
        return resource_name

    except Exception as e:
        logger.error(f"[Cache] {agent_name} - Creation failed: {e}")
        return None


def _build_cache_key(
    agent_name: str,
    prompt_variation: Optional[str],
    state: Optional[RAGState],
    include_enterprise: bool,
    include_references: bool,
    include_documents: bool
) -> str:
    """Build unique cache key based on agent, variation, and session."""
    if not state:
        return f"{agent_name}:v1"

    if include_enterprise or include_references or include_documents:
        session_id = state.get("session_id", "default")
        variation_suffix = f":{prompt_variation}" if prompt_variation else ""
        return f"{agent_name}{variation_suffix}:{session_id}"
    elif prompt_variation:
        # Handle prompt variations without dynamic content
        return f"{agent_name}:{prompt_variation}:v1"
    else:
        return f"{agent_name}:v1"


def _is_cache_valid(
    cache_key: str,
    state: Optional[RAGState],
    include_enterprise: bool,
    include_references: bool,
    include_documents: bool
) -> bool:
    """Check if cache is still valid by comparing content timestamps."""
    if not state:
        return True

    metadata = _cache_registry[cache_key]
    cached_timestamps = metadata["content_timestamps"]
    current_timestamps = _get_content_timestamps(state, include_enterprise, include_references, include_documents)

    for content_type, current_ts in current_timestamps.items():
        cached_ts = cached_timestamps.get(content_type, "")
        if current_ts and cached_ts and current_ts > cached_ts:
            logger.info(f"[Cache] Invalidated: {content_type} regenerated ({cached_ts} -> {current_ts})")
            return False

    return True


def _get_content_timestamps(
    state: Optional[RAGState],
    include_enterprise: bool,
    include_references: bool,
    include_documents: bool
) -> Dict[str, str]:
    """Extract timestamps for cached content from state."""
    timestamps = {}

    if not state:
        return timestamps

    if include_enterprise:
        enterprise = state.get("enterprise_agent_output", {}).get("metadata", {})
        enterprise_ts = enterprise.get("timestamp", "")
        if enterprise_ts:
            timestamps["enterprise"] = enterprise_ts

    if include_references:
        annotations = state.get("image_annotations", {})
        if annotations:
            ref_timestamps = [ann.get("timestamp", "") for ann in annotations.values() if ann.get("timestamp")]
            if ref_timestamps:
                timestamps["references"] = max(ref_timestamps)

    if include_documents:
        docs = state.get("enterprise_resources", {}).get("documents", [])
        if docs:
            doc_timestamps = [doc.get("timestamp", "") for doc in docs if doc.get("timestamp")]
            if doc_timestamps:
                timestamps["documents"] = max(doc_timestamps)

    return timestamps


def _build_cache_contents(
    state: Optional[RAGState],
    include_enterprise: bool,
    include_references: bool,
    include_documents: bool
) -> List[Content]:
    """Build Content objects for cache from state."""
    if not state:
        return []

    parts = []

    if include_enterprise:
        enterprise_output = state.get("enterprise_agent_output", {})
        analysis = enterprise_output.get("analysis", "")
        if analysis:
            parts.append(Part(text=f"### Enterprise Analysis:\n{analysis}"))

    if include_references:
        annotations = state.get("image_annotations", {})
        if annotations:
            ref_text = "### Reference Images:\n"
            for url, annotation in annotations.items():
                description = annotation.get("description", "")
                ref_text += f"\n**Reference**: {description}\n"
            parts.append(Part(text=ref_text))

    if include_documents:
        from ..agents.base import load_enterprise_documents
        try:
            document_parts = load_enterprise_documents(state, agent_name="cache_registry")
            parts.extend(document_parts)
        except Exception as e:
            logger.warning(f"[Cache] Failed to load documents: {e}")

    if not parts:
        return []

    return [Content(role="user", parts=parts)]


def _count_cache_tokens(system_instruction: str, contents: List[Content]) -> int:
    """
    Count tokens using official Gemini API for accurate validation.
    Returns total token count for system instruction + contents.
    """
    try:
        client = genai.Client(
            vertexai=True,
            project=PROJECT_ID,
            location=LOCATION,
            credentials=CREDENTIALS
        )

        # Count tokens for contents (if any)
        token_count = 0
        if contents:
            result = client.models.count_tokens(
                model="gemini-3-pro-preview",
                contents=contents
            )
            token_count = result.total_tokens

        # Add system instruction tokens (approximate since API doesn't count system instruction separately)
        # Use character-based estimation for system instruction only
        token_count += len(system_instruction) // 4

        return token_count

    except Exception as e:
        logger.warning(f"[Cache] Token counting failed: {e}, using fallback estimation")
        # Fallback to character-based estimation
        total = len(system_instruction) // 4
        for content in contents:
            for part in content.parts:
                if hasattr(part, 'text'):
                    total += len(part.text) // 4
        return total


def _create_gemini_cache(
    system_instruction: str,
    contents: List[Content],
    ttl: str,
    display_name: str
) -> str:
    """Create Gemini cache via API and return resource name."""
    client = genai.Client(
        vertexai=True,
        project=PROJECT_ID,
        location=LOCATION,
        credentials=CREDENTIALS
    )

    # Gemini API requires non-empty contents array
    # Add minimal context header for prompt-only caching
    if not contents:
        contents = [Content(
            role="user",
            parts=[Part(text="### Current Turn Task and Context")]
        )]

    cache = client.caches.create(
        model="gemini-3-pro-preview",
        config=CreateCachedContentConfig(
            contents=contents,
            system_instruction=system_instruction,
            display_name=display_name,
            ttl=ttl
        )
    )

    logger.info(f"[Cache] Created - Full resource name: {cache.name}")
    logger.info(f"[Cache] Created - Token count: {cache.usage_metadata.total_token_count}")

    return cache.name
