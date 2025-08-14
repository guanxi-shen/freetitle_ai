"""
Supplementary creative agent for AI Video Studio - handles dynamic user requests
Can generate any type of content not covered by specialized agents
"""

# Export the main function with both names for compatibility
__all__ = ['supplementary_agent']

import json
import time
import os
import traceback
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
from ...core.state import RAGState
from ...core.llm import get_llm
from ...core.config import ENABLE_MULTIMODAL_VALIDATION
from ..base import get_agent_context, emit_event
from .tools_image import call_image_tool
try:
    from ...storage.gcs_utils import upload_bytes_to_gcs
except ImportError:
    upload_bytes_to_gcs = None  # Graceful fallback

logger = logging.getLogger(__name__)


def get_supplementary_llm_choice(state: RAGState) -> str:
    """Get LLM model for supplementary agent. Always returns gemini."""
    return "gemini"


def list_state_assets(state: RAGState) -> Dict[str, List[str]]:
    """List all assets from state.
    
    Args:
        state: Current RAG state
        
    Returns:
        Dictionary organized by asset type with GCS URLs
    """
    # Use utility function from base.py for consistent metadata access
    from ..base import get_all_asset_urls
    return get_all_asset_urls(state)


def generate_supplementary_content(
    content_type: str,
    content_name: str,
    prompt: str,
    session_id: Optional[str] = None,
    reference_images: Optional[List[str]] = None,
    subfolder: Optional[str] = None,
    aspect_ratio: str = "vertical",
    image_tool: str = "nano_banana"
) -> dict:
    """Generate supplementary content with smart organization.

    Args:
        content_type: Type of content (concept_art, mood_board, prop, etc.)
        content_name: Name for the content
        prompt: Generation prompt
        session_id: Session ID for GCS operations
        reference_images: Optional reference image paths
        subfolder: Optional subfolder under supplementary/
        aspect_ratio: Aspect ratio preset (vertical, horizontal, square, cinematic, portrait, landscape)

    Returns:
        Dictionary with success status and file path
    """
    # Create safe filename
    safe_name = content_name.replace(' ', '_').replace('/', '_')
    filename = f"{safe_name}.png"
    
    # Always use GCS, no local saving
    save_path = None
    
    print(f"[Supplementary Agent] Generating {content_type}: {content_name}")
    print(f"[Supplementary Agent] Save path: {save_path}")
    
    # Generate the image using selected tool
    result = call_image_tool(
        tool_name=image_tool,
        prompt=prompt,
        reference_images=reference_images,
        save_path=None,
        aspect_ratio=aspect_ratio,
        session_id=session_id
    )
    
    gcs_url = ""
    if result["success"]:
        # Always upload to GCS
        if session_id and upload_bytes_to_gcs and "image_bytes" in result:
            asset_type = f"supplementary/{subfolder}" if subfolder else "supplementary"
            gcs_url = upload_bytes_to_gcs(
                data=result["image_bytes"],
                session_id=session_id,
                asset_type=asset_type,
                filename=filename,
                content_type="image/png",
                return_format="gs"  # Request gs:// path format
            )
            if gcs_url:
                print(f"  [SUCCESS] Image uploaded to GCS: {filename}")
                save_path = gcs_url  # Use GCS URL as path
            else:
                print(f"  [WARNING] Error uploading to GCS")
                save_path = None
        else:
            print(f"  [SUCCESS] Generated: {save_path if save_path else 'in memory'}")
        
        # No need to organize references - using GCS URLs
    else:
        print(f"  [ERROR] Generation error: {result.get('error', 'Unknown error')}")
    
    return_value = {
        "success": result["success"],
        "path": save_path if result["success"] else None,
        "content_type": content_type,
        "content_name": content_name,
        "error": result.get("error")
    }

    # Add GCS URL if available
    if gcs_url:
        return_value["gcs_url"] = gcs_url

    # Add multimodal function response
    if result["success"] and save_path:
        from ...core.config import INCLUDE_REFERENCE_IMAGES_IN_MULTIMODAL

        return_value["multimodal_response"] = {
            "generated": [{
                "file_uri": save_path,
                "mime_type": "image/png",
                "description": f"Generated: {content_name} ({content_type})"
            }],
            "references": [
                {
                    "file_uri": ref,
                    "mime_type": "image/png",
                    "description": f"Reference: {ref.split('/')[-1]}"
                }
                for ref in (reference_images or [])
            ] if INCLUDE_REFERENCE_IMAGES_IN_MULTIMODAL else []
        }
        logger.info(f"[Multimodal] Built response: 1 generated, {len(reference_images or [])} references (include_refs={INCLUDE_REFERENCE_IMAGES_IN_MULTIMODAL})")

    return return_value


def supplementary_agent(state: RAGState, task_instruction: str = None) -> RAGState:
    """Supplementary agent - handles any creative request not covered by specialized agents
    
    Can generate:
    - Concept art, mood boards, reference images
    - Props, environments, costumes  
    - Style guides, color palettes
    - Any custom visual or text content
    - Multiple related items with smart organization
    """
    print("[Supplementary Agent] Starting supplementary content generation...")
    start_time = time.time()
    
    # Set current agent for event tracking
    state["current_agent"] = "supplementary_agent"

    # Emit agent started event
    state = emit_event(state, "agent_started", {"agent": "supplementary_agent"}, agent_name="supplementary_agent")

    # Emit processing event
    state = emit_event(state, "processing", {"message": "Generating content..."}, agent_name="supplementary_agent")

    # Extract session ID from state
    session_id = state.get("session_id", "")
    print(f"[Supplementary Agent] Session ID: {session_id}")

    # Initialize supplementary_content_metadata if not present
    if "supplementary_content_metadata" not in state:
        state["supplementary_content_metadata"] = {}

    # Get selected tools from user preferences
    selected_tools = state.get("user_preferences", {}).get("tool_selections", {}).get("supplementary_agent", ["nano_banana"])
    print(f"[Supplementary Agent] User selected tools: {selected_tools}")

    # Create wrapper functions with session context
    def list_assets_wrapper() -> dict:
        """List all assets in session from state"""
        assets = list_state_assets(state)
        total = sum(len(urls) for urls in assets.values())
        return {
            "success": True,
            "assets": assets,
            "total_count": total
        }
    
    def generate_content_wrapper(
        content_type: str,
        content_name: str,
        prompt: str,
        reference_images: Optional[List[str]] = None,
        subfolder: Optional[str] = None,
        aspect_ratio: str = "vertical",
        image_tool: str = "nano_banana",
        description: str = "",
        usage_notes: str = "",
        related_to: Optional[List[str]] = None,
        _batch_mode: bool = False
    ) -> dict:
        """Generate supplementary content (synchronous, blocks until complete)

        Args:
            content_type: Type of content (concept_art, mood_board, prop, etc.)
            content_name: Name for the content
            prompt: Generation prompt
            reference_images: Reference image paths (max 14)
            subfolder: Optional subfolder under supplementary/
            aspect_ratio: Aspect ratio preset - vertical (9:16), horizontal (16:9), square (1:1),
                         cinematic (21:9), portrait (3:4), landscape (4:3)
            image_tool: Tool to use for generation (nano_banana, flux)
            description: What this content is and why it exists (semantic metadata)
            usage_notes: How downstream agents should use this (exact replication vs inspiration)
            related_to: List of related scene/character identifiers
            _batch_mode: Internal flag - skip state updates when True (handled by batch wrapper)

        Returns:
            Dict with success status, path, and metadata
        """
        # Validate LLM's tool choice against user's selected tools
        validated_tool = image_tool
        if image_tool not in selected_tools:
            validated_tool = selected_tools[0] if selected_tools else "nano_banana"
            print(f"[Supplementary Agent] LLM chose {image_tool} but not in selected tools, using {validated_tool}")
        else:
            print(f"[Supplementary Agent] Using LLM-chosen tool: {validated_tool}")

        # Validate reference count limit (upgraded: 14 total)
        if reference_images and len(reference_images) > 14:
            return {
                "success": False,
                "error": f"Too many references: {len(reference_images)} provided, maximum is 14."
            }

        # Generate content synchronously (blocks until complete)
        result = generate_supplementary_content(
            content_type=content_type,
            content_name=content_name,
            prompt=prompt,
            session_id=session_id,
            reference_images=reference_images,
            subfolder=subfolder,
            aspect_ratio=aspect_ratio,
            image_tool=validated_tool
        )

        # Build metadata for the generated content
        if result.get("success"):
            safe_name = content_name.replace(' ', '_').replace('/', '_')
            filename = f"{safe_name}.png"

            metadata = {
                "content_type": content_type,
                "content_name": content_name,
                "filename": filename,
                "path": result.get("path"),
                "gcs_url": result.get("gcs_url"),
                "prompt": prompt,
                "reference_images": reference_images,
                "subfolder": subfolder,
                "aspect_ratio": aspect_ratio,
                "tool": validated_tool,
                "description": description,
                "usage_notes": usage_notes,
                "related_to": related_to if related_to else [],
                "timestamp": datetime.now().isoformat(),
                "success": True
            }

            result["metadata"] = metadata

            # Skip state updates and events when in batch mode (handled by batch wrapper in main thread)
            if not _batch_mode:
                # Store metadata
                state["supplementary_content_metadata"][filename] = metadata
                print(f"[Supplementary Agent] Stored metadata for {filename}")

                # Store in generated_supplementary
                key = content_name.replace(" ", "_").replace("/", "_")
                if key:
                    if "generated_supplementary" not in state:
                        state["generated_supplementary"] = {}

                    state["generated_supplementary"][key] = {
                        "title": content_name,
                        "category": content_type,
                        "description": description,
                        "usage_notes": usage_notes,
                        "related_to": related_to if related_to else [],
                        "image_path": result.get("path")
                    }

                # Emit per-item event for real-time streaming
                emit_event(state, "supplementary_image_generated", {
                    "content_type": content_type,
                    "content_name": content_name,
                    "path": result.get("path"),
                    "filename": filename
                }, agent_name="supplementary_agent")

                # Update monitor output for answer_parser (single-item generation)
                state["supplementary_monitor_output"] = {
                    "status": "complete",
                    "failed": 0 if result.get("success") else 1,
                    "failed_tasks": [] if result.get("success") else [{
                        "item_key": content_name.replace(" ", "_"),
                        "content_name": content_name,
                        "error": result.get("error", "Unknown error")
                    }]
                }

        return result

    def batch_generate_content_wrapper(
        content_configs: List[dict],
        max_workers: int = 12
    ) -> dict:
        """Generate multiple supplementary content items in parallel.

        Internally parallelizes using ThreadPoolExecutor while appearing synchronous.
        Blocks until all content generation completes.

        Args:
            content_configs: List of content generation configs, each containing:
                - content_type: Type of content (concept_art, mood_board, prop, etc.)
                - content_name: Name for the content
                - prompt: Generation prompt
                - reference_images: Reference image paths (optional, max 14)
                - subfolder: Optional subfolder under supplementary/ (default: None)
                - aspect_ratio: Aspect ratio preset (default: "vertical")
                - image_tool: Tool to use for generation (default: "nano_banana")
                - description: What this content is and why it exists (optional)
                - usage_notes: How downstream agents should use this (optional)
                - related_to: List of related scene/character identifiers (optional)
            max_workers: Maximum parallel workers (default 12)

        Returns:
            Dict with:
                - success: bool (True if at least one succeeded)
                - results: List[Dict] (individual generation results)
                - completed: int (count of successful generations)
                - failed: int (count of failed generations)
        """
        from langchain_core.runnables.config import ContextThreadPoolExecutor
        from concurrent.futures import as_completed

        if not content_configs:
            return {
                "success": False,
                "error": "No content configs provided",
                "results": [],
                "completed": 0,
                "failed": 0
            }

        print(f"[Supplementary Agent - Batch] Starting batch generation of {len(content_configs)} items")

        results = []
        completed_count = 0
        failed_count = 0

        # Use ContextThreadPoolExecutor for context propagation
        with ContextThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_config = {}
            for idx, config in enumerate(content_configs):
                try:
                    # Extract parameters with defaults
                    content_type = config.get("content_type", "concept_art")
                    content_name = config.get("content_name", f"item_{idx}")
                    prompt = config.get("prompt", "")
                    reference_images = config.get("reference_images")
                    subfolder = config.get("subfolder")
                    aspect_ratio = config.get("aspect_ratio", "vertical")
                    image_tool = config.get("image_tool", "nano_banana")
                    description = config.get("description", "")
                    usage_notes = config.get("usage_notes", "")
                    related_to = config.get("related_to")

                    # Validate required parameters
                    if not content_name or not prompt:
                        error_result = {
                            "success": False,
                            "error": f"Missing required parameters in config {idx}",
                            "config_index": idx
                        }
                        results.append(error_result)
                        failed_count += 1
                        continue

                    # Submit to thread pool with batch mode enabled
                    future = executor.submit(
                        generate_content_wrapper,
                        content_type=content_type,
                        content_name=content_name,
                        prompt=prompt,
                        reference_images=reference_images,
                        subfolder=subfolder,
                        aspect_ratio=aspect_ratio,
                        image_tool=image_tool,
                        description=description,
                        usage_notes=usage_notes,
                        related_to=related_to,
                        _batch_mode=True
                    )
                    future_to_config[future] = (idx, content_type, content_name)

                except Exception as e:
                    error_result = {
                        "success": False,
                        "error": f"Failed to submit config {idx}: {str(e)}",
                        "config_index": idx
                    }
                    results.append(error_result)
                    failed_count += 1

            # Collect results as they complete (blocks here until all done)
            # Process in main thread for thread-safe state updates and event emission
            for future in as_completed(future_to_config):
                idx, content_type, content_name = future_to_config[future]
                try:
                    result = future.result()  # Blocks until this specific task completes
                    result["config_index"] = idx
                    results.append(result)

                    if result.get("success"):
                        completed_count += 1
                        print(f"[Supplementary Agent - Batch] ✓ Completed {idx+1}/{len(content_configs)}: {content_name} ({content_type})")

                        # Update state and emit events in main thread (thread-safe)
                        if "metadata" in result:
                            metadata = result["metadata"]
                            filename = metadata.get("filename")
                            if filename:
                                # State update in main thread - safe from race conditions
                                if "supplementary_content_metadata" not in state:
                                    state["supplementary_content_metadata"] = {}
                                state["supplementary_content_metadata"][filename] = metadata
                                print(f"[Supplementary Agent - Batch] Stored metadata for {filename}")

                                # Store in generated_supplementary
                                key = content_name.replace(" ", "_").replace("/", "_")
                                if key:
                                    if "generated_supplementary" not in state:
                                        state["generated_supplementary"] = {}

                                    state["generated_supplementary"][key] = {
                                        "title": content_name,
                                        "category": content_type,
                                        "description": metadata.get("description", ""),
                                        "usage_notes": metadata.get("usage_notes", ""),
                                        "related_to": metadata.get("related_to", []),
                                        "image_path": result.get("path")
                                    }

                                # Emit event in main thread - works correctly with LangGraph StreamWriter
                                emit_event(state, "supplementary_image_generated", {
                                    "content_type": content_type,
                                    "content_name": content_name,
                                    "path": metadata.get("path"),
                                    "filename": filename
                                }, agent_name="supplementary_agent")
                    else:
                        failed_count += 1
                        print(f"[Supplementary Agent - Batch] ✗ Failed {idx+1}/{len(content_configs)}: {content_name} ({content_type})")

                except Exception as e:
                    error_result = {
                        "success": False,
                        "error": f"Execution error for {content_name} ({content_type}): {str(e)}",
                        "config_index": idx
                    }
                    results.append(error_result)
                    failed_count += 1
                    print(f"[Supplementary Agent - Batch] ✗ Error {idx+1}/{len(content_configs)}: {str(e)}")

        # Sort results by config_index to maintain order
        results.sort(key=lambda x: x.get("config_index", 0))

        print(f"[Supplementary Agent - Batch] Batch complete: {completed_count} succeeded, {failed_count} failed")

        # Build monitor-compatible output for answer_parser (drop-in replacement for async monitor)
        failed_items = []
        for result in results:
            if not result.get("success"):
                failed_items.append({
                    "item_key": result.get("content_name", "").replace(" ", "_"),
                    "content_name": result.get("content_name", "unknown"),
                    "error": result.get("error", "Unknown error")
                })

        state["supplementary_monitor_output"] = {
            "status": "complete",
            "failed": failed_count,
            "failed_tasks": failed_items
        }
        print(f"[Supplementary Agent - Batch] Updated supplementary_monitor_output: {failed_count} failed")

        # Aggregate multimodal_response from all results
        multimodal_response = {"generated": [], "references": []}

        for result in results:
            if result.get("success") and result.get("multimodal_response"):
                # Extend generated list
                multimodal_response["generated"].extend(
                    result["multimodal_response"]["generated"]
                )
                # Deduplicate references by file_uri
                for ref in result["multimodal_response"].get("references", []):
                    if ref not in multimodal_response["references"]:
                        multimodal_response["references"].append(ref)

        logger.info(f"[Multimodal Batch] Aggregated: {len(multimodal_response['generated'])} generated, {len(multimodal_response['references'])} references")

        return {
            "success": completed_count > 0,
            "results": results,
            "completed": completed_count,
            "failed": failed_count,
            "multimodal_response": multimodal_response
        }

    # Configure LLM with automatic function calling
    # Note: LLM will be initialized after loading prompt template

    user_query = state["user_query"]
    instruction = task_instruction or state["agent_instructions"].get("supplementary_agent", "Generate creative content based on the request")
    
    if "component_timings" not in state:
        state["component_timings"] = {}
    
    # Build agent-specific context sections
    agent_specific_parts = []

    # Existing supplementary content
    supplementary_metadata = state.get("generated_supplementary", {})
    if supplementary_metadata:
        agent_specific_parts.append(f"### Existing Supplementary Content:\n{json.dumps(supplementary_metadata, indent=2)}")

    # Tool selection guidelines
    selected_tools_text = ", ".join(selected_tools)
    tool_selection = f"""### Selected Image Generation Tools:
{selected_tools_text}

Tool Selection Guidelines:
- Use nano_banana for all supplementary materials (mood boards, concept art, environments, props, aesthetics)
- Always use a tool from the selected list"""
    agent_specific_parts.append(tool_selection)

    agent_specific = "\n\n".join(agent_specific_parts)

    # Build complete context with all dynamic content
    from ..base import build_full_context
    context_content = build_full_context(
        state,
        agent_name="supplementary_agent",
        user_query=user_query,
        instruction=instruction,
        agent_specific_context=agent_specific,
        context_summary=True,
        window_memory=True,
        include_generated_content=True,
        include_reference_images=True,
        include_image_annotations=True
    )

    # Get static template (no variables)
    from ...prompts import SUPPLEMENTARY_AGENT_PROMPT_TEMPLATE
    supplementary_prompt = SUPPLEMENTARY_AGENT_PROMPT_TEMPLATE["template"]

    # Initialize LLM with prompt template as system_instruction
    supplementary_llm = get_llm(
        model="gemini",
        gemini_configs={
            'max_output_tokens': 8400,
            'temperature': 1.0,
            'top_p': 0.95,
            'tools': [
                list_assets_wrapper,
                generate_content_wrapper,
                batch_generate_content_wrapper
            ],
            'automatic_function_calling': False,  # Manual mode for multimodal support
            'enable_multimodal_responses': ENABLE_MULTIMODAL_VALIDATION,  # Enable images in function responses
            'maximum_remote_calls': 40
        },
        system_instruction=supplementary_prompt
    )

    # Context and prompt are already available in the function scope if needed for debugging

    try:
        # Generate with automatic function calling
        response = supplementary_llm.invoke(
            context_content,  # Only dynamic context, prompt is in system_instruction
            add_context=False,  # Don't concatenate, prompt already in system_instruction
            state=state,
            stream_callback=lambda event_type, content: emit_event(
                state,
                f"llm_{event_type}",
                {"content": content},
                agent_name="supplementary_agent"
            ) if content else None
        )
        
        # Parse LLM response - extract thinking only, don't parse text as JSON
        if isinstance(response, str):
            parsed_response = json.loads(response)
        else:
            parsed_response = response

        thinking = parsed_response['content'][0].get('thinking', '')
        response_text = parsed_response['content'][1].get('text', '')  # For logging only

        # Get function calls for logging (metadata already collected in function wrappers)
        function_calls = parsed_response.get('function_calls', [])
        print(f"[Supplementary Agent] Made {len(function_calls)} function calls")

        # Count generated content (metadata already captured in generate_content_wrapper)
        generated_content = state.get("generated_supplementary", {})
        content_count = len(generated_content)

        print(f"\n[Supplementary Agent] Generated {content_count} supplementary content items")
        print(f"[Supplementary Agent] All content generation complete (blocking execution)")

        # Store supplementary output in state with completion status
        state["supplementary_output"] = {
            "status": "completed",
            "thinking": thinking,
            "timestamp": datetime.now().isoformat(),
            "agent": "supplementary_agent",
            "function_calls": function_calls if function_calls else [],
            "content_count": content_count,
            "generated_content": generated_content
        }

        # Store thinking process for UI display
        if "thinking_processes" not in state:
            state["thinking_processes"] = {}
        state["thinking_processes"]["supplementary_agent"] = thinking

        print(f"[Supplementary Agent] Generation complete - {content_count} items created")

        # Emit event to notify UI that generation is complete
        state = emit_event(state, "supplementary_generation_complete", {
            "content_count": content_count,
            "message": f"Generated {content_count} supplementary items"
        }, agent_name="supplementary_agent")

    except (json.JSONDecodeError, KeyError, IndexError, AttributeError, ValueError) as e:
        # Handle processing errors gracefully
        error_msg = f"Error processing supplementary response: {str(e)}"
        print(f"[Supplementary Agent] Error: {error_msg}")

        # Use existing generated content from state (populated by function wrappers)
        generated_content = state.get("generated_supplementary", {})
        content_count = len(generated_content)

        state["supplementary_output"] = {
            "error": error_msg,
            "status": "completed_with_errors",
            "content_count": content_count,
            "generated_content": generated_content,
            "timestamp": datetime.now().isoformat(),
            "agent": "supplementary_agent"
        }

        if "thinking_processes" not in state:
            state["thinking_processes"] = {}
        state["thinking_processes"]["supplementary_agent"] = f"Error processing response but {content_count} items generated successfully"

    # Emit supplementary_generated event (always, even on error) using state data
    state = emit_event(state, "supplementary_generated", {
        "content": state.get("generated_supplementary", {}),
        "supplementary_content_metadata": state.get("supplementary_content_metadata", {})
    }, agent_name="supplementary_agent")

    # Record timing
    execution_time = time.time() - start_time
    state["component_timings"]["supplementary_agent"] = execution_time
    print(f"[Supplementary Agent] Completed in {execution_time:.2f}s")

    # Emit agent ended event
    state = emit_event(state, "agent_ended", {"agent": "supplementary_agent"}, agent_name="supplementary_agent")

    return state

