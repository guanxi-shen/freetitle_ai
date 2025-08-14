"""
Character design agents for AI Video Studio - Stage 2 Production
"""

import json
import time
import os
import glob
import shutil
import uuid
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
from ...core.state import RAGState
from ...core.llm import get_llm
from ...core.config import ENABLE_MULTIMODAL_VALIDATION
from ..base import get_agent_context, emit_event, clean_json_response

logger = logging.getLogger(__name__)
from .tools_image import call_image_tool
try:
    from ...storage.gcs_utils import upload_bytes_to_gcs
except ImportError:
    upload_bytes_to_gcs = None  # Graceful fallback


def get_character_llm_choice(state: RAGState) -> str:
    """Get LLM model for character agent. Always returns gemini."""
    return "gemini"


def generate_character_visualization(
    character_name: str,
    image_type: str,
    visual_description: Dict[str, str],
    session_id: str = None,
    character_json: str = "",
    reference_images: List[str] = None,
    image_tool: str = "nano_banana"
) -> Dict[str, Any]:
    """Generate character visualization for production reference and consistency.

    Args:
        character_name: Character's name for filename
        image_type: Type - 'turnaround' for standard reference, 'variation' for creative poses
        visual_description: Dict with 'description' and 'style' keys
        character_json: Full character design JSON as string
        reference_images: List of reference images (max 14)
        image_tool: Tool selection

    Returns:
        Dictionary with image_path and success status
    """
    
    # Validate that variations must have reference_image_paths (turnaround)
    if "variation" in image_type and not reference_images:
        error_msg = f"Variations require reference_images containing the character's turnaround. Missing for {character_name}"
        print(f"[Character Agent - Viz] ERROR: {error_msg}")
        return {"success": False, "error": error_msg}
    
    # Build prompt based on type
    if image_type == "turnaround":
        # Standardized turnaround sheet - fixed format for production consistency
        prompt = f"""Create a 2x2 turnaround reference sheet:
- Top left: Front view
- Top right: 3/4 angle view
- Bottom left: Side profile
- Bottom right: Back view
- No text overlays or captions
- Make subject clearly distinguishable from background, make background color neutral and differ from major subject colors

{json.dumps(visual_description)}"""

    elif "variation" in image_type:
        # Variation - just use the JSON directly
        prompt = json.dumps(visual_description)

    else:
        return {"success": False, "error": f"Unknown image type: {image_type}"}
    
    # Prepare filename
    safe_name = character_name.replace(' ', '_')
    
    # Simple filename without versioning (GCS handles that)
    if image_type == "turnaround":
        filename = f"{safe_name}_{image_type}.png"
    else:
        # For variations, add timestamp to ensure uniqueness
        import uuid
        var_id = str(uuid.uuid4())[:8]
        filename = f"{safe_name}_{image_type}_{var_id}.png"
    
    save_path = None  # No local saving
    
    print(f"[Character Agent - Viz] Generating {image_type} for {character_name}")
    print(f"[Character Agent - Viz] Prompt length: {len(prompt)} characters")
    print(f"[Character Agent - Viz] Save path: {save_path}")
    
    # Prepare reference images - for nano_banana only
    all_references = []

    if reference_images:
        for path in reference_images:
            if path.startswith(("gs://", "https://storage.googleapis.com")) or os.path.exists(path):
                all_references.append(path)

        if all_references:
            print(f"  Using {len(all_references)} reference images")
            for ref_path in all_references:
                ref_name = ref_path.split('/')[-1] if '/' in ref_path else os.path.basename(ref_path)
                print(f"    - {ref_name}")

    # Prepare metadata for nano_banana
    reference_metadata = {
        "reference_count": len(reference_images) if reference_images else 0
    }
    
    # Determine aspect ratio based on image type
    if image_type == "turnaround":
        aspect_ratio = "square"  # Force square for 2x2 grid layout
    else:
        aspect_ratio = "vertical"  # Default for variations
    
    # Always use GCS
    use_gcs = session_id and upload_bytes_to_gcs

    # Use selected image tool
    from .tools_image import call_image_tool

    result = call_image_tool(
        tool_name=image_tool,
        prompt=prompt,
        reference_images=all_references if all_references else None,
        save_path=None,
        reference_metadata=reference_metadata,
        aspect_ratio=aspect_ratio,
        session_id=session_id
    )
    # Store parameters in result metadata for state persistence
    if result.get("success") and "metadata" not in result:
        result["metadata"] = {
            "prompt": prompt,
            "reference_images": all_references,
            "aspect_ratio": aspect_ratio,
            "image_tool": image_tool,
            "filename": filename,
            "path": None  # Will be set after GCS upload
        }
    
    gs_path = ""
    if result["success"]:
        # Upload to GCS if configured
        if use_gcs and "image_bytes" in result:
            # Upload and get gs:// path (not signed URL)
            gs_path = upload_bytes_to_gcs(
                data=result["image_bytes"],
                session_id=session_id,
                asset_type="characters",
                filename=filename,
                content_type="image/png",
                return_format="gs"  # Explicitly request gs:// format
            )
            if gs_path:
                print(f"  [SUCCESS] Image uploaded to GCS: {filename}")
                print(f"  [DEBUG] Using gs:// path (50 chars) instead of signed URL (500+ chars): {gs_path}")
                save_path = gs_path  # Use gs:// path
            else:
                print(f"  [WARNING] Error uploading to GCS")
                save_path = None
        else:
            print(f"  [SUCCESS] Image generated: {save_path if save_path else 'in memory'}")

        # No need to organize references - using gs:// paths
    else:
        print(f"  [ERROR] Image generation error: {result.get('error', 'Unknown error')}")
        save_path = None
    
    # Build metadata for this image
    # Update metadata if generate_character_visualization() parameters change
    # Top-level fields: frequently accessed data (character_name, image_type, path, filename)
    # Params dict: complete function arguments for reproducibility
    metadata = {
        "character_name": character_name,
        "image_type": image_type,
        "path": save_path if result["success"] else None,
        "success": result["success"],
        "timestamp": datetime.now().isoformat(),
        "filename": filename,
        "prompt": prompt,
        "params": {
            "character_name": character_name,
            "image_type": image_type,
            "visual_description": visual_description,
            "session_id": session_id,
            "character_json": character_json,
            "reference_images": reference_images,
            "image_tool": image_tool
        }
    }
    
    return_value = {
        "success": result["success"],
        "image_path": save_path if result["success"] else None,
        "error": result.get("error"),
        "metadata": metadata  # Include metadata in return
    }

    # Add GCS path if available (for backward compatibility)
    if gs_path:
        return_value["gcs_url"] = gs_path  # Keep field name for compatibility, but it's a gs:// path now

    # Add multimodal function response
    if result["success"] and save_path:
        from ...core.config import INCLUDE_REFERENCE_IMAGES_IN_MULTIMODAL

        return_value["multimodal_response"] = {
            "generated": [{
                "file_uri": save_path,
                "mime_type": "image/png",
                "description": f"Generated: {character_name} - {image_type}"
            }],
            "references": [
                {
                    "file_uri": ref,
                    "mime_type": "image/png",
                    "description": f"Reference: {ref.split('/')[-1]}"
                }
                for ref in (all_references or [])
            ] if INCLUDE_REFERENCE_IMAGES_IN_MULTIMODAL else []
        }
        logger.info(f"[Multimodal] Built response: 1 generated, {len(all_references or [])} references (include_refs={INCLUDE_REFERENCE_IMAGES_IN_MULTIMODAL})")

    return return_value


# Removed filesystem helper functions - no longer needed with GCS-only approach


def character_agent(state: RAGState, task_instruction: str = None) -> RAGState:
    """Character image generation agent
    
    Generates character visualizations:
    - Turnarounds and variations
    - Integration with reference images
    - Returns task status
    """
    print("[Character Agent] Starting character design generation...")
    start_time = time.time()
    
    # Set current agent for event tracking
    state["current_agent"] = "character_agent"
    
    # Emit agent started event
    state = emit_event(state, "agent_started", {"agent": "character_agent"}, agent_name="character_agent")

    # Emit processing event
    state = emit_event(state, "processing", {"message": "Generating characters..."}, agent_name="character_agent")

    # Extract session ID from state
    session_id = state.get("session_id", "")
    print(f"[Character Agent] Session ID: {session_id}, ref_images: {len(state.get('reference_images', []))}")
    
    # Initialize character_image_metadata if not present
    if "character_image_metadata" not in state:
        state["character_image_metadata"] = {}

    # Get selected tools from user preferences
    selected_tools = state.get("user_preferences", {}).get("tool_selections", {}).get("character_agent", ["nano_banana"])
    print(f"[Character Agent] User selected tools: {selected_tools}")

    # Create wrapper functions with session context
    def generate_visualization_wrapper(
        character_name: str,
        image_type: str,
        visual_description: Dict[str, str],
        character_json: str = "",
        reference_images: List[str] = None,
        image_tool: str = "nano_banana",
        _batch_mode: bool = False
    ) -> dict:
        """Generate character visualization with session context"""

        # Validate LLM's tool choice against user's selected tools
        validated_tool = image_tool
        if image_tool not in selected_tools:
            validated_tool = selected_tools[0] if selected_tools else "nano_banana"
            print(f"[Character Agent] LLM chose {image_tool} but not in selected tools, using {validated_tool}")
        else:
            print(f"[Character Agent] Using LLM-chosen tool: {validated_tool}")

        # Auto-resolve reference filenames to gs:// paths
        if reference_images:
            user_ref_paths = state.get("reference_images", [])
            char_metadata = state.get("character_image_metadata", {})
            char_paths = [m.get("path") for m in char_metadata.values()
                          if m.get("success") and m.get("path")]

            # Also search supplementary content (props, environments, etc.)
            supp_metadata = state.get("supplementary_content_metadata", {})
            supp_paths = [m.get("path") for m in supp_metadata.values()
                          if m.get("success") and m.get("path")]

            all_ref_paths = user_ref_paths + char_paths + supp_paths
            resolved = []

            for ref in reference_images:
                if ref.startswith("gs://"):
                    resolved.append(ref)
                else:
                    found = False
                    for path in all_ref_paths:
                        if path.endswith(ref) or ref in path:
                            resolved.append(path)
                            print(f"[Character Agent] Resolved '{ref}' → gs:// path")
                            found = True
                            break
                    if not found:
                        print(f"[Character Agent] ERROR: Reference image '{ref}' not found (have {len(all_ref_paths)} paths)")

            reference_images = resolved

        result = generate_character_visualization(
            character_name=character_name,
            image_type=image_type,
            visual_description=visual_description,
            session_id=session_id,
            character_json=character_json,
            reference_images=reference_images,
            image_tool=validated_tool
        )

        # Store metadata if generation was successful
        # Metadata format: {prompt, path, success, timestamp, filename, params{...all function args}}
        # Skip state updates and events when in batch mode (handled by batch wrapper in main thread)
        if not _batch_mode:
            if result.get("success") and "metadata" in result:
                metadata = result["metadata"]
                filename = metadata.get("filename")
                if filename:
                    state["character_image_metadata"][filename] = metadata
                    print(f"[Character Agent] Stored metadata for {filename}")

                    # Emit per-image event for real-time streaming (matches storyboard pattern)
                    emit_event(state, "character_image_generated", {
                        "character_name": character_name,
                        "image_type": image_type,
                        "path": metadata.get("path"),
                        "filename": filename
                    }, agent_name="character_agent")

        return result

    def batch_generate_visualizations_wrapper(
        visualization_configs: List[dict],
        max_workers: int = 12
    ) -> dict:
        """Generate multiple character visualizations in parallel.

        Internally parallelizes using ThreadPoolExecutor while appearing synchronous
        from the LLM's perspective. Blocks until all images complete.

        Args:
            visualization_configs: List of dicts, each containing:
                - character_name: str (required)
                - image_type: str (required, e.g., "turnaround", "variation_1")
                - visual_description: Dict[str, str] (required, with "description" and "style")
                - character_json: str (optional, default "")
                - reference_images: List[str] (optional, max 14)
                - image_tool: str (optional, default "nano_banana")
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

        if not visualization_configs:
            return {
                "success": False,
                "error": "No visualization configs provided",
                "results": [],
                "completed": 0,
                "failed": 0
            }

        print(f"[Character Agent - Batch] Starting batch generation of {len(visualization_configs)} images")

        results = []
        completed_count = 0
        failed_count = 0

        # Use ContextThreadPoolExecutor for context propagation
        with ContextThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_config = {}
            for idx, config in enumerate(visualization_configs):
                try:
                    # Extract parameters with defaults
                    character_name = config.get("character_name")
                    image_type = config.get("image_type")
                    visual_description = config.get("visual_description")
                    character_json = config.get("character_json", "")
                    reference_images = config.get("reference_images")
                    image_tool = config.get("image_tool", "nano_banana")

                    # Validate required parameters
                    if not character_name or not image_type or not visual_description:
                        error_result = {
                            "success": False,
                            "error": f"Missing required parameters in config {idx}",
                            "config_index": idx
                        }
                        results.append(error_result)
                        failed_count += 1
                        continue

                    # Validate visual_description type
                    if not isinstance(visual_description, dict):
                        error_result = {
                            "success": False,
                            "error": f"visual_description must be dict, got {type(visual_description).__name__} in config {idx}",
                            "config_index": idx
                        }
                        results.append(error_result)
                        failed_count += 1
                        continue

                    # Submit to thread pool with batch mode enabled
                    future = executor.submit(
                        generate_visualization_wrapper,
                        character_name=character_name,
                        image_type=image_type,
                        visual_description=visual_description,
                        character_json=character_json,
                        reference_images=reference_images,
                        image_tool=image_tool,
                        _batch_mode=True
                    )
                    future_to_config[future] = (idx, character_name, image_type)

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
                idx, character_name, image_type = future_to_config[future]
                try:
                    result = future.result()  # Blocks until this specific task completes
                    result["config_index"] = idx
                    results.append(result)

                    if result.get("success"):
                        completed_count += 1
                        print(f"[Character Agent - Batch] ✓ Completed {idx+1}/{len(visualization_configs)}: {character_name} ({image_type})")

                        # Update state and emit events in main thread (thread-safe)
                        if "metadata" in result:
                            metadata = result["metadata"]
                            filename = metadata.get("filename")
                            if filename:
                                # State update in main thread - safe from race conditions
                                if "character_image_metadata" not in state:
                                    state["character_image_metadata"] = {}
                                state["character_image_metadata"][filename] = metadata
                                print(f"[Character Agent - Batch] Stored metadata for {filename}")

                                # Emit event in main thread - works correctly with LangGraph StreamWriter
                                emit_event(state, "character_image_generated", {
                                    "character_name": metadata.get("character_name"),
                                    "image_type": metadata.get("image_type"),
                                    "path": metadata.get("path"),
                                    "filename": filename
                                }, agent_name="character_agent")
                    else:
                        failed_count += 1
                        print(f"[Character Agent - Batch] ✗ Failed {idx+1}/{len(visualization_configs)}: {character_name} ({image_type})")

                except Exception as e:
                    error_result = {
                        "success": False,
                        "error": f"Execution error for {character_name} ({image_type}): {str(e)}",
                        "config_index": idx
                    }
                    results.append(error_result)
                    failed_count += 1
                    print(f"[Character Agent - Batch] ✗ Error {idx+1}/{len(visualization_configs)}: {str(e)}")

        # Sort results by config_index to maintain order
        results.sort(key=lambda x: x.get("config_index", 0))

        print(f"[Character Agent - Batch] Batch complete: {completed_count} succeeded, {failed_count} failed")

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
            "total": len(visualization_configs),
            "multimodal_response": multimodal_response
        }

    def list_images_wrapper() -> dict:
        """List all generated character images from metadata"""
        return {
            "success": True,
            "metadata": state.get("character_image_metadata", {})
        }
    
    # Configure LLM with automatic function calling for image generation
    # Pass the wrapper functions with session included
    # Note: LLM will be initialized after loading prompt template
    
    user_query = state["user_query"]
    instruction = task_instruction or state["agent_instructions"].get("character_agent", "Create comprehensive character designs from the script")
    
    if "component_timings" not in state:
        state["component_timings"] = {}
    
    # Build tool selection context section
    selected_tools_text = ", ".join(selected_tools)
    tool_selection_context = f"""### Selected Image Generation Tools:
{selected_tools_text}

Tool Selection Guidelines:
- Use nano_banana for all character generation (turnarounds, variations, precision work)
- Always use a tool from the selected list"""

    # Build complete context with all dynamic content
    from ..base import build_full_context
    context_content = build_full_context(
        state,
        agent_name="character_agent",
        user_query=user_query,
        instruction=instruction,
        agent_specific_context=tool_selection_context,
        context_summary=True,
        window_memory=True,
        include_generated_content=True,
        include_reference_images=True,
        include_image_annotations=True,
        include_generated_assets=True
    )

    # Get static template (no variables)
    from ...prompts import CHARACTER_AGENT_PROMPT_TEMPLATE
    character_prompt = CHARACTER_AGENT_PROMPT_TEMPLATE["template"]

    # Initialize LLM with prompt template as system_instruction
    character_llm = get_llm(
        model="gemini",
        gemini_configs={
            'max_output_tokens': 6000,
            'temperature': 1.0,
            'top_p': 0.92,
            'tools': [
                generate_visualization_wrapper,
                batch_generate_visualizations_wrapper,
                list_images_wrapper
            ],
            'automatic_function_calling': False,  # Manual mode for multimodal support
            'enable_multimodal_responses': ENABLE_MULTIMODAL_VALIDATION,  # Enable images in function responses
            'maximum_remote_calls': 35
        },
        system_instruction=character_prompt
    )

    # Context and prompt are already available in the function scope if needed for debugging

    try:
        # Generate character designs with automatic function calling
        response = character_llm.invoke(
            context_content,  # Only dynamic context, prompt is in system_instruction
            add_context=False,  # Don't concatenate, prompt already in system_instruction
            state=state,
            stream_callback=lambda event_type, content: emit_event(
                state,
                f"llm_{event_type}",
                {"content": content},
                agent_name="character_agent"
            ) if content else None
        )
        
        # # Save raw response for debugging
        # import pickle
        # raw_response_path = "generation_output/debug_raw_character_response.pkl"
        # os.makedirs("generation_output", exist_ok=True)
        # with open(raw_response_path, "wb") as f:
        #     pickle.dump(response, f)
        # print(f"\n[DEBUG] Raw response saved to: {raw_response_path}")
        # print(f"[DEBUG] Raw response type: {type(response)}")
        # print(f"[DEBUG] Raw response (first 1000 chars): {str(response)[:1000]}")
        
        # # Also save as JSON if possible
        # try:
        #     json_response_path = "generation_output/debug_raw_character_response.json"
        #     with open(json_response_path, "w") as f:
        #         json.dump(response, f, indent=2, default=str)
        #     print(f"[DEBUG] Raw response also saved as JSON to: {json_response_path}")
        # except Exception as json_err:
        #     print(f"[DEBUG] Could not save as JSON: {json_err}")
        
        # Parse LLM response
        if isinstance(response, str):
            parsed_response = json.loads(response)
        else:
            parsed_response = response
        response_text = parsed_response['content'][1]['text']
        thinking = parsed_response['content'][0]['thinking']

        # # Clean and parse the task status JSON
        # response_text = clean_json_response(response_text)
        # task_status = json.loads(response_text)
        #
        # # Validate response type
        # if not isinstance(task_status, dict) or "task_status" not in task_status:
        #     raise ValueError(f"Expected task_status JSON, got {type(task_status).__name__}")
        #
        # status_data = task_status["task_status"]

        # Store function calls for debugging
        function_calls = parsed_response.get('function_calls', [])
        print(f"[Character Agent] Found {len(function_calls)} function calls")

        # Extract character names from metadata for summary
        character_image_metadata = state.get("character_image_metadata", {})
        all_character_names = set()
        for filename, metadata in character_image_metadata.items():
            char_name = metadata.get("character_name")
            if metadata.get("success") and char_name:
                all_character_names.add(char_name)

        print(f"[Character Agent] Generated images for {len(all_character_names)} characters: {sorted(all_character_names)}")

        # Store character output in state (minimal - just for answer_parser compatibility)
        state["character_output"] = {
            "thinking": thinking,
            "timestamp": datetime.now().isoformat(),
            "agent": "character_agent"
        }

        # Add function calls if they exist
        if "function_calls" in parsed_response:
            function_calls = parsed_response["function_calls"]
            cleaned_calls = [fc for fc in function_calls if isinstance(fc, dict)]
            state["character_output"]["function_calls"] = cleaned_calls

        # Store thinking process for tracking
        if "thinking_processes" not in state:
            state["thinking_processes"] = {}
        state["thinking_processes"]["character_agent"] = thinking

        # Moved to after try-except to ensure it always fires
        # # Emit completion event for workflow stage tracking (matches storyboard pattern)
        # state = emit_event(state, "characters_generated", {
        #     "character_names": sorted(list(all_character_names)),
        #     "count": len(all_character_names),
        #     "total_images": len(character_image_metadata),
        #     "character_image_metadata": character_image_metadata
        # }, agent_name="character_agent")

        # Store task status for debugging
        # state["character_task_status"] = status_data
        print(f"[Character Agent] Completed - {len(all_character_names)} characters with {len(character_image_metadata)} total images")
        
    except Exception as e:
        # Handle character generation errors
        error_msg = f"Character generation error: {str(e)}"
        state["character_output"] = {
            "error": error_msg,
            "timestamp": datetime.now().isoformat(),
            "agent": "character_agent"
        }

        if "thinking_processes" not in state:
            state["thinking_processes"] = {}
        state["thinking_processes"]["character_agent"] = f"Character generation encountered an error: {error_msg}"

    # Extract character metadata from state and emit completion event (always, even on error)
    character_image_metadata = state.get("character_image_metadata", {})
    all_character_names = set()
    for filename, metadata in character_image_metadata.items():
        char_name = metadata.get("character_name")
        if metadata.get("success") and char_name:
            all_character_names.add(char_name)

    # Emit completion event for workflow stage tracking
    state = emit_event(state, "characters_generated", {
        "character_names": sorted(list(all_character_names)),
        "count": len(all_character_names),
        "total_images": len(character_image_metadata),
        "character_image_metadata": character_image_metadata
    }, agent_name="character_agent")

    # Record timing
    execution_time = time.time() - start_time
    state["component_timings"]["character_agent"] = execution_time
    print(f"[Character Agent] Completed in {execution_time:.2f}s")

    # Emit agent ended event
    state = emit_event(state, "agent_ended", {"agent": "character_agent"}, agent_name="character_agent")

    return state