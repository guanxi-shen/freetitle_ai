"""
Sub-agent for processing individual storyboard shot groups.
Handles 1-3 shots with intelligent image generation using function calling.
"""

import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from ...core.state import RAGState
logger = logging.getLogger(__name__)


def process_storyboard_shots(
    shots: List[Dict],
    state: RAGState,
    storyboard_instruction: str,
    reference_images: List[str],
    function_results_collector: List[Dict[str, Any]],
    selected_tools: List[str]
) -> Dict[str, Any]:
    """
    Process 1-3 shots with intelligent storyboard generation.

    Args:
        shots: List of shots to process [{"scene": 1, "shot": 1}, ...]
        state: Current RAGState
        storyboard_instruction: High-level guidance from orchestrator
        reference_images: Pre-selected reference images (gs:// paths)
        function_results_collector: List to collect all function call metadata
        selected_tools: User-selected image tools

    Returns:
        Dict with success status and processed shots info
    """
    print(f"\n[Storyboard Shot Processor] ========== SUB-AGENT START ==========")
    print(f"[Storyboard Shot Processor] Processing {len(shots)} shot(s)")
    for shot in shots:
        print(f"[Storyboard Shot Processor]   - Scene {shot['scene']}, Shot {shot['shot']}")
    print(f"[Storyboard Shot Processor] Instruction: {storyboard_instruction[:100]}...")
    print(f"[Storyboard Shot Processor] References provided: {len(reference_images)}")

    # Extract script context for these shots
    print(f"[Storyboard Shot Processor] Extracting script context...")
    script_context = _extract_script_context(state, shots)
    print(f"[Storyboard Shot Processor] Script context extracted ({len(script_context)} chars)")

    # Get session ID
    session_id = state.get("session_id", "")

    # Initialize storyboard_frame_metadata if not present
    if "storyboard_frame_metadata" not in state:
        state["storyboard_frame_metadata"] = {}

    # Storage for sub-agent thinking (accessible to wrappers via closure)
    sub_agent_thinking = {"thinking": ""}

    # Operation tracker for each frame (accessible to wrappers via closure)
    # Format: {(scene, shot, frame): [operations]}
    frame_operations = {}

    # Define function wrappers for sub-agent
    def generate_frame_sync(
        scene: int,
        shot: int,
        frame: int,
        prompt: str,
        reference_images: Optional[List[str]] = None,
        tool: str = "nano_banana",
        intermediate_name: Optional[str] = None,
        aspect_ratio: str = "horizontal"
    ) -> Dict[str, Any]:
        """
        Generate or edit a storyboard frame synchronously.

        Use reference_images (list of up to 14 references).
        """
        from .tools_image import generate_or_edit_frame_sync

        result = generate_or_edit_frame_sync(
            scene=scene,
            shot=shot,
            frame=frame,
            prompt=prompt,
            reference_images=reference_images,
            tool=tool,
            intermediate_name=intermediate_name,
            aspect_ratio=aspect_ratio,
            session_id=session_id
        )

        # Enhance path-related errors with available paths for LLM self-correction
        if not result.get("success"):
            error_msg = result.get("error", "")
            if any(keyword in str(error_msg) for keyword in ["404", "NOT_FOUND", "No such object", "path"]):
                from ...agents.base import get_all_asset_urls
                all_assets = get_all_asset_urls(state)

                available_paths = []
                for asset_type, urls in all_assets.items():
                    if urls:
                        available_paths.append(f"{asset_type.upper()}:")
                        for url in urls:
                            available_paths.append(f"  {url}")

                available_list = "\n".join(available_paths) if available_paths else "(No assets available)"

                result["error"] = (
                    f"{error_msg}\n\n"
                    f"Available correct paths:\n{available_list}\n\n"
                    f"Retry with exact path from list above."
                )

        # Track operation for this frame
        frame_key = (scene, shot, frame)
        if frame_key not in frame_operations:
            frame_operations[frame_key] = []

        operation = {
            "step": len(frame_operations[frame_key]) + 1,
            "tool": tool,
            "prompt": prompt,
            "reference_images": reference_images,
            "intermediate_name": intermediate_name,
            "result_path": result.get("gcs_url", ""),
            "success": result.get("success", False),
            "timestamp": result.get("timestamp", datetime.now().isoformat())
        }
        frame_operations[frame_key].append(operation)

        # Store metadata if successful
        if result.get("success"):
            filename = result["filename"]
            gcs_path = result.get("gcs_url", "")

            # Defensive validation: ensure gs:// format
            if gcs_path and not gcs_path.startswith("gs://"):
                print(f"[Storyboard Shot Processor] WARNING: Expected gs:// path, got: {gcs_path[:100]}")

            # Determine if this is final frame or intermediate
            is_final = intermediate_name is None
            final_filename = f"sc{scene:02d}_sh{shot:02d}_fr{frame}.png"

            # Store or update metadata (always overwrite to ensure regeneration updates prompt)
            state["storyboard_frame_metadata"][final_filename] = {
                "scene": scene,
                "shot": shot,
                "frame": frame,
                "path": gcs_path if is_final else "",
                "success": is_final,
                "timestamp": result.get("timestamp", datetime.now().isoformat()),
                "prompt": prompt,
                "thinking": sub_agent_thinking["thinking"],
                "operations": [],
                "orchestrator_config": {
                    "shots_assigned": shots,
                    "instruction": storyboard_instruction,
                    "reference_images": reference_images
                },
                "params": {
                    "scene_number": scene,
                    "shot_number": shot,
                    "frame_number": frame,
                    "prompt": prompt,
                    "reference_images": reference_images,
                    "tool": tool,
                    "intermediate_name": intermediate_name,
                    "aspect_ratio": aspect_ratio
                }
            }
            # Old conditional (only created metadata if new):
            # if final_filename not in state["storyboard_frame_metadata"]:
            #     state["storyboard_frame_metadata"][final_filename] = {...}

            # Update with operation chain and final path
            state["storyboard_frame_metadata"][final_filename]["operations"] = frame_operations[frame_key]
            if is_final:
                state["storyboard_frame_metadata"][final_filename]["path"] = gcs_path
                state["storyboard_frame_metadata"][final_filename]["success"] = True

            # Emit event
            from ..base import emit_event
            emit_event(state, "storyboard_frame_generated", {
                "scene": scene,
                "shot": shot,
                "frame": frame,
                "path": gcs_path,
                "filename": filename
            }, agent_name="storyboard_shot_processor")

        # Collect metadata
        function_results_collector.append({
            "function": "generate_frame_sync",
            "params": {
                "scene": scene,
                "shot": shot,
                "frame": frame,
                "prompt": prompt,
                "tool": tool,
                "intermediate_name": intermediate_name
            },
            "result": result.copy(),
            "timestamp": datetime.now().isoformat()
        })

        # Add multimodal function response
        if result.get("success") and result.get("gcs_url"):
            from ...core.config import INCLUDE_REFERENCE_IMAGES_IN_MULTIMODAL

            result["multimodal_response"] = {
                "generated": [{
                    "file_uri": result["gcs_url"],
                    "mime_type": "image/png",
                    "description": f"Generated: sc{scene:02d}_sh{shot:02d}_fr{frame}"
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

        return result

    def reuse_existing_frame(
        source_path: str,
        scene: int,
        shot: int,
        frame: int
    ) -> Dict[str, Any]:
        """Reuse an existing frame by copying its reference"""
        filename = f"sc{scene:02d}_sh{shot:02d}_fr{frame}.png"

        # Update metadata
        state["storyboard_frame_metadata"][filename] = {
            "scene": scene,
            "shot": shot,
            "frame": frame,
            "path": source_path,
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "reused_from": source_path,
            "params": {
                "scene_number": scene,
                "shot_number": shot,
                "frame_number": frame,
                "reused": True
            }
        }

        # Emit event
        from ..base import emit_event
        emit_event(state, "storyboard_frame_generated", {
            "scene": scene,
            "shot": shot,
            "frame": frame,
            "path": source_path,
            "filename": filename,
            "reused": True
        }, agent_name="storyboard_shot_processor")

        result = {
            "success": True,
            "gcs_url": source_path,
            "filename": filename,
            "reused_from": source_path
        }

        # Collect metadata
        function_results_collector.append({
            "function": "reuse_existing_frame",
            "params": {"source_path": source_path, "scene": scene, "shot": shot, "frame": frame},
            "result": result.copy(),
            "timestamp": datetime.now().isoformat()
        })

        return result

    def list_available_frames() -> Dict[str, Any]:
        """List all available storyboard frames"""
        frames = []
        metadata = state.get("storyboard_frame_metadata", {})

        for filename, meta in metadata.items():
            if meta.get("success") and meta.get("path"):
                frames.append({
                    "filename": filename,
                    "scene": meta.get("scene"),
                    "shot": meta.get("shot"),
                    "frame": meta.get("frame"),
                    "path": meta.get("path")
                })

        return {
            "success": True,
            "frames": frames,
            "count": len(frames)
        }

    # Configure sub-agent LLM with visual context
    from ...core.llm import get_llm
    from ...core.config import ENABLE_MULTIMODAL_VALIDATION

    # Note: LLM will be initialized after loading prompt template

    # Build prompt using template
    from ...prompts.storyboard import STORYBOARD_SUB_AGENT_PROMPT

    # Build context block with all dynamic content
    selected_tools_text = f"User selected tools: {', '.join(selected_tools)}\nChoose the appropriate tool for each frame from this list."

    reference_images_list = "\n".join([
        f"{idx}. {ref_path.split('/')[-1]}\n   Path: {ref_path}"
        for idx, ref_path in enumerate(reference_images, 1)
        if ref_path is not None
    ]) if reference_images else "None"

    script_data = state.get("generated_scripts", {})
    aspect_ratio = script_data.get("script_details", {}).get("aspect_ratio", "horizontal")

    # Build context block
    context_content = f"""### Script Context:
{script_context}

### Storyboard Workflow Notes:
{storyboard_instruction}

### Selected Image Generation Tools:
{selected_tools_text}

### Reference Images ({len(reference_images)} available):
{reference_images_list}

### Target Aspect Ratio:
{aspect_ratio}

### Important:
Review the task instructions and guidelines provided at the beginning of this prompt before proceeding. Ensure your response follows all specified rules, formats, and requirements."""

    # Get static template (no variables)
    prompt = STORYBOARD_SUB_AGENT_PROMPT

    # Initialize LLM with prompt template as system_instruction
    sub_agent_llm = get_llm(
        model="gemini",
        gemini_configs={
            'max_output_tokens': 5000,
            'temperature': 1.0,
            'top_p': 0.95,
            'tools': [generate_frame_sync, reuse_existing_frame, list_available_frames],
            'automatic_function_calling': False,  # Manual mode for multimodal support
            'enable_multimodal_responses': ENABLE_MULTIMODAL_VALIDATION,  # Enable images in function responses
            'maximum_remote_calls': 20
        },
        enable_images=True,
        system_instruction=prompt
    )

    # Prepare images for visual context
    print(f"[Storyboard Shot Processor] Preparing {len(reference_images)} reference images...")
    from ...storage.gcs_utils import prepare_image_for_llm

    images_list = []
    failed_refs = []
    if reference_images:
        for idx, ref_path in enumerate(reference_images, 1):
            try:
                ref_name = ref_path.split('/')[-1] if '/' in ref_path else ref_path
                print(f"[Storyboard Shot Processor]   Preparing ref {idx}/{len(reference_images)}: {ref_name}")

                # Model-aware prep: Gemini uses gs:// directly, others download+base64
                image_data = prepare_image_for_llm(ref_path, "gemini")
                if image_data:
                    images_list.append(image_data)
                    mode = "file_uri" if 'file_uri' in image_data else "base64"
                    print(f"[Storyboard Shot Processor]   SUCCESS - {mode}")
                else:
                    print(f"[Storyboard Shot Processor]   WARNING - Failed to prepare")
                    failed_refs.append(ref_name)
            except Exception as e:
                print(f"[Storyboard Shot Processor]   ERROR at idx={idx}: {e}")
                failed_refs.append(str(ref_path) if ref_path else "None")

    # Add warning to prompt if any references failed
    if failed_refs:
        warning_msg = f"\n\nWARNING: {len(failed_refs)} reference image(s) failed to load: {', '.join(failed_refs)}"
        prompt += warning_msg
        print(f"[Storyboard Shot Processor] Added failure warning to prompt")

    # Invoke sub-agent with visual context
    print(f"[Storyboard Shot Processor] Invoking sub-agent LLM...")
    print(f"[Storyboard Shot Processor]   Visual context: {len(images_list)}/{len(reference_images)} images loaded")
    print(f"[Storyboard Shot Processor]   Max function calls: 20")
    print(f"[Storyboard Shot Processor]   Mode: ANY (function-call-only)")

    try:
        response = sub_agent_llm.invoke(
            context_content,  # Only dynamic context, prompt is in system_instruction
            add_context=False,  # Don't concatenate, prompt already in system_instruction
            state=state,
            images=images_list if images_list else None
        )

        # Extract thinking from response (like video sub-agent)
        print(f"[Storyboard Shot Processor] LLM execution complete")
        print(f"[Storyboard Shot Processor] Extracting thinking content...")

        try:
            if isinstance(response, str):
                parsed_response = json.loads(response)
            else:
                parsed_response = response

            thinking = parsed_response['content'][0]['thinking']
            sub_agent_thinking["thinking"] = thinking
            print(f"[Storyboard Shot Processor] Thinking extracted ({len(thinking)} chars)")
        except (KeyError, IndexError, TypeError) as e:
            print(f"[Storyboard Shot Processor] WARNING - Could not extract thinking: {e}")
            sub_agent_thinking["thinking"] = "[Thinking extraction failed]"  # Explicit marker

        # Update all processed frames with extracted thinking
        print(f"[Storyboard Shot Processor] Updating all frames with extracted thinking...")
        thinking_update_count = 0
        for filename in state["storyboard_frame_metadata"]:
            meta = state["storyboard_frame_metadata"][filename]
            # Only update frames from this batch (matching assigned shots)
            for shot_info in shots:
                if meta.get("scene") == shot_info["scene"] and meta.get("shot") == shot_info["shot"]:
                    state["storyboard_frame_metadata"][filename]["thinking"] = sub_agent_thinking["thinking"]
                    thinking_update_count += 1
                    break
        print(f"[Storyboard Shot Processor] Updated thinking for {thinking_update_count} frame(s)")

        print(f"[Storyboard Shot Processor] Analyzing function call results...")

        # Track successes and failures from function calls
        successful_frames = []
        failed_frames = []
        total_function_calls = len([r for r in function_results_collector if r.get("function") == "generate_frame_sync"])

        for func_result in function_results_collector:
            if func_result.get("function") == "generate_frame_sync":
                result_data = func_result.get("result", {})
                params = func_result.get("params", {})
                if result_data.get("success"):
                    successful_frames.append({
                        "scene": params["scene"],
                        "shot": params["shot"],
                        "frame": params["frame"],
                        "filename": result_data.get("filename"),
                        "path": result_data.get("gcs_url")
                    })
                    print(f"[Storyboard Shot Processor]   SUCCESS - Scene {params['scene']}, Shot {params['shot']}, Frame {params['frame']}")
                else:
                    failed_frames.append({
                        "scene": params["scene"],
                        "shot": params["shot"],
                        "frame": params["frame"],
                        "error": result_data.get("error", "Unknown error")
                    })
                    print(f"[Storyboard Shot Processor]   FAILED - Scene {params['scene']}, Shot {params['shot']}, Frame {params['frame']}: {result_data.get('error', 'Unknown')}")

        print(f"[Storyboard Shot Processor] Results: {len(successful_frames)}/{total_function_calls} succeeded, {len(failed_frames)}/{total_function_calls} failed")
        print(f"[Storyboard Shot Processor] ========== SUB-AGENT COMPLETE ==========\n")

        return {
            "success": len(failed_frames) == 0,
            "shots_processed": len(shots),
            "successful_frames": successful_frames,
            "failed_frames": failed_frames,
            "total_frames": len(successful_frames) + len(failed_frames)
        }

    except Exception as e:
        print(f"[Storyboard Shot Processor] ERROR - Sub-agent exception: {str(e)}")
        print(f"[Storyboard Shot Processor] ========== SUB-AGENT FAILED ==========\n")
        import traceback
        print(f"[Storyboard Shot Processor] Traceback: {traceback.format_exc()}")
        return {
            "success": False,
            "shots_processed": 0,
            "error": str(e),
            "failed_frames": [{"scene": s["scene"], "shot": s["shot"], "error": str(e)} for s in shots]
        }


def _extract_script_context(state: Dict, shots: List[Dict]) -> str:
    """
    Extract relevant script context for the given shots.

    Args:
        state: Current RAGState
        shots: List of shots [{"scene": 1, "shot": 1}, ...]

    Returns:
        Formatted script context string
    """
    script = state.get("generated_scripts", {})
    script_details = script.get("script_details", {})
    production_notes = script.get("production_notes", {})
    scenes = script_details.get("scenes", [])

    # Get all high-level script_details fields (before 'scenes')
    high_level_fields = {k: v for k, v in script_details.items() if k != "scenes"}

    # Get all production_notes
    context_parts = [
        "=== HIGH-LEVEL SCRIPT INFORMATION ===",
        json.dumps(high_level_fields, indent=2),
        "",
        "=== PRODUCTION NOTES ===",
        json.dumps(production_notes, indent=2),
        "",
        "=== SHOTS TO PROCESS ===",
        "IMPORTANT: You are ONLY responsible for generating storyboards for the specific shots listed below.",
        "The scene may contain additional shots that are NOT your responsibility.",
        "Focus ONLY on the shots assigned to you.",
        ""
    ]

    # Extract context for each assigned shot
    for shot_info in shots:
        scene_num = shot_info["scene"]
        shot_num = shot_info["shot"]

        # Find the scene
        for scene_data in scenes:
            if scene_data.get("scene_number") == scene_num:
                # Get all scene-level fields except 'shots'
                scene_fields = {k: v for k, v in scene_data.items() if k != "shots"}

                # Find the specific shot
                shot_list = scene_data.get("shots", [])
                target_shot = next((s for s in shot_list if s.get("shot_number") == shot_num), None)

                if target_shot:
                    context_parts.append(f"--- SCENE {scene_num}, SHOT {shot_num} ---")
                    context_parts.append("Scene Context:")
                    context_parts.append(json.dumps(scene_fields, indent=2))
                    context_parts.append("")
                    context_parts.append("Shot Details:")
                    context_parts.append(json.dumps(target_shot, indent=2))
                    context_parts.append("")
                break

    return "\n".join(context_parts)
