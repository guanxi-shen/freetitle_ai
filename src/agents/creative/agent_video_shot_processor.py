"""
Intelligent sub-agent for individual video shot processing

Handles single shot end-to-end:
1. Reads storyboard + script + available assets
2. Decides which tool to use based on requirements
3. Selects reference images if needed
4. Generates tool-specific prompt
5. Submits video generation task
6. Returns task_id + metadata
"""

import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from ...prompts.video import VIDEO_SUB_AGENT_PROMPT_TEMPLATE as SUB_AGENT_PROMPT_TEMPLATE
logger = logging.getLogger(__name__)


def process_single_shot(
    scene: int,
    shot: int,
    state: Dict[str, Any],
    available_tools: List[str],
    storyboard_data: Dict[str, Any],
    function_results_collector: List[Dict]
) -> Dict[str, Any]:
    """
    Intelligent processing for single video shot

    Workflow:
    1. Read storyboard frames
    2. Read script context
    3. Check available assets (characters, supplementary)
    4. Decide tool based on context
    5. Generate prompt
    6. Submit task
    7. Return result

    Args:
        scene: Scene number
        shot: Shot number
        state: Full RAGState
        available_tools: List of user-selected tool names
        storyboard_data: Storyboard frames for this shot
        function_results_collector: List to collect function call metadata

    Returns:
        {
            "success": bool,
            "task_id": str,
            "tool_used": str,
            "generation_prompt": dict,
            "parameters": dict,
            "error": Optional[str]
        }
    """
    try:
        from ...core.llm import get_llm
        from .tools_video_registry import (
            get_selected_tools_info,
            build_tools_description_section,
            build_selection_rules_section
        )
        from .tools_video import call_video_tool

        logger.info(f"[Shot Processor] Processing Scene {scene}, Shot {shot}")

        # Get selected tools info from registry
        selected_tools = get_selected_tools_info(state)

        # Build dynamic tool sections
        tools_description = build_tools_description_section(selected_tools)
        selection_rules = build_selection_rules_section(selected_tools)

        # Extract storyboard info
        frames = storyboard_data.get("frames", [])
        start_frame = None
        end_frame = None
        start_frame_filename = None
        end_frame_filename = None

        for frame in frames:
            if frame.get("frame_number") == 1:
                start_frame = frame.get("path")
                start_frame_filename = f"sc{scene:02d}_sh{shot:02d}_fr1.png"
            elif frame.get("frame_number") == 2:
                end_frame = frame.get("path")
                end_frame_filename = f"sc{scene:02d}_sh{shot:02d}_fr2.png"

        # Get storyboard generation metadata
        storyboard_metadata = state.get("storyboard_frame_metadata", {})
        start_frame_prompt = "N/A"
        end_frame_prompt = "N/A"

        if start_frame_filename and start_frame_filename in storyboard_metadata:
            start_frame_prompt = storyboard_metadata[start_frame_filename].get("prompt", "N/A")

        if end_frame_filename and end_frame_filename in storyboard_metadata:
            end_frame_prompt = storyboard_metadata[end_frame_filename].get("prompt", "N/A")

        # Build storyboard info with generation prompts
        storyboard_info = f"""Frames available (you can see these images attached to this message):

Start frame: {start_frame if start_frame else 'None'}
  Generation prompt: "{start_frame_prompt}"
"""

        if end_frame:
            storyboard_info += f"""
End frame: {end_frame}
  Generation prompt: "{end_frame_prompt}"
"""
        else:
            storyboard_info += f"""
End frame: None
"""

        storyboard_info += f"""
Note: These generation prompts were used in creating the storyboard frames. Use as reference but don't treat as ground truth - rely on the actual frame images you see.

Frame count: {len(frames)}"""

        # Prepare storyboard frames for LLM visual analysis
        from ...storage.gcs_utils import prepare_image_for_llm

        images_list = []

        # Prepare start frame (always required)
        if start_frame:
            try:
                logger.info(f"[Shot Processor] Preparing start frame: {start_frame}")
                image_data = prepare_image_for_llm(start_frame, "gemini")
                if image_data:
                    images_list.append(image_data)
                    mode = "file_uri" if 'file_uri' in image_data else "base64"
                    logger.info(f"[Shot Processor] Start frame ready ({mode})")
            except Exception as e:
                logger.warning(f"[Shot Processor] Error preparing start frame: {str(e)}")

        # Prepare end frame if available (for dual-frame tools)
        if end_frame:
            try:
                logger.info(f"[Shot Processor] Preparing end frame: {end_frame}")
                image_data = prepare_image_for_llm(end_frame, "gemini")
                if image_data:
                    images_list.append(image_data)
                    mode = "file_uri" if 'file_uri' in image_data else "base64"
                    logger.info(f"[Shot Processor] End frame ready ({mode})")
            except Exception as e:
                logger.warning(f"[Shot Processor] Error preparing end frame: {str(e)}")

        # Extract script context for this scene/shot
        script = state.get("generated_scripts", {})
        script_details = script.get("script_details", {})
        scenes = script_details.get("scenes", [])
        script_context = "No script available"

        for scene_data in scenes:
            if scene_data.get("scene_number") == scene:
                # Find the specific shot within this scene
                shots = scene_data.get("shots", [])
                shot_data = next((s for s in shots if s.get("shot_number") == shot), None)

                # Build comprehensive script context
                script_context = f"""=== SCRIPT CONTEXT FOR SCENE {scene}, SHOT {shot} ===

IMPORTANT: You are handling ONLY Scene {scene}, Shot {shot}. The scene may contain other shots that you are NOT generating (they may be handled separately). Focus on THIS specific shot while maintaining alignment with the overall scene context.

--- HIGH-LEVEL SCRIPT DETAILS ---
Title: {script_details.get('title', 'N/A')}
Duration: {script_details.get('duration', 'N/A')}
Video Summary: {script_details.get('video_summary', 'N/A')}
Creative Vision: {script_details.get('creative_vision', 'N/A')}
Aspect Ratio: {script_details.get('aspect_ratio', 'N/A')}

--- SCENE {scene} CONTEXT ---
Scene Summary: {scene_data.get('scene_summary', 'N/A')}
Setting: {scene_data.get('setting', 'N/A')}
Duration: {scene_data.get('duration', 'N/A')}
Characters: {', '.join(scene_data.get('characters', []))}
Consistency Notes: {scene_data.get('consistency_notes', 'N/A')}

--- SHOT {shot} SPECIFIC DETAILS ---
{json.dumps(shot_data, indent=2) if shot_data else 'No shot-level details available'}

--- PRODUCTION NOTES (ENTIRE VIDEO) ---
{json.dumps(script.get('production_notes', {}), indent=2)}

--- AUDIO DESIGN GUIDANCE ---
{json.dumps(script.get('audio_design', {}), indent=2) if script.get('audio_design') else 'No audio design guidance'}

--- CHARACTER PROFILES ---
{json.dumps(script.get('characters', []), indent=2) if script.get('characters') else 'No character profiles'}"""

                break

        # Assets context not needed - video sub-agents already see storyboard frames visually
        # All character/supplementary/reference assets are already incorporated in the storyboard frames
        # Re-enable this section in the future if we add multi-reference video generation methods that need explicit asset selection

        # # Extract available assets for reference selection
        # # Transform character metadata into list format for LLM consumption
        # character_metadata = state.get("character_image_metadata", {})
        # character_visualizations = [
        #     {
        #         "url": metadata["path"],
        #         "character": metadata["character_name"],
        #         "type": metadata.get("image_type", "unknown"),
        #         "filename": filename,
        #         "prompt": metadata.get("prompt", "")
        #     }
        #     for filename, metadata in character_metadata.items()
        #     if metadata.get("success") and metadata.get("path")
        # ]

        # # Get supplementary assets from proper state fields (aligns with base.py:625-693)
        # supp_metadata = state.get("supplementary_content_metadata", {})
        # supp_semantic = state.get("generated_supplementary", {})

        # supplementary_assets = []
        # for item_key, sem_data in supp_semantic.items():
        #     # Find matching technical metadata
        #     tech_data = None
        #     for filename, meta in supp_metadata.items():
        #         if meta.get('content_name') == item_key or filename.replace(".png", "") == item_key:
        #             tech_data = meta
        #             break

        #     # Merge semantic + technical data
        #     asset_entry = {
        #         "name": sem_data.get('title', item_key),
        #         "category": sem_data.get('category', 'uncategorized'),
        #         "description": sem_data.get('description', ''),
        #         "usage_notes": sem_data.get('usage_notes', ''),
        #         "related_to": sem_data.get('related_to', []),
        #         "path": sem_data.get('image_path') or (tech_data.get('path') if tech_data else None)
        #     }
        #     supplementary_assets.append(asset_entry)

        # # Get user references with annotations
        # user_refs_list = state.get("reference_images", [])
        # image_annotations = state.get("image_annotations", {})

        # user_references_with_context = []
        # for url in user_refs_list:
        #     # Extract filename key (annotations are keyed by filename, not full URL)
        #     filename = url.split("/")[-1].split("?")[0] if "/" in url else url
        #     # Match by filename key (aligns with base.py:792-802)
        #     annotation_data = image_annotations.get(filename, {})
        #     user_references_with_context.append({
        #         "url": url,
        #         "filename": filename,
        #         "content": annotation_data.get("description", "No annotation available")
        #     })

        # available_assets = f"""Character Visualizations:
        # {json.dumps(character_visualizations, indent=2) if character_visualizations else 'None'}

        # Supplementary Assets:
        # {json.dumps(supplementary_assets, indent=2) if supplementary_assets else 'None'}

        # User-Uploaded References:
        # {json.dumps(user_references_with_context, indent=2) if user_references_with_context else 'None'}"""

        # Build context block with all dynamic content
        context_content = f"""### Shot Assignment:
Scene {scene}, Shot {shot}

### Available Video Generation Tools:
{tools_description}

### Tool Selection Rules:
{selection_rules}

### Storyboard Information:
{storyboard_info}

### Script Context:
{script_context}

### Important:
Review the task instructions and guidelines provided at the beginning of this prompt before proceeding. Ensure your response follows all specified rules, formats, and requirements."""

        # Get static template (no variables)
        sub_agent_prompt = SUB_AGENT_PROMPT_TEMPLATE

        # Track last attempt for this shot (accessible via closure)
        last_attempt_data = {"count": 0, "params": None, "result": None, "timestamp": None}

        # Define wrapper function for automatic function calling (must be defined before LLM init)
        def submit_video_generation_task(
            tool_name: str,
            generation_prompt: dict,
            start_frame_path: str,
            end_frame_path: Optional[str] = None,
            reference_image_paths: Optional[List[str]] = None,
            aspect_ratio: str = "horizontal",
            duration: int = 8,
            resolution: str = "720p",
            generate_audio: bool = False
        ) -> Dict[str, Any]:
            """
            Submit video generation task - called by LLM via automatic function calling.

            Validates parameters, calls video generation tool, returns result.
            LLM sees return value and can retry on errors.
            State updates and event emissions handled by process_video_shots wrapper.

            Args:
                tool_name: Selected video generation tool
                generation_prompt: 7-field nested dict {camera, motion, style, dialogue, sound, note, negative}
                start_frame_path: gs:// path to storyboard start frame (REQUIRED)
                end_frame_path: gs:// path to end frame (ONLY for dual-frame capable tools)
                reference_image_paths: List of 3 gs:// paths (ONLY for multi-reference tools)
                aspect_ratio: "vertical" or "horizontal"
                duration: Video duration in seconds
                resolution: "720p" or "1080p" (tool-dependent)
                generate_audio: Enable audio generation (tool-dependent)

            Returns:
                Dict with success, task_id, tool_used, generation_prompt, parameters, error
            """
            logger.info(f"[submit_video_generation_task] CALLED - Scene {scene}, Shot {shot}, tool={tool_name}, prompt_type={type(generation_prompt).__name__}, start_frame={start_frame_path[:60]}..., end_frame={end_frame_path[:60] if end_frame_path else None}...")

            from .tools_video import call_video_tool, VIDEO_TOOLS

            # Validate tool exists
            if tool_name not in VIDEO_TOOLS:
                available = ", ".join(VIDEO_TOOLS.keys())
                error_msg = f"Tool '{tool_name}' not available. Available: {available}"
                logger.error(f"[submit_video_generation_task] {error_msg}")
                return {
                    "success": False,
                    "error": error_msg,
                    "error_context": {"requested_tool": tool_name, "available_tools": list(VIDEO_TOOLS.keys())}
                }

            # Validate start_frame_path
            if not start_frame_path:
                error_msg = "start_frame_path required"
                logger.error(f"[submit_video_generation_task] {error_msg}")
                return {
                    "success": False,
                    "error": error_msg,
                    "error_context": {"scene": scene, "shot": shot}
                }

            if not start_frame_path.startswith("gs://"):
                error_msg = f"start_frame_path must be gs:// path, got: {start_frame_path[:100]}"
                logger.error(f"[submit_video_generation_task] {error_msg}")
                return {
                    "success": False,
                    "error": error_msg,
                    "error_context": {"provided_path": start_frame_path}
                }

            # Convert generation_prompt to JSON string for API with fixed field order
            if isinstance(generation_prompt, dict):
                # Ensure consistent field order: summary, camera, motion, style, dialogue, sound, note, negative
                ordered_prompt = {
                    "summary": generation_prompt.get("summary", ""),
                    "camera": generation_prompt.get("camera", ""),
                    "motion": generation_prompt.get("motion", ""),
                    "style": generation_prompt.get("style", ""),
                    "dialogue": generation_prompt.get("dialogue", ""),
                    "sound": generation_prompt.get("sound", ""),
                    "note": generation_prompt.get("note", ""),
                    "negative": generation_prompt.get("negative", "")
                }
                generation_prompt_string = json.dumps(ordered_prompt)
            else:
                generation_prompt_string = str(generation_prompt)

            # Build tool parameters
            tool_kwargs = {
                "generation_prompt": generation_prompt_string,
                "start_frame_path": start_frame_path,
                "scene": scene,
                "shot": shot,
                "aspect_ratio": aspect_ratio,
                "duration": duration
            }

            # Add optional parameters
            mode = "single-frame"
            if end_frame_path:
                tool_kwargs["end_frame_path"] = end_frame_path
                mode = "dual-frame"
            if reference_image_paths:
                tool_kwargs["reference_image_paths"] = reference_image_paths
            if resolution:
                tool_kwargs["resolution"] = resolution
            if generate_audio:
                tool_kwargs["generate_audio"] = generate_audio

            # Log mode and parameters
            logger.info(f"[Shot Processor] Scene {scene} Shot {shot}: {mode} mode, tool={tool_name}")

            # Call video generation tool
            logger.info(f"[submit_video_generation_task] Calling call_video_tool, tool={tool_name}, kwargs_keys={list(tool_kwargs.keys())}")
            result = call_video_tool(tool_name=tool_name, **tool_kwargs)
            logger.info(f"[submit_video_generation_task] Tool returned - success={result.get('success')}, task_id={result.get('task_id')}, error={result.get('error')}")

            # Track this attempt (don't append to collector yet)
            last_attempt_data["count"] += 1
            last_attempt_data["params"] = {
                "scene": scene,
                "shot": shot,
                "tool_name": tool_name
            }
            last_attempt_data["result"] = {
                "success": result.get("success"),
                "task_id": result.get("task_id"),
                "tool_used": tool_name,
                "generation_prompt": generation_prompt,
                "parameters": tool_kwargs,
                "error": result.get("error"),
                "error_context": result.get("error_context")
            }
            last_attempt_data["timestamp"] = datetime.now().isoformat()

            # Return result (LLM sees this!)
            return_value = {
                "success": result.get("success"),
                "task_id": result.get("task_id"),
                "tool_used": tool_name,
                "generation_prompt": generation_prompt,
                "parameters": tool_kwargs,
                "error": result.get("error"),
                "error_context": result.get("error_context")
            }

            # Log what LLM will see
            if not return_value["success"]:
                logger.error(f"[submit_video_generation_task] Returning error to LLM: {return_value['error']}")

            return return_value

        # Initialize LLM with prompt template as system_instruction
        sub_agent_llm = get_llm(
            model="gemini",
            gemini_configs={
                'max_output_tokens': 3000,
                'temperature': 1.0,
                'top_p': 0.95,
                'automatic_function_calling': True,
                'maximum_remote_calls': 10,
                'tool_config': {'function_calling_config': {'mode': 'ANY'}}
            },
            enable_images=True,
            system_instruction=sub_agent_prompt
        )
        sub_agent_llm.tools = [submit_video_generation_task]

        # Invoke sub-agent with visual context
        logger.info(f"[Shot Processor] Invoking sub-agent LLM - Scene {scene}, Shot {shot}, images={len(images_list)}, tools={available_tools}")

        response = sub_agent_llm.invoke(
            context_content,  # Only dynamic context, prompt is in system_instruction
            add_context=False,  # Don't concatenate, prompt already in system_instruction
            state=state,
            images=images_list if images_list else None
        )

        logger.info(f"[Shot Processor] Sub-agent returned, response_type={type(response).__name__}")

        # Append last attempt to collector
        if last_attempt_data.get("result"):
            function_results_collector.append({
                "function": "submit_video_generation_task",
                "params": {
                    **last_attempt_data["params"],
                    "attempts": last_attempt_data["count"]
                },
                "result": last_attempt_data["result"],
                "timestamp": last_attempt_data["timestamp"]
            })
            logger.info(f"[Shot Processor] Recorded attempt {last_attempt_data['count']} for Scene {scene} Shot {shot}")

        submit_calls = len([r for r in function_results_collector if r.get('function') == 'submit_video_generation_task'])
        logger.info(f"[Shot Processor] Collector has {submit_calls} submit_video_generation_task calls")

        # Extract thinking from response
        try:
            if isinstance(response, str):
                parsed_response = json.loads(response)
            else:
                parsed_response = response

            thinking = parsed_response['content'][0]['thinking']
        except (KeyError, IndexError, TypeError) as e:
            logger.warning(f"[Shot Processor] Could not extract thinking: {e}")
            thinking = ""

        # Find submit_video_generation_task result from collector
        logger.info(f"[Shot Processor] Extracting result, collector_size={len(function_results_collector)}, functions={[r.get('function') for r in function_results_collector]}")

        for func_result in function_results_collector:
            if (func_result["function"] == "submit_video_generation_task" and
                func_result["params"]["scene"] == scene and
                func_result["params"]["shot"] == shot):
                result_data = func_result["result"]
                logger.info(f"[Shot Processor] Found result for Scene {scene} Shot {shot} - success={result_data.get('success')}, task_id={result_data.get('task_id')}, error={result_data.get('error')}")

                if result_data.get("success"):
                    return {
                        "success": True,
                        "task_id": result_data["task_id"],
                        "tool_used": result_data["tool_used"],
                        "generation_prompt": result_data["generation_prompt"],
                        "parameters": result_data["parameters"],
                        "reasoning": "",
                        "thinking": thinking
                    }
                else:
                    return {
                        "success": False,
                        "error": result_data.get("error", "Tool call failed"),
                        "error_context": result_data.get("error_context", {
                            "scene": scene,
                            "shot": shot,
                            "tool_attempted": result_data.get("tool_used")
                        })
                    }

        # No function call found
        logger.error(f"[Shot Processor] No submit_video_generation_task call found - LLM did not call function")
        return {
            "success": False,
            "error": "Sub-agent did not call submit_video_generation_task function",
            "error_context": {"scene": scene, "shot": shot}
        }

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"[Shot Processor] Processing error for Scene {scene}, Shot {shot}: {str(e)}\n{error_trace}")
        return {
            "success": False,
            "error": str(e),
            "error_context": {
                "scene": scene,
                "shot": shot,
                "exception_type": type(e).__name__,
                "traceback": error_trace
            }
        }
