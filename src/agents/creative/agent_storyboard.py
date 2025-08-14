"""
Storyboard generation orchestrator for AI Video Studio
Uses sub-agents for parallel shot processing
"""

import json
import time
from datetime import datetime
from typing import Dict, Any, List
from langchain_core.runnables.config import ContextThreadPoolExecutor
from concurrent.futures import as_completed
from ...core.state import RAGState
from ...core.llm import get_llm
from ..base import get_agent_context, emit_event


def get_storyboard_llm_choice(state: RAGState) -> str:
    """Get LLM model for storyboard agent. Always returns gemini."""
    return "gemini"


def storyboard_agent(state: RAGState, task_instruction: str = None) -> RAGState:
    """
    Storyboard orchestrator - manages storyboard generation workflow

    Uses orchestrator pattern to:
    1. Analyze user request to determine shots to process
    2. Select reference materials
    3. Spawn sub-agents for shot processing
    4. Build final storyboard structures
    """
    print("[Storyboard Agent] Starting storyboard orchestration...")
    start_time = time.time()

    # Reset validation counter at start of turn
    from .tools_storyboard_validation import reset_validation_count
    reset_validation_count(state)

    # Set current agent
    state["current_agent"] = "storyboard_agent"

    # Emit events
    state = emit_event(state, "agent_started", {"agent": "storyboard_agent"}, agent_name="storyboard_agent")
    state = emit_event(state, "processing", {"message": "Generating storyboards..."}, agent_name="storyboard_agent")

    # Get session ID
    session_id = state.get("session_id", "")

    # Initialize metadata
    if "storyboard_frame_metadata" not in state:
        state["storyboard_frame_metadata"] = {}

    # Get selected tools
    selected_tools = state.get("user_preferences", {}).get("tool_selections", {}).get("storyboard_agent", ["nano_banana"])
    print(f"[Storyboard Agent] User selected tools: {selected_tools}")

    # Get script (for error checking only - context provides full access)
    script_data = state.get("generated_scripts", {})
    if not script_data:
        print("[Storyboard Agent] ERROR - No script found")
        state["storyboard_execution_output"] = {
            "status": "error",
            "error": "No script available for storyboard generation",
            "timestamp": datetime.now().isoformat(),
            "successful_count": 0,
            "failed_count": 0
        }
        return state

    script_details = script_data.get("script_details", {})
    script_title = script_details.get("title", "Unknown")
    scenes = script_details.get("scenes", [])
    print(f"[Storyboard Agent] Script: {script_title} ({len(scenes)} scenes)")

    # Create collector for function results
    function_results_collector: List[Dict[str, Any]] = []

    # Define orchestrator functions
    def get_available_shots() -> Dict[str, Any]:
        """Check which shots have existing storyboards"""
        all_shots = []
        existing_frames = []

        # Get all shots from script
        for scene in scenes:
            scene_num = scene.get("scene_number")
            for shot in scene.get("shots", []):
                shot_num = shot.get("shot_number")
                all_shots.append({"scene": scene_num, "shot": shot_num})

        # Check existing frames
        metadata = state.get("storyboard_frame_metadata", {})
        for filename, meta in metadata.items():
            if meta.get("success") and meta.get("path"):
                existing_frames.append({
                    "scene": meta.get("scene"),
                    "shot": meta.get("shot"),
                    "frame": meta.get("frame"),
                    "filename": filename,
                    "path": meta.get("path")
                })

        result = {
            "total_shots": len(all_shots),
            "all_shots": all_shots,
            "existing_frames": existing_frames,
            "existing_count": len(existing_frames)
        }

        # Collect function call metadata
        function_results_collector.append({
            "function": "get_available_shots",
            "params": {},
            "result": result.copy(),
            "timestamp": datetime.now().isoformat()
        })

        return result

    def process_storyboard_shots_wrapper(shot_configs_json: str) -> Dict[str, Any]:
        """
        Wrapper function for LLM automatic function calling.
        Process shot configs in parallel using sub-agents.

        Args:
            shot_configs_json: JSON string containing array of shot configuration objects.
                Format: '[{"shots": [{"scene": 1, "shot": 1}, ...], "instruction": "...", "reference_images": ["gs://...", ...]}, ...]'

                Each config object must have:
                - shots: Array of {scene: int, shot: int} objects (1-3 shots per config)
                - instruction: String describing the storyboard requirements
                - reference_images: Array of GCS paths (gs://) to reference images

        Returns:
            Dict with:
                - groups_processed: Number of configs successfully processed
                - total_groups: Total configs received
                - errors: List of error messages (or None)
                - successful_frames: List of successfully generated frame metadata
                - failed_frames: List of failed frame identifiers
        """
        try:
            print(f"\n[Storyboard Wrapper] ========== ENTRY ==========")
            print(f"[Storyboard Wrapper] Received JSON string ({len(shot_configs_json)} chars)")
            print(f"[Storyboard Wrapper] JSON preview: {shot_configs_json[:200]}...")

            # Parse JSON string
            from ..base import clean_json_response
            cleaned_json = clean_json_response(shot_configs_json)

            # Fix invalid escape sequences from Gemini SDK
            cleaned_json = cleaned_json.replace("\\'", "'")

            try:
                shot_configs = json.loads(cleaned_json)
            except json.JSONDecodeError as e:
                error_msg = f"JSON parsing error at position {e.pos}: {e.msg}\nReceived JSON:\n{shot_configs_json}"
                print(f"[Storyboard Wrapper] ERROR - {error_msg}")
                return {
                    "groups_processed": 0,
                    "total_groups": 0,
                    "errors": [f"Invalid JSON format: {e.msg} at position {e.pos}. Please ensure the shot_configs_json parameter is a valid JSON array."],
                    "successful_frames": [],
                    "failed_frames": [],
                    "json_error_detail": error_msg
                }

            # Validate it's a list
            if not isinstance(shot_configs, list):
                error_msg = f"Expected JSON array, got {type(shot_configs).__name__}. The shot_configs_json must be a JSON array of config objects."
                print(f"[Storyboard Wrapper] ERROR - {error_msg}")
                return {
                    "groups_processed": 0,
                    "total_groups": 0,
                    "errors": [error_msg],
                    "successful_frames": [],
                    "failed_frames": []
                }

            print(f"[Storyboard Wrapper] Successfully parsed {len(shot_configs)} shot config(s)")
            if shot_configs:
                print(f"[Storyboard Wrapper] First config preview: {str(shot_configs[0])[:200]}...")

            # Validate non-empty
            if not shot_configs:
                print(f"[Storyboard Wrapper] WARNING - Empty shot_configs array")
                return {
                    "groups_processed": 0,
                    "total_groups": 0,
                    "errors": ["Empty shot_configs array provided. Please include at least one shot configuration."],
                    "successful_frames": [],
                    "failed_frames": []
                }

            # Validate structure of each config
            validation_errors = []
            for idx, config in enumerate(shot_configs):
                if not isinstance(config, dict):
                    validation_errors.append(f"Config {idx}: Expected object/dict, got {type(config).__name__}")
                    continue

                if "shots" not in config:
                    validation_errors.append(f"Config {idx}: Missing required field 'shots'")
                elif not isinstance(config["shots"], list):
                    validation_errors.append(f"Config {idx}: Field 'shots' must be an array")
                elif len(config["shots"]) == 0:
                    validation_errors.append(f"Config {idx}: Field 'shots' cannot be empty")

                if "instruction" not in config:
                    validation_errors.append(f"Config {idx}: Missing required field 'instruction'")

                if "reference_images" not in config:
                    validation_errors.append(f"Config {idx}: Missing required field 'reference_images'")
                elif not isinstance(config["reference_images"], list):
                    validation_errors.append(f"Config {idx}: Field 'reference_images' must be an array")

            if validation_errors:
                error_summary = f"Shot config validation error with {len(validation_errors)} error(s):\n" + "\n".join(validation_errors)
                print(f"[Storyboard Wrapper] VALIDATION ERRORS:\n{error_summary}")
                return {
                    "groups_processed": 0,
                    "total_groups": len(shot_configs),
                    "errors": validation_errors,
                    "successful_frames": [],
                    "failed_frames": []
                }

            from .agent_storyboard_shot_processor import process_storyboard_shots

            print(f"\n[Storyboard Agent] ========== PROCESSING SHOT CONFIGS ==========")
            print(f"[Storyboard Agent] Total configs: {len(shot_configs)}")

            # Calculate total expected frames for progress tracking
            # Check script for each shot to see if it has end_frame (dual-frame workflow)
            total_expected_frames = 0
            shot_config_summaries = []
            for idx, config in enumerate(shot_configs, 1):
                shots = config.get("shots", [])
                refs = config.get("reference_images", [])

                # Count expected frames by checking script structure
                frame_count = 0
                for shot_info in shots:
                    scene_num = shot_info.get("scene")
                    shot_num = shot_info.get("shot")

                    # Find shot in script to check for end_frame
                    has_end_frame = False
                    for scene_data in scenes:
                        if scene_data.get("scene_number") == scene_num:
                            for shot_data in scene_data.get("shots", []):
                                if shot_data.get("shot_number") == shot_num:
                                    has_end_frame = "end_frame" in shot_data
                                    break
                            break

                    # Dual-frame shots need 2 frames, single-frame shots need 1
                    frame_count += 2 if has_end_frame else 1

                total_expected_frames += frame_count
                shot_config_summaries.append({
                    "config_index": idx,
                    "shots": [{"scene": s.get("scene"), "shot": s.get("shot")} for s in shots],
                    "expected_frames": frame_count,
                    "reference_count": len(refs)
                })
                print(f"[Storyboard Agent]   Config {idx}: {len(shots)} shot(s), {frame_count} expected frame(s), {len(refs)} ref(s)")

            # Emit batch-started event BEFORE spawning threads
            # This allows frontend to show placeholder cards immediately
            print(f"[Storyboard Agent] Emitting batch-started event for {total_expected_frames} expected frame(s)")
            emit_event(state, "storyboard_batch_started", {
                "shot_configs": shot_config_summaries,
                "total_configs": len(shot_configs),
                "total_expected_frames": total_expected_frames
            }, agent_name="storyboard_agent")

            results = []
            errors = []
            completed_count = 0

            # Process configs in parallel with context propagation
            print(f"[Storyboard Agent] Spawning {len(shot_configs)} parallel sub-agent(s)...")
            with ContextThreadPoolExecutor(max_workers=12) as executor:
                futures = {}
                for idx, config in enumerate(shot_configs):
                    print(f"[Storyboard Agent]   Spawning sub-agent {idx + 1}/{len(shot_configs)}...")
                    future = executor.submit(
                        process_storyboard_shots,
                        shots=config.get("shots", []),
                        state=state,
                        storyboard_instruction=config.get("instruction", ""),
                        reference_images=config.get("reference_images", []),
                        function_results_collector=function_results_collector,
                        selected_tools=selected_tools
                    )
                    futures[future] = idx

                # Wait for completion and emit progress events
                print(f"[Storyboard Agent] Waiting for all sub-agents to complete...")
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        result = future.result(timeout=180)  # 3 minute timeout

                        # Validate result structure
                        if result is None or not isinstance(result, dict):
                            error_msg = f"Config {idx + 1} returned invalid result: {type(result).__name__}"
                            print(f"[Storyboard Agent] ERROR - {error_msg}")
                            errors.append(error_msg)
                            continue

                        results.append(result)
                        success = result.get("success", False)
                        status = "SUCCESS" if success else "PARTIAL/FAILED"

                        # Emit progress event immediately after sub-agent completes
                        # This allows frontend to update frames as each batch finishes
                        completed_frames = result.get("successful_frames", [])
                        failed_frames = result.get("failed_frames", [])
                        completed_count += len(completed_frames)

                        print(f"[Storyboard Agent] Config {idx + 1}/{len(shot_configs)} completed - {status}")
                        print(f"[Storyboard Agent] Emitting progress event: {completed_count}/{total_expected_frames} frames completed")

                        emit_event(state, "storyboard_batch_progress", {
                            "config_index": idx + 1,
                            "config_total": len(shot_configs),
                            "completed_frames": completed_frames,
                            "failed_frames": failed_frames,
                            "total_completed": completed_count,
                            "total_expected": total_expected_frames,
                            "progress_percent": int((completed_count / total_expected_frames) * 100) if total_expected_frames > 0 else 0
                        }, agent_name="storyboard_agent")

                    except Exception as e:
                        error_msg = f"Config {idx + 1} error: {str(e)}"
                        print(f"[Storyboard Agent] ERROR - {error_msg}")
                        errors.append(error_msg)

                        # Emit error progress event
                        emit_event(state, "storyboard_batch_progress", {
                            "config_index": idx + 1,
                            "config_total": len(shot_configs),
                            "error": error_msg,
                            "total_completed": completed_count,
                            "total_expected": total_expected_frames
                        }, agent_name="storyboard_agent")

            print(f"[Storyboard Agent] All sub-agents completed")
            print(f"[Storyboard Agent] ==============================================\n")

            # Aggregate all results
            all_successful = [f for r in results for f in r.get("successful_frames", [])]
            all_failed = [f for r in results for f in r.get("failed_frames", [])]

            # Emit batch-complete event
            print(f"[Storyboard Agent] Emitting batch-complete event: {len(all_successful)} succeeded, {len(all_failed)} failed")
            emit_event(state, "storyboard_batch_complete", {
                "total_successful": len(all_successful),
                "total_failed": len(all_failed),
                "total_processed": len(all_successful) + len(all_failed),
                "successful_frames": all_successful,
                "failed_frames": all_failed,
                "errors": errors if errors else None
            }, agent_name="storyboard_agent")

            result = {
                "groups_processed": len(results),
                "total_groups": len(shot_configs),
                "errors": errors if errors else None,
                "successful_frames": all_successful,
                "failed_frames": all_failed
            }

            # Collect metadata
            function_results_collector.append({
                "function": "process_storyboard_shots_wrapper",
                "params": {"shot_configs_json": shot_configs_json, "parsed_count": len(shot_configs)},
                "result": result.copy(),
                "timestamp": datetime.now().isoformat()
            })

            print(f"[Storyboard Wrapper] Returning result: {result['groups_processed']}/{result['total_groups']} groups processed")
            print(f"[Storyboard Wrapper] ========== EXIT SUCCESS ==========\n")
            return result

        except Exception as e:
            print(f"\n[Storyboard Wrapper] ========== EXIT ERROR ==========")
            print(f"[Storyboard Wrapper] Exception type: {type(e).__name__}")
            print(f"[Storyboard Wrapper] Exception message: {str(e)}")
            import traceback
            print(f"[Storyboard Wrapper] Full traceback:\n{traceback.format_exc()}")
            print(f"[Storyboard Wrapper] ==============================================\n")
            raise  # Re-raise to propagate to orchestrator

    # Configure orchestrator LLM
    from ...prompts.storyboard import STORYBOARD_ORCHESTRATOR_PROMPT
    from ...core.config import ENABLE_STORYBOARD_VALIDATION

    # Check if validation is enabled from user preference (config flag as default)
    validation_enabled = state.get("user_preferences", {}).get("enable_validation", ENABLE_STORYBOARD_VALIDATION)
    print(f"[Storyboard Agent] Validation enabled: {validation_enabled}")

    # Conditional tool registration
    tools = [get_available_shots, process_storyboard_shots_wrapper]
    if validation_enabled:
        from .tools_storyboard_validation import validate_storyboards_wrapper

        # Wrap validate_storyboards_wrapper to pass state
        def validate_storyboards_wrapper_with_state(shots_json: str) -> Dict[str, Any]:
            return validate_storyboards_wrapper(shots_json, state)

        tools.append(validate_storyboards_wrapper_with_state)
        print(f"[Storyboard Agent] Validation enabled - registering validation tool")

    print(f"[Storyboard Agent] Registering tools: {[f.__name__ for f in tools]}")
    # Note: LLM will be initialized after loading prompt template

    # Get context (includes full script via FULL_ACCESS_AGENTS)
    user_query = state["user_query"]
    instruction = task_instruction or state["agent_instructions"].get("storyboard_agent", "Create storyboards from the script")

    # Build agent-specific context (validation instructions if enabled)
    agent_specific = None
    if validation_enabled:
        from ...prompts.storyboard import STORYBOARD_VALIDATION_INSTRUCTIONS
        from .tools_storyboard_validation import MAX_VALIDATIONS_PER_TURN
        validation_instructions = STORYBOARD_VALIDATION_INSTRUCTIONS.format(MAX_VALIDATIONS_PER_TURN=MAX_VALIDATIONS_PER_TURN)
        agent_specific = f"\n\n### Validation Requirements:\n{validation_instructions}"
        print(f"[Storyboard Agent] Validation instructions added to context (limit: {MAX_VALIDATIONS_PER_TURN})")

    # Build complete context with all dynamic content
    from ..base import build_full_context
    context_content = build_full_context(
        state,
        agent_name="storyboard_agent",
        user_query=user_query,
        instruction=instruction,
        agent_specific_context=agent_specific,
        context_summary=True,
        window_memory=True,
        include_generated_content=True,
        include_reference_images=True,
        include_image_annotations=True,
        include_generated_assets=True
    )

    # Get static template (no variables)
    orchestration_prompt = STORYBOARD_ORCHESTRATOR_PROMPT["template"]

    # Initialize LLM with prompt template as system_instruction
    storyboard_llm = get_llm(
        model="gemini",
        gemini_configs={
            'max_output_tokens': 8000,
            'temperature': 1.0,
            'top_p': 0.9,
            'tools': tools,
            'automatic_function_calling': False,  # Manual mode - SDK AFC has thought_signature issues
            'maximum_remote_calls': 20,
        },
        system_instruction=orchestration_prompt
    )

    try:
        # Execute orchestration
        print(f"\n[Storyboard Agent] ========== INVOKING ORCHESTRATOR LLM ==========")
        print(f"[Storyboard Agent] About to call storyboard_llm.invoke()...")

        response = storyboard_llm.invoke(
            context_content,  # Only dynamic context, prompt is in system_instruction
            add_context=False,  # Don't concatenate, prompt already in system_instruction
            state=state,
            stream_callback=lambda event_type, content: emit_event(
                state,
                f"llm_{event_type}",
                {"content": content},
                agent_name="storyboard_agent"
            ) if content else None
        )

        print(f"[Storyboard Agent] LLM invocation completed successfully")
        print(f"[Storyboard Agent] Response type: {type(response)}")
        print(f"[Storyboard Agent] ========== ORCHESTRATOR LLM COMPLETE ==========\n")

        # Parse LLM response and extract thinking
        if isinstance(response, str):
            parsed_response = json.loads(response)
        else:
            parsed_response = response

        thinking = parsed_response.get('content', [{}])[0].get('thinking', '')

        # Store thinking process for tracking
        if "thinking_processes" not in state:
            state["thinking_processes"] = {}
        state["thinking_processes"]["storyboard_agent"] = thinking

        # Collect execution results from function_results_collector
        print(f"[Storyboard Agent] Collecting results from {len(function_results_collector)} function call(s)...")
        all_successful = []
        all_failed = []

        for func_result in function_results_collector:
            func_name = func_result.get("function")
            print(f"[Storyboard Agent]   Processing result from: {func_name}")
            if func_name == "process_storyboard_shots_wrapper":
                result_data = func_result.get("result", {})
                # Each sub-agent returns lists of successful/failed frames
                successful = result_data.get("successful_frames", [])
                failed = result_data.get("failed_frames", [])
                all_successful.extend(successful)
                all_failed.extend(failed)
                print(f"[Storyboard Agent]   Wrapper result: {len(successful)} succeeded, {len(failed)} error")

        print(f"[Storyboard Agent] TOTAL RESULTS: {len(all_successful)} succeeded, {len(all_failed)} failed")

        # Build final storyboard structures
        print(f"[Storyboard Agent] Building final storyboard structures...")
        _build_final_structures(state)

        # Create turn-based execution output (replaces storyboard_monitor_output)
        print(f"[Storyboard Agent] Storing execution output...")
        state["storyboard_execution_output"] = {
            "status": "completed",
            "successful_frames": all_successful,
            "failed_frames": all_failed,
            "total_attempted": len(all_successful) + len(all_failed),
            "successful_count": len(all_successful),
            "failed_count": len(all_failed),
            "timestamp": datetime.now().isoformat()
        }

        # Store agent output
        state["storyboard_output"] = {
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "agent": "storyboard_agent"
        }

        # Record timing
        execution_time = time.time() - start_time
        if "component_timings" not in state:
            state["component_timings"] = {}
        state["component_timings"]["storyboard_agent"] = execution_time

        print(f"\n[Storyboard Agent] ========== ORCHESTRATION COMPLETE ==========")
        print(f"[Storyboard Agent] Execution time: {execution_time:.2f}s")
        print(f"[Storyboard Agent] Final results: {len(all_successful)} succeeded, {len(all_failed)} failed")
        print(f"[Storyboard Agent] ==============================================\n")

    except Exception as e:
        print(f"\n[Storyboard Agent] ========== ORCHESTRATION ERROR ==========")
        print(f"[Storyboard Agent] ERROR: {str(e)}")
        import traceback
        print(f"[Storyboard Agent] Traceback: {traceback.format_exc()}")
        print(f"[Storyboard Agent] ==============================================\n")

        # Set empty storyboards to prevent answer_parser errors
        state["generated_storyboards"] = {
            "storyboards": [],
            "generated_frames": [],
            "workspace_frames": []
        }
        # Create execution output for error case
        state["storyboard_execution_output"] = {
            "status": "error",
            "error": str(e),
            "successful_count": 0,
            "failed_count": 0,
            "timestamp": datetime.now().isoformat()
        }

    # Emit agent ended event
    state = emit_event(state, "agent_ended", {"agent": "storyboard_agent"}, agent_name="storyboard_agent")

    return state


def _build_final_structures(state: RAGState) -> None:
    """Build final storyboard structures from metadata (same as old version)"""
    storyboards = []
    workspace_shots = {}
    workspace_frames = []

    frame_metadata = state.get("storyboard_frame_metadata", {})
    print(f"[Storyboard Agent] Processing {len(frame_metadata)} frame metadata entries...")

    for filename, metadata in frame_metadata.items():
        try:
            if metadata.get("success") and metadata.get("path"):
                scene = metadata.get("scene")
                shot = metadata.get("shot")
                frame_num = metadata.get("frame")

                # Build workspace frames
                workspace_frames.append({
                    "path": metadata["path"],
                    "filename": filename,
                    "scene": scene,
                    "shot": shot,
                    "frame": frame_num
                })

                # Build shots structure
                if scene and shot:
                    key = (scene, shot)
                    if key not in workspace_shots:
                        workspace_shots[key] = {
                            "scene_number": scene,
                            "shot_number": shot,
                            "frames": []
                        }
                    workspace_shots[key]["frames"].append({
                        "frame_number": frame_num,
                        "filename": filename,
                        "path": metadata["path"]
                    })
        except Exception as e:
            print(f"[Storyboard Agent] ERROR processing metadata for {filename}: {e}")
            import traceback
            print(f"[Storyboard Agent] Traceback:\n{traceback.format_exc()}")
            continue  # Skip this frame

    # Defensive validation before unpacking
    print(f"[Storyboard Agent] Validating workspace_shots structure before unpacking...")
    print(f"[Storyboard Agent]   Type: {type(workspace_shots)}, Keys count: {len(workspace_shots)}")
    if workspace_shots:
        print(f"[Storyboard Agent]   Sample keys: {list(workspace_shots.keys())[:3]}")
        for key in workspace_shots.keys():
            if not isinstance(key, tuple) or len(key) != 2:
                print(f"[Storyboard Agent] ERROR - Invalid key structure: {key} (type: {type(key)}, len: {len(key) if isinstance(key, (tuple, list)) else 'N/A'})")
                raise ValueError(f"workspace_shots has invalid key format: {key}")

    # Convert to storyboards array
    for (scene, shot), data in sorted(workspace_shots.items()):
        storyboards.append({
            "scene_number": data["scene_number"],
            "shot_number": data["shot_number"],
            "description": f"Scene {scene}, Shot {shot}",
            "frames": sorted(data["frames"], key=lambda x: x["frame_number"]),
            "generated": True
        })

    # Store in state (exact same structure as before)
    state["generated_storyboards"] = {
        "storyboards": storyboards,
        "generated_frames": workspace_frames,
        "workspace_frames": workspace_frames
    }

    print(f"[Storyboard Agent] Built final structures: {len(storyboards)} shots, {len(workspace_frames)} frames")
