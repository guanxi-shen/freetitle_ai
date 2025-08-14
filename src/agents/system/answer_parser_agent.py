"""System agents - core video production coordination and workflow management"""

import json
import logging
import time
import uuid
from datetime import datetime
from ...core.state import RAGState
from ...core.llm import get_llm
from ...core.config import ROLLING_WINDOW_SIZE
from ...prompts import (
    ORCHESTRATOR_PROMPT_TEMPLATE,
    ORCHESTRATOR_RETRY_PROMPT_TEMPLATE,
    ANSWER_SYNTHESIS_BASE_PROMPT_TEMPLATE,
    ANSWER_SYNTHESIS_WITH_SUFFICIENCY_TEMPLATE,
    MEMORY_SUMMARY_PROMPT_TEMPLATE,
    MEMORY_UPDATE_SUMMARY_PROMPT_TEMPLATE,
    MEMORY_UPDATE_CONVERSATION_PROMPT_TEMPLATE,
    AGENT_REGISTRY
)

def get_answer_parser_llm_choice(state: RAGState) -> str:
    """Get LLM model for answer parser agent. Always returns gemini."""
    return "gemini"

def answer_parser_agent(state: RAGState) -> RAGState:
    """Answer parser that synthesizes outputs from multiple agents with context awareness and routes back if material is incomplete

    Handles retry system by:
    - Evaluating material sufficiency before final answer generation
    - Setting retry context for orchestrator when material is incomplete
    - Preserving attempt history and rationale for retry decision
    - Using different prompts based on retry eligibility status
    """
    print("[Answer Parser Agent] Starting answer synthesis...")
    start_time = time.time()

    # Set current agent for event tracking
    state["current_agent"] = "answer_parser"

    # Import utilities from parent module
    from .. import base as agents
    from ..base import emit_event, clean_json_response

    # Emit agent started event for frontend tracking
    state = emit_event(state, "agent_started", {"agent": "answer_parser"}, agent_name="answer_parser")

    # Note: LLM will be initialized after template selection based on retry logic

    user_query = state["user_query"]
    context_summary = state.get("context_summary", "")
    window_memory = state.get("window_memory", [])

    # Initialize retry state if not present
    if "retry_count" not in state:
        state["retry_count"] = 0

    # Recalculate max_retries at the start of each conversation turn
    # (when retry_count is 0) or when max_retries doesn't exist
    if "max_retries" not in state or state.get("retry_count", 0) == 0:
        from ...core.config import DEFAULT_MAX_RETRIES
        # Dynamic retry calculation for multi-agent workflows
        selected_agents = state.get("selected_agents", [])
        initial_agent_count = len(selected_agents) if selected_agents else 1
        # Store initial count for reference
        state["initial_agent_count"] = initial_agent_count
        # Base 3 attempts for 1 agent, +1 for each additional agent
        state["max_retries"] = DEFAULT_MAX_RETRIES + max(0, initial_agent_count - 1)
        print(f"[Answer Parser] Multi-agent workflow: {initial_agent_count} agents, max_retries set to {state['max_retries']}")
    if "previous_attempts" not in state:
        state["previous_attempts"] = []

    if "component_timings" not in state:
        state["component_timings"] = {}

    available_info = []
    print(f"[Answer Parser Agent] Retry state: attempt {state.get('retry_count', 0)}/{state.get('max_retries', 0)}")

    # Check for script agent output with error handling
    if state.get("script_output"):
        if state["script_output"].get("error"):
            available_info.append(f"Script Agent EXECUTED with ERROR: {state['script_output']['error']}")
        else:
            # Also check generated_scripts for successful data
            script_data = state.get("generated_scripts", {})
            if script_data:
                script_details = script_data.get("script_details", {})
                script_info = f"Script Agent EXECUTED - Output:\n"
                script_info += f"  Script: {script_details.get('title', 'Untitled')}"
                script_info += f"\n  Duration: {script_details.get('duration', 'Unknown')}"

                # Add scene details for verification
                scenes = script_details.get('scenes', [])
                script_info += f"\n  Total Scenes: {len(scenes)}"
                if scenes:
                    # Count dialogue correctly at SHOT level (not scene level)
                    total_dialogue = sum(
                        len(shot.get('dialogue', []))
                        for scene in scenes
                        for shot in scene.get('shots', [])
                    )
                    script_info += f"\n  Total Dialogue Lines: {total_dialogue}"
                    # List scene summaries
                    scene_summaries = []
                    for i, scene in enumerate(scenes[:3], 1):  # Show first 3 scenes
                        setting = scene.get('setting', 'Unknown')
                        chars = ', '.join(scene.get('characters', []))
                        scene_summaries.append(f"Scene {i}: {setting} ({chars})")
                    if scene_summaries:
                        script_info += f"\n  Scenes: {'; '.join(scene_summaries)}"
                        if len(scenes) > 3:
                            script_info += f" ...and {len(scenes)-3} more"

                # Add production notes if present
                production_notes = script_data.get('production_notes', {})
                if production_notes:
                    script_info += f"\n  Style: {production_notes.get('style_guide', 'Not specified')}"
                    script_info += f"\n  Tone: {production_notes.get('tone', 'Not specified')}"

                available_info.append(script_info)

    # Check for character agent output with error handling
    if state.get("character_output"):
        if state["character_output"].get("error"):
            available_info.append(f"Character Agent EXECUTED with ERROR: {state['character_output']['error']}")
        else:
            # Check character_image_metadata for successful data
            character_metadata = state.get("character_image_metadata", {})
            char_names = set()
            image_count = 0
            # tool_usage = {}  # COMMENTED OUT: params field filtered from context
            for filename, meta in character_metadata.items():
                if meta.get("success"):
                    char_name = meta.get("character_name")
                    if char_name:
                        char_names.add(char_name)
                    image_count += 1
                    # # Track tool usage
                    # tool = meta.get("params", {}).get("image_tool", "unknown")
                    # tool_usage[tool] = tool_usage.get(tool, 0) + 1

            char_info = f"Character Agent EXECUTED - Output:\n"
            char_info += f"  Character count: {len(char_names)}"

            if char_names:
                char_info += f"\n  Names: {', '.join(sorted(char_names))}"
                char_info += f"\n  Total images generated: {image_count}"

                # # Add tool usage info
                # if tool_usage:
                #     tool_summary = ', '.join([f"{count} {tool}" for tool, count in sorted(tool_usage.items())])
                #     char_info += f"\n  Tools used: {tool_summary}"

                # Add image generation details by character
                char_image_details = {}
                for filename, meta in character_metadata.items():
                    if meta.get("success"):
                        char_name = meta.get("character_name")
                        if char_name:
                            if char_name not in char_image_details:
                                char_image_details[char_name] = 0
                            char_image_details[char_name] += 1

                image_details = [f"{name}: {count} images" for name, count in sorted(char_image_details.items())]
                char_info += f"\n  Details: {', '.join(image_details)}"
            else:
                char_info += " (No characters generated)"

            available_info.append(char_info)

    # Check for storyboard agent output with error handling
    if state.get("storyboard_output"):
        if state["storyboard_output"].get("error"):
            available_info.append(f"Storyboard Agent EXECUTED with ERROR: {state['storyboard_output']['error']}")
        else:
            # Also check generated_storyboards for successful data
            storyboard_data = state.get("generated_storyboards", {})
            if storyboard_data:
                storyboards = storyboard_data.get("storyboards", [])
                # Count actual generated frames from the generated_frames array (current run only)
                generated_frames = storyboard_data.get("generated_frames", [])
                current_run_frame_count = len(generated_frames)

                # Count shots generated in current run (from generated_frames)
                current_run_shots = set()
                for frame in generated_frames:
                    scene = frame.get("scene", 0)
                    shot = frame.get("shot", 0)
                    if scene and shot:
                        current_run_shots.add((scene, shot))
                current_run_shot_count = len(current_run_shots)

                # # Track tool usage from metadata - COMMENTED OUT: params field filtered from context
                # storyboard_metadata = state.get("storyboard_frame_metadata", {})
                # tool_usage = {}
                # for filename, meta in storyboard_metadata.items():
                #     if meta.get("success"):
                #         tool = meta.get("params", {}).get("image_tool", "unknown")
                #         tool_usage[tool] = tool_usage.get(tool, 0) + 1

                # Also get total workspace frames if available
                workspace_frames = storyboard_data.get("workspace_frames", [])
                workspace_count = len(workspace_frames)

                # Count total shots in workspace (from storyboards array)
                total_shots = len(storyboards)

                storyboard_info = f"Storyboard Agent EXECUTED - Output:\n"
                storyboard_info += f"  Generated this run: {current_run_shot_count} shots, {current_run_frame_count} frames\n"
                if workspace_count > 0:
                    storyboard_info += f"  Total in workspace: {total_shots} shots, {workspace_count} frames available"

                # # Add tool usage info
                # if tool_usage:
                #     tool_summary = ', '.join([f"{count} {tool}" for tool, count in sorted(tool_usage.items())])
                #     storyboard_info += f"\n  Tools used: {tool_summary}"

                # Check storyboard execution output for failures (current turn only)
                storyboard_exec = state.get("storyboard_execution_output", {})
                if storyboard_exec:
                    failed_count = storyboard_exec.get("failed_count", 0)
                    if failed_count > 0:
                        storyboard_info += f"\n\nFAILED: {failed_count} frame(s) did not generate"
                        # Extract failure details from execution output (turn-based)
                        failed_frames = storyboard_exec.get("failed_frames", [])
                        for failed in failed_frames[:3]:  # Show first 3 failures
                            scene = failed.get("scene", "?")
                            shot = failed.get("shot", "?")
                            error = failed.get("error", "Unknown error")
                            error_msg = error[:80] if len(error) > 80 else error
                            storyboard_info += f"\n- Scene {scene}, Shot {shot}: {error_msg}"

                available_info.append(storyboard_info)

    # Check for supplementary agent output with error handling
    if state.get("supplementary_output"):
        if state["supplementary_output"].get("error"):
            available_info.append(f"Supplementary Agent EXECUTED with ERROR: {state['supplementary_output']['error']}")
        else:
            # Also check generated_supplementary for successful data
            supplementary_data = state.get("generated_supplementary", {})
            if supplementary_data:
                # # Track tool usage from metadata - COMMENTED OUT: params field filtered from context
                # supplementary_metadata = state.get("supplementary_content_metadata", {})
                # tool_usage = {}
                # for key, meta in supplementary_metadata.items():
                #     if meta.get("success"):
                #         tool = meta.get("params", {}).get("image_tool", "unknown")
                #         tool_usage[tool] = tool_usage.get(tool, 0) + 1

                supp_info = f"Supplementary Agent EXECUTED - Output:\n"
                supp_info += f"  Generated {len(supplementary_data)} items:\n"
                for key, item in supplementary_data.items():
                    title = item.get("title", key)
                    category = item.get("category", "general")
                    has_image = bool(item.get("image_path"))
                    supp_info += f"    - {title} ({category}){' with image' if has_image else ''}\n"

                # # Add tool usage info
                # if tool_usage:
                #     tool_summary = ', '.join([f"{count} {tool}" for tool, count in sorted(tool_usage.items())])
                #     supp_info += f"  Tools used: {tool_summary}\n"

                # Check supplementary monitor output for failures (current turn only)
                supplementary_monitor = state.get("supplementary_monitor_output", {})
                if supplementary_monitor and supplementary_monitor.get("status") != "no_tasks":
                    failed_count = supplementary_monitor.get("failed", 0)
                    if failed_count > 0:
                        supp_info += f"\nFAILED: {failed_count} item(s) did not generate\n"
                        # Extract failure details from monitor output (turn-based)
                        failed_tasks = supplementary_monitor.get("failed_tasks", [])
                        for task in failed_tasks[:3]:  # Show first 3 failures
                            item_key = task.get("content_name", task.get("item_key", "unknown"))
                            error = task.get("error", "Unknown error")
                            error_msg = error[:80] if len(error) > 80 else error
                            supp_info += f"- {item_key}: {error_msg}\n"

                available_info.append(supp_info.rstrip())

    # Check for audio agent output with error handling
    if state.get("audio_output"):
        if state["audio_output"].get("error"):
            available_info.append(f"Audio Agent EXECUTED with ERROR: {state['audio_output']['error']}")
        else:
            # Check audio_generation_metadata for successful data
            audio_metadata = state.get("audio_generation_metadata", {})
            track_count = len(audio_metadata) if audio_metadata else 0

            if track_count > 0:
                available_info.append(f"Audio Agent EXECUTED - {track_count} music tracks in metadata")
            else:
                available_info.append("Audio Agent EXECUTED")

    # Check for video agent output
    if state.get("video_output"):
        if state["video_output"].get("error"):
            available_info.append(f"Video Agent EXECUTED with ERROR: {state['video_output']['error']}")
        else:
            # Check for prompts in the correct location (generated_video_prompts)
            video_prompts_data = state.get("generated_video_prompts", {})
            if video_prompts_data:
                prompts_list = video_prompts_data.get("video_prompts", [])
                metadata = video_prompts_data.get("metadata", {})
                total_prompts = len(prompts_list)

                # Also check video_output for additional execution info
                prompts_generated = state["video_output"].get("prompts_generated", 0)
                tasks_submitted = state["video_output"].get("tasks_submitted", 0)
                submission_errors = state["video_output"].get("submission_errors", [])
                failed_shots = state["video_output"].get("failed_shots", [])

                if total_prompts > 0:
                    info_msg = f"Video Agent EXECUTED - Total prompts in state: {total_prompts}"
                    if prompts_generated > 0:
                        info_msg += f" (generated {prompts_generated} this run)"
                    if tasks_submitted > 0:
                        info_msg += f", submitted {tasks_submitted} tasks"
                    available_info.append(info_msg)
                elif prompts_generated > 0:
                    # Fallback if prompts were generated but not yet in state
                    available_info.append(f"Video Agent EXECUTED - Generated {prompts_generated} prompts")
            elif state["video_output"].get("prompts_generated", 0) > 0:
                # Fallback to video_output info if no prompts in state yet
                prompts_generated = state["video_output"].get("prompts_generated", 0)
                available_info.append(f"Video Agent EXECUTED - Generated {prompts_generated} prompts")

    # Check for video task monitor output
    if state.get("video_monitor_output"):
        if state["video_monitor_output"].get("error"):
            available_info.append(f"Video Task Monitor EXECUTED with ERROR: {state['video_monitor_output']['error']}")
        else:
            # Format video monitor output with public URLs
            monitor_output = state["video_monitor_output"]
            completed_tasks = monitor_output.get("completed_tasks", [])

            video_summary = f"Video Task Monitor EXECUTED - {monitor_output.get('completed_count', 0)} videos completed"

            if completed_tasks:
                video_summary += "\n\n✅ **Completed Videos (Click to view):**"
                for task in completed_tasks:
                    scene = task.get('scene', 0)
                    shot = task.get('shot', 0)
                    public_url = task.get('public_url', '')

                    if public_url:
                        video_summary += f"\n- Scene {scene}, Shot {shot}: [View Video]({public_url})"
                    else:
                        video_summary += f"\n- Scene {scene}, Shot {shot}: Video processed (local only)"

            failed_count = monitor_output.get('failed_count', 0)
            failed_tasks = monitor_output.get('failed_tasks', [])
            if failed_count > 0:
                video_summary += f"\n\n❌ {failed_count} videos error:"
                for task in failed_tasks:
                    scene = task.get('scene', 0)
                    shot = task.get('shot', 0)
                    error = task.get('error', 'Unknown error')
                    error_context = task.get('error_context', {})
                    error_source = error_context.get('source', 'unknown')
                    tool = error_context.get('tool', 'unknown')

                    if error_source == 'exception':
                        is_timeout = error_context.get('is_timeout', False)
                        description = error_context.get('description', '')
                        if is_timeout:
                            video_summary += f"\n- Scene {scene}, Shot {shot} ({tool}): ⏱️ Timeout - {description}"
                        else:
                            error_type = error_context.get('exception_type', 'Exception')
                            video_summary += f"\n- Scene {scene}, Shot {shot} ({tool}): Connection error - {error_type}"
                    elif error_source == 'api_query':
                        video_summary += f"\n- Scene {scene}, Shot {shot} ({tool}): API query failed - {error}"
                    elif error_source == 'task_status':
                        video_summary += f"\n- Scene {scene}, Shot {shot} ({tool}): ⚠️ Generation failed - {error}"
                    else:
                        video_summary += f"\n- Scene {scene}, Shot {shot} ({tool}): {error}"

            pending_count = monitor_output.get('pending_count', 0)
            if pending_count > 0:
                video_summary += f"\n\n⏳ {pending_count} videos still pending"

            video_summary += f"\n\nTotal monitoring time: {monitor_output.get('monitoring_duration', 'N/A')}"
            available_info.append(video_summary)

    # Check for video editor output
    if state.get("video_editor_output"):
        if state["video_editor_output"].get("error"):
            error_msg = state['video_editor_output']['error']
            print(f"[Answer Parser] Video editor error detected: {error_msg[:200]}")
            available_info.append(f"Video Editor EXECUTED with ERROR: {error_msg}")
        else:
            # Format video editor output with clickable link
            editor_output = state["video_editor_output"]
            editor_summary = f"Video Editor EXECUTED - Combined {editor_output.get('videos_processed', 0)} videos"

            public_url = editor_output.get("public_url", "")
            if public_url:
                editor_summary += f"\n\n✏️ **Edited Video:** [View Video]({public_url})"
            else:
                editor_summary += f"\n\n✏️ **Edited Video:** Saved locally as {os.path.basename(editor_output.get('final_video', 'edited.mp4'))}"

            transitions_count = len(editor_output.get("transitions_applied", {}))
            if transitions_count > 0:
                editor_summary += f"\n- Applied {transitions_count} transitions"

            if editor_output.get("editing_plan", {}).get("add_audio"):
                editor_summary += "\n- Added audio track"

            available_info.append(editor_summary)


    # Prepare retry context for the prompt
    retry_context = ""
    if state["previous_attempts"]:
        retry_context = f"Attempt {state['retry_count'] + 1} of {state['max_retries'] + 1}. Previous attempts: "
        for i, attempt in enumerate(state["previous_attempts"]):
            retry_context += f"Attempt {i+1}: {attempt.get('agents_used', [])} -> {attempt.get('outcome', 'unknown')}. "
            # Show what the answer parser decided in previous attempts
            if attempt.get('retry_rationale'):
                retry_context += f"(You decided: {attempt['retry_rationale'][:300]}). "
            if attempt.get('additional_assistance_needed'):
                retry_context += f"(You requested: {attempt['additional_assistance_needed'][:300]}). "
    else:
        retry_context = "First attempt"

    # Get selected_agents for prompt context
    selected_agents = state.get("selected_agents", [])

    if available_info and state["retry_count"] < state["max_retries"] - 1:
        # Retry-eligible case
        all_context = "\n\n".join(available_info)
        agent_specific = f"""### Agent Outputs:
{all_context}

### Retry Context:
{retry_context}

### Selected Agents:
{str(selected_agents)}"""

        base_prompt = ANSWER_SYNTHESIS_WITH_SUFFICIENCY_TEMPLATE["template"]
        answer_schema = ANSWER_SYNTHESIS_WITH_SUFFICIENCY_TEMPLATE["schema"]
        enable_streaming = False
    else:
        # Final attempt
        all_context = "\n\n".join(available_info) if available_info else "No relevant information found in knowledge base"
        agent_specific = f"""### Agent Outputs:
{all_context}

### Selected Agents:
{str(selected_agents)}"""

        base_prompt = ANSWER_SYNTHESIS_BASE_PROMPT_TEMPLATE["template"]
        answer_schema = ANSWER_SYNTHESIS_BASE_PROMPT_TEMPLATE["schema"]
        enable_streaming = True

    # Initialize LLM
    answer_llm = get_llm(
        model="gemini",
        gemini_configs={'max_output_tokens': 8000, 'temperature': 1.0, 'top_p': 0.95},
        system_instruction=base_prompt
    )

    # Build complete context with all dynamic content
    context_content = agents.build_full_context(
        state,
        agent_name="answer_parser",
        user_query=user_query,
        agent_specific_context=agent_specific,
        context_summary=True,
        window_memory=True,
        conversation_history=0,
        include_generated_content=True,
        include_reference_images=True,
        include_image_annotations=True,
        include_tool_selections=True,
        include_expert_info=True
    )


    # Context and prompt are already available in the function scope if needed for debugging

    try:
        response = answer_llm.invoke(
            context_content,  # Only dynamic context, prompt is in system_instruction
            add_context=False,  # Don't concatenate, prompt already in system_instruction
            response_schema=answer_schema,
            state=state,
            stream_callback=lambda event_type, content: emit_event(
                state,
                f"llm_{event_type}",
                {"content": content},
                agent_name="answer_parser"
            ) if content and enable_streaming else None
        )
        parsed_response = json.loads(response)
        response_text = parsed_response['content'][1]['text']
        thinking = parsed_response['content'][0]['thinking']

        # Log raw output for debugging
        print(f"[Answer Parser Agent - Raw Output] {response_text[:1000]}...")

        # Handle sufficiency evaluation response
        if state["retry_count"] < state["max_retries"] and available_info:
            try:
                sufficiency_data = json.loads(clean_json_response(response_text))
                if sufficiency_data.get("task_completion") == "incomplete" and sufficiency_data.get("route_back"):
                    print(f"[Answer Parser Agent] Task incomplete, routing back for retry")
                    # Material is incomplete, route back to orchestrator
                    # Increment retry count before routing decision
                    state["retry_count"] = state.get("retry_count", 0) + 1

                    state["task_completion"] = {
                        "status": "incomplete",
                        "evaluation_time": time.time()
                    }
                    state["retry_rationale"] = sufficiency_data.get("retry_rationale", "Material deemed incomplete")
                    state["additional_assistance_needed"] = sufficiency_data.get("additional_assistance_needed", "No specific assistance specified")

                    # Record this attempt (retry_count was already incremented)
                    state["previous_attempts"].append({
                        "attempt_number": state["retry_count"],
                        "agents_used": state.get("selected_agents", []),
                        "outcome": "incomplete_task",
                        # Add answer parser's own decisions for next run to see
                        "retry_rationale": state.get("retry_rationale", ""),
                        "additional_assistance_needed": state.get("additional_assistance_needed", "")
                    })

                    # Set context for orchestrator retry
                    state["orchestrator_retry_context"] = {
                        "is_retry": True,
                        "retry_reason": state["retry_rationale"],
                        "additional_assistance_needed": state["additional_assistance_needed"],
                        "previous_attempts": state["previous_attempts"]
                    }

                    state["thinking_processes"]["answer_parser"] = f"Material incomplete for query. Routing back to orchestrator. {thinking}"

                    # Don't set final_answer when routing back - let the retry cycle continue
                    # The final answer will be set in the subsequent attempt

                    execution_time = time.time() - start_time
                    state["component_timings"]["answer_parser"] = execution_time
                    print(f"[Answer Parser Agent] Completed in {execution_time:.2f}s (routing for retry)")

                    # Clear monitor/execution outputs after consuming for retry decision
                    # Ensures next iteration sees only fresh data
                    state["video_monitor_output"] = None
                    state["storyboard_execution_output"] = None
                    state["supplementary_monitor_output"] = None

                    return state
                elif sufficiency_data.get("task_completion") == "complete":
                    # Material is complete, use provided answer
                    print(f"[Answer Parser Agent] Task complete, generating final answer")
                    final_answer = sufficiency_data.get("final_answer", response_text)
                else:
                    # Fallback to treating as regular answer
                    # Try to extract final_answer from sufficiency_data if available
                    final_answer = sufficiency_data.get("final_answer", response_text)
            except (json.JSONDecodeError, KeyError) as e:
                # If JSON parsing fails, still try to extract if it looks like JSON
                try:
                    if response_text.strip().startswith('{'):
                        fallback_json = json.loads(clean_json_response(response_text))
                        final_answer = fallback_json.get("final_answer", response_text)
                    else:
                        final_answer = response_text
                except (json.JSONDecodeError, ValueError) as e:
                    logging.info(f"JSON parse error, using raw text: {e}")
                    final_answer = response_text
        else:
            # Final attempt or no retries allowed
            # Try to parse as JSON first to extract final_answer field
            try:
                json_data = json.loads(clean_json_response(response_text))
                final_answer = json_data.get("final_answer", response_text)
            except (json.JSONDecodeError, ValueError) as e:
                logging.info(f"JSON parse error, using raw text: {e}")
                final_answer = response_text

    except (json.JSONDecodeError, KeyError, IndexError) as e:
        final_answer = f"Error generating response: {str(e)}"
        thinking = f"Error: {str(e)}"

    # Mark as complete material (either complete or max retries reached)
    # NOTE: This should only run when generating final_answer, not when routing back
    state["task_completion"] = {
        "status": "complete" if state["retry_count"] < state["max_retries"] else "max_retries_reached",
        "evaluation_time": time.time()
    }

    # Clear route-back fields when generating final answer
    if state.get("retry_rationale"):
        state["retry_rationale"] = ""
    if state.get("additional_assistance_needed"):
        state["additional_assistance_needed"] = ""
    print(f"[Answer Parser Agent] Generating final answer")

    state["final_answer"] = final_answer
    state["thinking_processes"]["answer_parser"] = thinking

    # Defensive initialization
    if "metadata" not in state:
        state["metadata"] = {}

    state["metadata"]["answer_generated"] = True

    # Clear monitor/execution outputs after consuming for final answer generation
    # Ensures consistent state lifecycle regardless of retry decision
    state["video_monitor_output"] = None
    state["storyboard_execution_output"] = None
    state["supplementary_monitor_output"] = None

    execution_time = time.time() - start_time
    state["component_timings"]["answer_parser"] = execution_time
    print(f"[Answer Parser Agent] Completed in {execution_time:.2f}s")

    return state
