"""
Video production orchestrator agent for AI Video Studio
Uses automatic function calling to manage the video generation workflow
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from ...core.state import RAGState
from ...core.llm import get_llm
from ...prompts.video import get_video_agent_prompt
from ..base import get_agent_context, emit_event


def get_video_llm_choice(state: RAGState) -> str:
    """Get LLM model for video agent. Always returns gemini."""
    return "gemini"


def video_agent(state: RAGState, task_instruction: str = None) -> RAGState:
    """Video production orchestrator - manages prompt generation and video creation
    
    This agent acts as a production orchestrator using automatic function calling:
    1. Analyzes context and decides which shots need processing
    2. Calls functions to generate prompts in parallel
    3. Merges results with existing prompts
    4. Decides which shots need video generation
    5. Submits video tasks and confirms submission (doesn't wait for completion)
    """
    print("[Video Agent] Starting video production orchestration...")
    start_time = time.time()
    
    # Set current agent for event tracking
    state["current_agent"] = "video_agent"
    
    # Emit agent started event
    state = emit_event(state, "agent_started", {"agent": "video_agent"}, agent_name="video_agent")

    # Emit processing event
    state = emit_event(state, "processing", {"message": "Generating videos..."}, agent_name="video_agent")

    # Extract session ID from state
    session_id = state.get("session_id", "")
    print(f"[Video Agent] Session ID: {session_id}")
    
    # Create collector for function results (accessible to nested functions via closure)
    function_results_collector = []
    
    # Get existing prompts and storyboards
    existing_prompts = state.get("generated_video_prompts", {})
    existing_prompts_count = len(existing_prompts.get("video_prompts", []))
    
    # Get storyboards from state
    # Extract frames from metadata (utility function)
    from ..base import get_storyboard_frames
    workspace_frames = get_storyboard_frames(state)

    if not workspace_frames:
        print("[Video Agent] No storyboard frames found in state")
        state["video_output"] = {
            "error": "No storyboard frames available",
            "timestamp": datetime.now().isoformat(),
            "agent": "video_agent"
        }
        return state
    
    # Build storyboards structure
    workspace_shots = {}
    for frame in workspace_frames:
        scene_num = frame.get("scene", 0)
        shot_num = frame.get("shot", 0)
        frame_num = frame.get("frame", 0)
        
        if scene_num and shot_num:
            key = (scene_num, shot_num)
            if key not in workspace_shots:
                workspace_shots[key] = {
                    "scene_number": scene_num,
                    "shot_number": shot_num,
                    "frames": []
                }
            
            workspace_shots[key]["frames"].append({
                "frame_number": frame_num,
                "path": frame["url"],  # Use URL instead of path
                "filename": frame.get("filename", "")
            })
    
    storyboards = sorted(workspace_shots.values(), key=lambda x: (x["scene_number"], x["shot_number"]))
    storyboards_count = len(storyboards)
    
    print(f"[Video Agent] Found {storyboards_count} shots, {existing_prompts_count} existing prompts")

    # Get selected tools from user preferences
    # Map frontend tool IDs to backend tool names
    TOOL_ID_MAPPING = {
        "google_veo_i2v": "google_veo_i2v",
    }

    selected_tools_raw = state.get("user_preferences", {}).get("tool_selections", {}).get("video_agent", ["google_veo_i2v"])
    selected_tools = [TOOL_ID_MAPPING.get(tool, tool) for tool in selected_tools_raw]
    print(f"[Video Agent] User selected tools (raw): {selected_tools_raw}")
    print(f"[Video Agent] Mapped to backend tools: {selected_tools}")

    # Import function tools
    from .tools_video import (
        create_function_wrappers
    )
    
    # Create function wrappers with state binding and results collector
    get_available_shots, process_video_shots = create_function_wrappers(
        state, storyboards, function_results_collector, session_id, selected_tools
    )

    # Configure video agent LLM with automatic function calling
    # Note: LLM will be initialized after loading prompt template

    # Get context
    user_query = state["user_query"]
    instruction = task_instruction or state["agent_instructions"].get("video_agent", "Create video generation prompts")
    
    if "component_timings" not in state:
        state["component_timings"] = {}
    
    # Get template config with dynamic tool sections
    prompt_config = get_video_agent_prompt(state)

    # Build agent-specific context with tool information
    agent_specific = f"""### Available Video Generation Tools:
{prompt_config['tools_description']}

### Tool Selection Rules:
{prompt_config['selection_rules']}"""

    # Build complete context with all dynamic content
    from ..base import build_full_context
    context_content = build_full_context(
        state,
        agent_name="video_agent",
        user_query=user_query,
        instruction=instruction,
        agent_specific_context=agent_specific,
        context_summary=True,
        window_memory=True,
        include_generated_content=True,
        # include_reference_images=True,  # COMMENTED: video_agent orchestrator only routes tasks, sub-agents handle references
        include_reference_images=False,
        # include_image_annotations=True,  # COMMENTED: annotations are for reference images (disabled above), sub-agents build own context
        include_image_annotations=False,
        include_generated_assets=True,
        include_tool_selections=True
    )

    # Get 100% static template
    orchestration_prompt = prompt_config["template"]

    # Initialize LLM with prompt template as system_instruction
    video_agent_llm = get_llm(
        model="gemini",
        gemini_configs={
            'max_output_tokens': 5000,
            'temperature': 1.0,
            'top_p': 0.93,
            'tools': [get_available_shots, process_video_shots],
            'automatic_function_calling': True
        },
        system_instruction=orchestration_prompt
    )

    # Context and prompt are already available in the function scope if needed for debugging
    
    try:
        # Let the LLM orchestrate with automatic function calling
        print("[Video Agent] Executing video production workflow...")
        response = video_agent_llm.invoke(
            context_content,  # Only dynamic context, prompt is in system_instruction
            add_context=False,  # Don't concatenate, prompt already in system_instruction
            state=state,
            stream_callback=lambda event_type, content: emit_event(
                state,
                f"llm_{event_type}",
                {"content": content},
                agent_name="video_agent"
            ) if content else None
        )
        
        # Parse response
        if isinstance(response, str):
            parsed_response = json.loads(response)
        else:
            parsed_response = response
        
        # Extract thinking and final message
        thinking = ""
        final_message = ""
        
        for content in parsed_response.get('content', []):
            if content.get('thinking'):
                thinking = content['thinking']
            elif content.get('text'):
                final_message = content['text']
        
        # Store thinking process
        if "thinking_processes" not in state:
            state["thinking_processes"] = {}
        state["thinking_processes"]["video_agent"] = thinking

        # Backfill thinking into video tasks that were just submitted
        # Tasks were created during function execution, before thinking was extracted
        if thinking and "video_generation_tasks" in state:
            print(f"[Video Agent] Backfilling thinking into {len(state['video_generation_tasks'])} video tasks...")
            for task in state["video_generation_tasks"]:
                if task.get("metadata") and not task["metadata"].get("thinking"):
                    task["metadata"]["thinking"] = thinking

        # Process function call results from collector
        prompts_generated = 0
        tasks_submitted = 0
        errors = []

        print(f"\n[Video Agent] Processing {len(function_results_collector)} captured function results:")
        print("=" * 80)

        # Log all function calls made with their parameters (like character agent)
        for idx, captured in enumerate(function_results_collector, 1):
            func_name = captured.get("function", "unknown")
            params = captured.get("params", {})
            timestamp = captured.get("timestamp", "")
            result = captured.get("result", {})

            print(f"\n[Function Call #{idx}] {func_name}")
            print(f"  Timestamp: {timestamp}")
            print(f"  Parameters: {params}")

            # Show key result metrics
            if "prompts_generated" in result:
                prompts_generated = result['prompts_generated']
                print(f"  Result: Generated {prompts_generated} prompts")
            if "tasks_submitted" in result:
                tasks_submitted = result['tasks_submitted']
                print(f"  Result: Submitted {tasks_submitted} tasks")
            if "summary" in result:
                print(f"  Summary: {result['summary']}")

            # Check for submission_errors
            if 'submission_errors' in result and result['submission_errors']:
                errors.extend(result['submission_errors'])

        print("=" * 80)
        
        # Build final output
        state["video_output"] = {
            "prompts_generated": prompts_generated,
            "tasks_submitted": tasks_submitted,
            "submission_errors": errors if errors else None,
            "failed_shots": [err['shot'] for err in errors] if errors else [],
            "function_calls_metadata": function_results_collector if function_results_collector else None,
            "summary": final_message,
            "timestamp": datetime.now().isoformat(),
            "agent": "video_agent"
        }

        # Emit real-time events for all activities
        if prompts_generated > 0:
            state = emit_event(state, "video_prompts_generated", {
                "count": prompts_generated,
                "message": f"Generated {prompts_generated} video prompts"
            }, agent_name="video_agent")

        if tasks_submitted > 0:
            state = emit_event(state, "video_tasks_batch_submitted", {
                "count": tasks_submitted,
                "message": f"Submitted {tasks_submitted} video generation tasks"
            }, agent_name="video_agent")

        if errors:
            state = emit_event(state, "video_submission_errors", {
                "errors": errors,
                "failed_count": len(errors),
                "failed_shots": [err['shot'] for err in errors],
                "message": f"{len(errors)} shot(s) failed during submission"
            }, agent_name="video_agent")

        # Emit overall summary
        state = emit_event(state, "video_agent_summary", {
            "prompts_generated": prompts_generated,
            "tasks_submitted": tasks_submitted,
            "errors_count": len(errors),
            "success_rate": f"{tasks_submitted}/{prompts_generated}" if prompts_generated > 0 else "0/0"
        }, agent_name="video_agent")

        print(f"[Video Agent] Execution complete:")
        print(f"  - Prompts generated: {prompts_generated}")
        print(f"  - Tasks submitted: {tasks_submitted}")
        if errors:
            print(f"  - Submission errors: {len(errors)}")
            failed_shot_list = [f"Scene {e['shot']['scene']}, Shot {e['shot']['shot']}" for e in errors]
            print(f"  - Failed shots: {failed_shot_list}")
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        error_msg = f"Orchestration error: {str(e)}"
        print(f"[Video Agent] Error: {error_msg}\n{error_trace}")
        state["video_output"] = {
            "error": error_msg,
            "timestamp": datetime.now().isoformat(),
            "agent": "video_agent",
            "traceback": error_trace
        }
        
        if "thinking_processes" not in state:
            state["thinking_processes"] = {}
        state["thinking_processes"]["video_agent"] = f"Orchestration error: {error_msg}"
    
    # Record timing
    execution_time = time.time() - start_time
    state["component_timings"]["video_agent"] = execution_time
    print(f"[Video Agent] Completed in {execution_time:.2f}s")

    # Emit agent ended event
    state = emit_event(state, "agent_ended", {"agent": "video_agent"}, agent_name="video_agent")

    return state