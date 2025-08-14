"""
Audio generation agent for AI Video Studio
"""

import json
import time
from datetime import datetime
from typing import Dict, Any
from ...core.state import RAGState
from ...core.llm import get_llm
from ...prompts import AUDIO_AGENT_PROMPT_TEMPLATE
from ..base import get_agent_context, emit_event, clean_json_response


def get_audio_llm_choice(state: RAGState) -> str:
    """Get LLM model for audio agent. Always returns gemini."""
    return "gemini"


def audio_agent(state: RAGState, task_instruction: str = None) -> RAGState:
    """Audio generation agent - creates background music for video scenes

    Analyzes script and generates appropriate background music using ElevenLabs API.
    Uses automatic function calling to generate music based on scene requirements.
    """
    print("[Audio Agent] Starting audio generation...")
    start_time = time.time()

    # Set current agent for event tracking
    state["current_agent"] = "audio_agent"

    # Emit agent started event
    state = emit_event(state, "agent_started", {"agent": "audio_agent"}, agent_name="audio_agent")

    # Emit processing event
    state = emit_event(state, "processing", {"message": "Generating music..."}, agent_name="audio_agent")

    # Extract session ID from state
    session_id = state.get("session_id", "")
    print(f"[Audio Agent] Session ID: {session_id}")

    # Initialize audio_generation_metadata if not present
    if "audio_generation_metadata" not in state:
        state["audio_generation_metadata"] = {}

    # Create wrapper function with session context
    def generate_music_wrapper(
        prompt: str,
        name: str,
        duration_ms: int = 60000,
        force_instrumental: bool = True,
        output_format: str = "mp3_44100_128"
    ) -> Dict[str, Any]:
        """Generate background music with session context"""
        from .tools_audio import generate_music

        print(f"[Audio Agent] Generating music: {name}")
        print(f"  Prompt: {prompt[:100]}...")
        print(f"  Duration: {duration_ms}ms")

        try:
            # Call actual generate_music function
            gcs_url = generate_music(
                prompt=prompt,
                name=name,
                duration_ms=duration_ms,
                force_instrumental=force_instrumental,
                output_format=output_format,
                session_id=session_id
            )

            # Build metadata for this generation
            filename = f"{name}.mp3"  # Assuming mp3 format
            metadata = {
                "name": name,
                "prompt": prompt,
                "duration_ms": duration_ms,
                "path": gcs_url,
                "success": True,
                "timestamp": datetime.now().isoformat(),
                "params": {
                    "prompt": prompt,
                    "name": name,
                    "duration_ms": duration_ms,
                    "force_instrumental": force_instrumental,
                    "output_format": output_format
                }
            }

            # Store metadata in state
            state["audio_generation_metadata"][filename] = metadata
            print(f"[Audio Agent] Stored metadata for {filename}")

            # Register audio in asset_urls (following video pattern)
            if "asset_urls" not in state:
                state["asset_urls"] = {}
            if "generated_audio" not in state["asset_urls"]:
                state["asset_urls"]["generated_audio"] = []

            state["asset_urls"]["generated_audio"].append({
                "url": gcs_url,
                "name": name,
                "filename": filename,
                "duration_ms": duration_ms
            })

            # Emit per-generation event for real-time streaming
            emit_event(state, "audio_track_generated", {
                "name": name,
                "path": gcs_url,
                "filename": filename,
                "duration_ms": duration_ms
            }, agent_name="audio_agent")

            return {
                "success": True,
                "url": gcs_url,
                "filename": filename,
                "duration_ms": duration_ms
            }

        except Exception as e:
            error_msg = f"Music generation failed for {name}: {str(e)}"
            print(f"[Audio Agent] Error: {error_msg}")

            # Store error in metadata
            filename = f"{name}.mp3"
            metadata = {
                "name": name,
                "prompt": prompt,
                "duration_ms": duration_ms,
                "path": None,
                "success": False,
                "error": error_msg,
                "timestamp": datetime.now().isoformat(),
                "params": {
                    "prompt": prompt,
                    "name": name,
                    "duration_ms": duration_ms,
                    "force_instrumental": force_instrumental,
                    "output_format": output_format
                }
            }
            state["audio_generation_metadata"][filename] = metadata

            # Register failed audio in asset_urls (for consistency)
            if "asset_urls" not in state:
                state["asset_urls"] = {}
            if "generated_audio" not in state["asset_urls"]:
                state["asset_urls"]["generated_audio"] = []

            state["asset_urls"]["generated_audio"].append({
                "url": None,
                "name": name,
                "filename": filename,
                "duration_ms": duration_ms,
                "error": error_msg
            })

            return {
                "success": False,
                "error": error_msg
            }

    # Configure LLM with automatic function calling
    # Note: LLM will be initialized after loading prompt template

    user_query = state["user_query"]
    instruction = task_instruction or state["agent_instructions"].get("audio_agent", "Create background music for the video")

    if "component_timings" not in state:
        state["component_timings"] = {}

    # Build script data section for agent-specific context
    script_data_dict = state.get("generated_scripts", {})

    if script_data_dict:
        from ..base import get_filtered_script_data
        filtered_script = get_filtered_script_data(script_data_dict, "audio_agent")
        script_data = json.dumps(filtered_script, indent=2)
        script_title = filtered_script.get("script_details", {}).get("title", "Unknown")
        print(f"[Audio Agent] Found script: {script_title}")
        agent_specific = f"### Script Data for Audio Generation:\n{script_data}"
    else:
        print(f"[Audio Agent] No script in generated_scripts")
        agent_specific = "### Script Data:\nNo script data available - creating music based on user request"

    # Build complete context with all dynamic content
    from ..base import build_full_context
    context_content = build_full_context(
        state,
        agent_name="audio_agent",
        user_query=user_query,
        instruction=instruction,
        agent_specific_context=agent_specific,
        context_summary=True,
        window_memory=True,
        include_generated_content=True,
        include_generated_assets=True,
        include_metadata=False
    )

    # Get static template (no variables)
    audio_prompt = AUDIO_AGENT_PROMPT_TEMPLATE["template"]

    # Initialize LLM with prompt template as system_instruction
    audio_llm = get_llm(
        model="gemini",
        gemini_configs={
            'max_output_tokens': 5000,
            'temperature': 1.0,
            'top_p': 0.92,
            'tools': [generate_music_wrapper],
            'automatic_function_calling': True,
            'maximum_remote_calls': 25
        },
        system_instruction=audio_prompt
    )

    try:
        # Generate music with automatic function calling
        response = audio_llm.invoke(
            context_content,  # Only dynamic context, prompt is in system_instruction
            add_context=False,  # Don't concatenate, prompt already in system_instruction
            state=state,
            stream_callback=lambda event_type, content: emit_event(
                state,
                f"llm_{event_type}",
                {"content": content},
                agent_name="audio_agent"
            ) if content else None
        )

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

        # Extract function calls for debugging
        function_calls = parsed_response.get('function_calls', [])
        print(f"[Audio Agent] Found {len(function_calls)} function calls")

        # Extract track names from metadata for summary
        audio_metadata = state.get("audio_generation_metadata", {})
        all_track_names = []
        for filename, metadata in audio_metadata.items():
            track_name = metadata.get("name")
            if metadata.get("success") and track_name:
                all_track_names.append(track_name)

        print(f"[Audio Agent] Generated {len(all_track_names)} music tracks: {all_track_names}")

        # Store audio output in state
        state["audio_output"] = {
            "thinking": thinking,
            "timestamp": datetime.now().isoformat(),
            "agent": "audio_agent"
        }

        # Add function calls if they exist
        if "function_calls" in parsed_response:
            function_calls = parsed_response["function_calls"]
            cleaned_calls = [fc for fc in function_calls if isinstance(fc, dict)]
            state["audio_output"]["function_calls"] = cleaned_calls

        # Store thinking process for tracking
        if "thinking_processes" not in state:
            state["thinking_processes"] = {}
        state["thinking_processes"]["audio_agent"] = thinking

        # Store task status for debugging
        # state["audio_task_status"] = status_data
        print(f"[Audio Agent] Completed - {len(all_track_names)} tracks generated")

    except (json.JSONDecodeError, KeyError, IndexError, AttributeError, ValueError) as e:
        # Handle parsing errors gracefully
        error_msg = f"Error parsing audio response: {str(e)}"
        print(f"[Audio Agent] Error: {error_msg}")

        state["audio_output"] = {
            "error": error_msg,
            "timestamp": datetime.now().isoformat(),
            "agent": "audio_agent"
        }

        if "thinking_processes" not in state:
            state["thinking_processes"] = {}
        state["thinking_processes"]["audio_agent"] = f"Audio generation encountered an error: {error_msg}"

    # Extract audio metadata and emit completion event
    audio_metadata = state.get("audio_generation_metadata", {})
    all_track_names = []
    for filename, metadata in audio_metadata.items():
        track_name = metadata.get("name")
        if metadata.get("success") and track_name:
            all_track_names.append(track_name)

    # Emit completion event for workflow stage tracking
    state = emit_event(state, "audio_generated", {
        "track_names": all_track_names,
        "count": len(all_track_names),
        "total_tracks": len(audio_metadata),
        "audio_generation_metadata": audio_metadata
    }, agent_name="audio_agent")

    # Record timing
    execution_time = time.time() - start_time
    state["component_timings"]["audio_agent"] = execution_time
    print(f"[Audio Agent] Completed in {execution_time:.2f}s")

    # Emit agent ended event
    state = emit_event(state, "agent_ended", {"agent": "audio_agent"}, agent_name="audio_agent")

    return state
