"""
Video editor agent for post-production editing
Simple interface for combining videos with transitions and audio
"""

import os
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
from ...core.state import RAGState
from ...core.llm import get_llm
from ...prompts import VIDEO_EDITOR_PROMPT_TEMPLATE
from ..base import get_agent_context, emit_event, clean_json_response

def get_editor_llm_choice(state: RAGState) -> str:
    """Get LLM model for video editor agent. Always returns gemini."""
    return "gemini"

def video_editor_agent(state: RAGState) -> RAGState:
    """Video editor agent - combines generated videos with transitions and audio
    
    This agent:
    1. Finds available generated videos
    2. Determines editing sequence based on instructions
    3. Applies transitions and audio as requested
    4. Outputs final edited video
    """
    print("[Video Editor Agent] Starting video editing workflow...")
    start_time = time.time()
    
    # Set current agent for event tracking
    state["current_agent"] = "video_editor_agent"

    # Emit agent started event
    state = emit_event(state, "agent_started", {"agent": "video_editor_agent"}, agent_name="video_editor_agent")

    # Emit processing event
    state = emit_event(state, "processing", {"message": "Editing video..."}, agent_name="video_editor_agent")

    # Initialize thinking process tracking
    thinking_steps = []

    # Get session info and instructions
    session_id = state.get("session_id", "")
    user_query = state["user_query"]
    instruction = state.get("agent_instructions", {}).get("video_editor_agent", "")

    # Get aspect ratio from script to provide as context to the agent
    script_aspect_ratio = "vertical"  # Default suggestion
    generated_scripts = state.get("generated_scripts", {})
    if generated_scripts:
        # Check both old and new schema locations
        # New schema: metadata.aspect_ratio
        if "metadata" in generated_scripts:
            script_aspect_ratio = generated_scripts["metadata"].get("aspect_ratio", "vertical")
        # Old schema: script_details.aspect_ratio or video_metadata.aspect_ratio
        elif "script_details" in generated_scripts:
            script_aspect_ratio = generated_scripts["script_details"].get("aspect_ratio", "vertical")
        elif "video_metadata" in generated_scripts:
            script_aspect_ratio = generated_scripts["video_metadata"].get("aspect_ratio", "vertical")

    thinking_steps.append(f"Script suggests aspect ratio: {script_aspect_ratio}")
    
    
    thinking_steps.append(f"Initializing video editor with session ID: {session_id if session_id else 'default'}")
    thinking_steps.append(f"Processing request: {user_query[:50]}...")
    
    # Import editing tools
    from .tools_video_editor import (
        combine_videos_with_transitions,
        add_audio_to_video
    )
    
    # List all available videos from asset_urls (source of truth)
    all_videos = []

    # Extract videos from asset_urls["generated_videos"]
    generated_videos = state.get("asset_urls", {}).get("generated_videos", [])
    print(f"[Video Editor Agent] Found {len(generated_videos)} videos in asset_urls")

    for idx, video in enumerate(generated_videos):
        video_url = video.get("url")
        scene = video.get("scene", 0)
        shot = video.get("shot", 0)
        version = video.get("version", 1)

        if video_url and scene > 0 and shot > 0:
            filename = f"sc{scene:02d}_sh{shot:02d}_video_v{version}.mp4"

            all_videos.append({
                "scene": scene,
                "shot": shot,
                "version": version,
                "path": video_url,
                "filename": filename,
                "size_mb": 0
            })
            print(f"[Video Editor Agent] Added: {filename}")
        else:
            print(f"[Video Editor Agent] Skipping invalid video entry: scene={scene}, shot={shot}, url={video_url[:50] if video_url else 'None'}...")

    # Sort by scene, shot, and version
    all_videos.sort(key=lambda x: (x["scene"], x["shot"], x["version"]))

    thinking_steps.append(f"Located {len(all_videos)} video files for editing")
    
    # Check if we have videos
    if not all_videos:
        print("[Video Editor Agent] No videos found to edit")
        thinking_steps.append("No videos available to edit")
        
        if "thinking_processes" not in state:
            state["thinking_processes"] = {}
        state["thinking_processes"]["video_editor_agent"] = " -> ".join(thinking_steps)
        
        state["video_editor_output"] = {
            "error": "No videos found in clips folder",
            "timestamp": datetime.now().isoformat(),
            "agent": "video_editor_agent"
        }
        state = emit_event(state, "agent_ended", {"agent": "video_editor_agent"}, agent_name="video_editor_agent")
        return state
    
    print(f"[Video Editor Agent] Found {len(all_videos)} videos to edit")

    # Extract available music tracks from audio_generation_metadata
    available_audio = {}
    audio_metadata = state.get("audio_generation_metadata", {})
    print(f"[Video Editor Agent] DEBUG: audio_generation_metadata has {len(audio_metadata)} entries")

    for filename, meta in audio_metadata.items():
        print(f"[Video Editor Agent] DEBUG: Processing '{filename}' - success={meta.get('success')}, has_path={bool(meta.get('path'))}")
        if meta.get("success") and meta.get("path"):
            track_name = meta["name"]
            audio_path = meta["path"]

            # Convert gs:// to signed URL for audio analysis
            if audio_path.startswith("gs://"):
                from ...storage.gcs_utils import generate_signed_url
                audio_url_for_analysis = generate_signed_url(audio_path, expiration_days=1)
                print(f"[Video Editor Agent] DEBUG: Converted gs:// to signed URL for analysis")
            else:
                audio_url_for_analysis = audio_path

            available_audio[track_name] = {
                "url": audio_url_for_analysis,
                "duration_ms": meta.get("duration_ms"),
                "prompt": meta.get("prompt", "")[:100]
            }
            print(f"[Video Editor Agent] DEBUG: Added track '{track_name}' to available_audio")

    if available_audio:
        thinking_steps.append(f"Found {len(available_audio)} available music tracks")
        print(f"[Video Editor Agent] Available audio tracks: {list(available_audio.keys())}")
        print(f"[Video Editor Agent] DEBUG: Full available_audio dict: {json.dumps(available_audio, indent=2)}")
    else:
        thinking_steps.append("No audio tracks available")
        print(f"[Video Editor Agent] No audio tracks available for mixing")

    # Extract only relevant script data (production notes + shot continuity)
    # Video editor doesn't need full script - only style/tone guidance and continuity constraints
    full_script = state.get("generated_scripts", {})

    minimal_script_context = {
        "production_notes": full_script.get("production_notes", {}),
        "aspect_ratio": full_script.get("script_details", {}).get("aspect_ratio", "vertical")
    }

    # Extract shot-level continuity map
    shot_continuity = []
    script_details = full_script.get("script_details", {})
    scenes = script_details.get("scenes", [])
    for scene in scenes:
        scene_num = scene.get("scene_number", 0)
        for shot in scene.get("shots", []):
            shot_num = shot.get("shot_number", 0)
            shot_id = f"sc{scene_num:02d}_sh{shot_num:02d}"
            continuity = shot.get("continuity_notes", {})
            if continuity:  # Only include if continuity notes exist
                shot_continuity.append({
                    "shot_id": shot_id,
                    "continuity": continuity
                })

    script_json = json.dumps(minimal_script_context, indent=2) if minimal_script_context else "No script available"
    continuity_json = json.dumps(shot_continuity, indent=2) if shot_continuity else "No shot continuity information"

    thinking_steps.append(f"Extracted minimal script context: production_notes + {len(shot_continuity)} shots with continuity info")

    # Run analysis upfront (system-side, before LLM)
    from .tools_video_analysis import analyze_video_clips
    from .tools_audio_analysis import analyze_music_beats
    from langchain_core.runnables.config import ContextThreadPoolExecutor
    from concurrent.futures import as_completed

    video_analysis = None
    music_analysis = None

    # Determine what needs to be analyzed
    needs_video_analysis = len(all_videos) > 0
    needs_audio_analysis = available_audio and len(available_audio) > 0

    # Run video and audio analysis in parallel with context propagation
    if needs_video_analysis or needs_audio_analysis:
        print(f"[Video Editor Agent] Starting parallel analysis (video: {needs_video_analysis}, audio: {needs_audio_analysis})...")

        with ContextThreadPoolExecutor(max_workers=2) as executor:
            futures = {}

            # Submit video analysis task
            if needs_video_analysis:
                clip_urls = [v["path"] for v in all_videos]
                futures['video'] = executor.submit(analyze_video_clips, clip_urls, state)
                thinking_steps.append(f"Running video analysis on {len(all_videos)} clips")
                print(f"[Video Editor Agent] → Submitted video analysis for {len(all_videos)} clips")

            # Submit audio analysis task
            if needs_audio_analysis:
                first_audio_track = list(available_audio.values())[0]
                first_audio_name = list(available_audio.keys())[0]
                audio_url = first_audio_track["url"]
                futures['audio'] = executor.submit(analyze_music_beats, audio_url, state)
                thinking_steps.append("Running music beat analysis")
                print(f"[Video Editor Agent] → Submitted music analysis for '{first_audio_name}'")

            # Collect results as they complete
            for task_name, future in futures.items():
                try:
                    result = future.result()

                    if task_name == 'video':
                        if "clips" in result and not result.get("error"):
                            video_analysis = json.dumps(result, indent=2)
                            thinking_steps.append("Video analysis completed successfully")
                            print(f"[Video Editor Agent] ✓ Video analysis completed: {result.get('total_clips', 0)} clips analyzed")
                        else:
                            thinking_steps.append(f"Video analysis error: {result.get('error', 'Unknown error')}")
                            print(f"[Video Editor Agent] ✗ Video analysis error: {result.get('error', 'Unknown error')}")

                    elif task_name == 'audio':
                        if result.get("status") != "error":
                            music_analysis = json.dumps(result, indent=2)
                            tempo = result.get("tempo_bpm", "N/A")
                            beats = len(result.get("beat_timestamps", []))
                            thinking_steps.append("Music analysis completed successfully")
                            print(f"[Video Editor Agent] ✓ Music analysis completed: {tempo} BPM, {beats} beats detected")
                        else:
                            thinking_steps.append(f"Music analysis error: {result.get('error', 'Unknown error')}")
                            print(f"[Video Editor Agent] ✗ Music analysis error: {result.get('error', 'Unknown error')}")

                except Exception as e:
                    thinking_steps.append(f"{task_name.title()} analysis error: {str(e)}")
                    print(f"[Video Editor Agent] ✗ {task_name.title()} analysis error: {e}")

        print(f"[Video Editor Agent] Parallel analysis complete")
    else:
        print(f"[Video Editor Agent] No analysis needed (no videos or audio tracks)")

    # Configure editor LLM (no tools - declarative mode)
    # Note: LLM will be initialized after loading prompt template

    # Build agent-specific context with all editing resources
    agent_specific = f"""### Available Videos for Editing:
{json.dumps(all_videos, indent=2)}

### Script Information:
Aspect Ratio: {script_aspect_ratio}
Production Notes and Continuity:
{script_json}

### Shot Continuity Information:
{continuity_json}

### Available Audio Tracks:
{json.dumps(available_audio, indent=2) if available_audio else "No audio tracks available"}

### Video Analysis Results:
{video_analysis if video_analysis else "No video analysis available (use for simple edits)"}

### Music Analysis Results:
{music_analysis if music_analysis else "No music analysis available"}"""

    # Build minimal context (video editor works from analysis data, not conversation history)
    from ..base import build_full_context
    context_content = build_full_context(
        state,
        agent_name="video_editor_agent",
        user_query=user_query,
        instruction=instruction or user_query,
        agent_specific_context=agent_specific,
        context_summary=False,
        window_memory=False,
        include_generated_content=True
    )

    # Get static template (no variables)
    prompt = VIDEO_EDITOR_PROMPT_TEMPLATE["template"]

    # Get or create cache
    from src.core.cache_registry import get_or_create_cache
    cached_content = get_or_create_cache(
        agent_name="video_editor",
        system_instruction=prompt,
        state=state
    )

    # Initialize LLM
    editor_llm = get_llm(
        model="gemini",
        gemini_configs={'max_output_tokens': 8000, 'temperature': 1.0, 'top_p': 0.9},
        system_instruction=prompt,
        cached_content_name=cached_content
    )

    # Context and prompt are already available in the function scope if needed for debugging

    try:
        # Get editing decisions from LLM
        response = editor_llm.invoke(
            context_content,  # Only dynamic context, prompt is in system_instruction
            add_context=False,  # Don't concatenate, prompt already in system_instruction
            response_schema="video_editor_agent",
            state=state,
            stream_callback=lambda event_type, content: emit_event(
                state,
                f"llm_{event_type}",
                {"content": content},
                agent_name="video_editor_agent"
            ) if content else None
        )
        
        thinking_steps.append(f"Analyzing {len(all_videos)} videos for editing plan")

        # Parse the extracted response
        print(f"[Video Editor Agent] Parsing LLM response (type: {type(response).__name__})...")

        # Extract thinking content and full response for metadata
        llm_thinking = ""
        llm_full_response = response

        try:
            if isinstance(response, str):
                parsed = json.loads(response)

                # Extract thinking content
                if isinstance(parsed, dict) and "content" in parsed:
                    for item in parsed.get("content", []):
                        if isinstance(item, dict) and "thinking" in item:
                            llm_thinking = item["thinking"]
                            break

                    # Extract the actual JSON from the text field
                    for item in parsed.get("content", []):
                        if isinstance(item, dict) and "text" in item:
                            editing_plan = json.loads(clean_json_response(item["text"]))
                            break
                    else:
                        editing_plan = parsed
                else:
                    editing_plan = parsed
            else:
                editing_plan = response

            thinking_steps.append("Received editing plan from LLM")
            print(f"[Video Editor Agent] ✓ Editing plan parsed: '{editing_plan.get('edit_name', 'unnamed')}'")
            print(f"[Video Editor Agent]   Videos selected: {len(editing_plan.get('selected_videos', []))}")
            print(f"[Video Editor Agent]   Transitions specified: {len(editing_plan.get('transitions', []))}")
            print(f"[Video Editor Agent]   Add audio: {editing_plan.get('add_audio', False)}")

            # Extract aspect ratio from agent's decision
            aspect_ratio = editing_plan.get("aspect_ratio", script_aspect_ratio)
            thinking_steps.append(f"Agent chose aspect ratio: {aspect_ratio}")

        except Exception as e:
            # If not JSON, use simple config with all videos
            editing_plan = {
                "edit_name": "fallback_simple",
                "selected_videos": [{"filename": v["filename"], "trim": None} for v in all_videos],
                "transitions": [],
                "add_audio": False,
                "selected_audio": None,
                "audio_volume": 0.7,
                "mix_original_audio": True,
                "aspect_ratio": script_aspect_ratio,
                "notes": "Fallback plan - LLM response parsing error"
            }
            aspect_ratio = script_aspect_ratio
            thinking_steps.append("Using fallback editing plan - LLM parsing error")
            print(f"[Video Editor Agent] WARNING: LLM parsing error, using fallback: {str(e)}")
        
        
        # Log the keys if it's a dict
        if isinstance(editing_plan, dict):
            print(f"[Video Editor] Editing plan keys: {editing_plan.keys()}")

            # Ensure we have aspect_ratio from the editing plan
            if "aspect_ratio" not in editing_plan:
                aspect_ratio = script_aspect_ratio
                thinking_steps.append(f"No aspect ratio in plan, using script suggestion: {script_aspect_ratio}")
            else:
                aspect_ratio = editing_plan.get("aspect_ratio")
                thinking_steps.append(f"Using agent's aspect ratio choice: {aspect_ratio}")
        
        # Store thinking process (update continuously)
        if "thinking_processes" not in state:
            state["thinking_processes"] = {}

        # Hybrid format: real LLM thinking + execution steps
        if llm_thinking:
            state["thinking_processes"]["video_editor_agent"] = f"{llm_thinking}\n\n---\n\n**Execution Steps:**\n{' -> '.join(thinking_steps)}"
        else:
            state["thinking_processes"]["video_editor_agent"] = " -> ".join(thinking_steps)

        # Execute editing based on plan
        results = []

        # Get selected videos from editing plan (new format with inline trim specs)
        selected_video_specs = editing_plan.get("selected_videos", [])
        if not selected_video_specs:
            # If no selection provided, use all videos without trimming
            selected_video_specs = [{"filename": v["filename"], "trim": None, "mute_audio": None} for v in all_videos]
            print("[Video Editor Agent] No specific selection, using all available videos")

        # Build ordered list of video specs with URLs
        video_specs = []
        for spec in selected_video_specs:
            filename = spec.get("filename")
            trim_spec = spec.get("trim")
            mute_audio = spec.get("mute_audio")

            # Find matching video URL - try exact match first
            found = False
            for video in all_videos:
                if video["filename"] == filename:
                    video_specs.append({
                        "url": video["path"],
                        "filename": filename,
                        "trim": trim_spec,
                        "mute_audio": mute_audio
                    })
                    found = True
                    break

            # Fuzzy matching fallback: extract scene/shot from requested filename
            if not found:
                # Parse scene/shot from requested filename
                match = re.match(r'sc(\d+)_sh(\d+)_video_v(\d+)\.mp4', filename)
                if match:
                    req_scene = int(match.group(1))
                    req_shot = int(match.group(2))
                    req_version = int(match.group(3))

                    # Try to find by scene/shot (ignore version mismatch)
                    for video in all_videos:
                        if video["scene"] == req_scene and video["shot"] == req_shot:
                            print(f"[Video Editor Agent] WARNING - FUZZY MATCH: Requested '{filename}' but using '{video['filename']}' (scene={req_scene}, shot={req_shot})")
                            video_specs.append({
                                "url": video["path"],
                                "filename": video["filename"],  # Use actual filename
                                "trim": trim_spec,
                                "mute_audio": mute_audio
                            })
                            found = True
                            break

            if not found:
                print(f"[Video Editor Agent] WARNING - No match found for: {filename}")
                print(f"[Video Editor Agent] Available videos: {[v['filename'] for v in all_videos]}")

        if not video_specs:
            print("[Video Editor Agent] ERROR: Could not find paths for selected videos")
            state["video_editor_output"] = {
                "error": "Selected videos not found",
                "timestamp": datetime.now().isoformat(),
                "agent": "video_editor_agent"
            }
            state = emit_event(state, "agent_ended", {"agent": "video_editor_agent"}, agent_name="video_editor_agent")
            return state

        print(f"[Video Editor Agent] Using {len(video_specs)} selected videos")
        thinking_steps.append(f"Selected {len(video_specs)} videos for editing")

        # Log trim specs
        for spec in video_specs:
            if spec["trim"]:
                print(f"[Video Editor Agent]   {spec['filename']}: trim {spec['trim']['start']}s-{spec['trim']['end']}s")
            else:
                print(f"[Video Editor Agent]   {spec['filename']}: full clip")
        
        # Parse transitions from editing plan
        transitions_list = editing_plan.get("transitions", [])
        transitions = {}
        transition_durations = {}

        if transitions_list:
            # Convert list of transition objects to tuple-keyed dicts
            for trans in transitions_list:
                from_video = trans.get("from")
                to_video = trans.get("to")
                trans_type = trans.get("type")
                trans_dur = trans.get("duration", 0.5)

                transitions[(from_video, to_video)] = trans_type
                transition_durations[(from_video, to_video)] = trans_dur
                print(f"[Video Editor Agent] Added transition: ({from_video}, {to_video}) -> {trans_type}, {trans_dur}s")

        if transitions:
            thinking_steps.append(f"Configured {len(transitions)} transitions")
        else:
            thinking_steps.append("Using hard cuts (no transitions)")

        # Determine if audio mixing is needed (hybrid mode decision)
        add_audio = editing_plan.get("add_audio", False)
        selected_audio = editing_plan.get("selected_audio")
        use_audio = add_audio and selected_audio and selected_audio in available_audio

        # Fuzzy matching: Try stripping common audio extensions if exact match fails
        if not use_audio and add_audio and selected_audio:
            for ext in ['.mp3', '.wav', '.opus', '.m4a', '.aac']:
                if selected_audio.endswith(ext):
                    name_without_ext = selected_audio[:-len(ext)]
                    if name_without_ext in available_audio:
                        print(f"[Video Editor Agent] Fuzzy match: '{selected_audio}' → '{name_without_ext}'")
                        selected_audio = name_without_ext
                        use_audio = True
                        break

        print(f"[Video Editor Agent] DEBUG: LLM selected_audio='{editing_plan.get('selected_audio')}', fuzzy_matched='{selected_audio}', available={list(available_audio.keys())}, match={use_audio}")

        if use_audio:
            thinking_steps.append(f"Audio requested: {selected_audio}")
            print(f"[Video Editor Agent] Audio mixing enabled with track '{selected_audio}' at URL: {available_audio[selected_audio]['url'][:80]}...")
        elif selected_audio:
            print(f"[Video Editor Agent] WARNING: Audio track '{selected_audio}' not found in available_audio (name mismatch)")
        else:
            print(f"[Video Editor Agent] No audio selected by LLM")

        # Step 1: Combine videos with transitions
        if video_specs:
            print(f"[Video Editor Agent] Combining {len(video_specs)} videos...")

            thinking_steps.append(f"Starting video combination for {len(video_specs)} clips")

            # Hybrid mode: use file mode if audio needed, buffer mode otherwise
            if use_audio:
                import tempfile
                fd, temp_combined_path = tempfile.mkstemp(suffix=".mp4", prefix="combined_")
                os.close(fd)
                output_path = temp_combined_path
                thinking_steps.append("Using file mode for audio mixing")
                print(f"[Video Editor Agent] MODE: File (audio mixing) - temp: {temp_combined_path}")
            else:
                output_path = None
                thinking_steps.append("Using buffer mode for fast processing")
                print(f"[Video Editor Agent] MODE: Buffer (no audio, fast path)")

            combine_result = combine_videos_with_transitions(
                video_specs=video_specs,
                transitions=transitions,
                transition_durations=transition_durations,
                output_path=output_path,
                session_id=session_id,
                aspect_ratio=aspect_ratio,
                process_audio=use_audio
            )

            # Store sanitized result without buffer
            result_for_state = {k: v for k, v in combine_result.items() if k != "output_buffer"}
            results.append(result_for_state)

            if combine_result["status"] == "success":
                print(f"[Video Editor Agent] Videos combined successfully")
                thinking_steps.append(f"Successfully combined {len(video_specs)} videos")

                # Update thinking in state
                if llm_thinking:
                    state["thinking_processes"]["video_editor_agent"] = f"{llm_thinking}\n\n---\n\n**Execution Steps:**\n{' -> '.join(thinking_steps)}"
                else:
                    state["thinking_processes"]["video_editor_agent"] = " -> ".join(thinking_steps)

                # Step 2: Add audio if requested (file mode only)
                audio_was_added = False
                selected_audio_track = None
                final_path = None

                if use_audio:
                    audio_url = available_audio[selected_audio]["url"]
                    audio_volume = editing_plan.get("audio_volume", 0.7)
                    mix_original = True  # Always mix - preserves video audio (dialogue/sounds) and overlays music
                    print(f"[Video Editor Agent] Adding audio '{selected_audio}' - volume={audio_volume}, mix_original={mix_original}")
                    print(f"[Video Editor Agent] DEBUG: audio_url={audio_url[:80]}..., video_path={temp_combined_path}")
                    thinking_steps.append(f"Mixing audio: {selected_audio} (volume: {audio_volume})")

                    audio_result = add_audio_to_video(
                        video_path=temp_combined_path,
                        audio_path=audio_url,
                        audio_volume=audio_volume,
                        mix_original=mix_original,
                        session_id=session_id
                    )

                    if audio_result["status"] == "success":
                        final_path = audio_result["output_path"]
                        audio_was_added = True
                        selected_audio_track = selected_audio
                        thinking_steps.append("Audio successfully mixed into video")
                        print(f"[Video Editor Agent] Audio mixed successfully - output: {final_path}")
                    else:
                        final_path = temp_combined_path
                        error_msg = audio_result.get("message", "Unknown error")
                        thinking_steps.append(f"Audio mixing error: {error_msg}")
                        print(f"[Video Editor Agent] ERROR: Audio mixing error - {error_msg}, using video without audio")
                else:
                    # No audio requested
                    final_path = None

                # Determine version for edited video
                from .tools_video import upload_video_to_gcs_public
                existing_edited = state.get("edited_videos", [])
                version = len(existing_edited) + 1

                # Upload to GCS (buffer or file depending on mode)
                upload_session_id = session_id if session_id else "unknown"

                # Determine filename suffix
                music_suffix = "_music" if audio_was_added else ""

                if use_audio:
                    # Upload file to GCS
                    print(f"[Video Editor Agent] Uploading edited video with audio to GCS (version {version})...")
                    public_url = upload_video_to_gcs_public(
                        video_source=final_path,
                        session_id=upload_session_id,
                        scene=0,
                        shot=0,
                        version=version,
                        suffix=music_suffix,
                        return_format="gs"  # Request gs:// path format
                    )

                    # Cleanup temp files
                    if os.path.exists(temp_combined_path):
                        os.unlink(temp_combined_path)
                    if audio_was_added and final_path != temp_combined_path and os.path.exists(final_path):
                        os.unlink(final_path)
                else:
                    # Upload buffer to GCS (zero-latency)
                    video_buffer = combine_result.get("output_buffer")
                    print(f"[Video Editor Agent] Uploading buffer to GCS (version {version})...")
                    public_url = upload_video_to_gcs_public(
                        video_source=video_buffer,
                        session_id=upload_session_id,
                        scene=0,
                        shot=0,
                        version=version,
                        suffix=music_suffix,
                        return_format="gs"  # Request gs:// path format
                    )

                # Set logical filename (matches GCS filename now)
                final_video = f"edited{music_suffix}_v{version}.mp4"
                print(f"[Video Editor Agent] Final video: {final_video} (suffix='{music_suffix}', audio_added={audio_was_added}, track={selected_audio_track})")
                
                if public_url:
                    print(f"[Video Editor Agent] Uploaded to GCS: {public_url}")
                    thinking_steps.append("Uploaded edited video to cloud storage")
                else:
                    print(f"[Video Editor Agent] GCS upload error")
                    thinking_steps.append("Cloud upload error")
                    public_url = ""  # Ensure it's defined

                # Build analysis summaries (analysis ran upfront in declarative mode)
                video_analysis_summary = {"provided": video_analysis is not None} if video_analysis else None
                audio_analysis_summary = {"provided": music_analysis is not None} if music_analysis else None

                # Convert transitions dict to JSON-serializable format for metadata
                # This is done early so we can include it in edited_videos entry
                transitions_for_metadata = []
                if transitions:
                    for (from_video, to_video), trans_type in transitions.items():
                        duration = transition_durations.get((from_video, to_video), 0.5) if transition_durations else 0.5
                        transitions_for_metadata.append({
                            "from": from_video,
                            "to": to_video,
                            "type": trans_type,
                            "duration": duration
                        })

                # Update state with edited video (append to existing)
                edited_videos = state.get("edited_videos", [])
                edited_videos.append({
                    "path": public_url,
                    "logical_name": final_video,
                    "public_url": public_url,
                    "timestamp": datetime.now().isoformat(),
                    "source_videos": len(video_specs),
                    "transitions_applied": len(transitions) if transitions else 0,
                    "audio_added": audio_was_added,
                    "audio_track": selected_audio_track,
                    "audio_volume": editing_plan.get("audio_volume", 0.7) if audio_was_added else None,
                    "version": version,
                    "edit_name": editing_plan.get("edit_name", f"edit_v{version}"),
                    "analysis_used": {
                        "video_analysis": video_analysis_summary,
                        "audio_analysis": audio_analysis_summary
                    } if (video_analysis_summary or audio_analysis_summary) else None,
                    # Additional metadata for display
                    "operations_log": " -> ".join(thinking_steps),
                    "llm_thinking": llm_thinking,
                    "llm_full_response": llm_full_response,
                    "video_analysis_data": video_analysis,
                    "audio_analysis_data": music_analysis,
                    "aspect_ratio": aspect_ratio,
                    "transitions_list": transitions_for_metadata,
                    "editing_notes": editing_plan.get("notes", "")
                })
                state["edited_videos"] = edited_videos

                thinking_steps.append(f"Final output: {final_video}")
                thinking_steps.append(f"Total operations: {len(results)} completed")

                # Success output - declarative editing complete
                state["video_editor_output"] = {
                    "status": "success",
                    "final_video": final_video,
                    "path": public_url,
                    "public_url": public_url,
                    "editing_plan": editing_plan,
                    "transitions_applied": transitions_for_metadata,
                    "operations": results,
                    "videos_processed": len(video_specs),
                    "timestamp": datetime.now().isoformat(),
                    "agent": "video_editor_agent",
                    "version": version
                }

                print(f"[Video Editor Agent] Editing complete: {final_video}")

                # Emit editor_generated event for real-time streaming
                state = emit_event(state, "editor_generated", {
                    "edited_videos": state.get("edited_videos", []),
                    "final_video": final_video,
                    "public_url": public_url
                }, agent_name="video_editor_agent")
            else:
                # Combine failed
                error_msg = combine_result.get("message", "Failed to combine videos")
                thinking_steps.append(f"Video combination error: {error_msg}")

                print(f"[Video Editor Agent] Video combination error: {error_msg}")

                state["video_editor_output"] = {
                    "status": "error",
                    "error": error_msg,
                    "timestamp": datetime.now().isoformat(),
                    "agent": "video_editor_agent"
                }

                # Emit error event for frontend display
                state = emit_event(state, "editor_generated", {
                    "status": "error",
                    "error": error_msg
                }, agent_name="video_editor_agent")
        else:
            # No videos to edit
            thinking_steps.append("No videos available to edit")
            
            if "thinking_processes" not in state:
                state["thinking_processes"] = {}
            state["thinking_processes"]["video_editor_agent"] = " -> ".join(thinking_steps)
            
            state["video_editor_output"] = {
                "status": "error", 
                "error": "No videos available to edit",
                "timestamp": datetime.now().isoformat(),
                "agent": "video_editor_agent"
            }
            
    except Exception as e:
        import traceback
        error_details = f"{str(e)}\n{traceback.format_exc()}"
        print(f"[Video Editor Agent] Exception: {error_details}")

        thinking_steps.append(f"Error occurred: {str(e)}")

        if "thinking_processes" not in state:
            state["thinking_processes"] = {}

        # Hybrid format: real LLM thinking + execution steps
        if llm_thinking:
            state["thinking_processes"]["video_editor_agent"] = f"{llm_thinking}\n\n---\n\n**Execution Steps:**\n{' -> '.join(thinking_steps)}"
        else:
            state["thinking_processes"]["video_editor_agent"] = " -> ".join(thinking_steps)

        state["video_editor_output"] = {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "agent": "video_editor_agent"
        }

        # Emit error event for frontend
        state = emit_event(state, "editor_generated", {
            "status": "error",
            "error": str(e)
        }, agent_name="video_editor_agent")
    
    # Final thinking update
    if "thinking_processes" not in state:
        state["thinking_processes"] = {}
    if thinking_steps:
        # Hybrid format: real LLM thinking + execution steps
        if llm_thinking:
            state["thinking_processes"]["video_editor_agent"] = f"{llm_thinking}\n\n---\n\n**Execution Steps:**\n{' -> '.join(thinking_steps)}"
        else:
            state["thinking_processes"]["video_editor_agent"] = " -> ".join(thinking_steps)
    
    # Record timing
    execution_time = time.time() - start_time
    if "component_timings" not in state:
        state["component_timings"] = {}
    state["component_timings"]["video_editor_agent"] = execution_time

    print(f"[Video Editor Agent] Completed in {execution_time:.2f}s")

    # Emit agent ended event
    state = emit_event(state, "agent_ended", {"agent": "video_editor_agent"}, agent_name="video_editor_agent")

    return state