"""
Video task monitoring agent for AI Video Studio
Monitors video generation tasks submitted by the video agent
"""

import os
import json
import time
import logging
from datetime import datetime
from typing import Dict, Any, List
from ...core.state import RAGState
from ...core.llm import get_llm
from ..base import emit_event

logger = logging.getLogger(__name__)


def video_task_monitor(state: RAGState) -> RAGState:
    """Monitor video generation tasks until completion

    This agent:
    1. Retrieves submitted video tasks from state
    2. Polls video API for task status
    3. Waits for completion or timeout
    4. Reports final results to answer parser
    """
    print("[Video Task Monitor] Starting task monitoring...")
    start_time = time.time()

    # Set current agent for event tracking
    state["current_agent"] = "video_task_monitor"

    # Emit agent started event
    state = emit_event(state, "agent_started", {"agent": "video_task_monitor"}, agent_name="video_task_monitor")

    # Get tasks from state and filter to only unprocessed tasks
    all_tasks = state.get("video_generation_tasks", [])
    tasks = [t for t in all_tasks if t.get("status") not in ["completed", "failed"]]

    if not tasks:
        print("[Video Task Monitor] No tasks to monitor (all tasks already completed/error)")
        state["video_monitor_output"] = {
            "status": "no_tasks",
            "message": "No video generation tasks found to monitor",
            "timestamp": datetime.now().isoformat()
        }
        state = emit_event(state, "agent_ended", {"agent": "video_task_monitor"}, agent_name="video_task_monitor")
        return state

    print(f"[Video Task Monitor] Found {len(tasks)} tasks to monitor ({len(all_tasks)} total in state)")

    # Import registry for automatic routing
    from ..creative.tools_video import get_generator_for_tool
    
    # Monitoring configuration
    max_wait = int(state.get("video_monitor_timeout", 600))  # 10 minutes default
    check_interval = 12  # Check every 12 seconds (optimized from 8s for fewer API calls)
    
    # Track task states
    completed = []
    failed = []
    pending = tasks.copy()
    
    print(f"[Video Task Monitor] Monitoring {len(pending)} tasks (timeout: {max_wait}s)")
    
    # Monitoring loop
    while pending and (time.time() - start_time) < max_wait:
        still_pending = []
        
        for task in pending:
            task_id = task.get("task_id")
            scene = task.get("scene")
            shot = task.get("shot")
            tool_used = task.get("tool_used", "google_veo_i2v")

            try:
                # Use registry for automatic routing
                provider_info = get_generator_for_tool(tool_used)
                generator = provider_info["generator"]
                task_id_type = provider_info["task_id_type"]

                # Format task_id based on provider requirements
                task_id_param = int(task_id) if task_id_type == "integer" else str(task_id)

                logger.info(f"[VideoMonitor] Polling {tool_used} task={task_id_param}, scene={scene}, shot={shot}")
                print(f"[Monitor] Polling {tool_used} task {task_id_param} (scene {scene}, shot {shot})")

                # Query task status (unified interface across all providers)
                result = generator.query_task(task_id_param)
                
                if result.get("code") == 0:
                    data = result.get("data", {})
                    status = data.get("task_status", "unknown")
                    
                    if status == "succeed":
                        # Extract video URL
                        videos = data.get("task_result", {}).get("videos", [])
                        video_url = videos[0].get("url") if videos else None

                        logger.info(f"[VideoMonitor] Task {task_id_param} succeeded, downloading video")

                        # Initialize version tracking
                        version = 1  # Default version

                        # Stream video directly to memory and upload to GCS
                        local_path = None
                        public_url = ""  # Initialize to avoid NameError
                        if video_url:
                            try:
                                # Get session ID for GCS upload
                                session_id = state.get("session_id", "")

                                # Find next available version number from asset_urls (source of truth)
                                existing_videos = state.get("asset_urls", {}).get("generated_videos", [])
                                existing_versions = [
                                    v.get("version", 1)
                                    for v in existing_videos
                                    if v.get("scene") == scene and v.get("shot") == shot
                                ]
                                version = max(existing_versions) + 1 if existing_versions else 1

                                base_name = f"sc{scene:02d}_sh{shot:02d}_video"
                                video_filename = f"{base_name}_v{version}.mp4"

                                # Stream video directly to memory (no disk I/O)
                                import requests
                                from io import BytesIO

                                print(f"  Streaming video from API to memory...")
                                response = requests.get(video_url, stream=True, timeout=60)
                                response.raise_for_status()

                                video_buffer = BytesIO()
                                for chunk in response.iter_content(chunk_size=1024*1024):  # 1MB chunks
                                    video_buffer.write(chunk)
                                video_buffer.seek(0)  # Reset buffer position for reading

                                print(f"  Streamed: {video_filename} (version {version})")

                                # Upload from memory to GCS - request gs:// path format
                                from ...agents.creative.tools_video import upload_video_to_gcs_public
                                upload_session_id = session_id if session_id else "unknown"
                                public_url = upload_video_to_gcs_public(
                                    video_source=video_buffer,
                                    session_id=upload_session_id,
                                    scene=scene,
                                    shot=shot,
                                    version=version,
                                    return_format="gs"  # Request gs:// path (50 chars vs 500+ for signed URL)
                                )
                                if public_url:
                                    print(f"  Uploaded to GCS: {public_url}")
                                else:
                                    print(f"  GCS upload error")

                            except Exception as e:
                                print(f"  Video streaming/upload error: {str(e)}")
                                public_url = ""  # No public URL if streaming failed
                        
                        # Update task with completion info
                        completed_task = {**task}
                        # Always update top-level fields
                        completed_task["status"] = "completed"
                        completed_task["path"] = public_url or video_url
                        completed_task["success"] = True
                        completed_task["video_url"] = video_url
                        completed_task["local_path"] = None  # Not saved to disk (streamed to GCS)
                        completed_task["public_url"] = public_url
                        completed_task["version"] = version
                        completed_task["completed_at"] = datetime.now().isoformat()

                        # Also update params for completeness
                        if "params" in completed_task:
                            completed_task["params"] = {
                                **task["params"],
                                "status": "completed",
                                "video_url": video_url,
                                "local_path": None,  # Not saved to disk
                                "public_url": public_url,
                                "completed_at": datetime.now().isoformat()
                            }
                        completed.append(completed_task)
                        print(f"  Completed: Scene {scene}, Shot {shot}")

                        # Emit completion event
                        state = emit_event(state, "video_task_completed", {
                            "task_id": task_id,
                            "scene": scene,
                            "shot": shot,
                            "video_url": public_url or video_url,
                            "tool": tool_used,
                            "message": f"Video generated for Scene {scene}, Shot {shot}"
                        }, agent_name="video_task_monitor")

                        # Register video in asset_urls + metadata (following character/storyboard pattern)
                        if "asset_urls" not in state:
                            state["asset_urls"] = {}
                        if "generated_videos" not in state["asset_urls"]:
                            state["asset_urls"]["generated_videos"] = []
                        if "video_generation_metadata" not in state:
                            state["video_generation_metadata"] = {}

                        # Generate filename key (consistent with GCS naming)
                        video_filename = f"sc{scene:02d}_sh{shot:02d}_video_v{version}.mp4"

                        # Check if this video already exists (avoid duplicates during retries)
                        video_exists = any(
                            v.get("scene") == scene and v.get("shot") == shot and v.get("version") == version
                            for v in state["asset_urls"]["generated_videos"]
                        )

                        if not video_exists:
                            # Add lightweight entry to asset_urls
                            state["asset_urls"]["generated_videos"].append({
                                "url": public_url or video_url,
                                "scene": scene,
                                "shot": shot,
                                "version": version,
                                "filename": video_filename
                            })

                            # Add rich metadata to separate dict (like character/storyboard pattern)
                            state["video_generation_metadata"][video_filename] = {
                                "scene": scene,
                                "shot": shot,
                                "version": version,
                                "url": public_url or video_url,
                                "task_id": task_id,
                                "tool_used": tool_used,
                                "timestamp": datetime.now().isoformat(),
                                "status": "completed",
                                # Copy rich metadata from task for UI display
                                "params": task.get("params", {}),
                                "metadata": task.get("metadata", {})
                            }
                            print(f"  Registered video in asset_urls + metadata: Scene {scene}, Shot {shot}, Version {version}")

                    elif status == "failed":
                        error_msg = data.get("task_status_msg", "Unknown error")
                        logger.error(f"[VideoMonitor] Task {task_id_param} error: {error_msg}")
                        failed_task = {**task}
                        # Always update top-level fields
                        failed_task["status"] = "failed"
                        failed_task["success"] = False
                        failed_task["error"] = error_msg
                        failed_task["failed_at"] = datetime.now().isoformat()
                        failed_task["error_context"] = {
                            "source": "task_status",
                            "tool": tool_used,
                            "api_status": status,
                            "api_status_msg": data.get("task_status_msg"),
                            "api_data": data,
                            "stage": "task_monitoring"
                        }

                        # Also update params for completeness
                        if "params" in failed_task:
                            failed_task["params"] = {
                                **task["params"],
                                "status": "failed",
                                "error": error_msg,
                                "failed_at": datetime.now().isoformat()
                            }
                        failed.append(failed_task)
                        print(f"  Error: Scene {scene}, Shot {shot}: {error_msg}")

                        # Emit failure event
                        state = emit_event(state, "video_task_failed", {
                            "task_id": task_id,
                            "scene": scene,
                            "shot": shot,
                            "error": error_msg,
                            "error_context": failed_task["error_context"],
                            "tool": tool_used,
                            "message": f"Video generation error for Scene {scene}, Shot {shot}"
                        }, agent_name="video_task_monitor")

                    else:  # Still processing
                        # Emit status update event
                        state = emit_event(state, "video_task_updated", {
                            "task_id": task_id,
                            "scene": scene,
                            "shot": shot,
                            "status": status
                        }, agent_name="video_task_monitor")
                        still_pending.append(task)
                else:
                    # API error - treat as failed
                    error_msg = result.get("message", "API error")
                    api_code = result.get("code")
                    failed_task = {**task}
                    # Always update top-level fields
                    failed_task["status"] = "failed"
                    failed_task["success"] = False
                    failed_task["error"] = error_msg
                    failed_task["failed_at"] = datetime.now().isoformat()
                    failed_task["error_context"] = {
                        "source": "api_query",
                        "tool": tool_used,
                        "api_code": api_code,
                        "api_result": result,
                        "stage": "task_monitoring"
                    }

                    # Also update params for completeness
                    if "params" in failed_task:
                        failed_task["params"] = {
                            **task["params"],
                            "status": "failed",
                            "error": error_msg,
                            "failed_at": datetime.now().isoformat()
                        }
                    failed.append(failed_task)

                    # Enhanced error logging
                    logger.error(f"[VideoMonitor] API error for task {task_id_param} (Scene {scene}, Shot {shot}) | Error code: {api_code}, Message: {error_msg} | Full API response: {json.dumps(result, ensure_ascii=False)}")
                    print(f"  API Error for Scene {scene}, Shot {shot}:")
                    print(f"    Task ID: {task_id_param}")
                    print(f"    Code: {api_code}")
                    print(f"    Message: {error_msg}")
                    print(f"    Full response: {json.dumps(result, indent=2, ensure_ascii=False)}")

                    # Emit API error event
                    state = emit_event(state, "video_task_failed", {
                        "task_id": task_id,
                        "scene": scene,
                        "shot": shot,
                        "error": error_msg,
                        "error_context": failed_task["error_context"],
                        "tool": tool_used,
                        "message": f"API error for Scene {scene}, Shot {shot}"
                    }, agent_name="video_task_monitor")

            except Exception as e:
                # Exception - distinguish between timeout and other errors
                exception_type = type(e).__name__
                is_timeout = "Timeout" in exception_type or "timeout" in str(e).lower()

                failed_task = {**task}
                # Always update top-level fields
                failed_task["status"] = "failed"
                failed_task["success"] = False
                failed_task["error"] = str(e)
                failed_task["failed_at"] = datetime.now().isoformat()
                failed_task["error_context"] = {
                    "source": "exception",
                    "tool": tool_used,
                    "exception_type": exception_type,
                    "is_timeout": is_timeout,
                    "stage": "task_monitoring",
                    "description": "Timeout checking video status" if is_timeout else "Connection error during status check",
                    "task_id": task_id,
                    "scene": scene,
                    "shot": shot
                }

                # Also update params for completeness
                if "params" in failed_task:
                    failed_task["params"] = {
                        **task["params"],
                        "status": "failed",
                        "error": str(e),
                        "failed_at": datetime.now().isoformat()
                    }
                failed.append(failed_task)

                # Enhanced exception logging
                logger.error(f"[VideoMonitor] Monitoring error (exception) for task {task_id_param} (Scene {scene}, Shot {shot}) | Exception type: {exception_type}, Is timeout: {is_timeout} | Exception message: {str(e)}", exc_info=True)
                print(f"  Exception for Scene {scene}, Shot {shot}:")
                print(f"    Task ID: {task_id_param}")
                print(f"    Tool: {tool_used}")
                print(f"    Exception Type: {exception_type}")
                print(f"    Message: {str(e)}")
                if is_timeout:
                    print(f"    Note: This is a timeout error - API response took too long")

                # Emit exception error event
                state = emit_event(state, "video_task_failed", {
                    "task_id": task_id,
                    "scene": scene,
                    "shot": shot,
                    "error": str(e),
                    "error_context": failed_task["error_context"],
                    "tool": tool_used,
                    "message": f"Exception during monitoring for Scene {scene}, Shot {shot}"
                }, agent_name="video_task_monitor")

        pending = still_pending
        
        if pending:
            print(f"[Video Task Monitor] {len(pending)} still processing, {len(completed)} completed, {len(failed)} error")
            print(f"[Video Task Monitor] Waiting {check_interval}s before next check...")
            time.sleep(check_interval)
    
    # Check for timeout
    if pending:
        print(f"[Video Task Monitor] Timeout reached with {len(pending)} tasks still pending")
        for task in pending:
            # Always update top-level fields
            task["status"] = "timeout"
            task["success"] = False
            task["timeout_at"] = datetime.now().isoformat()

            # Also update params for completeness
            if "params" in task:
                task["params"]["status"] = "timeout"
                task["params"]["timeout_at"] = datetime.now().isoformat()
    
    # Calculate monitoring duration
    total_time = time.time() - start_time
    
    # Prepare monitor output
    monitor_output = {
        "status": "complete" if not pending else "timeout",
        "completed_count": len(completed),
        "failed_count": len(failed),
        "pending_count": len(pending),
        "total_tasks": len(tasks),
        "completed_tasks": completed,
        "failed_tasks": failed,
        "pending_tasks": pending,
        "monitoring_duration": f"{total_time:.1f}s",
        "timestamp": datetime.now().isoformat()
    }
    # Store monitor output
    state["video_monitor_output"] = monitor_output

    # Update task states in main list
    all_tasks = completed + failed + pending
    state["video_generation_tasks"] = all_tasks

    # Emit videos_generated event for completed videos (real-time streaming)
    if completed:
        state = emit_event(state, "videos_generated", {
            "completed_videos": [{"scene": t.get("scene"), "shot": t.get("shot"), "url": t.get("public_url")} for t in completed],
            "count": len(completed),
            "video_generation_tasks": all_tasks,
            "video_generation_metadata": state.get("video_generation_metadata", {})  # Include metadata dict for UI (like characters_generated pattern)
        }, agent_name="video_task_monitor")

    # Emit monitoring summary event
    state = emit_event(state, "video_monitoring_complete", {
        "total_tasks": len(tasks),
        "completed": len(completed),
        "failed": len(failed),
        "pending": len(pending),
        "monitoring_duration": total_time,
        "status": monitor_output["status"],
        "message": f"Video monitoring complete: {len(completed)}/{len(tasks)} succeeded"
    }, agent_name="video_task_monitor")

    # Auto-backup removed - no longer needed with GCS storage
    # Videos are already persisted to GCS with signed URLs
    
    # Store thinking process
    if "thinking_processes" not in state:
        state["thinking_processes"] = {}
    state["thinking_processes"]["video_task_monitor"] = f"Monitored {len(tasks)} video generation tasks for {total_time:.1f}s"
    
    # Record timing
    if "component_timings" not in state:
        state["component_timings"] = {}
    state["component_timings"]["video_task_monitor"] = total_time
    
    print(f"[Video Task Monitor] Monitoring complete:")
    print(f"  - Completed: {len(completed)}")
    print(f"  - Error: {len(failed)}")
    print(f"  - Pending: {len(pending)}")
    print(f"  - Total time: {total_time:.1f}s")

    # Emit agent ended event
    state = emit_event(state, "agent_ended", {"agent": "video_task_monitor"}, agent_name="video_task_monitor")

    return state