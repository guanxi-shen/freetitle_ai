"""
Tool-based video generation system using Google Veo

Key features:
- Official Google Veo via Vertex AI SDK
- Flexible frame inputs (single or dual frame)
- GCS-native paths (gs:// URIs)
- Standardized function signatures

Active Tools:
- google_veo_i2v: Google Veo 3.1 single/dual-frame (PRIMARY)
  * Auto-detects mode: single-frame when end_frame_path=None, dual-frame when provided
  * 8 seconds duration, multiple aspect ratios
  * Requires GCS URIs for images (gs://bucket/path)

Naming Convention: {provider}_{model}_{type}
- Provider: google
- Model: veo
- Type: i2v (image-to-video)
"""

import os
import json
import time
import requests
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import Google Veo client
from .client_veo_google import GoogleVeoGenerator

# Import utilities
from .util_image import image_to_base64, get_image_url_for_api, upload_image_to_url

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def upload_video_to_gcs_public(video_source, session_id: str, scene: int, shot: int, version: int, suffix: str = "", return_format: str = "gs") -> str:
    """
    Upload video to GCS and return path

    Args:
        video_source: Local path (str) or file-like object (BytesIO) containing video
        session_id: Session ID for organization
        scene: Scene number (use 0 for edited videos)
        shot: Shot number (use 0 for edited videos)
        version: Version number
        suffix: Optional suffix for edited videos (e.g., "_music")
        return_format: "gs" for gs:// path (default), "signed" for signed URL

    Returns:
        GCS path (gs://) or signed URL based on return_format
    """
    try:
        from google.cloud import storage
        from ...core.config import CREDENTIALS, SESSION_BUCKET_NAME, DEPLOYMENT_ENV

        # Initialize storage client
        storage_client = storage.Client(credentials=CREDENTIALS)
        bucket = storage_client.bucket(SESSION_BUCKET_NAME)

        # Create organized path with version
        environment = DEPLOYMENT_ENV.lower() if DEPLOYMENT_ENV else 'local'

        # Generate filename based on scene/shot
        if scene == 0 and shot == 0:
            # Edited video (no scene/shot prefix)
            video_filename = f"edited{suffix}_v{version}.mp4"
        else:
            # Shot video (with scene/shot prefix)
            video_filename = f"sc{scene:02d}_sh{shot:02d}_video_v{version}{suffix}.mp4"

        blob_name = f"{environment}/videos/{session_id}/{video_filename}"

        # Upload file (handle both file path and file-like object)
        blob = bucket.blob(blob_name)
        if isinstance(video_source, str):
            # File path provided
            logger.info(f"Uploading video from file to {blob_name}")
            blob.upload_from_filename(video_source)
        else:
            # File-like object (BytesIO) provided - zero-latency mode
            logger.info(f"Uploading video from memory buffer to {blob_name}")
            blob.upload_from_file(video_source, content_type='video/mp4')

        # Return format based on parameter
        if return_format == "signed":
            # Generate signed URL with 7 days expiration (maximum allowed)
            signed_url = blob.generate_signed_url(
                version="v4",
                expiration=timedelta(days=7),  # 7 days maximum expiration
                method="GET"
            )
            logger.info(f"Video uploaded to GCS with signed URL (7 days): {blob_name}")
            return signed_url
        else:
            # Return gs:// path (default)
            gs_path = f"gs://{SESSION_BUCKET_NAME}/{blob_name}"
            logger.info(f"Video uploaded to GCS with gs:// path: {gs_path}")
            return gs_path

    except Exception as e:
        logger.error(f"Error uploading video to GCS: {str(e)}")
        # Return empty string on failure so process can continue
        return ""


# ==============================================================================
# TOOL-BASED VIDEO GENERATION FUNCTIONS
# ==============================================================================

def google_veo_i2v(
    generation_prompt: str,
    start_frame_path: str,
    end_frame_path: Optional[str] = None,
    duration: int = 8,
    session_id: str = None,
    aspect_ratio: str = "horizontal",
    scene: Optional[int] = None,
    shot: Optional[int] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Google Veo image-to-video tool (single-frame or dual-frame)

    Primary tool for storyboard-to-video generation using official Google API.
    Automatically detects mode based on end_frame_path:
    - Single-frame: When end_frame_path is None
    - Dual-frame: When end_frame_path is provided (start/end frame interpolation)

    Args:
        generation_prompt: Video generation prompt with motion description
        start_frame_path: GCS path (gs://) to starting frame
        end_frame_path: Optional GCS path to ending frame (triggers dual-frame mode)
        duration: Video duration in seconds (default: 8)
        session_id: Session ID for tracking and GCS output
        aspect_ratio: Aspect ratio preset (horizontal or vertical)
        scene: Scene number for tracking
        shot: Shot number for tracking
        **kwargs: Additional tool-specific parameters

    Returns:
        Dict with success, task_id, tool_used, and optional error
    """
    # Detect mode based on end_frame_path
    is_dual_frame = end_frame_path is not None
    mode_str = "dual-frame" if is_dual_frame else "single-frame"
    logger.info(f"[google_veo_i2v] Scene {scene}, Shot {shot}, mode={mode_str}, aspect_ratio={aspect_ratio}")

    try:
        from ...core.config import BUCKET_NAME

        # Initialize Google Veo generator
        generator = GoogleVeoGenerator()

        # Map aspect ratio to Veo format
        veo_ratios = {
            "vertical": "9:16",
            "horizontal": "16:9"
        }
        ratio = veo_ratios.get(aspect_ratio, "16:9")

        # Convert paths to GCS URIs if needed
        def to_gs_uri(path: str) -> str:
            """Ensure path is in gs:// format"""
            if path.startswith("gs://"):
                return path
            elif path.startswith("https://storage.googleapis.com/"):
                # Convert HTTPS to gs://
                parts = path.replace("https://storage.googleapis.com/", "").split("/", 1)
                return f"gs://{parts[0]}/{parts[1]}" if len(parts) > 1 else path
            else:
                # Assume it's a local path that needs uploading
                logger.warning(f"[google_veo_i2v] Path is not GCS format: {path}")
                return path

        start_image_uri = to_gs_uri(start_frame_path)
        end_image_uri = to_gs_uri(end_frame_path) if end_frame_path else None

        logger.info(f"[google_veo_i2v] Start frame: {start_image_uri}")
        if end_image_uri:
            logger.info(f"[google_veo_i2v] End frame (dual-frame): {end_image_uri}")

        # Generate output GCS path for video
        output_gcs_uri = f"gs://{BUCKET_NAME}/videos/{session_id or 'default'}/"

        # Submit video generation
        logger.info(f"[google_veo_i2v] Submitting to Google Veo: mode={mode_str}, ratio={ratio}, duration={duration}s")
        response = generator.generate_video(
            prompt=generation_prompt,
            image_uri=start_image_uri,
            last_frame_uri=end_image_uri,
            aspect_ratio=ratio,
            duration_seconds=duration,
            output_gcs_uri=output_gcs_uri
        )

        if response.get("code") == 0:
            task_id = response.get("data", {}).get("task_id")
            logger.info(f"[google_veo_i2v] Task submitted successfully: {task_id}")
            return {
                "success": True,
                "task_id": task_id,
                "tool_used": "google_veo_i2v"
            }
        else:
            # API error during submission
            error_msg = response.get("message", "Unknown error")
            logger.error(f"[google_veo_i2v] API error (Scene {scene}, Shot {shot}): {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "error_context": {
                    "tool": "google_veo_i2v",
                    "source": "google_veo_api",
                    "api_response": response,
                    "request_params": {
                        "mode": mode_str,
                        "ratio": ratio,
                        "duration": duration,
                        "scene": scene,
                        "shot": shot
                    }
                }
            }

    except Exception as e:
        exception_type = type(e).__name__
        logger.error(f"[google_veo_i2v] Exception: {exception_type} - {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "error_context": {
                "tool": "google_veo_i2v",
                "source": "exception",
                "exception_type": exception_type,
                "scene": scene,
                "shot": shot
            }
        }



# ==============================================================================
# TOOL REGISTRY
# ==============================================================================

# Active tools
VIDEO_TOOLS = {
    "google_veo_i2v": google_veo_i2v,  # Primary: Google Veo single/dual-frame
}


def call_video_tool(tool_name: str, **kwargs) -> Dict[str, Any]:
    """
    Simple dispatcher for video generation tools.

    Args:
        tool_name: Name of the video generation tool
        **kwargs: Tool-specific parameters

    Returns:
        Tool execution result
    """
    if tool_name not in VIDEO_TOOLS:
        return {
            "success": False,
            "error": f"Unknown video tool: {tool_name}"
        }

    return VIDEO_TOOLS[tool_name](**kwargs)


# ==============================================================================
# PROVIDER REGISTRY FOR AUTOMATIC ROUTING
# ==============================================================================

# Active provider mappings: tool_name -> (ClientClass, task_id_type)
PROVIDER_MAP = {
    "google_veo_i2v": (GoogleVeoGenerator, "string"),  # Operation names are strings
}

# Cached generator instances (lazy initialization)
_generator_cache = {}


def get_generator_for_tool(tool_name: str):
    """
    Get generator instance for a tool (with caching).

    Args:
        tool_name: Name of the video generation tool

    Returns:
        Dict with generator instance and task_id_type
    """
    if tool_name not in _generator_cache:
        if tool_name not in PROVIDER_MAP:
            logger.warning(f"[get_generator_for_tool] Unknown tool '{tool_name}', falling back to google_veo_i2v")
            tool_name = "google_veo_i2v"

        logger.info(f"[get_generator_for_tool] Initializing generator for '{tool_name}'")
        generator_class, task_id_type = PROVIDER_MAP[tool_name]
        _generator_cache[tool_name] = {
            "generator": generator_class(),
            "task_id_type": task_id_type
        }

    return _generator_cache[tool_name]


# ==============================================================================
# CONVENIENCE FUNCTION
# ==============================================================================

def generate_video_from_frames(
    start_frame_path: str,
    end_frame_path: Optional[str] = None,
    prompt: str = "Smooth transition between frames",
    output_dir: str = "./output",
    duration: int = 8,
    **kwargs
) -> Optional[str]:
    """
    End-to-end video generation from image files (legacy function).
    Redirects to google_veo_i2v.

    Args:
        start_frame_path: Path to start frame image (GCS gs:// path)
        end_frame_path: Path to end frame image (optional, GCS gs:// path)
        prompt: Text description for video generation
        output_dir: Directory to save output video (ignored - uses GCS)
        duration: Video duration in seconds (default: 8)
        **kwargs: Additional parameters (ignored)

    Returns:
        GCS path to generated video or None if failed
    """
    try:
        # Use Google Veo generator
        generator = GoogleVeoGenerator()

        # Submit video generation request
        logger.info("Submitting video generation request to Google Veo...")
        response = generator.generate_video(
            prompt=prompt,
            image_uri=start_frame_path,
            last_frame_uri=end_frame_path,
            duration_seconds=duration
        )

        # Check response
        if response.get("code") != 0:
            error_msg = response.get("message", "Unknown error")
            raise Exception(f"API error: {error_msg}")

        task_id = response.get("data", {}).get("task_id")
        if not task_id:
            raise ValueError("No task_id in response")

        # Wait for completion
        logger.info(f"Waiting for task {task_id} to complete...")
        final_result = generator.wait_for_completion(task_id)

        # Extract video URL
        data = final_result.get("data", {})
        videos = data.get("task_result", {}).get("videos", [])
        if not videos:
            raise ValueError("No video in completed task")
        
        video_url = videos[0].get("url")
        if not video_url:
            raise ValueError("No video URL in result")

        # Veo outputs directly to GCS, return the GCS path
        logger.info(f"Video generated successfully: {video_url}")
        return video_url
        
    except Exception as e:
        logger.error(f"Video generation error: {str(e)}")
        return None


# Orchestration functions for video agent with automatic function calling

def create_function_wrappers(state: Dict, storyboards: List[Dict], function_results_collector: List, session_id: str = "", selected_tools: List[str] = None, thinking: str = ""):
    """Create function wrappers with state binding for automatic function calling

    Args:
        state: Current RAGState
        storyboards: List of storyboard data
        function_results_collector: List to collect function results
        session_id: Session ID for GCS operations
        selected_tools: List of user-selected video generation tools
        thinking: Agent thinking process to include in task metadata

    Returns:
        Tuple of wrapped functions (get_available_shots, process_video_shots)
    """

    # Default to all tools if not specified
    if selected_tools is None:
        selected_tools = ["google_veo_i2v"]
    
    def get_available_shots() -> Dict[str, Any]:
        """Get current video generation status from tasks and files
        
        Returns:
            Dictionary with current task status and existing video files
        """
        import re
        
        # Get fresh task information from state
        current_tasks = state.get("video_generation_tasks", [])
        
        # List existing videos from state
        video_files = []
        for task in current_tasks:
            if task.get("status") == "completed" and task.get("path"):
                video_url = task.get("path")
                scene = task.get("scene", 0)
                shot = task.get("shot", 0)
                version = task.get("version", 1)

                filename = f"sc{scene:02d}_sh{shot:02d}_video_v{version}.mp4"

                video_files.append({
                    "filename": filename,
                    "path": video_url,
                    "scene": scene,
                    "shot": shot,
                    "version": f"_v{version}",
                    "size_mb": 0
                })

        # Get prompts status
        existing_prompts = state.get("generated_video_prompts", {})
        prompt_list = existing_prompts.get("video_prompts", [])

        # Build comprehensive status - just pass data directly
        result = {
            "current_tasks": current_tasks,  # Already has all task fields
            "existing_videos": video_files,
            "video_prompts_available": prompt_list,  # Already has scene_number, shot_number, etc
            "summary": {
                "total_tasks": len(current_tasks),
                "total_videos_on_disk": len(video_files),
                "total_prompts": len(prompt_list)
            }
        }

        # Count tasks by status
        status_counts = {}
        for task in current_tasks:
            status = task.get("status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
        result["summary"]["tasks_by_status"] = status_counts
        
        # Capture result with full metadata before returning
        function_results_collector.append({
            "function": "get_available_shots",
            "params": {},  # No parameters for this function
            "result": result.copy(),
            "timestamp": datetime.now().isoformat()
        })

        return result

    def process_video_shots(shots_list: List[Dict[str, int]]) -> Dict[str, Any]:
        """
        Unified function for video generation (replaces generate_shots_prompts + create_video_tasks)

        Processes each shot with intelligent sub-agent that:
        - Reads storyboard + script + assets
        - Decides tool based on context
        - Selects reference images if needed
        - Generates prompt
        - Submits task

        Args:
            shots_list: List of dicts with 'scene' and 'shot' keys
                       Example: [{"scene": 1, "shot": 1}, {"scene": 2, "shot": 1}]

        Returns:
            Dictionary with task submission results
        """
        from .agent_video_shot_processor import process_single_shot

        logger.info(f"[process_video_shots] Processing {len(shots_list)} shots with selected tools: {selected_tools}")
        print(f"[Video Tools] Processing {len(shots_list)} shots with intelligent sub-agents...")

        # Validate shots_list structure
        validated_shots = []
        for idx, shot in enumerate(shots_list):
            if not isinstance(shot, dict):
                logger.error(f"[process_video_shots] Shot {idx} is not a dict: {type(shot)} - {shot}")
                continue

            if "scene" not in shot or "shot" not in shot:
                logger.error(f"[process_video_shots] Shot {idx} missing required keys: {shot}")
                continue

            try:
                validated_shots.append({
                    "scene": int(shot["scene"]),
                    "shot": int(shot["shot"])
                })
            except (ValueError, TypeError) as e:
                logger.error(f"[process_video_shots] Shot {idx} has invalid scene/shot values: {shot} - {e}")
                continue

        if not validated_shots:
            return {
                "success": False,
                "error": "No valid shots in shots_list",
                "prompts_generated": 0,
                "tasks_submitted": 0,
                "submission_errors": [{"error": "Invalid shots_list structure"}]
            }


        # Build storyboard lookup
        shot_to_storyboard = {
            (sb["scene_number"], sb["shot_number"]): sb
            for sb in storyboards
        }

        tasks = []
        prompts = []
        errors = []

        # Process shots in parallel
        with ThreadPoolExecutor(max_workers=12) as executor:
            futures = {
                executor.submit(
                    process_single_shot,
                    shot["scene"],
                    shot["shot"],
                    state,
                    selected_tools,
                    shot_to_storyboard.get((shot["scene"], shot["shot"]), {}),
                    function_results_collector
                ): shot
                for shot in validated_shots
            }

            for future in as_completed(futures):
                shot = futures[future]
                try:
                    result = future.result()

                    if result["success"]:
                        # Store task info with thinking metadata for UI display
                        task_data = {
                            "task_id": result["task_id"],
                            "scene": shot["scene"],
                            "shot": shot["shot"],
                            "status": "submitted",
                            "tool_used": result["tool_used"],
                            "params": {
                                **result["parameters"],
                                "generation_prompt": result.get("generation_prompt", {})
                            },
                            "metadata": {
                                # "generation_prompt": result.get("generation_prompt", {}),  # Duplicate of params.generation_prompt
                                "reasoning": result.get("reasoning", ""),
                                "thinking": result.get("thinking", ""),
                                # "tool_selected": result["tool_used"],  # Duplicate of tool_used
                                # "parameters": result["parameters"]  # Duplicate of entire params object
                            },
                            "timestamp": datetime.now().isoformat()
                        }
                        tasks.append(task_data)

                        # Store prompt info (for backward compatibility)
                        prompt_data = {
                            "scene_number": shot["scene"],
                            "shot_number": shot["shot"],
                            "generation_prompt": result.get("generation_prompt", {}),
                            "start_frame_path": result["parameters"].get("start_frame_path"),
                            "end_frame_path": result["parameters"].get("end_frame_path")
                        }
                        prompts.append(prompt_data)

                        print(f"  [OK] Scene {shot['scene']}, Shot {shot['shot']} → {result['task_id']} (using {result['tool_used']})")

                        # Emit real-time event with metadata for UI
                        from ..base import emit_event
                        emit_event(state, "video_task_submitted", {
                            "task_id": result["task_id"],
                            "scene": shot["scene"],
                            "shot": shot["shot"],
                            "status": "submitted",
                            "tool_used": result["tool_used"],
                            "params": task_data["params"],
                            "metadata": task_data["metadata"],  # Include metadata for UI display
                            "timestamp": task_data["timestamp"]
                        }, agent_name="video_agent")

                    else:
                        errors.append({
                            "shot": shot,
                            "error": result.get("error", "Unknown error"),
                            "error_context": result.get("error_context", {})
                        })
                        print(f"  [FAIL] Scene {shot['scene']}, Shot {shot['shot']}: {result.get('error', 'Unknown')}")

                except Exception as e:
                    errors.append({
                        "shot": shot,
                        "error": str(e),
                        "exception_type": type(e).__name__
                    })
                    print(f"  [ERROR] Scene {shot['scene']}, Shot {shot['shot']}: {e}")

        # Store tasks in state
        if tasks:
            if "video_generation_tasks" not in state:
                state["video_generation_tasks"] = []
            state["video_generation_tasks"].extend(tasks)

        # Store prompts in state (for backward compatibility)
        if prompts:
            merge_prompts_into_state(state, prompts)

        result = {
            "tasks_submitted": len(tasks),
            "task_ids": [t["task_id"] for t in tasks],
            "errors": errors if errors else None
        }

        logger.info(f"[process_video_shots] Completed: {len(tasks)} tasks submitted, {len(errors)} errors")

        # Capture result with full metadata
        function_results_collector.append({
            "function": "process_video_shots",
            "params": {"shots_list": shots_list},
            "result": result.copy(),
            "timestamp": datetime.now().isoformat()
        })

        return result

    # Return the wrapped functions (new unified function)
    return get_available_shots, process_video_shots


def process_single_shot_prompt(shot: Dict, context: str, user_query: str = "") -> Dict[str, Any]:
    """Generate prompt for a single shot (worker function for parallel processing)

    Args:
        shot: Shot data with frames
        context: Context content for the shot
        user_query: User's original query

    Returns:
        Generated prompt data
    """
    from ...core.llm import get_llm
    from ...prompts import VIDEO_GENERATION_PROMPT_TEMPLATE
    from ..base import clean_json_response

    scene_num = shot["scene_number"]
    shot_num = shot["shot_number"]
    frames = shot.get("frames", [])

    # Find frame paths (flexible - handle both single and dual frame)
    start_frame_path = None
    end_frame_path = None

    for frame in frames:
        if frame.get("frame_number") == 1:
            start_frame_path = frame.get("path")
        elif frame.get("frame_number") == 2:
            end_frame_path = frame.get("path")

    # Only require start_frame (end_frame is optional now)
    if not start_frame_path:
        raise ValueError(f"Missing start frame for Scene {scene_num}, Shot {shot_num}")
    
    # Load and encode frame images (flexible for single or dual frame)
    start_image = load_frame_image_for_llm(start_frame_path)
    end_image = load_frame_image_for_llm(end_frame_path) if end_frame_path else None

    # Configure LLM for this shot
    # Note: LLM will be initialized after loading prompt template

    # Build context block with all dynamic content
    frame_context_info = f"Scene {scene_num}, Shot {shot_num}.\n"
    if end_image:
        frame_context_info += "DUAL-FRAME MODE: Describe transformation from frame A to frame B."
    else:
        frame_context_info += "SINGLE-FRAME MODE (Veo3): Describe motion beginning from the visible start frame only."

    context_content = f"""{frame_context_info}

### Script and Shot Context:
{context}

### Current Request:
User Query: {user_query}

### Important:
Review the task instructions and guidelines provided at the beginning of this prompt before proceeding. Ensure your response follows all specified rules, formats, and requirements."""

    # Get static template
    shot_prompt = VIDEO_GENERATION_PROMPT_TEMPLATE["template"]

    # Initialize LLM with prompt template as system_instruction
    shot_llm = get_llm(
        gemini_configs={
            'max_output_tokens': 5000,
            'temperature': 1.0,
            'top_p': 0.95,
        },
        enable_images=True,
        system_instruction=shot_prompt  # Use full prompt template as system instruction
    )

    # Build images list based on what's available
    images = []
    if start_image:
        images.append(start_image)
    if end_image:
        images.append(end_image)

    # Generate prompt with images
    response = shot_llm.invoke(
        context_content,  # Only dynamic context, prompt is in system_instruction
        add_context=False,  # Don't concatenate, prompt already in system_instruction
        images=images if images else None,
        state=state
    )
    
    # Parse response with debug logging
    print(f"\n[DEBUG Scene {scene_num}, Shot {shot_num}] Raw response type: {type(response)}")
    
    if isinstance(response, str):
        print(f"[DEBUG] Response is string, first 500 chars: {response[:500]}")
        parsed_response = json.loads(response)
    else:
        parsed_response = response
        print(f"[DEBUG] Response is object, keys: {parsed_response.keys() if hasattr(parsed_response, 'keys') else 'N/A'}")
    
    print(f"[DEBUG] Parsed response 'content' length: {len(parsed_response.get('content', []))}")
    
    # Log content items
    for i, content in enumerate(parsed_response.get('content', [])):
        print(f"[DEBUG] Content[{i}] keys: {content.keys() if isinstance(content, dict) else type(content)}")
        if isinstance(content, dict) and 'text' in content:
            print(f"[DEBUG] Content[{i}] text (first 200 chars): {content['text'][:200]}")
    
    response_text = parsed_response['content'][1].get('text', '')
    print(f"\n[DEBUG] Attempting to parse as JSON, text length: {len(response_text)}")
    print(f"[DEBUG] Text to parse (first 1000 chars):\n{response_text[:1000]}")
    print(f"[DEBUG] Text to parse (last 500 chars):\n{response_text[-500:]}")
    
    try:
        # Clean JSON response from markdown code blocks
        response_text = clean_json_response(response_text)
        video_prompt_data = json.loads(response_text)
    except json.JSONDecodeError as e:
        print(f"\n[ERROR] JSON parsing error for Scene {scene_num}, Shot {shot_num}")
        print(f"[ERROR] Parse error: {str(e)}")
        print(f"[ERROR] Error position: Line {e.lineno}, Column {e.colno}, Char {e.pos}")

        # Show the problematic area
        if e.pos and e.pos < len(response_text):
            start = max(0, e.pos - 50)
            end = min(len(response_text), e.pos + 50)
            print(f"[ERROR] Text around error position ({start}-{end}):")
            print(f"[ERROR] ...{response_text[start:end]}...")

        # Re-raise to maintain original behavior
        raise

    # Validate new nested structure
    if "generation_prompt" not in video_prompt_data:
        raise ValueError(f"Missing 'generation_prompt' field in response for Scene {scene_num}, Shot {shot_num}")

    gen_prompt = video_prompt_data["generation_prompt"]
    if not isinstance(gen_prompt, dict):
        raise ValueError(f"'generation_prompt' must be a dict, got {type(gen_prompt)}")

    # Validate and fill missing fields with empty strings
    required_fields = ["summary", "camera", "motion", "style", "dialogue", "sound", "note", "negative"]
    for field in required_fields:
        if field not in gen_prompt:
            print(f"[WARNING] Missing field '{field}' in generation_prompt for Scene {scene_num}, Shot {shot_num}, setting to empty string")
            gen_prompt[field] = ""

    # Add frame paths at top level (metadata for system tracking)
    video_prompt_data["start_frame_path"] = start_frame_path
    video_prompt_data["end_frame_path"] = end_frame_path

    return video_prompt_data


def load_frame_image_for_llm(image_ref: str, model_type: str = "gemini") -> Optional[Dict]:
    """Load and prepare image for LLM input
    Handles both GCS URLs and local file paths

    Args:
        image_ref: GCS URL or local file path
        model_type: "gemini" (file_uri, zero latency) or others (base64)

    Returns:
        Image dict for LLM or None
    """
    try:
        # GCS URL - use model-aware preparation
        if image_ref.startswith(("gs://", "https://storage.googleapis.com")):
            from ...storage.gcs_utils import prepare_image_for_llm
            return prepare_image_for_llm(image_ref, model_type)

        # Local file path - always base64
        elif os.path.exists(image_ref):
            with open(image_ref, 'rb') as img_file:
                image_data = img_file.read()
                encoded_image = base64.b64encode(image_data).decode('utf-8')

            ext = Path(image_ref).suffix.lower()
            media_type = {'.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.webp': 'image/webp'}.get(ext, 'image/png')

            return {'data': encoded_image, 'media_type': media_type}

        else:
            logger.warning(f"Image not found: {image_ref}")
            return None

    except Exception as e:
        logger.error(f"Error loading image {image_ref}: {str(e)}")
        return None


def merge_prompts_into_state(state: Dict, new_prompts: List[Dict]) -> None:
    """Merge new prompts into existing state
    
    Args:
        state: Current RAGState
        new_prompts: List of new prompt data
    """
    existing = state.get("generated_video_prompts", {})
    
    if not existing:
        # First time - create structure
        state["generated_video_prompts"] = {
            "video_prompts": new_prompts,
            "metadata": {
                "total_shots": len(new_prompts),
                "timestamp": datetime.now().isoformat()
            }
        }
    else:
        # Merge with existing
        existing_list = existing.get("video_prompts", [])
        
        # Create lookup
        prompt_map = {
            (p["scene_number"], p["shot_number"]): p 
            for p in existing_list
        }
        
        # Update with new prompts
        for prompt in new_prompts:
            key = (prompt["scene_number"], prompt["shot_number"])
            prompt_map[key] = prompt
        
        # Convert back to sorted list
        merged_list = sorted(
            prompt_map.values(),
            key=lambda x: (x["scene_number"], x["shot_number"])
        )
        
        existing["video_prompts"] = merged_list
        existing["metadata"]["total_shots"] = len(merged_list)
        existing["metadata"]["timestamp"] = datetime.now().isoformat()
        existing["metadata"]["last_updated"] = f"Updated {len(new_prompts)} prompts"
        
        state["generated_video_prompts"] = existing
    
    print(f"[Video Tools] Merged {len(new_prompts)} prompts into state")
