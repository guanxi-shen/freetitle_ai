"""
Storyboard validation tools for visual quality checking
Validates frames for AI glitches, split screens, text errors, and reference matching
"""

import json
import base64
from datetime import datetime
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from ...core.state import RAGState
from ...core.llm import get_llm
from ..base import clean_json_response


# Validation limit constant
MAX_VALIDATIONS_PER_TURN = 1


def reset_validation_count(state: RAGState):
    """Reset validation counter at start of each storyboard agent turn"""
    state["_validation_count"] = 0
    print("[Validation] Counter reset to 0")


def validate_storyboards_wrapper(shots_json: str, state: RAGState) -> Dict[str, Any]:
    """
    Validate storyboard frames for specified shots.

    Args:
        shots_json: JSON array of shot identifiers
            Format: '[{"scene": 1, "shot": 1}, {"scene": 1, "shot": 2}, ...]'
        state: Current RAGState

    Returns:
        {
            "validated_shots": N,
            "total_frames_checked": N,
            "frames_good": N,
            "frames_with_issues": N,
            "results": [shot_validation_results]
        }
    """
    # Get session-isolated validation count
    validation_count = state.get("_validation_count", 0)

    # Check validation limit
    if validation_count >= MAX_VALIDATIONS_PER_TURN:
        print(f"[Validation Wrapper] Validation limit reached ({validation_count}/{MAX_VALIDATIONS_PER_TURN})")
        return {
            "validated_shots": 0,
            "total_frames_checked": 0,
            "frames_good": 0,
            "frames_with_issues": 0,
            "error": f"Only {MAX_VALIDATIONS_PER_TURN} validation is allowed per turn",
            "results": []
        }

    # Increment counter and proceed
    state["_validation_count"] = validation_count + 1
    print(f"[Validation Wrapper] Validation count: {state['_validation_count']}/{MAX_VALIDATIONS_PER_TURN}")

    print(f"\n[Validation Wrapper] Called with shots_json ({len(shots_json)} chars)")

    try:
        # Parse shots JSON
        cleaned_json = clean_json_response(shots_json)
        shots = json.loads(cleaned_json)
        print(f"[Validation Wrapper] Parsed {len(shots)} shot(s) to validate")

        if not isinstance(shots, list):
            return {
                "validated_shots": 0,
                "total_frames_checked": 0,
                "frames_good": 0,
                "frames_with_issues": 0,
                "error": "shots_json must be a JSON array",
                "results": []
            }

        print(f"[Storyboard Validation] Validating {len(shots)} shot(s)")

        # Spawn parallel validation sub-agents
        results = []
        with ThreadPoolExecutor(max_workers=12) as executor:
            futures = {}
            for shot_info in shots:
                scene = shot_info.get("scene")
                shot = shot_info.get("shot")
                if scene and shot:
                    future = executor.submit(validate_single_shot, scene, shot, state)
                    futures[future] = (scene, shot)

            # Collect results
            for future in as_completed(futures):
                scene, shot = futures[future]
                try:
                    result = future.result(timeout=120)
                    results.append(result)
                    print(f"[Storyboard Validation] Shot {scene}.{shot} validation complete")
                except Exception as e:
                    print(f"[Storyboard Validation] Shot {scene}.{shot} validation error: {e}")
                    results.append({
                        "scene": scene,
                        "shot": shot,
                        "frames": [],
                        "error": str(e)
                    })

        # Aggregate statistics
        total_frames = sum(len(r.get("frames", [])) for r in results)
        frames_good = sum(
            1 for r in results
            for f in r.get("frames", [])
            if f.get("is_good", False)
        )
        frames_with_issues = total_frames - frames_good

        # Build return structure
        validation_results = {
            "validated_shots": len(results),
            "total_frames_checked": total_frames,
            "frames_good": frames_good,
            "frames_with_issues": frames_with_issues,
            "results": results,
            "timestamp": datetime.now().isoformat()
        }

        # Store in state
        state["storyboard_validation_results"] = validation_results

        print(f"[Storyboard Validation] Summary: {total_frames} frames, {frames_good} good, {frames_with_issues} issues")
        print(f"[Storyboard Validation] Full results: {json.dumps(validation_results, indent=2)}")
        print(f"[Storyboard Validation] ========== VALIDATION COMPLETE ==========\n")

        return validation_results

    except Exception as e:
        print(f"[Storyboard Validation] ERROR: {e}")
        import traceback
        print(f"[Storyboard Validation] Traceback:\n{traceback.format_exc()}")
        return {
            "validated_shots": 0,
            "total_frames_checked": 0,
            "frames_good": 0,
            "frames_with_issues": 0,
            "error": str(e),
            "results": []
        }


def validate_single_shot(scene: int, shot: int, state: RAGState) -> Dict[str, Any]:
    """
    Validate all frames for a single shot using Gemini 2.5 Pro with visual context.

    Args:
        scene: Scene number
        shot: Shot number
        state: Current RAGState

    Returns:
        {
            "scene": N,
            "shot": N,
            "frames": [{"filename": "...", "is_good": bool, "issues": "string"}]
        }
    """
    print(f"\n[Validation Sub-Agent] ========== SHOT {scene}.{shot} ==========")

    # Extract shot description, start_frame, and end_frame from script for context
    shot_description = ""
    shot_start_frame = ""
    shot_end_frame = ""
    try:
        scripts = state.get("generated_scripts", {})
        if scripts and isinstance(scripts, dict):
            scenes = scripts.get("script_details", {}).get("scenes", [])
            for scene_data in scenes:
                if scene_data.get("scene_number") == scene:
                    shots = scene_data.get("shots", [])
                    for shot_data in shots:
                        if shot_data.get("shot_number") == shot:
                            shot_description = shot_data.get("description", "")
                            shot_start_frame = shot_data.get("start_frame", "")
                            shot_end_frame = shot_data.get("end_frame", "")
                            break
                    break
    except Exception as e:
        print(f"[Validation Sub-Agent] Could not extract shot details from script: {e}")

    # Fetch frames for this shot from metadata
    metadata = state.get("storyboard_frame_metadata", {})
    shot_frames = {
        filename: meta
        for filename, meta in metadata.items()
        if meta.get("scene") == scene and meta.get("shot") == shot and meta.get("success")
    }

    if not shot_frames:
        print(f"[Validation Sub-Agent] No frames found for shot {scene}.{shot}")
        return {
            "scene": scene,
            "shot": shot,
            "frames": [],
            "error": f"No frames found for shot {scene}.{shot}"
        }

    print(f"[Validation Sub-Agent] Found {len(shot_frames)} frame(s) to validate")

    # Extract generation prompt from first frame (all frames in shot use same prompt)
    generation_prompt = ""
    first_frame_meta = next(iter(shot_frames.values()), {})
    if first_frame_meta:
        params = first_frame_meta.get("params", {})
        generation_prompt = params.get("generation_prompt", "")
        if generation_prompt:
            print(f"[Validation Sub-Agent] Found generation prompt: {generation_prompt[:100]}...")

    # Extract unique references from all frames
    all_references = set()
    for meta in shot_frames.values():
        params = meta.get("params", {})

        # Reference images
        refs = params.get("reference_images", [])
        if refs:
            all_references.update(refs)

    print(f"[Validation Sub-Agent] Found {len(all_references)} reference image(s)")

    # Prepare frame images (always Gemini - use file_uri for zero latency)
    from ...storage.gcs_utils import prepare_image_for_llm

    frame_images = []
    frame_filenames = []
    for filename, meta in shot_frames.items():
        try:
            path = meta.get("path")
            if path:
                image_data = prepare_image_for_llm(path, "gemini")
                if image_data:
                    frame_images.append(image_data)
                    frame_filenames.append(filename)
                    print(f"[Validation Sub-Agent] Prepared frame: {filename} (file_uri)")
        except Exception as e:
            print(f"[Validation Sub-Agent] Error preparing frame {filename}: {e}")

    # Prepare reference images in sorted order
    reference_images = []
    reference_paths_loaded = []
    for ref_path in sorted(all_references):
        try:
            image_data = prepare_image_for_llm(ref_path, "gemini")
            if image_data:
                reference_images.append(image_data)
                reference_paths_loaded.append(ref_path)
                ref_name = ref_path.split('/')[-1]
                print(f"[Validation Sub-Agent] Prepared reference: {ref_name} (file_uri)")
        except Exception as e:
            print(f"[Validation Sub-Agent] Error preparing reference {ref_path}: {e}")

    # Prepare all images for Gemini (frames first, then references)
    all_images = frame_images + reference_images

    # Build dynamic image context - ONLY list images with minimal tags
    image_context_lines = [
        "=== IMAGE LIST ===",
        f"Total images: {len(frame_images) + len(reference_images)}",
        "",
        "STORYBOARD FRAMES:"
    ]

    # List storyboard frames
    for idx, filename in enumerate(frame_filenames, start=1):
        if len(frame_filenames) == 1:
            image_context_lines.append(f"- Image {idx}: {filename} (storyboard)")
        else:
            frame_type = "start" if idx == 1 else "end"
            image_context_lines.append(f"- Image {idx}: {filename} (storyboard-{frame_type})")

    # List reference images with type tags only
    if reference_images:
        image_context_lines.append("")
        image_context_lines.append("REFERENCE IMAGES:")
        ref_start_idx = len(frame_images) + 1

        for idx, ref_path in enumerate(reference_paths_loaded, start=ref_start_idx):
            ref_name = ref_path.split('/')[-1]

            # Detect type from path/filename
            if 'turnaround' in ref_name.lower():
                image_context_lines.append(f"- Image {idx}: {ref_name} (turnaround)")
            elif 'variation' in ref_name.lower() or '/characters/' in ref_path:
                image_context_lines.append(f"- Image {idx}: {ref_name} (character-variation)")
            elif '/supplementary/' in ref_path:
                image_context_lines.append(f"- Image {idx}: {ref_name} (supplementary)")
            elif '/references/' in ref_path:
                image_context_lines.append(f"- Image {idx}: {ref_name} (user-upload)")
            else:
                image_context_lines.append(f"- Image {idx}: {ref_name} (reference)")
    else:
        image_context_lines.append("")
        image_context_lines.append("REFERENCE IMAGES: None")

    image_context = "\n".join(image_context_lines)

    # Build context block with all validation data
    context_content = f"""### Shot Being Validated:
Scene {scene}, Shot {shot}

### Frames to Check:
{len(frame_images)} frame(s) for this shot:
{chr(10).join([f"- {fn}" for fn in frame_filenames])}

### Reference Images:
{len(reference_images)} reference(s) used during generation

### Script Context (for understanding intent):
Description: {shot_description}
Start Frame: {shot_start_frame}
End Frame: {shot_end_frame}

### Generation Context (actual prompt used to create this frame):
{generation_prompt}

This shows what was INTENDED for this frame. Use this to understand creative choices.
Example: If generation prompt says "character in party dress", outfit differs from turnaround intentionally.

### Image Context:
{image_context}

### Important:
Review the task instructions and guidelines provided at the beginning of this prompt before proceeding. Ensure your response follows all specified rules, formats, and requirements."""

    # Get static template
    from ...prompts.storyboard import STORYBOARD_VALIDATION_SUB_AGENT_PROMPT
    prompt = STORYBOARD_VALIDATION_SUB_AGENT_PROMPT

    # Get or create cache
    from src.core.cache_registry import get_or_create_cache
    cached_content = get_or_create_cache(
        agent_name="storyboard_validation",
        system_instruction=prompt,
        state=state
    )

    # Initialize LLM with prompt template as system_instruction
    validation_llm = get_llm(
        gemini_configs={
            'max_output_tokens': 5000,
            'temperature': 1.0,
            'top_p': 0.95
        },
        enable_images=True,
        system_instruction=prompt,  # Use full prompt template as system instruction
        cached_content_name=cached_content  # Use cache if available
    )

    try:
        # Invoke validation with images (with retry on empty response)
        print(f"[Validation Sub-Agent] Invoking Gemini with {len(all_images)} images")
        response_text = None

        for attempt in range(2):  # Try twice
            response = validation_llm.invoke(
                context_content,  # Only dynamic context, prompt is in system_instruction
                add_context=False,  # Don't concatenate, prompt already in system_instruction
                images=all_images if all_images else None,
                state=state
                # response_schema="storyboard_validation"  # Commented out - structured output may not work with images
            )

            # Parse response (aligned with script agent pattern)
            if isinstance(response, str):
                parsed_response = json.loads(response)
            else:
                parsed_response = response

            # Extract text content using direct indexing (matches script_agent.py:112)
            response_text = parsed_response['content'][1]['text']
            print(f"[Validation Sub-Agent] Attempt {attempt + 1}: Extracted text length: {len(response_text)}")

            # Retry if empty
            if response_text and len(response_text) > 0:
                break
            else:
                print(f"[Validation Sub-Agent] WARNING: Empty response on attempt {attempt + 1}, retrying...")

        # Clean JSON response from markdown code blocks
        response_text = clean_json_response(response_text) if response_text else ""

        # Parse the JSON validation output with fallback
        try:
            if not response_text:
                raise ValueError("Empty response after retry")
            validation_data = json.loads(response_text)
            frames_result = validation_data.get("frames", [])
        except (json.JSONDecodeError, ValueError) as e:
            # Fallback: null is_good means "validation error, result unknown"
            print(f"[Validation Sub-Agent] WARNING: Validation error - {e}")
            frames_result = [
                {
                    "filename": fn,
                    "is_good": None,
                    "issues": f"Validation error: {str(e)[:250]}"
                }
                for fn in frame_filenames
            ]

        print(f"[Validation Sub-Agent] Validation complete: {len(frames_result)} frame(s) checked")
        print(f"[Validation Sub-Agent] ========== SHOT {scene}.{shot} COMPLETE ==========\n")

        return {
            "scene": scene,
            "shot": shot,
            "frames": frames_result
        }

    except Exception as e:
        print(f"[Validation Sub-Agent] ERROR during validation: {e}")
        import traceback
        print(f"[Validation Sub-Agent] Traceback:\n{traceback.format_exc()}")

        # Return error result with null is_good (validation error, result unknown)
        return {
            "scene": scene,
            "shot": shot,
            "frames": [
                {
                    "filename": fn,
                    "is_good": None,
                    "issues": f"Validation error: {str(e)[:250]}"
                }
                for fn in frame_filenames
            ],
            "error": str(e)
        }
